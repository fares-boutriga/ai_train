from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from .config import ConfigError, load_train_config, parse_override_pairs, save_resolved_config
from .dataset import ensure_path_exists, load_chat_records, records_to_sft_dataset
from .env import load_env_file
from .utils.hf import (
    bf16_supported,
    build_4bit_config,
    choose_model_dtype,
    default_lora_target_modules,
    maybe_hf_login,
)
from .utils.logging import configure_wandb, resolve_report_to, setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--env-file", default=None, help="Optional dotenv file path.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE (can repeat).",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-train-path", default=None)
    parser.add_argument("--data-eval-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run config/dataset checks and exit without training.",
    )
    return parser.parse_args()


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = parse_override_pairs(args.override)
    direct = {
        "model_id": args.model_id,
        "run_name": args.run_name,
        "data_train_path": args.data_train_path,
        "data_eval_path": args.data_eval_path,
        "output_dir": args.output_dir,
        "resume_from_checkpoint": args.resume_from_checkpoint,
    }
    for key, value in direct.items():
        if value is not None:
            overrides[key] = value
    return overrides


def _preflight_checks() -> None:
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. CPU fallback is possible but very slow.")
    else:
        logger.info("CUDA devices available: %s", torch.cuda.device_count())


def _resolve_target_modules(config_target_modules: list[str] | None, model_id: str) -> list[str]:
    return config_target_modules or default_lora_target_modules(model_id)


def _set_eval_strategy_kwargs(
    training_kwargs: Dict[str, Any],
    eval_strategy: str,
    has_eval_dataset: bool,
    eval_steps: int,
    parameter_names: set[str],
) -> None:
    """Support eval strategy naming differences across argument classes."""
    strategy_value = eval_strategy if has_eval_dataset else "no"
    if "evaluation_strategy" in parameter_names:
        training_kwargs["evaluation_strategy"] = strategy_value
    elif "eval_strategy" in parameter_names:
        training_kwargs["eval_strategy"] = strategy_value
    else:  # pragma: no cover - defensive fallback for future API changes
        raise SystemExit(
            "Unsupported trainer args class: no eval strategy parameter found."
        )

    if strategy_value == "steps":
        training_kwargs["eval_steps"] = eval_steps


def _load_tokenizer(model_id: str, trust_remote_code: bool, token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=token,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model(cfg, target_modules: list[str]):
    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": cfg.model_id,
        "trust_remote_code": cfg.trust_remote_code,
        "token": cfg.hf_token,
        "device_map": "auto",
    }
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    if cfg.use_qlora or cfg.load_in_4bit:
        model_kwargs["quantization_config"] = build_4bit_config(
            quant_type=cfg.bnb_4bit_quant_type,
            compute_dtype_name=cfg.bnb_4bit_compute_dtype,
            load_in_4bit=cfg.load_in_4bit,
        )
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )
    else:
        model_dtype = choose_model_dtype(cfg.bf16, cfg.fp16)
        if model_dtype is not None:
            model_kwargs["torch_dtype"] = model_dtype
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, lora_config


def _should_use_sft_config() -> bool:
    params = inspect.signature(SFTTrainer.__init__).parameters
    return "processing_class" in params


def _build_trainer_args(cfg, training_kwargs: Dict[str, Any]):
    if _should_use_sft_config():
        sft_param_names = set(inspect.signature(SFTConfig.__init__).parameters)
        _set_eval_strategy_kwargs(
            training_kwargs=training_kwargs,
            eval_strategy=cfg.eval_strategy,
            has_eval_dataset=True,
            eval_steps=cfg.eval_steps,
            parameter_names=sft_param_names,
        )
        sft_kwargs = {
            key: value
            for key, value in training_kwargs.items()
            if key in sft_param_names and value is not None
        }
        if "dataset_text_field" in sft_param_names:
            sft_kwargs["dataset_text_field"] = "text"
        if "max_length" in sft_param_names:
            sft_kwargs["max_length"] = cfg.max_seq_len
        elif "max_seq_length" in sft_param_names:
            sft_kwargs["max_seq_length"] = cfg.max_seq_len
        if "packing" in sft_param_names:
            sft_kwargs["packing"] = cfg.packing
        return SFTConfig(**sft_kwargs)

    train_arg_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    _set_eval_strategy_kwargs(
        training_kwargs=training_kwargs,
        eval_strategy=cfg.eval_strategy,
        has_eval_dataset=True,
        eval_steps=cfg.eval_steps,
        parameter_names=train_arg_names,
    )
    return TrainingArguments(**training_kwargs)


def main() -> None:
    setup_logging()
    args = parse_args()
    load_env_file(args.env_file)

    try:
        cli_overrides = _build_cli_overrides(args)
        cfg = load_train_config(config_path=args.config, cli_overrides=cli_overrides)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    _preflight_checks()
    if cfg.bf16 and torch.cuda.is_available() and not bf16_supported():
        raise SystemExit(
            "BF16 requested but not supported on this GPU. Set BF16=false and FP16=true."
        )
    if not torch.cuda.is_available() and (cfg.bf16 or cfg.fp16):
        logger.warning("Disabling bf16/fp16 because CUDA is unavailable (CPU fallback).")
        cfg.bf16 = False
        cfg.fp16 = False

    ensure_path_exists(cfg.data_train_path, "Train dataset")
    if cfg.data_eval_path:
        ensure_path_exists(cfg.data_eval_path, "Eval dataset")

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, cfg.run_dir / "resolved_config.json")

    train_records, eval_records = load_chat_records(
        train_path=cfg.data_train_path,
        eval_path=cfg.data_eval_path,
        eval_split_ratio=cfg.eval_split_ratio,
        seed=cfg.seed,
        system_prompt=cfg.system_prompt,
    )
    if not train_records:
        raise SystemExit("No training records found after dataset processing.")

    logger.info(
        "Dataset loaded: %d train records, %d eval records",
        len(train_records),
        len(eval_records),
    )
    if args.preflight_only:
        logger.info("Preflight checks passed.")
        return

    set_seed(cfg.seed)
    maybe_hf_login(cfg.hf_token)
    configure_wandb(cfg.wandb_project, cfg.wandb_entity, cfg.wandb_api_key)

    tokenizer = _load_tokenizer(
        model_id=cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
        token=cfg.hf_token,
    )
    target_modules = _resolve_target_modules(cfg.lora_target_modules, cfg.model_id)
    logger.info("LoRA target modules: %s", ", ".join(target_modules))

    model, lora_config = _load_model(cfg, target_modules=target_modules)
    train_dataset = records_to_sft_dataset(train_records, tokenizer=tokenizer)
    eval_dataset = (
        records_to_sft_dataset(eval_records, tokenizer=tokenizer) if eval_records else None
    )
    if _should_use_sft_config():
        # Keep only raw text for TRL>=0.28 to avoid prompt/completion mask shape issues.
        train_drop_cols = [c for c in train_dataset.column_names if c != "text"]
        if train_drop_cols:
            train_dataset = train_dataset.remove_columns(train_drop_cols)
        if eval_dataset is not None:
            eval_drop_cols = [c for c in eval_dataset.column_names if c != "text"]
            if eval_drop_cols:
                eval_dataset = eval_dataset.remove_columns(eval_drop_cols)

    report_to = resolve_report_to(cfg.wandb_project)
    training_kwargs: Dict[str, Any] = {
        "output_dir": str(cfg.checkpoint_dir),
        "per_device_train_batch_size": cfg.micro_batch_size,
        "per_device_eval_batch_size": max(1, cfg.micro_batch_size),
        "gradient_accumulation_steps": cfg.grad_accum_steps,
        "num_train_epochs": cfg.num_epochs,
        "max_steps": cfg.max_steps,
        "learning_rate": cfg.learning_rate,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": cfg.lr_scheduler,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_strategy": "steps",
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "seed": cfg.seed,
        "report_to": report_to,
        "run_name": cfg.run_name,
        "save_total_limit": 3,
        "logging_first_step": True,
        "remove_unused_columns": False,
    }
    if eval_dataset is None:
        cfg.eval_strategy = "no"
    if torch.cuda.device_count() > 1:
        training_kwargs["ddp_find_unused_parameters"] = False

    trainer_args = _build_trainer_args(cfg=cfg, training_kwargs=training_kwargs)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": trainer_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora_config,
    }
    trainer_param_names = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "tokenizer" in trainer_param_names:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_param_names:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in trainer_param_names:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_param_names:
        trainer_kwargs["max_seq_length"] = cfg.max_seq_len
    if "packing" in trainer_param_names:
        trainer_kwargs["packing"] = cfg.packing

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training run: %s", cfg.run_name)
    train_output = trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_state()

    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(cfg.adapter_dir))
    tokenizer.save_pretrained(str(cfg.adapter_dir))

    trainer.log_metrics("train", train_output.metrics)
    trainer.save_metrics("train", train_output.metrics)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if cfg.push_to_hub:
        if not cfg.hub_repo_id:
            raise SystemExit("PUSH_TO_HUB=true requires HUB_REPO_ID.")
        trainer.model.push_to_hub(
            cfg.hub_repo_id, private=cfg.hub_private, token=cfg.hf_token
        )
        tokenizer.push_to_hub(
            cfg.hub_repo_id, private=cfg.hub_private, token=cfg.hf_token
        )
        logger.info("Pushed adapter/tokenizer to Hub repo: %s", cfg.hub_repo_id)

    logger.info("Training complete. Adapter saved to %s", cfg.adapter_dir)


if __name__ == "__main__":
    main()
