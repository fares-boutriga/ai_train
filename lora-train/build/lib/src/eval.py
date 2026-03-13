from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .config import ConfigError, load_train_config, parse_override_pairs, save_resolved_config
from .dataset import ensure_path_exists, load_chat_records
from .env import load_env_file
from .formatting import messages_to_text, prompt_messages_from_chat
from .utils.hf import build_4bit_config, choose_model_dtype
from .utils.logging import setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter or merged model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--env-file", default=None, help="Optional dotenv file path.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE (can repeat).",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to LoRA adapter folder. Defaults to outputs/{run_name}/adapter.",
    )
    parser.add_argument(
        "--merged-model-path",
        default=None,
        help="Path to merged model. If set, adapter loading is skipped.",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return parse_override_pairs(args.override)


def _load_model_and_tokenizer(cfg, adapter_path: Path | None, merged_model_path: Path | None):
    if merged_model_path:
        tokenizer = AutoTokenizer.from_pretrained(
            str(merged_model_path),
            trust_remote_code=cfg.trust_remote_code,
            token=cfg.hf_token,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(merged_model_path),
            trust_remote_code=cfg.trust_remote_code,
            token=cfg.hf_token,
            torch_dtype=choose_model_dtype(cfg.bf16, cfg.fp16),
            device_map="auto",
            attn_implementation=cfg.attn_implementation,
        )
        return model, tokenizer

    if adapter_path is None:
        raise ValueError("adapter_path is required when merged_model_path is not provided.")

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
    else:
        model_dtype = choose_model_dtype(cfg.bf16, cfg.fp16)
        if model_dtype is not None:
            model_kwargs["torch_dtype"] = model_dtype

    base_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=cfg.trust_remote_code,
        token=cfg.hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def _compute_eval_loss(model, tokenizer, records: List[Dict[str, Any]], max_seq_len: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = model.device

    for record in records:
        text = messages_to_text(record["messages"], tokenizer, add_generation_prompt=False)
        batch = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
        tokens = int(attention_mask.sum().item())
        total_tokens += tokens
        total_loss += float(outputs.loss.item()) * max(1, tokens)

    if total_tokens == 0:
        return {"eval_loss": float("nan"), "perplexity": float("nan")}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return {"eval_loss": avg_loss, "perplexity": ppl}


def _generate_samples(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    sample_count: int,
    max_seq_len: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    model.eval()
    device = model.device
    samples: List[Dict[str, Any]] = []

    for index, record in enumerate(records[:sample_count]):
        prompt_messages = prompt_messages_from_chat(record["messages"])
        prompt_text = messages_to_text(
            prompt_messages, tokenizer, add_generation_prompt=True
        )
        prompt_tokens = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **prompt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = int(prompt_tokens["input_ids"].shape[-1])
        text = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()
        samples.append(
            {
                "sample_id": index,
                "prompt": prompt_text,
                "generated": text,
            }
        )
    return samples


def main() -> None:
    setup_logging()
    args = parse_args()
    load_env_file(args.env_file)

    try:
        cfg = load_train_config(
            config_path=args.config, cli_overrides=_build_cli_overrides(args)
        )
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    ensure_path_exists(cfg.data_train_path, "Train dataset")
    if cfg.data_eval_path:
        ensure_path_exists(cfg.data_eval_path, "Eval dataset")

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, cfg.run_dir / "resolved_config.eval.json")

    _, eval_records = load_chat_records(
        train_path=cfg.data_train_path,
        eval_path=cfg.data_eval_path,
        eval_split_ratio=cfg.eval_split_ratio,
        seed=cfg.seed,
        system_prompt=cfg.system_prompt,
    )
    if not eval_records:
        raise SystemExit(
            "No eval records found. Provide DATA_EVAL_PATH or use a non-zero EVAL_SPLIT_RATIO."
        )

    adapter_path = Path(args.adapter_path) if args.adapter_path else cfg.adapter_dir
    merged_model_path = Path(args.merged_model_path) if args.merged_model_path else None
    if merged_model_path is None and not adapter_path.exists():
        raise SystemExit(
            f"Adapter path not found: {adapter_path}. Train first or pass --merged-model-path."
        )
    if merged_model_path is not None and not merged_model_path.exists():
        raise SystemExit(f"Merged model path not found: {merged_model_path}")

    set_seed(cfg.seed)
    model, tokenizer = _load_model_and_tokenizer(
        cfg=cfg, adapter_path=adapter_path, merged_model_path=merged_model_path
    )

    metrics = _compute_eval_loss(
        model=model,
        tokenizer=tokenizer,
        records=eval_records,
        max_seq_len=cfg.max_seq_len,
    )
    samples = _generate_samples(
        model=model,
        tokenizer=tokenizer,
        records=eval_records,
        sample_count=args.num_samples,
        max_seq_len=cfg.max_seq_len,
        max_new_tokens=args.max_new_tokens,
    )

    metrics_path = cfg.run_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    samples_path = cfg.run_dir / "eval_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Evaluation metrics: %s", metrics)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved %d sample generations to %s", len(samples), samples_path)


if __name__ == "__main__":
    main()
