from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ConfigError, load_train_config, parse_override_pairs, save_resolved_config
from .env import load_env_file
from .utils.hf import choose_model_dtype, maybe_hf_login
from .utils.logging import setup_logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--env-file", default=None, help="Optional dotenv file path.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config values with KEY=VALUE (can repeat).",
    )
    parser.add_argument("--adapter-path", default=None, help="Path to adapter folder.")
    parser.add_argument("--output-path", default=None, help="Path to merged model output.")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push merged model to Hub (requires HUB_REPO_ID/HF_TOKEN).",
    )
    return parser.parse_args()


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return parse_override_pairs(args.override)


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

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, cfg.run_dir / "resolved_config.merge.json")

    adapter_path = Path(args.adapter_path) if args.adapter_path else cfg.adapter_dir
    output_path = Path(args.output_path) if args.output_path else cfg.merged_dir
    if not adapter_path.exists():
        raise SystemExit(f"Adapter path not found: {adapter_path}")

    maybe_hf_login(cfg.hf_token)
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=cfg.trust_remote_code,
        token=cfg.hf_token,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        trust_remote_code=cfg.trust_remote_code,
        token=cfg.hf_token,
        torch_dtype=choose_model_dtype(cfg.bf16, cfg.fp16),
        device_map="auto",
        attn_implementation=cfg.attn_implementation,
    )
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = peft_model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    logger.info("Merged model saved to %s", output_path)

    push_now = args.push or cfg.push_to_hub
    if push_now:
        if not cfg.hub_repo_id:
            raise SystemExit("HUB_REPO_ID is required to push merged model.")
        merged.push_to_hub(cfg.hub_repo_id, private=cfg.hub_private, token=cfg.hf_token)
        tokenizer.push_to_hub(
            cfg.hub_repo_id, private=cfg.hub_private, token=cfg.hf_token
        )
        logger.info("Pushed merged model to Hub: %s", cfg.hub_repo_id)


if __name__ == "__main__":
    main()
