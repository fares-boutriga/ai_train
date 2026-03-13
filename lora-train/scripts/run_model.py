#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env import load_env_file  # noqa: E402
from src.formatting import messages_to_text  # noqa: E402
from src.utils.hf import build_4bit_config, choose_model_dtype  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one prompt against a base, adapter, or merged model."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["base", "adapter", "merged"],
        help="Model mode to run.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model ID/path (required for base and adapter modes).",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Adapter directory path (required for adapter mode).",
    )
    parser.add_argument(
        "--merged-model-path",
        default=None,
        help="Merged model directory path (required for merged mode).",
    )
    parser.add_argument("--message", required=True, help="User message/prompt.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load model with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--env-file", default=None, help="Optional dotenv file.")
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save prompt + generation result as JSON.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.mode in {"base", "adapter"} and not args.base_model:
        raise SystemExit("--base-model is required for base or adapter mode.")
    if args.mode == "adapter" and not args.adapter_path:
        raise SystemExit("--adapter-path is required for adapter mode.")
    if args.mode == "merged" and not args.merged_model_path:
        raise SystemExit("--merged-model-path is required for merged mode.")
    if args.bf16 and args.fp16:
        raise SystemExit("Use only one of --bf16 or --fp16.")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be > 0.")


def _build_model_kwargs(args: argparse.Namespace, hf_token: str | None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "token": hf_token,
        "device_map": "auto",
    }
    if args.use_4bit:
        kwargs["quantization_config"] = build_4bit_config(
            quant_type=args.bnb_4bit_quant_type,
            compute_dtype_name=args.bnb_4bit_compute_dtype,
            load_in_4bit=True,
        )
    else:
        model_dtype = choose_model_dtype(args.bf16, args.fp16)
        if model_dtype is not None:
            kwargs["torch_dtype"] = model_dtype
    return kwargs


def _load_model_and_tokenizer(args: argparse.Namespace, hf_token: str | None):
    kwargs = _build_model_kwargs(args, hf_token=hf_token)
    if args.mode == "merged":
        merged_path = Path(args.merged_model_path)
        if not merged_path.exists():
            raise SystemExit(f"Merged model path not found: {merged_path}")
        model = AutoModelForCausalLM.from_pretrained(str(merged_path), **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            str(merged_path), trust_remote_code=args.trust_remote_code, token=hf_token
        )
        return model, tokenizer

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    if args.mode == "adapter":
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            raise SystemExit(f"Adapter path not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        tokenizer_source = str(adapter_path)
    else:
        tokenizer_source = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=args.trust_remote_code, token=hf_token
    )
    return model, tokenizer


def _resolve_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    _validate_args(args)

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Inference will run on CPU (slow).")

    set_seed(args.seed)
    model, tokenizer = _load_model_and_tokenizer(args, hf_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.message},
    ]
    prompt_text = messages_to_text(messages, tokenizer, add_generation_prompt=True)

    device = _resolve_device(model)
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature if not args.greedy else None,
            top_p=args.top_p if not args.greedy else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = int(model_inputs["input_ids"].shape[-1])
    completion = tokenizer.decode(
        generated[0][prompt_len:], skip_special_tokens=True
    ).strip()

    result = {
        "mode": args.mode,
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "merged_model_path": args.merged_model_path,
        "message": args.message,
        "response": completion,
    }

    print("\n=== Model Response ===")
    print(completion)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved output JSON to: {out_path}")


if __name__ == "__main__":
    main()

