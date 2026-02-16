from __future__ import annotations

import logging
from typing import Optional

import torch
from huggingface_hub import login
from transformers import BitsAndBytesConfig


logger = logging.getLogger(__name__)


def default_lora_target_modules(model_id: str) -> list[str]:
    model_key = model_id.lower()
    if "mistral" in model_key:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    if "qwen" in model_key:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def torch_dtype_from_name(name: str) -> torch.dtype:
    norm = name.strip().lower()
    if norm in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if norm in {"fp16", "float16"}:
        return torch.float16
    if norm in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {name!r}")


def choose_model_dtype(bf16: bool, fp16: bool) -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def build_4bit_config(
    quant_type: str,
    compute_dtype_name: str,
    load_in_4bit: bool,
) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch_dtype_from_name(compute_dtype_name),
        bnb_4bit_use_double_quant=True,
    )


def maybe_hf_login(token: Optional[str]) -> None:
    if not token:
        return
    logger.info("Logging into Hugging Face Hub from HF_TOKEN.")
    login(token=token, add_to_git_credential=True)
