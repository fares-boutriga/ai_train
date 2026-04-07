from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi, login
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


def ensure_hub_repo(repo_id: str, private: bool, token: Optional[str]) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def create_version_tag(
    repo_id: str,
    run_name: str,
    tag_prefix: str,
    token: Optional[str],
    revision: str = "main",
) -> str:
    safe_run = re.sub(r"[^a-zA-Z0-9._-]+", "-", run_name).strip("-") or "run"
    safe_prefix = re.sub(r"[^a-zA-Z0-9._-]+", "-", tag_prefix).strip("-") or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag_name = f"{safe_prefix}-{safe_run}-{timestamp}"

    api = HfApi(token=token)
    api.create_tag(
        repo_id=repo_id,
        repo_type="model",
        tag=tag_name,
        revision=revision,
        tag_message=f"Auto tag for run: {run_name}",
    )
    return tag_name


def upload_folder_to_hub(
    repo_id: str,
    folder_path: str | Path,
    token: Optional[str],
    commit_message: str,
    delete_patterns: Optional[list[str]] = None,
) -> None:
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        commit_message=commit_message,
        delete_patterns=delete_patterns,
    )
