from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import yaml


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"Cannot parse boolean value: {value!r}")


def parse_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    return int(str(value).strip())


def parse_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def parse_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def parse_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def parse_csv_list(value: Any) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        values = [str(v).strip() for v in value if str(v).strip()]
        return values or None
    text = str(value).strip()
    if not text:
        return None
    values = [part.strip() for part in text.split(",") if part.strip()]
    return values or None


@dataclass(frozen=True)
class FieldSpec:
    env: str
    parser: Callable[[Any], Any]
    default: Any
    required: bool = False


FIELD_SPECS: Dict[str, FieldSpec] = {
    "model_id": FieldSpec("MODEL_ID", parse_str, "", required=True),
    "data_train_path": FieldSpec("DATA_TRAIN_PATH", parse_str, "", required=True),
    "data_eval_path": FieldSpec("DATA_EVAL_PATH", parse_optional_str, None),
    "output_dir": FieldSpec("OUTPUT_DIR", parse_str, "outputs"),
    "run_name": FieldSpec("RUN_NAME", parse_str, "lora-run"),
    "seed": FieldSpec("SEED", parse_int, 42),
    "max_seq_len": FieldSpec("MAX_SEQ_LEN", parse_int, 1024),
    "micro_batch_size": FieldSpec("MICRO_BATCH_SIZE", parse_int, 1),
    "grad_accum_steps": FieldSpec("GRAD_ACCUM_STEPS", parse_int, 16),
    "num_epochs": FieldSpec("NUM_EPOCHS", parse_float, 1.0),
    "learning_rate": FieldSpec("LEARNING_RATE", parse_float, 2e-4),
    "warmup_ratio": FieldSpec("WARMUP_RATIO", parse_float, 0.03),
    "weight_decay": FieldSpec("WEIGHT_DECAY", parse_float, 0.0),
    "lr_scheduler": FieldSpec("LR_SCHEDULER", parse_str, "cosine"),
    "max_steps": FieldSpec("MAX_STEPS", parse_int, -1),
    "eval_strategy": FieldSpec("EVAL_STRATEGY", parse_str, "steps"),
    "save_steps": FieldSpec("SAVE_STEPS", parse_int, 50),
    "eval_steps": FieldSpec("EVAL_STEPS", parse_int, 50),
    "logging_steps": FieldSpec("LOGGING_STEPS", parse_int, 10),
    "use_qlora": FieldSpec("USE_QLORA", parse_bool, True),
    "load_in_4bit": FieldSpec("LOAD_IN_4BIT", parse_bool, True),
    "bnb_4bit_quant_type": FieldSpec("BNB_4BIT_QUANT_TYPE", parse_str, "nf4"),
    "bnb_4bit_compute_dtype": FieldSpec(
        "BNB_4BIT_COMPUTE_DTYPE", parse_str, "bfloat16"
    ),
    "gradient_checkpointing": FieldSpec("GRADIENT_CHECKPOINTING", parse_bool, True),
    "bf16": FieldSpec("BF16", parse_bool, True),
    "fp16": FieldSpec("FP16", parse_bool, False),
    "lora_r": FieldSpec("LORA_R", parse_int, 16),
    "lora_alpha": FieldSpec("LORA_ALPHA", parse_int, 32),
    "lora_dropout": FieldSpec("LORA_DROPOUT", parse_float, 0.05),
    "lora_target_modules": FieldSpec(
        "LORA_TARGET_MODULES", parse_csv_list, None
    ),
    "packing": FieldSpec("PACKING", parse_bool, False),
    "eval_split_ratio": FieldSpec("EVAL_SPLIT_RATIO", parse_float, 0.1),
    "system_prompt": FieldSpec("SYSTEM_PROMPT", parse_str, "You are a helpful assistant."),
    "resume_from_checkpoint": FieldSpec(
        "RESUME_FROM_CHECKPOINT", parse_optional_str, None
    ),
    "wandb_project": FieldSpec("WANDB_PROJECT", parse_optional_str, None),
    "wandb_entity": FieldSpec("WANDB_ENTITY", parse_optional_str, None),
    "wandb_api_key": FieldSpec("WANDB_API_KEY", parse_optional_str, None),
    "hf_token": FieldSpec("HF_TOKEN", parse_optional_str, None),
    "push_to_hub": FieldSpec("PUSH_TO_HUB", parse_bool, False),
    "hub_repo_id": FieldSpec("HUB_REPO_ID", parse_optional_str, None),
    "hub_private": FieldSpec("HUB_PRIVATE", parse_bool, True),
    "trust_remote_code": FieldSpec("TRUST_REMOTE_CODE", parse_bool, False),
    "attn_implementation": FieldSpec("ATTN_IMPLEMENTATION", parse_optional_str, None),
}

ENV_TO_FIELD = {spec.env: key for key, spec in FIELD_SPECS.items()}


@dataclass
class TrainConfig:
    model_id: str
    data_train_path: str
    data_eval_path: Optional[str]
    output_dir: str
    run_name: str
    seed: int
    max_seq_len: int
    micro_batch_size: int
    grad_accum_steps: int
    num_epochs: float
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler: str
    max_steps: int
    eval_strategy: str
    save_steps: int
    eval_steps: int
    logging_steps: int
    use_qlora: bool
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: Optional[list[str]]
    packing: bool
    eval_split_ratio: float
    system_prompt: str
    resume_from_checkpoint: Optional[str]
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_api_key: Optional[str]
    hf_token: Optional[str]
    push_to_hub: bool
    hub_repo_id: Optional[str]
    hub_private: bool
    trust_remote_code: bool
    attn_implementation: Optional[str]

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapter"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def merged_dir(self) -> Path:
        return self.run_dir / "merged"

    def to_safe_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("hf_token"):
            payload["hf_token"] = "***"
        if payload.get("wandb_api_key"):
            payload["wandb_api_key"] = "***"
        return payload

    def validate(self) -> None:
        if not self.model_id:
            raise ConfigError("`model_id` is required.")
        if not self.data_train_path:
            raise ConfigError("`data_train_path` is required.")
        if self.max_seq_len <= 0:
            raise ConfigError("`max_seq_len` must be > 0.")
        if self.micro_batch_size <= 0:
            raise ConfigError("`micro_batch_size` must be > 0.")
        if self.grad_accum_steps <= 0:
            raise ConfigError("`grad_accum_steps` must be > 0.")
        if self.num_epochs <= 0 and self.max_steps <= 0:
            raise ConfigError("Set `num_epochs > 0` or `max_steps > 0`.")
        if self.learning_rate <= 0:
            raise ConfigError("`learning_rate` must be > 0.")
        if self.eval_strategy not in {"no", "steps", "epoch"}:
            raise ConfigError("`eval_strategy` must be one of: no, steps, epoch.")
        if self.eval_strategy == "steps" and self.eval_steps <= 0:
            raise ConfigError("`eval_steps` must be > 0 when eval_strategy=steps.")
        if self.save_steps <= 0:
            raise ConfigError("`save_steps` must be > 0.")
        if self.logging_steps <= 0:
            raise ConfigError("`logging_steps` must be > 0.")
        if self.bf16 and self.fp16:
            raise ConfigError("Only one of `bf16` or `fp16` can be true.")
        if self.use_qlora and not self.load_in_4bit:
            raise ConfigError("`use_qlora=true` requires `load_in_4bit=true`.")
        if self.eval_split_ratio < 0 or self.eval_split_ratio >= 1:
            raise ConfigError("`eval_split_ratio` must be in [0, 1).")
        if self.push_to_hub and not self.hub_repo_id:
            raise ConfigError("`hub_repo_id` is required when push_to_hub=true.")
        if self.attn_implementation and self.attn_implementation not in {
            "eager",
            "sdpa",
            "flash_attention_2",
        }:
            raise ConfigError(
                "`attn_implementation` must be one of: eager, sdpa, flash_attention_2."
            )


def _normalize_key(key: str) -> Optional[str]:
    direct = key.strip().lower().replace("-", "_")
    if direct in FIELD_SPECS:
        return direct
    upper = key.strip().upper()
    if upper in ENV_TO_FIELD:
        return ENV_TO_FIELD[upper]
    return None


def _normalize_input_map(raw: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in raw.items():
        normalized = _normalize_key(key)
        if normalized is None:
            continue
        output[normalized] = value
    return output


def _coerce_field(field_name: str, value: Any) -> Any:
    parser = FIELD_SPECS[field_name].parser
    try:
        return parser(value)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ConfigError(f"Invalid value for `{field_name}`: {value!r}") from exc


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"YAML config must be a mapping: {config_path}")
    return _normalize_input_map(payload)


def parse_override_pairs(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not items:
        return overrides
    for item in items:
        if "=" not in item:
            raise ConfigError(
                f"Override must use KEY=VALUE format. Invalid item: {item!r}"
            )
        key, raw_value = item.split("=", 1)
        normalized = _normalize_key(key)
        if normalized is None:
            raise ConfigError(f"Unknown override key: {key!r}")
        overrides[normalized] = raw_value
    return overrides


def load_train_config(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    yaml_values = _load_yaml_config(Path(config_path))
    env_values = _normalize_input_map(dict(os.environ))
    cli_values = _normalize_input_map(cli_overrides or {})

    merged: Dict[str, Any] = {}
    for field_name, spec in FIELD_SPECS.items():
        merged[field_name] = spec.default
        if field_name in yaml_values:
            merged[field_name] = _coerce_field(field_name, yaml_values[field_name])
        env_key = spec.env
        if env_key in os.environ and os.environ[env_key] != "":
            merged[field_name] = _coerce_field(field_name, env_values[field_name])
        if field_name in cli_values and cli_values[field_name] is not None:
            merged[field_name] = _coerce_field(field_name, cli_values[field_name])

        if spec.required and not merged[field_name]:
            raise ConfigError(f"Required configuration `{field_name}` is missing.")

    cfg = TrainConfig(**merged)
    cfg.validate()
    return cfg


def save_resolved_config(config: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_safe_dict(), handle, indent=2)
