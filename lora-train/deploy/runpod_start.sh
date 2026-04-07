#!/usr/bin/env bash
set -euo pipefail

cd /app/lora-train

PY_BIN="python3"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  PY_BIN="python"
fi
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  echo "Error: neither python3 nor python is available in this container."
  exit 1
fi

echo "== GPU Info =="
nvidia-smi || true

ENV_FILE="${ENV_FILE:-.env.prod}"
ENV_ARGS=()
if [ -f "${ENV_FILE}" ]; then
  echo "Using env file: ${ENV_FILE}"
  set -a
  source "${ENV_FILE}"
  set +a
  ENV_ARGS=(--env-file "${ENV_FILE}")
else
  echo "Warning: ${ENV_FILE} not found. Using runtime environment variables + YAML config defaults."
fi

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
mkdir -p "${HF_HOME}"
mkdir -p "${OUTPUT_DIR:-/workspace/outputs}"

if [ -n "${HF_TOKEN:-}" ]; then
  if command -v hf >/dev/null 2>&1; then
    hf auth login --token "${HF_TOKEN}" || true
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
  elif "${PY_BIN}" -c "import huggingface_hub.cli.hf" >/dev/null 2>&1; then
    "${PY_BIN}" -m huggingface_hub.cli.hf auth login --token "${HF_TOKEN}" || true
  else
    echo "Warning: no Hugging Face CLI found in container; continuing without explicit login."
  fi
fi

"${PY_BIN}" scripts/sanity_check_gpu.py
"${PY_BIN}" -m src.train --config configs/prod.yaml "${ENV_ARGS[@]}" --preflight-only
if command -v accelerate >/dev/null 2>&1; then
  accelerate launch -m src.train --config configs/prod.yaml "${ENV_ARGS[@]}"
else
  "${PY_BIN}" -m accelerate.commands.launch -m src.train --config configs/prod.yaml "${ENV_ARGS[@]}"
fi
