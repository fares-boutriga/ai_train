#!/usr/bin/env bash
set -euo pipefail

cd /workspace/lora-train

echo "== GPU Info =="
nvidia-smi || true

if [ -f .env.prod ]; then
  set -a
  source .env.prod
  set +a
fi

export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
mkdir -p "${HF_HOME}"
mkdir -p "${OUTPUT_DIR:-/workspace/outputs}"

if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
fi

python scripts/sanity_check_gpu.py
python -m src.train --config configs/prod.yaml --env-file .env.prod --preflight-only
accelerate launch -m src.train --config configs/prod.yaml --env-file .env.prod

