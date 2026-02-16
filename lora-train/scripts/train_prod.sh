#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env.prod ]; then
  set -a
  source .env.prod
  set +a
fi

python scripts/sanity_check_gpu.py
accelerate launch -m src.train --config configs/prod.yaml --env-file .env.prod "$@"

