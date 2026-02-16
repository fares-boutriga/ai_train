#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
fi

python scripts/sanity_check_gpu.py
accelerate launch -m src.train --config configs/local.yaml --env-file .env.local "$@"

