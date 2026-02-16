#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE="${ENV_FILE:-.env.local}"
CONFIG_FILE="${CONFIG_FILE:-configs/local.yaml}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

python -m src.eval --config "${CONFIG_FILE}" --env-file "${ENV_FILE}" "$@"

