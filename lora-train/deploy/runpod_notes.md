# RunPod Notes (Production)

## Recommended GPU
- Minimum: RTX A5000 / RTX A6000 for QLoRA on Qwen3.5-4B.
- Better throughput: RTX A6000, H100, or A100 80GB.

## Storage and Mounts
- The training code lives in the container image at `/app/lora-train`.
- Mount a persistent volume for outputs and cache:
  - `/workspace/outputs` for checkpoints/adapters/merged models
  - `/workspace/hf-cache` for Hugging Face cache
- In RunPod template/env settings:
  - `OUTPUT_DIR=/workspace/outputs`
  - `HF_HOME=/workspace/hf-cache`

## Environment Variables in RunPod UI
Set these at pod launch (or via `.env.prod`):
- `MODEL_ID=Qwen/Qwen3.5-4B`
- `DATA_TRAIN_PATH=/app/lora-train/data/train.jsonl`
- `DATA_EVAL_PATH=/app/lora-train/data/eval.jsonl`
- `RUN_NAME=qwen-prod-run`
- `HF_TOKEN=...` (optional, required for private model/repo or Hub push)
- `WANDB_API_KEY=...` (optional)

## Start Training
Container entrypoint already runs:
1. `nvidia-smi`
2. sanity check (`scripts/sanity_check_gpu.py`)
3. preflight (`python -m src.train --preflight-only`)
4. training via `accelerate launch`

## Resume From Checkpoint
- Set `RESUME_FROM_CHECKPOINT` to a checkpoint path, for example:
  - `/workspace/outputs/qwen-prod-run/checkpoints/checkpoint-1200`
- Or pass CLI override:
```bash
accelerate launch -m src.train \
  --config configs/prod.yaml \
  --env-file .env.prod \
  --override RESUME_FROM_CHECKPOINT=/workspace/outputs/qwen-prod-run/checkpoints/checkpoint-1200
```
