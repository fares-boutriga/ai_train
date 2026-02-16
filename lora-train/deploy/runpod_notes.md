# RunPod Notes (Production)

## Recommended GPU
- Minimum: RTX A6000 (48 GB) for QLoRA on Qwen2.5-14B.
- Better throughput: H100, A100 80GB, or multi-GPU nodes.

## Storage and Mounts
- Mount a persistent volume for outputs and cache:
  - `/workspace/outputs` for checkpoints/adapters/merged models
  - `/workspace/hf-cache` for Hugging Face cache
- In RunPod template/env settings:
  - `OUTPUT_DIR=/workspace/outputs`
  - `HF_HOME=/workspace/hf-cache`

## Environment Variables in RunPod UI
Set these at pod launch (or via `.env.prod`):
- `MODEL_ID=Qwen/Qwen2.5-14B-Instruct`
- `DATA_TRAIN_PATH=/workspace/lora-train/data/train.jsonl`
- `DATA_EVAL_PATH=/workspace/lora-train/data/eval.jsonl`
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

