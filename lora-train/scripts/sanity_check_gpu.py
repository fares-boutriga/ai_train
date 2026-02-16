#!/usr/bin/env python3
from __future__ import annotations

import platform

import torch


def main() -> None:
    print(f"Python platform: {platform.platform()}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Training will fall back to CPU (very slow).")
        return

    count = torch.cuda.device_count()
    print(f"GPU count: {count}")
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        print(f"[{idx}] {props.name} | VRAM: {total_gb:.2f} GB")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")


if __name__ == "__main__":
    main()

