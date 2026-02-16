from __future__ import annotations

import logging
import os
from typing import List, Optional


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def configure_wandb(
    project: Optional[str], entity: Optional[str], api_key: Optional[str]
) -> None:
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    if project:
        os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity


def resolve_report_to(project: Optional[str]) -> List[str]:
    targets = ["tensorboard"]
    if project:
        targets.append("wandb")
    return targets

