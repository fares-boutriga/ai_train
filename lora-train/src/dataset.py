from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from .formatting import (
    assistant_response_from_chat,
    messages_to_text,
    prompt_messages_from_chat,
    row_to_messages,
)


def ensure_path_exists(path: str, label: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    return file_path


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if "messages" in row and row["messages"]:
                try:
                    row["messages"] = json.loads(row["messages"])
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"CSV column `messages` must be valid JSON list: {path}"
                    ) from exc
            rows.append(row)
    return rows


def load_rows(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(file_path)
    if suffix == ".csv":
        return _load_csv(file_path)
    raise ValueError(
        f"Unsupported dataset extension: {suffix}. Use .jsonl or .csv files."
    )


def rows_to_chat_records(
    rows: List[Dict[str, Any]], system_prompt: str = ""
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            messages = row_to_messages(row, system_prompt=system_prompt)
        except Exception as exc:
            raise ValueError(f"Failed to parse row {index}: {exc}") from exc
        records.append({"messages": messages})
    return records


def _train_eval_split(
    records: List[Dict[str, Any]], eval_split_ratio: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not records:
        return [], []
    if eval_split_ratio <= 0:
        return records, []

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(len(records) * eval_split_ratio))
    eval_idx = set(indices[:eval_size])

    train_records = [records[i] for i in range(len(records)) if i not in eval_idx]
    eval_records = [records[i] for i in range(len(records)) if i in eval_idx]
    return train_records, eval_records


def load_chat_records(
    train_path: str,
    eval_path: Optional[str],
    eval_split_ratio: float,
    seed: int,
    system_prompt: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows = load_rows(train_path)
    train_records = rows_to_chat_records(train_rows, system_prompt=system_prompt)

    if eval_path:
        eval_rows = load_rows(eval_path)
        eval_records = rows_to_chat_records(eval_rows, system_prompt=system_prompt)
        return train_records, eval_records

    return _train_eval_split(train_records, eval_split_ratio=eval_split_ratio, seed=seed)


def records_to_sft_dataset(records: List[Dict[str, Any]], tokenizer: Any) -> Dataset:
    prepared: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        messages = record["messages"]
        prompt_messages = prompt_messages_from_chat(messages)
        prepared.append(
            {
                "id": index,
                "messages": messages,
                "text": messages_to_text(messages, tokenizer, add_generation_prompt=False),
                "prompt": messages_to_text(
                    prompt_messages, tokenizer, add_generation_prompt=True
                ),
                "target": assistant_response_from_chat(messages),
            }
        )
    return Dataset.from_list(prepared)

