#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare BIM tabular data (CSV/JSONL) into chat JSONL for SFT. "
            "Default task is Risk_Level prediction."
        )
    )
    parser.add_argument("--input", required=True, help="Input BIM dataset file (.csv/.jsonl).")
    parser.add_argument(
        "--train-output",
        required=True,
        help="Output chat JSONL path for training records.",
    )
    parser.add_argument(
        "--eval-output",
        default="",
        help="Output chat JSONL path for eval records. If omitted, no split is written.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Eval split ratio when --eval-output is provided (default: 0.1).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--system-prompt",
        default="You are a BIM expert assistant.",
        help="System prompt used in each chat example.",
    )
    parser.add_argument(
        "--target-column",
        default="Risk_Level",
        help="Column to predict (default: Risk_Level).",
    )
    parser.add_argument(
        "--task-instruction",
        default="Given this BIM project telemetry, predict the project risk level.",
        help="Instruction line for the user message.",
    )
    parser.add_argument(
        "--feature-columns",
        default="",
        help=(
            "Comma-separated feature columns to include. "
            "If omitted, all columns except target/drop columns are used."
        ),
    )
    parser.add_argument(
        "--drop-columns",
        default="Project_ID,Start_Date,End_Date",
        help="Comma-separated columns to exclude from features.",
    )
    parser.add_argument(
        "--input-title",
        default="Project data",
        help="Heading used before feature lines in user prompt.",
    )
    parser.add_argument(
        "--target-prefix",
        default="Risk level",
        help="Prefix used in assistant output sentence.",
    )
    parser.add_argument(
        "--output-style",
        choices=("label", "sentence", "json"),
        default="sentence",
        help="Assistant output format.",
    )
    parser.add_argument(
        "--drop-empty-rows",
        action="store_true",
        help="Skip rows missing target or usable features.",
    )
    parser.add_argument(
        "--max-input-fields",
        type=int,
        default=0,
        help="Optional cap on number of feature fields per sample (0 = no cap).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _parse_columns(value: str) -> list[str]:
    chunks = [item.strip() for item in value.split(",")]
    return [item for item in chunks if item]


def _ensure_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(parsed, dict):
                raise SystemExit(f"Expected object JSON at {path}:{line_number}")
            rows.append(parsed)
    return rows


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    raise SystemExit(
        f"Unsupported dataset extension '{suffix}'. Use .csv or .jsonl for this script."
    )


def _resolve_feature_columns(
    sample_row: dict[str, Any],
    target_column: str,
    feature_columns: list[str],
    drop_columns: list[str],
) -> list[str]:
    available_columns = list(sample_row.keys())
    if target_column not in available_columns:
        raise SystemExit(
            f"Target column '{target_column}' not found. "
            f"Available columns: {', '.join(available_columns)}"
        )

    if feature_columns:
        missing = [col for col in feature_columns if col not in available_columns]
        if missing:
            raise SystemExit(
                f"Feature columns not found: {', '.join(missing)}. "
                f"Available columns: {', '.join(available_columns)}"
            )
        return feature_columns

    drop_set = set(drop_columns)
    drop_set.add(target_column)
    return [col for col in available_columns if col not in drop_set]


def _assistant_output(value: str, style: str, target_column: str, target_prefix: str) -> str:
    if style == "label":
        return value
    if style == "json":
        return json.dumps({target_column: value}, ensure_ascii=False)
    return f"{target_prefix}: {value}."


def _build_record(
    row: dict[str, Any],
    feature_columns: list[str],
    task_instruction: str,
    system_prompt: str,
    input_title: str,
    target_column: str,
    output_style: str,
    target_prefix: str,
    max_input_fields: int,
) -> dict[str, Any] | None:
    target_value = _to_text(row.get(target_column))
    if not target_value:
        return None

    feature_lines: list[str] = []
    for column in feature_columns:
        value = _to_text(row.get(column))
        if not value:
            continue
        feature_lines.append(f"{column}: {value}")
        if max_input_fields > 0 and len(feature_lines) >= max_input_fields:
            break

    if not feature_lines:
        return None

    user_content = f"{task_instruction}\n\n{input_title}:\n" + "\n".join(feature_lines)
    assistant_content = _assistant_output(
        value=target_value,
        style=output_style,
        target_column=target_column,
        target_prefix=target_prefix,
    )
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _split_records(
    records: list[dict[str, Any]],
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    if eval_ratio <= 0:
        return records, []
    if eval_ratio >= 1:
        raise SystemExit("--eval-ratio must be < 1.0")
    if len(records) == 1:
        return records, []

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(len(records) * eval_ratio))
    eval_indices = set(indices[:eval_size])
    train_records = [records[i] for i in range(len(records)) if i not in eval_indices]
    eval_records = [records[i] for i in range(len(records)) if i in eval_indices]
    return train_records, eval_records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    train_output = Path(args.train_output)
    eval_output = Path(args.eval_output) if args.eval_output else None

    if not input_path.exists():
        raise SystemExit(f"Input dataset not found: {input_path}")
    _ensure_writable(train_output, overwrite=args.overwrite)
    if eval_output is not None:
        _ensure_writable(eval_output, overwrite=args.overwrite)

    rows = _load_rows(input_path)
    if not rows:
        raise SystemExit(f"No rows found in dataset: {input_path}")

    feature_columns_arg = _parse_columns(args.feature_columns)
    drop_columns_arg = _parse_columns(args.drop_columns)
    feature_columns = _resolve_feature_columns(
        sample_row=rows[0],
        target_column=args.target_column,
        feature_columns=feature_columns_arg,
        drop_columns=drop_columns_arg,
    )

    records: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        record = _build_record(
            row=row,
            feature_columns=feature_columns,
            task_instruction=args.task_instruction,
            system_prompt=args.system_prompt,
            input_title=args.input_title,
            target_column=args.target_column,
            output_style=args.output_style,
            target_prefix=args.target_prefix,
            max_input_fields=args.max_input_fields,
        )
        if record is None:
            if args.drop_empty_rows:
                skipped += 1
                continue
            raise SystemExit(
                "Encountered row with missing target/features. "
                "Use --drop-empty-rows to skip invalid rows."
            )
        records.append(record)

    if not records:
        raise SystemExit("No valid records produced after conversion.")

    if eval_output is None:
        _write_jsonl(train_output, records)
        print(
            f"Converted {len(records)} rows"
            f"{f' (skipped {skipped})' if skipped else ''} -> {train_output}"
        )
        return

    train_records, eval_records = _split_records(
        records=records, eval_ratio=args.eval_ratio, seed=args.seed
    )
    _write_jsonl(train_output, train_records)
    _write_jsonl(eval_output, eval_records)
    print(
        f"Converted {len(records)} rows"
        f"{f' (skipped {skipped})' if skipped else ''}. "
        f"Train: {len(train_records)} -> {train_output}, "
        f"Eval: {len(eval_records)} -> {eval_output}"
    )


if __name__ == "__main__":
    main()
