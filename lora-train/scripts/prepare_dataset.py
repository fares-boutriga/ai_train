#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_rows  # noqa: E402
from src.formatting import row_to_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert instruction/chat JSONL or CSV into chat JSONL format. "
            "Can also map arbitrary tabular columns (question/answer style) "
            "to instruction/input/output."
        )
    )
    parser.add_argument("--input", required=True, help="Input .jsonl or .csv file path.")
    parser.add_argument("--output", required=True, help="Output .jsonl file path.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt prepended when not already present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--instruction-column",
        default="instruction",
        help=(
            "Source column for instruction when converting generic tabular rows. "
            "Defaults to `instruction`."
        ),
    )
    parser.add_argument(
        "--output-column",
        default="output",
        help=(
            "Source column for assistant answer when converting generic tabular rows. "
            "Defaults to `output`."
        ),
    )
    parser.add_argument(
        "--input-columns",
        default="input",
        help=(
            "Comma-separated columns to pack into the instruction `input` section "
            "(e.g. `context,category,metadata`). Use empty string to disable."
        ),
    )
    parser.add_argument(
        "--include-other-columns-as-input",
        action="store_true",
        help=(
            "Include all non-empty columns except instruction/output columns "
            "in the input section."
        ),
    )
    parser.add_argument(
        "--drop-empty-rows",
        action="store_true",
        help=(
            "Drop rows where mapped instruction/output are empty instead of failing."
        ),
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
    raw = [chunk.strip() for chunk in value.split(",")]
    return [chunk for chunk in raw if chunk]


def _map_tabular_row(
    row: dict[str, Any],
    instruction_column: str,
    output_column: str,
    input_columns: list[str],
    include_other_columns_as_input: bool,
) -> dict[str, str]:
    instruction = _to_text(row.get(instruction_column))
    output = _to_text(row.get(output_column))

    columns_for_input = list(input_columns)
    if include_other_columns_as_input:
        excluded = {instruction_column, output_column, *input_columns}
        for key in row.keys():
            if key not in excluded:
                columns_for_input.append(key)

    seen: set[str] = set()
    input_lines: list[str] = []
    for column in columns_for_input:
        if column in seen:
            continue
        seen.add(column)
        value = _to_text(row.get(column))
        if value:
            input_lines.append(f"{column}: {value}")

    return {
        "instruction": instruction,
        "input": "\n".join(input_lines).strip(),
        "output": output,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    rows = load_rows(str(input_path))
    input_columns = _parse_columns(args.input_columns)
    mapping_requested = (
        args.instruction_column != "instruction"
        or args.output_column != "output"
        or args.input_columns.strip() != "input"
        or args.include_other_columns_as_input
    )
    records = []
    skipped = 0
    for index, row in enumerate(rows):
        try:
            source_row = row
            if "messages" not in row and mapping_requested:
                source_row = _map_tabular_row(
                    row=row,
                    instruction_column=args.instruction_column,
                    output_column=args.output_column,
                    input_columns=input_columns,
                    include_other_columns_as_input=args.include_other_columns_as_input,
                )
            messages = row_to_messages(source_row, system_prompt=args.system_prompt)
        except Exception as exc:
            if args.drop_empty_rows:
                skipped += 1
                continue
            raise SystemExit(f"Row {index} conversion failed: {exc}") from exc
        records.append({"messages": messages})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if skipped:
        print(f"Converted {len(records)} rows (skipped {skipped}) -> {output_path}")
    else:
        print(f"Converted {len(records)} rows -> {output_path}")


if __name__ == "__main__":
    main()
