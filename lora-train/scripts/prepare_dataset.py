#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_rows  # noqa: E402
from src.formatting import row_to_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert instruction/chat JSONL or CSV into chat JSONL format."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    rows = load_rows(str(input_path))
    records = []
    for index, row in enumerate(rows):
        try:
            messages = row_to_messages(row, system_prompt=args.system_prompt)
        except Exception as exc:
            raise SystemExit(f"Row {index} conversion failed: {exc}") from exc
        records.append({"messages": messages})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(records)} rows -> {output_path}")


if __name__ == "__main__":
    main()

