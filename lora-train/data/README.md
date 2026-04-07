# Dataset Formats

Supported input formats:

1. Instruction JSONL
```json
{"instruction":"...", "input":"optional", "output":"..."}
```

2. Chat JSONL
```json
{"messages":[
  {"role":"system","content":"..."},
  {"role":"user","content":"..."},
  {"role":"assistant","content":"..."}
]}
```

3. CSV with either:
- columns `instruction,input,output`
- or a `messages` column containing a JSON list

Use `scripts/prepare_dataset.py` to convert instruction-style rows to chat JSONL.

For generic tabular datasets (custom columns), use:
```bash
python scripts/prepare_dataset.py \
  --input data/raw.csv \
  --output data/train_chat.jsonl \
  --instruction-column question \
  --output-column answer \
  --input-columns context,metadata \
  --include-other-columns-as-input \
  --drop-empty-rows \
  --overwrite
```

For BIM risk prediction from tabular CSV:
```bash
python scripts/prepare_bim_dataset.py \
  --input "/c/Users/boutr/Downloads/bim_ai_civil_engineering_dataset.csv" \
  --train-output data/bim_train_chat.jsonl \
  --eval-output data/bim_eval_chat.jsonl \
  --target-column Risk_Level \
  --drop-empty-rows \
  --overwrite
```
