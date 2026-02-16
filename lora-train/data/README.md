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

