# Optional GGUF Export

This project outputs:
- LoRA adapters in `outputs/<run_name>/adapter`
- Optional merged model in `outputs/<run_name>/merged`

To export GGUF, first merge the adapter:

```bash
bash scripts/merge_lora.sh
```

Then use `llama.cpp` conversion tools (outside this repo), for example:

```bash
python convert_hf_to_gguf.py outputs/<run_name>/merged --outfile model.gguf --outtype q8_0
```

Notes:
- Use the merged model directory as input, not the adapter-only directory.
- Choose quantization format based on target hardware (`q4_k_m`, `q5_k_m`, `q8_0`, etc.).
- Conversion scripts and flags can change with `llama.cpp` updates.

