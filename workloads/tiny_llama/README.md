# TinyLlama

Causal LM (TinyLlama-1.1B architecture). Weights from the local HF cache.

## Venv
Uses the main `model2MLIR/.venv` (torch 2.10 + transformers + torch-mlir).

## Run
```bash
cd /scratch/agustin/projects/model2MLIR
# fast smoke: fewer layers
M2M_LLAMA_LAYERS=2 uv run --no-sync m2m convert workloads/tiny_llama/loader.py --out workloads/tiny_llama/tinyllama.mlir
# int8
M2M_LLAMA_LAYERS=2 uv run --no-sync m2m convert workloads/tiny_llama/loader.py --quant int8_weight_only --out workloads/tiny_llama/tinyllama_int8.mlir
```
Env: `M2M_LLAMA_LAYERS=N` (truncate layers), `M2M_SEQ=N` (sequence length).

## Status
- fp32: 282 `linalg`, 0 opaque.
- int8: 342 `linalg`, 0 opaque (quantization preserved).
- fp8: captures; blocked at xDSL float8 element typing.
