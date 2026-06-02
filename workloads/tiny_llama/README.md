# TinyLlama

Causal LM (TinyLlama-1.1B architecture). Weights from the local HF cache.

## Venv
Uses the main `model2MLIR/.venv` (torch 2.10 + transformers + torch-mlir).

## Run
```bash
cd /scratch/agustin/projects/model2MLIR
# fast smoke: fewer layers
TORCH2MLIR_LLAMA_LAYERS=2 uv run --no-sync model2mlir convert workloads/tiny_llama/loader.py --out workloads/tiny_llama/tinyllama.mlir
# int8
TORCH2MLIR_LLAMA_LAYERS=2 uv run --no-sync model2mlir convert workloads/tiny_llama/loader.py --quant int8_weight_only --out workloads/tiny_llama/tinyllama_int8.mlir
```
Env: `TORCH2MLIR_LLAMA_LAYERS=N` (truncate layers), `TORCH2MLIR_SEQ=N` (sequence length).

## Status
- fp32: 282 `linalg`, 0 opaque.
- int8: 342 `linalg`, 0 opaque (quantization preserved).
- fp8: captures; blocked at xDSL float8 element typing.
