# Workloads

One folder per model. Each holds a `loader.py` exposing `get_model_and_inputs() ->
(nn.Module, tuple[Tensor, ...])` and a `README.md` documenting its environment.

Models pin incompatible stacks, so **each model may need its own venv** (that's why they
live in separate folders). The general recipe for a hard-stack model:

1. Dedicated venv with the model's pinned deps.
2. `uv pip install -e /scratch/agustin/projects/model2MLIR --no-deps`.
3. Capture only the tensor compute (e.g. one flow-matching `denoise_step`); run host-side
   preprocessing eagerly outside the graph.
4. `m2m convert workloads/<model>/loader.py --out workloads/<model>/<model>.mlir`

## Backends

`m2m.convert(model, inputs, backend=...)`:
- `auto` (default): torch-mlir if available, else FXImporter.
- `torch_mlir`: torch-mlir only — lowers to standard dialects (linalg/tensor/arith/func/math/cf)
  with **zero opaque ops** (the correct/complete path).
- `fx_importer`: decomposition-based importer — linalg/tensor/arith/func with **opaque `func.call`s**
  for ops not yet in the decomposition table. Use for the VLAs: torch-mlir's lowering of their
  SmolVLM/PaliGemma vision stacks **OOMs** on this box (an OOM SIGKILL can't be caught to fall back).

## Status (fp32 unless noted)

| Model | venv | backend | result |
| ----- | ---- | ------- | ------ |
| tiny_llama | main `model2MLIR/.venv` (torch 2.10 + torch-mlir) | torch_mlir | fp32: **282 linalg / 0 opaque**; int8: **342 linalg / 0 opaque** (standard dialects). fp8: torch-mlir lacks float8 constants — pending. |
| pi05 (3.62B) | `/scratch/agustin/projects/openpi/.venv` | fx_importer | fp32: 1557 linalg + 8786 opaque (torch-mlir OOMs). int8/fp8: pending. |
| smolvla (0.45B) | `/scratch/agustin/projects/smolvla_capture/.venv` | fx_importer | fp32: 605 linalg + 5206 opaque (torch-mlir OOMs). int8/fp8: pending. |

## Known remaining work
- Cut VLA opaque counts: register decompositions (→ linalg/tensor/arith) for the high-frequency
  opaque ops (softmax, layernorm/rms_norm, gelu/silu, scaled_dot_product_attention, embedding,
  rsqrt, ...) in `m2m/ir/decompositions.py` (FXImporter coverage path, since torch-mlir OOMs here).
- VLA int8/fp8 capture (`convert(quantization=...)`).
- fp8: torch-mlir can't emit float8 constants; xDSL 0.65 has no native float8 `AnyFloat` for the
  FXImporter linalg path — needs proper float8 type support.
- A matching torch-mlir for torch 2.7.1 + more memory would let the VLAs use `backend="torch_mlir"`.
