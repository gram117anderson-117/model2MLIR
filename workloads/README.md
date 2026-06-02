# Workloads

One folder per model. Each holds a `loader.py` exposing `get_model_and_inputs() ->
(nn.Module, tuple[Tensor, ...])` and a `README.md` documenting its environment.

Models pin incompatible stacks, so **each model may need its own venv** (that's why they
live in separate folders). The general recipe for a hard-stack model:

1. Dedicated venv with the model's pinned deps.
2. `uv pip install -e /scratch/agustin/projects/model2MLIR --no-deps` (FXImporter works without
   torch-mlir; install the torch-mlir wheel too if a matching cp/torch wheel exists).
3. Capture only the tensor compute (e.g. one flow-matching `denoise_step`); run host-side
   preprocessing eagerly outside the graph.
4. `model2mlir convert workloads/<model>/loader.py --out workloads/<model>/<model>.mlir`

## Status (fp32 unless noted)

| Model | venv | result |
| ----- | ---- | ------ |
| tiny_llama | main `model2MLIR/.venv` (torch 2.10 + torch-mlir) | fp32: 282 linalg / 0 opaque; int8: 342 linalg / 0 opaque; fp8: blocked (xDSL float8 typing) |
| pi05 | `/scratch/agustin/projects/openpi/.venv` (torch 2.7.1, transformers 4.53.2 + transformers_replace, jax) | fp32: 1557 linalg captured (FXImporter; torch-mlir gap ops pending) |
| smolvla | `/scratch/agustin/projects/smolvla_capture/.venv` (lerobot 0.5.1, transformers 5.3.0, torch-mlir) | fp32: 605 linalg captured (FXImporter; torch-mlir aborts on gap op `aten.empty_permuted` → decomposition pending) |

## Known remaining work
- VLA int8/fp8 capture (apply torchao via `convert(quantization=...)`).
- Cut VLA opaque counts by registering torch-mlir-friendly decompositions for gap ops
  (e.g. `aten.empty_permuted` → `aten.empty`); each fix may reveal the next gap op.
- fp8 element-type support (xDSL 0.65 lacks a native float8 `AnyFloat`).
