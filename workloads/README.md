# Workloads — the standardized model→MLIR capture flow

One command captures any model to MLIR in every datatype/quant format:

```bash
python workloads/capture.py <model>                 # fp32 + int8 + fp8
python workloads/capture.py <model> --formats fp32  # just one
python workloads/capture.py <model> --level high-level
python workloads/capture.py --all                   # every model
python workloads/capture.py --list                  # what's available
```

Output: `workloads/<model>/<model>.mlir` (fp32), `_int8.mlir`, `_fp8.mlir` — each asserted
to lower to **0 opaque ops**. `.mlir` files are gitignored (build artifacts).

## What a model directory contains

```
workloads/<model>/
├── loader.py        # REQUIRED: get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])
├── capture.toml     # OPTIONAL: venv / deps / env / upstream config
└── <model>*.mlir    # OUTPUT (gitignored)
```

### `loader.py` — the one contract
```python
def get_model_and_inputs() -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    """Build the model and example inputs such that model(*inputs) runs eagerly."""
```
Rules that make capture stable and scalable:
- **Small config, no big/gated downloads.** Build the real architecture from a *reduced*
  config (fewer layers/hidden/vocab) and random init — capture is about the op set, not weights.
- **Capture ONE unit.** A single `forward` / denoise step — not generation loops or sampling.
- **`sys.path` the upstream repo** (cloned at `/scratch/agustin/projects/<Repo>`, a sibling of
  `model2MLIR`). Stub out heavy side-effect imports if the package `__init__` pulls the world.
- **Eager attention**, `use_cache=False` — keep the traced graph export-friendly.

### `capture.toml` — environment (optional)
```toml
venv   = ".venv"          # default: workloads/<model>/.venv (the STANDARD location).
                          # may be an absolute path to reuse an existing dedicated venv.
python = "3.11"           # python version for a fresh venv
deps   = ["torch==2.7.1", "transformers==4.53.3", ...]   # pip specs for a fresh venv
upstream = "/scratch/agustin/projects/<Repo>"            # documentation
[env]                     # env vars set for the capture (e.g. small-config knobs)
M2M_<MODEL>_LAYERS = "4"
```
The driver always also installs `xdsl structlog ml_dtypes torchao` and the `m2m` package
(editable, `--no-deps`) into the venv.

## How the driver works
1. Reads `capture.toml`, resolves the model's dedicated venv (default
   `workloads/<model>/.venv`; builds it from `deps`/`python` if missing — idempotent).
2. Runs the capture **inside that venv** (subprocess) so each model's conflicting
   torch/transformers/torchao versions stay isolated.
3. For each format, calls `m2m.convert(model, inputs, backend="fx_importer",
   quantization=..., level=...)`, writes the `.mlir`, and checks 0 opaque ops.

The two output levels (see `docs/OP_TAXONOMY.md`): `linalg-on-tensors` (default, portable
standard dialects) and `high-level` (opt-in `linalg_ext.*` named ops; `m2m.expand_to_linalg`
lowers it back). Quantization is QDQ-preserved by default (`quant_ext.dequantize` on the
weight); fp8 renders the native `f8E4M3FN` spelling.

## Adding a new model (the whole recipe)
1. `git clone <upstream>` → `/scratch/agustin/projects/<Repo>`.
2. `mkdir workloads/<model>` ; write `loader.py` (the contract above).
3. Write `capture.toml` (deps + any small-config env vars).
4. `python workloads/capture.py <model>` → three `.mlir` files at 0 opaque.

If a new op shows up opaque, add its decomposition (one entry in
`m2m/ir/decompositions.py::DECOMPOSITION_TABLE` + a family in
`m2m/ir/import_fx.py::_FAMILY_OF`) — see `docs/OP_TAXONOMY.md`. Brand-new architectures so
far have needed at most one new op each (RDT=`squeeze`, GR00T=`bitwise_not`, xr0=`repeat`,
BitVLA=`mean.default`).

## Models & status
Each has a `loader.py` + `capture.toml`; per-model env notes in `workloads/<model>/README.md`.
```bash
python workloads/capture.py --all     # capture/refresh all 10 in fp32+int8+fp8
```

| Model | fp32 | int8 | fp8 | notes |
|---|---|---|---|---|
| tiny_llama | 0 | 0 | 0 | |
| smolvla | 0 | 0 | 0 | |
| rdt | 0 | 0 | 0 | |
| rdt2 | 0 | 0 | 0 | |
| molmoact | 0 | 0 | 0 | |
| openvla | 0 | 0 | 0 | real DINO+SigLIP+Llama stack, no 14 GB download |
| xr0 | 0 | 0 | 0 | |
| bitvla | 0 | 0 | 0 | int8/fp8 quantize lm_head only (BitNet W1.58 stays) |
| groot_n1d7 | 0 | 0 | 0 | |
| pi05 | 0 | 357* | 357* | *torchao-0.11 (openpi's torch-2.7.1 pin) retains SDPA + a few ops on the **quantized** export, which doesn't route through the gap-decomp table. fp32 is 0 opaque. Fix: newer torch/torchao in the openpi venv, or wire the gap decomps into the quant export path. |

"0" = 0 opaque ops. Re-run `capture.py <model>` to regenerate after converter changes.
