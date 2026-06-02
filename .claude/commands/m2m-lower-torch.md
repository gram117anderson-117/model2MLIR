---
description: Lower a PyTorch model to standard-dialect MLIR (linalg-on-tensors) with model2MLIR
argument-hint: <loader.py> [--quant int8_weight_only|float8_weight_only_e4m3] [--backend auto|torch_mlir|fx_importer] [--out file.mlir]
---
Lower a **PyTorch** model to standard-dialect MLIR using model2MLIR (the `m2m` package).

Arguments: `$ARGUMENTS`

The first argument is a path to a Python file exposing
`get_model_and_inputs() -> (torch.nn.Module, tuple[Tensor, ...])`
(see `workloads/tiny_llama/loader.py` for the convention). Then run:

```bash
m2m lower-torch $ARGUMENTS
```

Guidance:
- Run inside the venv that has the model's deps **and** `m2m` installed
  (`uv pip install -e /path/to/model2MLIR --no-deps`). For in-repo workloads, the right
  venv + command are documented in `workloads/<model>/README.md`.
- Backends: `auto` tries torch-mlir (0-opaque when it works) then the FXImporter; pass
  `--backend fx_importer` for vision-heavy VLAs where torch-mlir OOMs.
- Quantization is torchAO (`--quant int8_weight_only`, `--quant float8_weight_only_e4m3`).
- After running, report: `frontend/path_taken`, `output_type`, the `linalg.*` vs opaque
  `func.call` counts (use `m2m.coverage.opaque_report`), and the output `.mlir` path.
