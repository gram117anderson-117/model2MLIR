---
description: Lower a JAX function to standard-dialect MLIR (StableHLO) with model2MLIR
argument-hint: <loader.py> [--out file.mlir]
---
Lower a **JAX** function to standard-dialect MLIR (StableHLO) using model2MLIR (`m2m`).

Arguments: `$ARGUMENTS`

The first argument is a path to a Python file exposing
`get_model_and_inputs() -> (callable, tuple[jax.Array, ...])` (a jax function + example
inputs). Then run:

```bash
m2m lower-jax $ARGUMENTS --out model.mlir
```

Guidance:
- Run inside a venv with `jax` **and** `m2m` installed.
- JAX lowers to **StableHLO** (a standard MLIR dialect) via `jax.export`; the same
  `m2m.convert()` entrypoint auto-detects jax callables, so the output is the same
  `ConversionResult` as the torch path (`frontend="jax"`, `output_type="stablehlo"`).
- The reusable artifact is the emitted `.mlir` file. StableHLO → linalg-on-tensors (to
  match the torch path exactly) is `stablehlo-legalize-to-linalg` — planned, drivable by
  the same differential-oracle in `m2m.coverage`.
- Report the output `.mlir` path and byte size.
