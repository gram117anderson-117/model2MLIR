# mlp_basic

3-layer MLP (`fc1 -> relu -> fc2 -> relu -> fc3`, 16 -> 32 -> 32 -> 8) in plain PyTorch, CPU.
No Triton. This is the reference for [`mlp_basic_triton`](../mlp_basic_triton/README.md): same
seed (0), same weights, same input.

## Run
```bash
python workloads/capture.py mlp_basic
```
Uses the main `model2MLIR/.venv`.
