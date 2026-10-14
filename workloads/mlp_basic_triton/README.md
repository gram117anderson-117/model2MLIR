# mlp_basic_triton

First Triton-kernel workload. Same MLP and weights as [`mlp_basic`](../mlp_basic/README.md), but
`fc1 -> relu -> fc2 -> relu` is ONE `@triton_op` (`triton_exp::fused_mlp2`, `kernels.py`) and
`fc3` is plain PyTorch.

- `kernels.py` — the `@triton.jit` kernel + `@triton_op` registration (opaque node to `torch.export`).
- `loader.py` — `TritonMLP3`; imports `kernels` so `torch.ops.triton_exp.*` is registered.

Uses the main `model2MLIR/.venv` (torch 2.10, triton 3.6). The `triton` package is only needed to
define/register the op; capture runs on CPU and never executes the kernel. In the emitted MLIR the
kernel is a black-box placeholder `func.call` -- its real IR is swapped in later in the stack, for
the targeted backend (CPU or a custom accelerator), not lowered from Triton here.

## Run
```bash
python workloads/capture.py mlp_basic_triton --formats fp32
```

## Status
Captures at **0 opaque, 1 kernel**. `triton_exp::fused_mlp2` is registered in
`m2m/ir/kernel_registry.py`, so it is emitted as a black-box kernel call (inputs as SSA operands,
output as an SSA result) to a private `func.func @triton_exp__fused_mlp2` declared after
`@forward` and tagged `prov.kernel = "triton_exp::fused_mlp2"`. fc3 lowers exactly as in
`mlp_basic`. merlin replaces the placeholder with a real kernel body it looks up by `prov.kernel`
(`src/merlin/llvmlower/kernel_table.py`). Covered by
`tests/test_kernels.py::test_mlp_basic_triton_mvp`.
