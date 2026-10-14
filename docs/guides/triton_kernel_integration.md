# Triton kernel integration (model2MLIR side)

> **Work in progress.** One registered kernel (`triton_exp::fused_mlp2`) on one toy model is captured,
> consumed by merlin and run on spike rv64gcv, matching PyTorch. Kernels that write into their
> arguments are not supported yet — see [Limitations](#limitations).

# THIS DOC IS TEMP AND ALSO SERVES A STATUS WILL BE UPDATED LATER

## What it does

A model can call a hand-written kernel, e.g. a `@torch.library.triton_op`. `torch.export` keeps each
such call as one custom-op node. model2MLIR now keeps a **global kernel table**, `KERNEL_REGISTRY` in
`m2m/ir/kernel_registry.py` (the sibling of `DECOMPOSITION_TABLE`). A custom op listed there is
emitted as a **black-box kernel call** — a typed `func.call` to a private declaration — instead of an
opaque op:

- tensor inputs become SSA operands, in the op schema's argument order;
- every output becomes an SSA result;
- non-tensor arguments (scales, block sizes, omitted defaults) are recorded as an attribute;
- the call is tagged with a stable identity, `prov.kernel = "<ns>::<name>"`.

model2MLIR never looks inside the kernel. The declaration is a placeholder that a downstream compiler
binds to a real implementation; merlin does this by looking up `prov.kernel` in its own kernel table
and linking a CPU body into the model's LLVM IR. Triton is only how the kernel is authored — capture
needs no GPU, and the kernel is never executed during export.

## End-to-end: the MLP toy example

Two workloads, the same 3-layer MLP (16 → 32 → 32 → 8) with the same seed, weights and input:

| workload | what |
|---|---|
| `workloads/mlp_basic/` | plain PyTorch: `fc1 → relu → fc2 → relu → fc3` |
| `workloads/mlp_basic_triton/` | fc1+relu+fc2+relu is one Triton op, `torch.ops.triton_exp.fused_mlp2` (`kernels.py`); fc3 is plain PyTorch |

### 1. Setup

```bash
uv venv && uv pip install -e .
uv pip install triton        # only to define/register the @triton_op; no GPU needed
```

Both workloads use the repo's own venv (`capture.toml`: `venv = "../../.venv"`).

### 2. Capture

```bash
python workloads/capture.py mlp_basic_triton --formats fp32
python workloads/capture.py mlp_basic        --formats fp32
```

```
OK mlp_basic_triton fp32  opaque=0    kernels=1   linalg=5     families=4   -> workloads/mlp_basic_triton/mlp_basic_triton.mlir
OK mlp_basic      fp32  opaque=0    kernels=0   linalg=19    families=4   -> workloads/mlp_basic/mlp_basic.mlir
```

Each run writes `workloads/<name>/<name>.mlir` plus `<name>.safetensors` and its `.manifest.json` (the
weights, keyed by func-argument index). These are gitignored build outputs.

The same from Python:

```python
import sys; sys.path.insert(0, "workloads/mlp_basic_triton")
from loader import get_model_and_inputs
import m2m
from m2m.coverage import kernel_report, opaque_report

model, inputs = get_model_and_inputs()
r = m2m.convert(model, inputs)        # backend="auto"
r.path_taken                          # "fx_importer" -- torch-mlir is skipped when a kernel is present
opaque_report(r.mlir_text)            # {}
kernel_report(r.mlir_text)            # {"triton_exp::fused_mlp2": 1}
```

### 3. The emitted IR

`workloads/mlp_basic_triton/mlp_basic_triton.mlir`:

```mlir
builtin.module attributes {prov.weights_file = ".../mlp_basic_triton.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32x16xf32>, %1: tensor<32xf32>, %2: tensor<32x32xf32>, %3: tensor<32xf32>,
                     %4: tensor<8x32xf32>, %5: tensor<8xf32>, %6: tensor<4x16xf32>) -> tensor<4x8xf32> {
    %7 = func.call @triton_exp__fused_mlp2(%6, %0, %1, %2, %3) {prov.kernel = "triton_exp::fused_mlp2",
           prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"], prov.region_id = "triton_exp__fused_mlp2_0",
           prov.op = "kernel", prov.family = "kernel", prov.aten = "triton_exp.fused_mlp2.default", ...}
         : (tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>) -> tensor<4x32xf32>
    ... fc3: linalg.transpose / linalg.fill / linalg.matmul / bias linalg.generic, identical to mlp_basic ...
    func.return %15 : tensor<4x8xf32>
  }
  func.func private @triton_exp__fused_mlp2(tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>,
                                            tensor<32xf32>) -> tensor<4x32xf32> attributes {prov.kernel = "triton_exp::fused_mlp2"}
}
```

Compared with `mlp_basic.mlir`: the same `@forward` signature (7 arguments in the same order), the
same weights manifest, and fc3 op-for-op identical — only fc1..relu2 is replaced by the one call.

**The contract a consumer reads** (nothing else is emitted for a kernel):

| attribute | on | meaning |
|---|---|---|
| `prov.kernel` | call + declaration | identity, `"<ns>::<name>"`. The callee symbol is `<ns>__<name>`, with a `_<n>` suffix per new shape, so match on this attribute, never the symbol. |
| `prov.kernel_operands` | call | schema argument name of each SSA operand, in order |
| `prov.kernel_args` | call | non-tensor arguments by name (an absent optional is `unit`); **omitted** when there are none |

Plus the usual provenance tags every op gets (`prov.op` / `prov.family = "kernel"`, `prov.region_id`,
`prov.aten`, `prov.orig_dtype`). Declarations are placed after `@forward`, so the entry function is
always the first `func.func`.

### 4. Downstream (merlin)

merlin consumes this IR as-is: its kernel table maps `triton_exp::fused_mlp2` to CPU bodies, links one
into the model's LLVM IR and runs it on spike rv64gcv. Measured there against PyTorch: max abs error
3.0e-8 with the kernel (`llvm_mlir` body, 15,558 retired instructions) and 4.5e-8 for `mlp_basic`
(16,156). In merlin, `build_tools/scripts/run_mlp_kernel_spike.py --form llvm_mlir` runs it; see
merlin's `docs/guides/triton_kernel_integration.md`.

### 5. Tests

```bash
python -m pytest tests/test_kernels.py
```

13 tests. Most register plain CPU `torch.library.custom_op`s with fake implementations as kernels, so
they need neither Triton nor a GPU; `test_mlp_basic_triton_mvp` captures the real workload and needs
the `triton` package.

### Adding a kernel

Append a `KernelSpec` to `_BUILTIN_KERNELS` in `m2m/ir/kernel_registry.py`:

```python
KernelSpec("my_ns::my_kernel", description="what it computes, which workload uses it"),
```

The qualname matches every overload of `torch.ops.my_ns.my_kernel`. A consumer needs a matching
entry (merlin: `src/merlin/llvmlower/kernel_table.py`; its `test_table_keys_match_m2m_registry` fails
until it has one). An unregistered Triton op still lowers as an opaque call, with a diagnostic that
points at the registry.

## Changes made

Branch `feat/triton-integration` on `main`, pushed to ucb-bar/model2MLIR. The code changes are these 4 commits (plus this page):

| commit | change |
|---|---|
| `feat(ir): lower registered custom kernels to black-box func.calls` | `m2m/ir/kernel_registry.py` (new); FXImporter branch in `m2m/ir/import_fx.py` (before the decomposition table; declarations after `@forward`; `kernel` family; kernel identity kept inside `while_loop` bodies; diagnostic for unregistered Triton ops; import error for a registered kernel that mutates its arguments); `m2m/coverage.py` (`opaque_report` excludes kernel calls, new `kernel_report`, `kernel` in `CANONICAL_FAMILIES`); `m2m/capture/torch_export.py` (registered kernels count as supported in the unsupported-op loop); `m2m/api.py` (`backend="auto"` skips torch-mlir when a kernel is present); `m2m/capture/torch_mlir_bridge.py` (kernel-call count in diagnostics); `m2m/ir/__init__.py` exports |
| `fix(transforms): keep kernel calls intact through sections and int2 fusion` | `split_by_section` took a body-less declaration as the entry point whenever a module had opaque or kernel calls, and returned no sections — it now picks the first function with a body and re-declares each section's callees; `fuse_int2_unpack` no longer matches kernel calls by callee substring |
| `feat(workloads): add mlp_basic and mlp_basic_triton kernel workloads` | the two workloads; `workloads/capture.py` prints `kernels=N` |
| `docs: document the kernel registry and its IR contract` | `AGENTS.md` (custom-kernels section), `docs/OP_TAXONOMY.md` (attributes, `kernel` family), `workloads/README.md` |

How the IR contract was chosen: an earlier draft also emitted `prov.kernel_backend`,
`prov.triton_jit` and `prov.kernel_source`, and an empty `prov.kernel_args = {}`. merlin read none of
them — where a kernel's implementation lives belongs to the consumer's table — so they were removed.
The contract is now exactly what merlin reads.

## Limitations

| gap | today | plan |
|---|---|---|
| Kernels that write into their arguments (`mutates_args`, e.g. an `out=` buffer) | `torch.export` wraps them in `auto_functionalized_v2(op, **kwargs, _<arg>_base_index, _all_bases=[...])`; a registered one is an **import error**; an unregistered one goes through the generic opaque path, which loses its operands (pre-existing behavior, not yet flagged) | rebuild the schema arguments (mutated buffer from `_all_bases[...]`); emit destination-passing style — the buffer's initial value as an operand, results = the op's returns + one new value per mutated argument, tagged `prov.kernel_mutated` / `prov.kernel_tied`; remap `getitem`; refuse view arguments (`_<arg>_size` / `_stride`). Probe the `meta["val"]` slot layout for ops that both return and mutate first |
| Unregistered custom ops | still the generic opaque path: non-tensor args dropped, only the first output kept | register them, or extend the same binding to the opaque path |
| Symbols per shape only | two calls with the same shapes but different `prov.kernel_args` share a symbol | key symbols by args too, once a consumer passes `kernel_args` |
| Registration | one global built-in table; no per-workload registration | by design for now |
| Dynamic shapes | dims become `-1` like any op; no consumer handles a dynamic kernel yet | — |

## Current state

- **Branch.** `feat/triton-integration` on ucb-bar/model2MLIR (also on the `gram117anderson-117` fork),
  based on `main` `905d2c9`. No PR is open.
- **Tests.** `tests/test_kernels.py` 13 passed; full suite 731 passed, 304 skipped. Each of the 4
  commits passes the kernel tests on its own.
- **No regression for kernel-free models.** Captures of `mlp_basic`, `tiny_llama` and `whisper_tiny` are
  byte-identical to `main`.
- **Cross-repo checks.** merlin's `kernel_link.find_kernel_uses` and `_check_use` accept the emitted IR
  for both of its body forms; merlin's `test_mlp_bundles_fresh.py` re-captures both workloads from this
  checkout and requires its tracked copies to match byte for byte.
- **Unrelated local work** (a conv `padding="valid"` test, a local pi05 setup, `efficient_wam` /
  `small_llama` configs) is kept apart on the local branch `wip/pi05-local-workloads`.
- **Next.** In-place kernels (the plan above), then per-call `prov.kernel_args` once merlin can pass them.
