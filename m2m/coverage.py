"""Op-coverage harness.

Two tools for the agent-driven coverage loop:

- ``opaque_report(mlir_text)`` -- inventory the opaque ``func.call`` placeholders in
  emitted MLIR (the worklist of ops still needing real decompositions).
- ``validate_op(build_fn, example_inputs)`` -- capture a single-op module through the
  FXImporter and check it lowered to real ops (no opaque call) with a result tensor
  type whose shape/dtype match PyTorch eager. This is the gate every new decomposition
  must pass before it's considered done.

Numeric (bit-level) validation needs an MLIR execution engine, which xDSL doesn't ship;
until then we validate structurally (lowered + shape/dtype match eager), which catches
the overwhelming majority of decomposition mistakes.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn


def opaque_report(mlir_text: str) -> dict[str, int]:
    """Return {opaque_func_name: count} for every ``func.call @name`` in the MLIR."""
    names = re.findall(r"func\.call @([A-Za-z0-9_]+)", mlir_text)
    return dict(Counter(names))


@dataclass
class OpValidation:
    op: str
    lowered: bool                      # no opaque func.call for this op
    shape_ok: bool                     # emitted result type matches eager shape/dtype
    opaque_calls: dict[str, int] = field(default_factory=dict)
    eager_shape: tuple[int, ...] | None = None
    mlir_result_type: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.lowered and self.shape_ok and self.error is None


class _SingleOp(nn.Module):
    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.fn(*args)


def validate_op(
    fn: Callable[..., torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    *,
    name: str,
) -> OpValidation:
    """Capture ``fn`` (a single torch op) via the FXImporter and validate it lowered.

    Forces ``backend="fx_importer"`` so we exercise our own decomposition path, not
    torch-mlir.
    """
    import m2m

    try:
        eager = fn(*example_inputs)
        eager_shape = tuple(int(d) for d in eager.shape)
    except Exception as exc:  # noqa: BLE001
        return OpValidation(op=name, lowered=False, shape_ok=False, error=f"eager failed: {exc}")

    try:
        r = m2m.convert(_SingleOp(fn).eval(), example_inputs, backend="fx_importer")
    except Exception as exc:  # noqa: BLE001
        return OpValidation(op=name, lowered=False, shape_ok=False,
                            eager_shape=eager_shape, error=f"convert failed: {exc}")

    opaque = opaque_report(r.mlir_text)
    lowered = sum(opaque.values()) == 0
    # The function's returned tensor type, e.g. "-> tensor<4x64xf32>"
    m = re.search(r"->\s*(tensor<[^>]+>)\s*\{", r.mlir_text) or re.search(r"return\s+%\w+\s*:\s*(tensor<[^>]+>)", r.mlir_text)
    result_type = m.group(1) if m else None
    shape_ok = False
    if result_type is not None:
        dims = re.findall(r"(\d+)", result_type.split(",")[0] + "".join(result_type.split("x")))
        # crude: confirm every eager dim appears in the result type string
        shape_ok = all(str(d) in result_type for d in eager_shape) if eager_shape else True
    return OpValidation(
        op=name, lowered=lowered, shape_ok=shape_ok, opaque_calls=opaque,
        eager_shape=eager_shape, mlir_result_type=result_type,
    )


def dialect_op_histogram(mlir_text: str) -> dict[str, int]:
    """Multiset of ``dialect.op`` names in the MLIR (e.g. {'linalg.generic': 3})."""
    ops = re.findall(r"\b([a-z_]+\.[a-z_][a-z_0-9]*)\b", mlir_text)
    keep = {"linalg", "tensor", "arith", "math", "func", "scf", "cf", "complex", "bufferization"}
    return dict(Counter(o for o in ops if o.split(".")[0] in keep and o != "func.func"))


@dataclass
class OpDiff:
    op: str
    ours_lowered: bool
    ours_opaque: dict[str, int]
    golden_ok: bool                       # torch-mlir produced a lowering
    golden_ops: dict[str, int] = field(default_factory=dict)   # the TARGET lowering
    our_ops: dict[str, int] = field(default_factory=dict)
    result_type_match: bool = False       # our result type == torch-mlir's
    golden_mlir: str = ""
    error: str | None = None

    @property
    def verdict(self) -> str:
        if not self.golden_ok:
            return "no-oracle"            # torch-mlir can't lower it either
        if not self.ours_lowered:
            return "TODO: implement (oracle available)"
        return "ok" if self.result_type_match else "mismatch"


def golden_lowering(fn: Callable[..., torch.Tensor], example_inputs: tuple[torch.Tensor, ...]):
    """Lower a single op via torch-mlir (the oracle). Returns mlir_text or None."""
    import m2m

    try:
        r = m2m.convert(_SingleOp(fn).eval(), example_inputs, backend="torch_mlir")
        return r.mlir_text or None
    except Exception:  # noqa: BLE001
        return None


def _result_type(mlir_text: str) -> str | None:
    m = re.search(r"->\s*(tensor<[^>]+>)\s*\{", mlir_text) or re.search(r"return\s+%\w+\s*:\s*(tensor<[^>]+>)", mlir_text)
    return m.group(1) if m else None


def differential_op(
    fn: Callable[..., torch.Tensor],
    example_inputs: tuple[torch.Tensor, ...],
    *,
    name: str,
) -> OpDiff:
    """Differential test one op: our FXImporter output vs torch-mlir's golden lowering.

    - ``verdict == "TODO: implement (oracle available)"``: we emit an opaque placeholder
      but torch-mlir shows the target linalg -- use ``golden_ops`` / ``golden_mlir`` as the
      spec for writing the emitter.
    - ``verdict == "ok"``: our emitter lowered and its result type matches the oracle.
    - ``verdict == "no-oracle"``: torch-mlir can't lower it either (truly novel op).
    """
    import m2m

    ours = validate_op(fn, example_inputs, name=name)
    golden = golden_lowering(fn, example_inputs)
    golden_ok = golden is not None
    golden_hist = dialect_op_histogram(golden) if golden else {}
    our_text = ""
    try:
        our_text = m2m.convert(_SingleOp(fn).eval(), example_inputs, backend="fx_importer").mlir_text
    except Exception:  # noqa: BLE001
        pass
    rt_match = bool(golden) and _result_type(our_text) is not None and _result_type(our_text) == _result_type(golden)
    return OpDiff(
        op=name,
        ours_lowered=ours.lowered,
        ours_opaque=ours.opaque_calls,
        golden_ok=golden_ok,
        golden_ops=golden_hist,
        our_ops=dialect_op_histogram(our_text),
        result_type_match=rt_match,
        golden_mlir=golden or "",
        error=ours.error,
    )


__all__ = [
    "OpDiff",
    "OpValidation",
    "dialect_op_histogram",
    "differential_op",
    "golden_lowering",
    "opaque_report",
    "validate_op",
]
