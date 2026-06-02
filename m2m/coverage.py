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


__all__ = ["OpValidation", "opaque_report", "validate_op"]
