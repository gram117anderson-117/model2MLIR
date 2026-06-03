"""IR transforms over the m2m representation (distinct from import-time decompositions).

- ``expand_to_linalg``: lower the opt-in high-level form (``linalg_ext.*`` named ops) to the
  default portable standard-dialect form. The inverse of emitting named ops at import.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp, StringAttr

from m2m.transforms.expand_ext import EXPANDERS, expand_to_linalg
from m2m.transforms.expand_quant import expand_quant_to_standard
from m2m.transforms.fuse_qdq import fuse_qdq
from m2m.transforms.sections import split_by_section


def to_standard(module: ModuleOp) -> ModuleOp:
    """Lower ALL m2m extension ops to pure upstream-MLIR core dialects: expand the high-level
    linalg_ext.* named ops AND the quant_ext.* QDQ ops to linalg/scf/tensor/arith/math. The
    result uses only standard dialects (the "purely MLIR supported" form); any remaining
    non-core op is an opaque func.call for a still-unconverted aten op."""
    expand_to_linalg(module)
    expand_quant_to_standard(module)
    try:
        module.attributes["m2m.level"] = StringAttr("standard")
    except Exception:  # noqa: BLE001
        pass
    return module


__all__ = ["EXPANDERS", "expand_quant_to_standard", "expand_to_linalg", "fuse_qdq",
           "split_by_section", "to_standard"]
