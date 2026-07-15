"""IR construction: FX -> xDSL linalg-on-tensors, plus the growable
aten/torchAO decomposition table.

`DECOMPOSITION_TABLE` is the registry the self-updating coverage loop grows:
torch-mlir handles breadth; entries here cover what torch-mlir lacks (torchAO,
new ops). `FXImporter` is the fallback path used when torch-mlir is absent.

These names are resolved **lazily** (PEP 562) so that importing the pure-xDSL
``m2m.ir.quant`` dialect does not pull in the torch-bound FX importer /
decomposition table.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY: dict[str, str] = {
    "DECOMPOSITION_TABLE": "m2m.ir.decompositions",
    "DecompResult": "m2m.ir.decompositions",
    "FXImporter": "m2m.ir.import_fx",
    "fx_to_xdsl": "m2m.ir.import_fx",
}

__all__ = ["DECOMPOSITION_TABLE", "DecompResult", "FXImporter", "fx_to_xdsl"]


def __getattr__(name: str):  # PEP 562
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'm2m.ir' has no attribute {name!r}")
    obj = getattr(importlib.import_module(target), name)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


if TYPE_CHECKING:
    from m2m.ir.decompositions import DECOMPOSITION_TABLE, DecompResult
    from m2m.ir.import_fx import FXImporter, fx_to_xdsl
