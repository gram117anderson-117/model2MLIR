"""IR construction: FX -> xDSL linalg-on-tensors, plus the growable
aten/torchAO decomposition table.

`DECOMPOSITION_TABLE` is the registry the self-updating coverage loop grows:
torch-mlir handles breadth; entries here cover what torch-mlir lacks (torchAO,
new ops). `FXImporter` is the fallback path used when torch-mlir is absent.
"""

from __future__ import annotations

from model2mlir.ir.decompositions import DECOMPOSITION_TABLE, DecompResult
from model2mlir.ir.import_fx import FXImporter, fx_to_xdsl

__all__ = ["DECOMPOSITION_TABLE", "DecompResult", "FXImporter", "fx_to_xdsl"]
