"""model2mlir: a self-updating PyTorch / torchAO -> MLIR frontend.

Built on torch-mlir (primary lowering path) with a decomposition-based fallback
and a self-updating op-coverage loop that grows support for new and torchAO ops.
Emits linalg-on-tensors by default.

Quick start:

    import model2mlir
    result = model2mlir.convert(model, (example_input,))
    print(result.mlir_text)        # linalg-on-tensors MLIR
    print(result.path_taken)       # "torch_mlir" or "fx_importer"
"""

from __future__ import annotations

from model2mlir.api import ConversionResult, convert, coverage_report
from model2mlir.capture.torch_mlir_bridge import torch_mlir_available

__version__ = "0.0.1"

__all__ = [
    "ConversionResult",
    "__version__",
    "convert",
    "coverage_report",
    "torch_mlir_available",
]
