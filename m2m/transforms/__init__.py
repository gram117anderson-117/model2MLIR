"""IR transforms over the m2m representation (distinct from import-time decompositions).

- ``expand_to_linalg``: lower the opt-in high-level form (``linalg_ext.*`` named ops) to the
  default portable standard-dialect form. The inverse of emitting named ops at import.
"""

from __future__ import annotations

from m2m.transforms.expand_ext import EXPANDERS, expand_to_linalg

__all__ = ["EXPANDERS", "expand_to_linalg"]
