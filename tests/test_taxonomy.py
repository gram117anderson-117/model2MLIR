"""Scalability governance: the matchable op vocabulary stays small, fixed, and complete.

These guard the invariant that downstream passes match a bounded set of families/op-kinds
rather than inspecting bespoke linalg.generic bodies -- and that no op escapes tagging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import m2m
from m2m.coverage import (
    CANONICAL_FAMILIES,
    family_histogram,
    op_vocabulary,
    untagged_compute_ops,
)


class _Net(nn.Module):
    """Exercises many families: matmul/linear (contraction), softmax + layer_norm
    (normalization), relu/add/mul (elementwise), mean (reduce), transpose, etc."""

    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(32)
        self.q = nn.Linear(32, 32)
        self.k = nn.Linear(32, 32)
        self.v = nn.Linear(32, 32)
        self.o = nn.Linear(32, 32)

    def forward(self, x):
        x = self.ln(x)
        q, k, v = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / 5.0, dim=-1)
        out = self.o(attn @ v)
        return torch.relu(out) + x.mean(-1, keepdim=True)


def _capture():
    return m2m.convert(_Net().eval(), (torch.randn(2, 8, 32),), backend="fx_importer")


def test_no_untagged_generics():
    """Every linalg.generic carries an prov.family tag (named ops are matchable by type)."""
    r = _capture()
    assert r.ok
    assert untagged_compute_ops(r.mlir_text) == 0


def test_families_within_canonical_vocabulary():
    """The whole network collapses into the small, fixed coarse vocabulary -- no
    proliferation of bespoke families."""
    r = _capture()
    fams = set(family_histogram(r.mlir_text))
    extra = fams - CANONICAL_FAMILIES
    assert not extra, f"non-canonical families introduced: {extra}"


def test_vocabulary_is_small():
    """A whole transformer block matches a handful of families, not one-per-op."""
    r = _capture()
    vocab = op_vocabulary(r.mlir_text)
    assert len(vocab) <= len(CANONICAL_FAMILIES)
    # contraction (matmul/linear) and normalization (softmax/layer_norm) must be present
    assert "contraction" in vocab
