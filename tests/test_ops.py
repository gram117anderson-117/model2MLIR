"""Regression tests: ops that must lower to real standard-dialect MLIR (no opaque).

Gated by the torch-mlir differential oracle where available; falls back to a structural
check (lowered + no opaque func.call) otherwise.
"""

from __future__ import annotations

import pytest
import torch

from m2m.coverage import validate_op

X = (torch.randn(4, 8),)
XY = (torch.randn(4, 8), torch.randn(4, 8))
XPOS = (torch.randn(4, 8).abs() + 1,)

CASES = [
    ("view", lambda a: a.view(2, 16), X),
    ("reshape", lambda a: a.reshape(8, 4), X),
    ("unsqueeze", lambda a: a.unsqueeze(1), X),
    ("flatten", lambda a: a.flatten(), X),
    ("mul", lambda a, b: a * b, XY),
    ("add", lambda a, b: a + b, XY),
    ("sub", lambda a, b: a - b, XY),
    ("div", lambda a, b: a / b, XY),
    ("neg", lambda a: -a, X),
    ("rsqrt", torch.rsqrt, XPOS),
    ("sigmoid", torch.sigmoid, X),
    ("silu", torch.nn.functional.silu, X),
    ("cos", torch.cos, X),
    ("sin", torch.sin, X),
    ("to_f16", lambda a: a.to(torch.float16), X),
    ("exp", torch.exp, X),
    ("sqrt", lambda a: torch.sqrt(a.abs() + 1), X),
    ("tanh", torch.tanh, X),
    ("abs", torch.abs, X),
    ("floor", torch.floor, X),
    ("log", lambda a: torch.log(a.abs() + 1), X),
    ("gelu", torch.nn.functional.gelu, X),
    ("permute3d", lambda a: a.permute(0, 2, 1), (torch.randn(2, 3, 4),)),
    ("clone", torch.clone, X),
    ("mul_bcast", lambda a, b: a * b, (torch.randn(4, 8), torch.randn(8))),
    ("mm", torch.mm, (torch.randn(4, 8), torch.randn(8, 16))),
    ("addmm", lambda b, x, w: torch.addmm(b, x, w), (torch.randn(16), torch.randn(4, 8), torch.randn(8, 16))),
    ("relu", torch.relu, X),
    ("clamp", lambda a: a.clamp(-1, 1), X),
    ("maximum", torch.maximum, XY),
    ("minimum", torch.minimum, XY),
    ("where", lambda c, a, b: torch.where(c, a, b), (torch.randn(4, 8) > 0, torch.randn(4, 8), torch.randn(4, 8))),
    ("lt", lambda a, b: a < b, XY),
    ("eq_scalar", lambda a: a == 0, X),
    ("logical_not", lambda a: torch.logical_not(a > 0), X),
    ("expand", lambda a: a.expand(4, 8), (torch.randn(1, 8),)),
    ("slice", lambda a: a[:, 2:6], X),
    ("pow2", lambda a: a ** 2, X),
    ("mean", lambda a: a.mean(-1), X),
    ("softmax", lambda a: torch.softmax(a, -1), X),
    ("layer_norm", lambda a: torch.nn.functional.layer_norm(a, (8,), torch.ones(8), torch.zeros(8)), X),
    ("bmm", torch.bmm, (torch.randn(2, 4, 8), torch.randn(2, 8, 16))),
    # scan / arg-reduce / dtype-edge families
    ("cumsum", lambda a: torch.cumsum(a, -1), X),
    ("cumsum_bool", lambda a: torch.cumsum(a > 0, -1), X),
    ("min_dim_vals", lambda a: torch.min(a, dim=1, keepdim=True)[0], X),
    ("min_dim_idx", lambda a: torch.min(a, dim=1)[1], X),
    ("argmax", lambda a: torch.argmax(a, dim=-1), X),
    ("sum_bool_keepdim", lambda a: (a > 0).sum(0, keepdim=True), (torch.randn(8),)),
    ("reciprocal_int", lambda a: torch.reciprocal(a), (torch.tensor([2, 3, 4]),)),
    ("to_copy_bool2int", lambda a: (a > 0).to(torch.int64), X),
    ("rsub_scalar", lambda a: 1.0 - a, X),            # reverse sub: scalar - tensor
    ("rdiv_scalar", lambda a: 2.0 / (a.abs() + 1), X),
    ("bucketize", lambda a: torch.bucketize(a, torch.linspace(-1, 1, 7)), X),
    ("index_gather", lambda s, i0, i1: s[i0, i1],
     (torch.randn(4, 8), torch.zeros(1, 1, dtype=torch.int64), torch.arange(8).view(1, 8))),
]

# Data-dependent ops: output has a dynamic (?) dim, so shape can't match eager exactly --
# assert only that they lower to standard dialects (scf + tensor) with no opaque calls.
DYNAMIC_CASES = [
    ("bool_mask_gather", lambda x, m: x[m], (torch.arange(8), torch.tensor([True, False] * 4))),
    ("masked_scatter", _masked_scatter := (lambda s, src, m: s.clone().index_put_((m,), src[m])),
     (torch.zeros(8, dtype=torch.int64), torch.arange(8), torch.tensor([True, False] * 4))),
]


@pytest.mark.parametrize("name,fn,inputs", CASES, ids=[c[0] for c in CASES])
def test_op_lowers_to_standard_dialects(name, fn, inputs):
    v = validate_op(fn, inputs, name=name)
    assert v.error is None, v.error
    assert v.lowered, f"{name} left opaque calls: {v.opaque_calls}"
    assert v.shape_ok, f"{name} result type {v.mlir_result_type} != eager {v.eager_shape}"


@pytest.mark.parametrize("name,fn,inputs", DYNAMIC_CASES, ids=[c[0] for c in DYNAMIC_CASES])
def test_dynamic_op_lowers_to_standard_dialects(name, fn, inputs):
    """Data-dependent ops lower to scf+tensor with a dynamic (?) dim -- no opaque calls."""
    v = validate_op(fn, inputs, name=name)
    assert v.error is None, v.error
    assert v.lowered, f"{name} left opaque calls: {v.opaque_calls}"
