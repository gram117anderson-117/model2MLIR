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
]


@pytest.mark.parametrize("name,fn,inputs", CASES, ids=[c[0] for c in CASES])
def test_op_lowers_to_standard_dialects(name, fn, inputs):
    v = validate_op(fn, inputs, name=name)
    assert v.error is None, v.error
    assert v.lowered, f"{name} left opaque calls: {v.opaque_calls}"
    assert v.shape_ok, f"{name} result type {v.mlir_result_type} != eager {v.eager_shape}"
