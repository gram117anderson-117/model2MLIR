"""Expansion pass: high-level *_ext named ops -> standard linalg, single-source-of-truth."""

from __future__ import annotations

import torch

from xdsl.dialects.builtin import ModuleOp, TensorType, f32
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

from m2m.api import module_to_text
from m2m.coverage import dialect_op_histogram
from m2m.ir.linalg_ext.ops import LayerNormOp, SoftmaxOp
from m2m.transforms import EXPANDERS, expand_to_linalg


def _module_to_text(m):
    return module_to_text(m)


def _wrap(build_op, in_shape):
    """Build a module: func(arg: tensor<in_shape xf32>) { %r = build_op(arg); return %r }."""
    t = TensorType(f32, in_shape)
    blk = Block(arg_types=[t])
    op = build_op(blk.args[0], t)
    blk.add_op(op)
    blk.add_op(ReturnOp(op.results[0]))
    fn = FuncOp("f", ((t,), (t,)), region=Region(blk))
    return ModuleOp([fn])


def test_expand_softmax_lowers_to_standard():
    m = _wrap(lambda a, t: SoftmaxOp(a, 1, t), [2, 8])
    assert any(isinstance(o, SoftmaxOp) for o in m.walk())
    expand_to_linalg(m)
    assert not any(isinstance(o, SoftmaxOp) for o in m.walk())  # named op gone
    hist = dialect_op_histogram(_module_to_text(m))
    # softmax body = max-reduce + sub + exp + sum-reduce + div, all standard dialects
    assert hist.get("linalg.reduce", 0) >= 2
    assert hist.get("linalg.generic", 0) >= 3
    assert "linalg_ext.softmax" not in _module_to_text(m)


def test_expand_layer_norm_lowers_to_standard():
    m = _wrap(lambda a, t: LayerNormOp(a, t, eps=1e-5, axis=1), [2, 8])
    expand_to_linalg(m)
    assert not any(isinstance(o, LayerNormOp) for o in m.walk())
    assert "linalg_ext.layer_norm" not in _module_to_text(m)


def test_high_level_then_expand_equals_standard():
    """convert(level=high-level) then expand_to_linalg == convert(level=linalg-on-tensors):
    proves the importer's standard path and the expansion pass are one source of truth."""
    import torch.nn as nn
    import m2m

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(16)
            self.fc = nn.Linear(16, 16)

        def forward(self, x):
            return torch.softmax(self.fc(self.ln(x)), dim=-1)

    x = (torch.randn(2, 16),)
    std = m2m.convert(Net().eval(), x, backend="fx_importer", level="linalg-on-tensors")
    hl = m2m.convert(Net().eval(), x, backend="fx_importer", level="high-level")
    # high-level form carries the named ops
    assert "linalg_ext.softmax" in hl.mlir_text and "linalg_ext.layer_norm" in hl.mlir_text
    # expand the high-level module and compare op histograms to the standard form
    m2m.expand_to_linalg(hl.module)
    from m2m.api import module_to_text
    assert dialect_op_histogram(module_to_text(hl.module)) == dialect_op_histogram(std.mlir_text)


def test_every_named_op_has_expander():
    """Registry-completeness: every named op the high-level form can emit must have an
    expansion handler (else expand_to_linalg would leave an unloadable op for backends)."""
    from m2m.ir.import_fx import high_level_named_ops
    emittable = high_level_named_ops()
    assert set(emittable) <= set(EXPANDERS), f"named ops without expander: {set(emittable) - set(EXPANDERS)}"
