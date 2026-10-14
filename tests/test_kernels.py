"""Kernel registry: registered custom-kernel ops lower to black-box ``func.call``s.

The registry is backend-agnostic, so most tests register plain CPU ``torch.library.custom_op``s
(with fake impls) in ``KERNEL_REGISTRY`` -- no Triton needed.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import m2m
from m2m.coverage import kernel_report, opaque_report
from m2m.ir.kernel_registry import KERNEL_REGISTRY, KernelSpec, lookup_kernel

# --- test kernels (CPU custom ops standing in for Triton launches) -----------------------


@torch.library.custom_op("m2m_test::scale_add", mutates_args=())
def scale_add(x: torch.Tensor, y: torch.Tensor, alpha: float, block: int = 64) -> torch.Tensor:
    return x * alpha + y


@scale_add.register_fake
def _(x, y, alpha, block=64):
    return torch.empty_like(x)


@torch.library.custom_op("m2m_test::two_out", mutates_args=())
def two_out(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x * 2, x + 1


@two_out.register_fake
def _(x):
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("m2m_test::scale_add_out", mutates_args={"out"})
def scale_add_out(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, alpha: float) -> None:
    out.copy_(x * alpha + y)


@scale_add_out.register_fake
def _(x, y, out, alpha):
    return None


@torch.library.custom_op("m2m_test::stack_and_shift", mutates_args=())
def stack_and_shift(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return a + b + c + d


@stack_and_shift.register_fake
def _(a, b, c, d):
    return torch.empty_like(a)


_TEST_KERNELS = ("m2m_test::scale_add", "m2m_test::two_out", "m2m_test::scale_add_out",
                 "m2m_test::stack_and_shift")


@pytest.fixture
def registered(monkeypatch):
    """Register the test kernels in the global table for the duration of one test."""
    for q in _TEST_KERNELS:
        monkeypatch.setitem(KERNEL_REGISTRY, q, KernelSpec(q))


class _Net(nn.Module):
    """lin -> scale_add (kwarg + omitted default) -> scale_add at a NEW shape -> two_out,
    whose SECOND output is returned. Exercises operand order, scalar args, symbol dedup,
    multi-output and ordering in one graph."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x, y):
        h = self.lin(x)
        a = torch.ops.m2m_test.scale_add(h, y=y, alpha=0.5)
        b = torch.ops.m2m_test.scale_add(a.reshape(-1), y.reshape(-1), 2.0, 32)
        p, q = torch.ops.m2m_test.two_out(a)
        return p + b.reshape(p.shape), q


def _inputs():
    torch.manual_seed(0)
    return torch.randn(4, 8), torch.randn(4, 8)


def _convert(model, inputs):
    r = m2m.convert(model.eval(), inputs, backend="fx_importer")
    assert r.ok, r.diagnostics
    return r


def _calls(text):
    return [line for line in text.splitlines() if "func.call" in line]


# --- registry -------------------------------------------------------------------------------


def test_builtin_table_has_mvp_kernel():
    spec = KERNEL_REGISTRY["triton_exp::fused_mlp2"]
    assert spec.callee == "triton_exp__fused_mlp2"
    # FX target strings and qualnames both resolve
    assert lookup_kernel("triton_exp.fused_mlp2.default") is spec
    assert lookup_kernel("triton_exp::fused_mlp2") is spec
    assert lookup_kernel("aten.addmm.default") is None


# --- emission -------------------------------------------------------------------------------


def test_kernel_not_opaque(registered):
    r = _convert(_Net(), _inputs())
    assert opaque_report(r.mlir_text) == {}
    assert kernel_report(r.mlir_text) == {"m2m_test::scale_add": 2, "m2m_test::two_out": 1}
    assert 'prov.family = "kernel"' in r.mlir_text


def test_operands_in_schema_order_with_scalar_args(registered):
    """kwarg call + omitted default normalize to schema order; scalars are attributes."""
    calls = [c for c in _calls(_convert(_Net(), _inputs()).mlir_text) if "m2m_test::scale_add" in c]
    assert 'prov.kernel_operands = ["x", "y"]' in calls[0]
    assert "alpha = 5.000000e-01 : f64" in calls[0] and "block = 64 : i64" in calls[0]
    assert "alpha = 2.000000e+00 : f64" in calls[1] and "block = 32 : i64" in calls[1]


def test_new_shape_gets_new_symbol_same_identity(registered):
    text = _convert(_Net(), _inputs()).mlir_text
    syms = re.findall(r"func\.call @(\w+)\([^\n]*prov\.kernel = \"m2m_test::scale_add\"", text)
    assert syms == ["m2m_test__scale_add", "m2m_test__scale_add_1"]
    assert "func.func private @m2m_test__scale_add_1(tensor<32xf32>" in text


def test_multi_output_all_results_kept(registered):
    """The SECOND output of two_out is a graph output: it must survive to func.return
    (the generic opaque path drops getitem(_, 1))."""
    r = _convert(_Net(), _inputs())
    call = next(c for c in _calls(r.mlir_text) if "m2m_test::two_out" in c)
    lhs = call.split("=")[0].strip()
    assert re.fullmatch(r"%\d+, %\d+", lhs), call
    second = lhs.split(", ")[1]
    ret = next(line for line in r.mlir_text.splitlines() if "func.return" in line)
    assert second in ret
    assert "-> (tensor<4x8xf32>, tensor<4x8xf32>)" in call


def test_ordering_and_entry_function(registered):
    """Kernel calls sit between their producers and consumers; @forward is the first func
    (kernel decls follow it); the text re-parses; capture is deterministic."""
    from xdsl.dialects.func import CallOp, FuncOp

    from m2m.capture.torch_mlir_bridge import _parse_mlir_text_to_xdsl

    r = _convert(_Net(), _inputs())
    funcs = [o for o in r.module.body.block.ops if isinstance(o, FuncOp)]
    assert funcs[0].sym_name.data == "forward" and funcs[0].body.blocks
    assert all(not f.body.blocks for f in funcs[1:])

    ops = list(funcs[0].body.block.ops)
    pos = {op: i for i, op in enumerate(ops)}
    for call in (o for o in ops if isinstance(o, CallOp)):
        for v in call.operands:
            if v.owner in pos:
                assert pos[v.owner] < pos[call]
        for res in call.results:
            for use in res.uses:
                assert pos[use.operation] > pos[call]

    assert _parse_mlir_text_to_xdsl(r.mlir_text) is not None
    assert _convert(_Net(), _inputs()).mlir_text == r.mlir_text


def test_func_signature_matches_unkerneled_model(registered):
    """Placeholders (func args) are untouched by kernel lowering: same args, same order."""

    class _Plain(_Net):
        def forward(self, x, y):
            h = self.lin(x)
            return h * 0.5 + y, h

    sig = lambda t: re.search(r"func\.func @forward\(([^)]*)\)", t).group(1)  # noqa: E731
    assert sig(_convert(_Net(), _inputs()).mlir_text) == sig(_convert(_Plain(), _inputs()).mlir_text)


def test_unregistered_custom_op_stays_opaque():
    """Without registration nothing changes: the op is an opaque call, as before."""
    r = _convert(_Net(), _inputs())
    assert kernel_report(r.mlir_text) == {}
    assert any(k.startswith("m2m_test_scale_add") for k in opaque_report(r.mlir_text))


def test_mutating_kernel_is_an_error(registered):
    """In-place kernels (auto_functionalized) aren't supported yet: a registered one must fail
    loudly instead of lowering to an operand-less call."""

    class _M(nn.Module):
        def forward(self, x, y):
            out = torch.empty_like(x)
            torch.ops.m2m_test.scale_add_out(x, y, out, 3.0)
            return out * 2

    r = m2m.convert(_M().eval(), _inputs(), backend="fx_importer")
    assert not r.ok
    assert any("in-place kernels are not supported" in d for d in r.diagnostics)
