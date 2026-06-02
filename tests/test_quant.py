"""Quantization lowering: int8/fp8 weight-only (QDQ) + true-quantized int8 (no QDQ) +
mixed precision all lower to standard dialects with 0 opaque ops."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import m2m
from m2m.capture.torchao_pipeline import QuantizationConfig
from m2m.coverage import opaque_report

torchao = pytest.importorskip("torchao")


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(64, 128)
        self.b = nn.Linear(128, 64)

    def forward(self, x):  # noqa: D102
        return self.b(torch.relu(self.a(x)))


X = (torch.randn(4, 64),)

# (scheme, expect_int8_storage) -- weight-only keeps QDQ (i8 + dequant), dyn-act is true int matmul
SCHEMES = [
    "int8_weight_only",
    "float8_weight_only_e4m3",
    "float8_weight_only_e5m2",
    "int8_dyn_act_int8_weight",
]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_quant_scheme_lowers(scheme):
    r = m2m.convert(MLP().eval(), X, backend="fx_importer", quantization=QuantizationConfig(scheme=scheme))
    assert r.ok, f"{scheme} did not produce a valid module"
    opaque = opaque_report(r.mlir_text)
    assert sum(opaque.values()) == 0, f"{scheme} left opaque: {opaque}"
    assert "m2m.quantization" in r.mlir_text


def test_int8_weight_only_preserves_qdq():
    """preserve_qdq (default) folds the implicit weight dequant into explicit
    quant_ext.dequantize ops so the quantization is first-class/matchable."""
    r = m2m.convert(MLP().eval(), X, backend="fx_importer",
                    quantization=QuantizationConfig(scheme="int8_weight_only"))
    assert "quant_ext.dequantize" in r.mlir_text
    assert sum(opaque_report(r.mlir_text).values()) == 0
    # opting out leaves the unfused (still 0-opaque) form
    r2 = m2m.convert(MLP().eval(), X, backend="fx_importer", preserve_qdq=False,
                     quantization=QuantizationConfig(scheme="int8_weight_only"))
    assert "quant_ext.dequantize" not in r2.mlir_text


def test_true_int_matmul_present():
    """Dynamic-activation int8 lowers to a real i8->i32 integer matmul (no dequant roundtrip)."""
    r = m2m.convert(MLP().eval(), X, backend="fx_importer",
                    quantization=QuantizationConfig(scheme="int8_dyn_act_int8_weight"))
    assert "i32" in r.mlir_text  # int32 accumulation


def test_fp8_type_renders_native_spelling():
    """The shim fp8 type prints with the MLIR-native spelling (f8E4M3FN) on text emission,
    so an artifact carrying f8 storage parses in a standard MLIR toolchain."""
    from xdsl.dialects.builtin import ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    from m2m.capture.torch_mlir_bridge import module_to_text
    from m2m.ir.types import Float8E4M3FNType

    t = TensorType(Float8E4M3FNType(), [4, 8])
    blk = Block(arg_types=[t])
    blk.add_op(ReturnOp(blk.args[0]))
    txt = module_to_text(ModuleOp([FuncOp("f", ((t,), (t,)), region=Region(blk))]))
    assert "f8E4M3FN" in txt and "builtin_ext" not in txt


def test_mixed_precision_lowers():
    """One network mixing int8 + fp8 + full-precision submodules lowers to 0 opaque."""
    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc_a = nn.Linear(64, 64)
            self.fc_b = nn.Linear(64, 64)
            self.fc_c = nn.Linear(64, 64)

        def forward(self, x):
            return self.fc_c(torch.relu(self.fc_b(torch.relu(self.fc_a(x)))))

    cfg = QuantizationConfig(scheme="none", per_module={
        r"fc_a": "int8_weight_only",
        r"fc_b": "float8_weight_only_e4m3",
    })
    r = m2m.convert(Net().eval(), X, backend="fx_importer", quantization=cfg)
    assert r.ok
    assert sum(opaque_report(r.mlir_text).values()) == 0
    assert "m2m.quantization_mixed" in r.mlir_text
