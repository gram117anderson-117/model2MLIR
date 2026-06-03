"""Weight externalization: graph stays inspectable, real data goes to safetensors."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

import m2m


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 8)

    def forward(self, x):  # noqa: D102
        return self.b(torch.relu(self.a(x)))


def test_weights_externalized_and_recoverable(tmp_path):
    pytest_safetensors = __import__("pytest").importorskip("safetensors")
    from safetensors.torch import load_file

    m = _MLP().eval()
    wp = str(tmp_path / "mlp.safetensors")
    r = m2m.convert(m, (torch.randn(2, 8),), backend="fx_importer", weights_path=wp)
    assert r.ok
    assert "prov.weights_file" in r.mlir_text          # graph references the data file

    st = load_file(wp)                                 # real data is recoverable
    assert set(st.keys()) == {"a.weight", "a.bias", "b.weight", "b.bias"}
    assert torch.allclose(st["a.weight"], m.a.weight)
    assert torch.allclose(st["b.bias"], m.b.bias)

    manifest = json.load(open(wp + ".manifest.json"))  # arg-index -> weight/dtype/shape
    weights = {v["weight"]: v for v in manifest.values() if "weight" in v}
    assert weights["a.weight"]["shape"] == [16, 8]
    assert weights["a.weight"]["kind"] == "param"
