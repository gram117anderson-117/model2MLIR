"""prov.fqn: each op carries its DEEPEST nn.Module path, so a downstream tool can tell
backbone from action head (prov.module only keeps the first/wrapper component)."""

from __future__ import annotations

import re

import torch
import torch.nn as nn

import m2m


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 32)

    def forward(self, x):
        return torch.relu(self.proj(x))


class _ActionExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc(x)


class _VLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = _Backbone()
        self.action_expert = _ActionExpert()

    def forward(self, x):
        return self.action_expert(self.vision_backbone(x))


def _matmul_fqns(mlir_text: str) -> list[str]:
    out = []
    for line in mlir_text.splitlines():
        if "linalg.matmul" in line:
            m = re.search(r'prov\.fqn = "([^"]*)"', line)
            out.append(m.group(1) if m else None)
    return out


def test_prov_fqn_distinguishes_submodules():
    r = m2m.convert(_VLA(), (torch.randn(2, 16),), backend="fx_importer")
    fqns = _matmul_fqns(r.mlir_text)
    assert fqns, "expected at least one linalg.matmul with a prov.fqn"
    joined = " ".join(f for f in fqns if f)
    # The two submodules are distinguishable in the FQNs (impossible from prov.module alone
    # for a monolithic wrapper).
    assert "vision_backbone" in joined
    assert "action_expert" in joined
    # prov.fqn is the deep path, not just the first component.
    assert any("." in (f or "") for f in fqns)
