"""Per-section split: VLA module boundaries -> separate func.func per section."""

from __future__ import annotations

import torch
import torch.nn as nn

import m2m
from m2m.api import module_to_text
from m2m.coverage import module_sections, opaque_report
from m2m.transforms import split_by_section


class _Sub(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l1 = nn.Linear(n, n)
        self.l2 = nn.Linear(n, n)

    def forward(self, x):
        return torch.softmax(self.l2(torch.relu(self.l1(x))), -1)


class _VLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = _Sub(16)
        self.action_expert = _Sub(16)

    def forward(self, x):
        return self.action_expert(self.vlm(x))


def _capture():
    return m2m.convert(_VLA().eval(), (torch.randn(2, 16),), backend="fx_importer")


def test_module_provenance_tags():
    r = _capture()
    secs = module_sections(r.mlir_text)
    assert set(secs) == {"vlm", "action_expert"}
    assert all(v > 0 for v in secs.values())


def test_split_by_section_produces_valid_per_section_funcs():
    r = _capture()
    secs = split_by_section(r.module)
    assert set(secs) == {"vlm", "action_expert"}
    for name, mod in secs.items():
        mod.verify()                                            # each section is valid MLIR
        t = module_to_text(mod)
        assert f"func.func @section_{name}" in t
        assert sum(opaque_report(t).values()) == 0             # and fully lowered
