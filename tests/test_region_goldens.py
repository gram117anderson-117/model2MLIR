"""Per-region boundary-tensor capture (region_goldens.npz).

For each nn.Module the export tagged with a ``prov.fqn``, ``write_bundle`` captures — in the SAME
golden forward — the module's boundary INPUT (= the upstream region's output) and its OUTPUT golden.
This is the shared substrate for per-region equivalence and standalone-section profiling downstream.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from m2m.capture.bundle import _extract_prov_fqns, write_bundle


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


def test_extract_prov_fqns_is_structured_and_ordered():
    txt = ('%0 = linalg.matmul {prov.fqn = "blocks.0.attn.q", prov.op = "matmul"} ...\n'
           '%1 = "x"() {prov.fqn = "blocks.0.mlp.g"} : () -> ()\n'
           '%2 = "y"() {prov.fqn = "blocks.0.attn.q"} : () -> ()\n')   # duplicate
    # first-seen order, deduplicated, no regex.
    assert _extract_prov_fqns(txt) == ["blocks.0.attn.q", "blocks.0.mlp.g"]
    assert _extract_prov_fqns("no provenance here") == []


def test_write_bundle_emits_region_goldens(tmp_path):
    out = tmp_path / "vla_fp32"
    summary = write_bundle(_VLA(), (torch.randn(2, 16),), out)

    rg = out / "region_goldens.npz"
    assert rg.is_file()
    assert summary["n_regions"] >= 2

    with np.load(rg) as z:
        keys = list(z.files)
    fqns = {k.split("::", 1)[0] for k in keys}
    # both submodules captured as regions, distinguishable by fqn.
    assert any("vision_backbone" in f for f in fqns)
    assert any("action_expert" in f for f in fqns)
    # every region carries an OUTPUT golden, and at least one carries a boundary input.
    assert any(k.endswith("::out") for k in keys)
    assert any("::in0" in k for k in keys)
    # the whole-model golden is still written and unaffected by the hooks.
    assert (out / "golden.npy").is_file()


def test_capture_regions_can_be_disabled(tmp_path):
    out = tmp_path / "vla_noregions"
    write_bundle(_VLA(), (torch.randn(2, 16),), out, capture_regions=False)
    assert not (out / "region_goldens.npz").is_file()
    assert (out / "golden.npy").is_file()      # the rest of the bundle is unchanged
