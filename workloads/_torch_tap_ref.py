#!/usr/bin/env python3
"""Torch reference at a given M2M_SMOLVLA_TAP, mirroring capture_consistent.py 40-66.

Usage: python _torch_tap_ref.py <out_npy>   (env M2M_SMOLVLA_TAP/VLM_LAYERS/EXPERT_LAYERS set)
Builds the SAME seeded+perturbed+quantized instance the consistent capture uses, runs
mdl(*inputs), saves the (float) output to <out_npy>.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
WORKLOADS = REPO / "workloads"


def main(out_npy: str, fmt: str = "int8") -> None:
    import m2m  # noqa: F401  (parity with capture env)
    sys.path.insert(0, str(WORKLOADS))
    from capture import _load_toml, _quant_for
    sys.path.insert(0, str(WORKLOADS / "smolvla"))
    from loader import get_model_and_inputs

    cfg = _load_toml(WORKLOADS / "smolvla")
    torch.manual_seed(0)
    np.random.seed(0)
    mdl, inputs = get_model_and_inputs()
    mdl.eval()
    inputs = tuple(inputs)
    with torch.no_grad():
        for p in mdl.parameters():
            if float(p.detach().abs().max()) == 0.0:
                p.copy_(torch.randn_like(p) * 0.02)
    # Apply the SAME quantization the capture applies (mutates mdl in place via m2m.convert).
    q = _quant_for(cfg, fmt)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = m2m.convert(mdl, inputs, backend="fx_importer", quantization=q,
                        level="linalg-on-tensors",
                        weights_path=str(Path(td) / "w.safetensors"))
        assert r.ok
    with torch.no_grad():
        g = mdl(*inputs)
    g = g[0] if isinstance(g, (tuple, list)) else g
    np.save(out_npy, g.detach().float().cpu().numpy())
    print("__TORCH_TAP_OK__", out_npy, list(g.shape))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "int8")
