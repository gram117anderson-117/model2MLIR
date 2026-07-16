"""Write a self-contained Merlin capture bundle (mlir + weights + golden + inputs + extra +
input_order) from ONE seeded model instance.

Factored out of ``workloads/capture_consistent.py`` so any capture path (per-model loader, or the
GGUF frontend) produces a byte-compatible bundle for the Merlin RVV runtime. It recovers the two
argument classes m2m elides to the runtime — registered buffers and lifted get_attr constants — by
type, plus quantized subclass inner tensors, exactly as the consistent-capture worker does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean ``input_ids -> logits`` forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, use_cache=False).logits


def capture_hf_bundle(model_dir, out, *, quant_scheme: str | None = None, seq_len: int = 8,
                      dtype: "torch.dtype" = torch.float32) -> dict:
    """Capture a HuggingFace causal-LM (local dir or hub id) into a Merlin bundle via the torch path.

    ``quant_scheme`` is a torchAO scheme name (e.g. ``"int8_weight_only"``) or ``None`` for fp. This
    is the torch/torchAO ingestion arm (the GGUF arm is :func:`m2m.frontends.gguf.capture_gguf_bundle`).
    """
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype, use_cache=False).eval()
    model = _LogitsOnly(lm).eval()
    vocab = int(lm.config.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (1, seq_len))
    quant = None
    if quant_scheme:
        from m2m.capture.torchao_pipeline import QuantizationConfig
        quant = QuantizationConfig(scheme=quant_scheme)
    summary = write_bundle(model, (input_ids,), out, quant=quant)
    summary["hf"] = str(model_dir)
    summary["quant_scheme"] = quant_scheme
    return summary


def _flatten_subclass(obj: Any, prefix: str, out: dict) -> None:
    """Recursively flatten a tensor-subclass parameter to its leaf tensors (qinner::<attr-path>)."""
    flat = getattr(obj, "__tensor_flatten__", None)
    if callable(flat):
        try:
            names, _ = flat()
        except Exception:  # noqa: BLE001
            names = []
        for nm in names:
            child = getattr(obj, nm, None)
            if child is not None:
                _flatten_subclass(child, f"{prefix}.{nm}", out)
    elif hasattr(obj, "detach"):                       # a leaf torch.Tensor
        t = obj.detach().cpu()
        _dt = str(t.dtype)
        arr = t.float().numpy() if ("float8" in _dt or "bfloat16" in _dt) else t.numpy()
        out[f"qinner::{prefix}"] = arr


def _lifted_constants(mdl, inputs, extra: dict) -> None:
    """Populate c_lifted_tensor_<i> from m2m's own export (graph order matches the importer)."""
    try:
        from m2m.capture.torch_export import capture_frontend_artifact
        from m2m.ir.torchmlir_decomps import torch_mlir_gap_decompositions
        from torch.export.graph_signature import InputKind

        artifact = capture_frontend_artifact(
            mdl, inputs, export_decomposition_table=torch_mlir_gap_decompositions())
        ep = artifact.exported_program or artifact.original_exported_program
        consts = dict(getattr(ep, "constants", {}) or {})
        sd = dict(getattr(ep, "state_dict", {}) or {})
        li = 0
        for spec in ep.graph_signature.input_specs:
            if spec.kind != InputKind.CONSTANT_TENSOR:
                continue
            val = consts.get(str(spec.target), sd.get(str(spec.target)))
            if val is not None and hasattr(val, "detach"):
                extra[f"c_lifted_tensor_{li}"] = val.detach().cpu().numpy()
            li += 1
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[warn] lifted-constant export failed: {exc}\n{traceback.format_exc()[-800:]}")


def write_bundle(mdl, inputs, out: str | Path, *, quant=None) -> dict:
    """Convert ``mdl`` and write the full bundle to ``out``. Returns a summary dict.

    ``quant`` is an m2m ``QuantizationConfig`` (or ``None`` for an unquantized/fp bundle). The golden
    is the forward of the SAME instance on the SAME inputs, so the bundle is self-consistent.
    """
    import m2m

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mdl.eval()
    inputs = tuple(inputs)

    weights_path = str(out / "weights.safetensors")
    r = m2m.convert(mdl, inputs, backend="fx_importer", quantization=quant,
                    level="linalg-on-tensors", weights_path=weights_path)
    assert r.ok, "m2m.convert failed"
    (out / "model.mlir").write_text(r.mlir_text)

    with torch.no_grad():
        g = mdl(*inputs)
    golden = g[0] if isinstance(g, (tuple, list)) else g
    np.save(out / "golden.npy", golden.detach().float().cpu().numpy())

    np.savez(out / "inputs.npz",
             **{f"in{i}": x.detach().cpu().numpy() for i, x in enumerate(inputs)})

    extra: dict = {}
    for name, t in mdl.named_buffers():
        extra["buf::" + name] = t.detach().float().cpu().numpy()
    for pname, p in mdl.named_parameters():
        if type(p).__name__ not in ("Parameter", "Tensor") or hasattr(p, "__tensor_flatten__"):
            _flatten_subclass(p, pname, extra)
    _lifted_constants(mdl, inputs, extra)
    np.savez(out / "extra.npz", **extra)

    man = json.loads(Path(weights_path + ".manifest.json").read_text())
    order: dict = {}
    k = 0
    for i in range(len(man)):
        meta = man[str(i)]
        nm = meta.get("name", "") or ""
        if meta["kind"] in ("param", "buffer") or "lifted_tensor" in nm:
            continue
        order[nm] = k
        k += 1
    (out / "input_order.json").write_text(json.dumps(order, indent=2))

    return {
        "out": str(out), "n_inputs": len(inputs),
        "n_buffers": sum(1 for kk in extra if kk.startswith("buf::")),
        "n_lifted": sum(1 for kk in extra if kk.startswith("c_lifted")),
        "n_qinner": sum(1 for kk in extra if kk.startswith("qinner::")),
        "golden_shape": list(golden.shape), "linalg": r.mlir_text.count("linalg."),
        "input_order": order,
    }
