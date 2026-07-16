"""GGUF frontend — reconstruct a decoder graph from a ``.gguf`` checkpoint and export to MLIR.

GGUF stores quantized weights plus architecture metadata (but no compute graph). This frontend
reconstructs the graph from that metadata via transformers' GGUF support (``GGUF_CONFIG_MAPPING``
covers llama / qwen2 / gemma2 — TinyLlama, DeepSeek-R1-Distill-Qwen, Gemma-2), wraps it to a clean
tensor->tensor forward, and runs the **existing** m2m torch export path — so all of m2m's ATen->linalg
lowering, decomposition, and quant handling is reused unchanged.

Two stages (this module implements v1; v2 is layered on the same seam):

* **v1 (dequantized baseline):** ``AutoModelForCausalLM.from_pretrained(dir, gguf_file=...)`` builds the
  graph *and* dequantizes the GGUF weights to fp. This proves GGUF -> linalg-on-tensors end to end and
  is the correctness reference (it matches transformers' own dequantized forward).
* **v2 (quant-preserving, staged):** keep the GGUF weights in their native format and inject them as
  ``quant_ext`` types instead of dequantizing — reusing m2m's quant_ext dialect and the FXImporter's
  quantized-weight handling. The GGUF block layouts / dtype classification for that already exist
  (gguf-py + the quant_formats registry).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class _LogitsOnly(nn.Module):
    """Wrap a HF causal LM so export sees a clean ``input_ids -> logits`` forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm(input_ids=input_ids, use_cache=False).logits


def load_gguf_torch_model(gguf_path: str | Path, *, dtype: torch.dtype = torch.float32) -> nn.Module:
    """Build a torch causal-LM from a GGUF file (graph from metadata, weights dequantized by HF)."""
    from transformers import AutoModelForCausalLM

    p = Path(gguf_path)
    model = AutoModelForCausalLM.from_pretrained(str(p.parent), gguf_file=p.name, dtype=dtype)
    return model.eval()


def convert_gguf(gguf_path: str | Path, example_inputs: tuple[Any, ...] | None = None, *,
                 seq_len: int = 8, level: str = "linalg-on-tensors"):
    """Convert a GGUF checkpoint to a :class:`m2m.api.ConversionResult` (linalg-on-tensors MLIR).

    ``example_inputs`` defaults to a single seeded ``input_ids`` row of length ``seq_len``.
    """
    from m2m.api import convert

    lm = load_gguf_torch_model(gguf_path)
    model = _LogitsOnly(lm).eval()
    if not example_inputs:
        vocab = int(lm.config.vocab_size)
        example_inputs = (torch.randint(0, vocab, (1, seq_len)),)
    result = convert(model, tuple(example_inputs), backend="fx_importer", level=level)
    # Record the true frontend (the torch bridge stamps its own path).
    try:
        object.__setattr__(result, "frontend", "gguf")
    except Exception:  # noqa: BLE001 - ConversionResult may not be frozen; best-effort provenance
        pass
    return result


def capture_gguf_bundle(gguf_path: str | Path, out: str | Path, *, seq_len: int = 8,
                        dtype: torch.dtype = torch.float32, quant_scheme: str | None = None) -> dict:
    """Produce a runnable Merlin capture bundle from a GGUF checkpoint (mlir + weights + golden + ...).

    Reconstructs the graph from GGUF metadata, seeds an ``input_ids`` row, and writes a self-consistent
    bundle via :func:`m2m.capture.bundle.write_bundle`.

    ``quant_scheme`` controls how the quantization is carried into the compiled IR:

    * ``None`` (default) — the fp baseline: transformers dequantizes the GGUF weights, so the compiled
      model is fp (the low-bit format is not carried through compute).
    * a torchAO scheme (e.g. ``"int8_weight_only"``) — **preserves a quantized datapath**: the model is
      (re-)quantized so the emitted MLIR carries ``quant_ext.dequantize_per_channel`` (int8 storage +
      per-channel scale in the IR), which the Merlin pipeline lowers to the weight-only-int8 / W8A8 RVV
      datapath. This closes the "GGUF ingests but compiles to fp" gap for the int8 case — the GGUF
      model now compiles WITH a quant_ext datapath and runs on RVV. (The int8 is re-derived per-channel
      from the GGUF weights; bit-exact per-block/sub-byte preservation is the deeper quant_ext-native
      path, tied to a low-bit target backend.)
    """
    from m2m.capture.bundle import write_bundle

    lm = load_gguf_torch_model(gguf_path, dtype=dtype)
    model = _LogitsOnly(lm).eval()
    vocab = int(lm.config.vocab_size)
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (1, seq_len))
    quant = None
    if quant_scheme:
        from m2m.capture.torchao_pipeline import QuantizationConfig
        quant = QuantizationConfig(scheme=quant_scheme)
    summary = write_bundle(model, (input_ids,), out, quant=quant)
    summary["gguf"] = str(gguf_path)
    summary["quant_scheme"] = quant_scheme
    return summary
