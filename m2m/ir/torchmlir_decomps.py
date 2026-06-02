"""Torch-MLIR gap-op decompositions.

torch-mlir lowers hundreds of aten ops to linalg, but a handful have no converter
and torch-mlir marks them *illegal*, aborting the whole module. Several of these are
not even in the original graph -- torch's own decompositions introduce them (e.g.
``aten.empty_permuted``). The fix (not a monkeypatch): provide torch-level
decompositions that rewrite each gap op into ops torch-mlir *does* support, and feed
them into ``run_decompositions`` so the ExportedProgram handed to torch-mlir is free
of gap ops.

This is the growable registry: when torch-mlir reports a new "failed to legalize
operation 'torch.aten.<op>'", add a decomposition here.
"""

from __future__ import annotations

from typing import Any

import torch

_aten = torch.ops.aten


def _empty_permuted(size, physical_layout, *, dtype=None, layout=None, device=None, pin_memory=None):
    """aten.empty_permuted -> aten.empty.

    Contents are uninitialized, so the physical (permuted) layout is irrelevant to the
    computation; the logical shape ``size`` is preserved. torch-mlir has no
    empty_permuted converter but lowers empty fine.
    """
    return torch.empty(
        size,
        dtype=dtype,
        device=device,
        pin_memory=bool(pin_memory) if pin_memory is not None else False,
    )


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
          scale=None, enable_gqa=False):
    """aten.scaled_dot_product_attention -> the math reference, so export expands it into
    matmul/softmax/(mask) that m2m lowers. Quantized graphs retain SDPA where fp32 does
    not; expanding it here covers both."""
    import math

    L, S = query.size(-2), key.size(-2)
    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn = attn.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, value)


def _dropout(input, p=0.5, train=False):
    """aten.dropout in inference (train=False) is the identity."""
    return input


def _chunk(input, chunks, dim=0):
    """aten.chunk -> tensor_split (slices); export lowers the slices to ops m2m handles."""
    return torch.tensor_split(input, chunks, dim=dim)


# op-overload -> decomposition function. Extend as new torch-mlir gap ops surface.
_GAP_DECOMPS: dict[Any, Any] = {
    _aten.empty_permuted.default: _empty_permuted,
    _aten.scaled_dot_product_attention.default: _sdpa,
    _aten.dropout.default: _dropout,
    _aten.chunk.default: _chunk,
}


def torch_mlir_gap_decompositions() -> dict[Any, Any]:
    """Decompositions to merge into the export decomposition table so torch-mlir's
    pipeline does not abort on unsupported ops."""
    return dict(_GAP_DECOMPS)
