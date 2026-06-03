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


def inline_set_grad_hops(gm) -> bool:
    """Inline ``wrap_with_*`` higher-order ops (torch.no_grad / torch.autocast regions) into
    the parent FX graph.

    A ``torch.no_grad()`` / ``torch.autocast()`` region is exported (on torch 2.7.x) as a HOP
    wrapping a sub-GraphModule -- e.g. ``wrap_with_set_grad_enabled(enabled, submod, *ops)`` or
    ``wrap_with_autocast(device, dtype, enabled, cache, submod, *ops)`` -- with
    ``getitem(hop, i)`` extracting its outputs. The FXImporter doesn't recurse into the
    subgraph, so its outputs aren't mapped and downstream ops lose their operand. Inlining the
    subgraph's nodes into the parent removes the HOP. Returns True if anything changed (caller
    re-runs to a fixpoint for nested HOPs). Idempotent.
    """
    import operator

    g = gm.graph
    changed = False
    for node in list(g.nodes):
        # Only the set_grad (torch.no_grad) HOP. Autocast HOPs wrap a nested constant-
        # precompute subgraph whose operand remapping during inlining is fragile (it
        # mismaps a sliced tensor), so we leave autocast as a clean opaque boundary -- the
        # correct floor (4 opaque on pi05) vs. a corrupt 41 if inlined.
        if node.op != "call_function" or "wrap_with_set_grad_enabled" not in str(node.target):
            continue
        # the submodule is whichever arg is a get_attr to a GraphModule; subgraph inputs are
        # the node args that follow it (the flags precede it).
        sub_idx = next((i for i, a in enumerate(node.args)
                        if getattr(a, "op", None) == "get_attr"
                        and hasattr(getattr(gm, a.target, None), "graph")), None)
        if sub_idx is None:
            continue
        # Decide erasability BEFORE touching the graph (never leave orphaned duplicates):
        # either every consumer is an int getitem (multi-output), or there's a single direct
        # consumer (single-output). Anything else -> skip this HOP (stays opaque, no spin).
        users = list(node.users)
        getitems = [u for u in users if u.op == "call_function" and u.target is operator.getitem
                    and isinstance(u.args[1], int)]
        direct = [u for u in users if u not in getitems]
        if direct and (len(users) != 1):
            continue  # mixed / multiple direct users: unsafe to fully erase
        sub = getattr(gm, node.args[sub_idx].target)
        operands = list(node.args[sub_idx + 1:])
        placeholders = [n for n in sub.graph.nodes if n.op == "placeholder"]
        env = {ph: val for ph, val in zip(placeholders, operands)}
        outputs = None
        with g.inserting_before(node):
            for sn in sub.graph.nodes:
                if sn.op == "placeholder":
                    continue
                if sn.op == "output":
                    outs = sn.args[0]
                    outputs = [env.get(o, o) for o in (outs if isinstance(outs, (tuple, list)) else [outs])]
                    continue
                env[sn] = g.node_copy(sn, lambda n: env[n])
        if not outputs or (getitems and any(u.args[1] >= len(outputs) for u in getitems)):
            continue
        if direct:                       # single-output HOP used directly
            node.replace_all_uses_with(outputs[0])
        else:
            for u in getitems:
                u.replace_all_uses_with(outputs[u.args[1]])
                g.erase_node(u)
        g.erase_node(node)
        changed = True
    if changed:
        g.eliminate_dead_code()
        g.lint()
        gm.recompile()
    return changed


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
