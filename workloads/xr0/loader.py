"""Xiaomi-Robotics-0 (XR0) capture loader for m2m.

XR0 is a VLA model: a Qwen3-VL-4B backbone encodes vision+language into a
KV-cache, and a DiT (Diffusion Transformer) head decodes an action chunk via
rectified flow (Euler integration over `num_steps`).

The full `XR0.forward(batch)` is NOT a capture target:
  - it instantiates Qwen3-VL via `from_pretrained("Qwen/Qwen3-VL-4B-Instruct")`
    (multi-GB download, flash_attention_2),
  - inference runs a Python `for`-loop Euler integrator (`_flow_generate`),
  - control flow depends on `prefix_length` resolved via `.item()`.

Capture unit: ONE DiT denoise step (`XR0.dit_forward`) -- a single rectified-flow
velocity prediction. This is pure tensor ops (AdaLN-modulated DiT decoder layers
cross-attending to the VLM KV-cache) with no data-dependent control flow when
`prefix_length == 0`. The VLM KV-cache, RoPE position embeds, attention mask and
projected state are supplied as already-computed input tensors (host-side prefix
encoding stays out of the graph), mirroring the smolVLA denoise-step convention.

We build only the DiT head + projectors + embedders from a small random config;
the Qwen3-VL backbone is never constructed, so no weights are downloaded.

Env:
    XR0_DIT_LAYERS=N   number of DiT decoder layers (default 2 for a fast smoke;
                       the real model uses 16)
"""

from __future__ import annotations

import os

import torch
from torch import nn

# Real XR0 defaults (see xr0/mibot/models/VLA/XR0.py and configs/model).
# Qwen3-VL-4B text config: hidden 2560, 36 layers, head_dim 128, 8 KV heads.
_VLM_NUM_LAYERS = 36
_VLM_KV_HEADS = 8
_HEAD_DIM = 128
_DIT_HIDDEN = 1024
_STATE_LEN, _STATE_DIM = 1, 32
_ACTION_LEN, _ACTION_DIM = 30, 32
_KV_HEADS_DIT = 8


class XR0DenoiseStep(nn.Module):
    """Wrap XR0.dit_forward as a clean tensor->tensor capture unit (one flow step)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, noisy_action, t, action_mask, state, cos, sin, attn_mask, *kv):
        m = self.model
        # Reassemble the flat (k0, v0, k1, v1, ...) tensor list into the
        # per-layer [(k, v), ...] structure dit_forward expects.
        past_key_values = [(kv[2 * i], kv[2 * i + 1]) for i in range(len(kv) // 2)]
        state_embed = m.state_projector(state)
        return m.dit_forward(
            noisy_action=noisy_action,
            t=t,
            action_mask=action_mask,
            state_embed=state_embed,
            position_embeds=(cos, sin),
            past_key_values=past_key_values,
            attn_mask=attn_mask,
            prefix_length=0,
        )


def _build_dit_head(dit_layers: int) -> nn.Module:
    """Build the DiT head + projectors standalone, without the Qwen3-VL backbone.

    XR0.__init__ calls Qwen3VLForConditionalGeneration.from_pretrained (a multi-GB
    download). We bypass that by constructing the lightweight sub-modules directly,
    re-using the real XR0 classes so dit_forward semantics are identical.
    """
    from mibot.models.VLA.XR0 import DiT, MLPProjector, TimestepEmbedder
    from transformers import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
    )

    class _DiTHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit_hidden_size = _DIT_HIDDEN
            self.action_shape = (_ACTION_LEN, _ACTION_DIM)
            self.state_shape = (_STATE_LEN, _STATE_DIM)
            self.dit = DiT(hidden_size=_DIT_HIDDEN, kv_heads=_KV_HEADS_DIT, layer_num=dit_layers)
            self.state_projector = MLPProjector(_STATE_DIM, _DIT_HIDDEN, num_layers=2)
            self.action_projector = MLPProjector(_ACTION_DIM, _DIT_HIDDEN, num_layers=2)
            self.action_output_layer = MLPProjector(_DIT_HIDDEN, _ACTION_DIM, num_layers=2)
            self.t_embedder = TimestepEmbedder(_DIT_HIDDEN)
            self.t_projector = MLPProjector(_DIT_HIDDEN, 6 * _DIT_HIDDEN, bias=True)
            self.sink = nn.Embedding(1, _DIT_HIDDEN)
            # Small random text config for RoPE (no weights downloaded).
            txt_cfg = Qwen3VLTextConfig(
                hidden_size=_DIT_HIDDEN, head_dim=_HEAD_DIM, num_attention_heads=8,
                num_key_value_heads=_KV_HEADS_DIT, num_hidden_layers=2,
            )
            self.rotary_emb = Qwen3VLTextRotaryEmbedding(txt_cfg)

        # dit_forward is defined on XR0; bind the unbound method here.
        dit_forward = staticmethod(None)

    head = _DiTHead()
    # Reuse the real, identical dit_forward implementation.
    from mibot.models.VLA.XR0 import XR0
    head.dit_forward = XR0.dit_forward.__get__(head, _DiTHead)
    return head.to(torch.float32).eval()


def get_model_and_inputs():
    dit_layers = int(os.environ.get("XR0_DIT_LAYERS", "2"))
    head = _build_dit_head(dit_layers)

    b = 1
    q_len = _STATE_LEN + 1 + _ACTION_LEN  # sink + state + action tokens
    # Prefix (VLM cache) length is arbitrary for a single denoise step; pick small.
    prefix_len = 16
    total_len = prefix_len + q_len

    noisy_action = torch.randn(b, _ACTION_LEN, _ACTION_DIM, dtype=torch.float32)
    t = torch.ones(b, 1, 1, dtype=torch.float32) * 0.5
    action_mask = torch.ones(b, _ACTION_LEN, _ACTION_DIM, dtype=torch.float32)
    state = torch.randn(b, _STATE_LEN, _STATE_DIM, dtype=torch.float32)

    # RoPE (cos, sin) for the q_len DiT tokens, head_dim = 128.
    cos = torch.randn(b, q_len, _HEAD_DIM, dtype=torch.float32)
    sin = torch.randn(b, q_len, _HEAD_DIM, dtype=torch.float32)

    # Boolean attention mask: (B, 1, q_len, prefix_len + q_len).
    attn_mask = torch.ones(b, 1, q_len, total_len, dtype=torch.bool)

    # VLM KV-cache: the DiT consumes the tail `dit_layers` cache entries, each
    # (B, kv_heads, prefix_len, head_dim). Supply `dit_layers` (k, v) pairs.
    kv = []
    for _ in range(dit_layers):
        kv.append(torch.randn(b, _KV_HEADS_DIT, prefix_len, _HEAD_DIM, dtype=torch.float32))  # k
        kv.append(torch.randn(b, _KV_HEADS_DIT, prefix_len, _HEAD_DIM, dtype=torch.float32))  # v

    inputs = (noisy_action, t, action_mask, state, cos, sin, attn_mask, *kv)
    return XR0DenoiseStep(head).eval(), inputs
