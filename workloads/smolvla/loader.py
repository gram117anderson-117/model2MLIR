"""smolVLA (lerobot) capture loader for model2mlir.

Builds smolVLA's VLAFlowMatching (SmolVLM2-500M backbone + action expert) and exposes
one flow-matching denoise step (prefix embed + one expert pass) as the capture unit,
on already-preprocessed tensors (host-side preprocessing stays out of the graph).
"""

from __future__ import annotations

import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, make_att_2d_masks


class SmolVLADenoiseStep(nn.Module):
    def __init__(self, model: VLAFlowMatching) -> None:
        super().__init__()
        self.model = model

    def forward(self, img, img_mask, lang_tokens, lang_masks, state, noise):
        m = self.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            [img], [img_mask], lang_tokens, lang_masks, state=state
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = m.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=m.config.use_cache,
            fill_kv_cache=True,
        )
        timestep = torch.ones(state.shape[0], dtype=torch.float32)
        return m.denoise_step(prefix_pad_masks, past_key_values, noise, timestep)


def get_model_and_inputs():
    cfg = SmolVLAConfig()
    cfg.device = "cpu"
    model = VLAFlowMatching(cfg).eval()

    b = 1
    H = W = cfg.resize_imgs_with_padding[0] if cfg.resize_imgs_with_padding else 512
    tok = cfg.tokenizer_max_length
    sd, ah, ad = cfg.max_state_dim, cfg.chunk_size, cfg.max_action_dim

    inputs = (
        torch.rand(b, 3, H, W, dtype=torch.float32) * 2 - 1,   # 1 camera image [-1,1]
        torch.ones(b, dtype=torch.bool),                       # image mask
        torch.randint(0, 256, (b, tok), dtype=torch.long),     # language tokens
        torch.ones(b, tok, dtype=torch.bool),                  # language mask
        torch.randn(b, sd, dtype=torch.float32),               # robot state
        torch.randn(b, ah, ad, dtype=torch.float32),           # noise x_t
    )
    return SmolVLADenoiseStep(model).eval(), inputs
