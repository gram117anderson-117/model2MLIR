"""pi0.5 (PyTorch) capture loader for model2mlir.

Builds the real pi0.5 architecture (PaliGemma gemma_2b + gemma_300m action expert,
pi05=True) with random init (no checkpoint download), and exposes a single
flow-matching `denoise_step` (prefix embed + one expert pass) as the capture unit
-- the representative heavy graph, without the diffusion while-loop.

    get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])
"""

from __future__ import annotations

import torch
from torch import nn

from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

_IMG_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


class Pi05DenoiseStep(nn.Module):
    """One flow-matching step: prefix (SigLIP+Gemma) pass + one expert denoise_step.

    Takes ALREADY-PREPROCESSED tensors. Observation preprocessing (resize/normalize +
    Python set/dict glue) is host-side and not traceable, so it runs eagerly outside
    the captured graph.
    """

    def __init__(self, model: PI0Pytorch) -> None:
        super().__init__()
        self.model = model

    def forward(self, i0, i1, i2, m0, m1, m2, lang_tokens, lang_masks, state, noise):
        m = self.model
        images, img_masks = [i0, i1, i2], [m0, m1, m2]
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_4d = m._prepare_attention_masks_4d(prefix_att_2d)
        m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = m.paligemma_with_expert.forward(
            attention_mask=prefix_att_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        timestep = torch.ones(state.shape[0], dtype=torch.float32)
        return m.denoise_step(state, prefix_pad_masks, past_key_values, noise, timestep)


def get_model_and_inputs():
    cfg = Pi0Config(pi05=True, dtype="float32")
    model = PI0Pytorch(cfg).eval()

    b = 1
    res = 224
    ad, ah, tok = cfg.action_dim, cfg.action_horizon, cfg.max_token_len

    def img():
        # channels-first [B, 3, H, W] in [-1, 1] (preprocessing detects shape[1]==3)
        return torch.rand(b, 3, res, res, dtype=torch.float32) * 2 - 1

    def mask():
        return torch.ones(b, dtype=torch.bool)

    raw = Observation(
        images={k: img() for k in _IMG_KEYS},
        image_masks={k: mask() for k in _IMG_KEYS},
        state=torch.randn(b, ad, dtype=torch.float32),
        tokenized_prompt=torch.randint(0, 256, (b, tok), dtype=torch.long),
        tokenized_prompt_mask=torch.ones(b, tok, dtype=torch.bool),
    )
    # Run host-side preprocessing eagerly (not part of the captured graph).
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(raw, train=False)
    noise = torch.randn(b, ah, ad, dtype=torch.float32)
    inputs = (
        images[0], images[1], images[2],
        img_masks[0], img_masks[1], img_masks[2],
        lang_tokens, lang_masks, state, noise,
    )
    return Pi05DenoiseStep(model).eval(), inputs
