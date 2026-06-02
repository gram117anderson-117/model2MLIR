"""BitVLA (1-bit VLA, OFT variant) capture loader for m2m.

BitVLA is a Llava-style VLA: a W1.58-A8 (BitNet) language model + a (optionally
1-bit) SigLIP vision tower, fine-tuned with OpenVLA-OFT to predict a chunk of
continuous actions in a single bi-directional forward pass (no autoregressive
generation loop).

`BitVLAForActionPrediction.predict_action` is NOT a capture target:
  - heavy host-side prep with `.item()` (`(input_ids == idx).sum().item()`),
    `masked_scatter`, numpy unnormalization, and a proprio/action-head plumbing,
  - requires the vendored `transformers` fork + the `prismatic` package +
    real checkpoints + an image processor.

Capture unit: the inner VLM forward on already-built `inputs_embeds`, i.e.
`LlavaForConditionalGeneration.forward(inputs_embeds=..., use_bi_attn=True)`
returning logits. This is exactly what `predict_action` calls after all the
host-side embedding assembly; it carries the full BitNet LM + bi-directional
attention and the W1.58-A8 BitLinear math (round/clamp/absmean quant, traced
through each autograd.Function's forward). Vision-tower image embedding and the
masked_scatter of image/proprio tokens stay out of the graph (done host-side),
mirroring the smolVLA / tiny_llama conventions.

We instantiate a SMALL random Llava+BitNet config (no weights downloaded). The
vendored BitVLA `transformers` fork MUST be installed (it hardcodes
`BitNetForCausalLM` + `SiglipVisionModel` and the `use_bi_attn` kwarg).

Env:
    BITVLA_LLM_LAYERS=N   BitNet decoder layers (default 2 fast smoke; real: 30)
    BITVLA_SEQ=N          prefix sequence length of the embeds (default 32)
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

# Make the OFT bitvla package + the prismatic package importable.
# NOTE: bitvla_for_action_prediction imports `prismatic.vla.constants` and
# `prismatic.training.train_utils` at MODULE LOAD time, so `openvla-oft/` must be
# on sys.path even though we never run the host-side path that uses them.
_BITVLA_PKG = "/scratch/agustin/projects/BitVLA/openvla-oft/bitvla"
_OFT_ROOT = "/scratch/agustin/projects/BitVLA/openvla-oft"
for _p in (_BITVLA_PKG, _OFT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _VLMLogits(nn.Module):
    """Run the inner bi-directional Llava VLM forward on inputs_embeds -> logits."""

    def __init__(self, vla: nn.Module) -> None:
        super().__init__()
        self.vla = vla

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        from transformers import LlavaForConditionalGeneration

        out = LlavaForConditionalGeneration.forward(
            self.vla,
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            pixel_values=None,
            labels=None,
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_bi_attn=True,  # OFT uses bi-directional attention over action tokens
        )
        return out.logits


def get_model_and_inputs():
    from configuration_bit_vla import Bitvla_Config

    llm_layers = int(os.environ.get("BITVLA_LLM_LAYERS", "2"))
    seq = int(os.environ.get("BITVLA_SEQ", "32"))

    # Small random BitNet text config (W1.58 LM) + small SigLIP vision config.
    # The vendored fork's LlavaForConditionalGeneration hardcodes BitNetForCausalLM
    # and SiglipVisionModel, so only the config dicts matter here.
    text_config = {
        "model_type": "BitNet",
        "vocab_size": 1024,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": llm_layers,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
    }
    vision_config = {
        "model_type": "siglip_vision_model",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_channels": 3,
        "image_size": 224,
        "patch_size": 16,
    }

    cfg = Bitvla_Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=10,
        vision_feature_layer=-1,
        vision_feature_select_strategy="full",
        norm_stats={},
        n_action_bins=256,
    )

    from bitvla_for_action_prediction import BitVLAForActionPrediction

    vla = BitVLAForActionPrediction(cfg).to(torch.float32).eval()
    # Constants only matter for the host-side path we skip; set benign values.
    vla.set_constant(
        image_token_idx=10,
        proprio_pad_idx=11,
        ignore_idx=-100,
        action_token_begin_idx=12,
        stop_index=2,
    )

    hidden = text_config["hidden_size"]
    inputs_embeds = torch.randn(1, seq, hidden, dtype=torch.float32)
    attention_mask = torch.ones(1, seq, dtype=torch.long)
    return _VLMLogits(vla).eval(), (inputs_embeds, attention_mask)
