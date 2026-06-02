"""OpenVLA-7b (vision-language-action) capture loader for m2m.

Loads openvla/openvla-7b (Llama-2 7B + DINOv2/SigLIP vision, custom HF code) and exposes
its forward(input_ids, pixel_values) -> logits as the capture unit. `predict_action` wraps
a generation loop (not export-friendly), so we capture the single forward.

Weights (~14 GB) download from the HF hub on first use; needs a dedicated venv (see README).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

_MODEL_ID = "openvla/openvla-7b"


class _LogitsOnly(nn.Module):
    def __init__(self, vla: nn.Module) -> None:
        super().__init__()
        self.vla = vla

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vla(input_ids=input_ids, pixel_values=pixel_values, use_cache=False).logits


def get_model_and_inputs():
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        _MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    image = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    prompt = "In: What action should the robot take to pick up the block?\nOut:"
    inputs = processor(prompt, image)
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(torch.float32)
    return _LogitsOnly(vla).eval(), (input_ids, pixel_values)
