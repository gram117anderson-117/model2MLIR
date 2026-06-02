"""AllenAI MolmoAct (PyTorch) capture loader for m2m.

MolmoAct is a Molmo-style VLA: a SigLIP2 ViT vision backbone + an adapter + a
Qwen2-style LLM decoder that autoregressively emits reasoning / depth / action
tokens (the "action reasoning" generation), then decodes action bins. Source code:
https://github.com/allenai/molmoact  (olmo/hf_model/molmoact/modeling_molmoact.py).

NOTE on which repo: the task pointed at allenai/molmoact2, but that repo is only a
FastAPI *inference server* -- it loads the model from the HF Hub
(`AutoModelForImageTextToText.from_pretrained("allenai/MolmoAct2-DROID",
trust_remote_code=True)`) via a hub-side `modeling_molmoact2.py`; no model class
lives in that repo. The full PyTorch model definition lives in allenai/molmoact
(the `olmo` training package), which is what this loader builds against. (MolmoAct2
swaps the autoregressive action head for a flow-matching one, but the LLM/ViT
backbone is the same Molmo lineage.)

Capture unit: the **LLM decoder** as a clean causal-LM `input_ids -> logits` forward
(`MolmoActForCausalLM`), exactly mirroring the tiny_llama example. This is the
representative heavy transformer graph and avoids the generation loop, the ViT image
preprocessing, and the image-token splicing (all data-dependent / host-side).

    get_model_and_inputs() -> (nn.Module, tuple[Tensor, ...])

Env:
    M2M_MOLMOACT_LAYERS=N   LLM decoder layers (default 4; full config is 48)
    M2M_MOLMOACT_VOCAB=N    override vocab_size (default 4096 smoke; full is 152064 ~ a
                            2 GB fp32 embedding+lm_head, so default is shrunk)
    M2M_SEQ=N               sequence length for the example input (default 8)
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

# The model code lives in the (un-installed) molmoact repo; make it importable.
_MOLMOACT_REPO = os.environ.get("MOLMOACT_REPO", "/scratch/agustin/projects/molmoact")
if _MOLMOACT_REPO not in sys.path:
    sys.path.insert(0, _MOLMOACT_REPO)


class _LogitsOnly(nn.Module):
    """Wrap the causal LM so export sees a clean tensor->tensor forward."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, use_cache=False)
        return out.logits


def get_model_and_inputs():
    from olmo.hf_model.molmoact.configuration_molmoact import MolmoActLlmConfig
    from olmo.hf_model.molmoact.modeling_molmoact import MolmoActForCausalLM

    n_layers = int(os.environ.get("M2M_MOLMOACT_LAYERS", "4"))
    vocab = int(os.environ.get("M2M_MOLMOACT_VOCAB", "4096"))
    seq = int(os.environ.get("M2M_SEQ", "8"))

    # Small random-init LLM (full default is 48 layers / hidden 3584 / vocab 152064).
    cfg = MolmoActLlmConfig(
        num_hidden_layers=n_layers,
        vocab_size=vocab,
        use_cache=False,
    )
    cfg._attn_implementation = "eager"  # avoid flash/sdpa quirks during trace
    model = MolmoActForCausalLM(cfg).to(torch.float32).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, seq), dtype=torch.long)
    return _LogitsOnly(model).eval(), (input_ids,)
