"""OpenVLA-7b (vision-language-action) capture loader for m2m.

OpenVLA (`openvla/openvla-7b`) is a Prismatic VLM: a *fused* DINOv2 + SigLIP ViT
vision backbone -> a 3-layer GELU MLP projector -> a Llama-2 7B causal LM. The custom
HF modeling code (`modeling_prismatic.py` / `configuration_prismatic.py`) is fetched from
the hub (just the .py + config.json, ~tens of KB, NO 14 GB weights).

Capture unit: the single multimodal `forward(input_ids, pixel_values) -> logits`
(the `PrismaticForConditionalGeneration.forward` multimodal branch:
vision_backbone -> projector -> concat-with-text-embeds -> language_model -> logits).
`predict_action` wraps a `.generate()` loop (data-dependent, not export-friendly), so we
capture the single forward, mirroring the other VLA loaders.

To stay tractable we build the *real* OpenVLA architecture from a SMALL RANDOM config
(no weights downloaded, no OOM):
  - LLM: Llama with very few layers / small hidden / small vocab (env-overridable),
  - vision: two tiny timm ViTs (still a fused DINO+SigLIP-shaped backbone),
  - eager attention, short token sequence, small image.

The upstream custom code hard-pins `timm` (0.9.x) and warns on `transformers != 4.40.1`;
we run modern timm/transformers, so we (a) neutralize the timm version assert and
(b) fix `get_intermediate_layers` (timm 1.x returns a *list*, the upstream `unpack_tuple`
only unwraps tuples) by re-wrapping each featurizer's forward to return the last entry.

Env:
    M2M_OPENVLA_LLM_LAYERS=N   Llama decoder layers (default 2; real is 32)
    M2M_OPENVLA_HIDDEN=N       Llama hidden size (default 128; real is 4096)
    M2M_OPENVLA_VOCAB=N        Llama vocab size (default 512; real is 32064)
    M2M_OPENVLA_VIT_LAYERS=N   ViT blocks per vision backbone (default 2; real is 24/27)
    M2M_OPENVLA_IMG=N          image resolution per backbone (default 64; real is 224)
    M2M_SEQ=N                  text token sequence length (default 4)
"""

from __future__ import annotations

import os

import torch
from torch import nn

_MODEL_ID = "openvla/openvla-7b"


class _LogitsOnly(nn.Module):
    """Wrap the VLM so export sees a clean (input_ids, pixel_values) -> logits forward."""

    def __init__(self, vla: nn.Module) -> None:
        super().__init__()
        self.vla = vla

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vla(
            input_ids=input_ids,
            pixel_values=pixel_values,
            use_cache=False,
            return_dict=True,
        ).logits


def _build_small_config():
    """Real OpenVLAConfig with a drastically shrunk LLM + tiny ViT vision backbones."""
    from transformers import AutoConfig

    llm_layers = int(os.environ.get("M2M_OPENVLA_LLM_LAYERS", "2"))
    hidden = int(os.environ.get("M2M_OPENVLA_HIDDEN", "128"))
    vocab = int(os.environ.get("M2M_OPENVLA_VOCAB", "512"))

    cfg = AutoConfig.from_pretrained(_MODEL_ID, trust_remote_code=True)

    # Shrink the Llama-2 text backbone (this is the 14 GB part).
    tc = cfg.text_config
    tc.num_hidden_layers = llm_layers
    tc.hidden_size = hidden
    tc.intermediate_size = 2 * hidden
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 4
    tc.vocab_size = vocab
    tc.pad_token_id = None  # upstream pad_token_id=32000 is out of our small vocab range
    tc.bos_token_id = 1
    tc.eos_token_id = 2
    tc.max_position_embeddings = 2048
    tc.tie_word_embeddings = False  # let torchao swap lm_head / embeddings independently
    tc._attn_implementation = "eager"
    cfg._attn_implementation = "eager"
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"

    # Shrink the fused DINO+SigLIP vision backbone to two tiny ViTs (keeps the fused
    # 6-channel-split + dual-featurizer + channel-stack structure intact).
    img = int(os.environ.get("M2M_OPENVLA_IMG", "64"))
    cfg.timm_model_ids = ["vit_tiny_patch16_224", "vit_small_patch16_224"]
    cfg.timm_override_act_layers = [None, None]
    cfg.image_sizes = [img, img]
    cfg.use_fused_vision_backbone = True
    return cfg


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    import timm

    cfg = _build_small_config()

    # Resolve the custom `OpenVLAForActionPrediction` class from the hub-side dynamic
    # module. (transformers 5.x dropped `AutoModelForVision2Seq`, the class the config's
    # auto_map points at, so we load the modeling module directly instead of via Auto*.)
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module(
        "modeling_prismatic.OpenVLAForActionPrediction", _MODEL_ID
    )

    # transformers 5.x calls `tie_weights(recompute_mapping=...)`, but the upstream custom
    # class overrides `tie_weights(self)` (no kwargs) -> make it kwargs-tolerant. Llama
    # doesn't tie weights, so this stays a no-op delegate to the language model.
    def _tie_weights(self, *args, **kwargs):
        lm = getattr(self, "language_model", None)
        if lm is not None and hasattr(lm, "tie_weights"):
            try:
                lm.tie_weights()
            except TypeError:
                pass

    for _cls in (model_cls, model_cls.__mro__[1]):
        _cls.tie_weights = _tie_weights
    # Neutralize the `timm.__version__ in {...}` gate (we run timm 1.x intentionally).
    if not hasattr(timm, "_m2m_version_patched"):
        timm._m2m_orig_version = timm.__version__
        timm.__version__ = "0.9.16"
        timm._m2m_version_patched = True

    vit_layers = int(os.environ.get("M2M_OPENVLA_VIT_LAYERS", "2"))

    # Build tiny ViTs ourselves and shrink depth, then hand them to timm.create_model
    # via a wrapper so PrismaticVisionBackbone picks up the small featurizers.
    _orig_create = timm.create_model

    def _small_create_model(name, *args, **kwargs):
        m = _orig_create(name, *args, **kwargs)
        # Truncate transformer depth to keep the graph small.
        if hasattr(m, "blocks") and len(m.blocks) > vit_layers:
            m.blocks = nn.Sequential(*list(m.blocks[:vit_layers]))
        return m

    timm.create_model = _small_create_model
    try:
        model = model_cls(cfg).eval()
    finally:
        timm.create_model = _orig_create

    # timm 1.x `get_intermediate_layers` returns a *list*; upstream `unpack_tuple` only
    # unwraps tuples, so the backbone forward would return a list and break torch.cat.
    # Re-wrap each featurizer forward to return the second-to-last block's patch tensor.
    vb = model.vision_backbone

    def _make_forward(feat):
        n_blocks = len(feat.blocks)
        idx = {max(n_blocks - 2, 0)}

        def _fwd(x, _feat=feat, _idx=idx):
            out = _feat.get_intermediate_layers(x, n=_idx)
            return out[0] if isinstance(out, (list, tuple)) else out

        return _fwd

    vb.featurizer.forward = _make_forward(vb.featurizer)
    if getattr(vb, "use_fused_vision_backbone", False):
        vb.fused_featurizer.forward = _make_forward(vb.fused_featurizer)
    # Recompute the (possibly changed) embed_dim from the truncated backbones.
    vb.embed_dim = vb.featurizer.embed_dim + (
        vb.fused_featurizer.embed_dim if getattr(vb, "use_fused_vision_backbone", False) else 0
    )

    model = model.to(torch.float32).eval()

    # Inputs: short token sequence + fused 6-channel image (two stacked RGB views).
    seq = int(os.environ.get("M2M_SEQ", "4"))
    img = cfg.image_sizes[0]
    vocab = cfg.text_config.vocab_size
    input_ids = torch.randint(0, vocab, (1, seq), dtype=torch.long)
    pixel_values = torch.randn(1, 6, img, img, dtype=torch.float32)

    return _LogitsOnly(model).eval(), (input_ids, pixel_values)
