"""Whisper tiny (openai/whisper-tiny, via HF transformers) -> MLIR.

    python workloads/capture.py whisper_tiny --formats fp32,int8
    <venv>/bin/python workloads/capture_consistent.py whisper_tiny int8 <bundle_dir>

Capture unit: the audio encoder plus ONE cross-attending decoder step -- the graph a decode
loop executes repeatedly, not the loop itself (``use_cache=False``, so no KV cache is threaded
in or out). That is the same convention the LLM workloads here use, and it means the capture
does NOT prove autoregressive behaviour: it proves the per-step graph.

Log-mel / STFT feature extraction is deliberately outside the capture. It is host-side
preprocessing in Whisper, not part of the model graph, so including it would measure a
spectrogram front end rather than the network.

Env:
    M2M_WHISPER_MODEL     HF id (default: openai/whisper-tiny)
    M2M_WHISPER_LAYERS    truncate BOTH encoder and decoder to N layers (fast smoke;
                          default: the real 4 + 4)
    M2M_WHISPER_FRAMES    mel frames (default: the full 3000 = 30 s; 2 frames per encoder pos)

Upstream: https://github.com/openai/whisper -- weights come from the HF port, which is the
same checkpoint and already in the local HF cache.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

_MODEL_ID = os.environ.get("M2M_WHISPER_MODEL", "openai/whisper-tiny")


class _LogitsOnly(nn.Module):
    """Wrap the seq2seq model so export sees a clean tensors->tensor forward.

    Whisper returns a ``Seq2SeqLMOutput``; the capture contract is a single tensor, and the
    logits are what a decode step consumes.
    """

    def __init__(self, m: nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, input_features: torch.Tensor, decoder_input_ids: torch.Tensor):
        return self.m(input_features=input_features, decoder_input_ids=decoder_input_ids,
                      use_cache=False).logits


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    layers = os.environ.get("M2M_WHISPER_LAYERS")
    # eager attention: the SDPA/flash paths export as opaque fused calls, and the math form is
    # what the frontend decomposes.
    if layers:
        cfg = WhisperConfig.from_pretrained(_MODEL_ID)
        cfg.encoder_layers = cfg.decoder_layers = int(layers)
        cfg.use_cache = False
        cfg._attn_implementation = "eager"
        model = WhisperForConditionalGeneration(cfg)
        print(f"[whisper_tiny] reduced to {layers} enc/dec layers — RANDOM INIT",
              file=sys.stderr)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            _MODEL_ID, dtype=torch.float32, attn_implementation="eager")
        cfg = model.config
    model.config.use_cache = False
    model = model.eval()

    frames = int(os.environ.get("M2M_WHISPER_FRAMES", cfg.max_source_positions * 2))
    if frames != cfg.max_source_positions * 2:
        # Shortening the audio window is not just a smaller input tensor. Whisper's encoder asserts
        # the mel length matches `max_source_positions * 2` exactly ("expects the mel input features
        # to be of length 3000"), because its encoder position embedding is a fixed [1500, d_model]
        # table. Passing fewer frames alone raises; the config and that table have to come with it.
        #
        # This matters because the 30-second window is what makes the model unrunnable on a small
        # board: encoder self-attention materializes [heads, pos, pos] tensors, so at 1500 positions
        # each is 54 MB and the peak working set is ~2 GB regardless of how well the compiler frees.
        # Positions enter that quadratically, so a shorter window is the only lever with the right
        # exponent.
        #
        # The position table is TRUNCATED, not re-initialized: rows 0..N-1 of the pretrained
        # embedding are the ones a real N-position utterance would use, so every weight in the model
        # stays the trained one. What changes is the audio context, and that is a property of the
        # capture -- it must be named in the bundle, never passed off as full whisper_tiny.
        if frames % 2:
            raise ValueError(f"M2M_WHISPER_FRAMES must be even (2 mel frames per encoder "
                             f"position); got {frames}")
        positions = frames // 2
        if positions > cfg.max_source_positions:
            raise ValueError(f"M2M_WHISPER_FRAMES={frames} asks for {positions} encoder positions "
                             f"but the checkpoint only has {cfg.max_source_positions}")
        emb = model.model.encoder.embed_positions
        with torch.no_grad():
            kept = emb.weight[:positions].clone()
        model.model.encoder.embed_positions = nn.Embedding(positions, kept.shape[1])
        with torch.no_grad():
            model.model.encoder.embed_positions.weight.copy_(kept)
        model.model.encoder.embed_positions.requires_grad_(False)
        cfg.max_source_positions = positions
        model.config.max_source_positions = positions
        print(f"[whisper_tiny] audio window shortened to {positions} encoder positions "
              f"({frames} mel frames, {frames / 100:.1f}s) — pretrained weights, TRUNCATED "
              f"position table", file=sys.stderr)
    features = torch.randn(1, cfg.num_mel_bins, frames)
    # One decode step from the start-of-transcript token.
    start = getattr(cfg, "decoder_start_token_id", None) or 0
    decoder_input_ids = torch.tensor([[int(start)]], dtype=torch.long)
    return _LogitsOnly(model).eval(), (features, decoder_input_ids)
