"""MX-Gemmini quantization for any PyTorch model.

Implements the exact quantization semantics of MX-Gemmini's golden model
(chipyard gemmini-rocc-tests golden_model.py):

  - per-row, per-32-element groups along the contraction (input) dim
  - power-of-two scales: fpE8M0, scale = 2^ceil(log2(group_absmax / 240))
  - element quantization: fp8 E4M3 with round-toward-zero, subnormals -> 0,
    saturation at +-240 (= golden_model e4m3 max-finite)
  - dequant = q * scale

Provides:
  - quantize_mx_gemmini(w):    fake-quantized tensor + (codes, scale exponents)
  - MXGemminiFakeQuantConfig:  torchAO config; works with torchao.quantize_()
  - apply_mx_gemmini_(model):  no-torchao fallback for plain fake quantization

A model fake-quantized with this scheme is "MX-Gemmini viable": its linear
weights take only values representable on the accelerator (fp8 e4m3 x 2^k),
so spike/RTL kernels reproduce its numerics exactly.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

GROUP = 32
E4M3_MAX = 240.0  # golden_model e4m3 max-finite: IEEE-style emax=7 -> 240 (not OCP 448)
_MAN = 3       # mantissa bits
_BIAS = 7      # e4m3 exponent bias


def fp8_e4m3_rtz(x: Tensor) -> Tensor:
    """fp8 E4M3 quantization, round-toward-zero, subnormals flushed, saturate.

    Matches golden_model.float_quantize_trunc(exp=4, man=3).
    """
    out = torch.zeros_like(x, dtype=torch.float32)
    x = x.to(torch.float32)
    ax = x.abs()
    nz = ax > 0
    if not torch.any(nz):
        return out

    sign = torch.sign(x[nz])
    a = ax[nz]
    E = torch.floor(torch.log2(a))
    emin, emax = 1 - _BIAS, _BIAS

    base = torch.pow(torch.tensor(2.0, dtype=a.dtype), E.clamp(emin, emax))
    delta = base / (2 ** _MAN)
    k = torch.floor(torch.clamp((a - base) / delta, 0, 2 ** _MAN - 1 - 1e-7))
    q = base + k * delta
    q = torch.where(E < emin, torch.zeros_like(q), q)   # flush subnormal
    q = torch.where(E > emax, torch.full_like(q, E4M3_MAX), q)  # saturate

    out[nz] = sign * q
    return out


def quantize_mx_gemmini(w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize a [out, in] weight tensor with per-row, per-32-group MX scales.

    Returns (w_fake_quant, q_codes_float, scale_exponents):
      - w_fake_quant:    q * scale (e4m3-on-grid * power-of-two)
      - q_codes_float:   fp8 e4m3 values (pre-scale)
      - scale_exponents: integer exponents e, scale = 2**e (fpE8M0 grid)
    """
    O, I = w.shape
    if I % GROUP != 0:
        raise ValueError(f"input dim {I} not a multiple of MX group {GROUP}")
    wg = w.to(torch.float32).reshape(O, I // GROUP, GROUP)
    gmax = wg.abs().amax(dim=-1, keepdim=True)
    e = torch.ceil(torch.log2(gmax / E4M3_MAX))
    e = torch.where(torch.isfinite(e), e, torch.zeros_like(e)).clamp(-127, 127)
    scale = torch.pow(torch.tensor(2.0), e)
    q = fp8_e4m3_rtz(wg / scale)
    return (q * scale).reshape(O, I), q.reshape(O, I), e.squeeze(-1)


@torch.no_grad()
def apply_mx_gemmini_(model: nn.Module) -> nn.Module:
    """Fake-quantize every nn.Linear weight in-place (no torchao required)."""
    for mod in model.modules():
        if isinstance(mod, nn.Linear) and mod.weight.shape[1] % GROUP == 0:
            mod.weight.copy_(quantize_mx_gemmini(mod.weight)[0])
    return model


# --------------------------- torchAO integration ---------------------------
try:
    from torchao.core.config import AOBaseConfig
    from torchao.quantization.transform_module import register_quantize_module_handler

    class MXGemminiFakeQuantConfig(AOBaseConfig):
        """torchAO config: MX-Gemmini fp8(e4m3) weights, 32-elem fpE8M0 scales."""

    @register_quantize_module_handler(MXGemminiFakeQuantConfig)
    def _mx_gemmini_transform(module: nn.Module, config: MXGemminiFakeQuantConfig) -> nn.Module:
        if isinstance(module, nn.Linear) and module.weight.shape[1] % GROUP == 0:
            with torch.no_grad():
                module.weight.copy_(quantize_mx_gemmini(module.weight)[0])
        return module

except ImportError:  # torchao not installed — fallback API still available
    MXGemminiFakeQuantConfig = None  # type: ignore
