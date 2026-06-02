"""TorchAO quantization pipeline integration.

Integrates TorchAO's quantization and sparsity workflows into the CompGen
capture pipeline. Quantization decisions affect kernel contracts, layout
requirements, and verification (quantized paths need separate golden outputs).

Invariants:
    - Quantization config must be serializable (part of the recipe).
    - Accuracy degradation from quantization is measured and reported.
    - Quantized models produce separate golden outputs for verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization configuration.

    Attributes:
        scheme: Quantization scheme (e.g., "int8_weight_only", "int4_weight_only", "fp8").
        calibration_samples: Number of calibration samples.
        group_size: Group size for grouped quantization.
        extra_args: Additional scheme-specific arguments.
    """

    scheme: str
    calibration_samples: int = 100
    group_size: int | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccuracyReport:
    """Quantization accuracy report.

    Attributes:
        l2_error: L2 norm of output difference.
        max_abs_error: Maximum absolute output difference.
        cosine_similarity: Cosine similarity between original and quantized outputs.
        within_tolerance: Whether errors are within acceptable bounds.
        tolerance: The tolerance threshold used.
    """

    l2_error: float
    max_abs_error: float
    cosine_similarity: float
    within_tolerance: bool
    tolerance: float


def _rematerialize_parameters(model: Any) -> None:
    """Replace every parameter with a fresh nn.Parameter holding a cloned tensor.

    This drops any outstanding weakrefs on the original parameter objects so
    torchao's swap_tensors-based quantize_ can proceed on HuggingFace models.
    """
    import torch.nn as nn

    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if param is not None:
                module._parameters[name] = nn.Parameter(
                    param.detach().clone(), requires_grad=param.requires_grad
                )


def apply_quantization(model: Any, config: QuantizationConfig) -> Any:
    """Apply TorchAO quantization to a model.

    Args:
        model: PyTorch nn.Module.
        config: Quantization configuration.

    Returns:
        Quantized model (modified in-place or new module).

    """
    # NOTE: CompGen's NPU-custom FP8 schemes ("fp8_e4m3_po2[_npu]") were dropped
    # during extraction (they depended on NPU-specific modules). Reintroduce them
    # as a m2m.quant extension if needed.

    # torchao's quantize_ (and the later .to()/.eval() during capture) swap weight
    # tensors via torch.utils.swap_tensors, which refuses to swap a tensor that has
    # an outstanding weakref -- HuggingFace models leave such weakrefs, producing
    # "Cannot swap ... because it has weakref associated with it". Two mitigations:
    #   1. re-materialize every parameter as a fresh object (drops weakrefs), and
    #   2. disable swap-on-conversion process-wide so the post-quantization capture
    #      (.to()/.eval() on quantized-subclass weights) uses copy semantics.
    _rematerialize_parameters(model)
    try:
        torch.__future__.set_swap_module_params_on_conversion(False)
    except Exception:  # noqa: BLE001 - older torch without the toggle
        pass

    # Legacy TorchAO schemes -- handle both pre-0.17 (function-based) and
    # 0.17+ (Config-class-based) APIs.
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError("torchao is not installed") from exc

    scheme_map: dict[str, Any] = {}

    # torchao 0.17+: Config classes (preferred)
    try:
        from torchao.quantization import Int8WeightOnlyConfig

        scheme_map["int8_weight_only"] = Int8WeightOnlyConfig
    except ImportError:
        pass
    try:
        from torchao.quantization import Int4WeightOnlyConfig

        scheme_map["int4_weight_only"] = Int4WeightOnlyConfig
    except ImportError:
        pass
    try:
        from torchao.quantization import Float8WeightOnlyConfig

        scheme_map["fp8"] = Float8WeightOnlyConfig
    except ImportError:
        pass
    # int8 W + int8 A for systolic / RVV int8 datapaths (Gemmini, Saturn/OPU).
    # Per-channel weights, per-tensor dynamic activations.
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig

        scheme_map["int8_dyn_act_int8_weight"] = Int8DynamicActivationInt8WeightConfig
    except ImportError:
        pass
    try:
        from torchao.quantization import Int8StaticActivationInt8WeightConfig

        scheme_map["int8_static_act_int8_weight"] = Int8StaticActivationInt8WeightConfig
    except ImportError:
        pass

    # torchao <0.17: function-based API (fallback)
    if "int8_weight_only" not in scheme_map:
        try:
            from torchao.quantization import int8_weight_only

            scheme_map["int8_weight_only"] = int8_weight_only
        except ImportError:
            pass
    if "int4_weight_only" not in scheme_map:
        try:
            from torchao.quantization import int4_weight_only

            scheme_map["int4_weight_only"] = int4_weight_only
        except ImportError:
            pass
    if "fp8" not in scheme_map:
        try:
            from torchao.quantization import float8_weight_only

            scheme_map["fp8"] = float8_weight_only
        except ImportError:
            pass

    factory = scheme_map.get(config.scheme)
    if factory is None:
        # Fall back to the scheme catalog (covers fp8 variants, mx, nvfp4, etc.):
        # resolve the scheme's config_class_path to a TorchAO Config class.
        from m2m.capture.torchao_schemes import TORCHAO_SCHEMES, resolve_config

        scheme_obj = TORCHAO_SCHEMES.get(config.scheme)
        if scheme_obj is not None:
            factory = resolve_config(scheme_obj)
    if factory is None:
        raise ValueError(f"Unsupported TorchAO scheme: {config.scheme}")

    quantizer = factory()
    quantize_(model, quantizer)
    # torchao's quantize_ re-enables swap-on-conversion; turn it back off so the
    # subsequent capture (.to()/.eval() on quantized-subclass weights) uses copy
    # semantics and doesn't trip the weakref guard.
    try:
        torch.__future__.set_swap_module_params_on_conversion(False)
    except Exception:  # noqa: BLE001
        pass
    return model


def verify_quant_accuracy(
    original_model: Any,
    quantized_model: Any,
    test_inputs: Any,
    tolerance: float = 0.01,
) -> AccuracyReport:
    """Verify quantization accuracy against the original model."""
    with torch.no_grad():
        reference = original_model(*test_inputs)
        candidate = quantized_model(*test_inputs)

    diff = (reference - candidate).float()
    ref_norm = torch.linalg.vector_norm(reference.float()).item()
    cand_norm = torch.linalg.vector_norm(candidate.float()).item()
    l2_error = torch.linalg.vector_norm(diff).item()
    max_abs_error = diff.abs().max().item()

    denom = max(ref_norm * cand_norm, 1e-12)
    cosine_similarity = torch.sum(reference.float() * candidate.float()).item() / denom
    within_tolerance = max_abs_error <= tolerance

    return AccuracyReport(
        l2_error=l2_error,
        max_abs_error=max_abs_error,
        cosine_similarity=float(cosine_similarity),
        within_tolerance=within_tolerance,
        tolerance=tolerance,
    )


__all__ = ["AccuracyReport", "QuantizationConfig", "apply_quantization", "verify_quant_accuracy"]
