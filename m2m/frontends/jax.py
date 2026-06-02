"""JAX frontend: a jax callable -> StableHLO MLIR (a standard MLIR dialect).

JAX lowers to StableHLO natively, so we get standard MLIR without maintaining a
converter. The path to the shared linalg-on-tensors target is `stablehlo-legalize-to-linalg`
(documented; future work, drivable by the same differential-oracle approach in
m2m.coverage). For now the common output is StableHLO, reported through the same
ConversionResult as the torch path.
"""

from __future__ import annotations

from typing import Any, Callable


def to_stablehlo(fn: Callable[..., Any], example_inputs: tuple[Any, ...]) -> str:
    """Lower a jax function to StableHLO MLIR text."""
    import jax

    lowered = jax.jit(fn).lower(*example_inputs)
    return str(lowered.compiler_ir(dialect="stablehlo"))


def convert_jax(fn: Callable[..., Any], example_inputs: tuple[Any, ...]):
    """Convert a jax callable to MLIR (StableHLO). Returns the shared ConversionResult."""
    from m2m.api import ConversionResult

    text = to_stablehlo(fn, tuple(example_inputs))
    return ConversionResult(
        mlir_text=text,
        module=None,
        path_taken="jax_stablehlo",
        output_type="stablehlo",
        frontend="jax",
        diagnostics=[f"jax -> stablehlo: {len(text)} chars"],
    )


__all__ = ["convert_jax", "to_stablehlo"]
