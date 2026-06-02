"""Frontends: torch (capture/ + ir/) and jax (jax.export -> StableHLO).

Both produce the same `m2m.api.ConversionResult` (standard-dialect MLIR text + metadata),
so downstream consumers don't care which framework the model came from.
"""
