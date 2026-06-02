# Op taxonomy & scalability strategy

The point of this document: keep the *number of distinct things a downstream pass must
match* small and stable, even as we add support for hundreds of `aten` ops across many
models. We do **not** want "one bespoke pattern per aten op." We want a small, fixed,
matchable vocabulary, with new ops slotting into it rather than inventing new shapes.

## The problem

xDSL (0.65) ships **no named elementwise linalg ops** (`linalg.add`, `linalg.mul`,
`linalg.elementwise`, …) and `AnyFloat` is a closed union. So most pointwise/reduction
ops *must* be emitted as `linalg.generic` with a bespoke body. If a pass had to recognize
"an add" by inspecting generic bodies + indexing maps, every op would effectively be
unique and transforms would not scale.

## The strategy — three tiers

### Tier 1 — standard ops + a universal metadata taxonomy (DONE)
Lower to canonical standard dialects (`linalg`, `tensor`, `arith`, `math`, `scf`) and stamp
**every** emitted op, centrally and automatically (in `import_fx._forward_fx_meta`), with a
two-level tag:

- `m2m.op`     — the fine canonical op-kind: `add`, `mul`, `matmul`, `softmax`, `cumsum`, …
- `m2m.family` — the coarse family (a small fixed set, see below), **authoritative** from
  the central `_FAMILY_OF` map so the whole module uses ONE vocabulary.

A pass matches on `op.attributes["m2m.family"]` (broad sweeps) or `["m2m.op"]` (specific
rewrites) — never by walking a `linalg.generic` body. Because the tagging lives at the
single per-op chokepoint, coverage is 100% and *cannot* be forgotten when a new
decomposition is added.

**Coarse families** (keep this list small; add one only for a *fundamentally* different op):
`elementwise, cast, fill, iota, compare, select, minmax, logical, bitwise, reduce,
arg_reduce, contraction, normalization, layout, concat, gather_scatter, scan, search,
quantize`.

To add a new aten op: write its decomposition, set a `pattern_hint` (the fine kind), and add
that hint to the right family in `_FAMILY_OF`. No new pass-matching surface is created.

### Tier 2 — named extension-dialect ops for *fundamentally distinct* operations
For ops that are semantically distinct AND awkward to match as a tagged generic/scf —
attention-shaped composites and irregular structural ops — prefer a **named op** in an
extension dialect over a bespoke generic. These already exist:

- `linalg_ext`: `softmax`, `layer_norm`, `rms_norm`, `rope`, `swiglu`, `gelu`, `silu`
- `tensor_ext`: `concat`, `pack`, `unpack`
- `quant_ext`:  `quantize/dequantize_per_{tensor,channel,group}`, `weight_int{4,8}pack_mm`,
  `choose_qparams_*`

Naming convention: extension dialects use **bare, standard-adjacent** names (`linalg_ext`
reads as "an extension of linalg", portable / close to upstream MLIR) — *not* a vendor
prefix. `quant_ext` is named to avoid colliding with MLIR's real `quant` dialect. FP8 types
mirror the MLIR-native spelling (`!builtin_ext.f8E4M3FN`) so swapping to native xDSL f8 later
is trivial. Only the **discardable annotation attributes** keep a namespace prefix (`m2m.op`,
`m2m.family`, `m2m.region_id`, …) — MLIR requires one, and standard tooling simply ignores
them, so the *ops* stay standard MLIR.

A named op is matched by **op type** (the strongest possible match) and carries its
parameters as attributes. A single **expansion/legalization pass** then lowers each named
op to the standard-dialect form (the generic/scf we already build) when a target wants pure
linalg-on-tensors. This gives two views from one pipeline: a *high-level, trivially
matchable* IR and a *portable standard* IR — without scattering the lowering.

Good Tier-2 candidates still emitted as bespoke generic/scf today (worth promoting):
`scan` (cumsum), `search` (bucketize), `gather_scatter` (gather / mask_gather /
mask_scatter / index_put). These are exactly the ops where a named op >> a tagged generic.

### Tier 3 — new extension ops only for genuine gaps
When neither a standard op nor an existing extension op fits, add a new op to the relevant
`*_ext` dialect (or a new `arith_ext` if needed). Rule of thumb: a new op must be
*fundamentally* different from everything in the families above. Otherwise extend an
existing family. Take the op's shape from torch-mlir (the differential oracle) so it stays
portable.

## Governance (scalability + coverage)
- **Coverage** is enforced by: the importer verify-fallback (invalid → opaque, never silent
  wrong IR), the differential oracle (`m2m.coverage.differential_op` vs torch-mlir), and the
  per-model gap reports. New gaps surface as opaque `func.call @aten_*` you can grep for.
- **Scalability** is enforced by this taxonomy: the matchable vocabulary is bounded
  (~19 families) and fixed; adding ops grows the `aten → family` map, not the pass surface.
- Measure anytime with `m2m.coverage.dialect_op_histogram(text)` and by grepping
  `m2m.family = "…"` / `m2m.op = "…"` across the emitted `.mlir`.
