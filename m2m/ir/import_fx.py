"""FX graph to xDSL/MLIR conversion.

Converts PyTorch FX graphs (from torch.export) into CompGen's canonical
Payload IR using real xDSL linalg/arith/tensor ops where decompositions
exist, and opaque func.call for ops without known decompositions.

Invariants:
    - Every FX node maps to at least one xDSL op (or a diagnostic).
    - Decomposed ops get ``compgen.region_id`` attributes for Recipe IR targeting.
    - Unsupported ops fall back to ``func.call`` (flagged as opaque).
    - The output module passes the xDSL verifier.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

import torch


# Strip Python object memory addresses from built-in callable names so
# region/dispatch ids are deterministic across reruns. dynamo records
# certain targets as ``<built-in method tanh of type object at 0x...>``;
# the address moves heap-randomly and would leak into the MLIR func.call
# callee name and every downstream identifier (region_id, candidate_id,
# region_dossier filenames). Mirrors lower.py::_canonicalize_target_string
# but applied at the IR-emission step rather than only on diagnostics.
_HEX_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+>")


def _canonicalize_fx_target_str(s: str) -> str:
    """Return a canonical, address-free string for an FX target.

    For ``<built-in method tanh of type object at 0x70adfe129b40>``
    returns ``<built-in method tanh of type object>`` (stable across
    reruns / machines). Other forms pass through unchanged.
    """
    return _HEX_ADDR_RE.sub(">", s)
from xdsl.dialects.builtin import (
    BFloat16Type,
    FlatSymbolRefAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    FunctionType,
    ModuleOp,
    StringAttr,
    TensorType,
)
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp


def FlatSymbolRefAttr_ref(name: str) -> FlatSymbolRefAttr:
    """Helper: build a FlatSymbolRefAttr from a plain string name."""
    return FlatSymbolRefAttr(name)


from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
from xdsl.printer import Printer

from m2m.ir.decompositions import (
    DECOMPOSITION_TABLE,
    DecompFn,
    reset_region_counters,
)
from m2m.ir.types import Float8E4M3FNType, Float8E5M2Type

# Tags the FX-side graph passes (in ``compgen.transforms.graph_passes``) set
# on ``node.meta``. ``FXImporter`` forwards each onto the emitted xDSL ops
# so downstream Recipe-IR passes don't have to re-detect patterns the FX
# stage already recognized.
_FX_META_FORWARD_KEYS = (
    "_compgen_pattern",
    "_compgen_transpose_absorbed",
    "_compgen_fuse_dequant",
)


# ---------------------------------------------------------------------------
# Canonical op taxonomy (the matchable vocabulary for downstream passes).
#
# Every emitted op is stamped with TWO attributes, centrally and automatically:
#   m2m.op     -- the fine canonical op-kind  (e.g. "add", "matmul", "softmax")
#   m2m.family -- the coarse family it belongs to (a small, fixed set below)
# so a pass matches on an attribute, never by inspecting linalg.generic bodies.
# Adding a new aten op means slotting its hint into one of these families --
# NOT inventing a new pattern. Keep this vocabulary small; only add a family
# when an op is *fundamentally* different from every existing one.
# ---------------------------------------------------------------------------
_FAMILY_OF: dict[str, str] = {}
def _reg(family: str, *hints: str) -> None:
    for h in hints:
        _FAMILY_OF[h] = family

_reg("elementwise", "add", "mul", "sub", "div", "neg", "reciprocal", "pow",
     "pow_tensor_scalar", "exp", "sqrt", "rsqrt", "tanh", "abs", "floor", "ceil",
     "log", "cos", "sin", "erf", "round", "sigmoid", "silu", "gelu", "relu", "clamp",
     "copy", "clone", "contiguous", "identity")
_reg("cast", "dtype_cast", "to_dtype")
_reg("fill", "fill", "empty", "scalar_tensor")
_reg("iota", "arange")
_reg("compare", "compare")
_reg("select", "where")
_reg("minmax", "maximum", "minimum", "minmax")
_reg("logical", "logical_not")
_reg("bitwise", "bitwise_and", "bitwise_not")
_reg("reduce", "reduce", "reduce_sum", "reduce_mean", "sum", "mean", "any", "bool_reduce")
_reg("arg_reduce", "aten_min_dim", "aten_max_dim", "aten_argmin", "aten_argmax")
_reg("contraction", "matmul", "batch_matmul", "int_matmul", "addmm", "linear",
     "conv2d", "convolution", "convolution_im2col_matmul",
     "weight_int8pack_mm", "weight_int4pack_mm", "weight_int4pack_qm")
_reg("normalization", "softmax", "layer_norm")
_reg("layout", "view", "reshape", "unsqueeze", "squeeze", "flatten", "permute",
     "transpose", "expand", "slice", "select", "slice_scatter", "split", "repeat")
_reg("concat", "cat", "concat")
_reg("gather_scatter", "gather", "embedding", "embedding_lookup", "index_gather",
     "mask_gather", "index_put", "mask_scatter")
_reg("scan", "cumsum")
_reg("search", "bucketize")
_reg("quantize", "quantize_per_tensor", "quantize_per_channel", "quantize_per_group",
     "dequantize_per_tensor", "dequantize_per_channel", "dequantize_per_group",
     "choose_qparams_per_tensor", "choose_qparams_per_channel", "dequantize", "quant_param")


def family_of(hint: str | None) -> str | None:
    """Coarse family for a fine op-kind hint (the matchable vocabulary). None if unknown."""
    return _FAMILY_OF.get(hint) if hint else None


def high_level_named_ops() -> set:
    """The named ``*_ext`` op types the high-level form (``emit_named_ops``) can emit.
    Every one MUST have an expander in ``m2m.transforms.expand_ext.EXPANDERS`` (enforced by
    tests/test_transforms.py). Lazily imported to avoid a dialect import at module load."""
    from m2m.ir.linalg_ext.ops import LayerNormOp, SoftmaxOp
    return {SoftmaxOp, LayerNormOp}


# Fallback when a decomposition emitted a named op but set no pattern_hint: infer the
# (op-kind, family) straight from the MLIR op name, so tagging is 100% regardless of
# whether each decomposition remembered a hint. Maps op.name -> (m2m.op, m2m.family).
_OP_TYPE_TAXONOMY: dict[str, tuple[str, str]] = {
    "linalg.matmul": ("matmul", "contraction"),
    "linalg.batch_matmul": ("batch_matmul", "contraction"),
    "linalg.quantized_matmul": ("matmul", "contraction"),
    "linalg.transpose": ("transpose", "layout"),
    "linalg.broadcast": ("broadcast", "layout"),
    "linalg.reduce": ("reduce", "reduce"),
    "linalg.fill": ("fill", "fill"),
    "linalg.softmax": ("softmax", "normalization"),
    "linalg.conv_2d": ("conv2d", "contraction"),
    "tensor.concat": ("concat", "concat"),
    "tensor.extract_slice": ("slice", "layout"),
    "tensor.insert_slice": ("slice_scatter", "layout"),
    "tensor.collapse_shape": ("reshape", "layout"),
    "tensor.expand_shape": ("reshape", "layout"),
}


def _forward_fx_meta(
    op: Operation,
    fx_meta: dict[str, Any],
    decomp_hint: str | None = None,
) -> None:
    """Copy FX node meta + DecompResult.pattern_hint onto ``op.attributes`` and stamp the
    canonical taxonomy (``m2m.op`` fine kind + ``m2m.family`` coarse family) on EVERY op.

    - ``_compgen_pattern`` (FX-level tag) -> ``m2m._pattern_hint``
    - ``decomp_hint`` (decomp-side explicit tag) wins when FX didn't set one.
    - ``m2m.op`` <- effective hint; ``m2m.family`` <- family_of(hint), backfilled only
      when a decomposition didn't already set an (authoritative) family. This makes
      family tagging 100% consistent regardless of whether each decomposition remembered.

    Idempotent: won't overwrite an existing attribute.
    """
    fx_hint = fx_meta.get("_compgen_pattern") if isinstance(fx_meta, dict) else None
    effective_hint = fx_hint or decomp_hint
    if effective_hint and "m2m._pattern_hint" not in op.attributes:
        op.attributes["m2m._pattern_hint"] = StringAttr(str(effective_hint))
    # canonical two-level taxonomy (matchable by downstream passes):
    #   m2m.op     = fine op-kind (the hint)            -- preserved as-is
    #   m2m.family = coarse family from the fixed map   -- AUTHORITATIVE: overwrite any
    #     ad-hoc family a decomposition set, so the whole module uses ONE vocabulary.
    fam = family_of(effective_hint)
    op_kind = str(effective_hint) if effective_hint else None
    if fam is None:
        # no usable hint -> infer from the op's own type (named ops only)
        type_tax = _OP_TYPE_TAXONOMY.get(getattr(op, "name", ""))
        if type_tax is not None:
            op_kind, fam = type_tax
    if op_kind and "m2m.op" not in op.attributes:
        op.attributes["m2m.op"] = StringAttr(op_kind)
    if fam is not None:
        op.attributes["m2m.family"] = StringAttr(fam)  # authoritative coarse family

    if isinstance(fx_meta, dict):
        if fx_meta.get("_compgen_transpose_absorbed") and "m2m.transpose_absorbed" not in op.attributes:
            op.attributes["m2m.transpose_absorbed"] = StringAttr("true")
        if fx_meta.get("_compgen_fuse_dequant") and "m2m.fuse_dequant" not in op.attributes:
            op.attributes["m2m.fuse_dequant"] = StringAttr("true")
        # provenance: trace each op back to its PyTorch origin (the source aten op) and the
        # original torch dtype before any coercion (e.g. fp8 -> f32). Lets downstream/merlin
        # recover "what this represents in PyTorch" and the true low-precision dtype.
        aten = fx_meta.get("_aten_target")
        if aten and "m2m.provenance.aten" not in op.attributes:
            op.attributes["m2m.provenance.aten"] = StringAttr(str(aten))
        if "m2m.provenance.orig_dtype" not in op.attributes:
            val = fx_meta.get("val")
            dt = getattr(val[0] if isinstance(val, (tuple, list)) and val else val, "dtype", None)
            if dt is not None:
                op.attributes["m2m.provenance.orig_dtype"] = StringAttr(str(dt).replace("torch.", ""))


def _torch_dtype_to_xdsl(dtype: torch.dtype) -> Attribute:
    """Convert a torch dtype to an xDSL element type."""
    mapping: dict[torch.dtype, type] = {
        torch.float32: Float32Type,
        torch.float64: Float64Type,
        torch.float16: Float16Type,
        torch.bfloat16: BFloat16Type,
    }
    # FP8: torch's float8 dtypes (e4m3fn / e5m2) have MLIR-native equivalents
    # (f8E4M3FN, f8E5M2) but xDSL 0.65 ships no builtin Float8 type and its
    # `AnyFloat` is a closed union, so linalg body-builders (transpose/generic)
    # reject our custom Float8 type with an AnyFloat assertion. Since we can't
    # patch xDSL (and fp8 here is for representation, not execution), fp8 tensors
    # flow through the compute path as f32; the exact torchAO fp8 scheme is recorded
    # as the module attribute `m2m.quantization` so the quantization is still captured.
    # `Float8E4M3FNType`/`Float8E5M2Type` remain available for a future xDSL that
    # supports f8 natively (or a torch-mlir round-trip).
    _f8 = {getattr(torch, n, None) for n in ("float8_e4m3fn", "float8_e5m2",
                                             "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu")}
    if dtype in _f8:
        return Float32Type()
    # Bool -> i1 (so arith.select conds / comparison results verify) and integers ->
    # their bitwidth, instead of silently defaulting to f32.
    from xdsl.dialects.builtin import IntegerType

    if dtype == torch.bool:
        return IntegerType(1)
    int_bits = {torch.int8: 8, torch.uint8: 8, torch.int16: 16, torch.int32: 32, torch.int64: 64}
    if dtype in int_bits:
        return IntegerType(int_bits[dtype])
    factory = mapping.get(dtype, Float32Type)
    return factory()  # type: ignore[abstract]


def _coerce_static_dim(dim: Any) -> int:
    """Concrete dim → ``int(dim)``; symbolic / data-dependent dim → ``-1``.

    xDSL's ``TensorType`` rejects ``SymInt`` (``"u6 should be of base
    attribute builtin.int"``). For models with dynamic shapes (SmolVLA's
    image-tile counts, etc.) we emit ``-1`` (xDSL's dynamic-dim convention)
    so capture continues; downstream passes that need static shapes will
    short-circuit through their own dynamic-shape paths.
    """
    try:
        return int(dim)
    except Exception:
        return -1


def _tensor_type_from_meta(val: Any) -> TensorType | None:
    """Extract a TensorType from an FX node's meta['val'].

    Tuple/list-returning ops (``aten.native_layer_norm`` →
    ``(out, mean, rstd)``, ``aten.var_mean`` → ``(mean, var)``, …)
    surface as Python tuples in ``meta['val']``. We use the *primary*
    element (index 0) as the node's representative type; downstream
    ``operator.getitem(node, 0)`` consumers then resolve to the
    decomposition's single-tensor result rather than crashing as
    orphan opaque calls (REQ-020).
    """
    if val is None:
        return None
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        elem = _torch_dtype_to_xdsl(val.dtype)
        shape = [_coerce_static_dim(d) for d in val.shape]
        return TensorType(elem, shape)
    if isinstance(val, (tuple, list)) and val:
        # Tuple-returning op — recurse on the primary element.
        return _tensor_type_from_meta(val[0])
    return None


@dataclass
class ImportDiagnostic:
    """Diagnostic from an import operation.

    Attributes:
        fx_node: Name of the FX node that produced this diagnostic.
        level: "error", "warning", or "info".
        message: Human-readable description.
    """

    fx_node: str
    level: str
    message: str


@dataclass
class FXImporter:
    """Converts a PyTorch FX graph to an xDSL module.

    Uses the decomposition table from ``decompositions.py`` to produce
    real xDSL ops (linalg.matmul, linalg.transpose, etc.) where possible.
    Falls back to opaque func.call for undecomposed ops.
    """

    diagnostics: list[ImportDiagnostic] = field(default_factory=list)
    decomposed_count: int = 0
    opaque_count: int = 0
    allow_opaque_fallback: bool = True
    explicit_blackboxes: set[str] = field(default_factory=set)
    dynamic_decompositions: dict[str, DecompFn] = field(default_factory=dict)
    emit_named_ops: bool = False  # high-level form: emit linalg_ext.* composites

    @property
    def decomposition_coverage(self) -> float:
        """Fraction of ops that were decomposed to real xDSL ops."""
        total = self.decomposed_count + self.opaque_count
        return self.decomposed_count / total if total > 0 else 1.0

    def import_graph(self, exported_program: Any) -> ModuleOp:
        """Convert an ExportedProgram's FX graph to an xDSL module."""
        reset_region_counters()
        graph = exported_program.graph
        nodes = list(graph.nodes)

        placeholders = [n for n in nodes if n.op == "placeholder"]
        call_nodes = [n for n in nodes if n.op == "call_function"]
        output_nodes = [n for n in nodes if n.op == "output"]

        # Build xDSL types for each node from meta
        node_types: dict[str, TensorType] = {}
        for node in nodes:
            val = node.meta.get("val")
            tt = _tensor_type_from_meta(val)
            if tt is not None:
                node_types[node.name] = tt

        # All placeholders become func args
        arg_types: list[Attribute] = []
        for p in placeholders:
            tt = node_types.get(p.name)
            if tt is None:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=p.name,
                        level="warning",
                        message=f"No type info for placeholder {p.name}, using f32[1]",
                    )
                )
                tt = TensorType(Float32Type(), [1])
            arg_types.append(tt)

        # Determine return types
        ret_types: list[Attribute] = []
        if output_nodes:
            out_args = output_nodes[0].args[0] if output_nodes[0].args else ()
            if not isinstance(out_args, (tuple, list)):
                out_args = (out_args,)
            for a in out_args:
                if hasattr(a, "name") and a.name in node_types:
                    ret_types.append(node_types[a.name])
        if not ret_types:
            ret_types = [TensorType(Float32Type(), [1])]

        func_type = FunctionType.from_lists(arg_types, ret_types)

        # Build the function body
        block = Block(arg_types=arg_types)
        value_map: dict[str, SSAValue] = {}
        multi_results: dict[str, list[SSAValue]] = {}  # split/unbind: node -> per-output SSA
        for i, p in enumerate(placeholders):
            value_map[p.name] = block.args[i]

        # Track external function declarations for opaque fallback
        extern_funcs: list[FuncOp] = []
        declared_sigs: dict[str, str] = {}
        name_counters: dict[str, int] = {}

        declared_callee_sig: dict[str, str] = {}

        def ensure_external_decl(call: CallOp) -> None:
            callee = call.callee.string_value()
            operand_types = tuple(str(value.type) for value in call.operands)
            result_types = tuple(str(value.type) for value in call.results)
            sig_key = f"{callee}:{operand_types}:{result_types}"
            if sig_key in declared_sigs:
                # Rewrite the call to use the canonical (possibly
                # disambiguated) name for this signature.
                call.properties["callee"] = FlatSymbolRefAttr_ref(declared_sigs[sig_key])
                return
            # If the callee name already exists with a different
            # signature, generate a unique suffixed name.
            chosen_name = callee
            if callee in declared_callee_sig:
                count = name_counters.get(callee, 1)
                while f"{callee}_{count}" in declared_callee_sig:
                    count += 1
                chosen_name = f"{callee}_{count}"
                name_counters[callee] = count + 1
            declared_callee_sig[chosen_name] = sig_key
            declared_sigs[sig_key] = chosen_name
            if chosen_name != callee:
                # Rewrite the existing CallOp to point at the new name.
                call.properties["callee"] = FlatSymbolRefAttr_ref(chosen_name)
            extern_funcs.append(
                FuncOp.external(
                    chosen_name,
                    [value.type for value in call.operands],
                    [value.type for value in call.results],
                )
            )

        # Process call_function nodes
        for node in call_nodes:
            target_str = _canonicalize_fx_target_str(str(node.target))
            result_type = node_types.get(node.name)
            if result_type is None:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=node.name,
                        level="warning",
                        message=f"No type info for {node.name}, skipping",
                    )
                )
                continue

            # REQ-020: ``operator.getitem`` on a tuple-producing aten op
            # (``native_layer_norm`` returns ``(out, mean, rstd)``,
            # ``var_mean`` returns ``(mean, var)``, etc.) shows up as
            # ``getitem(producer_node, idx)`` after FX export. The
            # producer was decomposed to a single-tensor representative
            # (the primary output, index 0), so:
            # - ``getitem(_, 0)`` → resolve to the producer's tensor.
            # - ``getitem(_, k)`` for ``k > 0`` → drop the node entirely
            #   (auxiliary outputs that user code typically doesn't
            #   reference; if anything DOES reference them, it'll fail
            #   loudly downstream — preferable to emitting an opaque
            #   ``<built-in function getitem>`` op no provider can match).
            # Acceptance per REQ-020: ``<built-in function getitem>``
            # never appears in payload.mlir.
            # Multi-output ops (split/unbind): getitem(node, i) -> the i-th real output.
            if (
                target_str == "<built-in function getitem>"
                and len(node.args) >= 2
                and hasattr(node.args[0], "name")
                and node.args[0].name in multi_results
            ):
                outs = multi_results[node.args[0].name]
                idx = node.args[1]
                if isinstance(idx, int) and 0 <= idx < len(outs):
                    value_map[node.name] = outs[idx]
                    continue

            if (
                target_str == "<built-in function getitem>"
                and len(node.args) >= 2
                and hasattr(node.args[0], "name")
                and node.args[0].name in value_map
            ):
                idx = node.args[1]
                if idx == 0:
                    value_map[node.name] = value_map[node.args[0].name]
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="info",
                            message=(
                                f"Resolved getitem({node.args[0].name}, 0) → primary result of decomposed tuple op"
                            ),
                        )
                    )
                else:
                    # Drop the auxiliary getitem; no IR emission, no
                    # value_map entry. Consumers that reference this
                    # node will surface as missing operands downstream.
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="info",
                            message=(
                                f"Dropped getitem({node.args[0].name}, {idx}) — "
                                f"auxiliary output of decomposed tuple op (REQ-020)"
                            ),
                        )
                    )
                continue

            # Resolve operands. Flatten list/tuple args (e.g. cat/stack take a tensor
            # list as their first arg) so the decomposition sees all tensor operands.
            operands: list[SSAValue] = []
            for arg in node.args:
                if hasattr(arg, "name") and arg.name in value_map:
                    operands.append(value_map[arg.name])
                elif isinstance(arg, (list, tuple)):
                    for sub in arg:
                        if hasattr(sub, "name") and sub.name in value_map:
                            operands.append(value_map[sub.name])

            # Try decomposition table first
            decomp_fn = self.dynamic_decompositions.get(target_str, DECOMPOSITION_TABLE.get(target_str))
            if decomp_fn is not None:
                meta = dict(node.meta)
                # Forward FX-level args / kwargs to the decomposition so it
                # can extract scalar properties (group_size, axis, quant_min,
                # quant_max, etc.) that don't show up as SSA operands.
                meta["_fx_args"] = tuple(node.args)
                meta["_fx_kwargs"] = dict(node.kwargs)
                meta["_aten_target"] = target_str  # provenance: source aten op
                meta["_emit_named_ops"] = self.emit_named_ops  # high-level form toggle
                try:
                    result = decomp_fn(operands, meta, node.name)
                except (IndexError, KeyError, TypeError) as decomp_err:
                    # Decomposition failed (e.g. missing operands from scalar constants).
                    # Fall through to opaque fallback instead of crashing.
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="warning",
                            message=f"Decomposition failed for {target_str}: {decomp_err}; falling back to opaque call",
                        )
                    )
                else:
                    # Verify the emitted ops before committing them. A single invalid op
                    # (e.g. a decomposition that hardcoded f32 for a bf16 tensor) would
                    # fail whole-module verification and lose the entire lowering, so on
                    # any verification failure we discard and fall through to the opaque
                    # path (typed from meta, which is always consistent downstream).
                    verify_err = None
                    try:
                        for op in result.ops:
                            # CallOp placeholders are opaque-by-design; their external
                            # decl is added below (ensure_external_decl), so don't verify
                            # them here -- verifying an undeclared callee would wrongly
                            # reject a decomposition that mixes real ops + opaque calls.
                            if isinstance(op, CallOp):
                                continue
                            op.verify()
                    except Exception as verr:  # noqa: BLE001
                        verify_err = verr
                    if verify_err is None:
                        for op in result.ops:
                            if isinstance(op, CallOp):
                                ensure_external_decl(op)
                            _forward_fx_meta(op, meta, result.pattern_hint)
                            block.add_op(op)

                        if result.result is not None:
                            value_map[node.name] = result.result
                        if result.results:
                            multi_results[node.name] = list(result.results)

                        self.decomposed_count += 1
                        hint_suffix = f", hint: {result.pattern_hint}" if result.pattern_hint else ""
                        self.diagnostics.append(
                            ImportDiagnostic(
                                fx_node=node.name,
                                level="info",
                                message=(
                                    f"Decomposed {target_str} -> {len(result.ops)} ops "
                                    f"(regions: {result.region_ids}{hint_suffix})"
                                ),
                            )
                        )
                        continue
                    self.diagnostics.append(
                        ImportDiagnostic(
                            fx_node=node.name,
                            level="warning",
                            message=f"Decomposition for {target_str} produced invalid IR ({verify_err}); opaque fallback",
                        )
                    )

            if not self.allow_opaque_fallback and target_str not in self.explicit_blackboxes:
                self.diagnostics.append(
                    ImportDiagnostic(
                        fx_node=node.name,
                        level="error",
                        message=f"Unsupported without explicit blackbox approval: {target_str}",
                    )
                )
                continue

            # Fallback: opaque func.call
            base_name = target_str.replace(".", "_")
            operand_types = tuple(str(v.type) for v in operands)
            sig_key = f"{base_name}:{operand_types}:{result_type}"

            if sig_key not in declared_sigs:
                count = name_counters.get(base_name, 0)
                unique_name = base_name if count == 0 else f"{base_name}_{count}"
                name_counters[base_name] = count + 1
                declared_sigs[sig_key] = unique_name
                real_operand_types = [v.type for v in operands]
                ext_func = FuncOp.external(unique_name, real_operand_types, [result_type])
                extern_funcs.append(ext_func)

            func_name = declared_sigs[sig_key]
            call_op = CallOp(func_name, operands, [result_type])
            block.add_op(call_op)
            value_map[node.name] = call_op.res[0]

            self.opaque_count += 1
            level = "warning" if target_str in self.explicit_blackboxes else "info"
            self.diagnostics.append(
                ImportDiagnostic(
                    fx_node=node.name,
                    level=level,
                    message=f"Opaque: {target_str} -> func.call @{func_name}",
                )
            )

        # Add return
        ret_values: list[SSAValue] = []
        if output_nodes:
            out_args = output_nodes[0].args[0] if output_nodes[0].args else ()
            if not isinstance(out_args, (tuple, list)):
                out_args = (out_args,)
            for a in out_args:
                if hasattr(a, "name") and a.name in value_map:
                    ret_values.append(value_map[a.name])

        if ret_values:
            block.add_op(ReturnOp(ret_values[0]))

        # Reconcile the func signature with the actual return-value types.
        # The original ret_types snapshot (line ~189) was taken from the
        # FX output-node metadata, which can disagree with what the body
        # actually produces — e.g. HF Llama checkpoints declare a bf16
        # output but the attention math upcasts to f32, leaving the
        # declared func.return type at bf16 and the live SSA value at f32.
        # Without this, xDSL's verifier rejects the module on real-scale
        # transformer captures with: "Expected arguments to have the same
        # types as the function output types".
        if ret_values:
            actual_ret_types: list[Attribute] = [v.type for v in ret_values]
            if actual_ret_types != ret_types:
                func_type = FunctionType.from_lists(arg_types, actual_ret_types)

        region = Region([block])
        main_func = FuncOp("forward", func_type, region)

        all_ops = list(extern_funcs) + [main_func]
        module = ModuleOp(all_ops)

        # REQ-023 (generalised): every ``linalg.transpose`` gets a
        # ``compgen.region_id`` + ``dispatch_id`` so the dispatch
        # graph can resolve consumers' operands. When a transpose
        # with permutation ``[1, 0]`` feeds a ``linalg.matmul``'s
        # B operand, the matmul is also tagged with
        # ``compgen.transposed_b="true"`` so providers can short-
        # circuit by emitting a B^T kernel.
        _annotate_transposes_and_matmuls(module)

        # Taxonomy backfill: a final sweep that stamps m2m.op / m2m.family on every named
        # compute op still missing them (ops emitted by helpers outside any DecompResult --
        # e.g. transpose/reduce built inside matmul/layer-norm helpers -- never pass through
        # _forward_fx_meta). Guarantees the matchable vocabulary covers 100% of named ops.
        for op in module.walk():
            if "m2m.family" in op.attributes:
                continue
            tax = _OP_TYPE_TAXONOMY.get(getattr(op, "name", ""))
            if tax is not None:
                kind, fam = tax
                if "m2m.op" not in op.attributes:
                    op.attributes["m2m.op"] = StringAttr(kind)
                op.attributes["m2m.family"] = StringAttr(fam)

        try:
            module.verify()
        except Exception as e:
            self.diagnostics.append(
                ImportDiagnostic(
                    fx_node="<module>",
                    level="error",
                    message=f"Module verification failed: {e}",
                )
            )

        return module

    def get_ir_text(self, module: ModuleOp) -> str:
        """Get the IR text representation of a module."""
        stream = io.StringIO()
        Printer(stream=stream).print(module)
        return stream.getvalue()


def _annotate_transposes_and_matmuls(module: ModuleOp) -> None:
    """Generalised dispatch-region annotation (REQ-023 + REQ-026).

    Walks the module once and stamps:

    - Every ``linalg.transpose`` with ``compgen.region_id`` +
      ``compgen.dispatch_id`` (idempotent — respects pre-existing
      tags from in-tree decompositions).
    - Every ``linalg.matmul`` whose B operand is a permutation-
      ``[1, 0]`` transpose with ``compgen.transposed_b="true"``.
    - Every ``func.call`` op (the opaque-fallback shape
      ``func.call @aten_relu_default`` for unmapped ATen ops) with
      ``compgen.region_id`` + ``compgen.dispatch_id``. Without this
      tag, codegen-fallback's contract extractor can't surface them
      as kernel boundaries (REQ-026's blocker for any opaque-call op).

    Counters are kept distinct per-op-family so synthesised ids
    don't collide with what decompositions assign.
    """
    from xdsl.dialects.func import CallOp
    from xdsl.dialects.linalg import MatmulOp, TransposeOp

    transpose_counter = 0
    for op in module.walk():
        if isinstance(op, TransposeOp):
            existing_rid = op.attributes.get("m2m.region_id")
            if existing_rid is None:
                rid = f"transpose_{transpose_counter}"
                transpose_counter += 1
                op.attributes["m2m.region_id"] = StringAttr(rid)
            else:
                rid = existing_rid.data if isinstance(existing_rid, StringAttr) else None
            if rid and "m2m.dispatch_id" not in op.attributes:
                op.attributes["m2m.dispatch_id"] = StringAttr(rid)

    # Opaque func.call annotation (REQ-026). Per-callee counters so
    # ids stay readable: ``aten_relu_default_0``, ``aten_add_1``, etc.
    callee_counters: dict[str, int] = {}
    for op in module.walk():
        if not isinstance(op, CallOp):
            continue
        if not (hasattr(op, "callee") and hasattr(op.callee, "string_value")):
            continue
        # External func declarations also surface as CallOps elsewhere
        # but those have no `results`; skip them defensively.
        if not op.results:
            continue
        existing_rid = op.attributes.get("m2m.region_id")
        if existing_rid is None:
            callee = op.callee.string_value()
            stem = callee.lstrip("@") if callee else "call"
            count = callee_counters.get(stem, 0)
            callee_counters[stem] = count + 1
            rid = f"{stem}_{count}"
            op.attributes["m2m.region_id"] = StringAttr(rid)
        else:
            rid = existing_rid.data if isinstance(existing_rid, StringAttr) else None
        if rid and "m2m.dispatch_id" not in op.attributes:
            op.attributes["m2m.dispatch_id"] = StringAttr(rid)

    for op in module.walk():
        if not isinstance(op, MatmulOp):
            continue
        if "m2m.transposed_b" in op.attributes:
            continue  # already tagged by a decomposition
        if len(op.operands) < 2:
            continue
        b_operand = op.operands[1]
        producer = b_operand.owner
        if not isinstance(producer, TransposeOp):
            continue
        # Check the permutation is [1, 0] — the only shape that's
        # safe to advertise as "transposed B" for a 2D matmul.
        perm_attr = producer.permutation
        try:
            perm = list(perm_attr.get_values()) if hasattr(perm_attr, "get_values") else None
        except Exception:
            perm = None
        if perm == [1, 0]:
            op.attributes["m2m.transposed_b"] = StringAttr("true")


def fx_to_xdsl(
    exported_program: Any,
    *,
    allow_opaque_fallback: bool = True,
    explicit_blackboxes: set[str] | None = None,
    dynamic_decompositions: dict[str, DecompFn] | None = None,
) -> tuple[ModuleOp, list[ImportDiagnostic]]:
    """Convenience function: export -> xDSL in one call.

    Returns:
        Tuple of (xDSL ModuleOp, list of diagnostics).
    """
    importer = FXImporter(
        allow_opaque_fallback=allow_opaque_fallback,
        explicit_blackboxes=set(explicit_blackboxes or ()),
        dynamic_decompositions=dict(dynamic_decompositions or {}),
    )
    module = importer.import_graph(exported_program)
    return module, importer.diagnostics


__all__ = ["FXImporter", "ImportDiagnostic", "fx_to_xdsl"]
