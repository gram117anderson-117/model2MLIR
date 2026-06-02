"""ATen to xDSL decomposition table.

Maps PyTorch ATen operator targets to functions that produce real xDSL
linalg/arith/tensor operations. This replaces the opaque ``func.call``
approach with structured IR the agent can reason about.

Each decomposition function takes the FX node's args (as xDSL SSAValues)
and metadata, and returns a list of xDSL Operations to insert into the block.

Ops without decompositions fall back to ``func.call`` (flagged as opaque).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import (
    DenseArrayBase,
    Float32Type,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.dialects.linalg import MatmulOp, TransposeOp
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Operation, SSAValue


def _static_shape(shape_like: Any) -> list[int]:
    """Coerce a sequence of possibly-symbolic dims to xDSL-friendly ints.

    ``torch.export`` may emit ``SymInt`` dims (rendered as ``u6`` etc.)
    when the graph is traced under data-dependent or unbacked shape
    constraints (e.g. SmolVLA's image-tile counts). xDSL's
    :class:`TensorType` only accepts :class:`builtin.int`; a symbolic dim
    would otherwise surface as ``VerifyException: u6 should be of base
    attribute builtin.int``.

    This helper mirrors the more-narrow
    :func:`m2m.ir.import_fx._coerce_static_dim`. Symbolic
    dims become xDSL's dynamic-dim sentinel (``-1``) so import
    completes; downstream passes that need static shapes handle the
    dynamic case through their own paths.
    """
    out: list[int] = []
    for dim in shape_like:
        try:
            out.append(int(dim))
        except Exception:
            out.append(-1)
    return out


@dataclass
class DecompResult:
    """Result of decomposing one FX node into xDSL ops.

    Attributes:
        ops: xDSL operations to insert into the block.
        result: The SSAValue that represents this node's output.
        region_ids: region_id labels attached to linalg ops.
        pattern_hint: Optional canonical pattern name (e.g. "layer_norm",
            "softmax", "rms_norm", "dequantize_per_channel"). The
            ``FXImporter`` propagates this onto every emitted op as the
            ``compgen._pattern_hint`` attribute so Phase 2 passes
            (``raise_special_ops``, ``fuse_dequant_matmul``, etc.) can
            recognize the op's origin without re-detecting.
    """

    ops: list[Operation] = field(default_factory=list)
    result: SSAValue | None = None
    region_ids: list[str] = field(default_factory=list)
    pattern_hint: str | None = None
    # Multi-output ops (split, unbind, ...): per-output SSA values so the importer can
    # resolve getitem(node, i) -> results[i]. Empty for single-output ops.
    results: list[SSAValue] = field(default_factory=list)


# Type for decomposition functions
DecompFn = Callable[
    [
        list[SSAValue],  # positional operands (resolved FX args)
        dict[str, Any],  # FX node metadata (shapes, dtypes)
        str,  # node name (for region_id generation)
    ],
    DecompResult,
]


# ============================================================================
# Counters for unique region IDs
# ============================================================================

_region_counters: dict[str, int] = {}


def _next_region_id(prefix: str) -> str:
    """Generate a unique region ID like 'matmul_0', 'matmul_1'."""
    count = _region_counters.get(prefix, 0)
    _region_counters[prefix] = count + 1
    return f"{prefix}_{count}"


def reset_region_counters() -> None:
    """Reset counters between imports."""
    _region_counters.clear()


def _make_empty(result_type: TensorType) -> EmptyOp:
    """Create a tensor.empty for an output tensor."""
    return EmptyOp([], result_type)


def _attach_region_id(op: Operation, region_id: str) -> None:
    """Attach a compgen.region_id attribute to an operation."""
    op.attributes["m2m.region_id"] = StringAttr(region_id)


def _reassoc(rank: int):
    """Reassociation grouping all ``rank`` dims into one group: [[0, 1, ..., rank-1]]."""
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr

    return ArrayAttr([ArrayAttr([IntegerAttr(j, i64) for j in range(rank)])])


def _emit_reshape(source: SSAValue, out_shape: list[int], elem: Any):
    """Emit a logical reshape via the canonical ``tensor.collapse_shape`` +
    ``tensor.expand_shape`` pair (collapse the source to 1-D, expand to the target) --
    the same form torch-mlir produces, with no shape-tensor / constant noise.

    Covers view/reshape/unsqueeze/squeeze/flatten. Returns ``(ops, result_ssa)``, or
    ``None`` on dynamic/0-rank shapes (caller falls back to an opaque placeholder)."""
    src_type = source.type
    if not isinstance(src_type, TensorType):
        return None
    in_shape = list(src_type.get_shape())
    if any(d < 0 for d in (*in_shape, *out_shape)):
        return None
    if in_shape == out_shape:
        return [], source  # identity reshape

    from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp

    # rank-0 (scalar) endpoints: a full reduction yields a 0-D tensor that a keepdim
    # reshape must turn back into an all-ones shape (and vice-versa). expand/collapse
    # can't bridge rank-0, so reshape a single-element tensor via extract + from_elements.
    if not in_shape or not out_shape:
        from xdsl.dialects.arith import ConstantOp
        from xdsl.dialects.builtin import IndexType, IntegerAttr
        from xdsl.dialects.tensor import ExtractOp, FromElementsOp

        cops: list[Operation] = []
        idxs = []
        for _ in in_shape:  # all source dims are size 1
            c = ConstantOp(IntegerAttr(0, IndexType()), IndexType())
            cops.append(c)
            idxs.append(c.results[0])
        ext = ExtractOp(source, idxs, elem)
        fe = FromElementsOp(ext.results[0], result_type=TensorType(elem, out_shape))
        return [*cops, ext, fe], fe.results[0]

    numel = 1
    for d in in_shape:
        numel *= d

    ops: list[Operation] = []
    cur = source
    if len(in_shape) != 1:  # collapse N-D -> 1-D
        flat = CollapseShapeOp(
            operands=[cur],
            result_types=[TensorType(elem, [numel])],
            properties={"reassociation": _reassoc(len(in_shape))},
        )
        ops.append(flat)
        cur = flat.results[0]
    if len(out_shape) != 1:  # expand 1-D -> target (static output shape, no dynamic dims)
        exp = ExpandShapeOp(
            cur,
            [],  # dynamic_output_shape (none -- fully static)
            _reassoc(len(out_shape)),
            list(out_shape),  # static_output_shape
            TensorType(elem, out_shape),
        )
        ops.append(exp)
        cur = exp.results[0]
    return ops, cur


def _broadcast_map(operand_shape, result_shape):
    """Affine map projecting the iteration space (result dims) onto an operand's dims.

    Handles exact match (identity) and rank-difference broadcasting (a lower-rank operand
    broadcasts over the leading dims, its dims aligned to the trailing result dims -- e.g.
    bias [N] over [M, N]). Returns None for unsupported broadcasts (size-1 within matching
    rank), so the caller falls back to opaque."""
    from xdsl.ir.affine import AffineExpr, AffineMap

    r, s = len(result_shape), len(operand_shape)
    if list(operand_shape) == list(result_shape):
        return AffineMap.identity(r)
    if s > r:
        return None
    # Right-align the operand to the result; leading dims broadcast (dropped), and within
    # the aligned region a size-1 operand dim broadcasts via a constant-0 index.
    offset = r - s
    exprs = []
    for i in range(s):
        od, rd = operand_shape[i], result_shape[offset + i]
        if od == rd:
            exprs.append(AffineExpr.dimension(offset + i))
        elif od == 1:
            exprs.append(AffineExpr.constant(0))
        else:
            return None
    return AffineMap(r, 0, tuple(exprs))


def _cast_scalar_arg(x, dst):
    """Cast a scalar SSA value to ``dst`` element type (for arith dtype promotion)."""
    src = x.type
    if src == dst:
        return [], x
    from xdsl.dialects.arith import ExtFOp, ExtSIOp, ExtUIOp, FPToSIOp, SIToFPOp, TruncFOp, TruncIOp
    from xdsl.dialects.builtin import AnyFloat, IntegerType

    if isinstance(src, AnyFloat) and isinstance(dst, AnyFloat):
        op = TruncFOp(x, dst) if dst.bitwidth < src.bitwidth else ExtFOp(x, dst)
    elif isinstance(src, IntegerType) and isinstance(dst, AnyFloat):
        op = SIToFPOp(x, dst)
    elif isinstance(src, AnyFloat) and isinstance(dst, IntegerType):
        op = FPToSIOp(x, dst)
    elif isinstance(src, IntegerType) and isinstance(dst, IntegerType):
        if dst.width.data < src.width.data:
            op = TruncIOp(x, dst)
        # widening: bool (i1) zero-extends (True->1), signed ints sign-extend
        elif src.width.data == 1:
            op = ExtUIOp(x, dst)
        else:
            op = ExtSIOp(x, dst)
    else:
        return [], x  # can't bridge; leave as-is (importer verify-fallback handles it)
    return [op], op.results[0]


def _cast_tensor(x: SSAValue, shape, dst_elem):
    """Elementwise cast a whole tensor to ``dst_elem`` (family ``cast``). Returns
    ``(ops, result_ssa)`` or ``None`` if shapes are dynamic. A no-op (``([], x)``)
    when the element type already matches."""
    if x.type.element_type == dst_elem:  # type: ignore[union-attr]
        return [], x
    rt = TensorType(dst_elem, list(shape))
    return _elementwise([x], rt, lambda args, oe: _cast_scalar_arg(args[0], oe))


def _elementwise(inputs: list[SSAValue], result_type: TensorType, scalar_build, input_maps=None, promote=False):
    """Emit a ``linalg.generic`` elementwise op (all-parallel).

    ``input_maps`` (one AffineMap per input) enables broadcasting; defaults to identity
    maps (inputs must then match ``result_type``'s shape). ``scalar_build(args, out_elem)
    -> (ops, yield_ssa)`` builds the scalar body. Returns ``(ops, result_ssa)`` or None."""
    shape = result_type.get_shape()
    if any(d < 0 for d in shape):
        return None

    from xdsl.dialects.builtin import AffineMapAttr
    from xdsl.dialects.linalg import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    rank = len(shape)
    out_elem = result_type.element_type
    empty = EmptyOp([], result_type)

    in_elems = [inp.type.element_type for inp in inputs]  # type: ignore[union-attr]
    block = Block(arg_types=[*in_elems, out_elem])
    args = list(block.args[: len(inputs)])
    if promote:  # cast each input element to the result dtype (dtype promotion, e.g. bf16*f32)
        casted = []
        for a in args:
            cops, ca = _cast_scalar_arg(a, out_elem)
            for c in cops:
                block.add_op(c)
            casted.append(ca)
        args = casted
    body_ops, yield_ssa = scalar_build(args, out_elem)
    for op in body_ops:
        block.add_op(op)
    block.add_op(YieldOp(yield_ssa))

    if input_maps is None:
        input_maps = [AffineMap.identity(rank)] * len(inputs)
    indexing_maps = [AffineMapAttr(m) for m in input_maps] + [AffineMapAttr(AffineMap.identity(rank))]
    generic = GenericOp(
        inputs=list(inputs),
        outputs=[empty.results[0]],
        body=Region(block),
        indexing_maps=indexing_maps,
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * rank,
        result_types=[result_type],
    )
    return [empty, generic], generic.results[0]


def _splat_scalar(scalar: Any, result_type: TensorType):
    """Build a full-shape constant tensor by splatting a scalar (for binary-with-scalar)."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType
    from xdsl.dialects.tensor import SplatOp

    elem = result_type.element_type
    if any(d < 0 for d in result_type.get_shape()):
        return None
    if isinstance(elem, IntegerType):
        const = ConstantOp(IntegerAttr(int(scalar), elem), elem)
    else:
        const = ConstantOp(FloatAttr(float(scalar), elem), elem)
    splat = SplatOp(const.result, [], result_type)
    return [const, splat], splat.results[0]


def _cast_scalar_build(target_elem):
    """scalar_build for a dtype cast: float trunc/ext, int<->float, or int<->int
    (bool zero-extends), picked by types -- delegates to ``_cast_scalar_arg``."""

    def build(args, out_elem):
        return _cast_scalar_arg(args[0], target_elem)

    return build


# ============================================================================
# Decomposition functions
# ============================================================================


def decompose_linear(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.linear.default(input, weight, bias?) -> matmul + bias add.

    linear(x, w, b) = x @ w^T + b
    - x: [M, K], w: [N, K] (note: weight is transposed), b: [N]
    - output: [M, N]
    """
    ops: list[Operation] = []
    region_ids: list[str] = []

    x = operands[0]  # input: [M, K]
    w = operands[1]  # weight: [N, K]

    # Get result type from metadata (preserve operand dtype, e.g. bf16)
    val: Any = meta["val"]
    result_type = TensorType(_t_elem(x), _static_shape(val.shape))

    # Step 1: Transpose weight [N, K] -> [K, N]
    w_type = w.type
    assert isinstance(w_type, TensorType)
    w_shape = w_type.get_shape()
    wt_type = TensorType(_t_elem(w), [w_shape[1], w_shape[0]])
    wt_empty = _make_empty(wt_type)
    ops.append(wt_empty)

    perm = DenseArrayBase.from_list(i64, [1, 0])
    transpose = TransposeOp(
        input=w,
        init=wt_empty.results[0],
        permutation=perm,
        result=wt_type,
    )
    # REQ-023: tag the transpose with its own region_id + dispatch_id
    # so the dispatch graph can resolve the matmul's B operand to a
    # producer node. Providers that don't want to materialize the
    # transpose can ignore this region and use the matmul's
    # ``compgen.transposed_b`` flag instead.
    transpose_rid = _next_region_id("transpose")
    _attach_region_id(transpose, transpose_rid)
    transpose.attributes["m2m.dispatch_id"] = StringAttr(transpose_rid)
    region_ids.append(transpose_rid)
    ops.append(transpose)

    # Step 2: Matmul: x [M, K] @ w^T [K, N] -> [M, N]
    mm_empty = _make_empty(result_type)
    ops.append(mm_empty)

    matmul = MatmulOp(
        inputs=[x, transpose.results[0]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    rid = _next_region_id("matmul")
    _attach_region_id(matmul, rid)
    matmul.attributes["m2m.dispatch_id"] = StringAttr(rid)
    # REQ-023: declare that this matmul's B operand is logically a
    # transposed weight. Providers that prefer to short-circuit the
    # transpose op (e.g. emit a B^T kernel kernel directly against
    # the original weight) can read this flag and skip the
    # transpose region.
    matmul.attributes["m2m.transposed_b"] = StringAttr("true")
    region_ids.append(rid)
    ops.append(matmul)

    result = matmul.results[0]

    # Step 3: Bias addition deferred — requires broadcast lowering
    # (bias is [N] but result is [M, N], needs linalg.generic with
    # indexing_maps for proper broadcast semantics)

    return DecompResult(ops=ops, result=result, region_ids=region_ids)


def _gelu_build(args, out_elem):
    """gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))  (matches torch-mlir's exact form)."""
    import math as _pymath

    from xdsl.dialects.arith import AddfOp, ConstantOp, MulfOp
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.math import ErfOp

    x = args[0]
    half = ConstantOp(FloatAttr(0.5, out_elem), out_elem)
    one = ConstantOp(FloatAttr(1.0, out_elem), out_elem)
    inv_sqrt2 = ConstantOp(FloatAttr(1.0 / _pymath.sqrt(2.0), out_elem), out_elem)
    scaled = MulfOp(x, inv_sqrt2.results[0])
    er = ErfOp(scaled.results[0])
    onep = AddfOp(one.results[0], er.results[0])
    hx = MulfOp(half.results[0], x)
    out = MulfOp(hx.results[0], onep.results[0])
    return [half, one, inv_sqrt2, scaled, er, onep, hx, out], out.results[0]


def decompose_gelu(operands, meta, node_name):
    """aten.gelu.default(input) -> 0.5*x*(1+erf(x/sqrt(2))) via linalg.generic."""
    real = _unary_elementwise(operands, meta, "gelu", _gelu_build)
    if real is not None:
        return real
    return _opaque_decomp("aten_gelu", operands[:1], meta, "elementwise", pattern_hint="gelu")


def _coerce_static_dim(d: Any) -> int:
    try:
        return int(d)
    except Exception:
        return -1


def _shape_of(ssa: SSAValue) -> list[int] | None:
    t = ssa.type
    if isinstance(t, TensorType):
        return list(t.get_shape())
    return None


def _t_elem(ssa: SSAValue):
    """Element type of a tensor SSA value (so matmul/transpose preserve dtype rather than
    hardcoding f32). Mismatched matmul operand dtypes are caught by the importer's
    verify-fallback -> opaque."""
    t = ssa.type
    return t.element_type if isinstance(t, TensorType) else Float32Type()


def _binary_elementwise(operands, meta, op_name, scalar_build):
    """Binary elementwise via linalg.generic. Handles (tensor, scalar) by splatting.

    Returns a DecompResult, or None to signal the caller to use the opaque fallback
    (dynamic shapes, or operand shapes that don't already match the result -> broadcast,
    which we don't emit yet)."""
    val: Any = meta["val"]
    out_shape = [_coerce_static_dim(d) for d in val.shape]
    if any(d < 0 for d in out_shape):
        return None
    lhs0_type = operands[0].type if operands else None
    if not isinstance(lhs0_type, TensorType):
        return None
    # Result dtype = meta dtype; operands are cast to it inside the body (promote=True),
    # so mixed-dtype promotion (e.g. bf16 * f32 -> f32) lowers instead of bailing.
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, out_shape)

    pre: list[Operation] = []
    if len(operands) >= 2:
        lhs, rhs = operands[0], operands[1]
        if not isinstance(rhs.type, TensorType):
            return None
    elif len(operands) == 1:
        # One tensor + one scalar. The scalar may be on EITHER side (e.g. the reverse
        # subtraction `511 - arange`), so detect which arg is numeric and splat it on
        # the matching side -- order matters for non-commutative ops (sub/div).
        a0, a1 = _fx_arg(meta, 0, None), _fx_arg(meta, 1, None)
        if isinstance(a0, (int, float)) and not isinstance(a0, bool) and not isinstance(a1, (int, float)):
            scalar, scalar_left = a0, True
        elif isinstance(a1, (int, float)) and not isinstance(a1, bool):
            scalar, scalar_left = a1, False
        else:
            return None
        sp = _splat_scalar(scalar, result_type)
        if sp is None:
            return None
        pre = sp[0]
        if scalar_left:
            lhs, rhs = sp[1], operands[0]
        else:
            lhs, rhs = operands[0], sp[1]
    else:
        return None

    # Broadcasting via affine maps (e.g. bias [N] over [M, N]); unsupported broadcasts
    # (size-1 within matching rank) return None -> opaque fallback.
    ml = _broadcast_map(_shape_of(lhs) or [], out_shape)
    mr = _broadcast_map(_shape_of(rhs) or [], out_shape)
    if ml is None or mr is None:
        return None

    em = _elementwise([lhs, rhs], result_type, scalar_build, input_maps=[ml, mr], promote=True)
    if em is None:
        return None
    ops, res = em
    rid = _next_region_id(op_name)
    for op in (*pre, *ops):
        _attach_region_id(op, rid)
    return DecompResult(ops=[*pre, *ops], result=res, region_ids=[rid], pattern_hint=op_name)


def _unary_elementwise(operands, meta, op_name, scalar_build, out_elem=None):
    """Unary elementwise via linalg.generic. Returns DecompResult or None (opaque).

    Result dtype defaults to the operand's dtype (dtype-preserving ops); pass
    ``out_elem`` for casts (where the output dtype differs from the input)."""
    src_type = operands[0].type if operands else None
    if not isinstance(src_type, TensorType):
        return None
    val: Any = meta["val"]
    out_shape = [_coerce_static_dim(d) for d in val.shape]
    if any(d < 0 for d in out_shape):
        return None
    meta_elem = _element_type_from_meta(meta)
    if out_elem is None:
        # dtype-preserving op: require operand dtype == meta dtype (consistency), else opaque.
        if src_type.element_type != meta_elem:
            return None
        elem = meta_elem
    else:
        # cast: result dtype is the explicit target; input dtype may differ (that's the cast).
        elem = out_elem
    result_type = TensorType(elem, out_shape)
    em = _elementwise([operands[0]], result_type, scalar_build)
    if em is None:
        return None
    ops, res = em
    rid = _next_region_id(op_name)
    for op in ops:
        _attach_region_id(op, rid)
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint=op_name)


def _bin_build(arith_cls):
    def build(args, out_elem):
        op = arith_cls(args[0], args[1])
        return [op], op.results[0]
    return build


def _arith_build2(float_name: str, int_name: str):
    """Binary build that picks the integer or float arith op based on the result dtype
    (so integer index/position math lowers, not just float)."""

    def build(args, out_elem):
        import xdsl.dialects.arith as _A
        from xdsl.dialects.builtin import IntegerType

        cls = getattr(_A, int_name) if isinstance(out_elem, IntegerType) else getattr(_A, float_name)
        op = cls(args[0], args[1])
        return [op], op.results[0]

    return build


def _un_build(op_cls):
    def build(args, out_elem):
        op = op_cls(args[0])
        return [op], op.results[0]
    return build


def _scalar_to_tensor(scalar: Any, like_type: TensorType) -> tuple[list[Operation], SSAValue]:
    """Materialize a Python scalar as a constant tensor matching ``like_type``.

    Returns ``(ops_to_emit, ssa_value_to_use_as_operand)``.

    xDSL's ``DenseIntOrFPElementsAttr.from_list`` packs data through the
    element type's ``pack`` method. Some element types (notably
    ``BFloat16Type`` as of xDSL 0.24) raise ``NotImplementedError`` in
    ``pack`` — SmolVLA's vision tower carries bf16 weights, so we hit
    this on import. The fallback below materialises the constant as f32
    and emits an ``arith.truncf`` / ``arith.extf`` cast when the target
    element type differs, keeping the IR well-typed.
    """
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType

    elem = like_type.element_type
    is_int = isinstance(elem, IntegerType)
    data = [int(scalar)] if is_int else [float(scalar)]
    try:
        attr = DenseIntOrFPElementsAttr.from_list(like_type, data)
        const = ConstantOp(attr, like_type)
        return [const], const.result
    except NotImplementedError:
        pass

    # Pack fallback: build the constant in f32 and cast to the target.
    # Emits an opaque ``func.call @_compgen_cast`` rather than an xDSL
    # arith cast, because the linalg-on-tensor cast path would require
    # shape-aware loop nests that we don't have here. The cast call
    # carries a ``compgen.cast_to`` attribute so downstream passes can
    # lower it alongside the rest of the opaque fallbacks.
    from xdsl.dialects.func import CallOp

    f32_like = TensorType(Float32Type(), like_type.get_shape())
    f32_attr = DenseIntOrFPElementsAttr.from_list(f32_like, [float(scalar)])
    f32_const = ConstantOp(f32_attr, f32_like)
    cast = CallOp("_compgen_cast_scalar", [f32_const.result], [like_type])
    cast.attributes["m2m.cast_to"] = StringAttr(str(elem))
    return [f32_const, cast], cast.res[0]


def _binary_operands(operands: list[SSAValue], meta: dict[str, Any]) -> tuple[list[Operation], SSAValue, SSAValue]:
    """Resolve a binary aten op's two operands, materializing a scalar second
    operand from FX args when only one SSA value was supplied.

    Pre-fix this raised ``IndexError: list index out of range`` for
    ``aten.add.Tensor(tensor, scalar_int)`` because the scalar arrives as
    a Python int via ``meta['_fx_args']`` rather than as an SSA operand.
    """
    if len(operands) >= 2:
        return [], operands[0], operands[1]
    if len(operands) == 1:
        # Scalar second operand — broadcastable constant of result dtype.
        val: Any = meta["val"]
        elem = _element_type_from_meta(meta)
        # 1-element tensor; the elementwise add will broadcast against it.
        like = TensorType(elem, [1])
        scalar = _fx_arg(meta, 1, 0)
        ops, ssa = _scalar_to_tensor(scalar, like)
        return ops, operands[0], ssa
    raise IndexError("binary op with zero SSA operands")


def decompose_add_tensor(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.add.Tensor(a, b) -> element-wise add.

    Handles the (tensor, scalar) form by materialising the scalar as a
    1-element constant tensor of the result dtype.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, [_coerce_static_dim(d) for d in val.shape])

    from xdsl.dialects.arith import AddfOp

    real = _binary_elementwise(operands, meta, "add", _arith_build2("AddfOp", "AddiOp"))
    if real is not None:
        return real

    pre, lhs, rhs = _binary_operands(operands, meta)
    rid = _next_region_id("add")
    call = CallOp("aten_add", [lhs, rhs], [result_type])
    _attach_region_id(call, rid)
    return DecompResult(ops=[*pre, call], result=call.res[0], region_ids=[rid])


def decompose_mul_tensor(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.mul.Tensor(a, b) -> element-wise mul.

    Handles the (tensor, scalar) form like ``decompose_add_tensor``.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, [_coerce_static_dim(d) for d in val.shape])

    from xdsl.dialects.arith import MulfOp

    real = _binary_elementwise(operands, meta, "mul", _arith_build2("MulfOp", "MuliOp"))
    if real is not None:
        return real

    pre, lhs, rhs = _binary_operands(operands, meta)
    rid = _next_region_id("mul")
    call = CallOp("aten_mul", [lhs, rhs], [result_type])
    _attach_region_id(call, rid)
    return DecompResult(ops=[*pre, call], result=call.res[0], region_ids=[rid])


def decompose_mm(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.mm.default(a, b) -> linalg.matmul."""
    val: Any = meta["val"]
    result_type = TensorType(_t_elem(operands[0]), _static_shape(val.shape))

    mm_empty = _make_empty(result_type)
    matmul = MatmulOp(
        inputs=[operands[0], operands[1]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    rid = _next_region_id("matmul")
    _attach_region_id(matmul, rid)

    return DecompResult(ops=[mm_empty, matmul], result=matmul.results[0], region_ids=[rid])


def decompose_transpose(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.t.default(input) -> linalg.transpose."""
    val: Any = meta["val"]
    result_type = TensorType(_t_elem(operands[0]), _static_shape(val.shape))

    t_empty = _make_empty(result_type)
    perm = DenseArrayBase.from_list(i64, [1, 0])
    transpose = TransposeOp(
        input=operands[0],
        init=t_empty.results[0],
        permutation=perm,
        result=result_type,
    )
    rid = _next_region_id("transpose")
    _attach_region_id(transpose, rid)

    return DecompResult(ops=[t_empty, transpose], result=transpose.results[0], region_ids=[rid])


def decompose_permute(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """aten.permute.default(input, dims) -> linalg.transpose for ANY rank."""
    operand_type = operands[0].type
    val: Any = meta["val"]
    result_shape = _static_shape(val.shape)
    meta_elem = _element_type_from_meta(meta)
    dims = _fx_arg(meta, 1, None)
    if (
        isinstance(operand_type, TensorType)
        and operand_type.element_type == meta_elem
        and dims is not None
        and all(d >= 0 for d in result_shape)
    ):
        elem = operand_type.element_type
        result_type = TensorType(elem, result_shape)
        empty = _make_empty(result_type)
        perm = DenseArrayBase.from_list(i64, [int(d) for d in dims])
        transpose = TransposeOp(
            input=operands[0], init=empty.results[0], permutation=perm, result=result_type
        )
        rid = _next_region_id("permute")
        _attach_region_id(transpose, rid)
        return DecompResult(ops=[empty, transpose], result=transpose.results[0], region_ids=[rid], pattern_hint="permute")

    return _opaque_decomp("aten_permute", operands[:1], meta, "layout", pattern_hint="permute")


def decompose_addmm(
    operands: list[SSAValue],
    meta: dict[str, Any],
    node_name: str,
) -> DecompResult:
    """Decompose aten.addmm.default(bias, mat1, mat2, ...) -> matmul + bias add."""

    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    result_type = TensorType(_t_elem(operands[1]), _static_shape(val.shape))
    ops: list[Operation] = []
    region_ids: list[str] = []

    mm_empty = _make_empty(result_type)
    ops.append(mm_empty)

    matmul = MatmulOp(
        inputs=[operands[1], operands[2]],
        outputs=[mm_empty.results[0]],
        res=[result_type],
    )
    matmul_rid = _next_region_id("matmul")
    _attach_region_id(matmul, matmul_rid)
    region_ids.append(matmul_rid)
    ops.append(matmul)

    # Bias add (broadcast): bias [N] over matmul result [M, N] via linalg.generic.
    from xdsl.dialects.arith import AddfOp

    out_shape = _static_shape(val.shape)
    mm_res = matmul.results[0]
    elem = result_type.element_type
    mb = _broadcast_map(_shape_of(operands[0]) or [], out_shape)
    if mb is not None and _t_elem(operands[0]) == elem and not any(d < 0 for d in out_shape):
        em = _elementwise(
            [mm_res, operands[0]], result_type, _bin_build(AddfOp),
            input_maps=[_broadcast_map(out_shape, out_shape), mb],
        )
        if em is not None:
            add_ops, res = em
            add_rid = _next_region_id("add")
            for op in add_ops:
                _attach_region_id(op, add_rid)
            region_ids.append(add_rid)
            ops.extend(add_ops)
            return DecompResult(ops=ops, result=res, region_ids=region_ids, pattern_hint="addmm")

    # Fallback: keep the real matmul, opaque bias add.
    bias_add = CallOp("aten_bias_add", [operands[0], mm_res], [result_type])
    add_rid = _next_region_id("add")
    _attach_region_id(bias_add, add_rid)
    region_ids.append(add_rid)
    ops.append(bias_add)
    return DecompResult(ops=ops, result=bias_add.res[0], region_ids=region_ids)


# ============================================================================
#  expansion — real-model coverage (smolVLA + Gemma-decode)
# ============================================================================
# Each entry below follows the established MVP pattern:
# - emit a real linalg op where cleanly supported (bmm, convolution as GEMM)
# - otherwise emit an opaque func.call (same pattern as decompose_gelu today)
# - set ``pattern_hint`` so downstream Phase 2 passes can reason about intent
#   even when the body is a black box.
# Destructive lowerings into full linalg.generic bodies land in a follow-up
# wave alongside the MVP-annotator → real-rewrite upgrade.


def _opaque_decomp(
    op_name: str,
    operands: list[SSAValue],
    meta: dict[str, Any],
    region_prefix: str,
    *,
    pattern_hint: str | None = None,
) -> DecompResult:
    """Shared helper for MVP decompositions that lower to a typed
    ``func.call @op_name`` carrying a ``compgen.region_id`` and
    optional pattern hint. Used for every op below whose full linalg
    body is deferred to the destructive-rewrite wave.
    """
    from xdsl.dialects.func import CallOp

    val: Any = meta["val"]
    # Tuple-returning ops (``native_layer_norm``, ``var_mean``, …) put a
    # tuple in ``meta['val']``. Surface only the primary tensor (index 0)
    # since that's what downstream ``getitem(_, 0)`` consumers care
    # about; auxiliary outputs (mean / rstd / etc.) are folded away.
    if isinstance(val, (tuple, list)) and val:
        val = val[0]
    result_type = TensorType(_t_elem(operands[1]), _static_shape(val.shape))
    rid = _next_region_id(region_prefix)
    call = CallOp(op_name, operands, [result_type])
    _attach_region_id(call, rid)
    return DecompResult(
        ops=[call],
        result=call.res[0],
        region_ids=[rid],
        pattern_hint=pattern_hint,
    )


def decompose_bmm(operands, meta, node_name):
    """aten.bmm.default(a[B,M,K], b[B,K,N]) -> batch matmul via linalg.generic (contraction)."""
    if len(operands) < 2:
        return _opaque_decomp("aten_bmm", operands, meta, "batch_matmul", pattern_hint="batch_matmul")
    a, b = operands[0], operands[1]
    sa, sb = _shape_of(a), _shape_of(b)
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    elem = _t_elem(a)
    if (sa is None or sb is None or len(sa) != 3 or len(sb) != 3 or len(out_shape) != 3
            or any(d < 0 for d in [*sa, *sb, *out_shape]) or _t_elem(b) != elem):
        return _opaque_decomp("aten_bmm", operands, meta, "batch_matmul", pattern_hint="batch_matmul")

    from xdsl.dialects.arith import AddfOp, ConstantOp, MulfOp
    from xdsl.dialects.builtin import AffineMapAttr, FloatAttr
    from xdsl.dialects.linalg import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    result_type = TensorType(elem, out_shape)
    zero = ConstantOp(FloatAttr(0.0, elem), elem)
    init = SplatOp(zero.result, [], result_type)
    D = AffineExpr.dimension  # iteration dims: (b=0, m=1, n=2, k=3)
    a_map = AffineMap(4, 0, (D(0), D(1), D(3)))
    b_map = AffineMap(4, 0, (D(0), D(3), D(2)))
    o_map = AffineMap(4, 0, (D(0), D(1), D(2)))
    blk = Block(arg_types=[elem, elem, elem])
    prod = MulfOp(blk.args[0], blk.args[1])
    acc = AddfOp(blk.args[2], prod.results[0])
    blk.add_op(prod)
    blk.add_op(acc)
    blk.add_op(YieldOp(acc.results[0]))
    par, red = IteratorType.PARALLEL, IteratorType.REDUCTION
    gen = GenericOp(
        inputs=[a, b],
        outputs=[init.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(a_map), AffineMapAttr(b_map), AffineMapAttr(o_map)],
        iterator_types=[IteratorTypeAttr(par), IteratorTypeAttr(par), IteratorTypeAttr(par), IteratorTypeAttr(red)],
        result_types=[result_type],
    )
    rid = _next_region_id("matmul")
    for op in (zero, init, gen):
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("matmul")
    return DecompResult(ops=[zero, init, gen], result=gen.results[0], region_ids=[rid], pattern_hint="batch_matmul")


def decompose_int_mm(operands, meta, node_name):
    """aten._int_mm(a:i8[M,K], b:i8[K,N]) -> i32[M,N] — the TRUE quantized matmul.

    This is the non-QDQ integer GEMM: sign-extend i8 operands to i32 and accumulate
    in i32 (linalg.generic contraction, family ``matmul``). No dequantize roundtrip --
    the low-precision op is preserved for a target that supports int8 matmul natively."""
    if len(operands) < 2:
        return _opaque_decomp("aten__int_mm", operands, meta, "matmul", pattern_hint="int_matmul")
    a, b = operands[0], operands[1]
    sa, sb = _shape_of(a), _shape_of(b)
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    out_elem = _element_type_from_meta(meta)  # i32
    if sa is None or sb is None or len(sa) != 2 or len(sb) != 2 or len(out_shape) != 2 \
            or any(d < 0 for d in [*sa, *sb, *out_shape]):
        return _opaque_decomp("aten__int_mm", operands, meta, "matmul", pattern_hint="int_matmul")

    from xdsl.dialects.arith import AddiOp, ConstantOp, ExtSIOp, MuliOp
    from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, IntegerType
    from xdsl.dialects.linalg import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    if not isinstance(out_elem, IntegerType):
        out_elem = IntegerType(32)
    a_elem, b_elem = a.type.element_type, b.type.element_type
    result_type = TensorType(out_elem, out_shape)
    zero = ConstantOp(IntegerAttr(0, out_elem), out_elem)
    init = SplatOp(zero.result, [], result_type)
    D = AffineExpr.dimension  # (m=0, n=1, k=2)
    a_map = AffineMap(3, 0, (D(0), D(2)))
    b_map = AffineMap(3, 0, (D(2), D(1)))
    o_map = AffineMap(3, 0, (D(0), D(1)))
    blk = Block(arg_types=[a_elem, b_elem, out_elem])
    ea = ExtSIOp(blk.args[0], out_elem) if a_elem != out_elem else None
    eb = ExtSIOp(blk.args[1], out_elem) if b_elem != out_elem else None
    av = ea.results[0] if ea else blk.args[0]
    bv = eb.results[0] if eb else blk.args[1]
    prod = MuliOp(av, bv)
    acc = AddiOp(blk.args[2], prod.results[0])
    for op in (ea, eb, prod, acc):
        if op is not None:
            blk.add_op(op)
    blk.add_op(YieldOp(acc.results[0]))
    par, red = IteratorType.PARALLEL, IteratorType.REDUCTION
    gen = GenericOp(
        inputs=[a, b],
        outputs=[init.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(a_map), AffineMapAttr(b_map), AffineMapAttr(o_map)],
        iterator_types=[IteratorTypeAttr(par), IteratorTypeAttr(par), IteratorTypeAttr(red)],
        result_types=[result_type],
    )
    rid = _next_region_id("matmul")
    for op in (zero, init, gen):
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("matmul")
    return DecompResult(ops=[zero, init, gen], result=gen.results[0], region_ids=[rid], pattern_hint="int_matmul")


def _make_amin_amax(is_min):
    """aten.amin/amax(input, dim, keepdim) — min/max reduction over dims (family reduce)."""
    name = "aten_amin" if is_min else "aten_amax"

    def decompose(operands, meta, node_name):
        from xdsl.dialects.arith import MaxSIOp, MinSIOp
        from xdsl.dialects.builtin import IntegerType

        x = operands[0]
        in_shape = _shape_of(x)
        if in_shape is None or any(d < 0 for d in in_shape):
            return _opaque_decomp(name, operands[:1], meta, "reduce", pattern_hint="reduce")
        elem = _t_elem(x)
        val: Any = meta["val"]
        out_shape = _static_shape(getattr(val, "shape", []))
        dims = _fx_arg(meta, 1, None)
        if dims is None:
            dims = list(range(len(in_shape)))
        elif isinstance(dims, int):
            dims = [dims]
        else:
            dims = list(dims)
        dims = [d % len(in_shape) for d in dims]
        is_int = isinstance(elem, IntegerType)
        if is_int:
            bits = elem.width.data
            ident = ((1 << (bits - 1)) - 1) if is_min else -(1 << (bits - 1))
            combine = MinSIOp if is_min else MaxSIOp
        else:
            from xdsl.dialects.arith import MaximumfOp, MinimumfOp
            ident = float("inf") if is_min else float("-inf")
            combine = MinimumfOp if is_min else MaximumfOp
        ops, red, rsh = _reduce(x, in_shape, dims, ident, combine, elem)
        res = red
        if rsh != out_shape:
            res = _keepdim_reshape(ops, res, rsh, out_shape, elem)
            if res is None:
                return _opaque_decomp(name, operands[:1], meta, "reduce", pattern_hint="reduce")
        rid = _next_region_id("reduce")
        for op in ops:
            _attach_region_id(op, rid)
            op.attributes["m2m.family"] = StringAttr("reduce")
        return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="reduce")

    return decompose


def decompose_round(operands, meta, node_name):
    """aten.round.default — round half-to-even (math.roundeven), family elementwise."""
    from xdsl.dialects.math import RoundEvenOp

    real = _unary_elementwise(operands, meta, "round", _un_build(RoundEvenOp))
    if real is not None:
        return real
    return _opaque_decomp("aten_round", operands[:1], meta, "elementwise", pattern_hint="round")


def build_layer_norm_body(x, weight, bias, *, eps, k):
    """Pure layer-norm lowering (mean/var/normalize + optional affine) on SSA tensors.

    Single source of truth for layer_norm: called by ``decompose_native_layer_norm`` (the
    standard path) and by the high-level expansion pass. ``weight``/``bias`` may be None;
    ``k`` is the number of trailing normalized dims. Returns ``(ops, result_ssa)`` or None."""
    from xdsl.dialects.arith import AddfOp, ConstantOp, DivfOp, MulfOp, SubfOp
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.math import RsqrtOp

    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return None
    elem = _t_elem(x)
    rank = len(in_shape)
    dims = list(range(rank - k, rank))
    keep = list(in_shape)
    for d in dims:
        keep[d] = 1
    rt = TensorType(elem, in_shape)
    id_map = _broadcast_map(in_shape, in_shape)
    keep_map = _broadcast_map(keep, in_shape)
    count = 1
    for d in dims:
        count *= in_shape[d]

    ops: list[Operation] = []

    def reduce_mean(src):
        o, s, rsh = _reduce(src, in_shape, dims, 0.0, AddfOp, elem)
        ops.extend(o)
        sp = _splat_scalar(float(count), TensorType(elem, rsh))
        if sp is None:
            return None
        ops.extend(sp[0])
        dv = _elementwise([s, sp[1]], TensorType(elem, rsh), _bin_build(DivfOp))
        if dv is None:
            return None
        ops.extend(dv[0])
        return _keepdim_reshape(ops, dv[1], rsh, keep, elem)

    mean = reduce_mean(x)
    if mean is None:
        return None
    cen = _elementwise([x, mean], rt, _bin_build(SubfOp), input_maps=[id_map, keep_map])
    ops += cen[0]
    sq = _elementwise([cen[1], cen[1]], rt, _bin_build(MulfOp))
    ops += sq[0]
    var = reduce_mean(sq[1])
    if var is None:
        return None

    def add_eps_rsqrt(v):
        c = ConstantOp(FloatAttr(float(eps), elem), elem)
        ce = SplatOp_like(c, keep, elem, ops)
        ve = _elementwise([v, ce], TensorType(elem, keep), _bin_build(AddfOp))
        ops.extend(ve[0])
        rs = _elementwise([ve[1]], TensorType(elem, keep), _un_build(RsqrtOp))
        ops.extend(rs[0])
        return rs[1]

    rstd = add_eps_rsqrt(var)
    normed = _elementwise([cen[1], rstd], rt, _bin_build(MulfOp), input_maps=[id_map, keep_map])
    ops += normed[0]
    res = normed[1]

    # optional affine: out = normed * weight + bias (weight/bias broadcast over norm dims)
    if weight is not None and isinstance(weight.type, TensorType):
        wmap = _broadcast_map(_shape_of(weight) or [], in_shape)
        if wmap is not None:
            sc = _elementwise([res, weight], rt, _bin_build(MulfOp), input_maps=[id_map, wmap])
            if sc is not None:
                ops += sc[0]
                res = sc[1]
    if bias is not None and isinstance(bias.type, TensorType):
        bmap = _broadcast_map(_shape_of(bias) or [], in_shape)
        if bmap is not None:
            sh = _elementwise([res, bias], rt, _bin_build(AddfOp), input_maps=[id_map, bmap])
            if sh is not None:
                ops += sh[0]
                res = sh[1]
    return ops, res


def decompose_native_layer_norm(operands, meta, node_name):
    """aten.native_layer_norm.default(input, normalized_shape, weight, bias, eps).

    Decomposes to mean/var/normalize/scale/shift (family: layer_norm). Returns the
    normalized output as the primary result (getitem(_,0)); mean/rstd aux outputs are
    folded away by the importer. Thin adapter over ``build_layer_norm_body``."""
    norm_shape = _fx_arg(meta, 1, None)
    k = len(norm_shape) if norm_shape is not None else 1
    eps = _fx_arg(meta, 4, 1e-5)
    weight = operands[1] if len(operands) >= 2 else None
    bias = operands[2] if len(operands) >= 3 else None
    built = build_layer_norm_body(operands[0], weight, bias, eps=eps, k=k)
    if built is None:
        return _opaque_decomp("aten_native_layer_norm", operands, meta, "layer_norm", pattern_hint="layer_norm")
    ops, res = built
    rid = _next_region_id("layer_norm")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("layer_norm")
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="layer_norm")


def SplatOp_like(const_op, shape, elem, ops):
    """Splat a freshly-built scalar ConstantOp to a tensor of ``shape``; appends ops."""
    from xdsl.dialects.tensor import SplatOp

    ops.append(const_op)
    sp = SplatOp(const_op.result, [], TensorType(elem, shape))
    ops.append(sp)
    return sp.results[0]


def _reduce(input_ssa, in_shape, dims, identity, combine_cls, elem):
    """linalg.reduce over ``dims`` (rank drops). Returns (ops, result_ssa, reduced_shape).
    ``identity``/``combine_cls`` define the monoid (0.0/AddfOp=sum, -inf/MaximumfOp=max)."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import DenseArrayBase, FloatAttr
    from xdsl.dialects.linalg import ReduceOp, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region

    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    dims = sorted(d % len(in_shape) for d in dims)
    reduced_shape = [s for i, s in enumerate(in_shape) if i not in dims]
    if isinstance(elem, IntegerType):
        c0 = ConstantOp(IntegerAttr(int(identity), elem), elem)
    else:
        c0 = ConstantOp(FloatAttr(float(identity), elem), elem)
    init = SplatOp(c0.result, [], TensorType(elem, reduced_shape))
    blk = Block(arg_types=[elem, elem])
    comb = combine_cls(blk.args[0], blk.args[1])
    blk.add_op(comb)
    blk.add_op(YieldOp(comb.results[0]))
    red = ReduceOp(input_ssa, init.results[0], DenseArrayBase.from_list(i64, dims), Region(blk))
    return [c0, init, red], red.results[0], reduced_shape


def _keepdim_reshape(ops, ssa, reduced_shape, keep_shape, elem):
    """Reshape a reduced tensor back to its keepdim shape (size-1 in reduced dims)."""
    if reduced_shape == keep_shape:
        return ssa
    re = _emit_reshape(ssa, keep_shape, elem)
    if re is None:
        return None
    ops.extend(re[0])
    return re[1]


def build_softmax_body(x, *, dim):
    """Pure softmax lowering (max/sub/exp/sum/div over ``dim``) on an SSA tensor.

    The single source of truth for softmax: called both by ``decompose_softmax`` (the
    importer/standard path) and by the high-level expansion pass (linalg_ext.softmax ->
    standard). Returns ``(ops, result_ssa)`` or ``None`` if shapes are dynamic."""
    from xdsl.dialects.arith import AddfOp, DivfOp, MaximumfOp, SubfOp
    from xdsl.dialects.math import ExpOp

    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return None
    elem = _t_elem(x)
    dim = int(dim) % len(in_shape)
    keep = list(in_shape)
    keep[dim] = 1
    rt = TensorType(elem, in_shape)
    id_map = _broadcast_map(in_shape, in_shape)
    keep_map = _broadcast_map(keep, in_shape)

    ops, mx, rsh = _reduce(x, in_shape, [dim], float("-inf"), MaximumfOp, elem)
    mx = _keepdim_reshape(ops, mx, rsh, keep, elem)
    sub = _elementwise([x, mx], rt, _bin_build(SubfOp), input_maps=[id_map, keep_map]) if mx is not None else None
    if sub is None:
        return None
    ops += sub[0]
    ex = _elementwise([sub[1]], rt, _un_build(ExpOp))
    ops += ex[0]
    o2, s, rsh2 = _reduce(ex[1], in_shape, [dim], 0.0, AddfOp, elem)
    ops += o2
    s = _keepdim_reshape(ops, s, rsh2, keep, elem)
    div = _elementwise([ex[1], s], rt, _bin_build(DivfOp), input_maps=[id_map, keep_map]) if s is not None else None
    if div is None:
        return None
    ops += div[0]
    return ops, div[1]


def decompose_softmax(operands, meta, node_name):
    """aten._softmax(input, dim, _) -> max/sub/exp/sum/div (family: softmax)."""
    built = build_softmax_body(operands[0], dim=_fx_arg(meta, 1, -1))
    if built is None:
        return _opaque_decomp("aten_softmax", operands, meta, "softmax", pattern_hint="softmax")
    ops, res = built
    rid = _next_region_id("softmax")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("softmax")
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="softmax")


def decompose_rsqrt(operands, meta, node_name):
    """aten.rsqrt.default(input) -> math.rsqrt via linalg.generic."""
    from xdsl.dialects.math import RsqrtOp

    real = _unary_elementwise(operands, meta, "rsqrt", _un_build(RsqrtOp))
    if real is not None:
        return real
    return _opaque_decomp("aten_rsqrt", operands, meta, "rsqrt", pattern_hint="rsqrt")


def decompose_pow_tensor_scalar(operands, meta, node_name):
    """aten.pow.Tensor_Scalar(input, exponent) -> math.powf via linalg.generic."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.math import PowFOp

    exp = _fx_arg(meta, 1, 2.0)

    def build(args, oe):
        c = ConstantOp(FloatAttr(float(exp), oe), oe)
        p = PowFOp(args[0], c.results[0])
        return [c, p], p.results[0]

    real = _pointwise(operands[:1], meta, build, family="pow", promote=True)
    if real is not None:
        return real
    return _opaque_decomp("aten_pow", operands, meta, "pow", pattern_hint="pow_tensor_scalar")


def _identity_decomp(operands, meta, node_name):
    """alias/detach/lift_fresh_copy -> identity: forward the operand SSA (no op)."""
    if operands:
        return DecompResult(ops=[], result=operands[0], pattern_hint="identity")
    return _opaque_decomp("aten_identity", operands[:1], meta, "identity", pattern_hint="identity")


def decompose_copy(operands, meta, node_name):
    """aten.copy.default(self, src) -> broadcast src to self's shape (family: copy)."""
    if len(operands) >= 2:
        src = operands[1]
        real = _pointwise([src], meta, lambda args, oe: ([], args[0]), family="copy")
        if real is not None:
            return real
    return _opaque_decomp("aten_copy", operands[:1], meta, "copy", pattern_hint="copy")


def decompose_pow_scalar(operands, meta, node_name):
    """aten.pow.Scalar(base_scalar, exponent_tensor) -> base ** x via math.powf."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.math import PowFOp

    base = _fx_arg(meta, 0, 2.0)

    def build(args, oe):
        c = ConstantOp(FloatAttr(float(base), oe), oe)
        p = PowFOp(c.results[0], args[0])
        return [c, p], p.results[0]

    real = _pointwise(operands[:1], meta, build, family="pow")
    if real is not None:
        return real
    return _opaque_decomp("aten_pow", operands[:1], meta, "pow", pattern_hint="pow")


def decompose_slice_scatter(operands, meta, node_name):
    """aten.slice_scatter(input, src, dim, start, end, step) -> tensor.insert_slice."""
    if len(operands) < 2 or not isinstance(operands[0].type, TensorType) or not isinstance(operands[1].type, TensorType):
        return _opaque_decomp("aten_slice_scatter", operands[:1], meta, "layout", pattern_hint="slice_scatter")
    inp, src = operands[0], operands[1]
    si = list(inp.type.get_shape())
    ss = list(src.type.get_shape())
    if any(d < 0 for d in si) or any(d < 0 for d in ss) or len(ss) != len(si):
        return _opaque_decomp("aten_slice_scatter", operands[:1], meta, "layout", pattern_hint="slice_scatter")
    rank = len(si)
    dim = int(_fx_arg(meta, 2, 0) or 0) % rank
    start = _fx_arg(meta, 3, 0)
    start = 0 if start is None else int(start)
    if start < 0:
        start += si[dim]
    step = _fx_arg(meta, 4, 1)
    step = 1 if step is None else max(1, int(step))
    offsets = [0] * rank
    offsets[dim] = max(0, min(start, si[dim]))
    strides = [1] * rank
    strides[dim] = step

    from xdsl.dialects.tensor import InsertSliceOp

    op = InsertSliceOp.from_static_parameters(src, inp, offsets, ss, strides)
    rid = _next_region_id("slice_scatter")
    _attach_region_id(op, rid)
    op.attributes["m2m.family"] = StringAttr("slice")
    return DecompResult(ops=[op], result=op.results[0], region_ids=[rid], pattern_hint="slice_scatter")


def decompose_any_real(operands, meta, node_name):
    """aten.any.dim(input, dim, keepdim) -> linalg.reduce with arith.ori (family reduce)."""
    from xdsl.dialects.arith import ConstantOp, OrIOp
    from xdsl.dialects.builtin import DenseArrayBase, IntegerAttr, IntegerType
    from xdsl.dialects.linalg import ReduceOp, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region

    x = operands[0]
    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_any", operands[:1], meta, "bool_reduce", pattern_hint="any")
    i1 = IntegerType(1)
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    dim = _fx_arg(meta, 1, None)
    if dim is None:
        dims = list(range(len(in_shape)))
    elif isinstance(dim, int):
        dims = [dim % len(in_shape)]
    else:
        dims = [d % len(in_shape) for d in dim]
    dims = sorted(set(dims))
    reduced = [s for i, s in enumerate(in_shape) if i not in dims]
    c0 = ConstantOp(IntegerAttr(0, i1), i1)
    init = SplatOp(c0.result, [], TensorType(i1, reduced))
    blk = Block(arg_types=[i1, i1])
    comb = OrIOp(blk.args[0], blk.args[1])
    blk.add_op(comb)
    blk.add_op(YieldOp(comb.results[0]))
    red = ReduceOp(x, init.results[0], DenseArrayBase.from_list(i64, dims), Region(blk))
    ops = [c0, init, red]
    res = red.results[0]
    if reduced != out_shape:
        res = _keepdim_reshape(ops, res, reduced, out_shape, i1)
        if res is None:
            return _opaque_decomp("aten_any", operands[:1], meta, "bool_reduce", pattern_hint="any")
    rid = _next_region_id("reduce")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("reduce")
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="any")


def decompose_pow_tensor_tensor(operands, meta, node_name):
    """aten.pow.Tensor_Tensor(a, b) -> math.powf via linalg.generic (family: pow)."""
    from xdsl.dialects.math import PowFOp

    def build(args, oe):
        p = PowFOp(args[0], args[1])
        return [p], p.results[0]

    real = _pointwise(operands[:2], meta, build, family="pow", promote=True)
    if real is not None:
        return real
    return _opaque_decomp("aten_pow", operands[:2], meta, "pow", pattern_hint="pow")


def decompose_empty(operands, meta, node_name):
    """aten.empty.memory_format / empty_like / new_empty -> tensor.empty (uninitialized)."""
    val: Any = meta["val"]
    if isinstance(val, (tuple, list)) and val:
        val = val[0]
    elem = _element_type_from_meta(meta)
    out_shape = _static_shape(getattr(val, "shape", []))
    if any(d < 0 for d in out_shape):
        return _opaque_decomp("aten_empty", [], meta, "empty", pattern_hint="empty")
    e = _make_empty(TensorType(elem, out_shape))
    rid = _next_region_id("empty")
    e.attributes["m2m.family"] = StringAttr("empty")
    _attach_region_id(e, rid)
    return DecompResult(ops=[e], result=e.results[0], region_ids=[rid], pattern_hint="empty")


def _make_fill(value_idx: int):
    """aten.{scalar_tensor,full,full_like} -> tensor.splat(const) (family: fill)."""

    def f(operands, meta, node_name):
        val: Any = meta["val"]
        if isinstance(val, (tuple, list)) and val:
            val = val[0]
        elem = _element_type_from_meta(meta)
        out_shape = _static_shape(getattr(val, "shape", []))
        if any(d < 0 for d in out_shape):
            return _opaque_decomp("aten_fill", [], meta, "fill", pattern_hint="fill")
        sp = _splat_scalar(_fx_arg(meta, value_idx, 0.0), TensorType(elem, out_shape))
        if sp is None:
            return _opaque_decomp("aten_fill", [], meta, "fill", pattern_hint="fill")
        ops, res = sp
        rid = _next_region_id("fill")
        for op in ops:
            _attach_region_id(op, rid)
            op.attributes["m2m.family"] = StringAttr("fill")
        return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="fill")

    return f


def decompose_mean_dim(operands, meta, node_name):
    """aten.mean.dim(input, dims, keepdim?) -> sum-reduce / count (family: reduce)."""
    from xdsl.dialects.arith import AddfOp, DivfOp

    x = operands[0]
    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_mean_dim", operands, meta, "reduce", pattern_hint="reduce_mean")
    elem = _t_elem(x)
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    dims = _fx_arg(meta, 1, None)
    if dims is None:
        dims = list(range(len(in_shape)))
    elif isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)
    dims = [d % len(in_shape) for d in dims]

    ops, summ, rsh = _reduce(x, in_shape, dims, 0.0, AddfOp, elem)
    count = 1
    for d in dims:
        count *= in_shape[d]
    rt = TensorType(elem, rsh)
    sp = _splat_scalar(float(count), rt)
    if sp is None:
        return _opaque_decomp("aten_mean_dim", operands, meta, "reduce", pattern_hint="reduce_mean")
    ops += sp[0]
    dv = _elementwise([summ, sp[1]], rt, _bin_build(DivfOp))
    if dv is None:
        return _opaque_decomp("aten_mean_dim", operands, meta, "reduce", pattern_hint="reduce_mean")
    ops += dv[0]
    res = dv[1]
    if rsh != out_shape:  # keepdim -> reshape to insert size-1 dims
        res = _keepdim_reshape(ops, res, rsh, out_shape, elem)
        if res is None:
            return _opaque_decomp("aten_mean_dim", operands, meta, "reduce", pattern_hint="reduce_mean")
    rid = _next_region_id("reduce")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("reduce")
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="reduce_mean")


def _try_direct_conv2d(operands, meta, in_shape, w_shape):
    """2-D conv as a single linalg.generic contraction (groups=1, no padding, dilation 1).

    out[n,f,oh,ow] = sum_{ci,kh,kw} in[n,ci, oh*sh+kh, ow*sw+kw] * w[f,ci,kh,kw] (+ bias).
    Returns a DecompResult or None (caller -> im2col path). Verify-fallback covers mistakes.
    """
    val: Any = meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return None
    out_shape = _static_shape(val.shape)
    if len(out_shape) != 4 or any(d < 0 for d in out_shape):
        return None
    stride = _fx_arg(meta, 3, [1, 1]) or [1, 1]
    padding = _fx_arg(meta, 4, [0, 0]) or [0, 0]
    dilation = _fx_arg(meta, 5, [1, 1]) or [1, 1]
    transposed = _fx_arg(meta, 6, False)
    groups = _fx_arg(meta, 8, 1)
    if transposed or int(groups or 1) != 1:
        return None
    if any(int(p) != 0 for p in padding) or any(int(d) != 1 for d in dilation):
        return None
    sh, sw = (int(stride[0]), int(stride[1])) if isinstance(stride, (list, tuple)) else (int(stride), int(stride))
    elem = _t_elem(operands[0])
    if _t_elem(operands[1]) != elem:
        return None

    from xdsl.dialects.arith import AddfOp, ConstantOp, MulfOp
    from xdsl.dialects.builtin import AffineMapAttr, FloatAttr
    from xdsl.dialects.linalg import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    inp, w = operands[0], operands[1]
    result_type = TensorType(elem, out_shape)
    zero = ConstantOp(FloatAttr(0.0, elem), elem)
    init = SplatOp(zero.result, [], result_type)
    D = AffineExpr.dimension  # dims: n=0,f=1,oh=2,ow=3,ci=4,kh=5,kw=6
    in_map = AffineMap(7, 0, (D(0), D(4), D(2) * sh + D(5), D(3) * sw + D(6)))
    w_map = AffineMap(7, 0, (D(1), D(4), D(5), D(6)))
    out_map = AffineMap(7, 0, (D(0), D(1), D(2), D(3)))
    blk = Block(arg_types=[elem, elem, elem])
    prod = MulfOp(blk.args[0], blk.args[1])
    acc = AddfOp(blk.args[2], prod.results[0])
    blk.add_op(prod)
    blk.add_op(acc)
    blk.add_op(YieldOp(acc.results[0]))
    par, red = IteratorType.PARALLEL, IteratorType.REDUCTION
    gen = GenericOp(
        inputs=[inp, w],
        outputs=[init.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(in_map), AffineMapAttr(w_map), AffineMapAttr(out_map)],
        iterator_types=[IteratorTypeAttr(par)] * 4 + [IteratorTypeAttr(red)] * 3,
        result_types=[result_type],
    )
    ops: list[Operation] = [zero, init, gen]
    res = gen.results[0]
    rid = _next_region_id("conv")
    # optional bias [F] over [N,F,Ho,Wo]
    if len(operands) >= 3 and isinstance(operands[2].type, TensorType):
        from xdsl.ir.affine import AffineExpr as _AE
        from xdsl.ir.affine import AffineMap as _AM

        bias = operands[2]
        bias_map = _AM(4, 0, (_AE.dimension(1),))
        em = _elementwise(
            [res, bias], result_type, _bin_build(AddfOp),
            input_maps=[_broadcast_map(out_shape, out_shape), bias_map], promote=True,
        )
        if em is not None:
            ops += em[0]
            res = em[1]
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("conv")
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="conv2d")


def decompose_convolution(operands, meta, node_name):
    """aten.convolution.default → im2col + linalg.matmul + reshape (REQ-021).

    Most SIMT targets ship a matmul provider but no conv provider.
    Decomposing here means every such target gets conv "for free" via
    its existing matmul kernel, with no per-pack convolution lowering.

    Shape contract:

    - ``input``: ``(N, C, H, W)``
    - ``weight``: ``(F, C, kH, kW)``
    - ``output``: ``(N, F, H', W')`` — read straight off ``meta['val'].shape``.

    Decomposition (im2col + matmul):

    - ``im2col``  ``(N, C, H, W)`` → ``(K, N*H'*W')`` where ``K = C*kH*kW``
    - ``matmul``  ``W_flat (F, K) @ im2col (K, N*H'*W')`` → ``(F, N*H'*W')``
    - ``reshape`` ``(F, N*H'*W')`` → ``(N, F, H', W')``

    The im2col + reshape steps are opaque ``func.call``s with their own
    ``region_id``s — providers can claim them or skip them (the pack
    composer falls back to its own im2col helper when no provider
    matches). The middle matmul is a real ``linalg.matmul`` with
    ``compgen.region_id`` so any matmul provider claims it.

    For unusual conv shapes (no static dimensions / non-MVP groups /
    transposed conv), this falls back to the prior opaque-conv path.
    """
    from xdsl.dialects.func import CallOp

    # Need at least input + weight; bias is optional.
    if len(operands) < 2:
        return _opaque_decomp(
            "aten_convolution",
            list(operands[:3]),
            meta,
            "convolution",
            pattern_hint="convolution",
        )

    in_v = operands[0]
    w_v = operands[1]

    # Pull static shapes off operand types. Bail to opaque for any
    # non-rank-4 / dynamic-shape case; the existing opaque path is the
    # safety net.
    in_type = in_v.type
    w_type = w_v.type
    if not (isinstance(in_type, TensorType) and isinstance(w_type, TensorType)):
        return _opaque_decomp(
            "aten_convolution",
            list(operands[:3]),
            meta,
            "convolution",
            pattern_hint="convolution",
        )
    in_shape = in_type.get_shape()
    w_shape = w_type.get_shape()
    if len(in_shape) != 4 or len(w_shape) != 4:
        return _opaque_decomp(
            "aten_convolution",
            list(operands[:3]),
            meta,
            "convolution",
            pattern_hint="convolution",
        )
    if any(d <= 0 for d in (*in_shape, *w_shape)):
        return _opaque_decomp(
            "aten_convolution",
            list(operands[:3]),
            meta,
            "convolution",
            pattern_hint="convolution",
        )

    # Direct convolution as a single linalg.generic contraction, for the common
    # 2-D, groups=1, no-padding, dilation-1 case (patch-embed convs). Falls through to
    # the im2col path otherwise.
    direct = _try_direct_conv2d(operands, meta, in_shape, w_shape)
    if direct is not None:
        return direct

    val: Any = meta.get("val")
    if val is None or not hasattr(val, "shape"):
        return _opaque_decomp(
            "aten_convolution",
            list(operands[:3]),
            meta,
            "convolution",
            pattern_hint="convolution",
        )
    out_shape = _static_shape(val.shape)
    if len(out_shape) != 4 or any(d <= 0 for d in out_shape):
        return _opaque_decomp(
            "aten_convolution",
            list(operands[:3]),
            meta,
            "convolution",
            pattern_hint="convolution",
        )

    n_in, c_in, _h_in, _w_in = in_shape
    f_out, c_w, kh, kw = w_shape
    n_out, f_check, h_out, w_out = out_shape
    if n_in != n_out or f_out != f_check or c_in != c_w:
        # Group conv / weird channel layout — opaque fallback.
        return _opaque_decomp(
            "aten_convolution",
            list(operands[:3]),
            meta,
            "convolution",
            pattern_hint="convolution",
        )

    elem = w_type.element_type
    k_dim = c_in * kh * kw
    nhw = n_out * h_out * w_out

    ops: list[Operation] = []
    region_ids: list[str] = []

    # 1. im2col: (N, C, H, W) → (K, N*H'*W'), opaque (target may
    # ship a real im2col kernel, or the pack composer can do it).
    im2col_type = TensorType(elem, [k_dim, nhw])
    im2col_call = CallOp("aten_im2col", [in_v], [im2col_type])
    im2col_rid = _next_region_id("im2col")
    _attach_region_id(im2col_call, im2col_rid)
    im2col_call.attributes["m2m.dispatch_id"] = StringAttr(im2col_rid)
    region_ids.append(im2col_rid)
    ops.append(im2col_call)

    # 2. flatten weight (F, C, kH, kW) → (F, K) — opaque.
    w_flat_type = TensorType(elem, [f_out, k_dim])
    w_flat_call = CallOp("aten_flatten_weight", [w_v], [w_flat_type])
    w_flat_rid = _next_region_id("flatten")
    _attach_region_id(w_flat_call, w_flat_rid)
    w_flat_call.attributes["m2m.dispatch_id"] = StringAttr(w_flat_rid)
    region_ids.append(w_flat_rid)
    ops.append(w_flat_call)

    # 3. linalg.matmul: W_flat (F, K) @ im2col (K, N*H'*W') → (F, N*H'*W').
    mm_out_type = TensorType(elem, [f_out, nhw])
    mm_empty = _make_empty(mm_out_type)
    ops.append(mm_empty)
    matmul = MatmulOp(
        inputs=[w_flat_call.res[0], im2col_call.res[0]],
        outputs=[mm_empty.results[0]],
        res=[mm_out_type],
    )
    mm_rid = _next_region_id("matmul")
    _attach_region_id(matmul, mm_rid)
    matmul.attributes["m2m.dispatch_id"] = StringAttr(mm_rid)
    region_ids.append(mm_rid)
    ops.append(matmul)

    # 4. reshape (F, N*H'*W') → (N, F, H', W') — opaque.
    out_type = TensorType(elem, [n_out, f_out, h_out, w_out])
    reshape_call = CallOp("aten_reshape", [matmul.res[0]], [out_type])
    reshape_rid = _next_region_id("reshape")
    _attach_region_id(reshape_call, reshape_rid)
    reshape_call.attributes["m2m.dispatch_id"] = StringAttr(reshape_rid)
    region_ids.append(reshape_rid)
    ops.append(reshape_call)

    return DecompResult(
        ops=ops,
        result=reshape_call.res[0],
        region_ids=region_ids,
        pattern_hint="convolution_im2col_matmul",
    )


def decompose_select_int(operands, meta, node_name):
    """aten.select.int(input, dim, index) -> rank-reducing tensor.extract_slice (family slice)."""
    if not operands or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_select", operands[:1], meta, "layout", pattern_hint="select")
    src = operands[0]
    in_shape = list(src.type.get_shape())
    if any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_select", operands[:1], meta, "layout", pattern_hint="select")
    rank = len(in_shape)
    dim = int(_fx_arg(meta, 1, 0) or 0) % rank
    index = int(_fx_arg(meta, 2, 0) or 0)
    if index < 0:
        index += in_shape[dim]
    offsets = [0] * rank
    offsets[dim] = max(0, min(index, in_shape[dim] - 1))
    sizes = list(in_shape)
    sizes[dim] = 1
    from xdsl.dialects.tensor import ExtractSliceOp

    op = ExtractSliceOp.from_static_parameters(src, offsets, sizes, [1] * rank, reduce_rank=True)
    rid = _next_region_id("select")
    _attach_region_id(op, rid)
    op.attributes["m2m.family"] = StringAttr("slice")
    return DecompResult(ops=[op], result=op.results[0], region_ids=[rid], pattern_hint="select")


def decompose_embedding(operands, meta, node_name):
    """aten.embedding(weight[V,D], indices[*I]) -> gather out[*I,D]=weight[indices] via
    linalg.generic + tensor.extract (family gather)."""
    if len(operands) < 2 or not isinstance(operands[0].type, TensorType) or not isinstance(operands[1].type, TensorType):
        return _opaque_decomp("aten_embedding", operands[:2], meta, "embedding", pattern_hint="embedding_lookup")
    weight, indices = operands[0], operands[1]
    welem = weight.type.element_type
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    idx_shape = _shape_of(indices)
    if idx_shape is None or any(d < 0 for d in out_shape) or any(d < 0 for d in idx_shape) or len(out_shape) != len(idx_shape) + 1:
        return _opaque_decomp("aten_embedding", operands[:2], meta, "embedding", pattern_hint="embedding_lookup")

    from xdsl.dialects.arith import IndexCastOp
    from xdsl.dialects.builtin import AffineMapAttr, IndexType
    from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import EmptyOp, ExtractOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    rank = len(out_shape)
    out_t = TensorType(welem, out_shape)
    empty = EmptyOp([], out_t)
    # indices map: (i0..i_{k-1}, d) -> (i0..i_{k-1})   (drop the trailing D dim)
    idx_map = AffineMap(rank, 0, tuple(AffineExpr.dimension(i) for i in range(rank - 1)))
    out_map = AffineMap.identity(rank)
    idx_elem = indices.type.element_type
    blk = Block(arg_types=[idx_elem, welem])
    ic = IndexCastOp(blk.args[0], IndexType())
    didx = IndexOp(rank - 1)
    ext = ExtractOp(weight, [ic.results[0], didx.results[0]], welem)
    blk.add_op(ic)
    blk.add_op(didx)
    blk.add_op(ext)
    blk.add_op(YieldOp(ext.results[0]))
    gen = GenericOp(
        inputs=[indices],
        outputs=[empty.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(idx_map), AffineMapAttr(out_map)],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * rank,
        result_types=[out_t],
    )
    rid = _next_region_id("gather")
    for op in (empty, gen):
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("gather")
    return DecompResult(ops=[empty, gen], result=gen.results[0], region_ids=[rid], pattern_hint="embedding")


def _sigmoid_build(args, out_elem):
    """sigmoid(x) = 1 / (1 + exp(-x))."""
    from xdsl.dialects.arith import AddfOp, ConstantOp, DivfOp, NegfOp
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.math import ExpOp

    x = args[0]
    one = ConstantOp(FloatAttr(1.0, out_elem), out_elem)
    neg = NegfOp(x)
    ex = ExpOp(neg.results[0])
    den = AddfOp(one.results[0], ex.results[0])
    r = DivfOp(one.results[0], den.results[0])
    return [one, neg, ex, den, r], r.results[0]


def _silu_build(args, out_elem):
    """silu(x) = x * sigmoid(x)."""
    from xdsl.dialects.arith import MulfOp

    ops, sig = _sigmoid_build(args, out_elem)
    m = MulfOp(args[0], sig)
    return [*ops, m], m.results[0]


def decompose_sigmoid(operands, meta, node_name):
    """aten.sigmoid.default(input) -> 1/(1+exp(-x)) via linalg.generic."""
    real = _unary_elementwise(operands, meta, "sigmoid", _sigmoid_build)
    if real is not None:
        return real
    return _opaque_decomp("aten_sigmoid", operands, meta, "elementwise", pattern_hint="sigmoid")


def decompose_neg(operands, meta, node_name):
    """aten.neg.default(input) -> arith.negf via linalg.generic."""
    from xdsl.dialects.arith import NegfOp

    real = _unary_elementwise(operands, meta, "neg", _un_build(NegfOp))
    if real is not None:
        return real
    return _opaque_decomp("aten_neg", operands, meta, "elementwise", pattern_hint="neg")


def decompose_silu(operands, meta, node_name):
    """aten.silu.default(input) -> x*sigmoid(x) via linalg.generic."""
    real = _unary_elementwise(operands, meta, "silu", _silu_build)
    if real is not None:
        return real
    return _opaque_decomp("aten_silu", operands, meta, "elementwise", pattern_hint="silu")


def decompose_sub_tensor(operands, meta, node_name):
    """aten.sub.Tensor(a, b) -> arith.subf via linalg.generic."""
    from xdsl.dialects.arith import SubfOp

    real = _binary_elementwise(operands, meta, "sub", _arith_build2("SubfOp", "SubiOp"))
    if real is not None:
        return real
    return _opaque_decomp("aten_sub", operands[:2], meta, "elementwise", pattern_hint="sub")


def decompose_div_tensor(operands, meta, node_name):
    """aten.div.Tensor(a, b) -> arith.divf via linalg.generic."""
    from xdsl.dialects.arith import DivfOp

    real = _binary_elementwise(operands, meta, "div", _arith_build2("DivfOp", "DivSIOp"))
    if real is not None:
        return real
    return _opaque_decomp("aten_div", operands[:2], meta, "elementwise", pattern_hint="div")


# ---- layout / structural (preserve shape metadata; no compute) ----


def _reshape_decomp(operands, meta, node_name, *, hint: str, prefix: str):
    """Shared logical-reshape decomposition (view/reshape/unsqueeze/squeeze/flatten).

    tensor.reshape preserves the element type, so the result dtype MUST equal the
    source SSA's dtype (meta['val'].dtype can disagree, e.g. bf16/f32), or module
    verification fails."""
    val: Any = meta["val"]
    src_type = operands[0].type
    meta_elem = _element_type_from_meta(meta)
    if not isinstance(src_type, TensorType):
        return _opaque_decomp(f"aten_{prefix}", operands[:1], meta, "layout", pattern_hint=hint)
    out_shape = _static_shape(val.shape)
    # reshape preserves dtype. When the meta dtype disagrees with the source SSA dtype
    # (a fused reshape+cast, e.g. bool unsqueeze captured as i64), reshape in the source
    # dtype then cast the result to the meta dtype.
    src_elem = src_type.element_type
    emitted = _emit_reshape(operands[0], out_shape, src_elem)
    if emitted is None:
        return _opaque_decomp(f"aten_{prefix}", operands[:1], meta, "layout", pattern_hint=hint)
    ops, result = emitted
    if src_elem != meta_elem:
        cast = _cast_tensor(result, out_shape, meta_elem)
        if cast is None:
            return _opaque_decomp(f"aten_{prefix}", operands[:1], meta, "layout", pattern_hint=hint)
        ops += cast[0]
        result = cast[1]
    rid = _next_region_id(prefix)
    for op in ops:
        _attach_region_id(op, rid)
    return DecompResult(ops=ops, result=result, region_ids=[rid], pattern_hint=hint)


def decompose_view(operands, meta, node_name):
    """aten.view.default(input, shape) -> tensor.reshape."""
    return _reshape_decomp(operands, meta, node_name, hint="view", prefix="view")


def decompose_unsqueeze(operands, meta, node_name):
    """aten.unsqueeze.default(input, dim) -> tensor.reshape inserting a size-1 dim."""
    return _reshape_decomp(operands, meta, node_name, hint="unsqueeze", prefix="unsqueeze")


def decompose_squeeze(operands, meta, node_name):
    """aten.squeeze[.dim[s]](input, ...) -> tensor.reshape dropping size-1 dims.

    Pure layout op; the target shape comes from meta['val'], so squeeze.default
    (drop all 1s), squeeze.dim, and squeeze.dims all share the reshape path."""
    return _reshape_decomp(operands, meta, node_name, hint="squeeze", prefix="squeeze")


def decompose_expand(operands, meta, node_name):
    """aten.expand.default(input, sizes) -> broadcast copy via linalg.generic."""
    val: Any = meta["val"]
    out_shape = _static_shape(val.shape)
    elem = _element_type_from_meta(meta)
    if operands and _t_elem(operands[0]) == elem and not any(d < 0 for d in out_shape):
        bm = _broadcast_map(_shape_of(operands[0]) or [], out_shape)
        if bm is not None:
            em = _elementwise(
                [operands[0]], TensorType(elem, out_shape),
                lambda args, oe: ([], args[0]),  # identity copy
                input_maps=[bm],
            )
            if em is not None:
                ops, res = em
                rid = _next_region_id("expand")
                for op in ops:
                    _attach_region_id(op, rid)
                return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="expand")
    return _opaque_decomp("aten_expand", operands[:1], meta, "layout", pattern_hint="expand")


def decompose_cat(operands, meta, node_name):
    """aten.cat.default(tensors, dim?) -> tensor concat.

    Concat preserves the tensor inputs but not the scalar ``dim`` — the
    lowering wave will read the axis from node.args[1].
    """
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    elem = _element_type_from_meta(meta)
    if (
        len(operands) >= 1
        and not any(d < 0 for d in out_shape)
        and all(isinstance(o.type, TensorType) for o in operands)
    ):
        dim = int(_fx_arg(meta, 1, 0) or 0)
        if dim < 0:
            dim += len(out_shape)
        from xdsl.dialects.tensor import ConcatOp

        # tensor.concat requires a uniform element type; torch.cat promotes mixed
        # dtypes (e.g. f32 + bf16 -> bf16). Cast each input to the result dtype first.
        ops: list = []
        casted: list = []
        for o in operands:
            if o.type.element_type == elem:
                casted.append(o)
                continue
            c = _cast_tensor(o, o.type.get_shape(), elem)
            if c is None:
                return _opaque_decomp("aten_cat", operands, meta, "layout", pattern_hint="cat")
            ops += c[0]
            casted.append(c[1])
        op = ConcatOp(inputs=casted, dim=dim, result_type=TensorType(elem, out_shape))
        ops.append(op)
        rid = _next_region_id("cat")
        for o in ops:
            _attach_region_id(o, rid)
            o.attributes["m2m.family"] = StringAttr("concat")
        return DecompResult(ops=ops, result=op.results[0], region_ids=[rid], pattern_hint="cat")
    return _opaque_decomp("aten_cat", operands, meta, "layout", pattern_hint="cat")


def decompose_split_with_sizes(operands, meta, node_name):
    """aten.split_with_sizes.default(input, split_sizes, dim?).

    Emits one tensor.extract_slice per chunk (multi-output); getitem(node, i) resolves to
    the i-th slice via the importer's multi_results map.
    """
    if not operands or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_split_with_sizes", operands[:1], meta, "layout", pattern_hint="split")
    src = operands[0]
    in_shape = list(src.type.get_shape())
    if any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_split_with_sizes", operands[:1], meta, "layout", pattern_hint="split")
    rank = len(in_shape)
    sizes_list = _fx_arg(meta, 1, None)
    dim = int(_fx_arg(meta, 2, 0) or 0) % rank
    if not isinstance(sizes_list, (list, tuple)) or not sizes_list:
        return _opaque_decomp("aten_split_with_sizes", operands[:1], meta, "layout", pattern_hint="split")

    from xdsl.dialects.tensor import ExtractSliceOp

    ops: list[Operation] = []
    results: list[SSAValue] = []
    rid = _next_region_id("split")
    off = 0
    for sz in sizes_list:
        sz = int(sz)
        offsets = [0] * rank
        offsets[dim] = off
        sizes = list(in_shape)
        sizes[dim] = sz
        op = ExtractSliceOp.from_static_parameters(src, offsets, sizes, [1] * rank)
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("slice")
        ops.append(op)
        results.append(op.results[0])
        off += sz
    return DecompResult(ops=ops, result=results[0], results=results, region_ids=[rid], pattern_hint="split")


def decompose_clone(operands, meta, node_name):
    """aten.clone.default(input) -> identity: forward the operand SSA (no op emitted)."""
    if operands:
        return DecompResult(ops=[], result=operands[0], pattern_hint="clone")
    return _opaque_decomp("aten_clone", operands[:1], meta, "identity", pattern_hint="clone")


# --- layout / reshape / contiguous (production-readiness fill-ins) -----------


def decompose_contiguous(operands, meta, node_name):
    """aten.contiguous.default(input) -> identity: forward the operand SSA (no op)."""
    if operands:
        return DecompResult(ops=[], result=operands[0], pattern_hint="contiguous")
    return _opaque_decomp("aten_contiguous", operands[:1], meta, "layout", pattern_hint="contiguous")


def decompose_transpose_int(operands, meta, node_name):
    """aten.transpose.int(input, dim0, dim1) -> shape-swapped tensor.

    ``dim0`` / ``dim1`` are scalar ints and don't appear as SSA
    operands. The result's shape comes from ``meta['val'].shape``
    which already reflects the transposition.
    """
    return _opaque_decomp(
        "aten_transpose",
        operands[:1],
        meta,
        "layout",
        pattern_hint="transpose",
    )


def decompose_matmul(operands, meta, node_name):
    """aten.matmul.default(a, b) -> linalg.matmul when rank-2, else opaque.

    Structural emission for the 2D × 2D case drops the opaque rate
    on real LLM fixtures. Higher-rank matmul stays opaque with
    ``pattern_hint="batch_matmul"`` so  dispatch handles it.
    """
    val = meta.get("val")
    shape = getattr(val, "shape", ()) if val is not None else ()
    out_rank = len(shape)

    if out_rank == 2 and len(operands) == 2:
        lhs, rhs = operands[0], operands[1]
        lhs_type = getattr(lhs, "type", None)
        rhs_type = getattr(rhs, "type", None)
        if (
            isinstance(lhs_type, TensorType)
            and isinstance(rhs_type, TensorType)
            and len(list(lhs_type.get_shape())) == 2
            and len(list(rhs_type.get_shape())) == 2
        ):
            result_type = TensorType(_t_elem(operands[0]), _static_shape(shape))
            init = _make_empty(result_type)
            mm = MatmulOp(
                inputs=[lhs, rhs],
                outputs=[init.results[0]],
                res=[result_type],
            )
            rid = _next_region_id("matmul")
            _attach_region_id(mm, rid)
            return DecompResult(
                ops=[init, mm],
                result=mm.results[0],
                region_ids=[rid],
                pattern_hint="matmul",
            )

    hint = "batch_matmul" if out_rank > 2 else "matmul"
    return _opaque_decomp(
        "aten_matmul",
        operands[:2],
        meta,
        "matmul",
        pattern_hint=hint,
    )


# ---------------------------------------------------------------------------
# Wave 7 — TinyLlama opaque-tail closure: 10 new families that previously
# fell through to the unhinted opaque fallback. Each emits a typed
# func.call with a pattern_hint so the kernel selector recognises them
# as members of a known family (not unknown-tail).
# ---------------------------------------------------------------------------


def decompose_to_copy(operands, meta, node_name):
    """aten._to_copy.default — dtype cast (linalg.generic arith.trunc/ext/sitofp), or
    identity when the dtype is unchanged (device-only copy -> no cast op)."""
    target_elem = _element_type_from_meta(meta)
    if operands and isinstance(operands[0].type, TensorType) and operands[0].type.element_type == target_elem:
        return DecompResult(ops=[], result=operands[0], pattern_hint="identity")
    real = _unary_elementwise(operands, meta, "dtype_cast", _cast_scalar_build(target_elem), out_elem=target_elem)
    if real is not None:
        return real
    return _opaque_decomp("aten_to_dtype", operands[:1], meta, "cast", pattern_hint="dtype_cast")


def decompose_where_self(operands, meta, node_name):
    """aten.where.self(condition, x, y) — elementwise selection."""
    return _opaque_decomp("aten_where", operands[:3], meta, "select", pattern_hint="where")


def decompose_scalar_tensor(operands, meta, node_name):
    """aten.scalar_tensor.default(value, ...) — 0-rank constant fill."""
    return _opaque_decomp("aten_scalar_tensor", [], meta, "fill", pattern_hint="fill")


def decompose_full_like(operands, meta, node_name):
    """aten.full_like.default(input, fill_value, ...) — same-shape fill."""
    return _opaque_decomp("aten_full_like", operands[:1], meta, "fill", pattern_hint="fill")


def decompose_full(operands, meta, node_name):
    """aten.full.default(size, fill_value, ...) — explicit-shape fill."""
    return _opaque_decomp("aten_full", [], meta, "fill", pattern_hint="fill")


def decompose_arange(operands, meta, node_name):
    """aten.arange[.start[_step]] -> 1-D iota via linalg.generic + linalg.index (family iota)."""
    from xdsl.dialects.arith import AddiOp, ConstantOp, IndexCastOp, MuliOp, SIToFPOp
    from xdsl.dialects.builtin import AffineMapAttr, FloatAttr, IntegerAttr, IntegerType
    from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import EmptyOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    elem = _element_type_from_meta(meta)
    if len(out_shape) != 1 or out_shape[0] < 0:
        return _opaque_decomp("aten_arange", [], meta, "arange", pattern_hint="arange")
    nargs = len(meta.get("_fx_args", ()))
    start = _fx_arg(meta, 0, 0) if nargs >= 2 else 0
    step = _fx_arg(meta, 2, 1) if nargs >= 3 else 1

    out_t = TensorType(elem, out_shape)
    empty = EmptyOp([], out_t)
    blk = Block(arg_types=[elem])
    idx = IndexOp(0)
    body = [idx]
    is_int = isinstance(elem, IntegerType)
    if is_int:
        ic = IndexCastOp(idx.results[0], elem)
        stepc = ConstantOp(IntegerAttr(int(step), elem), elem)
        mul = MuliOp(ic.results[0], stepc.results[0])
        startc = ConstantOp(IntegerAttr(int(start), elem), elem)
        add = AddiOp(startc.results[0], mul.results[0])
        body += [ic, stepc, mul, startc, add]
        yield_v = add.results[0]
    else:
        ic = IndexCastOp(idx.results[0], i64)
        f = SIToFPOp(ic.results[0], elem)
        stepc = ConstantOp(FloatAttr(float(step), elem), elem)
        from xdsl.dialects.arith import AddfOp, MulfOp

        mul = MulfOp(f.results[0], stepc.results[0])
        startc = ConstantOp(FloatAttr(float(start), elem), elem)
        add = AddfOp(startc.results[0], mul.results[0])
        body += [ic, f, stepc, mul, startc, add]
        yield_v = add.results[0]
    for op in body:
        blk.add_op(op)
    blk.add_op(YieldOp(yield_v))
    gen = GenericOp(
        inputs=[],
        outputs=[empty.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(AffineMap.identity(1))],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)],
        result_types=[out_t],
    )
    rid = _next_region_id("iota")
    for op in (empty, gen):
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("iota")
    return DecompResult(ops=[empty, gen], result=gen.results[0], region_ids=[rid], pattern_hint="arange")


def decompose_sum_dim(operands, meta, node_name):
    """aten.sum.dim_IntList(input, dims, keepdim?) -> linalg.reduce add (family reduce)."""
    from xdsl.dialects.arith import AddfOp, AddiOp
    from xdsl.dialects.builtin import IntegerType

    x = operands[0]
    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_sum", operands[:1], meta, "reduce", pattern_hint="reduce_sum")
    elem = _t_elem(x)
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    # torch sums in the output dtype (e.g. bool sum -> i64 counts); cast first so the
    # accumulation doesn't overflow/diverge from the captured result dtype.
    out_elem = _element_type_from_meta(meta)
    ops: list = []
    if out_elem != elem:
        cast = _cast_tensor(x, in_shape, out_elem)
        if cast is None:
            return _opaque_decomp("aten_sum", operands[:1], meta, "reduce", pattern_hint="reduce_sum")
        ops += cast[0]
        x = cast[1]
        elem = out_elem
    add_cls = AddiOp if isinstance(elem, IntegerType) else AddfOp
    dims = _fx_arg(meta, 1, None)
    if dims is None:
        dims = list(range(len(in_shape)))
    elif isinstance(dims, int):
        dims = [dims]
    else:
        dims = list(dims)
    dims = [d % len(in_shape) for d in dims]
    rops, summ, rsh = _reduce(x, in_shape, dims, 0, add_cls, elem)
    ops += rops
    res = summ
    if rsh != out_shape:
        res = _keepdim_reshape(ops, res, rsh, out_shape, elem)
        if res is None:
            return _opaque_decomp("aten_sum", operands[:1], meta, "reduce", pattern_hint="reduce_sum")
    rid = _next_region_id("reduce")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("reduce")
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="reduce_sum")


def decompose_reciprocal(operands, meta, node_name):
    """aten.reciprocal(x) -> 1/x via linalg.generic (family elementwise)."""
    from xdsl.dialects.arith import ConstantOp, DivfOp
    from xdsl.dialects.builtin import FloatAttr

    def build(args, oe):
        # reciprocal of an integer tensor yields float (e.g. 1/i64 -> f32): cast first.
        cops, x = _cast_scalar_arg(args[0], oe)
        c = ConstantOp(FloatAttr(1.0, oe), oe)
        d = DivfOp(c.results[0], x)
        return [*cops, c, d], d.results[0]

    real = _pointwise(operands[:1], meta, build, family="elementwise")
    if real is not None:
        return real
    return _opaque_decomp("aten_reciprocal", operands[:1], meta, "elementwise", pattern_hint="reciprocal")


def decompose_logical_not(operands, meta, node_name):
    """aten.logical_not.default — pointwise boolean NOT."""
    return _opaque_decomp("aten_logical_not", operands[:1], meta, "logical", pattern_hint="logical_not")


def decompose_bitwise_and(operands, meta, node_name):
    """aten.bitwise_and.Tensor(a, b) -> arith.andi via linalg.generic (family bitwise)."""
    from xdsl.dialects.arith import AndIOp

    def build(args, oe):
        op = AndIOp(args[0], args[1])
        return [op], op.results[0]

    real = _pointwise(operands[:2], meta, build, family="bitwise")
    if real is not None:
        return real
    return _opaque_decomp("aten_bitwise_and", operands[:2], meta, "bitwise", pattern_hint="bitwise_and")


def decompose_any_dim(operands, meta, node_name):
    """aten.any.dim(input, dim, keepdim) — boolean OR reduction along dim."""
    return _opaque_decomp("aten_any_dim", operands[:1], meta, "bool_reduce", pattern_hint="bool_reduce")


def decompose_bucketize(operands, meta, node_name):
    """aten.bucketize.Tensor(input, boundaries, *, right=False) — searchsorted.

    out[*s] = #{b : boundaries[b] < input[*s]}  (right=False, the default)
            = #{b : boundaries[b] <= input[*s]} (right=True)
    Emitted as a counting reduction ``linalg.generic`` (family ``search``): a parallel
    loop per input dim plus one reduction over the (sorted) boundary axis, accumulating
    a +1 each time the predicate holds."""
    if len(operands) < 2 or not isinstance(operands[0].type, TensorType) \
            or not isinstance(operands[1].type, TensorType):
        return _opaque_decomp("aten_bucketize", operands[:2], meta, "search", pattern_hint="bucketize")
    x, bnd = operands[0], operands[1]
    in_shape = _shape_of(x)
    bnd_shape = _shape_of(bnd)
    if in_shape is None or bnd_shape is None or len(bnd_shape) != 1 \
            or any(d < 0 for d in (*in_shape, *bnd_shape)):
        return _opaque_decomp("aten_bucketize", operands[:2], meta, "search", pattern_hint="bucketize")

    from xdsl.dialects.arith import AddiOp, CmpfOp, CmpiOp, ConstantOp, SelectOp
    from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, IntegerType
    from xdsl.dialects.linalg import GenericOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    right = bool(meta.get("_fx_kwargs", {}).get("right", False) or _fx_arg(meta, 3, False))
    in_elem = x.type.element_type
    out_elem = _element_type_from_meta(meta)
    if not isinstance(out_elem, IntegerType):
        out_elem = IntegerType(64)
    rank = len(in_shape)
    out_t = TensorType(out_elem, list(in_shape))

    zero = ConstantOp(IntegerAttr(0, out_elem), out_elem)
    one = ConstantOp(IntegerAttr(1, out_elem), out_elem)
    init = SplatOp(zero.result, [], out_t)

    in_map = AffineMap(rank + 1, 0, tuple(AffineExpr.dimension(d) for d in range(rank)))
    bnd_map = AffineMap(rank + 1, 0, (AffineExpr.dimension(rank),))
    out_map = in_map

    is_int = isinstance(in_elem, IntegerType)
    # predicate: boundaries[b] < input (right=False) or <= input (right=True)
    pred_kind = ("sle" if right else "slt") if is_int else ("ole" if right else "olt")
    blk = Block(arg_types=[in_elem, in_elem, out_elem])
    pred = CmpiOp(blk.args[1], blk.args[0], pred_kind) if is_int \
        else CmpfOp(blk.args[1], blk.args[0], pred_kind)   # arg0=input, arg1=boundary
    inc = SelectOp(pred.results[0], one.result, zero.result)
    acc = AddiOp(blk.args[2], inc.results[0])
    for op in (pred, inc, acc):
        blk.add_op(op)
    blk.add_op(YieldOp(acc.results[0]))

    iters = [IteratorTypeAttr(IteratorType.PARALLEL)] * rank + [IteratorTypeAttr(IteratorType.REDUCTION)]
    gen = GenericOp(
        inputs=[x, bnd],
        outputs=[init.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(in_map), AffineMapAttr(bnd_map), AffineMapAttr(out_map)],
        iterator_types=iters,
        result_types=[out_t],
    )
    ops = [zero, one, init, gen]
    rid = _next_region_id("search")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("search")
    return DecompResult(ops=ops, result=gen.results[0], region_ids=[rid], pattern_hint="bucketize")


def _flatten_1d(ssa, shape, elem, ops):
    """Collapse a tensor to 1-D [numel] (no-op if already 1-D). Appends to ``ops``,
    returns (flat_ssa, numel) or (None, 0) if the reshape can't be emitted."""
    numel = 1
    for d in shape:
        numel *= d
    if len(shape) == 1:
        return ssa, numel
    emitted = _emit_reshape(ssa, [numel], elem)
    if emitted is None:
        return None, 0
    ops.extend(emitted[0])
    return emitted[1], numel


def _bool_mask_gather(source, mask, shape, elem):
    """src[mask] (full boolean mask) -> tensor<?xELEM> via stream compaction.

    Counts the True entries (dynamic extent), allocates a tensor<?> of that size, then
    an scf.for/scf.if loop copies each selected element into the next write slot
    (tensor.insert). Family ``mask_gather`` -- a genuinely data-dependent op expressed
    with a dynamic dimension rather than bailed to opaque."""
    from xdsl.dialects.arith import AddiOp, ConstantOp, IndexCastOp
    from xdsl.dialects.builtin import DYNAMIC_INDEX, IndexType, IntegerAttr, i64
    from xdsl.dialects.linalg import ReduceOp  # noqa: F401 (via _reduce)
    from xdsl.dialects.scf import ForOp, IfOp
    from xdsl.dialects.scf import YieldOp as ScfYield
    from xdsl.dialects.tensor import EmptyOp, ExtractOp, InsertOp
    from xdsl.ir import Block, Region

    ops: list[Operation] = []
    src_flat, numel = _flatten_1d(source, shape, elem, ops)
    mask_flat, _ = _flatten_1d(mask, shape, mask.type.element_type, ops)
    if src_flat is None or mask_flat is None:
        return _opaque_decomp("aten_index", [source], {"val": None}, "gather", pattern_hint="gather")

    # count = sum(mask as i64) -> index
    cast = _cast_tensor(mask_flat, [numel], i64)
    if cast is None:
        return _opaque_decomp("aten_index", [source], {"val": None}, "gather", pattern_hint="gather")
    ops += cast[0]
    from xdsl.dialects.arith import AddiOp as _AddiOp
    rops, summ, _ = _reduce(cast[1], [numel], [0], 0, _AddiOp, i64)
    ops += rops
    cnt_ext = ExtractOp(summ, [], i64)
    cnt_idx = IndexCastOp(cnt_ext.results[0], IndexType())
    ops += [cnt_ext, cnt_idx]

    dyn_t = TensorType(elem, [DYNAMIC_INDEX])
    c0 = ConstantOp(IntegerAttr(0, IndexType()), IndexType())
    c1 = ConstantOp(IntegerAttr(1, IndexType()), IndexType())
    cN = ConstantOp(IntegerAttr(numel, IndexType()), IndexType())
    init = EmptyOp([cnt_idx.results[0]], dyn_t)
    ops += [c0, c1, cN, init]

    # loop body: (iv, acc: tensor<?>, wp: index)
    body = Block(arg_types=[IndexType(), dyn_t, IndexType()])
    iv, acc, wp = body.args
    m = ExtractOp(mask_flat, [iv], mask.type.element_type)
    body.add_op(m)
    then_blk = Block()
    v = ExtractOp(src_flat, [iv], elem)
    ins = InsertOp(v.results[0], acc, [wp])
    wp2 = AddiOp(wp, c1.results[0])
    then_blk.add_ops([v, ins, wp2, ScfYield(ins.results[0], wp2.results[0])])
    else_blk = Block()
    else_blk.add_op(ScfYield(acc, wp))
    iff = IfOp(m.results[0], [dyn_t, IndexType()], Region(then_blk), Region(else_blk))
    body.add_op(iff)
    body.add_op(ScfYield(iff.results[0], iff.results[1]))
    floop = ForOp(c0.results[0], cN.results[0], c1.results[0], [init.results[0], c0.results[0]], Region(body))
    ops.append(floop)

    rid = _next_region_id("mask_gather")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("mask_gather")
    return DecompResult(ops=ops, result=floop.results[0], region_ids=[rid], pattern_hint="mask_gather")


def decompose_index_tensor(operands, meta, node_name):
    """aten.index.Tensor(self, [idx0, idx1, ...]) — advanced integer-array gather.

    Lowered (family ``gather``) as a ``linalg.generic`` over the output (advanced-index
    broadcast) shape: each output coordinate reads the broadcast index tensors and does a
    ``tensor.extract`` from ``self``. Handles the full-indexing case (one integer index
    tensor per source dim). Boolean-mask indexing (``self[mask]``) yields a data-dependent
    shape that static lowering can't express, so it stays an opaque placeholder."""
    from xdsl.dialects.builtin import IntegerType

    if not operands or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_index", operands[:1], meta, "gather", pattern_hint="gather")
    source = operands[0]
    idxs = list(operands[1:])
    src_shape = _shape_of(source)
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    out_elem = source.type.element_type

    def bail():
        return _opaque_decomp("aten_index", operands[:1], meta, "gather", pattern_hint="gather")

    if not idxs or any(not isinstance(t.type, TensorType) for t in idxs):
        return bail()
    # boolean-mask indexing src[mask]: data-dependent #True -> a dynamic (?) output dim.
    # Lowered by stream compaction (scf.for + scf.if + tensor.insert) into tensor<?xELEM>.
    if len(idxs) == 1 and isinstance(idxs[0].type.element_type, IntegerType) \
            and idxs[0].type.element_type.width.data == 1:
        if src_shape is not None and _shape_of(idxs[0]) == src_shape and not any(d < 0 for d in src_shape):
            return _bool_mask_gather(source, idxs[0], src_shape, out_elem)
        return bail()
    if src_shape is None or any(d < 0 for d in src_shape) or any(d < 0 for d in out_shape):
        return bail()
    R = len(src_shape)

    # Recover the FULL index list (with None entries) from meta -- the importer drops
    # None when flattening operands, so only meta knows which source dims are indexed.
    index_list = _fx_arg(meta, 1)
    if not isinstance(index_list, (list, tuple)):
        index_list = [None] * (R - len(idxs)) + list(idxs)  # best-effort: assume trailing
    # positions of the integer (non-None) indices, in source-dim order
    int_positions = [d for d, e in enumerate(index_list) if e is not None]
    if len(int_positions) != len(idxs) or not int_positions:
        return bail()
    # require the integer indices to be CONTIGUOUS (numpy's adv-shape-inserted-at-first rule)
    if int_positions != list(range(int_positions[0], int_positions[0] + len(int_positions))):
        return bail()
    p0 = int_positions[0]
    pk = int_positions[-1]
    k = len(idxs)

    # advanced (broadcast) shape of the integer index tensors, and where it sits in output
    out_rank = len(out_shape)
    rank_adv = out_rank - (R - k)          # output dims contributed by the index broadcast
    if rank_adv < 1 or p0 + rank_adv > out_rank:
        return bail()
    adv_shape = out_shape[p0:p0 + rank_adv]
    # each index tensor broadcasts onto the adv block (output dims p0 .. p0+rank_adv-1)
    base_maps = [_broadcast_map(_shape_of(t) or [], adv_shape) for t in idxs]
    if any(m is None for m in base_maps):
        return bail()

    from xdsl.dialects.arith import IndexCastOp
    from xdsl.dialects.builtin import AffineMapAttr, IndexType
    from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import EmptyOp, ExtractOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    # lift each index map from the adv subspace (rank_adv dims, based at 0) to the full
    # output space: select output dims p0..p0+rank_adv-1 as the adv coords, then apply
    # the broadcast map. base_map(P(out_dims)) where P picks the adv block.
    from xdsl.ir.affine import AffineExpr as _AE
    _P = AffineMap(out_rank, 0, tuple(_AE.dimension(p0 + i) for i in range(rank_adv)))
    maps = [m.compose(_P) for m in base_maps]

    out_t = TensorType(out_elem, out_shape)
    empty = EmptyOp([], out_t)
    idx_elems = [t.type.element_type for t in idxs]
    blk = Block(arg_types=[*idx_elems, out_elem])
    # build the R source coordinates: free leading dims, gathered dims, free trailing dims
    coords: list[Any] = [None] * R
    for d in range(p0):                                  # free leading: output dim d == source dim d
        ix = IndexOp(d); blk.add_op(ix); coords[d] = ix.results[0]
    for j, a in enumerate(blk.args[:k]):                 # gathered dims p0..pk
        ic = IndexCastOp(a, IndexType()); blk.add_op(ic); coords[p0 + j] = ic.results[0]
    for s in range(pk + 1, R):                           # free trailing source dims
        out_dim = p0 + rank_adv + (s - (pk + 1))
        ix = IndexOp(out_dim); blk.add_op(ix); coords[s] = ix.results[0]
    if any(c is None for c in coords):
        return bail()
    ext = ExtractOp(source, coords, out_elem)
    blk.add_op(ext)
    blk.add_op(YieldOp(ext.results[0]))

    gen = GenericOp(
        inputs=idxs,
        outputs=[empty.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(m) for m in maps] + [AffineMapAttr(AffineMap.identity(out_rank))],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * out_rank,
        result_types=[out_t],
    )
    ops = [empty, gen]
    rid = _next_region_id("gather")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("gather")
    return DecompResult(ops=ops, result=gen.results[0], region_ids=[rid], pattern_hint="index_gather")


def decompose_index_put(operands, meta, node_name):
    """aten.index_put(self, [mask], values, accumulate=False) — masked scatter.

    For a single boolean mask: self[mask] = values, where values is the (dynamic-length)
    compacted set of replacements. Lowered (family ``mask_scatter``) by an scf.for/scf.if
    loop that walks self in flat order, consuming ``values`` sequentially at each True
    position (tensor.insert) -- the output keeps self's static shape; only the consumed
    ``values`` tensor carries the dynamic (?) dim."""
    if len(operands) < 3 or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_index_put", operands[:1], meta, "scatter", pattern_hint="index_put")
    self_t, mask, values = operands[0], operands[1], operands[2]
    self_shape = _shape_of(self_t)
    elem = self_t.type.element_type
    from xdsl.dialects.builtin import IntegerType

    if self_shape is None or any(d < 0 for d in self_shape) \
            or not isinstance(mask.type, TensorType) \
            or not isinstance(mask.type.element_type, IntegerType) or mask.type.element_type.width.data != 1 \
            or _shape_of(mask) != self_shape:
        return _opaque_decomp("aten_index_put", operands[:1], meta, "scatter", pattern_hint="index_put")

    from xdsl.dialects.arith import AddiOp, ConstantOp
    from xdsl.dialects.builtin import IndexType, IntegerAttr
    from xdsl.dialects.scf import ForOp, IfOp
    from xdsl.dialects.scf import YieldOp as ScfYield
    from xdsl.dialects.tensor import ExtractOp, InsertOp
    from xdsl.ir import Block, Region

    ops: list[Operation] = []
    self_flat, numel = _flatten_1d(self_t, self_shape, elem, ops)
    mask_flat, _ = _flatten_1d(mask, self_shape, mask.type.element_type, ops)
    if self_flat is None or mask_flat is None:
        return _opaque_decomp("aten_index_put", operands[:1], meta, "scatter", pattern_hint="index_put")

    c0 = ConstantOp(IntegerAttr(0, IndexType()), IndexType())
    c1 = ConstantOp(IntegerAttr(1, IndexType()), IndexType())
    cN = ConstantOp(IntegerAttr(numel, IndexType()), IndexType())
    ops += [c0, c1, cN]

    flat_t = TensorType(elem, [numel])
    body = Block(arg_types=[IndexType(), flat_t, IndexType()])
    iv, acc, rp = body.args            # induction var, running self, read pointer into values
    m = ExtractOp(mask_flat, [iv], mask.type.element_type)
    body.add_op(m)
    then_blk = Block()
    v = ExtractOp(values, [rp], elem)
    ins = InsertOp(v.results[0], acc, [iv])
    rp2 = AddiOp(rp, c1.results[0])
    then_blk.add_ops([v, ins, rp2, ScfYield(ins.results[0], rp2.results[0])])
    else_blk = Block()
    else_blk.add_op(ScfYield(acc, rp))
    iff = IfOp(m.results[0], [flat_t, IndexType()], Region(then_blk), Region(else_blk))
    body.add_op(iff)
    body.add_op(ScfYield(iff.results[0], iff.results[1]))
    floop = ForOp(c0.results[0], cN.results[0], c1.results[0], [self_flat, c0.results[0]], Region(body))
    ops.append(floop)

    result = floop.results[0]
    if len(self_shape) != 1:
        back = _emit_reshape(result, list(self_shape), elem)
        if back is None:
            return _opaque_decomp("aten_index_put", operands[:1], meta, "scatter", pattern_hint="index_put")
        ops += back[0]
        result = back[1]

    rid = _next_region_id("mask_scatter")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("mask_scatter")
    return DecompResult(ops=ops, result=result, region_ids=[rid], pattern_hint="index_put")


def decompose_compare(operands, meta, node_name):
    """aten.{eq,ne,le,lt,gt,ge}.{Tensor,Scalar} — pointwise comparison."""
    return _opaque_decomp("aten_compare", operands[:2], meta, "compare", pattern_hint="compare")


def decompose_cos(operands, meta, node_name):
    """aten.cos.default — math.cos via linalg.generic (RoPE)."""
    from xdsl.dialects.math import CosOp

    real = _unary_elementwise(operands, meta, "cos", _un_build(CosOp))
    if real is not None:
        return real
    return _opaque_decomp("aten_cos", operands[:1], meta, "trig", pattern_hint="cos")


def decompose_sin(operands, meta, node_name):
    """aten.sin.default — math.sin via linalg.generic (RoPE)."""
    from xdsl.dialects.math import SinOp

    real = _unary_elementwise(operands, meta, "sin", _un_build(SinOp))
    if real is not None:
        return real
    return _opaque_decomp("aten_sin", operands[:1], meta, "trig", pattern_hint="sin")


def decompose_cumsum(operands, meta, node_name):
    """aten.cumsum.default(input, dim) — prefix sum along ``dim``.

    Lowered as a single masked-reduction ``linalg.generic`` (family ``scan``):
    out[..., i, ...] = sum_j (j <= i) ? x[..., j, ...] : 0
    Adds a parallel loop per dim plus one reduction loop ``j`` over the scan dim,
    using ``linalg.index`` to compare i vs j. Handles a dtype change (e.g. bool
    ``cumsum`` -> i64 counts) by casting the read element to the output dtype."""
    if not operands or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_cumsum", operands[:1], meta, "scan", pattern_hint="cumsum")
    x = operands[0]
    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_cumsum", operands[:1], meta, "scan", pattern_hint="cumsum")

    from xdsl.dialects.arith import (
        AddfOp,
        AddiOp,
        CmpiOp,
        ConstantOp,
        ExtSIOp,
        ExtUIOp,
        SelectOp,
    )
    from xdsl.dialects.builtin import AffineMapAttr, FloatAttr, IntegerAttr, IntegerType
    from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    rank = len(in_shape)
    dim = int(_fx_arg(meta, 1, -1) or 0) % rank
    in_elem = x.type.element_type
    out_elem = _element_type_from_meta(meta)
    out_shape = list(in_shape)
    out_t = TensorType(out_elem, out_shape)

    # zero accumulator init (splat over the full result shape)
    if isinstance(out_elem, IntegerType):
        zero = ConstantOp(IntegerAttr(0, out_elem), out_elem)
    else:
        zero = ConstantOp(FloatAttr(0.0, out_elem), out_elem)
    init = SplatOp(zero.result, [], out_t)

    # loop dims: rank parallel (i0..i_{rank-1}) + 1 reduction (j, the last)
    in_exprs = [AffineExpr.dimension(d) for d in range(rank)]
    in_exprs[dim] = AffineExpr.dimension(rank)          # read position uses j at the scan dim
    in_map = AffineMap(rank + 1, 0, tuple(in_exprs))
    out_map = AffineMap(rank + 1, 0, tuple(AffineExpr.dimension(d) for d in range(rank)))

    blk = Block(arg_types=[in_elem, out_elem])
    i_idx = IndexOp(dim)
    j_idx = IndexOp(rank)
    cond = CmpiOp(j_idx.results[0], i_idx.results[0], "ule")   # j <= i (lower-triangular mask)
    body = [i_idx, j_idx, cond]
    # cast the read element to the accumulator dtype if needed
    src = blk.args[0]
    if in_elem != out_elem:
        if isinstance(in_elem, IntegerType) and isinstance(out_elem, IntegerType):
            cast = ExtUIOp(src, out_elem) if in_elem.width.data == 1 else ExtSIOp(src, out_elem)
            body.append(cast)
            src = cast.results[0]
        else:
            # non-int/int dtype change is uncommon for cumsum; bail to opaque
            return _opaque_decomp("aten_cumsum", operands[:1], meta, "scan", pattern_hint="cumsum")
    masked = SelectOp(cond.results[0], src, zero.result)
    add = AddiOp(blk.args[1], masked.results[0]) if isinstance(out_elem, IntegerType) \
        else AddfOp(blk.args[1], masked.results[0])
    body += [masked, add]
    for op in body:
        blk.add_op(op)
    blk.add_op(YieldOp(add.results[0]))

    iters = [IteratorTypeAttr(IteratorType.PARALLEL)] * rank + [IteratorTypeAttr(IteratorType.REDUCTION)]
    gen = GenericOp(
        inputs=[x],
        outputs=[init.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(in_map), AffineMapAttr(out_map)],
        iterator_types=iters,
        result_types=[out_t],
    )
    ops = [zero, init, gen]
    rid = _next_region_id("scan")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("scan")
    return DecompResult(ops=ops, result=gen.results[0], region_ids=[rid], pattern_hint="cumsum")


def _arg_reduce(x, dim, keepdim, val_elem, *, is_min):
    """Combined value+index reduction over ``dim`` (the argmin/argmax kernel).

    Emits ONE two-output ``linalg.generic`` (family ``arg_reduce``): it threads a
    running (best_value, best_index) pair, updating both when a strictly-better
    element is seen (strict ``<``/``>`` keeps the first extremum, matching torch).
    Returns ``(ops, values_ssa, indices_ssa)`` with both reshaped to keepdim shape
    when requested, or ``None`` if shapes are dynamic."""
    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return None
    rank = len(in_shape)
    dim = dim % rank
    reduced_shape = [s for i, s in enumerate(in_shape) if i != dim]

    from xdsl.dialects.arith import CmpfOp, CmpiOp, ConstantOp, IndexCastOp, SelectOp
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        FloatAttr,
        IndexType,
        IntegerAttr,
        IntegerType,
    )
    from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import SplatOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineExpr, AffineMap

    is_int = isinstance(val_elem, IntegerType)
    idx_elem = IntegerType(64)
    val_t = TensorType(val_elem, reduced_shape)
    idx_t = TensorType(idx_elem, reduced_shape)

    # accumulator seeds: +inf/INT_MAX for min, -inf/INT_MIN for max; index 0
    if is_int:
        bits = val_elem.width.data
        seed = (1 << (bits - 1)) - 1 if is_min else -(1 << (bits - 1))
        vseed = ConstantOp(IntegerAttr(seed, val_elem), val_elem)
    else:
        vseed = ConstantOp(FloatAttr(float("inf") if is_min else float("-inf"), val_elem), val_elem)
    iseed = ConstantOp(IntegerAttr(0, idx_elem), idx_elem)
    vinit = SplatOp(vseed.result, [], val_t)
    iinit = SplatOp(iseed.result, [], idx_t)

    in_map = AffineMap.identity(rank)
    out_exprs = tuple(AffineExpr.dimension(d) for d in range(rank) if d != dim)
    out_map = AffineMap(rank, 0, out_exprs)

    blk = Block(arg_types=[val_elem, val_elem, idx_elem])
    idx = IndexOp(dim)
    idx_cast = IndexCastOp(idx.results[0], idx_elem)
    pred_kind = ("slt" if is_min else "sgt") if is_int else ("olt" if is_min else "ogt")
    pred = CmpiOp(blk.args[0], blk.args[1], pred_kind) if is_int \
        else CmpfOp(blk.args[0], blk.args[1], pred_kind)
    new_val = SelectOp(pred.results[0], blk.args[0], blk.args[1])
    new_idx = SelectOp(pred.results[0], idx_cast.results[0], blk.args[2])
    for op in (idx, idx_cast, pred, new_val, new_idx):
        blk.add_op(op)
    blk.add_op(YieldOp(new_val.results[0], new_idx.results[0]))

    iters = [IteratorTypeAttr(IteratorType.PARALLEL if d != dim else IteratorType.REDUCTION)
             for d in range(rank)]
    gen = GenericOp(
        inputs=[x],
        outputs=[vinit.results[0], iinit.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(in_map), AffineMapAttr(out_map), AffineMapAttr(out_map)],
        iterator_types=iters,
        result_types=[val_t, idx_t],
    )
    ops = [vseed, iseed, vinit, iinit, gen]
    values, indices = gen.results[0], gen.results[1]

    if keepdim:
        keep = list(in_shape)
        keep[dim] = 1
        vr = _emit_reshape(values, keep, val_elem)
        ir = _emit_reshape(indices, keep, idx_elem)
        if vr is None or ir is None:
            return None
        ops += vr[0]
        ops += ir[0]
        values, indices = vr[1], ir[1]

    rid = _next_region_id("arg_reduce")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("arg_reduce")
    return ops, values, indices


def _make_dim_extremum(is_min, indices_only):
    """Build a decomposition for min.dim/max.dim (values+indices) or argmin/argmax
    (indices only) over a single dim."""
    name = ("aten_argmin" if is_min else "aten_argmax") if indices_only \
        else ("aten_min_dim" if is_min else "aten_max_dim")

    def decompose(operands, meta, node_name):
        if not operands or not isinstance(operands[0].type, TensorType):
            return _opaque_decomp(name, operands[:1], meta, "arg_reduce", pattern_hint=name)
        x = operands[0]
        in_shape = _shape_of(x)
        if in_shape is None:
            return _opaque_decomp(name, operands[:1], meta, "arg_reduce", pattern_hint=name)
        dim_arg = _fx_arg(meta, 1, None)
        if dim_arg is None:
            return _opaque_decomp(name, operands[:1], meta, "arg_reduce", pattern_hint=name)
        dim = int(dim_arg)
        keepdim = bool(_fx_arg(meta, 2, False))
        # value element type: from the values output (tuple meta) or the input
        v = meta.get("val")
        vmeta = v[0] if isinstance(v, (tuple, list)) and v else v
        val_elem = _element_type_from_meta({"val": vmeta}) if not indices_only else x.type.element_type
        if indices_only:
            val_elem = x.type.element_type
        out = _arg_reduce(x, dim, keepdim, val_elem, is_min=is_min)
        if out is None:
            return _opaque_decomp(name, operands[:1], meta, "arg_reduce", pattern_hint=name)
        ops, values, indices = out
        if indices_only:
            return DecompResult(ops=ops, result=indices, pattern_hint=name)
        return DecompResult(ops=ops, result=values, results=[values, indices], pattern_hint=name)

    return decompose


def decompose_slice_tensor(operands, meta, node_name):
    """aten.slice.Tensor(input, dim, start, end, step) -> tensor.extract_slice."""
    if not operands or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_slice", operands[:1], meta, "layout", pattern_hint="slice")
    src = operands[0]
    in_shape = list(src.type.get_shape())
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    elem = _t_elem(src)
    if any(d < 0 for d in in_shape) or any(d < 0 for d in out_shape):
        return _opaque_decomp("aten_slice", operands[:1], meta, "layout", pattern_hint="slice")
    # Rank divergence (our upstream select/squeeze collapsed size-1 dims that torch kept):
    # if element counts match, the slice is layout-only here -> reconcile via a reshape to
    # the meta shape rather than emitting an opaque placeholder.
    if len(out_shape) != len(in_shape):
        n_in = 1
        for d in in_shape:
            n_in *= d
        n_out = 1
        for d in out_shape:
            n_out *= d
        if n_in == n_out:
            emitted = _emit_reshape(src, list(out_shape), elem)
            if emitted is not None:
                ops, res = emitted
                rid = _next_region_id("slice")
                for op in ops:
                    _attach_region_id(op, rid)
                    op.attributes["m2m.family"] = StringAttr("slice")
                return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint="slice")
        return _opaque_decomp("aten_slice", operands[:1], meta, "layout", pattern_hint="slice")

    rank = len(in_shape)
    dim = int(_fx_arg(meta, 1, 0) or 0)
    if dim < 0:
        dim += rank
    start = _fx_arg(meta, 2, 0)
    start = 0 if start is None else int(start)
    if start < 0:
        start += in_shape[dim]
    step = _fx_arg(meta, 4, 1)
    step = 1 if step is None else max(1, int(step))

    offsets = [0] * rank
    offsets[dim] = max(0, min(start, in_shape[dim]))
    strides = [1] * rank
    strides[dim] = step

    from xdsl.dialects.tensor import ExtractSliceOp

    op = ExtractSliceOp.from_static_parameters(src, offsets, out_shape, strides)
    rid = _next_region_id("slice")
    _attach_region_id(op, rid)
    op.attributes["m2m.family"] = StringAttr("slice")
    return DecompResult(ops=[op], result=op.results[0], region_ids=[rid], pattern_hint="slice")


# ============================================================================
#  C.2 — TorchAO quantized_decomposed + _weight_int*pack_mm
# W0.1: lowered to real compgen.quant ops (no more opaque func.call).
# ============================================================================


def _element_type_from_meta(meta: dict[str, Any]) -> Any:
    """Map the meta['val'].dtype (torch dtype) to an xDSL element type.

    Defaults to ``Float32Type`` when dtype is unavailable (e.g. in unit
    tests that stub out ``meta``).
    """
    from xdsl.dialects.builtin import (
        BFloat16Type,
        Float16Type,
        Float64Type,
        IntegerType,
    )

    val = meta.get("val")
    if val is None or not hasattr(val, "dtype"):
        return Float32Type()
    try:
        import torch
    except ImportError:
        return Float32Type()

    d = val.dtype
    if d == torch.bool:
        return IntegerType(1)
    if d == torch.float32:
        return Float32Type()
    if d == torch.float64:
        return Float64Type()
    if d == torch.float16:
        return Float16Type()
    if d == torch.bfloat16:
        return BFloat16Type()
    if d == torch.int8 or d == torch.uint8:
        return IntegerType(8)
    if d == torch.int16:
        return IntegerType(16)
    if d == torch.int32:
        return IntegerType(32)
    if d == torch.int64:
        return IntegerType(64)
    # fp8 flows through compute as f32 (xDSL has no builtin Float8 / closed AnyFloat
    # union; see _torch_dtype_to_xdsl). The fp8 scheme is recorded as a module attribute.
    _f8 = {getattr(torch, n, None) for n in ("float8_e4m3fn", "float8_e5m2",
                                             "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu")}
    if d in _f8:
        return Float32Type()
    return Float32Type()


def _fx_arg(meta: dict[str, Any], index: int, default: Any = None) -> Any:
    """Read a scalar from the forwarded FX positional args, if present.

    ``import_fx.FXImporter`` attaches ``_fx_args`` (a tuple of raw FX
    call arguments) to ``meta`` before invoking a decomposition so
    scalar kwargs (group_size, axis, quant_min, quant_max) can flow
    through without becoming SSA operands.
    """
    args = meta.get("_fx_args") or ()
    if not isinstance(args, (tuple, list)):
        return default
    if index >= len(args):
        return default
    return args[index]


def _int_attr(value: int, width: int = 64) -> Any:
    """Shorthand for ``IntegerAttr(value, IntegerType(width))``."""
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    return IntegerAttr(int(value), IntegerType(width))


def _string_attr(value: str) -> Any:
    from xdsl.dialects.builtin import StringAttr

    return StringAttr(str(value))


def _torch_dtype_tag(val: Any) -> str:
    """Best-effort string tag for the result dtype.

    Used as the ``output_dtype`` / ``input_dtype`` informative property
    on quantize / dequantize ops.
    """
    if val is None or not hasattr(val, "dtype"):
        return ""
    return str(val.dtype).replace("torch.", "")


def decompose_quantize_per_tensor(operands, meta, node_name):
    """torch.ops.quantized_decomposed.quantize_per_tensor.default.

    FX signature: ``quantize_per_tensor(input, scale, zero_point,
    quant_min, quant_max, dtype)`` -- scale/zero_point are scalar
    tensor operands in the traced graph.
    """
    from m2m.ir.quant.ops import QuantizePerTensorOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    # Require at least input + scale + zero_point as SSA operands. In
    # the real FX path these all exist; in unit tests the caller passes
    # three tensor placeholders which we accept as-is.
    if len(operands) < 3:
        raise IndexError(
            f"decompose_quantize_per_tensor expects input + scale + zero_point (3 operands), got {len(operands)}"
        )

    properties: dict[str, Any] = {}
    qmin = _fx_arg(meta, 3)
    qmax = _fx_arg(meta, 4)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)
    tag = _torch_dtype_tag(val)
    if tag:
        properties["output_dtype"] = _string_attr(tag)

    rid = _next_region_id("quantize")
    op = QuantizePerTensorOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="quantize_per_tensor",
    )


def decompose_dequantize_per_tensor(operands, meta, node_name):
    """torch.ops.quantized_decomposed.dequantize_per_tensor.default."""
    from m2m.ir.quant.ops import DequantizePerTensorOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_dequantize_per_tensor expects input + scale + zero_point, got {len(operands)}")

    properties: dict[str, Any] = {}
    qmin = _fx_arg(meta, 3)
    qmax = _fx_arg(meta, 4)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("dequantize")
    op = DequantizePerTensorOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="dequantize_per_tensor",
    )


def decompose_quantize_per_channel(operands, meta, node_name):
    """torch.ops.quantized_decomposed.quantize_per_channel.default.

    FX signature: ``(input, scales, zero_points, axis, quant_min,
    quant_max, dtype)``. ``scales`` + ``zero_points`` are 1-D tensors
    along ``axis``.
    """
    from m2m.ir.quant.ops import QuantizePerChannelOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_quantize_per_channel expects input + scales + zero_points, got {len(operands)}")

    axis = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        "axis": _int_attr(axis if isinstance(axis, int) else 0),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)
    tag = _torch_dtype_tag(val)
    if tag:
        properties["output_dtype"] = _string_attr(tag)

    rid = _next_region_id("quantize")
    op = QuantizePerChannelOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="quantize_per_channel",
    )


def decompose_dequantize_per_channel(operands, meta, node_name):
    """torch.ops.quantized_decomposed.dequantize_per_channel.default."""
    from m2m.ir.quant.ops import DequantizePerChannelOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_dequantize_per_channel expects input + scales + zero_points, got {len(operands)}")

    axis = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        "axis": _int_attr(axis if isinstance(axis, int) else 0),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("dequantize")
    op = DequantizePerChannelOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="dequantize_per_channel",
    )


def decompose_quantize_per_group(operands, meta, node_name):
    """torch.ops.quantized_decomposed.quantize_per_group_along_last_dim.default.

    FX signature: ``(input, scales, zero_points, group_size, quant_min,
    quant_max, dtype)``.
    """
    from m2m.ir.quant.ops import QuantizePerGroupOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_quantize_per_group expects input + scales + zero_points, got {len(operands)}")

    gs = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        # Default group_size = 128 (TorchAO's most common setting) when
        # the FX arg is unavailable (e.g. under test fixtures).
        "group_size": _int_attr(gs if isinstance(gs, int) and gs > 0 else 128),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("quantize")
    op = QuantizePerGroupOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="quantize_per_group",
    )


def decompose_dequantize_per_group(operands, meta, node_name):
    """torch.ops.quantized_decomposed.dequantize_per_group_along_last_dim.default."""
    from m2m.ir.quant.ops import DequantizePerGroupOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_dequantize_per_group expects input + scales + zero_points, got {len(operands)}")

    gs = _fx_arg(meta, 3)
    properties: dict[str, Any] = {
        "group_size": _int_attr(gs if isinstance(gs, int) and gs > 0 else 128),
    }
    qmin = _fx_arg(meta, 4)
    qmax = _fx_arg(meta, 5)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("dequantize")
    op = DequantizePerGroupOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="dequantize_per_group",
    )


def decompose_weight_int8pack_mm(operands, meta, node_name):
    """aten._weight_int8pack_mm.default(input, weight_int8, scales)."""
    from m2m.ir.quant.ops import WeightInt8PackMMOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) < 3:
        raise IndexError(f"decompose_weight_int8pack_mm expects input + weight + scales, got {len(operands)}")

    rid = _next_region_id("quantized_matmul")
    op = WeightInt8PackMMOp(
        operands=[operands[0], operands[1], operands[2]],
        result_types=[result_type],
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="weight_int8pack_mm",
    )


def decompose_weight_int4pack_mm(operands, meta, node_name):
    """aten._weight_int4pack_mm.default(input, weight_int4, group_size, scales_and_zeros).

    ``group_size`` is a Python scalar in the FX signature so it is
    forwarded via ``meta['_fx_args'][2]`` rather than an SSA operand;
    in the traced graph the remaining tensor operands are ``[input,
    weight, scales_and_zeros]``. Tests may supply a 4-operand list
    with a placeholder in slot 2 which is skipped.
    """
    from m2m.ir.quant.ops import WeightInt4PackMMOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    if len(operands) == 3:
        tensor_operands = [operands[0], operands[1], operands[2]]
    elif len(operands) >= 4:
        tensor_operands = [operands[0], operands[1], operands[3]]
    else:
        raise IndexError(f"decompose_weight_int4pack_mm expects >= 3 operands, got {len(operands)}")

    gs = _fx_arg(meta, 2)
    if not isinstance(gs, int) or gs <= 0:
        gs = 128  # TorchAO default
    # Snap to nearest valid group_size (32/64/128/256); the verifier
    # enforces this set. 128 is the TorchAO / Marlin default.
    valid = (32, 64, 128, 256)
    if gs not in valid:
        gs = 128

    rid = _next_region_id("quantized_matmul")
    op = WeightInt4PackMMOp(
        operands=tensor_operands,
        result_types=[result_type],
        properties={"group_size": _int_attr(gs)},
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="weight_int4pack_mm",
    )


def decompose_weight_int4pack_qm(operands, meta, node_name):
    """aten._weight_int4pack_qm.default — batched int4 packed GEMM."""
    from m2m.ir.quant.ops import WeightInt4PackQMOp

    val: Any = meta["val"]
    elem = _element_type_from_meta(meta)
    result_type = TensorType(elem, _static_shape(val.shape))

    tensor_operands = [o for o in operands if hasattr(o, "type") and isinstance(o.type, TensorType)]
    if len(tensor_operands) < 3:
        raise IndexError(f"decompose_weight_int4pack_qm expects >= 3 tensor operands, got {len(tensor_operands)}")

    gs = _fx_arg(meta, 2)
    if not isinstance(gs, int) or gs <= 0:
        gs = 128

    rid = _next_region_id("quantized_matmul")
    op = WeightInt4PackQMOp(
        operands=[tensor_operands[0], tensor_operands[1], tensor_operands[2]],
        result_types=[result_type],
        properties={"group_size": _int_attr(gs)},
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="weight_int4pack_qm",
    )


def decompose_choose_qparams_per_tensor(operands, meta, node_name):
    """aten._choose_qparams_per_tensor.default."""
    from m2m.ir.quant.ops import ChooseQParamsPerTensorOp

    # Produces (scale: f32, zero_point: i64) scalar tensors.
    scale_type = TensorType(Float32Type(), [])
    from xdsl.dialects.builtin import IntegerType

    zp_type = TensorType(IntegerType(64), [])

    if len(operands) < 1:
        raise IndexError("decompose_choose_qparams_per_tensor needs an input")

    properties: dict[str, Any] = {}
    qmin = _fx_arg(meta, 1)
    qmax = _fx_arg(meta, 2)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("choose_qparams")
    op = ChooseQParamsPerTensorOp(
        operands=[operands[0]],
        result_types=[scale_type, zp_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    # DecompResult carries a single ``result`` SSAValue; downstream
    # consumers that need both scale + zero_point read them off the
    # op directly via op.results[0] / op.results[1]. We pick
    # ``scale`` as the canonical ``result`` since that's what the FX
    # node's downstream ops most commonly consume first.
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="choose_qparams_per_tensor",
    )


def decompose_choose_qparams_per_channel(operands, meta, node_name):
    """aten._choose_qparams_per_channel.default."""
    from xdsl.dialects.builtin import IntegerType

    from m2m.ir.quant.ops import ChooseQParamsPerChannelOp

    val: Any = meta.get("val")
    # For channel qparams we produce two 1-D vectors of size C along
    # the channel axis. When the test fixture lacks a concrete shape
    # we fall back to rank-0 scalars so the op still verifies.
    shape = _static_shape(val.shape) if val is not None and hasattr(val, "shape") else []
    scale_type = TensorType(Float32Type(), shape)
    zp_type = TensorType(IntegerType(64), shape)

    if len(operands) < 1:
        raise IndexError("decompose_choose_qparams_per_channel needs an input")

    axis = _fx_arg(meta, 1)
    properties: dict[str, Any] = {
        "axis": _int_attr(axis if isinstance(axis, int) else 0),
    }
    qmin = _fx_arg(meta, 2)
    qmax = _fx_arg(meta, 3)
    if isinstance(qmin, int):
        properties["quant_min"] = _int_attr(qmin)
    if isinstance(qmax, int):
        properties["quant_max"] = _int_attr(qmax)

    rid = _next_region_id("choose_qparams")
    op = ChooseQParamsPerChannelOp(
        operands=[operands[0]],
        result_types=[scale_type, zp_type],
        properties=properties,
    )
    _attach_region_id(op, rid)
    return DecompResult(
        ops=[op],
        result=op.results[0],
        region_ids=[rid],
        pattern_hint="choose_qparams_per_channel",
    )


# ============================================================================
# Decomposition table
# ============================================================================


def _pointwise(operands, meta, scalar_build, *, family: str, out_elem=None, promote=False):
    """Scalable pointwise core: build a broadcast-aware linalg.generic for ANY pointwise
    op, tagged with an ``m2m.family`` cluster attribute so transforms can match families
    rather than unique ops. ``out_elem`` defaults to the meta result dtype.

    Returns a DecompResult, or None (caller -> opaque) on dynamic shapes / unsupported
    broadcast. The importer verify-fallback turns any invalid scalar body into opaque too.
    """
    val: Any = meta["val"]
    if isinstance(val, (tuple, list)) and val:
        val = val[0]
    out_shape = [_coerce_static_dim(d) for d in getattr(val, "shape", [])]
    if not operands or any(d < 0 for d in out_shape):
        return None
    oe = out_elem if out_elem is not None else _element_type_from_meta(meta)
    result_type = TensorType(oe, out_shape)
    maps = []
    for o in operands:
        sh = _shape_of(o)
        if sh is None:
            return None
        m = _broadcast_map(sh, out_shape)
        if m is None:
            return None
        maps.append(m)
    em = _elementwise(operands, result_type, scalar_build, input_maps=maps, promote=promote)
    if em is None:
        return None
    ops, res = em
    rid = _next_region_id(family)
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr(family)
    return DecompResult(ops=ops, result=res, region_ids=[rid], pattern_hint=family)


# Compare / select family (cmp + select), all sharing the _pointwise core ----------
_CMP_PREDICATE = {"eq": "oeq", "ne": "one", "lt": "olt", "le": "ole", "gt": "ogt", "ge": "oge"}
_CMP_PREDICATE_INT = {"eq": "eq", "ne": "ne", "lt": "slt", "le": "sle", "gt": "sgt", "ge": "sge"}


def decompose_where_self(operands, meta, node_name):
    """aten.where.self(cond, a, b) -> arith.select via linalg.generic (family: select)."""
    from xdsl.dialects.arith import SelectOp

    def build(args, oe):
        s = SelectOp(args[0], args[1], args[2])
        return [s], s.results[0]

    real = _pointwise(operands[:3], meta, build, family="select") if len(operands) >= 3 else None
    if real is not None:
        return real
    return _opaque_decomp("aten_where", operands[:3], meta, "select", pattern_hint="where")


def _make_compare(kind: str):
    """aten.{eq,ne,lt,le,gt,ge}.{Tensor,Scalar} -> arith.cmpf (family: compare, i1 out)."""
    from xdsl.dialects.builtin import IntegerType

    pred_f = _CMP_PREDICATE[kind]
    pred_i = _CMP_PREDICATE_INT[kind]

    def f(operands, meta, node_name):
        from xdsl.dialects.arith import CmpfOp, CmpiOp
        from xdsl.dialects.builtin import IntegerType as _IT

        def build(args, oe):
            if isinstance(args[0].type, _IT):  # integer compare
                c = CmpiOp(args[0], args[1], pred_i)
            else:
                c = CmpfOp(args[0], args[1], pred_f)
            return [c], c.results[0]

        ops_in = list(operands)
        if len(ops_in) == 1:  # scalar form: splat the scalar to the lhs shape
            lhs_t = ops_in[0].type
            if isinstance(lhs_t, TensorType):
                sp = _splat_scalar(_fx_arg(meta, 1, 0), TensorType(lhs_t.element_type, lhs_t.get_shape()))
                if sp is not None:
                    pre, rhs = sp
                    real = _pointwise([ops_in[0], rhs], meta, build, family="compare", out_elem=IntegerType(1))
                    if real is not None:
                        real.ops[:0] = pre
                        return real
            return _opaque_decomp("aten_compare", operands[:2], meta, "compare", pattern_hint="compare")
        real = _pointwise(ops_in[:2], meta, build, family="compare", out_elem=IntegerType(1))
        if real is not None:
            return real
        return _opaque_decomp("aten_compare", operands[:2], meta, "compare", pattern_hint="compare")

    return f


def decompose_relu(operands, meta, node_name):
    """aten.relu.default(x) -> max(x, 0) via arith.maximumf (family: minmax)."""
    from xdsl.dialects.arith import ConstantOp, MaximumfOp
    from xdsl.dialects.builtin import FloatAttr

    def build(args, oe):
        z = ConstantOp(FloatAttr(0.0, oe), oe)
        m = MaximumfOp(args[0], z.results[0])
        return [z, m], m.results[0]

    real = _pointwise(operands[:1], meta, build, family="minmax", promote=True)
    if real is not None:
        return real
    return _opaque_decomp("aten_relu", operands[:1], meta, "elementwise", pattern_hint="relu")


def decompose_clamp(operands, meta, node_name):
    """aten.clamp.default(x, min?, max?) -> minimumf(maximumf(x,min),max) (family: minmax)."""
    from xdsl.dialects.arith import ConstantOp, MaximumfOp, MinimumfOp
    from xdsl.dialects.builtin import FloatAttr

    lo = _fx_arg(meta, 1, None)
    hi = _fx_arg(meta, 2, None)

    def build(args, oe):
        cur = args[0]
        ops = []
        if lo is not None:
            c = ConstantOp(FloatAttr(float(lo), oe), oe)
            mx = MaximumfOp(cur, c.results[0])
            ops += [c, mx]
            cur = mx.results[0]
        if hi is not None:
            c = ConstantOp(FloatAttr(float(hi), oe), oe)
            mn = MinimumfOp(cur, c.results[0])
            ops += [c, mn]
            cur = mn.results[0]
        return ops, cur

    real = _pointwise(operands[:1], meta, build, family="minmax", promote=True)
    if real is not None:
        return real
    return _opaque_decomp("aten_clamp", operands[:1], meta, "elementwise", pattern_hint="clamp")


def _make_minmax(op_cls_name: str, hint: str):
    """aten.{maximum,minimum}(a, b) -> arith.{maximumf,minimumf} (family: minmax)."""

    def f(operands, meta, node_name):
        int_name = {"MaximumfOp": "MaxSIOp", "MinimumfOp": "MinSIOp"}.get(op_cls_name, op_cls_name)
        build = _arith_build2(op_cls_name, int_name)
        real = _pointwise(operands[:2], meta, build, family="minmax", promote=True)
        if real is not None:
            return real
        return _opaque_decomp(f"aten_{hint}", operands[:2], meta, "elementwise", pattern_hint=hint)

    return f


def decompose_logical_not(operands, meta, node_name):
    """aten.logical_not.default(x) -> x XOR true (family: logical, i1 out)."""
    from xdsl.dialects.arith import ConstantOp, XOrIOp
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    i1 = IntegerType(1)

    def build(args, oe):
        true = ConstantOp(IntegerAttr(1, i1), i1)
        x = XOrIOp(args[0], true.results[0])
        return [true, x], x.results[0]

    real = _pointwise(operands[:1], meta, build, family="logical", out_elem=i1)
    if real is not None:
        return real
    return _opaque_decomp("aten_logical_not", operands[:1], meta, "logical", pattern_hint="logical_not")


def decompose_bitwise_not(operands, meta, node_name):
    """aten.bitwise_not.default(x) -> x XOR all-ones (family bitwise). For i1 this is
    boolean NOT; for wider integers it flips every bit (~x)."""
    from xdsl.dialects.arith import ConstantOp, XOrIOp
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    def build(args, oe):
        if not isinstance(oe, IntegerType):
            return None  # bitwise_not is integer-only; bail to opaque
        allones = 1 if oe.width.data == 1 else -1  # all-ones bit pattern for this width
        c = ConstantOp(IntegerAttr(allones, oe), oe)
        x = XOrIOp(args[0], c.results[0])
        return [c, x], x.results[0]

    real = _pointwise(operands[:1], meta, build, family="bitwise")
    if real is not None:
        return real
    return _opaque_decomp("aten_bitwise_not", operands[:1], meta, "bitwise", pattern_hint="bitwise_not")


def decompose_repeat(operands, meta, node_name):
    """aten.repeat(input, repeats) -> tile each dim by ``repeats`` (family ``tile``).

    repeats has length >= input rank; extra leading entries prepend new tiled dims.
    Lowered as a gather linalg.generic: out[coords] = input[coords_i % in_dim_i], with
    the input dims right-aligned to the output. An all-ones repeat (no growth) is the
    identity and forwards the operand unchanged."""
    if not operands or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_repeat", operands[:1], meta, "tile", pattern_hint="repeat")
    src = operands[0]
    in_shape = _shape_of(src)
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    elem = src.type.element_type
    if in_shape is None or any(d < 0 for d in in_shape) or any(d < 0 for d in out_shape):
        return _opaque_decomp("aten_repeat", operands[:1], meta, "tile", pattern_hint="repeat")
    if out_shape == list(in_shape):
        return DecompResult(ops=[], result=src, pattern_hint="repeat")  # identity tile
    R = len(in_shape)
    out_rank = len(out_shape)
    if out_rank < R:
        return _opaque_decomp("aten_repeat", operands[:1], meta, "tile", pattern_hint="repeat")
    offset = out_rank - R

    from xdsl.dialects.arith import ConstantOp, RemUIOp
    from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerAttr
    from xdsl.dialects.linalg import GenericOp, IndexOp, IteratorType, IteratorTypeAttr, YieldOp
    from xdsl.dialects.tensor import EmptyOp, ExtractOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    out_t = TensorType(elem, out_shape)
    empty = EmptyOp([], out_t)
    blk = Block(arg_types=[elem])
    coords = []
    for i in range(R):
        out_dim = offset + i
        ix = IndexOp(out_dim)
        blk.add_op(ix)
        if out_shape[out_dim] != in_shape[i]:   # tiled: wrap the index modulo the input extent
            c = ConstantOp(IntegerAttr(in_shape[i], IndexType()), IndexType())
            rem = RemUIOp(ix.results[0], c.results[0])
            blk.add_op(c)
            blk.add_op(rem)
            coords.append(rem.results[0])
        else:
            coords.append(ix.results[0])
    ext = ExtractOp(src, coords, elem)
    blk.add_op(ext)
    blk.add_op(YieldOp(ext.results[0]))
    gen = GenericOp(
        inputs=[],
        outputs=[empty.results[0]],
        body=Region(blk),
        indexing_maps=[AffineMapAttr(AffineMap.identity(out_rank))],
        iterator_types=[IteratorTypeAttr(IteratorType.PARALLEL)] * out_rank,
        result_types=[out_t],
    )
    ops = [empty, gen]
    rid = _next_region_id("tile")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("tile")
    return DecompResult(ops=ops, result=gen.results[0], region_ids=[rid], pattern_hint="repeat")


def _make_math_unary(op_cls_name: str, hint: str):
    """Build a decomposition that lowers a pointwise unary op to linalg.generic{math.X}
    (the exact form torch-mlir produces), with an opaque fallback."""

    def f(operands, meta, node_name):
        import xdsl.dialects.math as _M

        real = _unary_elementwise(operands, meta, hint, _un_build(getattr(_M, op_cls_name)))
        if real is not None:
            return real
        return _opaque_decomp(f"aten_{hint}", operands[:1], meta, "elementwise", pattern_hint=hint)

    return f


DECOMPOSITION_TABLE: dict[str, DecompFn] = {
    # --- pointwise unary math (linalg.generic{math.*}) ---
    "aten.exp.default": _make_math_unary("ExpOp", "exp"),
    "aten.sqrt.default": _make_math_unary("SqrtOp", "sqrt"),
    "aten.tanh.default": _make_math_unary("TanhOp", "tanh"),
    "aten.abs.default": _make_math_unary("AbsFOp", "abs"),
    "aten.floor.default": _make_math_unary("FloorOp", "floor"),
    "aten.ceil.default": _make_math_unary("CeilOp", "ceil"),
    "aten.log.default": _make_math_unary("LogOp", "log"),
    # --- pre-wave-6 entries (kept) ---
    "aten.addmm.default": decompose_addmm,
    "aten.linear.default": decompose_linear,
    "aten.gelu.default": decompose_gelu,
    "aten.add.Tensor": decompose_add_tensor,
    "aten.mul.Tensor": decompose_mul_tensor,
    # scalar overloads route to the same emitters (scalar is splatted)
    "aten.add.Scalar": decompose_add_tensor,
    "aten.mul.Scalar": decompose_mul_tensor,
    "aten.sub.Scalar": decompose_sub_tensor,
    "aten.div.Scalar": decompose_div_tensor,
    "aten.mm.default": decompose_mm,
    "aten.permute.default": decompose_permute,
    "aten.t.default": decompose_transpose,
    # --- wave 6: real-model coverage ---
    # compute / semantic
    "aten.bmm.default": decompose_bmm,
    "aten.native_layer_norm.default": decompose_native_layer_norm,
    "aten.layer_norm.default": decompose_native_layer_norm,
    "aten._softmax.default": decompose_softmax,
    "aten.softmax.int": decompose_softmax,
    "aten.rsqrt.default": decompose_rsqrt,
    "aten.pow.Tensor_Scalar": decompose_pow_tensor_scalar,
    "aten.mean.dim": decompose_mean_dim,
    "aten.mean.default": decompose_mean_dim,  # full reduction (dims=None -> all dims, scalar)
    "aten.convolution.default": decompose_convolution,
    # ``nn.Conv2d`` lowers to ``aten.conv2d.default`` after FX export
    # (not ``aten.convolution.default``). Same shape contract; same
    # decomposition function consumes both.
    "aten.conv2d.default": decompose_convolution,
    "aten.embedding.default": decompose_embedding,
    "aten.sigmoid.default": decompose_sigmoid,
    "aten.neg.default": decompose_neg,
    "aten.silu.default": decompose_silu,
    "aten.sub.Tensor": decompose_sub_tensor,
    "aten.div.Tensor": decompose_div_tensor,
    # layout / structural
    "aten.view.default": decompose_view,
    "aten.unsqueeze.default": decompose_unsqueeze,
    "aten.squeeze.default": decompose_squeeze,
    "aten.squeeze.dim": decompose_squeeze,
    "aten.squeeze.dims": decompose_squeeze,
    "aten.expand.default": decompose_expand,
    "aten.cat.default": decompose_cat,
    "aten.split_with_sizes.default": decompose_split_with_sizes,
    "aten.clone.default": decompose_clone,
    # production-readiness fill-ins:
    "aten.contiguous.default": decompose_contiguous,
    "aten.transpose.int": decompose_transpose_int,
    "aten.transpose.default": decompose_transpose_int,
    "aten.matmul.default": decompose_matmul,
    "aten.slice.Tensor": decompose_slice_tensor,
    # --- wave 6 C.2: TorchAO quantized_decomposed + packed GEMMs ---
    "torch.ops.quantized_decomposed.quantize_per_tensor.default": decompose_quantize_per_tensor,
    "torch.ops.quantized_decomposed.dequantize_per_tensor.default": decompose_dequantize_per_tensor,
    "torch.ops.quantized_decomposed.quantize_per_channel.default": decompose_quantize_per_channel,
    "torch.ops.quantized_decomposed.dequantize_per_channel.default": decompose_dequantize_per_channel,
    "torch.ops.quantized_decomposed.quantize_per_group_along_last_dim.default": decompose_quantize_per_group,
    "torch.ops.quantized_decomposed.dequantize_per_group_along_last_dim.default": decompose_dequantize_per_group,
    "aten._weight_int8pack_mm.default": decompose_weight_int8pack_mm,
    "aten._weight_int4pack_mm.default": decompose_weight_int4pack_mm,
    "aten._weight_int4pack_qm.default": decompose_weight_int4pack_qm,
    "aten._choose_qparams_per_tensor.default": decompose_choose_qparams_per_tensor,
    "aten._choose_qparams_per_channel.default": decompose_choose_qparams_per_channel,
    # Short-form aliases some PyTorch versions emit
    "quantized_decomposed.quantize_per_tensor.default": decompose_quantize_per_tensor,
    "quantized_decomposed.dequantize_per_tensor.default": decompose_dequantize_per_tensor,
    "quantized_decomposed.quantize_per_channel.default": decompose_quantize_per_channel,
    "quantized_decomposed.dequantize_per_channel.default": decompose_dequantize_per_channel,
    "quantized_decomposed.quantize_per_group_along_last_dim.default": decompose_quantize_per_group,
    "quantized_decomposed.dequantize_per_group_along_last_dim.default": decompose_dequantize_per_group,
    # --- wave 7: TinyLlama opaque-tail closure (10 new families) ---
    "aten._to_copy.default": decompose_to_copy,
    "aten.where.self": decompose_where_self,
    "aten.scalar_tensor.default": decompose_scalar_tensor,
    "aten.full_like.default": decompose_full_like,
    "aten.full.default": decompose_full,
    "aten.arange.start_step": decompose_arange,
    "aten.arange.default": decompose_arange,
    "aten.logical_not.default": decompose_logical_not,
    "aten.bitwise_and.Tensor": decompose_bitwise_and,
    "aten.bitwise_not.default": decompose_bitwise_not,
    "aten.repeat.default": decompose_repeat,
    "aten.any.dim": decompose_any_real,
    "aten.index.Tensor": decompose_index_tensor,
    # Comparisons + trig + scan (RoPE / mask construction)
    "aten.eq.Scalar": decompose_compare,
    "aten.eq.Tensor": decompose_compare,
    "aten.ne.Scalar": decompose_compare,
    "aten.ne.Tensor": decompose_compare,
    "aten.le.Scalar": decompose_compare,
    "aten.le.Tensor": decompose_compare,
    "aten.lt.Scalar": decompose_compare,
    "aten.lt.Tensor": decompose_compare,
    "aten.gt.Scalar": decompose_compare,
    "aten.gt.Tensor": decompose_compare,
    "aten.ge.Scalar": decompose_compare,
    "aten.ge.Tensor": decompose_compare,
    "aten.cos.default": decompose_cos,
    "aten.sin.default": decompose_sin,
    "aten.cumsum.default": decompose_cumsum,
}


# Compare / select / minmax family registrations (parameterized; one emitter per family).
DECOMPOSITION_TABLE.update(
    {
        "aten.relu.default": decompose_relu,
        "aten.clamp.default": decompose_clamp,
        "aten.clamp.Tensor": decompose_clamp,
        "aten.maximum.default": _make_minmax("MaximumfOp", "maximum"),
        "aten.minimum.default": _make_minmax("MinimumfOp", "minimum"),
        "aten.where.self": decompose_where_self,
        "aten.logical_not.default": decompose_logical_not,
        # identity aliases
        "aten.alias.default": _identity_decomp,
        "aten.detach.default": _identity_decomp,
        "aten.lift_fresh_copy.default": _identity_decomp,
        # empty / fill
        "aten.empty.memory_format": decompose_empty,
        "aten.empty_like.default": decompose_empty,
        "aten.new_empty.default": decompose_empty,
        "aten.scalar_tensor.default": _make_fill(0),
        "aten.full.default": _make_fill(1),
        "aten.full_like.default": _make_fill(1),
        # copy / pow / minmax extras
        "aten.copy.default": decompose_copy,
        "aten.pow.Tensor_Tensor": decompose_pow_tensor_tensor,
        "aten.pow.Scalar": decompose_pow_scalar,
        "aten.slice_scatter.default": decompose_slice_scatter,
        "aten.any.dim": decompose_any_real,
        "aten.any.default": decompose_any_real,
        "aten.any.dims": decompose_any_real,
        "aten.reshape.default": decompose_view,
        "aten._unsafe_view.default": decompose_view,
        "aten.sum.dim_IntList": decompose_sum_dim,
        "aten.reciprocal.default": decompose_reciprocal,
        "aten.select.int": decompose_select_int,
        "aten.min.other": _make_minmax("MinimumfOp", "minimum"),
        "aten.max.other": _make_minmax("MaximumfOp", "maximum"),
        "aten.min.dim": _make_dim_extremum(is_min=True, indices_only=False),
        "aten.max.dim": _make_dim_extremum(is_min=False, indices_only=False),
        "aten.argmin.default": _make_dim_extremum(is_min=True, indices_only=True),
        "aten.argmax.default": _make_dim_extremum(is_min=False, indices_only=True),
        "aten.bucketize.Tensor": decompose_bucketize,
        "aten.searchsorted.Tensor": decompose_bucketize,
        "aten.index_put.default": decompose_index_put,
        "aten.index_put_.default": decompose_index_put,
        "aten._index_put_impl.default": decompose_index_put,
        # true-quantized (non-QDQ) integer ops
        "aten._int_mm.default": decompose_int_mm,
        "aten.amin.default": _make_amin_amax(is_min=True),
        "aten.amax.default": _make_amin_amax(is_min=False),
        "aten.round.default": decompose_round,
    }
)
for _k in ("eq", "ne", "lt", "le", "gt", "ge"):
    DECOMPOSITION_TABLE[f"aten.{_k}.Tensor"] = _make_compare(_k)
    DECOMPOSITION_TABLE[f"aten.{_k}.Scalar"] = _make_compare(_k)


__all__ = [
    "DECOMPOSITION_TABLE",
    "DecompFn",
    "DecompResult",
    "reset_region_counters",
]
