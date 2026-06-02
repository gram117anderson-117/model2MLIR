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
    if not in_shape or not out_shape or any(d < 0 for d in (*in_shape, *out_shape)):
        return None
    if in_shape == out_shape:
        return [], source  # identity reshape

    from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp

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
    from xdsl.dialects.arith import ExtFOp, FPToSIOp, SIToFPOp, TruncFOp
    from xdsl.dialects.builtin import AnyFloat, IntegerType

    if isinstance(src, AnyFloat) and isinstance(dst, AnyFloat):
        op = TruncFOp(x, dst) if dst.bitwidth < src.bitwidth else ExtFOp(x, dst)
    elif isinstance(src, IntegerType) and isinstance(dst, AnyFloat):
        op = SIToFPOp(x, dst)
    elif isinstance(src, AnyFloat) and isinstance(dst, IntegerType):
        op = FPToSIOp(x, dst)
    else:
        return [], x  # can't bridge; leave as-is (importer verify-fallback handles it)
    return [op], op.results[0]


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
    """scalar_build for a dtype cast: float trunc/ext or int<->float, picked by types."""
    from xdsl.dialects.arith import ExtFOp, FPToSIOp, SIToFPOp, TruncFOp
    from xdsl.dialects.builtin import AnyFloat, IntegerType

    def build(args, out_elem):
        x = args[0]
        src = x.type
        dst = target_elem
        if isinstance(src, AnyFloat) and isinstance(dst, AnyFloat):
            op = TruncFOp(x, dst) if dst.bitwidth < src.bitwidth else ExtFOp(x, dst)
        elif isinstance(src, IntegerType) and isinstance(dst, AnyFloat):
            op = SIToFPOp(x, dst)
        elif isinstance(src, AnyFloat) and isinstance(dst, IntegerType):
            op = FPToSIOp(x, dst)
        else:
            op = ExtFOp(x, dst)  # best-effort
        return [op], op.results[0]

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
        sp = _splat_scalar(_fx_arg(meta, 1, 0), result_type)
        if sp is None:
            return None
        pre, rhs = sp[0], sp[1]
        lhs = operands[0]
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

    real = _binary_elementwise(operands, meta, "add", _bin_build(AddfOp))
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

    real = _binary_elementwise(operands, meta, "mul", _bin_build(MulfOp))
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


def decompose_native_layer_norm(operands, meta, node_name):
    """aten.native_layer_norm.default(input, normalized_shape, weight, bias, eps).

    Decomposes to mean/var/normalize/scale/shift (family: layer_norm). Returns the
    normalized output as the primary result (getitem(_,0)); mean/rstd aux outputs are
    folded away by the importer.
    """
    from xdsl.dialects.arith import AddfOp, ConstantOp, MulfOp, SubfOp
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.dialects.math import RsqrtOp

    x = operands[0]
    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_native_layer_norm", operands, meta, "layer_norm", pattern_hint="layer_norm")
    elem = _t_elem(x)
    norm_shape = _fx_arg(meta, 1, None)
    k = len(norm_shape) if norm_shape is not None else 1
    eps = _fx_arg(meta, 4, 1e-5)
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

    from xdsl.dialects.arith import DivfOp

    mean = reduce_mean(x)
    if mean is None:
        return _opaque_decomp("aten_native_layer_norm", operands, meta, "layer_norm", pattern_hint="layer_norm")
    cen = _elementwise([x, mean], rt, _bin_build(SubfOp), input_maps=[id_map, keep_map])
    ops += cen[0]
    sq = _elementwise([cen[1], cen[1]], rt, _bin_build(MulfOp))
    ops += sq[0]
    var = reduce_mean(sq[1])
    if var is None:
        return _opaque_decomp("aten_native_layer_norm", operands, meta, "layer_norm", pattern_hint="layer_norm")

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
    if len(operands) >= 2 and isinstance(operands[1].type, TensorType):
        w = operands[1]
        wmap = _broadcast_map(_shape_of(w) or [], in_shape)
        if wmap is not None:
            sc = _elementwise([res, w], rt, _bin_build(MulfOp), input_maps=[id_map, wmap])
            if sc is not None:
                ops += sc[0]
                res = sc[1]
    if len(operands) >= 3 and isinstance(operands[2].type, TensorType):
        b = operands[2]
        bmap = _broadcast_map(_shape_of(b) or [], in_shape)
        if bmap is not None:
            sh = _elementwise([res, b], rt, _bin_build(AddfOp), input_maps=[id_map, bmap])
            if sh is not None:
                ops += sh[0]
                res = sh[1]

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

    dims = sorted(d % len(in_shape) for d in dims)
    reduced_shape = [s for i, s in enumerate(in_shape) if i not in dims]
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


def decompose_softmax(operands, meta, node_name):
    """aten._softmax(input, dim, _) -> max/sub/exp/sum/div (family: softmax)."""
    from xdsl.dialects.arith import AddfOp, DivfOp, MaximumfOp, SubfOp
    from xdsl.dialects.math import ExpOp

    x = operands[0]
    in_shape = _shape_of(x)
    if in_shape is None or any(d < 0 for d in in_shape):
        return _opaque_decomp("aten_softmax", operands, meta, "softmax", pattern_hint="softmax")
    elem = _t_elem(x)
    dim = int(_fx_arg(meta, 1, -1)) % len(in_shape)
    keep = list(in_shape)
    keep[dim] = 1
    rt = TensorType(elem, in_shape)
    id_map = _broadcast_map(in_shape, in_shape)
    keep_map = _broadcast_map(keep, in_shape)

    ops, mx, rsh = _reduce(x, in_shape, [dim], float("-inf"), MaximumfOp, elem)
    mx = _keepdim_reshape(ops, mx, rsh, keep, elem)
    sub = _elementwise([x, mx], rt, _bin_build(SubfOp), input_maps=[id_map, keep_map]) if mx is not None else None
    if sub is None:
        return _opaque_decomp("aten_softmax", operands, meta, "softmax", pattern_hint="softmax")
    ops += sub[0]
    ex = _elementwise([sub[1]], rt, _un_build(ExpOp))
    ops += ex[0]
    o2, s, rsh2 = _reduce(ex[1], in_shape, [dim], 0.0, AddfOp, elem)
    ops += o2
    s = _keepdim_reshape(ops, s, rsh2, keep, elem)
    div = _elementwise([ex[1], s], rt, _bin_build(DivfOp), input_maps=[id_map, keep_map]) if s is not None else None
    if div is None:
        return _opaque_decomp("aten_softmax", operands, meta, "softmax", pattern_hint="softmax")
    ops += div[0]
    rid = _next_region_id("softmax")
    for op in ops:
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("softmax")
    return DecompResult(ops=ops, result=div[1], region_ids=[rid], pattern_hint="softmax")


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


def decompose_embedding(operands, meta, node_name):
    """aten.embedding.default(weight, indices, ...) -> gather-style op (MVP: opaque)."""
    # weight + indices are the first two operands; scalar-flag kwargs beyond that.
    tensor_operands = operands[:2] if len(operands) >= 2 else operands
    return _opaque_decomp(
        "aten_embedding",
        tensor_operands,
        meta,
        "embedding",
        pattern_hint="embedding_lookup",
    )


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

    real = _binary_elementwise(operands, meta, "sub", _bin_build(SubfOp))
    if real is not None:
        return real
    return _opaque_decomp("aten_sub", operands[:2], meta, "elementwise", pattern_hint="sub")


def decompose_div_tensor(operands, meta, node_name):
    """aten.div.Tensor(a, b) -> arith.divf via linalg.generic."""
    from xdsl.dialects.arith import DivfOp

    real = _binary_elementwise(operands, meta, "div", _bin_build(DivfOp))
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
    # reshape preserves dtype (source elem must == result elem), AND the result must
    # match the meta dtype the downstream consumers expect. Only emit when source SSA
    # dtype and meta dtype agree; otherwise fall back to opaque (typed from meta).
    if not isinstance(src_type, TensorType) or src_type.element_type != meta_elem:
        return _opaque_decomp(f"aten_{prefix}", operands[:1], meta, "layout", pattern_hint=hint)
    out_shape = _static_shape(val.shape)
    emitted = _emit_reshape(operands[0], out_shape, meta_elem)
    if emitted is None:
        return _opaque_decomp(f"aten_{prefix}", operands[:1], meta, "layout", pattern_hint=hint)
    ops, result = emitted
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
        and all(isinstance(o.type, TensorType) and o.type.element_type == elem for o in operands)
    ):
        dim = int(_fx_arg(meta, 1, 0) or 0)
        if dim < 0:
            dim += len(out_shape)
        from xdsl.dialects.tensor import ConcatOp

        op = ConcatOp(inputs=list(operands), dim=dim, result_type=TensorType(elem, out_shape))
        rid = _next_region_id("cat")
        _attach_region_id(op, rid)
        op.attributes["m2m.family"] = StringAttr("concat")
        return DecompResult(ops=[op], result=op.results[0], region_ids=[rid], pattern_hint="cat")
    return _opaque_decomp("aten_cat", operands, meta, "layout", pattern_hint="cat")


def decompose_split_with_sizes(operands, meta, node_name):
    """aten.split_with_sizes.default(input, split_sizes, dim?).

    Returns a list of tensors; for MVP we emit a single opaque call
    with pattern_hint='split' — the FX graph's getitem ops disambiguate
    which chunk each downstream consumer needs.
    """
    return _opaque_decomp(
        "aten_split_with_sizes",
        operands[:1],
        meta,
        "layout",
        pattern_hint="split",
    )


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
    """aten.arange.start_step(start, end, step, ...) — index generator."""
    return _opaque_decomp("aten_arange", [], meta, "arange", pattern_hint="arange")


def decompose_logical_not(operands, meta, node_name):
    """aten.logical_not.default — pointwise boolean NOT."""
    return _opaque_decomp("aten_logical_not", operands[:1], meta, "logical", pattern_hint="logical_not")


def decompose_bitwise_and(operands, meta, node_name):
    """aten.bitwise_and.Tensor — pointwise bitwise AND."""
    return _opaque_decomp("aten_bitwise_and", operands[:2], meta, "bitwise", pattern_hint="bitwise_and")


def decompose_any_dim(operands, meta, node_name):
    """aten.any.dim(input, dim, keepdim) — boolean OR reduction along dim."""
    return _opaque_decomp("aten_any_dim", operands[:1], meta, "bool_reduce", pattern_hint="bool_reduce")


def decompose_index_tensor(operands, meta, node_name):
    """aten.index.Tensor — multi-dim gather. Operand 0 is source; index
    tensors arrive via meta['_fx_args'][1] as a list."""
    # Forward only the source SSA value; the index list is a Python list of
    # tensors that the kernel will receive via the contract metadata.
    return _opaque_decomp("aten_index", operands[:1], meta, "gather", pattern_hint="gather")


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
    """aten.cumsum.default — prefix sum along dim."""
    return _opaque_decomp("aten_cumsum", operands[:1], meta, "scan", pattern_hint="cumsum")


def decompose_slice_tensor(operands, meta, node_name):
    """aten.slice.Tensor(input, dim, start, end, step) -> tensor.extract_slice."""
    if not operands or not isinstance(operands[0].type, TensorType):
        return _opaque_decomp("aten_slice", operands[:1], meta, "layout", pattern_hint="slice")
    src = operands[0]
    in_shape = list(src.type.get_shape())
    val: Any = meta["val"]
    out_shape = _static_shape(getattr(val, "shape", []))
    if any(d < 0 for d in in_shape) or any(d < 0 for d in out_shape) or len(out_shape) != len(in_shape):
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
    if hasattr(torch, "float8_e4m3fn") and d == torch.float8_e4m3fn:
        from m2m.ir.types import Float8E4M3FNType

        return Float8E4M3FNType()
    if hasattr(torch, "float8_e5m2") and d == torch.float8_e5m2:
        from m2m.ir.types import Float8E5M2Type

        return Float8E5M2Type()
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

    pred = _CMP_PREDICATE[kind]

    def f(operands, meta, node_name):
        from xdsl.dialects.arith import CmpfOp

        def build(args, oe):
            c = CmpfOp(args[0], args[1], pred)
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
        import xdsl.dialects.arith as _A

        cls = getattr(_A, op_cls_name)

        def build(args, oe):
            o = cls(args[0], args[1])
            return [o], o.results[0]

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
    "aten.any.dim": decompose_any_dim,
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
        "aten.min.other": _make_minmax("MinimumfOp", "minimum"),
        "aten.max.other": _make_minmax("MaximumfOp", "maximum"),
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
