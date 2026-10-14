"""Kernel registry: custom-kernel ops (Triton today) lowered as black-box ``func.call``s.

Sibling of ``decompositions.py``. ``DECOMPOSITION_TABLE`` maps an aten target to an emitter that
produces real linalg/tensor/arith ops; ``KERNEL_REGISTRY`` maps a custom-op target -- e.g. the
``@triton_op`` ``triton_exp::fused_mlp2`` -- to a ``KernelSpec``, and ``emit_kernel_call`` (the
kernel analogue of a ``DecompFn``) emits ONE typed ``func.call`` for it: tensor inputs as SSA
operands, outputs as SSA results, non-tensor arguments as an attribute. The call targets a
private ``func.func`` declaration that a downstream lowering (merlin) binds to the real kernel
implementation, looked up by ``prov.kernel`` in its own kernel table.

Why not just the generic opaque fallback: that path keys the callee on the node's type
signature (``@ns_op_default_1`` when the shape changes), drops every non-tensor argument
(alpha, block sizes, defaults), keeps only the first output of a multi-output op, and is
counted as an unsupported (opaque) op. A registered kernel instead gets a stable identity
(``prov.kernel``), its full schema-ordered argument binding, all of its results, the normal
provenance/taxonomy tags (``prov.family = "kernel"``), and does NOT count as opaque.

The emitted contract is deliberately small -- only what a consumer needs to bind the call:
    call: ``prov.kernel`` (identity), ``prov.kernel_operands`` (schema name of each SSA
          operand), ``prov.kernel_args`` (non-tensor args by name; omitted when there are none)
    decl: ``prov.kernel``
Where a kernel's implementation lives, and in what language, is the consumer's table's business.

Invariants:
    - The table is the single global source of truth: a custom op is a kernel iff its
      qualname (``ns::name``, all overloads) is a key of ``KERNEL_REGISTRY``.
    - Operands follow the op SCHEMA's argument order (kwargs + omitted defaults normalized),
      never FX's mixed positional/keyword order; ``prov.kernel_operands`` names each one.
    - Results follow the op's return order; ``getitem(node, i)`` resolves to result ``i``.
    - A registered kernel that cannot be represented faithfully (in-place/mutating schema,
      an unresolved tensor operand) raises ``KernelLoweringError`` -- the importer reports
      an error instead of emitting silently-wrong IR.

To register a kernel: add a ``KernelSpec`` to ``_BUILTIN_KERNELS`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    TensorType,
    UnitAttr,
    f64,
    i64,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import Attribute, SSAValue

from m2m.ir.decompositions import DecompResult, _attach_region_id, _next_region_id


class KernelLoweringError(RuntimeError):
    """A registered kernel node can't be represented as a black-box call (see invariants)."""


@dataclass(frozen=True)
class KernelSpec:
    """One registered kernel.

    Attributes:
        qualname: the torch custom-op name, ``"namespace::name"`` (matches every overload).
        description: free-form note (what it computes, which workload uses it).
    """

    qualname: str
    description: str = ""

    @property
    def callee(self) -> str:
        """Callee symbol: the qualname with ``::`` -> ``__`` (``triton_exp__fused_mlp2``). A
        shape-specialized duplicate gets a ``_<n>`` suffix, so match on ``prov.kernel``, never this."""
        return self.qualname.replace("::", "__")


# ============================================================================
# The global kernel table. Every kernel m2m treats as a black box is listed here.
# ============================================================================

_BUILTIN_KERNELS: tuple[KernelSpec, ...] = (
    KernelSpec(
        "triton_exp::fused_mlp2",
        description="linear+relu+linear+relu in one Triton kernel (workloads/mlp_basic_triton)",
    ),
)

KERNEL_REGISTRY: dict[str, KernelSpec] = {k.qualname: k for k in _BUILTIN_KERNELS}


# ============================================================================
# Lookup
# ============================================================================


def _qualname_of(target: Any) -> str | None:
    """``ns::name`` for an OpOverload / OpOverloadPacket, or for an FX target string
    (``"ns.name.overload"`` / ``"ns::name"``). None for anything else (getitem, HOPs, ...)."""
    if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return target._schema.name if isinstance(target, torch._ops.OpOverload) else target._qualified_op_name
    if isinstance(target, str):
        if "::" in target:
            return target.split(".")[0]
        parts = target.split(".")
        if len(parts) >= 2 and all(parts):
            return f"{parts[0]}::{parts[1]}"
    return None


def lookup_kernel(target: Any) -> KernelSpec | None:
    """The registered ``KernelSpec`` for an FX node target, or None."""
    q = _qualname_of(target)
    return KERNEL_REGISTRY.get(q) if q else None


def registered_kernel_targets(graph: Any) -> set[str]:
    """``str(node.target)`` of every call node in ``graph`` that is a registered kernel. Used
    to mark them supported for the unsupported-op loop and to skip torch-mlir (which can't
    legalize custom ops) when a graph contains any."""
    return {str(n.target) for n in graph.nodes
            if n.op == "call_function" and lookup_kernel(n.target) is not None}


def is_triton_op(target: Any) -> bool:
    """True if ``target`` was registered via ``torch.library.triton_op`` (registered in this
    table or not) -- used to flag unregistered Triton ops that fall back to opaque."""
    q = _qualname_of(target)
    try:
        from torch._library.triton import triton_ops_to_kernels
    except Exception:  # noqa: BLE001 - torch without triton_op support
        return False
    return q is not None and q in triton_ops_to_kernels


# ============================================================================
# Emission
# ============================================================================


def _scalar_attr(value: Any) -> Attribute:
    """Non-tensor argument -> attribute. bool before int (bool is an int subclass); ``None``
    (an absent optional) -> unit; anything unrecognized is kept as its ``str`` so the value
    is never silently dropped."""
    if value is None:
        return UnitAttr()
    if isinstance(value, bool):
        return IntegerAttr(int(value), IntegerType(1))
    if isinstance(value, int):
        return IntegerAttr(value, i64)
    if isinstance(value, float):
        return FloatAttr(value, f64)
    if isinstance(value, str):
        return StringAttr(value)
    if isinstance(value, (list, tuple)):
        return ArrayAttr([_scalar_attr(v) for v in value])
    if isinstance(value, torch.dtype):
        return StringAttr(str(value).replace("torch.", ""))
    return StringAttr(str(value))


def _is_fx_node(x: Any) -> bool:
    return hasattr(x, "op") and hasattr(x, "name") and hasattr(x, "meta")


def emit_kernel_call(
    spec: KernelSpec,
    node: Any,
    value_map: dict[str, SSAValue],
    result_types: list[TensorType],
) -> DecompResult:
    """Emit the black-box ``func.call`` for a registered kernel node.

    ``result_types`` are the node's output tensor types (from ``meta["val"]``, in return
    order). The binding is normalized against the op schema so operands/attributes don't
    depend on how the model spelled the call (positional vs keyword, defaults omitted).
    Returns a ``DecompResult`` whose ``ops`` is the single call, ``result`` its first result
    and ``results`` all of them. The matching declaration is built by ``kernel_decl``.
    """
    target = node.target
    schema = target._schema
    mutated = [a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write]
    if mutated:
        # In-place / destination-passing kernels reach the graph wrapped in
        # auto_functionalized_v2; representing them needs the tied-result form (TODO).
        raise KernelLoweringError(
            f"{spec.qualname} mutates {mutated}; in-place kernels are not supported yet")

    from torch.fx.operator_schemas import normalize_function

    bound = normalize_function(target, tuple(node.args), dict(node.kwargs),
                               normalize_to_only_use_kwargs=True)
    kwargs = dict(bound.kwargs) if bound is not None else dict(node.kwargs)

    operands: list[SSAValue] = []
    operand_names: list[str] = []
    scalars: dict[str, Attribute] = {}
    for arg in schema.arguments:
        if arg.name in kwargs:
            value = kwargs[arg.name]
        elif arg.has_default_value():
            value = arg.default_value
        else:
            raise KernelLoweringError(f"{spec.qualname}: no value for argument {arg.name!r}")
        # Tensor (or Tensor[]) arguments -> SSA operands in schema order; a list is
        # flattened as name[0], name[1], ... so every operand stays individually named.
        items = value if isinstance(value, (list, tuple)) and any(_is_fx_node(v) for v in value) else None
        if _is_fx_node(value) or items is not None:
            for k, v in enumerate(items if items is not None else [value]):
                if not _is_fx_node(v) or v.name not in value_map:
                    raise KernelLoweringError(
                        f"{spec.qualname}: operand {arg.name!r} ({v}) has no SSA value")
                operands.append(value_map[v.name])
                operand_names.append(f"{arg.name}[{k}]" if items is not None else arg.name)
        else:
            scalars[arg.name] = _scalar_attr(value)

    call = CallOp(spec.callee, operands, list(result_types))
    call.attributes["prov.kernel"] = StringAttr(spec.qualname)
    call.attributes["prov.kernel_operands"] = ArrayAttr([StringAttr(n) for n in operand_names])
    if scalars:  # absent == no non-tensor args; a consumer that can't pass them must refuse
        call.attributes["prov.kernel_args"] = DictionaryAttr(scalars)
    rid = _next_region_id(spec.callee)
    _attach_region_id(call, rid)
    results = list(call.results)
    return DecompResult(ops=[call], result=results[0] if results else None, region_ids=[rid],
                        pattern_hint="kernel", results=results)


def kernel_decl(spec: KernelSpec, symbol: str, call: CallOp) -> FuncOp:
    """Private ``func.func`` declaration for ``call`` (typed from its live operands/results),
    tagged ``prov.kernel`` -- the hook a downstream lowering binds to the real kernel."""
    decl = FuncOp.external(symbol, [v.type for v in call.operands], [r.type for r in call.results])
    decl.attributes["prov.kernel"] = StringAttr(spec.qualname)
    return decl


__all__ = [
    "KERNEL_REGISTRY",
    "KernelLoweringError",
    "KernelSpec",
    "emit_kernel_call",
    "is_triton_op",
    "kernel_decl",
    "lookup_kernel",
    "registered_kernel_targets",
]
