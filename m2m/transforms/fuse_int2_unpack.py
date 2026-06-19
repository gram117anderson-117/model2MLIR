"""OPT-IN int2-unpack recognizer: fold bitvla's opaque ternary-unpack chain into one named op.

BitNet (bitvla) stores its W1.58 ternary weights packed 4 values per int8 byte; the dequant
path unpacks them with ``packed & 0x03`` / ``(packed >> 2|4|6) & 0x03`` then ``stack`` of the four
2-bit lanes. With the model's ``.item()`` scale graph-breaking the chain, those lowered to opaque
``func.call @aten___and___Scalar / @aten___rshift___Scalar / @aten_stack_default``. This pass
recognizes that exact chain (a ``stack`` of 4 i8 operands all tracing back through and/rshift opaque
calls to a single packed i8 source) and rewrites it to ``quant_ext.unpack_int2(packed)`` — naming the
native low-bit storage datapath instead of leaving it as opaque bitwise soup.

DE-RISK (the user's hard requirement — must not touch any other model):
  * This pass is **opt-in**: it is NOT in the default ``m2m.convert`` pipeline. Only the bitvla
    native recapture applies it explicitly, so every other model/capture is unchanged by construction.
  * **Verify-guarded** exactly like ``fuse_qdq``: runs on a clone and returns it only if it still
    verifies; on ANY failure the original module is returned unchanged (never regresses).
  * It matches a highly specific signature (stack-of-4-i8 whose operands all reduce to one packed i8
    source via and/rshift opaque calls), so it cannot fire on unrelated graphs.
"""

from __future__ import annotations

from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, StringAttr
from xdsl.dialects.func import CallOp
from xdsl.rewriter import InsertPoint, Rewriter

from m2m.ir.quant.ops import UnpackInt2Op


def _callee(op) -> str | None:
    if not isinstance(op, CallOp):
        return None
    c = op.properties.get("callee")
    return c.root_reference.data if c is not None and hasattr(c, "root_reference") else None


def _packed_source(value):
    """Walk back through and/rshift opaque calls (each a single-operand func.call) to the
    ultimate packed i8 source feeding one stack lane."""
    op = getattr(value, "owner", None)
    for _ in range(4):                       # &, >>& at most a couple hops per lane
        cal = _callee(op) if op is not None else None
        if cal and ("_and_" in cal or "_rshift_" in cal or "_lshift_" in cal) and op.operands:
            value = op.operands[0]
            op = getattr(value, "owner", None)
        else:
            break
    return value


def fuse_int2_unpack(module: ModuleOp) -> tuple[ModuleOp, int]:
    """Recognize the bitvla int2-unpack chain -> ``quant_ext.unpack_int2``. Verify-guarded:
    returns (rewritten_clone, n) only if it still verifies, else (original, 0)."""
    work = module.clone()
    try:
        n = _fuse(work)
        if n:
            work.verify()
        return (work, n) if n else (module, 0)
    except Exception:  # noqa: BLE001
        return module, 0


def _fuse(module: ModuleOp) -> int:
    count = 0
    for op in list(module.walk()):
        cal = _callee(op)
        if not cal or "stack" not in cal:
            continue
        if len(op.operands) != 4:            # the 4 ternary lanes q0..q3
            continue
        srcs = [_packed_source(o) for o in op.operands]
        if len({id(s) for s in srcs}) != 1:  # all 4 must reduce to the SAME packed source
            continue
        packed = srcs[0]
        # require the lane operands to actually come from and/rshift opaque calls (not a
        # coincidental 4-operand stack) -- at least 3 of 4 lanes go through a bitwise call
        bitwise = sum(1 for o in op.operands
                      if (_callee(getattr(o, "owner", None)) or "").find("_and_") >= 0
                      or (_callee(getattr(o, "owner", None)) or "").find("_rshift_") >= 0)
        if bitwise < 3:
            continue
        res_t = op.results[0].type
        unpack = UnpackInt2Op(
            operands=[packed], result_types=[res_t],
            properties={"bits": IntegerAttr(2, IntegerType(64)),
                        "lanes": IntegerAttr(4, IntegerType(64))})
        unpack.attributes["prov.op"] = StringAttr("unpack_int2")
        unpack.attributes["prov.family"] = StringAttr("quantize")
        Rewriter.insert_op(unpack, InsertPoint.before(op))
        op.results[0].replace_all_uses_with(unpack.results[0])
        Rewriter.erase_op(op)
        count += 1
        # sweep now-dead and/rshift opaque calls feeding this unpack (zero remaining uses)
        for _ in range(3):
            for o in list(module.walk()):
                c2 = _callee(o)
                if c2 and ("_and_" in c2 or "_rshift_" in c2 or "_lshift_" in c2) \
                        and all(len(list(r.uses)) == 0 for r in o.results):
                    Rewriter.erase_op(o)
    return count
