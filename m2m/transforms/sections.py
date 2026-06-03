"""Partition a captured module into per-source-module sections.

VLAs are composed of sections (VLM backbone, action expert / DiT, vision encoder, ...) that
may run at different frequencies. The importer flattens the whole forward, but tags every op
with ``prov.module`` (its source nn.Module). This pass splits the flat function into one
``func.func`` per top-level section -- each taking the cross-section values it consumes and
returning the values other sections consume -- so each section can be compiled/scheduled/run
at its own cadence (and emitted to its own ``.mlir`` file).
"""

from __future__ import annotations

from xdsl.dialects.builtin import FunctionType, ModuleOp, StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region


def _section_of(op) -> str | None:
    a = op.attributes.get("prov.module")
    return a.data if isinstance(a, StringAttr) else None


def split_by_section(module: ModuleOp) -> dict[str, ModuleOp]:
    """Return ``{section_name: ModuleOp}`` -- one module per top-level source section, each a
    ``func.func @section_<name>`` with the section's ops and proper cross-section I/O. Ops
    without an ``prov.module`` tag inherit their operands' section (else 'shared')."""
    func = next((o for o in module.body.block.ops if isinstance(o, FuncOp)), None)
    if func is None or not func.body.blocks:
        return {}
    block = func.body.blocks[0]
    body_ops = [o for o in block.ops if not isinstance(o, ReturnOp)]
    ret = next((o for o in block.ops if isinstance(o, ReturnOp)), None)
    ret_vals = set(ret.operands) if ret is not None else set()

    # assign each op a section: its tag, else inherited from an operand's defining op.
    sec: dict = {}
    for op in body_ops:
        s = _section_of(op)
        if s is None:
            for v in op.operands:
                ow = v.owner
                if ow in sec:
                    s = sec[ow]
                    break
        sec[op] = s or "shared"

    order: list[str] = []
    for op in body_ops:
        if sec[op] not in order:
            order.append(sec[op])

    out: dict[str, ModuleOp] = {}
    for S in order:
        sops = [op for op in body_ops if sec[op] == S]
        sset = set(sops)
        defined = {r for op in sops for r in op.results}
        inputs: list = []
        iseen: set = set()
        for op in sops:
            for v in op.operands:
                if v not in defined and v not in iseen:
                    inputs.append(v)
                    iseen.add(v)
        outputs: list = []
        oseen: set = set()
        for op in sops:
            for r in op.results:
                if (r in ret_vals or any(u.operation not in sset for u in r.uses)) and r not in oseen:
                    outputs.append(r)
                    oseen.add(r)

        nblk = Block(arg_types=[v.type for v in inputs])
        vmap = {v: nblk.args[k] for k, v in enumerate(inputs)}
        for op in sops:
            cl = op.clone(value_mapper=vmap)
            for old, new in zip(op.results, cl.results):
                vmap[old] = new
            nblk.add_op(cl)
        nblk.add_op(ReturnOp(*[vmap[o] for o in outputs]))
        ft = FunctionType.from_lists([v.type for v in inputs], [o.type for o in outputs])
        nf = FuncOp(f"section_{S}", ft, region=Region(nblk))
        m = ModuleOp([nf])
        m.attributes["prov.section"] = StringAttr(S)
        out[S] = m
    return out
