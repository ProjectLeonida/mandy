r"""Which image instructions read a T# whose index word the layer wrote (docs/design.md, "Bindless index in the T#").

The layer writes the bindless index of every T# into dword 6 of the descriptor slot shaders load it from: the
copies of the console resource heaps it keeps (fix_console_slot) and its own shader-visible heaps (Create*View).
Shaders reach those heaps only through a descriptor-table or heap pointer, which arrives as a 32-bit pointer in an
entry SGPR (user data) with the high dword a constant 0 (docs/d3d12_vulkan.md section 3.4, D3D-329: 99.86% of the
corpus's descriptor loads). A T# loaded through anything else (a 64-bit pointer, a pointer read from memory, a
constant or structured buffer of the title's) or built from constants keeps the title's dword 6, so for those the
translator keeps the hash-table lookup.

`analyse` follows a forward dataflow over the basic blocks, as tools/heap_index_census.py does, tagging each SGPR
with where its value came from, and answers, for every image instruction, whether the SGPR that holds its T#'s
dword 6 holds dword 6 of a heap slot: a dword of an s_load from such a pointer whose byte position in its 32-byte
slot is 24 (a static offset is known; a run-time offset, the heap index times the slot size, is taken as slot
aligned, which the shader's tag check guards). Where two paths join with loads of different instructions, the value
keeps its slot position when both are heap dwords at the same position, so a T# loaded in either arm of a branch
still counts. Any other write, and anything the dataflow cannot follow, gives the lookup.

Standard library only; tools/rdna2spirv.py and tools/descriptor_source_census.py share it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heap_index_census as _hic  # noqa: E402
import shader_meta as _meta  # noqa: E402

SLOT_BYTES = 32       # kDescriptorSize: one T# per slot, at its start
INDEX_DWORD = 6       # the T# dword that carries the index (gfx10_descriptors.h, tsharp_index_word)
ENTRY_SGPRS = 106
SCALAR_LOADS = ("s_load_dword", "s_buffer_load_dword")


def _load_slot_base(ins, sym, user_count, system):
    """For an s_load whose base is a 32-bit heap pointer in a user SGPR: the byte position in its slot of the load's
    first dword (a static offset mod 32; 0 for a run-time one, the heap index times 32). None for any other load."""
    op = _meta.base_op(ins["op"])
    a = ins["args"]
    if not op.startswith("s_load_dword") or len(a) < 2 or a[1]["k"] != "sgpr" or a[1].get("n") != 2:
        return None
    b = a[1]["r"]
    lo, hi = sym.get(b), sym.get(b + 1)
    t = lo
    while t is not None and t[0] == "off":
        if not t[2]:
            return None   # a constant added to the pointer whose value the tag does not keep: position unknown
        t = t[1]
    if t is None or t[0] != "in" or hi != ("const", 0):
        return None
    n = t[1]
    if n in system or n >= user_count:
        return None
    off = _hic._src(sym, a[2]) if len(a) >= 3 else ("const", 0)
    if off is not None and off[0] == "const":
        return off[1] % SLOT_BYTES
    return 0              # a run-time offset (the heap index times the slot size) on top of the table pointer


def _slot_pos(tag, heap_loads):
    """The dword position in a heap slot of the value `tag` names, or None."""
    if tag is None:
        return None
    if tag[0] == "hw":
        return tag[1]
    if tag[0] == "ld" and tag[1] in heap_loads:
        byte = heap_loads[tag[1]] + 4 * tag[2]
        return (byte % SLOT_BYTES) // 4 if byte % 4 == 0 else None
    return None


def _merge(old, new, heap_loads):
    out = {}
    for k, v in old.items():
        w = new.get(k)
        if w == v:
            out[k] = v
            continue
        p, q = _slot_pos(v, heap_loads), _slot_pos(w, heap_loads)
        if p is not None and p == q:
            out[k] = ("hw", p)
    return out


def _vgprs(ins):
    """Every VGPR an instruction names, read or written."""
    out = set()
    for a in ins["args"]:
        if a["k"] == "vgpr":
            out.update(range(a["r"], a["r"] + a.get("n", 1)))
        elif a["k"] == "vlist":
            out.update(r for r in a["r"] if isinstance(r, int))
    return out


def _lane(sym, a):
    """The constant lane of a v_readlane/v_writelane lane operand, or None."""
    t = _hic._src(sym, a)
    return t[1] & 63 if t is not None and t[0] == "const" else None


def _spill_step(sym, ins, spill_vgprs):
    """SGPR values parked in VGPR lanes (v_writelane_b32 / v_readlane_b32, a compiler's SGPR spill): the lane keeps the
    tag of the SGPR written into it, as ("v", vgpr, lane), and a v_readlane of a constant lane gives it back. Any other
    instruction naming the VGPR forgets its lanes. Run after heap_index_census.step, which has forgotten the
    v_readlane's destination."""
    op = _meta.base_op(ins["op"])
    a = ins["args"]
    if op == "v_readlane_b32":     # reads the VGPR and leaves it as it is, whatever it writes (an SGPR, VCC)
        if len(a) >= 3 and a[0]["k"] == "sgpr" and a[0].get("n", 1) == 1 and a[1]["k"] == "vgpr":
            lane = _lane(sym, a[2])
            tag = sym.get(("v", a[1]["r"], lane)) if lane is not None else None
            if tag is not None:
                sym[a[0]["r"]] = tag
        return
    touched = _vgprs(ins) & spill_vgprs
    if not touched:
        return
    before = None
    if op == "v_writelane_b32" and len(a) >= 3 and a[0]["k"] == "vgpr":
        lane = _lane(sym, a[2])
        if lane is not None:     # only that lane changes; a source other than an SGPR (VCC, a constant) is unknown
            src = sym.get(a[1]["t"]) if a[1]["k"] == "special" and a[1].get("t") in VCC_HALVES else \
                _hic._src(sym, a[1])
            before = (a[0]["r"], lane, src)
    for k in [k for k in sym if isinstance(k, tuple) and k[1] in touched]:
        if before is None or k[1] != before[0] or k[2] == before[1]:
            del sym[k]
    if before is not None and before[2] is not None:
        sym[("v", before[0], before[1])] = before[2]


VCC_HALVES = ("vcc_lo", "vcc_hi")


def _vcc_names(a):
    if a["k"] == "special" and a.get("t") in ("vcc", "vcc_lo", "vcc_hi"):
        return {"vcc_lo", "vcc_hi"} if a["t"] == "vcc" else {a["t"]}
    if a["k"] == "slist":
        return {r for r in a["r"] if r in VCC_HALVES}
    return set()


def _vcc_step(sym, ins):
    """VCC used as two more SGPRs: a compiler short of SGPRs loads a T#'s last two dwords into vcc_lo and vcc_hi
    (s_load_dwordx8 [s100, ..., s105, vcc_lo, vcc_hi]) or parks a pointer there. Their tags live under "vcc_lo" and
    "vcc_hi". LLVM prints every VCC operand, the implicit ones of VOPC and carry instructions included, so any other
    instruction naming VCC forgets them (a reader as well, which only loses precision). Run after
    heap_index_census.step, which follows neither VCC nor a load into a register list."""
    op = _meta.base_op(ins["op"])
    a = ins["args"]
    if not a:
        return
    if op.startswith(SCALAR_LOADS) and a[0]["k"] == "slist":
        for i, r in enumerate(a[0]["r"]):
            sym[r] = ("ld", ins["pc"], i)
        return
    if op.startswith("image_"):
        return                      # reads its descriptor operands
    names = set().union(*(_vcc_names(x) for x in a))
    if not names:
        return
    if op == "v_readlane_b32" and len(a) >= 3 and a[0]["k"] == "special" and a[0].get("t") in VCC_HALVES \
            and a[1]["k"] == "vgpr":
        lane = _lane(sym, a[2])
        tag = sym.get(("v", a[1]["r"], lane)) if lane is not None else None
        sym.pop(a[0]["t"], None)
        if tag is not None:
            sym[a[0]["t"]] = tag
        return
    if op in ("s_mov_b64", "s_mov_b32") and len(a) >= 2 and a[0]["k"] == "sgpr" and not _vcc_names(a[0]):
        src = ["vcc_lo", "vcc_hi"] if op == "s_mov_b64" and a[1].get("t") == "vcc" else \
            [a[1]["t"]] if op == "s_mov_b32" and a[1].get("t") in VCC_HALVES else None
        if src is not None:
            for i, name in enumerate(src):
                tag = sym.get(name)
                if tag is not None:
                    sym[a[0]["r"] + i] = tag
            return
    for name in VCC_HALVES:
        sym.pop(name, None)


def tsharp_operand(ins):
    """The registers of an image instruction's T# (SGPR numbers, or "vcc_lo" / "vcc_hi" in a register list), or
    None."""
    a = ins["args"]
    if not _meta.base_op(ins["op"]).startswith("image_") or len(a) < 3:
        return None
    t = a[2]
    if t["k"] == "sgpr" and t.get("n", 1) >= 8:
        return [t["r"] + i for i in range(8)]
    if t["k"] == "slist" and len(t["r"]) >= 8 and all(isinstance(r, int) or r in VCC_HALVES for r in t["r"][:8]):
        return list(t["r"][:8])
    return None


def analyse(blocks, succ, user_count=32, system=()):
    """{id(ins): bool} for every image instruction of `blocks` (the translator's basic blocks, `succ` their
    successors): True when its T#'s dword 6 is dword 6 of a heap slot. Raises heap_index_census.Skip never; an
    instruction the analysis cannot place is simply absent (the caller keeps the lookup)."""
    system = set(system)
    if not blocks:
        return {}
    # The tags name a load by its pc. A listing without encodings (a hand-written test program) gives every
    # instruction the same pc, so the analysis then runs on copies numbered in order and maps its answers back.
    flat = [ins for blk in blocks for ins in blk]
    if len({ins["pc"] for ins in flat}) < len(flat):
        copies, k = [], 0
        for blk in blocks:
            copies.append([dict(ins, pc=4 * (k + n)) for n, ins in enumerate(blk)])
            k += len(blk)
        answers = analyse(copies, succ, user_count, system)
        return {id(ins): answers.get(id(copy), False)
                for ins, copy in zip(flat, (c for blk in copies for c in blk))}
    spill_vgprs = {ins["args"][0]["r"] for ins in flat if _meta.base_op(ins["op"]) == "v_writelane_b32" and
                   ins["args"] and ins["args"][0]["k"] == "vgpr"}
    # A load is a heap load when every path reaching it brings a heap pointer. The fixpoint below decides that as it
    # goes; a load first seen as one and later not (a join lost its pointer) is banned and the dataflow run again
    # without it, so no slot position stays derived from a load that is not a heap load in the end.
    banned = set()
    while True:
        heap_loads = {}   # load pc -> byte position of its first dword in its slot
        lost = set()
        entry = [None] * len(blocks)
        entry[0] = {r: ("in", r) for r in range(ENTRY_SGPRS)}

        def run(b, sym, out=None):
            for ins in blocks[b]:
                op = _meta.base_op(ins["op"])
                if op == "s_endpgm":
                    break
                if op.startswith(SCALAR_LOADS):
                    pc = ins["pc"]
                    pos = None if pc in banned else _load_slot_base(ins, sym, user_count, system)
                    if pos is None or heap_loads.get(pc, pos) != pos:
                        if pc in heap_loads:
                            lost.add(pc)
                        heap_loads.pop(pc, None)
                        banned.add(pc)
                    else:
                        heap_loads[pc] = pos
                if out is not None and op.startswith("image_"):
                    regs = tsharp_operand(ins)
                    out[id(ins)] = (regs is not None and
                                    _slot_pos(sym.get(regs[INDEX_DWORD]), heap_loads) == INDEX_DWORD)
                _hic.step(sym, ins)
                if spill_vgprs:
                    _spill_step(sym, ins, spill_vgprs)
                _vcc_step(sym, ins)
            return sym

        work = [0]
        while work:
            b = work.pop()
            sym = run(b, dict(entry[b]))
            for s in succ[b]:
                merged = dict(sym) if entry[s] is None else _merge(entry[s], sym, heap_loads)
                if merged != entry[s]:
                    entry[s] = merged
                    work.append(s)
        if lost:
            continue
        out = {}
        for b in range(len(blocks)):
            if entry[b] is not None:
                run(b, dict(entry[b]), out)
        if lost:
            continue
        return out
