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


def tsharp_operand(ins):
    """The SGPR numbers of an image instruction's T#, or None."""
    a = ins["args"]
    if not _meta.base_op(ins["op"]).startswith("image_") or len(a) < 3:
        return None
    t = a[2]
    if t["k"] == "sgpr" and t.get("n", 1) >= 8:
        return [t["r"] + i for i in range(8)]
    if t["k"] == "slist" and len(t["r"]) >= 8 and all(isinstance(r, int) for r in t["r"][:8]):
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
