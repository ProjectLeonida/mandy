r"""Where do the T#s and S#s of image instructions come from? (stdlib only; counts, no shader bytes)

A translated shader turns the T# (8 SGPRs) and S# (4 SGPRs) of every image instruction into a bindless index.
Whether that index can travel inside the descriptor (docs/design.md, "Bindless index in the T#") depends on where
the descriptor words were loaded from: the layer rewrites the console resource heaps' copies and its own heaps, but
not the title's other memory, and not the S#s of console sampler heaps, which shaders read in place.

This census follows the forward SGPR dataflow of tools/heap_index_census.py over every listing and classifies the
descriptor operand of each image instruction:

  * heap32          all words are one aligned descriptor of an s_load_dwordx8/x16 whose base is a 32-bit pointer in
                    an entry SGPR (a descriptor table or a directly indexed heap), static or dynamic offset;
  * heap32-partial  the word that carries the index (T#: word 6, S#: word 3) is such a load's, other words are not
                    (the shader edited the descriptor);
  * ptr64 / loaded / system / other-base   an s_load from a 64-bit entry pointer, a loaded pointer, a system SGPR,
                    anything else;
  * vsharp          an s_buffer_load through a V# (a constant or structured buffer of the title's);
  * literal         every word a translate-time constant;
  * unknown         the dataflow lost the words (a join of different loads, a v_readfirstlane, an ALU result).

It also counts, per descriptor word, scalar instructions that read a word of a heap32 T# or S# for anything but the
image instruction itself (a shader doing its own maths on a field).

Usage:
    python -I tools/descriptor_source_census.py <work>/shaders/dxbc/disasm [--json OUT] [--limit N]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import descriptor_sources  # noqa: E402
import heap_index_census as hic  # noqa: E402
import shader_ir  # noqa: E402
import shader_meta  # noqa: E402
from report_paths import same_report_path  # noqa: E402

LOADS = {"s_load_dwordx4": 4, "s_load_dwordx8": 8, "s_load_dwordx16": 16,
         "s_buffer_load_dwordx4": 4, "s_buffer_load_dwordx8": 8, "s_buffer_load_dwordx16": 16}
INDEX_WORD = {"T#": 6, "S#": 3}
WIDTH = {"T#": 8, "S#": 4}


def _regs(a, n):
    if a["k"] == "sgpr" and a.get("n", 1) >= n:
        return [a["r"] + i for i in range(n)]
    if a["k"] == "slist" and len(a["r"]) >= n and all(isinstance(r, int) for r in a["r"][:n]):
        return list(a["r"][:n])
    return None


def descriptor_operands(ins):
    """[(role, [sgpr numbers])] of an image instruction."""
    op = shader_meta.base_op(ins["op"])
    a = ins["args"]
    out = []
    if not op.startswith("image_") or len(a) < 3:
        return out
    t = _regs(a[2], 8)
    out.append(("T#", t))
    if op.startswith(shader_meta.SAMPLER_IMAGE_OPS) and len(a) >= 4:
        out.append(("S#", _regs(a[3], 4)))
    return out


def analyse(insns, stage, user_count):
    blocks, succ = hic.blocks_of(insns)
    system = set(hic.SYSTEM_SGPRS.get(stage, ()))
    entry = [None] * len(blocks)
    entry[0] = {r: ("in", r) for r in range(106)}
    work = [0]
    while work:
        b = work.pop()
        sym = dict(entry[b])
        for ins in blocks[b]:
            if shader_meta.base_op(ins["op"]) == "s_endpgm":
                break
            hic.step(sym, ins)
        for s in succ[b]:
            merged = sym if entry[s] is None else {k: v for k, v in entry[s].items() if sym.get(k) == v}
            if merged != entry[s]:
                entry[s] = dict(merged)
                work.append(s)

    loads = {}   # pc -> (base class, offset kind, static offset)

    def load_class(ins, sym):
        op = shader_meta.base_op(ins["op"])
        a = ins["args"]
        if op.startswith("s_buffer_"):
            return ("vsharp", None, None)
        if len(a) < 2 or a[1]["k"] != "sgpr" or a[1].get("n") != 2:
            return ("other-base", None, None)
        b = a[1]["r"]
        lo, hi = sym.get(b), sym.get(b + 1)
        rlo, dyn = hic.root(lo)
        off = hic._src(sym, a[2]) if len(a) >= 3 else ("const", 0)
        static = off[1] if off is not None and off[0] == "const" and not dyn else None
        kind = "static" if static is not None else "dynamic"
        if rlo is not None and rlo[0] == "in":
            n = rlo[1]
            if n in system or n >= user_count:
                return ("system", kind, static)
            if hi == ("const", 0):
                return ("heap32", kind, static)
            return ("ptr64", kind, static)
        if rlo is not None and rlo[0] == "ld":
            return ("loaded", kind, static)
        return ("other-base", kind, static)

    uses = []          # (pc, role, class)
    heap_loads = defaultdict(set)   # load pc -> roles of heap32 descriptors taken from it whole or in part
    pending_reads = []  # (load pc, dword) read by a scalar instruction that is not an image instruction
    for b in range(len(blocks)):
        if entry[b] is None:
            continue
        sym = dict(entry[b])
        for ins in blocks[b]:
            op = shader_meta.base_op(ins["op"])
            if op == "s_endpgm":
                break
            if op in LOADS and ins["args"] and ins["args"][0]["k"] == "sgpr":
                loads[ins["pc"]] = load_class(ins, sym)
            if op.startswith("image_"):
                for role, regs in descriptor_operands(ins):
                    uses.append((ins["pc"], role, classify(role, regs, sym, loads, heap_loads)))
            elif not op.startswith(("s_load_", "s_buffer_load_", "s_mov_b", "s_waitcnt")):
                reads, _ = shader_meta._split_operands(ins)
                for r in reads:
                    t = sym.get(r)
                    if t is not None and t[0] == "ld":
                        pending_reads.append((t[1], t[2], op))
            hic.step(sym, ins)
    verdict = descriptor_sources.analyse(blocks, succ, user_count, system)
    embedded = sum(1 for v in verdict.values() if v)
    reads = Counter()
    for pc, dword, op in pending_reads:
        for role in heap_loads.get(pc, ()):
            reads[(role, dword % WIDTH[role], op)] += 1
    return uses, reads, (embedded, len(verdict))


def classify(role, regs, sym, loads, heap_loads):
    if regs is None:
        return "unknown"
    tags = [sym.get(r) for r in regs]
    if all(t is not None and t[0] == "const" for t in tags):
        return "literal"
    w = WIDTH[role]
    first = tags[0]
    whole = (first is not None and first[0] == "ld" and first[2] % w == 0 and
             all(t == ("ld", first[1], first[2] + i) for i, t in enumerate(tags)))
    idx = tags[INDEX_WORD[role]]
    if whole:
        cls, kind, static = loads.get(first[1], ("other-base", None, None))
        if cls == "heap32":
            heap_loads[first[1]].add(role)
            if kind == "static" and static % 16:
                return "heap32-misaligned"
            return f"heap32-{kind}"
        return cls
    if idx is not None and idx[0] == "ld" and idx[2] % w == INDEX_WORD[role]:
        cls, kind, static = loads.get(idx[1], ("other-base", None, None))
        if cls == "heap32":
            heap_loads[idx[1]].add(role)
            return "heap32-partial"
        return cls + "-partial"
    return "unknown"


def fspx_user_count(disasm_dir, name):
    """USER_SGPR [5:1] of an FSPX program's COMPUTE_PGM_RSRC2 (<disasm>/../fspx/<name>.json, tools/fspx_unpack.py;
    the pipeline library's programs have no FSSM header), or None."""
    path = os.path.join(os.path.dirname(os.path.abspath(disasm_dir)), "fspx", name + ".json")
    try:
        with open(path, encoding="utf-8") as f:
            return (int(json.load(f)["rsrc2"]) >> 1) & 31
    except (OSError, ValueError, KeyError, TypeError):
        return None


def census(disasm_dir, limit=None, progress=None):
    stages = hic.stage_index(disasm_dir)
    names = sorted(n[:-2] for n in os.listdir(disasm_dir) if n.endswith(".s"))
    if limit:
        names = names[:limit]
    skipped = Counter()
    accesses = defaultdict(Counter)       # role -> class -> image instructions
    modules = defaultdict(Counter)        # role -> class -> modules with at least one
    only_heap = Counter()                 # role -> modules whose every operand of that role is heap32-*
    with_role = Counter()
    word_reads = Counter()
    per_stage = defaultdict(Counter)      # stage -> class (T#) -> accesses
    index_word = Counter()                # descriptor_sources.analyse, as the translator decides
    shaders = 0
    for k, name in enumerate(names):
        if progress and k % 2000 == 0:
            progress(k, len(names))
        stage = stages.get(name)
        if stage is None:
            skipped["no stage"] += 1
            continue
        info = shader_ir.parse_file(os.path.join(disasm_dir, name + ".s"))
        if not info["insns"]:
            skipped["no instructions"] += 1
            continue
        if info["invalid"]:
            skipped["embedded data words"] += 1
            continue
        user = hic.USER_SGPRS
        if stage == "cs":
            user = hic.cs_user_count(disasm_dir, name)
            if user is None:
                user = fspx_user_count(disasm_dir, name)
            if user is None:
                skipped["no FSSM header"] += 1
                continue
        try:
            uses, reads, (emb, images) = analyse(info["insns"], stage, user)
        except hic.Skip as e:
            skipped[str(e)] += 1
            continue
        shaders += 1
        index_word["image instructions"] += images
        index_word["T# index from dword 6"] += emb
        if images:
            index_word["modules with images"] += 1
            if emb == images:
                index_word["modules with no T# lookup"] += 1
        seen = defaultdict(set)
        for _, role, cls in uses:
            accesses[role][cls] += 1
            seen[role].add(cls)
            if role == "T#":
                per_stage[stage][cls] += 1
        for role, classes in seen.items():
            with_role[role] += 1
            for c in classes:
                modules[role][c] += 1
            if all(c.startswith("heap32-") and c != "heap32-misaligned" for c in classes):
                only_heap[role] += 1
        for (role, dword, op), n in reads.items():
            word_reads[(role, dword, op)] += n
    return {
        "shaders_read": len(names), "shaders_followed": shaders, "skipped": dict(skipped),
        "accesses": {r: dict(c.most_common()) for r, c in accesses.items()},
        "modules": {r: dict(c.most_common()) for r, c in modules.items()},
        "modules_with_role": dict(with_role),
        "modules_all_heap32": dict(only_heap),
        "t_sharp_accesses_per_stage": {s: dict(c.most_common()) for s, c in sorted(per_stage.items())},
        "tsharp_index_word": dict(index_word),
        "heap32_word_reads": [{"role": r, "word": d, "op": o, "count": n}
                              for (r, d, o), n in sorted(word_reads.items())],
    }


def print_report(out):
    print(f"shaders read: {out['shaders_read']}, followed: {out['shaders_followed']}; "
          f"skipped: {out['skipped'] or 'none'}")
    for role in ("T#", "S#"):
        acc = out["accesses"].get(role, {})
        total = sum(acc.values())
        print(f"\n{role}: {total} image instructions in {out['modules_with_role'].get(role, 0)} modules; "
              f"{out['modules_all_heap32'].get(role, 0)} modules take every {role} whole from a heap32 load")
        for cls, n in acc.items():
            print(f"  {cls:<24} {n:>9} accesses {out['modules'][role][cls]:>7} modules")
    print("\nT# accesses per stage:")
    for s, c in out["t_sharp_accesses_per_stage"].items():
        print(f"  {s}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    print("\nthe translator's verdict (tools/descriptor_sources.py): " +
          ", ".join(f"{k} {v}" for k, v in out["tsharp_index_word"].items()))
    words = Counter()
    for row in out["heap32_word_reads"]:
        words[(row["role"], row["word"])] += row["count"]
    print("\nscalar reads of heap32 descriptor words (not by the image instruction):")
    for (role, word), n in sorted(words.items()):
        print(f"  {role} word {word}: {n}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("disasm", help="<work>/shaders/dxbc/disasm")
    ap.add_argument("--json", help="write the full census here (outside the disasm folder)")
    ap.add_argument("--limit", type=shader_ir.positive_int, help="only the first N listings (sorted by name)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.disasm):
        print(f"ERROR: {args.disasm} is not a folder", file=sys.stderr)
        return 2
    if args.json:
        disasm = os.path.realpath(args.disasm)
        target = os.path.realpath(os.path.abspath(args.json))
        try:
            inside = os.path.commonpath([disasm, target]) == disasm
        except ValueError:
            inside = False
        index_path = os.path.join(os.path.dirname(os.path.abspath(args.disasm)), "index.json")
        if inside or same_report_path(args.json, index_path):
            print(f"ERROR: --json must not write inside or next to the folder it reads: {args.json}",
                  file=sys.stderr)
            return 2
    try:
        out = census(args.disasm, args.limit,
                     progress=lambda k, n: print(f"  {k}/{n}", file=sys.stderr, flush=True))
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print_report(out)
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1, sort_keys=True)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write {args.json}: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
