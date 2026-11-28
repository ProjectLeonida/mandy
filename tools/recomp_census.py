"""Recompiler census over a PE image's .pdata.

Usage:
    python -I recomp_census.py <image> [--out DIR] [--limit N] [--top K]

Walks every RUNTIME_FUNCTION entry in .pdata (the same table the SEH machinery and
tools/pe_inspect.py report on), decodes each function with the x86-64 decoder in
tools/x64_recomp.py and runs the translator over it. The report answers the
questions the arm64 port needs answered before any machine code gets emitted:

  * how many functions / code bytes does .pdata claim?
  * which mnemonics dominate the corpus?
  * how much of it does the current emitter cover (per-function and overall)?
  * what are the blockers: decode gaps, unhandled mnemonics, indirect control
    flow that needs the pc-table dispatch, x87/SIMD/sys instructions?

Writes <out>/<file>.recomp.json and <out>/<file>.recomp.md. A wrong input or an
--out that cannot be written is one ERROR line and exit code 2. Standard library
only; reads the file, never runs it.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pe_inspect  # noqa: E402
import x64_recomp  # noqa: E402

DIR_EXCEPTION = pe_inspect.DIR_EXCEPTION
MAX_FUNCS = 1 << 20  # .pdata this long is not a table; matches pe_inspect's guard spirit

UNW_FLAG_EHANDLER = 0x1
UNW_FLAG_UHANDLER = 0x2
UNW_FLAG_CHAININFO = 0x4
UNW_FLAG_KNOWN_MASK = UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER | UNW_FLAG_CHAININFO
UNW_FLAG_NAMES = {
    UNW_FLAG_EHANDLER: "EHANDLER",
    UNW_FLAG_UHANDLER: "UHANDLER",
    UNW_FLAG_CHAININFO: "CHAINED",
}

UNWIND_REG_NAMES = (
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
)
UNWIND_OP_NAMES = {
    0: "UWOP_PUSH_NONVOL",
    1: "UWOP_ALLOC_LARGE",
    2: "UWOP_ALLOC_SMALL",
    3: "UWOP_SET_FPREG",
    4: "UWOP_SAVE_NONVOL",
    5: "UWOP_SAVE_NONVOL_FAR",
    6: "UWOP_EPILOG",
    7: "UWOP_SPARE_CODE",
    8: "UWOP_SAVE_XMM128",
    9: "UWOP_SAVE_XMM128_FAR",
    10: "UWOP_PUSH_MACHFRAME",
}
MAX_UNWIND_CHAIN = 64

# insn kinds that always need the pc-table dispatch (target not statically known)
INDIRECT_KINDS = ("ret",)


def pdata_entries(img):
    """Sorted [(begin_rva, end_rva, unwind_rva)] from the exception directory."""
    rva, size = img.directory(DIR_EXCEPTION)
    if not rva or size < 12:
        return []
    off = img.offset(rva, size)
    if off is None:
        return []
    out = []
    for i in range(min(size // 12, MAX_FUNCS)):
        b, e, u = struct.unpack_from("<III", img.data, off + i * 12)
        if b and e >= b:
            out.append((b, e, u))
    out.sort()
    return out


def function_bytes(img, begin, end):
    """File bytes for [begin_rva, end_rva), or None when it spans sections/headers oddly."""
    off = img.offset(begin, end - begin)
    if off is None:
        return None
    return img.data[off:off + (end - begin)]


def unwind_flags(img, unwind_rva):
    """(version, flag_names, code_count) from an UNWIND_INFO header, or None."""
    off = img.offset(unwind_rva, 4)
    if off is None:
        return None
    b0, _prolog_size, code_count, _frame = img.data[off:off + 4]
    flags = b0 >> 3
    names = [n for bit, n in UNW_FLAG_NAMES.items() if flags & bit]
    return (b0 & 7, names, code_count)


def _decode_unwind_codes(data, off, count, version, frame_reg):
    """Decode x64 UNWIND_CODE slots starting at `off`. Returns (ops, error)."""
    ops = []
    idx = 0
    seen_epilog = False
    while idx < count:
        slot_off = off + idx * 2
        code_offset = data[slot_off]
        op_byte = data[slot_off + 1]
        op = op_byte & 0x0F
        info = (op_byte >> 4) & 0x0F
        name = UNWIND_OP_NAMES.get(op)
        if name is None:
            return ops, f"unknown unwind op {op}"

        if op == 0:  # UWOP_PUSH_NONVOL
            ops.append({"op": name, "code_offset": code_offset, "reg": UNWIND_REG_NAMES[info]})
            idx += 1
        elif op == 1:  # UWOP_ALLOC_LARGE
            if info == 0:
                if idx + 2 > count:
                    return ops, "truncated UWOP_ALLOC_LARGE"
                size = struct.unpack_from("<H", data, slot_off + 2)[0] * 8
                idx += 2
            elif info == 1:
                if idx + 3 > count:
                    return ops, "truncated UWOP_ALLOC_LARGE"
                lo, hi = struct.unpack_from("<HH", data, slot_off + 2)
                size = lo | (hi << 16)
                idx += 3
            else:
                return ops, f"invalid UWOP_ALLOC_LARGE info {info}"
            ops.append({"op": name, "code_offset": code_offset, "size": size})
        elif op == 2:  # UWOP_ALLOC_SMALL
            ops.append({"op": name, "code_offset": code_offset, "size": (info + 1) * 8})
            idx += 1
        elif op == 3:  # UWOP_SET_FPREG
            if frame_reg == 0:
                return ops, "UWOP_SET_FPREG with FrameRegister=0"
            ops.append({"op": name, "code_offset": code_offset, "reg": UNWIND_REG_NAMES[frame_reg]})
            idx += 1
        elif op == 4:  # UWOP_SAVE_NONVOL
            if idx + 2 > count:
                return ops, "truncated UWOP_SAVE_NONVOL"
            offset = struct.unpack_from("<H", data, slot_off + 2)[0] * 8
            ops.append({"op": name, "code_offset": code_offset,
                        "reg": UNWIND_REG_NAMES[info], "offset": offset})
            idx += 2
        elif op == 5:  # UWOP_SAVE_NONVOL_FAR
            if idx + 3 > count:
                return ops, "truncated UWOP_SAVE_NONVOL_FAR"
            lo, hi = struct.unpack_from("<HH", data, slot_off + 2)
            ops.append({"op": name, "code_offset": code_offset,
                        "reg": UNWIND_REG_NAMES[info], "offset": lo | (hi << 16)})
            idx += 3
        elif op == 6:  # UWOP_EPILOG (UNWIND_INFO version 2)
            if version < 2:
                return ops, "UWOP_EPILOG requires UNWIND_INFO version 2"
            if not seen_epilog:
                seen_epilog = True
                ops.append({"op": name, "code_offset": code_offset,
                            "at_end": bool(info & 1), "length": code_offset})
            else:
                ep_off = (info << 8) | code_offset
                entry = {"op": name, "code_offset": code_offset, "epilog_offset": ep_off}
                if ep_off == 0:
                    entry["padding"] = True
                ops.append(entry)
            idx += 1
        elif op == 7:  # UWOP_SPARE_CODE (3 slots in Windows/CoreCLR x64 unwinder)
            if idx + 3 > count:
                return ops, "truncated UWOP_SPARE_CODE"
            ops.append({"op": name, "code_offset": code_offset})
            idx += 3
        elif op == 8:  # UWOP_SAVE_XMM128
            if idx + 2 > count:
                return ops, "truncated UWOP_SAVE_XMM128"
            offset = struct.unpack_from("<H", data, slot_off + 2)[0] * 16
            ops.append({"op": name, "code_offset": code_offset,
                        "reg": f"xmm{info}", "offset": offset})
            idx += 2
        elif op == 9:  # UWOP_SAVE_XMM128_FAR
            if idx + 3 > count:
                return ops, "truncated UWOP_SAVE_XMM128_FAR"
            lo, hi = struct.unpack_from("<HH", data, slot_off + 2)
            ops.append({"op": name, "code_offset": code_offset,
                        "reg": f"xmm{info}", "offset": lo | (hi << 16)})
            idx += 3
        elif op == 10:  # UWOP_PUSH_MACHFRAME
            ops.append({"op": name, "code_offset": code_offset, "has_error_code": bool(info)})
            idx += 1
    return ops, None


def parse_unwind_info(img, unwind_rva):
    """Parse an x64 .xdata UNWIND_INFO entry, decoding UNWIND_CODEs and following UNW_FLAG_CHAININFO."""
    base = {
        "rva": unwind_rva,
        "version": None,
        "flags": [],
        "prolog_size": 0,
        "code_count": 0,
        "frame_register": None,
        "frame_offset": 0,
        "ops": [],
        "stack_alloc": 0,
        "saved_gprs": [],
        "saved_xmms": [],
        "has_machframe": False,
        "epilog_count": 0,
        "handler_rva": None,
        "chain_depth": 0,
        "chained_rvas": [],
        "error": None,
    }
    cur_rva = unwind_rva
    visited = set()
    saved_gpr_set = set()
    saved_xmm_set = set()

    for hop in range(MAX_UNWIND_CHAIN + 1):
        if cur_rva in visited:
            base["error"] = f"chained unwind cycle at {hex(cur_rva)}"
            return base
        visited.add(cur_rva)

        off = img.offset(cur_rva, 4)
        if off is None:
            base["error"] = f"unmapped unwind header at {hex(cur_rva)}"
            return base
        b0, prolog_size, code_count, b3 = img.data[off:off + 4]
        version = b0 & 0x07
        flags = (b0 >> 3) & 0x1F
        flag_names = [n for bit, n in UNW_FLAG_NAMES.items() if flags & bit]
        frame_reg = b3 & 0x0F
        frame_offset = ((b3 >> 4) & 0x0F) * 16 if frame_reg else 0

        if hop == 0:
            base["version"] = version
            base["flags"] = flag_names
            base["prolog_size"] = prolog_size
            base["code_count"] = code_count
            base["frame_register"] = UNWIND_REG_NAMES[frame_reg] if frame_reg else None
            base["frame_offset"] = frame_offset
        elif base["frame_register"] is None and frame_reg:
            base["frame_register"] = UNWIND_REG_NAMES[frame_reg]
            base["frame_offset"] = frame_offset

        if version not in (1, 2):
            base["error"] = f"unsupported unwind version {version}"
            return base
        if flags & ~UNW_FLAG_KNOWN_MASK:
            base["error"] = f"unknown unwind flags {hex(flags)}"
            return base
        if (flags & UNW_FLAG_CHAININFO) and (flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER)):
            base["error"] = "CHAINED combined with EHANDLER/UHANDLER"
            return base

        # The UNWIND_CODE array is padded to an even number of 2-byte slots before trailing data.
        padded_bytes = ((code_count + 1) & ~1) * 2
        if code_count:
            codes_off = img.offset(cur_rva + 4, code_count * 2)
            if codes_off is None:
                base["error"] = f"truncated unwind codes at {hex(cur_rva)}"
                return base
            ops, err = _decode_unwind_codes(img.data, codes_off, code_count, version, frame_reg)
            base["ops"].extend(ops)
            for op in ops:
                kind = op["op"]
                if kind == "UWOP_PUSH_NONVOL":
                    base["stack_alloc"] += 8
                    if op["reg"] not in saved_gpr_set:
                        saved_gpr_set.add(op["reg"])
                        base["saved_gprs"].append(op["reg"])
                elif kind in ("UWOP_ALLOC_SMALL", "UWOP_ALLOC_LARGE"):
                    base["stack_alloc"] += op["size"]
                elif kind in ("UWOP_SAVE_NONVOL", "UWOP_SAVE_NONVOL_FAR"):
                    if op["reg"] not in saved_gpr_set:
                        saved_gpr_set.add(op["reg"])
                        base["saved_gprs"].append(op["reg"])
                elif kind in ("UWOP_SAVE_XMM128", "UWOP_SAVE_XMM128_FAR"):
                    if op["reg"] not in saved_xmm_set:
                        saved_xmm_set.add(op["reg"])
                        base["saved_xmms"].append(op["reg"])
                elif kind == "UWOP_PUSH_MACHFRAME":
                    base["has_machframe"] = True
                elif kind == "UWOP_EPILOG":
                    if op.get("at_end") or (op.get("epilog_offset", 0) > 0):
                        base["epilog_count"] += 1
            if err is not None:
                base["error"] = err
                return base

        tail_rva = cur_rva + 4 + padded_bytes
        if flags & UNW_FLAG_CHAININFO:
            if hop >= MAX_UNWIND_CHAIN:
                base["error"] = f"chained unwind depth exceeds {MAX_UNWIND_CHAIN}"
                return base
            chain_off = img.offset(tail_rva, 12)
            if chain_off is None:
                base["error"] = f"truncated chained RUNTIME_FUNCTION at {hex(tail_rva)}"
                return base
            _cb, _ce, next_rva = struct.unpack_from("<III", img.data, chain_off)
            base["chain_depth"] += 1
            base["chained_rvas"].append(next_rva)
            cur_rva = next_rva
            continue

        if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER):
            handler_off = img.offset(tail_rva, 4)
            if handler_off is None:
                base["error"] = f"truncated exception handler RVA at {hex(tail_rva)}"
                return base
            base["handler_rva"] = struct.unpack_from("<I", img.data, handler_off)[0]
        return base

    base["error"] = f"chained unwind depth exceeds {MAX_UNWIND_CHAIN}"
    return base


def census_function(img, begin, end, image_base):
    """Decode + translate one .pdata function. Returns a stats dict."""
    code = function_bytes(img, begin, end)
    st = {"rva": begin, "size": end - begin, "insns": 0, "translated": 0,
          "unhandled": Counter(), "decode_errors": 0, "indirect": Counter(),
          "branches": 0, "mnem": Counter()}
    if code is None:
        st["decode_errors"] += 1
        return st
    good = []
    for ins in x64_recomp.disasm(code, image_base + begin):
        if ins.unknown or ins.error:
            st["decode_errors"] += 1
            st["unhandled"][f"decode:{ins.mnem}"] += 1
            break
        good.append(ins)
    st["insns"] = len(good)
    for ins in good:
        st["mnem"][ins.mnem] += 1
        if ins.kind in ("branch", "call") and ins.ops and ins.ops[0][0] != "a":
            st["indirect"][ins.mnem] += 1
        elif ins.kind == "ret":
            st["indirect"]["ret"] += 1
        if ins.kind == "branch":
            st["branches"] += 1
    try:
        tr = x64_recomp.FunctionTranslator(good)
        tr.translate()
        st["translated"] = tr.stats["insns"]
        for u in tr.stats["unhandled"]:
            st["unhandled"][u.split(":", 1)[0].split("@")[0]] += 1
    except Exception as e:  # translator-wide failure still gets counted
        st["unhandled"][f"translator:{type(e).__name__}"] += 1
    return st


def census(path, limit=None):
    with open(path, "rb") as f:
        img = pe_inspect.PeImage(f.read())
    if img.machine != 0x8664:
        raise pe_inspect.PeFormatError(
            f"machine {hex(img.machine)} is not x86-64; nothing to recompile")
    entries = pdata_entries(img)
    if limit:
        entries = entries[:limit]

    mnem, unhandled, indirect, uw_flags = Counter(), Counter(), Counter(), Counter()
    uw_ops, uw_versions, uw_saved_gprs, uw_saved_xmms, uw_errors = (
        Counter(), Counter(), Counter(), Counter(), Counter(),
    )
    uw_handlers = set()
    uw_parsed = uw_malformed = uw_with_fp = uw_with_xmm = uw_with_handler = 0
    uw_with_machframe = uw_chained = uw_max_chain = uw_max_alloc = 0
    funcs = covered = partial = insns = translated = code_bytes = 0
    decode_errs = branches = 0
    worst = []
    for begin, end, uw in entries:
        st = census_function(img, begin, end, img.image_base)
        funcs += 1
        insns += st["insns"]
        translated += st["translated"]
        code_bytes += st["size"]
        decode_errs += st["decode_errors"]
        branches += st["branches"]
        mnem += st["mnem"]
        unhandled += st["unhandled"]
        indirect += st["indirect"]
        uw_detail = parse_unwind_info(img, uw)
        if uw_detail["version"] is not None:
            uw_flags[",".join(uw_detail["flags"]) or "none"] += 1
            uw_versions[str(uw_detail["version"])] += 1
        if uw_detail["error"] is None:
            uw_parsed += 1
        else:
            uw_malformed += 1
            uw_errors[uw_detail["error"].split(" at ", 1)[0]] += 1
        for op in uw_detail["ops"]:
            uw_ops[op["op"]] += 1
        if uw_detail["frame_register"] is not None:
            uw_with_fp += 1
        if uw_detail["saved_xmms"]:
            uw_with_xmm += 1
        if uw_detail["handler_rva"] is not None:
            uw_with_handler += 1
            uw_handlers.add(uw_detail["handler_rva"])
        if uw_detail["has_machframe"]:
            uw_with_machframe += 1
        if uw_detail["chain_depth"]:
            uw_chained += 1
            uw_max_chain = max(uw_max_chain, uw_detail["chain_depth"])
        uw_max_alloc = max(uw_max_alloc, uw_detail["stack_alloc"])
        for reg in uw_detail["saved_gprs"]:
            uw_saved_gprs[reg] += 1
        for reg in uw_detail["saved_xmms"]:
            uw_saved_xmms[reg] += 1
        if not st["unhandled"]:
            covered += 1
        else:
            partial += 1
            worst.append((len(st["unhandled"]) and sum(st["unhandled"].values()) or 0,
                          st["rva"], st["size"], dict(st["unhandled"])))
    worst.sort(reverse=True)
    cov = (translated / insns * 100) if insns else 0.0
    return {
        "file": os.path.basename(path),
        "image_base": hex(img.image_base),
        "pdata_functions": len(entries),
        "code_bytes": code_bytes,
        "insns": insns,
        "insn_coverage_pct": round(cov, 2),
        "functions_fully_covered": covered,
        "functions_with_gaps": partial,
        "decode_errors": decode_errs,
        "branch_insns": branches,
        "indirect_control_flow": dict(indirect),
        "unwind_flags": dict(uw_flags),
        "unwind": {
            "parsed": uw_parsed,
            "malformed": uw_malformed,
            "versions": dict(uw_versions),
            "ops": uw_ops.most_common(20),
            "with_frame_pointer": uw_with_fp,
            "with_saved_xmm": uw_with_xmm,
            "with_handler": uw_with_handler,
            "with_machframe": uw_with_machframe,
            "chained": uw_chained,
            "max_chain_depth": uw_max_chain,
            "max_stack_alloc": uw_max_alloc,
            "saved_gprs": dict(uw_saved_gprs),
            "saved_xmms": dict(uw_saved_xmms),
            "unique_handlers": len(uw_handlers),
            "errors": dict(uw_errors),
        },
        "top_mnemonics": mnem.most_common(40),
        "unhandled": unhandled.most_common(40),
        "worst_functions": [{"rva": hex(r), "size": s, "unhandled": u}
                            for _c, r, s, u in worst[:20]],
    }


def to_markdown(r):
    out = [f"# recomp census: {r['file']}", "",
           f"- .pdata functions: {r['pdata_functions']:,} covering {r['code_bytes']:,} code bytes",
           f"- decoded instructions: {r['insns']:,}; translated: "
           f"{r['insn_coverage_pct']}% ({r['functions_fully_covered']:,} functions fully covered, "
           f"{r['functions_with_gaps']:,} with gaps)",
           f"- decode errors (unknown/error insns): {r['decode_errors']:,}",
           f"- branch instructions: {r['branch_insns']:,}; indirect control flow: "
           + (", ".join(f"{k}={v}" for k, v in sorted(r["indirect_control_flow"].items())) or "none"),
           "", "## Unwind info flags", ""]
    for k, v in sorted(r["unwind_flags"].items(), key=lambda kv: -kv[1]):
        out.append(f"- {k}: {v:,}")
    uw = r.get("unwind")
    if uw:
        out += [
            "", "## Unwind (.xdata) details", "",
            f"- parsed: {uw['parsed']:,}; malformed: {uw['malformed']:,}; "
            f"chained: {uw['chained']:,} (max depth {uw['max_chain_depth']})",
            f"- frame pointer: {uw['with_frame_pointer']:,}; saved XMM: {uw['with_saved_xmm']:,}; "
            f"handlers: {uw['with_handler']:,} ({uw['unique_handlers']:,} unique); "
            f"machframe: {uw['with_machframe']:,}; max stack alloc: {uw['max_stack_alloc']:,} bytes",
        ]
        if uw["ops"]:
            out += ["", "| unwind op | count |", "|---|---|"]
            for k, v in uw["ops"]:
                out.append(f"| {k} | {v:,} |")
    out += ["", "## Top mnemonics", "", "| mnemonic | count |", "|---|---|"]
    for k, v in r["top_mnemonics"]:
        out.append(f"| {k} | {v:,} |")
    out += ["", "## Unhandled / hazard mnemonics", "", "| mnemonic | count |", "|---|---|"]
    for k, v in r["unhandled"]:
        out.append(f"| {k} | {v:,} |")
    out += ["", "## Worst functions", "", "| rva | size | unhandled |", "|---|---|---|"]
    for w in r["worst_functions"]:
        u = ", ".join(f"{k}={v}" for k, v in sorted(w["unhandled"].items()))
        out.append(f"| {w['rva']} | {w['size']} | {u} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="x86-64 -> arm64 recompiler census over .pdata")
    p.add_argument("image")
    p.add_argument("--out", default=".")
    p.add_argument("--limit", type=int, default=0, help="only inspect the first N .pdata functions")
    p.add_argument("--top", type=int, default=40)
    a = p.parse_args(argv)
    try:
        r = census(a.image, a.limit or None)
    except pe_inspect.PeFormatError as e:
        print(f"{a.image}: {e}", file=sys.stderr)
        return 1
    except OSError as e:  # missing image, a folder, no permission: one line, not a traceback
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    r["top_mnemonics"] = r["top_mnemonics"][:a.top]
    r["unhandled"] = r["unhandled"][:a.top]
    name = os.path.basename(a.image)
    try:  # --out naming a file, or a folder that cannot be written
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, name + ".recomp.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(r, f, indent=1)
        with open(os.path.join(a.out, name + ".recomp.md"), "w", encoding="utf-8", newline="\n") as f:
            f.write(to_markdown(r))
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"{r['file']}: {r['pdata_functions']:,} fns, {r['insns']:,} insns, "
          f"{r['insn_coverage_pct']}% covered, {len(r['unhandled'])} hazard kinds")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
