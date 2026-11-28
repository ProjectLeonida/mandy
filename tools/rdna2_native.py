"""Loader of the optional Rust build of `rdna2_decode.disasm_bytes` (`tools/rdna2_decode_native`).

`rdna2_decode.disasm_bytes` calls `load(globals())` once and then the function it returns, which gives
`(listing, invalid)` or None; on None the listing is made in Python. The library is used only when:

- `LEONIDA_RDNA2_DECODE_NATIVE` is not `0` and `target/release/` holds the built library;
- the SHA-256 of the Python decoder functions it ports (`fingerprint`) is the one the library was built from, so a
  stale build, or a decoder change not yet ported to `src/lib.rs`, falls back to Python instead of printing
  different text;
- `_decode_one`, `format_insn` and `_iter_insns` are still the module's own (a test may wrap them).

The opcode tables and name maps are not duplicated in Rust: `table_blob` serialises the Python ones.
Standard library only.
"""
from __future__ import annotations

import ctypes
import hashlib
import inspect
import os
import sys

ENV = "LEONIDA_RDNA2_DECODE_NATIVE"

# The functions of rdna2_decode.py that src/lib.rs reproduces; disasm_bytes itself only dispatches.
PORTED = (
    "_s16", "_s12", "_s21", "_sreg", "_vreg", "_src", "_vcc_operand", "_width", "_format_hwreg",
    "_format_waitcnt", "_format_sendmsg", "_format_dpp16_ctrl", "_format_ds_swizzle", "_mimg_addr_count",
    "_mimg_data_count", "_decode_one", "_format_operand", "format_insn", "_iter_insns",
)
_WATCHED = ("_decode_one", "format_insn", "_iter_insns")

TABLES = (
    "DS", "FLAT", "GLOBAL", "MIMG", "MTBUF", "MUBUF", "SCRATCH", "SMEM", "SOP1", "SOP2", "SOPC", "SOPK", "SOPP",
    "VINTRP", "VOP1", "VOP2", "VOP3", "VOP3B", "VOP3P", "VOPC",
)


def library_path() -> str:
    name = {"win32": "rdna2_decode_native.dll",
            "darwin": "librdna2_decode_native.dylib"}.get(sys.platform, "librdna2_decode_native.so")
    # The workspace build (tools/native) first, then a build of the crate on its own.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "native", "target", "release", name),
                 os.path.join(here, "rdna2_decode_native", "target", "release", name)):
        if os.path.isfile(path):
            break
    return path


def fingerprint(g: dict) -> str:
    """SHA-256 of the source of the ported functions in the rdna2_decode namespace `g`."""
    h = hashlib.sha256()
    for name in PORTED:
        h.update(inspect.getsource(g[name]).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _field(value) -> str:
    text = str(value)
    if "\t" in text or "\n" in text:
        raise ValueError(f"unexpected separator in {text!r}")
    return text


def table_blob(g: dict) -> bytes:
    """The opcode tables and name maps of rdna2_decode, in the line format `Ctx::parse` reads."""
    lines = []
    tables = g["T"]
    for tname in TABLES:
        for opc, (name, spec) in sorted(getattr(tables, tname).items()):
            fields = ",".join(f"{_field(item[0])}:{int(item[1])}" for item in spec)
            lines.append(f"T\t{tname}\t{int(opc)}\t{_field(name)}\t{fields}")
    maps = {
        "HWREG": g["_HWREG_NAMES"], "SENDMSG": g["_SENDMSG_NAMES"], "GS_OPS": g["_GS_OPS"],
        "MIMG_DIMS": g["_MIMG_DIMS"], "SDWA_SELS": g["_SDWA_SELS"], "SDWA_UNUSED": g["_SDWA_UNUSED"],
        "OMOD": g["_OMOD_MODS"],
        # The text _format_operand prints for each inline float code _src accepts.
        "INLINE_FLOAT": {code: g["_format_operand"]({"k": "fimm", "v": v}) for code, v in g["_INLINE_FLOAT"].items()},
        # _decode_one prints a format only when its name is non-empty, shortened this way.
        "FORMATS": {code: name.replace("GFX10_FORMAT_", "BUF_FMT_")
                    for code, name in g["_GFX10_FORMATS"].items() if name},
    }
    for mname, mapping in maps.items():
        for key, text in sorted(mapping.items()):
            lines.append(f"M\t{mname}\t{int(key)}\t{_field(text)}")
    for mname, mapping in (("DIM_COORDS", g["_MIMG_DIM_COORDS"]), ("GRAD_COORDS", g["_MIMG_GRAD_COORDS"])):
        for key, value in sorted(mapping.items()):
            lines.append(f"I\t{mname}\t{int(key)}\t{int(value)}")
    for sname, members in (("VOP2_LITERAL", g["_VOP2_MANDATORY_LITERAL"]),
                           ("SOPK_LITERAL", g["_SOPK_MANDATORY_LITERAL"])):
        lines.extend(f"S\t{sname}\t{int(m)}" for m in sorted(members))
    for sname, members in (("SOPP_BRANCHES", g["_SOPP_BRANCHES"]), ("SOPP_NO_OPERAND", g["_SOPP_NO_OPERAND"])):
        lines.extend(f"S\t{sname}\t{_field(m)}" for m in sorted(members))
    return ("\n".join(lines) + "\n").encode("utf-8")


def load(g: dict):
    """The native `disasm(code, mcpu, wave32)` for the rdna2_decode namespace `g`, or None."""
    if "0" in (os.environ.get("LEONIDA_NATIVE", "1"), os.environ.get(ENV, "1")):
        return None
    path = library_path()
    if not os.path.isfile(path):
        return None
    try:
        lib = ctypes.CDLL(path)
        lib.rdna2_native_fingerprint.restype = ctypes.c_char_p
        lib.rdna2_native_fingerprint.argtypes = []
        if lib.rdna2_native_fingerprint().decode("ascii") != fingerprint(g):
            return None
        lib.rdna2_native_init.restype = ctypes.c_void_p
        lib.rdna2_native_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        disasm = lib.rdna2_native_disasm
        disasm.restype = ctypes.c_int
        disasm.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
                           ctypes.POINTER(ctypes.c_size_t)]
        free = lib.rdna2_native_free
        free.restype = None
        free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        blob = table_blob(g)
        ctx = lib.rdna2_native_init(blob, len(blob))
    except (OSError, AttributeError, KeyError, TypeError, ValueError, UnicodeError):
        return None
    if not ctx:
        return None
    originals = tuple(g[name] for name in _WATCHED)

    def run(code, mcpu, wave32):
        if any(g[name] is not fn for name, fn in zip(_WATCHED, originals)) or not isinstance(mcpu, str):
            return None
        if isinstance(code, memoryview):
            # Only format 'B': Python indexes 'b' and 'c' views as signed ints / 1-byte strings, so leave them to it.
            if code.format != "B" or code.ndim != 1 or code.itemsize != 1 or not code.c_contiguous:
                return None
            code = code.tobytes()
        elif isinstance(code, bytearray):
            code = bytes(code)
        elif not isinstance(code, bytes):
            return None
        out = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        invalid = ctypes.c_size_t()
        if disasm(ctx, code, len(code), 1 if wave32 else 0, 1 if mcpu.startswith("gfx101") else 0,
                  ctypes.byref(out), ctypes.byref(out_len), ctypes.byref(invalid)) != 0:
            return None
        try:
            text = ctypes.string_at(out, out_len.value).decode("utf-8") if out_len.value else ""
        finally:
            free(out, out_len.value)
        return text, invalid.value

    run.library = lib  # keeps the DLL loaded for as long as `run` lives
    return run
