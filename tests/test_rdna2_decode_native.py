"""The optional Rust disassembler (tools/rdna2_decode_native) against `rdna2_decode.disasm_bytes` in Python.

Synthetic words only (no game data): every opcode of every table in `tools/rdna2_isa_tables.py`, put in its
encoding with random operand and modifier bits, inline constants, literals and NSA dwords, in wave32 and wave64,
for gfx1013 and gfx1030, whole and truncated, plus random byte streams. The comparisons skip when the library is
not built, or was built from an older Python decoder (its fingerprint no longer matches):

    cargo build --release --locked --offline --manifest-path tools/rdna2_decode_native/Cargo.toml
    python -I tests/test_rdna2_decode_native.py
"""
import os
import random
import struct
import subprocess
import sys
import unittest
from unittest import mock

# These tests set the per-module switch themselves; the global LEONIDA_NATIVE=0 opt-out would override it.
os.environ.pop("LEONIDA_NATIVE", None)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import rdna2_decode  # noqa: E402
import rdna2_isa_tables as T  # noqa: E402
import rdna2_native  # noqa: E402

G = vars(rdna2_decode)
BUILT = os.path.isfile(rdna2_native.library_path())
NATIVE = rdna2_native.load(G)
CONFIGS = (("gfx1013", True), ("gfx1013", False), ("gfx1030", True), ("gfx1030", False))

# Source operand codes worth over-sampling: SGPRs next to vcc/ttmp/exec, inline integers and floats, the DPP/SDWA
# markers, specials, the literal marker and VGPRs.
INTERESTING_SRC = (0, 1, 103, 104, 105, 106, 107, 108, 122, 123, 124, 125, 126, 127, 128, 129, 192, 193, 208,
                   209, 233, 234, 239, 240, 241, 242, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 511)


def python_disasm(code, **kwargs):
    with mock.patch.object(rdna2_decode, "_NATIVE", [None]):
        return rdna2_decode.disasm_bytes(code, **kwargs)


def src9(rng):
    return rng.choice(INTERESTING_SRC) if rng.random() < 0.6 else rng.getrandbits(9)


def src8(rng):
    return rng.choice([c for c in INTERESTING_SRC if c < 256]) if rng.random() < 0.6 else rng.getrandbits(8)


def encode(family, opc, rng):
    """One instruction word pair of `family` with opcode `opc`, every other field random."""
    w0 = rng.getrandbits(32)
    w1 = rng.getrandbits(32)
    if family == "VOP1":
        w0 = (0x3F << 25) | ((w0 >> 17 & 0xFF) << 17) | (opc << 9) | src9(rng)
    elif family == "VOPC":
        w0 = (0x3E << 25) | (opc << 17) | ((w0 >> 9 & 0xFF) << 9) | src9(rng)
    elif family == "VOP2":
        w0 = (opc << 25) | (w0 & 0x01FFFE00) | src9(rng)
    elif family == "SOP1":
        w0 = (0x17D << 23) | (w0 & 0x7F0000) | (opc << 8) | src8(rng)
    elif family == "SOPC":
        w0 = (0x17E << 23) | (opc << 16) | (src8(rng) << 8) | src8(rng)
    elif family == "SOPP":
        w0 = (0x17F << 23) | (opc << 16) | (w0 & 0xFFFF)
    elif family == "SOPK":
        w0 = (0xB << 28) | (opc << 23) | (w0 & 0x7FFFFF)
    elif family == "SOP2":
        w0 = (0b10 << 30) | (opc << 23) | (w0 & 0x7F0000) | (src8(rng) << 8) | src8(rng)
    elif family == "VINTRP":
        w0 = (0x32 << 26) | (w0 & 0x03FC0000) | (opc << 16) | (w0 & 0xFFFF)
    elif family == "VOP3P":
        w0 = (0x33 << 26) | (opc << 16) | (w0 & 0xFFFF)
        w1 = (w1 & 0xF8000000) | (src9(rng) << 18) | (src9(rng) << 9) | src9(rng)
    elif family in ("VOP3", "VOP3B"):
        w0 = (0x35 << 26) | (opc << 16) | (w0 & 0xFFFF)
        w1 = (w1 & 0xF8000000) | (src9(rng) << 18) | (src9(rng) << 9) | src9(rng)
    elif family == "DS":
        w0 = (0x36 << 26) | (opc << 18) | (w0 & 0x3FFFF)
    elif family in ("FLAT", "SCRATCH", "GLOBAL"):
        seg = {"FLAT": 0, "SCRATCH": 1, "GLOBAL": 2}[family]
        w0 = (0x37 << 26) | (opc << 18) | (w0 & 0x30000) | (seg << 14) | (w0 & 0x3FFF)
        if rng.random() < 0.5:
            w1 = (w1 & 0xFF80FFFF) | (rng.choice((0x7D, 0x7F, 0, 2, 104)) << 16)
    elif family == "MUBUF":
        w0 = (0x38 << 26) | (opc << 18) | (w0 & 0x3FFFF)
        w1 = (w1 & 0x00FFFFFF) | (src8(rng) << 24)
    elif family == "MTBUF":
        w0 = (0x3A << 26) | (w0 & 0x03F80000) | ((opc & 7) << 16) | (w0 & 0xFFFF)
        w1 = (w1 & 0x00DFFFFF) | ((opc >> 3 & 1) << 21) | (src8(rng) << 24)
    elif family == "MIMG":
        w0 = (0x3C << 26) | (w0 & 0x02000000) | ((opc & 0x7F) << 18) | (w0 & 0x3FFFE) | (opc >> 7)
    elif family == "SMEM":
        w0 = (0x3D << 26) | (opc << 18) | (w0 & 0x3FFFF)
        if rng.random() < 0.5:
            w1 = (w1 & 0x01FFFFFF) | (rng.choice((125, 0, 5, 124)) << 25)
    elif family == "EXP":
        w0 = (0x3E << 26) | (w0 & 0x3FFFFFF)
    else:
        raise ValueError(family)
    return [w0 & 0xFFFFFFFF, w1 & 0xFFFFFFFF]


FAMILIES = ["DS", "FLAT", "GLOBAL", "MIMG", "MTBUF", "MUBUF", "SCRATCH", "SMEM", "SOP1", "SOP2", "SOPC", "SOPK",
            "SOPP", "VINTRP", "VOP1", "VOP2", "VOP3", "VOP3B", "VOP3P", "VOPC"]


def corpus(rng, variants):
    """`(family, opcode, bytes)` chunks: each opcode `variants` times, with three random trailing words."""
    out = []
    for family in FAMILIES:
        for opc in sorted(getattr(T, family)):
            for _ in range(variants):
                words = encode(family, opc, rng) + [rng.getrandbits(32) for _ in range(3)]
                if rng.random() < 0.25:
                    words[2] = rng.choice((0, 0x3F800000, 0x12345678, 0xFFFFFFFF, 99, 100))
                out.append((family, opc, struct.pack(f"<{len(words)}I", *words)))
    for _ in range(variants * 8):
        out.append(("EXP", 0, struct.pack("<5I", *encode("EXP", 0, rng), *(rng.getrandbits(32) for _ in range(3)))))
    return out


def opcode(family, name):
    return next(opc for opc, (n, _) in getattr(T, family).items() if n == name)


def pack(words):
    return struct.pack(f"<{len(words)}I", *words)


def sweeps(rng):
    """`(label, bytes, wave_sensitive)` streams that walk the structured fields one value at a time.

    Every instruction in a stream has its full length (literals and NSA dwords included), so each stays aligned.
    """
    out = []
    vop1_mov = opcode("VOP1", "v_mov_b32")
    vop2_add = opcode("VOP2", "v_add_f32")
    vop2_cnd = opcode("VOP2", "v_cndmask_b32")
    vopc_cmp = opcode("VOPC", "v_cmp_lt_f32")

    words = []
    for ctrl in range(512):  # DPP16: every dpp_ctrl, random masks / bound_ctrl / fi / neg / abs
        for w0 in ((0x3F << 25) | (rng.getrandbits(8) << 17) | (vop1_mov << 9) | 250,
                   (vop2_add << 25) | (rng.getrandbits(16) << 9) | 250):
            words += [w0, (rng.getrandbits(32) & ~(0x1FF << 8)) | (ctrl << 8)]
    for _ in range(512):  # DPP8 and SDWA on VOP1 / VOP2 / VOPC
        for src0 in (233, 234, 249):
            words += [(0x3F << 25) | (rng.getrandbits(8) << 17) | (vop1_mov << 9) | src0, rng.getrandbits(32)]
            words += [(rng.choice((vop2_add, vop2_cnd)) << 25) | (rng.getrandbits(16) << 9) | src0,
                      rng.getrandbits(32)]
        words += [(0x3E << 25) | (vopc_cmp << 17) | (rng.getrandbits(8) << 9) | 249, rng.getrandbits(32)]
    out.append(("dpp/sdwa", pack(words), True))

    simm = list(range(0x10000))
    for name in ("s_waitcnt", "s_sendmsg", "s_branch", "s_ttracedata_imm"):
        out.append((name, pack([(0x17F << 23) | (opcode("SOPP", name) << 16) | v for v in simm]), False))
    for name in ("s_getreg_b32", "s_movk_i32"):
        out.append((name, pack([(0xB << 28) | (opcode("SOPK", name) << 23) | (rng.getrandbits(7) << 16) | v
                                for v in simm]), False))
    swz = opcode("DS", "ds_swizzle_b32")
    out.append(("ds_swizzle", pack([w for v in simm for w in ((0x36 << 26) | (swz << 18) | v, rng.getrandbits(32))]),
                False))

    words = []
    for code in range(512):  # every source code in each source slot, at 32 and 64 bits
        lit = [rng.choice((0x3F800000, 0x12345678, 7, 100))] if code in (233, 234, 249, 250, 255) else []
        words += [(vop2_add << 25) | (rng.getrandbits(16) << 9) | code] + lit
        words += [(0x3F << 25) | (rng.getrandbits(8) << 17) | (opcode("VOP1", "v_cvt_f64_f32") << 9) | code] + lit
        for name in ("v_fma_f32", "v_fma_f64", "v_add_f64", "v_cmp_lt_f32", "v_add_co_u32", "v_readlane_b32",
                     "v_cndmask_b32"):
            vop3 = opcode("VOP3B", name) if name in [n for n, _ in T.VOP3B.values()] else opcode("VOP3", name)
            for slot in range(3):
                srcs = [rng.getrandbits(8) + 256 for _ in range(3)]
                srcs[slot] = code
                lit = [rng.getrandbits(32)] if 255 in srcs else []
                words += [(0x35 << 26) | (vop3 << 16) | rng.getrandbits(16),
                          (rng.getrandbits(5) << 27) | (srcs[2] << 18) | (srcs[1] << 9) | srcs[0]] + lit
        for slot in range(3):
            srcs = [rng.getrandbits(9) for _ in range(3)]
            srcs[slot] = code
            lit = [rng.getrandbits(32)] if 255 in srcs else []
            words += [(0x33 << 26) | (opcode("VOP3P", rng.choice(("v_pk_fma_f16", "v_fma_mix_f32"))) << 16)
                      | rng.getrandbits(16), (rng.getrandbits(5) << 27) | (srcs[2] << 18) | (srcs[1] << 9) | srcs[0]]
            words += lit
    for code in range(256):
        for name in ("s_add_u32", "s_and_b64", "s_lshl_b64"):
            other = rng.getrandbits(8) if rng.random() < 0.5 else rng.choice((106, 126, 104))
            lit = [rng.getrandbits(32)] if 255 in (code, other) else []
            words += [(0b10 << 30) | (opcode("SOP2", name) << 23) | (rng.getrandbits(7) << 16) | (other << 8) | code]
            words += lit
        for name in ("s_mov_b32", "s_mov_b64"):
            lit = [rng.getrandbits(32)] if code == 255 else []
            words += [(0x17D << 23) | (code & 0x7F) << 16 | (opcode("SOP1", name) << 8) | code] + lit
        lit = [rng.getrandbits(32)] if code == 255 else []
        words += [(0x17E << 23) | (opcode("SOPC", "s_cmp_eq_u64") << 16) | (rng.getrandbits(8) << 8) | code] + lit
    out.append(("sources", pack(words), True))

    words = []
    for name in ("s_load_dword", "s_load_dwordx2", "s_load_dwordx4", "s_load_dwordx8", "s_load_dwordx16",
                 "s_buffer_load_dwordx4"):
        for sdata in range(128):
            soff = rng.choice((125, 125, 0, 106, 124, rng.getrandbits(7)))
            off = rng.choice((0, 1, 0xFFFFF, 0x100000, 0x1FFFFF, rng.getrandbits(21)))
            words += [(0x3D << 26) | (opcode("SMEM", name) << 18) | (rng.getrandbits(2) << 14) | (sdata << 6)
                      | rng.getrandbits(6), (soff << 25) | off]
    for fmt in range(128):
        for _ in range(4):
            opc = rng.choice(sorted(T.MTBUF))
            words += [(0x3A << 26) | (fmt << 19) | ((opc & 7) << 16) | rng.getrandbits(16),
                      (rng.getrandbits(32) & ~(1 << 21)) | ((opc >> 3 & 1) << 21)]
    for seg, table in ((0, T.FLAT), (1, T.SCRATCH), (2, T.GLOBAL)):
        for opc in sorted(table):
            for saddr in rng.sample(range(128), 6) + [0x7D, 0x7F]:
                words += [(0x37 << 26) | (opc << 18) | (rng.getrandbits(2) << 16) | (seg << 14) | rng.getrandbits(14),
                          (rng.getrandbits(32) & ~(0x7F << 16)) | (saddr << 16)]
    for tgt in range(64):
        for en in range(16):
            words += [(0x3E << 26) | (rng.getrandbits(3) << 10 & 0x1C00) | (tgt << 4) | en, rng.getrandbits(32)]
    for opc in sorted(T.VINTRP):
        for vsrc in range(256):
            words += [(0x32 << 26) | (rng.getrandbits(8) << 18) | (opc << 16) | (rng.getrandbits(8) << 8) | vsrc]
    out.append(("memory/export", pack(words), True))

    words = []
    for opc in sorted(T.MIMG):
        for dim in range(8):
            for a16 in (0, 1):
                for d16 in (0, 1):
                    nsa = rng.randrange(4)
                    w0 = (0x3C << 26) | (rng.getrandbits(1) << 25) | ((opc & 0x7F) << 18) | (rng.getrandbits(10) << 8)
                    w0 |= (rng.getrandbits(1) << 7) | (dim << 3) | (nsa << 1) | (opc >> 7)
                    w1 = (d16 << 31) | (a16 << 30) | rng.getrandbits(30)
                    words += [w0, w1] + [rng.getrandbits(32) for _ in range(nsa)]
    out.append(("mimg", pack(words), True))
    return out


@unittest.skipIf(NATIVE is None, "tools/rdna2_decode_native is not built, or is older than tools/rdna2_decode.py")
class NativeMatchesPython(unittest.TestCase):
    def assert_same(self, code, mcpu, wave32, label=""):
        expected = python_disasm(code, mcpu=mcpu, wave32=wave32)
        native = NATIVE(code, mcpu, wave32)
        self.assertIsNotNone(native, f"native refused {label} {code.hex()}")
        if native != expected:
            self.fail(f"{label} {mcpu} wave32={wave32} {code.hex()}\nnative:\n{native[0]}{native[1]}\n"
                      f"python:\n{expected[0]}{expected[1]}")

    def test_every_opcode_of_every_family(self):
        rng = random.Random(0x0D1A)
        chunks = corpus(rng, 16)
        self.assertGreater(len(chunks), 20000)  # 1,470 opcodes x 16, plus 128 exports
        for i, (family, opc, code) in enumerate(chunks):
            mcpu, wave32 = CONFIGS[i % len(CONFIGS)]
            self.assert_same(code, mcpu, wave32, f"{family} 0x{opc:x}")

    def test_field_sweeps(self):
        for i, (label, code, wave_sensitive) in enumerate(sweeps(random.Random(0xF1E1D))):
            for mcpu, wave32 in (CONFIGS if wave_sensitive else CONFIGS[i % 2:i % 2 + 1]):
                with self.subTest(sweep=label, mcpu=mcpu, wave32=wave32):
                    text, invalid = python_disasm(code, mcpu=mcpu, wave32=wave32)
                    self.assertLess(invalid, len(code) // 400, f"{label}: the sweep should decode")
                    self.assert_same(code, mcpu, wave32, label)

    def test_both_wave_sizes_and_targets_per_opcode(self):
        rng = random.Random(0x1030)
        for family, opc, code in corpus(rng, 2):
            for mcpu, wave32 in CONFIGS:
                self.assert_same(code, mcpu, wave32, f"{family} 0x{opc:x}")

    def test_truncated_instructions(self):
        rng = random.Random(0x7A11)
        for family, opc, code in corpus(rng, 2):
            cut = rng.randrange(0, len(code))
            mcpu, wave32 = rng.choice(CONFIGS)
            self.assert_same(code[:cut], mcpu, wave32, f"{family} 0x{opc:x} cut {cut}")

    def test_one_long_stream(self):
        rng = random.Random(0x5EED)
        code = b"".join(c for _, _, c in corpus(rng, 3))
        for mcpu, wave32 in CONFIGS:
            self.assert_same(code, mcpu, wave32, "stream")

    def test_random_bytes(self):
        rng = random.Random(0xB17E5)
        for size in (0, 1, 2, 3, 4, 5, 7, 8, 12, 4093, 65536, 65539):
            code = rng.randbytes(size)
            for mcpu, wave32 in CONFIGS:
                self.assert_same(code, mcpu, wave32, f"random {size}")

    def test_hand_picked_words(self):
        # s_waitcnt, s_cbranch_scc1 -4, s_ttracedata_imm 0xffff, s_mov_b32 with a literal, v_add_f32 1.0, exp.
        words = [0xBF8C0070, 0xBF85FFFC, 0xBFA8FFFF, 0xBE8403FF, 0x12345678, 0x060204F2, 0xF800180F, 0x03020100,
                 0xBF810000]
        code = struct.pack(f"<{len(words)}I", *words)
        for mcpu, wave32 in CONFIGS:
            self.assert_same(code, mcpu, wave32, "hand-picked")

    def test_disasm_bytes_uses_the_native_path_for_every_buffer_type(self):
        code = struct.pack("<3I", 0xBE8403FF, 0x12345678, 0xBF810000)
        expected = python_disasm(code)
        calls = []

        def native(*args):
            calls.append(args)
            return NATIVE(*args)

        with mock.patch.object(rdna2_decode, "_NATIVE", [native]):
            for buf in (code, bytearray(code), memoryview(code)):
                with self.subTest(type=type(buf).__name__):
                    self.assertEqual(rdna2_decode.disasm_bytes(buf), expected)
                    self.assertEqual(NATIVE(buf, "gfx1013", True), expected)
        self.assertEqual(len(calls), 3)


class Fallback(unittest.TestCase):
    def test_disabled_by_environment(self):
        code = (
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import rdna2_decode\n"
            "text, bad = rdna2_decode.disasm_bytes(bytes.fromhex('70008cbf000081bf'))\n"
            "assert rdna2_decode._NATIVE == [None], rdna2_decode._NATIVE\n"
            "sys.stdout.write(text)\n"
        )
        env = dict(os.environ, LEONIDA_RDNA2_DECODE_NATIVE="0")
        result = subprocess.run([sys.executable, "-I", "-c", code, TOOLS], env=env, capture_output=True,
                                text=True, check=True)
        self.assertEqual(result.stdout, python_disasm(bytes.fromhex("70008cbf000081bf"))[0])

    def test_missing_library(self):
        with mock.patch.object(rdna2_native.os.path, "isfile", return_value=False):
            self.assertIsNone(rdna2_native.load(G))

    def test_changed_decoder_source_refuses_the_library(self):
        with mock.patch.object(rdna2_native, "fingerprint", return_value="0" * 64):
            self.assertIsNone(rdna2_native.load(G))

    @unittest.skipIf(NATIVE is None, "tools/rdna2_decode_native is not built or is stale")
    def test_wrapped_decoder_uses_python(self):
        code = struct.pack("<2I", 0xBF800000, 0xBF810000)
        expected = python_disasm(code)
        with mock.patch.object(rdna2_decode, "_NATIVE", [NATIVE]), \
                mock.patch.object(rdna2_decode, "_decode_one", wraps=rdna2_decode._decode_one) as decode:
            self.assertEqual(rdna2_decode.disasm_bytes(code), expected)
        self.assertEqual(decode.call_count, 2)

    @unittest.skipIf(NATIVE is None, "tools/rdna2_decode_native is not built or is stale")
    def test_unusual_arguments_use_python(self):
        self.assertIsNone(NATIVE([0xBF810000], "gfx1013", True))
        self.assertIsNone(NATIVE(b"\0\0\x81\xbf", None, True))
        self.assertIsNone(NATIVE(memoryview(struct.pack("<I", 0xBF810000)).cast("I"), "gfx1013", True))

    @unittest.skipIf(NATIVE is None, "tools/rdna2_decode_native is not built or is stale")
    def test_signed_and_char_memoryviews_use_python(self):
        raw = bytes.fromhex("6a0c49f2f04ec2d9e9e0b0ce")
        for fmt in ("b", "c"):
            view = memoryview(raw).cast(fmt)
            with self.subTest(format=fmt):
                self.assertIsNone(NATIVE(view, "gfx1013", True))
                try:
                    expected = python_disasm(view)
                except Exception as exc:  # Python's own result for these views must be reproduced exactly
                    with mock.patch.object(rdna2_decode, "_NATIVE", [NATIVE]):
                        with self.assertRaises(type(exc)) as caught:
                            rdna2_decode.disasm_bytes(view)
                    self.assertEqual(str(caught.exception), str(exc))
                else:
                    with mock.patch.object(rdna2_decode, "_NATIVE", [NATIVE]):
                        self.assertEqual(rdna2_decode.disasm_bytes(view), expected)

    def test_refused_call_falls_back(self):
        code = struct.pack("<2I", 0xBF800000, 0xBF810000)
        with mock.patch.object(rdna2_decode, "_NATIVE", [lambda *args: None]):
            self.assertEqual(rdna2_decode.disasm_bytes(code), python_disasm(code))

    def test_table_blob_covers_every_table(self):
        blob = rdna2_native.table_blob(G).decode("utf-8")
        rows = [line.split("\t") for line in blob.splitlines()]
        for family in FAMILIES:
            with self.subTest(family=family):
                self.assertEqual(sum(1 for r in rows if r[0] == "T" and r[1] == family), len(getattr(T, family)))

    @unittest.skipUnless(BUILT, "tools/rdna2_decode_native is not built")
    def test_built_library_reports_its_fingerprint(self):
        import ctypes
        lib = ctypes.CDLL(rdna2_native.library_path())
        lib.rdna2_native_fingerprint.restype = ctypes.c_char_p
        built = lib.rdna2_native_fingerprint().decode("ascii")
        current = rdna2_native.fingerprint(G)
        if built != current:
            self.skipTest(f"the library ports decoder {built[:12]}, tools/rdna2_decode.py is now {current[:12]}: "
                          "port the change to src/lib.rs and update FINGERPRINT")
        self.assertIsNotNone(NATIVE)


if __name__ == "__main__":
    unittest.main()
