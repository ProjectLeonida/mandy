"""Unit tests for SSE/SSE2 scalar floating-point and XMM move/bitwise support in tools/x64_recomp.py.

Covers four layers without game files or third-party packages:

  * SSEDecodeTests: prefix-aware legacy 0F SSE/SSE2 decoding (including REX.R/B/W,
    memory widths, operand order, and MMX non-aliasing),
  * Arm64FPEncodingTests: exact 32-bit arm64 FP/SIMD encodings in Asm,
  * SSETranslateTests: zero-unhandled translation, ucomis* flag-liveness paths,
    and EL1 FP/SIMD enable in build_test_kernel,
  * SSEExecTests: end-to-end execution on arm64 hardware (via the Mach-O runner
    when present) or qemu-system-aarch64, checking IEEE-754 arithmetic, upper-lane
    preservation, NaN/signed-zero min/max, ordered/unordered ucomis* flags,
    conversions + INT_MIN indefinite, and 128-bit moves/bitwise ops.

    python -I tests/test_x64_recomp_sse.py
"""
import os
import shutil
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import x64_recomp as x  # noqa: E402

QEMU = os.environ.get(
    "QEMU_AARCH64",
    "/home/nolan/.nix-profile/bin/qemu-system-aarch64"
    if os.path.exists("/home/nolan/.nix-profile/bin/qemu-system-aarch64")
    else shutil.which("qemu-system-aarch64") or "")
HAS_QEMU = bool(QEMU) and os.path.exists(QEMU)
USE_MACHO = bool(getattr(x, "CAN_RUN_MACHO", False)) and not os.environ.get("QEMU_AARCH64")
CAN_RUN = USE_MACHO or HAS_QEMU

GUEST_VA = 0x46000000
EXEC_SCRATCH = (getattr(x, "DARWIN_RW", 0x47000000) + 0x100000) if USE_MACHO else 0x47100000


def asm(*hexes):
    return bytes.fromhex("".join(h.replace(" ", "") for h in hexes))


def f32_hex(val):
    return struct.pack("<f", val).hex()


def f64_hex(val):
    return struct.pack("<d", val).hex()


def u32_bits(val):
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u64_bits(val):
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def run(code):
    return x.run_result(code, GUEST_VA)


class SSEDecodeTests(unittest.TestCase):
    def one(self, code, base=0x1000):
        return x.decode_one(code, 0, len(code), base)

    def test_scalar_and_vector_moves(self):
        ins = self.one(asm("f3 0f 10 c1"))          # movss xmm0, xmm1
        self.assertEqual((ins.mnem, ins.kind, ins.size), ("movss", "simd", 4))
        self.assertEqual(ins.ops, (("x", 0, 4), ("x", 1, 4)))

        ins = self.one(asm("f3 45 0f 10 c7"))       # movss xmm8, xmm15
        self.assertEqual(ins.mnem, "movss")
        self.assertEqual(ins.ops, (("x", 8, 4), ("x", 15, 4)))
        self.assertIn("xmm8/4", repr(ins))

        ins = self.one(asm("f3 0f 10 07"))          # movss xmm0, [rdi]
        self.assertEqual(ins.ops[0], ("x", 0, 4))
        self.assertEqual(ins.ops[1][0], "m")
        self.assertEqual(ins.ops[1][1]["size"], 4)

        ins = self.one(asm("f3 0f 11 07"))          # movss [rdi], xmm0
        self.assertEqual(ins.ops[0][0], "m")
        self.assertEqual(ins.ops[0][1]["size"], 4)
        self.assertEqual(ins.ops[1], ("x", 0, 4))

        ins = self.one(asm("f2 0f 10 07"))          # movsd xmm0, [rdi]
        self.assertEqual((ins.mnem, ins.ops[0], ins.ops[1][1]["size"]), ("movsd", ("x", 0, 8), 8))

        ins = self.one(asm("f2 0f 11 07"))          # movsd [rdi], xmm0
        self.assertEqual((ins.mnem, ins.ops[0][1]["size"], ins.ops[1]), ("movsd", 8, ("x", 0, 8)))

        for hex_bytes, mnem, dst_is_mem in (
            ("0f 28 c1", "movaps", False),
            ("0f 28 07", "movaps", False),
            ("0f 29 07", "movaps", True),
            ("0f 10 07", "movups", False),
            ("0f 11 07", "movups", True),
            ("66 0f 28 c1", "movapd", False),
            ("66 0f 10 c1", "movupd", False),
            ("66 0f 6f c1", "movdqa", False),
            ("66 0f 7f 07", "movdqa", True),
            ("f3 0f 6f 07", "movdqu", False),
            ("f3 0f 7f 07", "movdqu", True),
        ):
            ins = self.one(asm(hex_bytes))
            self.assertEqual((ins.mnem, ins.kind), (mnem, "simd"), hex_bytes)
            if dst_is_mem:
                self.assertEqual((ins.ops[0][0], ins.ops[0][1]["size"], ins.ops[1]), ("m", 16, ("x", 0, 16)))

    def test_movd_movq_variants(self):
        ins = self.one(asm("66 0f 6e c0"))          # movd xmm0, eax
        self.assertEqual((ins.mnem, ins.ops), ("movd", (("x", 0, 4), ("r", 0, 4))))

        ins = self.one(asm("66 48 0f 6e c0"))       # movq xmm0, rax
        self.assertEqual((ins.mnem, ins.ops), ("movq", (("x", 0, 8), ("r", 0, 8))))

        ins = self.one(asm("66 0f 7e c0"))          # movd eax, xmm0
        self.assertEqual((ins.mnem, ins.ops), ("movd", (("r", 0, 4), ("x", 0, 4))))

        ins = self.one(asm("66 48 0f 7e c0"))       # movq rax, xmm0
        self.assertEqual((ins.mnem, ins.ops), ("movq", (("r", 0, 8), ("x", 0, 8))))

        ins = self.one(asm("f3 0f 7e c8"))          # movq xmm1, xmm0
        self.assertEqual((ins.mnem, ins.ops), ("movq", (("x", 1, 8), ("x", 0, 8))))

        ins = self.one(asm("66 0f d6 07"))          # movq [rdi], xmm0
        self.assertEqual((ins.mnem, ins.ops[0][0], ins.ops[0][1]["size"], ins.ops[1]),
                         ("movq", "m", 8, ("x", 0, 8)))

    def test_arith_compare_cvt_and_bitwise(self):
        cases = (
            ("f3 0f 58 c1", "addss", (("x", 0, 4), ("x", 1, 4))),
            ("f2 0f 58 c1", "addsd", (("x", 0, 8), ("x", 1, 8))),
            ("f3 0f 5c c1", "subss", (("x", 0, 4), ("x", 1, 4))),
            ("f2 0f 5c c1", "subsd", (("x", 0, 8), ("x", 1, 8))),
            ("f3 0f 59 c1", "mulss", (("x", 0, 4), ("x", 1, 4))),
            ("f2 0f 59 c1", "mulsd", (("x", 0, 8), ("x", 1, 8))),
            ("f3 0f 5e c1", "divss", (("x", 0, 4), ("x", 1, 4))),
            ("f2 0f 5e c1", "divsd", (("x", 0, 8), ("x", 1, 8))),
            ("f3 0f 51 c1", "sqrtss", (("x", 0, 4), ("x", 1, 4))),
            ("f2 0f 51 c1", "sqrtsd", (("x", 0, 8), ("x", 1, 8))),
            ("f3 0f 5d c1", "minss", (("x", 0, 4), ("x", 1, 4))),
            ("f2 0f 5d c1", "minsd", (("x", 0, 8), ("x", 1, 8))),
            ("f3 0f 5f c1", "maxss", (("x", 0, 4), ("x", 1, 4))),
            ("f2 0f 5f c1", "maxsd", (("x", 0, 8), ("x", 1, 8))),
            ("0f 2e c1", "ucomiss", (("x", 0, 4), ("x", 1, 4))),
            ("66 0f 2e c1", "ucomisd", (("x", 0, 8), ("x", 1, 8))),
            ("0f 2f c1", "comiss", (("x", 0, 4), ("x", 1, 4))),
            ("66 0f 2f c1", "comisd", (("x", 0, 8), ("x", 1, 8))),
            ("f3 0f 5a c1", "cvtss2sd", (("x", 0, 8), ("x", 1, 4))),
            ("f2 0f 5a c1", "cvtsd2ss", (("x", 0, 4), ("x", 1, 8))),
            ("f3 0f 2a c0", "cvtsi2ss", (("x", 0, 4), ("r", 0, 4))),
            ("f3 48 0f 2a c0", "cvtsi2ss", (("x", 0, 4), ("r", 0, 8))),
            ("f2 0f 2a c0", "cvtsi2sd", (("x", 0, 8), ("r", 0, 4))),
            ("f2 48 0f 2a c0", "cvtsi2sd", (("x", 0, 8), ("r", 0, 8))),
            ("f3 0f 2c c0", "cvttss2si", (("r", 0, 4, False), ("x", 0, 4))),
            ("f3 48 0f 2c c0", "cvttss2si", (("r", 0, 8, False), ("x", 0, 4))),
            ("f2 0f 2c c0", "cvttsd2si", (("r", 0, 4, False), ("x", 0, 8))),
            ("f2 48 0f 2c c0", "cvttsd2si", (("r", 0, 8, False), ("x", 0, 8))),
            ("f3 0f 2d c0", "cvtss2si", (("r", 0, 4, False), ("x", 0, 4))),
            ("f2 48 0f 2d c0", "cvtsd2si", (("r", 0, 8, False), ("x", 0, 8))),
            ("0f 57 c0", "xorps", (("x", 0, 16), ("x", 0, 16))),
            ("66 0f 57 c1", "xorpd", (("x", 0, 16), ("x", 1, 16))),
            ("0f 54 c1", "andps", (("x", 0, 16), ("x", 1, 16))),
            ("66 0f 54 c1", "andpd", (("x", 0, 16), ("x", 1, 16))),
            ("0f 55 c1", "andnps", (("x", 0, 16), ("x", 1, 16))),
            ("66 0f 55 c1", "andnpd", (("x", 0, 16), ("x", 1, 16))),
            ("0f 56 c1", "orps", (("x", 0, 16), ("x", 1, 16))),
            ("66 0f 56 c1", "orpd", (("x", 0, 16), ("x", 1, 16))),
            ("66 0f ef c1", "pxor", (("x", 0, 16), ("x", 1, 16))),
            ("66 0f db c1", "pand", (("x", 0, 16), ("x", 1, 16))),
            ("66 0f df c1", "pandn", (("x", 0, 16), ("x", 1, 16))),
            ("66 0f eb c1", "por", (("x", 0, 16), ("x", 1, 16))),
        )
        for hex_bytes, mnem, ops in cases:
            ins = self.one(asm(hex_bytes))
            self.assertEqual((ins.mnem, ins.ops), (mnem, ops), hex_bytes)
            if mnem in x.FCMP_KIND:
                self.assertEqual(ins.flags_w, "NZCVPA")
            else:
                self.assertEqual(ins.flags_w, "")

    def test_prefix_precedence_and_mmx_stays_unhandled(self):
        # Mandatory F3/F2 takes precedence over 66 and does not shrink Ey/Gy to 16-bit.
        ins = self.one(asm("66 f3 0f 58 c1"))
        self.assertEqual(ins.mnem, "addss")
        ins = self.one(asm("66 f2 0f 2a c0"))
        self.assertEqual((ins.mnem, ins.ops), ("cvtsi2sd", (("x", 0, 8), ("r", 0, 4))))

        # MMX forms without 66/F2/F3 must stay generic unhandled "simd" (never XMM).
        for mmx_hex in ("0f 6e c0", "0f 7e c0", "0f 6f c1", "0f 7f c1",
                        "0f ef c1", "0f db c1", "0f df c1", "0f eb c1",
                        "0f 2a c0", "0f 2c c0", "0f 2d c0"):
            ins = self.one(asm(mmx_hex))
            self.assertEqual(ins.mnem, "simd", mmx_hex)
            tr = x.translate_function(asm(mmx_hex, "c3"), GUEST_VA)
            self.assertEqual(len(tr.stats["unhandled"]), 1, mmx_hex)


class Arm64FPEncodingTests(unittest.TestCase):
    def test_fp_simd_encodings_match_arm64_isa(self):
        a = x.Asm()
        a.fldr(1, 16, 0, size=4)
        a.fldr(1, 16, 12, size=4)
        a.fldr(2, 16, 0, size=8)
        a.fldr(2, 16, 16, size=8)
        a.fldr(3, 16, 0, size=16)
        a.fldr(3, 16, 32, size=16)
        a.fstr(1, 16, 0, size=4)
        a.fstr(2, 16, 0, size=8)
        a.fstr(3, 16, 0, size=16)
        a.fadd(1, 2, 3, dp=0)
        a.fadd(1, 2, 3, dp=1)
        a.fsub(1, 2, 3, dp=0)
        a.fsub(1, 2, 3, dp=1)
        a.fmul(1, 2, 3, dp=0)
        a.fmul(1, 2, 3, dp=1)
        a.fdiv(1, 2, 3, dp=0)
        a.fdiv(1, 2, 3, dp=1)
        a.fsqrt(1, 2, dp=0)
        a.fsqrt(1, 2, dp=1)
        a.fcmp(2, 3, dp=0)
        a.fcmp(2, 3, dp=1)
        a.fcsel(1, 2, 3, "mi", dp=0)
        a.fcsel(1, 2, 3, "gt", dp=1)
        a.fmov_r(1, 2, dp=0)
        a.fmov_r(1, 2, dp=1)
        a.fmov_to_fp(1, 2, dp=0)
        a.fmov_to_fp(1, 2, dp=1)
        a.fmov_to_gpr(1, 2, dp=0)
        a.fmov_to_gpr(1, 2, dp=1)
        a.fcvt_s2d(1, 2)
        a.fcvt_d2s(1, 2)
        a.scvtf(1, 2, sf=0, dp=0)
        a.scvtf(1, 2, sf=1, dp=0)
        a.scvtf(1, 2, sf=0, dp=1)
        a.scvtf(1, 2, sf=1, dp=1)
        a.fcvtzs(1, 2, sf=0, dp=0)
        a.fcvtzs(1, 2, sf=1, dp=0)
        a.fcvtzs(1, 2, sf=0, dp=1)
        a.fcvtzs(1, 2, sf=1, dp=1)
        a.fcvtns(1, 2, sf=0, dp=0)
        a.fcvtns(1, 2, sf=1, dp=0)
        a.fcvtns(1, 2, sf=0, dp=1)
        a.fcvtns(1, 2, sf=1, dp=1)
        a.vins0(1, 2, dp=0)
        a.vins0(1, 2, dp=1)
        a.vand16b(1, 2, 3)
        a.vbic16b(1, 2, 3)
        a.vorr16b(1, 2, 3)
        a.veor16b(1, 2, 3)
        a.mrs_cpacr_el1(0)
        a.msr_cpacr_el1(0)
        a.isb()
        expected = [
            0xBD400201, 0xBD400E01, 0xFD400202, 0xFD400A02, 0x3DC00203, 0x3DC00A03,
            0xBD000201, 0xFD000202, 0x3D800203,
            0x1E232841, 0x1E632841, 0x1E233841, 0x1E633841,
            0x1E230841, 0x1E630841, 0x1E231841, 0x1E631841,
            0x1E21C041, 0x1E61C041, 0x1E232040, 0x1E632040,
            0x1E234C41, 0x1E63CC41,
            0x1E204041, 0x1E604041, 0x1E270041, 0x9E670041, 0x1E260041, 0x9E660041,
            0x1E22C041, 0x1E624041,
            0x1E220041, 0x9E220041, 0x1E620041, 0x9E620041,
            0x1E380041, 0x9E380041, 0x1E780041, 0x9E780041,
            0x1E200041, 0x9E200041, 0x1E600041, 0x9E600041,
            0x6E040441, 0x6E080441,
            0x4E231C41, 0x4E631C41, 0x4EA31C41, 0x6E231C41,
            0xD5381040, 0xD5181040, 0xD5033FDF,
        ]
        self.assertEqual(a.words, expected)


class SSETranslateTests(unittest.TestCase):
    def test_all_supported_sse_translate_cleanly(self):
        code = asm(
            "0f 57 c0",             # xorps xmm0, xmm0
            "66 0f 6e c8",          # movd xmm1, eax
            "f3 0f 10 c1",          # movss xmm0, xmm1
            "f3 0f 58 c1",          # addss xmm0, xmm1
            "f3 0f 5c c1",          # subss xmm0, xmm1
            "f3 0f 59 c1",          # mulss xmm0, xmm1
            "f3 0f 5e c1",          # divss xmm0, xmm1
            "f3 0f 51 c0",          # sqrtss xmm0, xmm0
            "f3 0f 5d c1",          # minss xmm0, xmm1
            "f3 0f 5f c1",          # maxss xmm0, xmm1
            "f3 0f 5a c0",          # cvtss2sd xmm0, xmm0
            "f2 0f 58 c1",          # addsd xmm0, xmm1
            "f2 0f 5c c1",          # subsd xmm0, xmm1
            "f2 0f 59 c1",          # mulsd xmm0, xmm1
            "f2 0f 5e c1",          # divsd xmm0, xmm1
            "f2 0f 51 c0",          # sqrtsd xmm0, xmm0
            "f2 0f 5d c1",          # minsd xmm0, xmm1
            "f2 0f 5f c1",          # maxsd xmm0, xmm1
            "f2 0f 5a c0",          # cvtsd2ss xmm0, xmm0
            "f3 0f 2a c0",          # cvtsi2ss xmm0, eax
            "f2 48 0f 2a c0",       # cvtsi2sd xmm0, rax
            "f3 0f 2c c0",          # cvttss2si eax, xmm0
            "f2 48 0f 2d c0",       # cvtsd2si rax, xmm0
            "0f 2e c1",             # ucomiss xmm0, xmm1
            "66 0f 2e c1",          # ucomisd xmm0, xmm1
            "0f 2f c1",             # comiss xmm0, xmm1
            "66 0f 2f c1",          # comisd xmm0, xmm1
            "0f 28 d0",             # movaps xmm2, xmm0
            "66 0f 6f da",          # movdqa xmm3, xmm2
            "f3 0f 7e d9",          # movq xmm3, xmm1
            "66 0f 54 c1",          # andpd xmm0, xmm1
            "0f 55 c1",             # andnps xmm0, xmm1
            "66 0f 56 c1",          # orpd xmm0, xmm1
            "66 0f ef c1",          # pxor xmm0, xmm1
            "66 0f db c1",          # pand xmm0, xmm1
            "66 0f df c1",          # pandn xmm0, xmm1
            "66 0f eb c1",          # por xmm0, xmm1
            "c3",
        )
        tr = x.translate_function(code, GUEST_VA)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(tr.stats["insns"], 38)

    def test_ucomis_flag_liveness_and_intervening_inc(self):
        # ucomiss; inc eax; jb target -> inc clobbers NZV so ucomiss must materialize CTX_EFLAGS
        code = asm(
            "0f 2e c1",             # +0: ucomiss xmm0, xmm1
            "ff c0",                # +3: inc eax
            "72 01",                # +5: jb +1
            "90",                   # +7: nop
            "c3",                   # +8: ret
        )
        tr = x.translate_function(code, GUEST_VA)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertIn(GUEST_VA, tr.mat)
        self.assertIsNone(tr.arm_path[GUEST_VA + 5])

        # Cross-block ucomiss -> ja / jbe must materialize CTX_EFLAGS without invalid 0x41 bitmask imm
        code2 = asm(
            "0f 2e c1",             # +0: ucomiss xmm0, xmm1
            "eb 00",                # +3: jmp +0 (splits block)
            "77 02",                # +5: ja +2
            "76 00",                # +7: jbe +0
            "c3",                   # +9: ret
        )
        tr2 = x.translate_function(code2, GUEST_VA)
        self.assertEqual(tr2.stats["unhandled"], [])
        self.assertIn(GUEST_VA, tr2.mat)

    def test_kernel_stub_enables_el1_fp_simd(self):
        img, tr = x.build_test_kernel(asm("0f 57 c0", "c3"), GUEST_VA)
        self.assertEqual(tr.stats["unhandled"], [])
        w0, w1, w2, w3 = struct.unpack_from("<4I", img, 0)
        self.assertEqual([w0, w1, w2, w3], [0xD5381040, 0xB26C0400, 0xD5181040, 0xD5033FDF])


@unittest.skipUnless(CAN_RUN, "neither Apple Silicon Mach-O runner nor qemu-system-aarch64 available")
class SSEExecTests(unittest.TestCase):
    def test_scalar_float32_arith_and_sqrt(self):
        # ((1.5 + 2.25) * 4.0 - 6.0) / 3.0 = 3.0; sqrt(9.0) = 3.0 -> 3.0 + 3.0 = 6.0f
        s = EXEC_SCRATCH
        code = asm(
            "b8 " + f32_hex(1.5), "66 0f 6e c0",       # xmm0 = 1.5f
            "b8 " + f32_hex(2.25), "66 0f 6e c8",      # xmm1 = 2.25f
            "f3 0f 58 c1",                              # addss xmm0, xmm1 -> 3.75f
            "48 bf " + s.to_bytes(8, "little").hex(),
            "c7 07 " + f32_hex(4.0),
            "f3 0f 59 07",                              # mulss xmm0, [rdi] -> 15.0f
            "b8 " + f32_hex(6.0), "66 0f 6e c8",
            "f3 0f 5c c1",                              # subss xmm0, xmm1 -> 9.0f
            "f3 0f 51 d0",                              # sqrtss xmm2, xmm0 -> 3.0f
            "b8 " + f32_hex(3.0), "66 0f 6e c8",
            "f3 0f 5e c1",                              # divss xmm0, xmm1 -> 3.0f
            "f3 0f 58 c2",                              # addss xmm0, xmm2 -> 6.0f
            "66 0f 7e c7",                              # movd edi, xmm0
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, u32_bits(6.0))

    def test_scalar_float64_arith_and_sqrt(self):
        # ((10.5 + 5.5) * 2.0 - 7.0) = 25.0; sqrt(25.0) / 2.0 = 2.5 -> high32 ^ low32 in edi
        code = asm(
            "48 b8 " + f64_hex(10.5), "66 48 0f 6e c0", # xmm0 = 10.5
            "48 b8 " + f64_hex(5.5), "66 48 0f 6e c8",  # xmm1 = 5.5
            "f2 0f 58 c1",                              # addsd xmm0, xmm1 -> 16.0
            "48 b8 " + f64_hex(2.0), "66 48 0f 6e c8",
            "f2 0f 59 c1",                              # mulsd xmm0, xmm1 -> 32.0
            "48 b8 " + f64_hex(7.0), "66 48 0f 6e c8",
            "f2 0f 5c c1",                              # subsd xmm0, xmm1 -> 25.0
            "f2 0f 51 c0",                              # sqrtsd xmm0, xmm0 -> 5.0
            "48 b8 " + f64_hex(2.0), "66 48 0f 6e c8",
            "f2 0f 5e c1",                              # divsd xmm0, xmm1 -> 2.5
            "66 48 0f 7e c0",                           # movq rax, xmm0
            "48 89 c2",                                 # mov rdx, rax
            "48 c1 ea 20",                              # shr rdx, 32
            "31 d0",                                    # xor eax, edx
            "89 c7",                                    # mov edi, eax
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        bits = u64_bits(2.5)
        self.assertEqual(r, ((bits >> 32) ^ (bits & 0xFFFFFFFF)) & 0xFFFFFFFF)

    def test_upper_lane_preservation_and_zero_extension(self):
        # Seed xmm0 with [1.0f, 0x11223344, 0x55667788, 0x99AABBCC]; scalar reg ops must preserve
        # lanes 1..3, while memory movss and reg/mem movq must zero upper lanes.
        s = EXEC_SCRATCH
        code = asm(
            "48 bf " + s.to_bytes(8, "little").hex(),
            "c7 07 " + f32_hex(1.0),
            "c7 47 04 44 33 22 11",
            "c7 47 08 88 77 66 55",
            "c7 47 0c cc bb aa 99",
            "0f 10 07",                                 # movups xmm0, [rdi]
            "b8 " + f32_hex(2.0), "66 0f 6e c8",        # xmm1 = 2.0f
            "f3 0f 58 c1",                              # addss xmm0, xmm1 -> 3.0f
            "f3 0f 59 c1",                              # mulss xmm0, xmm1 -> 6.0f
            "0f 11 47 10",                              # movups [rdi+16], xmm0
            # Preserve upper 64 bits across movsd reg,reg
            "0f 10 0f",                                 # movups xmm1, [rdi]
            "f2 0f 10 c1",                              # movsd xmm0, xmm1 (preserves xmm0[127:64])
            "0f 11 47 40",                              # movups [rdi+64], xmm0
            # Memory movss must zero lanes 1..3
            "f3 0f 10 07",
            "0f 11 47 20",
            # Register movq must zero lanes 2..3 while keeping lane 1
            "0f 10 07",
            "f3 0f 7e d0",                              # movq xmm2, xmm0
            "0f 11 57 30",
            # Scalar memory stores (movss / movsd / movq) round-trip cleanly
            "f3 0f 11 47 50",                           # movss [rdi+80], xmm0
            "f2 0f 11 47 58",                           # movsd [rdi+88], xmm0
            "66 0f d6 47 60",                           # movq [rdi+96], xmm0
            # Compute checksum: XOR expected values with stored values (0 iff all match!)
            "31 c0",                                    # xor eax, eax
            "8b 57 10", "81 f2 " + f32_hex(6.0), "09 d0",
            "8b 57 14", "81 f2 44 33 22 11", "09 d0",
            "8b 57 18", "81 f2 88 77 66 55", "09 d0",
            "8b 57 1c", "81 f2 cc bb aa 99", "09 d0",
            "8b 57 48", "81 f2 88 77 66 55", "09 d0",
            "8b 57 4c", "81 f2 cc bb aa 99", "09 d0",
            "0b 47 24", "0b 47 28", "0b 47 2c",         # movss mem zeroed lanes 1..3
            "8b 57 34", "81 f2 44 33 22 11", "09 d0",   # movq kept lane 1
            "0b 47 38", "0b 47 3c",                     # movq zeroed lanes 2..3
            "8b 57 50", "81 f2 " + f32_hex(1.0), "09 d0",
            "8b 57 5c", "81 f2 44 33 22 11", "09 d0",
            "8b 57 64", "81 f2 44 33 22 11", "09 d0",
            "89 c7",
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0)

    def test_min_max_ieee_nan_signed_zero_and_nzcv_preservation(self):
        # x86 minss(a,b) = (a < b) ? a : b; maxss(a,b) = (a > b) ? a : b.
        # In particular: NaN in either operand returns the SECOND operand (b),
        # and (+0.0, -0.0) returns -0.0 while (-0.0, +0.0) returns +0.0!
        # Also verify integer NZCV flags survive across minss/maxss.
        qnan = "00 00 c0 7f"
        pos0 = "00 00 00 00"
        neg0 = "00 00 00 80"
        one = f32_hex(1.0)
        two = f32_hex(2.0)

        def run_mm(op_hex, a_hex, b_hex):
            code = asm(
                "b8 " + a_hex, "66 0f 6e c0",           # xmm0 = a
                "b8 " + b_hex, "66 0f 6e c8",           # xmm1 = b
                op_hex,                                  # minss/maxss xmm0, xmm1
                "66 0f 7e c7",                          # movd edi, xmm0
                "c3",
            )
            r, tr = run(code)
            self.assertEqual(tr.stats["unhandled"], [])
            return r

        # minss
        self.assertEqual(run_mm("f3 0f 5d c1", one, two), u32_bits(1.0))
        self.assertEqual(run_mm("f3 0f 5d c1", two, one), u32_bits(1.0))
        self.assertEqual(run_mm("f3 0f 5d c1", qnan, two), u32_bits(2.0))       # NaN first -> returns src!
        self.assertEqual(run_mm("f3 0f 5d c1", two, qnan), 0x7FC00000)          # NaN second -> returns src!
        self.assertEqual(run_mm("f3 0f 5d c1", pos0, neg0), 0x80000000)         # +0 vs -0 -> src (-0.0)!
        self.assertEqual(run_mm("f3 0f 5d c1", neg0, pos0), 0x00000000)         # -0 vs +0 -> src (+0.0)!
        # maxss
        self.assertEqual(run_mm("f3 0f 5f c1", one, two), u32_bits(2.0))
        self.assertEqual(run_mm("f3 0f 5f c1", qnan, one), u32_bits(1.0))       # NaN first -> returns src!
        self.assertEqual(run_mm("f3 0f 5f c1", one, qnan), 0x7FC00000)          # NaN second -> returns src!
        self.assertEqual(run_mm("f3 0f 5f c1", neg0, pos0), 0x00000000)         # -0 vs +0 -> src (+0.0)!
        self.assertEqual(run_mm("f3 0f 5f c1", pos0, neg0), 0x80000000)         # +0 vs -0 -> src (-0.0)!

        # Live integer NZCV flags must NOT be clobbered by minss/maxss
        code = asm(
            "b8 05 00 00 00",                           # mov eax, 5
            "83 f8 09",                                 # cmp eax, 9 (5 < 9 -> CF=1 / lt)
            "b9 " + two, "66 0f 6e c1",                 # xmm0 = 2.0f
            "ba " + one, "66 0f 6e ca",                 # xmm1 = 1.0f
            "f3 0f 5d c1",                              # minss xmm0, xmm1 (must preserve NZCV!)
            "0f 9c c0",                                 # setl al (must still be 1!)
            "0f b6 f8",                                 # movzx edi, al
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 1)

    def test_ucomiss_and_ucomisd_all_relations_and_flags(self):
        # Pack setcc results (bits 0..7: a, ae, b, be, e, ne, p, np) into edi for each relation.
        def probe_flags(a_hex, b_hex, is_double=False, cross_block=False):
            if is_double:
                setup = ("48 b8 " + a_hex, "66 48 0f 6e c0",
                         "48 b8 " + b_hex, "66 48 0f 6e c8")
            else:
                setup = ("b8 " + a_hex, "66 0f 6e c0",
                         "b8 " + b_hex, "66 0f 6e c8")
            ucomi = "66 0f 2e c1" if is_double else "0f 2e c1"
            parts = list(setup) + ["31 ff"]
            setccs = ("0f 97 c0", "0f 93 c0", "0f 92 c0", "0f 96 c0",
                      "0f 94 c0", "0f 95 c0", "0f 9a c0", "0f 9b c0")
            for bit, sc in enumerate(setccs):
                parts.append(ucomi)
                if cross_block:
                    parts.append("eb 00")               # force cross-block CTX_EFLAGS
                parts.extend([sc, "0f b6 c0"])
                if bit:
                    parts.append(f"c1 e0 {bit:02x}")
                parts.append("09 c7")
            parts.append("c3")
            r, tr = run(asm(*parts))
            self.assertEqual(tr.stats["unhandled"], [])
            return r

        # Bit mask: 0=a, 1=ae, 2=b, 3=be, 4=e, 5=ne, 6=p, 7=np
        exp_gt = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 7)              # 0b10100011 = 0xA3
        exp_lt = (1 << 2) | (1 << 3) | (1 << 5) | (1 << 7)              # 0b10101100 = 0xAC
        exp_eq = (1 << 1) | (1 << 3) | (1 << 4) | (1 << 7)              # 0b10011010 = 0x9A
        exp_un = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6)              # 0b01011100 = 0x5C

        qnan32 = "00 00 c0 7f"
        qnan64 = "00 00 00 00 00 00 f8 7f"
        for cross in (False, True):
            self.assertEqual(probe_flags(f32_hex(5.0), f32_hex(2.0), False, cross), exp_gt)
            self.assertEqual(probe_flags(f32_hex(2.0), f32_hex(5.0), False, cross), exp_lt)
            self.assertEqual(probe_flags(f32_hex(3.0), f32_hex(3.0), False, cross), exp_eq)
            self.assertEqual(probe_flags(f32_hex(0.0), f32_hex(-0.0), False, cross), exp_eq)
            self.assertEqual(probe_flags(qnan32, f32_hex(1.0), False, cross), exp_un)
            self.assertEqual(probe_flags(f64_hex(5.0), f64_hex(2.0), True, cross), exp_gt)
            self.assertEqual(probe_flags(f64_hex(2.0), f64_hex(5.0), True, cross), exp_lt)
            self.assertEqual(probe_flags(f64_hex(3.0), f64_hex(3.0), True, cross), exp_eq)
            self.assertEqual(probe_flags(qnan64, f64_hex(1.0), True, cross), exp_un)

    def test_ucomiss_pushf_adc_cmov_and_intervening_inc(self):
        # pushf after ucomiss(NaN, 1.0) -> ZF(0x40) | PF(0x04) | CF(0x01) | bit1(0x02) = 0x47
        code = asm(
            "b8 00 00 c0 7f", "66 0f 6e c0",            # xmm0 = qNaN
            "b8 " + f32_hex(1.0), "66 0f 6e c8",        # xmm1 = 1.0f
            "0f 2e c1",                                 # ucomiss xmm0, xmm1
            "9c",                                       # pushf
            "5f",                                       # pop rdi
            "81 e7 d5 08 00 00",                        # mask OF|SF|ZF|AF|PF|CF (0x8D5)
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x45)

        # ucomiss(1.0, 2.0) -> CF=1 -> adc edi, 10 == 11; then cmovnp selects 99
        code = asm(
            "31 ff",                                    # xor edi, edi
            "b8 " + f32_hex(1.0), "66 0f 6e c0",
            "b8 " + f32_hex(2.0), "66 0f 6e c8",
            "0f 2e c1",                                 # ucomiss (1.0 < 2.0 -> CF=1, PF=0)
            "83 d7 0a",                                 # adc edi, 10 -> 11
            "0f 2e c1",                                 # ucomiss again
            "bb 63 00 00 00",                           # mov ebx, 99
            "0f 4b fb",                                 # cmovnp edi, ebx -> 99
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 99)

        # ucomiss(1.0, 2.0) -> CF=1; inc eax (does not touch CF); jb taken -> edi=7
        code = asm(
            "b8 " + f32_hex(1.0), "66 0f 6e c0",
            "b8 " + f32_hex(2.0), "66 0f 6e c8",
            "31 c0",                                    # xor eax, eax
            "0f 2e c1",                                 # ucomiss -> CF=1
            "ff c0",                                    # inc eax (clobbers NZV, keeps CF)
            "72 06",                                    # jb +6 -> taken
            "bf 01 00 00 00", "c3",
            "bf 07 00 00 00", "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 7)

        # ucomiss(1.0, 2.0) -> CF=1; cmc toggles CF to 0 -> jae taken -> edi=13
        code = asm(
            "b8 " + f32_hex(1.0), "66 0f 6e c0",
            "b8 " + f32_hex(2.0), "66 0f 6e c8",
            "0f 2e c1",                                 # ucomiss -> CF=1
            "f5",                                       # cmc -> CF=0
            "73 06",                                    # jae +6 -> taken
            "bf 01 00 00 00", "c3",
            "bf 0d 00 00 00", "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 13)

        # lahf after ucomiss(NaN, 1.0) -> ah = ZF|PF|CF = 0x45
        code = asm(
            "b8 00 00 c0 7f", "66 0f 6e c0",
            "b8 " + f32_hex(1.0), "66 0f 6e c8",
            "0f 2e c1",                                 # ucomiss
            "9f",                                       # lahf
            "0f b6 fc",                                 # movzx edi, ah
            "83 e7 d5",                                 # mask SF|ZF|AF|PF|CF
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x45)

        # Live integer NZCV must survive across cvttss2si
        code = asm(
            "b8 02 00 00 00",                           # mov eax, 2
            "83 f8 09",                                 # cmp eax, 9 (2 < 9 -> lt)
            "b9 " + f32_hex(7.9), "66 0f 6e c1",        # xmm0 = 7.9f
            "f3 0f 2c d0",                              # cvttss2si edx, xmm0 -> edx=7 (preserves NZCV!)
            "0f 9c c0",                                 # setl al -> 1
            "0f b6 f8",                                 # movzx edi, al
            "01 d7",                                    # add edi, edx -> 8
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 8)

    def test_conversions_rounding_and_x86_indefinite(self):
        # cvtsi2ss / cvtss2sd / cvtsd2ss / cvtsi2sd
        code = asm(
            "b8 d6 ff ff ff",                           # mov eax, -42
            "f3 0f 2a c0",                              # cvtsi2ss xmm0, eax -> -42.0f
            "f3 0f 5a c8",                              # cvtss2sd xmm1, xmm0 -> -42.0
            "b8 0a 00 00 00",                           # mov eax, 10
            "f2 0f 2a d0",                              # cvtsi2sd xmm2, eax -> 10.0
            "f2 0f 58 ca",                              # addsd xmm1, xmm2 -> -32.0
            "f2 0f 5a c1",                              # cvtsd2ss xmm0, xmm1 -> -32.0f
            "f3 0f 2c f8",                              # cvttss2si edi, xmm0 -> -32
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, (-32) & 0xFFFFFFFF)

        # Truncate vs round-to-nearest-even (ties to even: 2.5->2, 3.5->4)
        def run_f2i(op_hex, val_hex, is_double=False, is_64=False):
            load = ("48 b8 " + val_hex, "66 48 0f 6e c0") if is_double else ("b8 " + val_hex, "66 0f 6e c0")
            if is_64:
                tail = (op_hex, "48 89 c2", "48 c1 ea 20", "31 d0", "89 c7", "c3")
            else:
                tail = (op_hex, "89 c7", "c3")
            r, tr = run(asm(*load, *tail))
            self.assertEqual(tr.stats["unhandled"], [])
            return r

        # cvttss2si eax, xmm0 (f3 0f 2c c0) vs cvtss2si eax, xmm0 (f3 0f 2d c0)
        self.assertEqual(run_f2i("f3 0f 2c c0", f32_hex(3.9)), 3)
        self.assertEqual(run_f2i("f3 0f 2c c0", f32_hex(-3.9)), (-3) & 0xFFFFFFFF)
        self.assertEqual(run_f2i("f3 0f 2d c0", f32_hex(2.5)), 2)               # tie to even
        self.assertEqual(run_f2i("f3 0f 2d c0", f32_hex(3.5)), 4)               # tie to even
        self.assertEqual(run_f2i("f3 0f 2c c0", "00 00 c0 7f"), 0x80000000)     # NaN -> INT32_MIN
        self.assertEqual(run_f2i("f3 0f 2c c0", "00 00 80 7f"), 0x80000000)     # +Inf -> INT32_MIN
        self.assertEqual(run_f2i("f3 0f 2c c0", "00 00 80 ff"), 0x80000000)     # -Inf -> INT32_MIN
        self.assertEqual(run_f2i("f3 0f 2c c0", f32_hex(float(2**31))), 0x80000000)

        # cvttsd2si vs cvtsd2si at exact 2147483647.5 boundary!
        self.assertEqual(run_f2i("f2 0f 2c c0", f64_hex(2147483647.5), True), 2147483647)
        self.assertEqual(run_f2i("f2 0f 2d c0", f64_hex(2147483647.5), True), 0x80000000)  # rounds to 2^31 -> overflow!
        self.assertEqual(run_f2i("f2 0f 2c c0", f64_hex(2147483648.0), True), 0x80000000)
        self.assertEqual(run_f2i("f2 0f 2c c0", "00 00 00 00 00 00 f8 7f", True), 0x80000000)

        # 64-bit cvttsd2si rax, xmm0 (f2 48 0f 2c c0): NaN & +2^63 -> INT64_MIN (high^low = 0x80000000)
        self.assertEqual(run_f2i("f2 48 0f 2c c0", "00 00 00 00 00 00 f8 7f", True, True), 0x80000000)
        self.assertEqual(run_f2i("f2 48 0f 2c c0", f64_hex(float(2**63)), True, True), 0x80000000)
        self.assertEqual(run_f2i("f2 48 0f 2c c0", f64_hex(5000000000.75), True, True),
                         ((5000000000 >> 32) ^ (5000000000 & 0xFFFFFFFF)))

    def test_128bit_bitwise_negate_abs_and_andnps_order(self):
        # Verify xorps (negate float), andps (fabsf), andnps (~dst & src), orps, and xmm8..xmm15.
        code = asm(
            "b8 " + f32_hex(3.5), "66 44 0f 6e c0",     # movd xmm8, eax (3.5f)
            "b8 00 00 00 80", "66 44 0f 6e c8",         # movd xmm9, eax (sign mask 0x80000000)
            "45 0f 57 c1",                              # xorps xmm8, xmm9 -> -3.5f
            "45 0f 28 d0",                              # movaps xmm10, xmm8 (-3.5f)
            "b8 ff ff ff 7f", "66 44 0f 6e d8",         # movd xmm11, eax (abs mask 0x7fffffff)
            "45 0f 54 d3",                              # andps xmm10, xmm11 -> +3.5f
            # Test andnps operand order: (~dst) & src. dst=0x00FF00FF, src=0x0F0F0F0F -> 0x0F000F00
            "b8 ff 00 ff 00", "66 0f 6e c0",            # xmm0 = 0x00FF00FF
            "b8 0f 0f 0f 0f", "66 0f 6e c8",            # xmm1 = 0x0F0F0F0F
            "0f 55 c1",                                 # andnps xmm0, xmm1 -> 0x0F000F00
            "66 0f 7e c2",                              # movd edx, xmm0
            "66 44 0f 7e d0",                           # movd eax, xmm10 (+3.5f)
            "35 " + f32_hex(3.5),                       # xor eax, bits(3.5f) -> 0
            "09 c2",                                    # or edx, eax -> 0x0F000F00
            "89 d7",                                    # mov edi, edx
            "c3",
        )
        r, tr = run(code)
        self.assertEqual(tr.stats["unhandled"], [])
        self.assertEqual(r, 0x0F000F00)


if __name__ == "__main__":
    unittest.main()
