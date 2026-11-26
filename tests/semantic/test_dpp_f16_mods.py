"""Semantic tests for DPP src0 neg/abs modifiers on f16 ops and sNaN-preserving f32 ops.

Verifies that Gen.dpp_insn forwards `neg`/`abs` on the post-select SSA operand instead of
eagerly applying 32-bit float `Gen.modify()` (bit 31 FAbs/OpFNegate):
  1. f16 DPP ops (`v_add_f16_dpp`, `v_mul_f16_dpp`, `v_min_f16_dpp`, `v_cvt_f32_f16_dpp`,
     `v_cmp_gt_f16_dpp`) apply `neg`/`abs` to the f16 sign bit (bit 15) after invalid-lane
     zero selection, preserving the destination high half and yielding `-0.0h` for `-v1`
     on invalid `bound_ctrl:1` lanes.
  2. Sign-bit-move / sNaN-preserving f32 DPP ops (`v_cndmask_b32_dpp`, `v_cmp_class_f32_dpp`,
     `v_min_f32_dpp`) use raw integer bit-31 sign moves (`src_u_sign_mods` / `src_f_sign_mods` /
     `cmp_class`) rather than `FAbs`/`OpFNegate`.
  3. Wave64 emulation (`--wave64-emulate`) produces the same f16 DPP modifier results.

    python -I tests/semantic/test_dpp_f16_mods.py
"""
import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

SEMANTIC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SEMANTIC_DIR))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for _p in (TOOLS_DIR, SEMANTIC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def translate_cs(name: str, body_lines: list[str], *, wave64: bool = False, emulate: bool = False) -> str:
    threads = 64 if wave64 else 32
    with tempfile.TemporaryDirectory() as td:
        os.mkdir(os.path.join(td, "disasm"))
        os.mkdir(os.path.join(td, "fssm"))
        src = os.path.join(td, "disasm", f"cs_{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; cs_{name}\n")
            for ln in body_lines:
                f.write(f"\t{ln}\n")
            f.write("\ts_endpgm\n")
        with open(os.path.join(td, f"cs_{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        fssm = bytearray(0x148)
        fssm[0:4] = b"FSSM"
        struct.pack_into("<I", fssm, 0xD4, 0x00 if wave64 else 0x20)
        struct.pack_into("<4I", fssm, 0x138, 0x8090, threads, 1, 1)
        with open(os.path.join(td, "fssm", f"cs_{name}.fssm"), "wb") as f:
            f.write(bytes(fssm))
        old_emu = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emu
        return spv


def run_cs(spv: str, *, wave64: bool = False, emulate: bool = False) -> dict[int, int]:
    threads = 64 if wave64 else 32
    sub = 32 if emulate else threads
    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv, threads, sub,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sub},
        push_constants=pc, memory={},
    )


class DppF16AndSignModsTests(unittest.TestCase):
    def test_v_add_f16_dpp_neg_and_abs_wave32_and_emulate(self):
        # Put f16(2.0) = 0x4000 in lane 0's low half and f16(-3.0) = 0xC200 in lane 1's low half
        # (with 0xABCD in the high half to verify high-half preservation on dst).
        # With quad_perm:[1,0,3,2]:
        #   lane 0 reads lane 1 (-3.0h): -(-3.0h) + 1.0h = +4.0h (0x4400); |-3.0h| * 2.0h = +6.0h (0x4600)
        #   lane 1 reads lane 0 (+2.0h): -(+2.0h) + 1.0h = -1.0h (0xBC00); |+2.0h| * 2.0h = +4.0h (0x4400)
        lines = [
            "v_and_b32_e32 v4, 1, v0",
            "v_cmp_eq_u32_e32 vcc_lo, 0, v4",
            "v_cndmask_b32_e32 v1, 0x0000c200, 0x00004000, vcc_lo",
            "v_mov_b32_e32 v2, 0xabcd0000",
            "v_mov_b32_e32 v3, 0x00003c00",  # +1.0h
            "v_add_f16_dpp v2, -v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_mov_b32_e32 v5, 0x12340000",
            "v_mov_b32_e32 v7, 0x00004000",  # +2.0h
            "v_mul_f16_dpp v5, |v1|, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v2, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
        ]
        # Wave32
        spv = translate_cs("w32_dpp_f16_add_mul", lines, wave64=False)
        mem = run_cs(spv, wave64=False)
        for lane in range(32):
            exp_add = (0xABCD << 16) | (f16_bits(4.0) if (lane & 1) == 0 else f16_bits(-1.0))
            exp_mul = (0x1234 << 16) | (f16_bits(6.0) if (lane & 1) == 0 else f16_bits(4.0))
            self.assertEqual(mem[4 * lane], exp_add, f"add lane {lane}")
            self.assertEqual(mem[256 + 4 * lane], exp_mul, f"mul lane {lane}")

        # Wave64 emulated (replace vcc_lo with vcc)
        lines_w64 = [ln.replace("vcc_lo", "vcc") for ln in lines]
        spv_emu = translate_cs("w64_dpp_f16_add_mul", lines_w64, wave64=True, emulate=True)
        mem_emu = run_cs(spv_emu, wave64=True, emulate=True)
        for lane in range(64):
            exp_add = (0xABCD << 16) | (f16_bits(4.0) if (lane & 1) == 0 else f16_bits(-1.0))
            exp_mul = (0x1234 << 16) | (f16_bits(6.0) if (lane & 1) == 0 else f16_bits(4.0))
            self.assertEqual(mem_emu[4 * lane], exp_add, f"emu add lane {lane}")
            self.assertEqual(mem_emu[256 + 4 * lane], exp_mul, f"emu mul lane {lane}")

    def test_v_cvt_f32_f16_dpp_neg_and_bound_ctrl_zero(self):
        # row_shr:1 with bound_ctrl:1: lane 0 and lane 16 read invalid source -> 0x0000 (+0.0h),
        # then `-` negates it to -0.0h -> widened to -0.0f (0x80000000).
        # Other lanes read previous lane's +2.0h (0x4000) -> negated to -2.0f.
        spv = translate_cs("w32_dpp_cvt_f32_f16", [
            "v_mov_b32_e32 v1, 0x00004000",  # +2.0h
            "v_cvt_f32_f16_dpp v2, -v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v2, v6, s[0:3], 0 offen",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        for lane in range(32):
            expected = 0x80000000 if (lane % 16) == 0 else f32_bits(-2.0)
            self.assertEqual(mem[4 * lane], expected, f"lane {lane}")

    def test_v_min_f16_and_v_cmp_gt_f16_dpp_neg(self):
        # All lanes have v1 = +3.0h (0x4200), v3 = +1.0h (0x3c00).
        # With -v1:
        #   v_min_f16_dpp(-3.0h, +1.0h) = -3.0h (0xC200) (on origin/main: min(+3.0h, +1.0h) = +1.0h)
        #   v_cmp_gt_f16_dpp(-3.0h, +1.0h) = false (0)   (on origin/main: +3.0h > +1.0h = true)
        spv = translate_cs("w32_dpp_f16_min_cmp", [
            "v_mov_b32_e32 v1, 0x00004200",  # +3.0h
            "v_mov_b32_e32 v3, 0x00003c00",  # +1.0h
            "v_mov_b32_e32 v2, 0",
            "v_min_f16_dpp v2, -v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_cmp_gt_f16_dpp vcc_lo, -v1, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_cndmask_b32_e32 v4, 0, 1, vcc_lo",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v2, v6, s[0:3], 0 offen",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:128",
        ], wave64=False)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [f16_bits(-3.0)] * 32)
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [0] * 32)

    def test_v_cndmask_b32_and_cmp_class_f32_dpp_preserve_sign_bit_moves(self):
        # v1 = 0x7f800001 (positive sNaN).
        # v_cndmask_b32_dpp with -v1 (and vcc = 0 so src0 is chosen) must flip bit 31 via
        # OpBitwiseXor (no FAbs/OpFNegate that could quiet the sNaN) -> 0xff800001.
        # v_cmp_class_f32_dpp with -v1 and class mask 0x001 (sNaN) must also keep raw bit ops
        # (no FAbs/OpFNegate) and report True.
        spv = translate_cs("w32_dpp_snan_sign_moves", [
            "s_mov_b32 vcc_lo, 0",
            "v_mov_b32_e32 v1, 0x7f800001",
            "v_mov_b32_e32 v3, 0",
            "v_cndmask_b32_dpp v2, -v1, v3, vcc_lo quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_cmp_class_f32_dpp vcc_lo, -v1, 1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "v_cndmask_b32_e32 v4, 0, 1, vcc_lo",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v2, v6, s[0:3], 0 offen",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:128",
        ], wave64=False)
        self.assertNotIn("OpFNegate", spv)
        self.assertNotIn("FAbs", spv)
        mem = run_cs(spv, wave64=False)
        self.assertEqual([mem[4 * i] for i in range(32)], [0xFF800001] * 32)
        self.assertEqual([mem[128 + 4 * i] for i in range(32)], [1] * 32)


if __name__ == "__main__":
    unittest.main()
