"""Semantic tests for s_cmov_b32, s_cmovk_i32, and s_movk_i32 with special and lane-mask destinations.

Verifies (wave32 and wave64, CPU interpreter, stdlib only):
  1. s_cmov_b32 / s_cmovk_i32 targeting exec_lo / exec_hi: per-lane bit extraction for constants and scalar
     SGPRs, cross-half wave64 gather (exec_lo <- vcc_hi), untouched-half preservation in wave64, and SCC kept.
  2. s_cmov_b32 / s_cmovk_i32 targeting vcc_lo / vcc_hi: per-lane mask update, untouched-half preservation in
     wave64, and lane-mask tracking preserved for subsequent v_cndmask_b32 / s_bcnt1_i32_b64.
  3. s_cmov_b32 / s_cmovk_i32 targeting m0 and null (SCC preserved).
  4. s_cmov_b32 / s_cmovk_i32 targeting a lane-mask SGPR (mask source, 0, -1, 0xffff): preserves lane_masks
     state so a subsequent s_bcnt1_i32_b32 counts the wave's bits instead of treating the lane bool as data.
  5. s_movk_i32 targeting wave64 vcc_lo / vcc_hi with 0 and -1 (0xffff) preserves the other half of VCC, and
     s_movk_i32 sN, 0/-1 produces EITHER in lane_masks so CFG merges with a lane mask stay a lane mask.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

W32_BUILD_VCC = [
    "v_mov_b32_e32 v1, s4",
    "v_and_b32_e32 v2, 31, v0",
    "v_lshrrev_b32_e32 v3, v2, v1",
    "v_and_b32_e32 v3, 1, v3",
    "v_cmp_ne_u32_e64 vcc_lo, 0, v3",
]

W64_BUILD_VCC = [
    "v_mov_b32_e32 v11, s5",
    "v_cmp_lt_u32_e64 vcc, v0, 32",
    "v_cndmask_b32_e64 v1, v11, s4, vcc",
    "v_and_b32_e32 v2, 31, v0",
    "v_lshrrev_b32_e32 v3, v2, v1",
    "v_and_b32_e32 v3, 1, v3",
    "v_cmp_ne_u32_e64 vcc, 0, v3",
]


def translate_cs(name: str, lines: list[str], threads: int, emulate: bool = False) -> str:
    has_labels = any(ln.endswith(":") for ln in lines)
    encoded_lines = lines
    if has_labels:
        lbl_idx = {}
        ins_count = 0
        for ln in lines:
            if ln.endswith(":"):
                lbl_idx[ln[:-1]] = ins_count
            else:
                ins_count += 1
        encoded_lines = []
        idx = 0
        for ln in lines:
            if ln.endswith(":"):
                continue
            enc = 0
            if ln.startswith(("s_cbranch_", "s_branch ")):
                op, target = ln.split()
                rel = (lbl_idx[target] - idx - 1) & 0xFFFF
                enc = 0xBF880000 | rel
                signed = rel if rel < 0x8000 else rel - 0x10000
                ln = f"{op} {signed}"
            enc_s = ",".join(f"0x{x:02x}" for x in enc.to_bytes(4, "little"))
            encoded_lines.append(f"{ln:<44} ; encoding: [{enc_s}]")
            idx += 1
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", f"cs_{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; cs_{name}\n")
            for ln in encoded_lines:
                f.write(f"\t{ln}\n")
        with open(os.path.join(tmp, f"cs_{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", f"cs_{name}.fssm"), "wb") as f:
            # FSSM magic + wave64 bit (u32[0xD4] bit 5 = 0) so wave64 shaders stay wave64 even with b32-only ops.
            hdr = bytearray(0x138)
            hdr[0:4] = b"FSSM"
            if threads == 32:
                struct.pack_into("<I", hdr, 0xD4, 0x20)
            f.write(bytes(hdr) + struct.pack("<4I", 0x8090, threads, 1, 1))
        old_emulate = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emulate
        return spv


def run_cs(spv: str, threads: int, s4: int = 0, s5: int = 0, s6: int = 0, s7: int = 0) -> dict[int, int]:
    pc = VSHARP + [s4 & M32, s5 & M32, s6 & M32, s7 & M32] + [0] * 24
    return si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )


def bits_list(mask: int, width: int) -> list[int]:
    return [1 if (mask >> i) & 1 else 0 for i in range(width)]


class Cmov32MaskAndDstTests(unittest.TestCase):
    def test_cmov_and_cmovk_exec_wave32_and_wave64(self):
        # Wave32: s_cmov_b32 exec_lo, s4 and s_cmovk_i32 exec_lo, 0x5 with SCC in {0, 1}.
        asm32 = [
            "s_mov_b32 s10, exec_lo",
            "v_mov_b32_e32 v4, 10",
            "s_cmp_eq_u32 s6, 1",
            "s_cmov_b32 exec_lo, s4",
            "s_cselect_b32 s20, 1, 0",
            "v_mov_b32_e32 v4, 20",
            "s_mov_b32 exec_lo, s10",
            "v_mov_b32_e32 v5, 30",
            "s_cmp_eq_u32 s6, 1",
            "s_cmovk_i32 exec_lo, 0x5",
            "s_cselect_b32 s21, 1, 0",
            "v_mov_b32_e32 v5, 40",
            "s_mov_b32 exec_lo, s10",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:128",
            "v_mov_b32_e32 v7, s20",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:256",
            "v_mov_b32_e32 v7, s21",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:384",
            "s_endpgm",
        ]
        spv32 = translate_cs("cmov_exec_w32", asm32, 32)
        for s4, scc in ((0x000000A5, 0), (0x000000A5, 1), (0x80000001, 1)):
            with self.subTest(wave=32, s4=hex(s4), scc=scc):
                mem = run_cs(spv32, 32, s4=s4, s6=scc)
                ex1 = s4 if scc else M32
                ex2 = 0x5 if scc else M32
                self.assertEqual([mem[4 * i] for i in range(32)],
                                 [20 if (ex1 >> i) & 1 else 10 for i in range(32)])
                self.assertEqual([mem[128 + 4 * i] for i in range(32)],
                                 [40 if (ex2 >> i) & 1 else 30 for i in range(32)])
                self.assertEqual([mem[256 + 4 * i] for i in range(32)], [scc] * 32)
                self.assertEqual([mem[384 + 4 * i] for i in range(32)], [scc] * 32)

        # Wave64: s_cmov_b32 exec_lo, vcc_hi (cross-half, keeps exec_hi) and s_cmovk_i32 exec_hi, 0x3 (keeps exec_lo).
        asm64 = W64_BUILD_VCC + [
            "s_mov_b64 s[10:11], exec",
            "v_mov_b32_e32 v4, 10",
            "s_cmp_eq_u32 s6, 1",
            "s_cmov_b32 exec_lo, vcc_hi",
            "v_mov_b32_e32 v4, 20",
            "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v5, 30",
            "s_cmp_eq_u32 s6, 1",
            "s_cmovk_i32 exec_hi, 0x3",
            "v_mov_b32_e32 v5, 40",
            "s_mov_b64 exec, s[10:11]",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
            "s_endpgm",
        ]
        spv64 = translate_cs("cmov_exec_w64", asm64, 64)
        for lo, hi, scc in ((0x00FF00FF, 0x00010005, 0), (0x00FF00FF, 0x00010005, 1), (0x0, 0x80000002, 1)):
            with self.subTest(wave=64, lo=hex(lo), hi=hex(hi), scc=scc):
                mem = run_cs(spv64, 64, s4=lo, s5=hi, s6=scc)
                ex1 = (hi | (M32 << 32)) if scc else ((M32 << 32) | M32)
                ex2 = (M32 | (0x3 << 32)) if scc else ((M32 << 32) | M32)
                self.assertEqual([mem[4 * i] for i in range(64)],
                                 [20 if (ex1 >> i) & 1 else 10 for i in range(64)])
                self.assertEqual([mem[256 + 4 * i] for i in range(64)],
                                 [40 if (ex2 >> i) & 1 else 30 for i in range(64)])

    def test_cmov_and_cmovk_vcc_m0_null(self):
        # Wave64: s_cmov_b32 vcc_lo, s7 (preserves vcc_hi), s_cmovk_i32 vcc_hi, 0x0f (preserves vcc_lo),
        # plus m0 and null destinations (SCC preserved).
        asm = W64_BUILD_VCC + [
            "s_mov_b32 m0, 0x11112222",
            "s_cmp_eq_u32 s6, 1",
            "s_cmov_b32 vcc_lo, s7",
            "s_cmovk_i32 vcc_hi, 0x0f",
            "s_cmov_b32 m0, s7",
            "s_cmovk_i32 null, 0x1234",
            "s_cselect_b32 s20, 1, 0",
            "s_bcnt1_i32_b64 s21, vcc",
            "v_cndmask_b32_e64 v4, 0, 1, vcc",
            "v_mov_b32_e32 v5, m0",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
            "v_mov_b32_e32 v7, s20",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:512",
            "v_mov_b32_e32 v7, s21",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:768",
            "s_endpgm",
        ]
        spv = translate_cs("cmov_vcc_m0_null", asm, 64)
        for lo, hi, s7, scc in ((0x0F0F0003, 0x80000001, 0x000000A5, 0),
                                (0x0F0F0003, 0x80000001, 0x000000A5, 1)):
            with self.subTest(lo=hex(lo), hi=hex(hi), s7=hex(s7), scc=scc):
                mem = run_cs(spv, 64, s4=lo, s5=hi, s6=scc, s7=s7)
                want_vcc = (s7 | (0x0F << 32)) if scc else (lo | (hi << 32))
                want_m0 = s7 if scc else 0x11112222
                self.assertEqual([mem[4 * i] for i in range(64)], bits_list(want_vcc, 64))
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [want_m0] * 64)
                self.assertEqual([mem[512 + 4 * i] for i in range(64)], [scc] * 64)
                self.assertEqual([mem[768 + 4 * i] for i in range(64)], [want_vcc.bit_count()] * 64)

    def test_cmov_and_cmovk_preserve_sgpr_lane_mask_state(self):
        # Wave32: s12 holds VCC mask; s_cmov_b32 s12, s13 (another lane mask) and s_cmovk_i32 s12, 0 / -1
        # must keep s12 in lane_masks so s_bcnt1_i32_b32 counts the wave's lanes rather than returning 32 or 0.
        asm = W32_BUILD_VCC + [
            "s_mov_b32 s12, vcc_lo",
            "v_mov_b32_e32 v1, s5",
            "v_and_b32_e32 v2, 31, v0",
            "v_lshrrev_b32_e32 v3, v2, v1",
            "v_and_b32_e32 v3, 1, v3",
            "v_cmp_ne_u32_e64 s13, 0, v3",
            "s_cmp_eq_u32 s6, 1",
            "s_cmov_b32 s12, s13",
            "s_bcnt1_i32_b32 s20, s12",
            "s_cmp_eq_u32 s7, 1",
            "s_cmovk_i32 s12, 0",
            "s_bcnt1_i32_b32 s21, s12",
            "s_cmp_eq_u32 s7, 2",
            "s_cmovk_i32 s12, 0xffff",
            "s_bcnt1_i32_b32 s22, s12",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "v_mov_b32_e32 v7, s20",
            "buffer_store_dword v7, v6, s[0:3], 0 offen",
            "v_mov_b32_e32 v7, s21",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:128",
            "v_mov_b32_e32 v7, s22",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:256",
            "s_endpgm",
        ]
        spv = translate_cs("cmov_sgpr_mask", asm, 32)
        for s4, s5, s6, s7 in ((0x00000007, 0x0000001F, 0, 0),
                               (0x00000007, 0x0000001F, 1, 1),
                               (0x00000007, 0x0000001F, 0, 2)):
            with self.subTest(s4=hex(s4), s5=hex(s5), s6=s6, s7=s7):
                mem = run_cs(spv, 32, s4=s4, s5=s5, s6=s6, s7=s7)
                m1 = s5 if s6 == 1 else s4
                m2 = 0 if s7 == 1 else m1
                m3 = M32 if s7 == 2 else m2
                self.assertEqual([mem[4 * i] for i in range(32)], [m1.bit_count()] * 32)
                self.assertEqual([mem[128 + 4 * i] for i in range(32)], [m2.bit_count()] * 32)
                self.assertEqual([mem[256 + 4 * i] for i in range(32)], [m3.bit_count()] * 32)

    def test_movk_i32_vcc_half_preserves_other_half_and_cfg_merge(self):
        # Wave64: s_movk_i32 vcc_lo, 0 preserves vcc_hi; s_movk_i32 vcc_hi, 0 clears lanes 32..63 and preserves vcc_lo;
        # s_movk_i32 vcc_hi, -1 sets lanes 32..63 and preserves vcc_lo.
        cases = [
            ("vcc_lo, 0", lambda lo, hi: hi << 32),
            ("vcc_hi, 0", lambda lo, hi: lo),
            ("vcc_lo, -1", lambda lo, hi: M32 | (hi << 32)),
            ("vcc_hi, 0xffff", lambda lo, hi: lo | (M32 << 32)),
        ]
        for idx, (op_args, ref) in enumerate(cases):
            asm = W64_BUILD_VCC + [
                f"s_movk_i32 {op_args}",
                "v_cndmask_b32_e64 v4, 0, 1, vcc",
                "s_bcnt1_i32_b64 s20, vcc",
                "v_lshlrev_b32_e32 v6, 2, v0",
                "buffer_store_dword v4, v6, s[0:3], 0 offen",
                "v_mov_b32_e32 v7, s20",
                "buffer_store_dword v7, v6, s[0:3], 0 offen offset:256",
                "s_endpgm",
            ]
            spv = translate_cs(f"movk_vcc_half_{idx}", asm, 64)
            lo, hi = 0x00FF0005, 0x80000013
            with self.subTest(op_args=op_args):
                mem = run_cs(spv, 64, s4=lo, s5=hi)
                want = ref(lo, hi)
                self.assertEqual([mem[4 * i] for i in range(64)], bits_list(want, 64))
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [want.bit_count()] * 64)

        # CFG join: s_movk_i32 s12, 0 on one path and vcc_lo on the other must merge as MASK so s_bcnt1 counts lanes.
        asm_cfg = W32_BUILD_VCC + [
            "s_movk_i32 s12, 0",
            "s_cmp_eq_u32 s6, 0",
            "s_cbranch_scc1 BB_join",
            "s_mov_b32 s12, vcc_lo",
            "BB_join:",
            "s_bcnt1_i32_b32 s20, s12",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "v_mov_b32_e32 v7, s20",
            "buffer_store_dword v7, v6, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv_cfg = translate_cs("movk_cfg_merge", asm_cfg, 32)
        for s6 in (0, 1):
            with self.subTest(cfg_s6=s6):
                mem = run_cs(spv_cfg, 32, s4=0x00000015, s6=s6)
                want = 0 if s6 == 0 else (0x00000015).bit_count()
                self.assertEqual([mem[4 * i] for i in range(32)], [want] * 32)


if __name__ == "__main__":
    unittest.main()
