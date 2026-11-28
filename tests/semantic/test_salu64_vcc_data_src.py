"""Regression tests for plain-data VCC sources in 64-bit scalar ops and Gen.src64_data.

Gen.is_mask_op previously returned True unconditionally for vcc/vcc_lo/vcc_hi, even when
lane_masks.py proved VCC holds ordinary scalar data (written via s_mov_b32 vcc_lo/vcc_hi).
That routed plain-data VCC operands in s_not_b64, s_{and,or,xor,andn2,orn2,nand,nor,xnor}_b64,
and s_cselect_b64 onto the per-lane mask path (dropping %vcc_hi and the high dword of the
paired SGPR, duplicating the low 32-bit result into both destination words, and voting SCC
across the wave), and caused s_cmov_b64 and s_quadmask_*/s_bitreplicate_b64_b32 to reject
plain-data VCC operands as lane masks. Gen.src64_data also declined plain-data VCC.

All shaders here are synthetic and hand-written; no game files, LLVM, or GPU are used.

Run directly:
    python -I tests/semantic/test_salu64_vcc_data_src.py -v
"""
import gc
import json
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
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def _translate_cs(name, lines, *, wave32=False):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for ln in lines:
                f.write(f"\t{ln}\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        if wave32:
            os.mkdir(os.path.join(tmp, "fspx"))
            with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": 0x8020, "threads": [32, 1, 1], "wave32": True}, f)
        else:
            os.mkdir(os.path.join(tmp, "fssm"))
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src)
            gc.collect()
        return spv


def _run_cs(spv, s4, s5, s6=0, s7=0, *, wave32=False, n_words=3):
    wg = 32 if wave32 else 64
    pc = VSHARP + [s4 & M32, s5 & M32, s6 & M32, s7 & M32] + [0] * 24
    mem = si.run_workgroup(
        spv, wg, wg,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory={},
    )
    lane0 = [mem[w * 4] for w in range(n_words)]
    for lane in range(1, wg):
        base = lane * n_words * 4
        got = [mem[base + w * 4] for w in range(n_words)]
        assert got == lane0, f"lane {lane} diverged: {got} != {lane0}"
    return lane0


def _store_s8_s9_scc():
    return [
        "s_cselect_b32 s10, 1, 0",
        "v_mov_b32 v1, s8",
        "v_mov_b32 v2, s9",
        "v_mov_b32 v3, s10",
        "v_mul_lo_u32 v4, v0, 12",
        "buffer_store_dword v1, v4, s[0:3], 0 offen",
        "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
        "buffer_store_dword v3, v4, s[0:3], 0 offen offset:8",
        "s_endpgm",
    ]


def _quadmask64(x):
    return sum(1 << i for i in range(16) if (x >> (4 * i)) & 0xF)


class Salu64VccDataSrcTests(unittest.TestCase):
    def test_bitwise_b64_ops_with_plain_vcc_source(self):
        # Test all 9 64-bit bitwise ops with VCC holding plain 64-bit data (s[4:5] -> vcc),
        # including high-word-only non-zero results to verify BOTH dwords and SCC.
        ops = [
            ("s_and_b64",   lambda a, b: a & b),
            ("s_or_b64",    lambda a, b: a | b),
            ("s_xor_b64",   lambda a, b: a ^ b),
            ("s_andn2_b64", lambda a, b: a & (~b & M64)),
            ("s_orn2_b64",  lambda a, b: a | (~b & M64)),
            ("s_nand_b64",  lambda a, b: (~(a & b)) & M64),
            ("s_nor_b64",   lambda a, b: (~(a | b)) & M64),
            ("s_xnor_b64",  lambda a, b: (~(a ^ b)) & M64),
        ]
        pairs = [
            # (vcc_64, sgpr_64) — includes high-only non-zero results where low word is 0
            (0x55667788_11223344, 0x0F0F0F0F_00000000),
            (0x00000000_00000000, 0x80000000_00000000),
            (0xFFFFFFFF_FFFFFFFF, 0xFFFFFFFF_FFFFFFFF),
            (0xDEADBEEF_00000000, 0xDEADBEEF_00000000),
        ]
        for op, fn in ops:
            # Test both operand orders: (s[6:7], vcc) and (vcc, s[6:7])
            for order in ("s_vcc", "vcc_s"):
                insn = f"{op} s[8:9], s[6:7], vcc" if order == "s_vcc" else f"{op} s[8:9], vcc, s[6:7]"
                lines = [
                    "s_mov_b32 vcc_lo, s4",
                    "s_mov_b32 vcc_hi, s5",
                    insn,
                ] + _store_s8_s9_scc()
                spv = _translate_cs(f"cs_{op}_{order}", lines, wave32=False)
                for vcc_val, s_val in pairs:
                    lhs, rhs = (s_val, vcc_val) if order == "s_vcc" else (vcc_val, s_val)
                    want = fn(lhs, rhs)
                    with self.subTest(op=op, order=order, vcc=hex(vcc_val), s=hex(s_val)):
                        got = _run_cs(
                            spv,
                            vcc_val & M32, vcc_val >> 32,
                            s_val & M32, s_val >> 32,
                            wave32=False, n_words=3,
                        )
                        self.assertEqual(got, [want & M32, want >> 32, int(want != 0)])

        # s_not_b64 s[8:9], vcc
        spv_not = _translate_cs(
            "cs_s_not_b64_vcc",
            ["s_mov_b32 vcc_lo, s4", "s_mov_b32 vcc_hi, s5", "s_not_b64 s[8:9], vcc"] + _store_s8_s9_scc(),
            wave32=False,
        )
        for vcc_val in (0, M64, 0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000, 0x55667788_11223344):
            want = (~vcc_val) & M64
            with self.subTest(op="s_not_b64", vcc=hex(vcc_val)):
                got = _run_cs(spv_not, vcc_val & M32, vcc_val >> 32, wave32=False, n_words=3)
                self.assertEqual(got, [want & M32, want >> 32, int(want != 0)])

    def test_cselect_and_cmov_b64_with_plain_vcc_source(self):
        vcc_val = 0xDEADBEEF_11223344
        s_val = 0xCAFEBABE_55667788
        for scc in (0, 1):
            cmp_ln = "s_cmp_eq_u32 1, 1" if scc else "s_cmp_eq_u32 0, 1"
            for order, insn, want in (
                ("cselect_vcc_s", "s_cselect_b64 s[8:9], vcc, s[6:7]", vcc_val if scc else s_val),
                ("cselect_s_vcc", "s_cselect_b64 s[8:9], s[6:7], vcc", s_val if scc else vcc_val),
                ("cmov_vcc", "s_cmov_b64 s[8:9], vcc", vcc_val if scc else s_val),
            ):
                with self.subTest(order=order, scc=scc):
                    lines = [
                        "s_mov_b32 vcc_lo, s4",
                        "s_mov_b32 vcc_hi, s5",
                        "s_mov_b32 s8, s6",
                        "s_mov_b32 s9, s7",
                        cmp_ln,
                        insn,
                    ] + _store_s8_s9_scc()
                    spv = _translate_cs(f"cs_{order}_{scc}", lines, wave32=False)
                    got = _run_cs(
                        spv,
                        vcc_val & M32, vcc_val >> 32,
                        s_val & M32, s_val >> 32,
                        wave32=False, n_words=3,
                    )
                    self.assertEqual(got, [want & M32, want >> 32, scc])

    def test_quadmask_and_bitreplicate_with_plain_vcc_source(self):
        # s_quadmask_b64 s[8:9], vcc
        spv_qm64 = _translate_cs(
            "cs_quadmask64_vcc",
            ["s_mov_b32 vcc_lo, s4", "s_mov_b32 vcc_hi, s5", "s_quadmask_b64 s[8:9], vcc"] + _store_s8_s9_scc(),
            wave32=False,
        )
        for vcc_val in (0, 0xF0000000_00000001, 0x00000000_01020408, 0x80000000_00000000):
            want = _quadmask64(vcc_val)
            with self.subTest(op="s_quadmask_b64", vcc=hex(vcc_val)):
                got = _run_cs(spv_qm64, vcc_val & M32, vcc_val >> 32, wave32=False, n_words=3)
                self.assertEqual(got, [want, 0, int(want != 0)])

        # s_bitreplicate_b64_b32 s[8:9], vcc_lo (wave32 FSPX)
        spv_br = _translate_cs(
            "cs_bitrep_vcclo",
            [
                "s_mov_b32 vcc_lo, s4",
                "s_cmp_eq_u32 1, 1",
                "s_bitreplicate_b64_b32 s[8:9], vcc_lo",
            ] + _store_s8_s9_scc(),
            wave32=True,
        )
        got = _run_cs(spv_br, 0x80000001, 0, wave32=True, n_words=3)
        self.assertEqual(got, [3, 0xC0000000, 1])

    def test_vcc_after_mask_overwritten_with_data_and_lane_mask_controls(self):
        # VCC starts as a lane mask (v_cmp), then both halves are overwritten with plain scalar data.
        # s_and_b64 must now operate on both 32-bit data words instead of the mask path.
        lines = [
            "v_cmp_lt_u32_e32 vcc, v0, 10",
            "s_mov_b32 vcc_lo, s4",
            "s_mov_b32 vcc_hi, s5",
            "s_and_b64 s[8:9], s[6:7], vcc",
        ] + _store_s8_s9_scc()
        spv = _translate_cs("cs_vcc_mask_then_data_and64", lines, wave32=False)
        got = _run_cs(spv, 0x11223344, 0xDEADBEEF, 0x00000000, 0x00FF00FF, wave32=False, n_words=3)
        self.assertEqual(got, [0, 0x00AD00EF, 1])

        # Control: when VCC actually holds a lane mask, s_and_b64 still takes the lane-mask path,
        # and s_cmov_b64 / s_quadmask_b64 still reject it as Unsupported.
        spv_mask = _translate_cs(
            "cs_vcc_mask_and64_ctrl",
            [
                "v_cmp_ge_u32_e32 vcc, v0, 40",
                "v_cmp_lt_u32_e64 s[6:7], v0, 50",
                "s_and_b64 s[8:9], vcc, s[6:7]",
                "s_cselect_b32 s10, 1, 0",
                "s_bcnt1_i32_b64 s11, s[8:9]",
                "v_mov_b32 v1, s11",
                "v_mov_b32 v2, s10",
                "v_mul_lo_u32 v4, v0, 8",
                "buffer_store_dword v1, v4, s[0:3], 0 offen",
                "buffer_store_dword v2, v4, s[0:3], 0 offen offset:4",
                "s_endpgm",
            ],
            wave32=False,
        )
        self.assertEqual(_run_cs(spv_mask, 0, 0, wave32=False, n_words=2), [10, 1])

        for bad in (
            ["v_cmp_lt_u32_e32 vcc, v0, 10", "s_cmov_b64 s[8:9], vcc", "s_endpgm"],
            ["v_cmp_lt_u32_e32 vcc, v0, 10", "s_quadmask_b64 s[8:9], vcc", "s_endpgm"],
            ["v_cmp_lt_u32_e32 vcc, v0, 10", "v_cmp_eq_u64_e64 s[8:9], vcc, v[0:1]", "s_endpgm"],
        ):
            with self.subTest(bad=bad[1]):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate_cs("cs_vcc_mask_reject", bad, wave32=False)


if __name__ == "__main__":
    unittest.main()
