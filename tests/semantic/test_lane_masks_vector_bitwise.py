"""Semantic interpreter tests for vector bitwise lane-mask tracking (`v_and_or_b32`, `v_xnor_b32`, `v_bfi_b32`)."""
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # OOB_SELECT 3 (raw): stores up to num_records
SENTINEL = 0xCAFEBABE


def _translate_cs(name, body_lines, wave_size=32, emulate=False):
    """Translate a synthetic compute shader and return `(spv, gen)`."""
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name}\n")
                for ln in body_lines:
                    f.write(f"\t{ln}\n")
                f.write("\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            # COMPUTE_PGM_RSRC2: 0x8088 -> 4 user SGPRs (s[0:3] = output V#), TGID_X.
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gc.collect()
            gen = rdna2spirv.translate.last_gen
            assert gen.wave64 == (wave_size == 64), f"{name}: expected wave64={wave_size == 64}, got {gen.wave64}"
            return spv, gen
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _run_cs(name, body_lines, wave_size=32, emulate=False):
    """Translate a synthetic compute shader and execute every lane in spirv_mini_interp."""
    spv, _ = _translate_cs(name, body_lines, wave_size=wave_size, emulate=emulate)
    sg_size = 32 if emulate else wave_size
    pc = VSHARP + [0] * 28
    return si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory={},
    )


def _bitwise_mask_lines(wave_size):
    """Build shader computing masks via v_and_or_b32, v_xnor_b32, and v_bfi_b32 under partial EXEC."""
    is64 = wave_size == 64
    r = {
        "exec": "exec" if is64 else "exec_lo",
        "mov": "s_mov_b64" if is64 else "s_mov_b32",
        "saveexec": "s_and_saveexec_b64" if is64 else "s_and_saveexec_b32",
        "bcnt": "s_bcnt1_i32_b64" if is64 else "s_bcnt1_i32_b32",
        "s4": "s[4:5]" if is64 else "s4",
        "s6": "s[6:7]" if is64 else "s6",
        "s8": "s[8:9]" if is64 else "s8",
        "s10": "s[10:11]" if is64 else "s10",
        "s12": "s[12:13]" if is64 else "s12",
        "s14": "s[14:15]" if is64 else "s14",
        "s16": "s[16:17]" if is64 else "s16",
        "s18": "s[18:19]" if is64 else "s18",
    }
    lines = [
        f"{r['mov']} {r['exec']}, -1",
        "v_and_b32 v1, 1, v0",
        f"v_cmp_eq_u32_e64 {r['s6']}, v1, 0",        # even lanes
        f"v_cmp_lt_u32_e64 {r['s8']}, v0, 10",       # lanes < 10
        "v_and_b32 v1, 2, v0",
        f"v_cmp_eq_u32_e64 {r['s10']}, v1, 0",       # (lane & 2) == 0
        f"v_cmp_lt_u32_e64 {r['s12']}, v0, 8",       # partial EXEC: lanes 0..7 active
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "v_mov_b32 v4, 0",
    ]
    if is64:
        lines += ["v_mov_b32 v5, 0", "v_mov_b32 v6, 0", "v_mov_b32 v7, 0"]
    lines += [
        f"{r['saveexec']} {r['s4']}, {r['s12']}",
        # Mask ops under partial EXEC must update inactive lanes too.
        "v_and_or_b32 v2, s6, s8, s10",
        "v_xnor_b32 v3, s6, s8",
        "v_bfi_b32 v4, s6, s8, s10",
    ]
    if is64:
        lines += [
            "v_and_or_b32 v5, s7, s9, s11",
            "v_xnor_b32 v6, s7, s9",
            "v_bfi_b32 v7, s7, s9, s11",
        ]
    lines += [
        f"{r['mov']} {r['exec']}, -1",
        "v_readfirstlane_b32 s14, v2",
        "v_readfirstlane_b32 s16, v3",
        "v_readfirstlane_b32 s18, v4",
    ]
    if is64:
        lines += [
            "v_readfirstlane_b32 s15, v5",
            "v_readfirstlane_b32 s17, v6",
            "v_readfirstlane_b32 s19, v7",
        ]
    lines += [
        f"{r['bcnt']} s20, {r['s14']}",
        f"{r['bcnt']} s21, {r['s16']}",
        f"{r['bcnt']} s22, {r['s18']}",
        f"v_cndmask_b32_e64 v8, 10, 20, {r['s14']}",
        f"v_cndmask_b32_e64 v9, 30, 40, {r['s16']}",
        f"v_cndmask_b32_e64 v10, 50, 60, {r['s18']}",
        "v_mov_b32 v11, s20",
        "v_mov_b32 v12, s21",
        "v_mov_b32 v13, s22",
        "v_mul_lo_u32 v14, v0, 36",
    ]
    for off, vreg in enumerate((2, 3, 4, 8, 9, 10, 11, 12, 13)):
        if off == 0:
            lines.append(f"buffer_store_dword v{vreg}, v14, s[0:3], 0 offen")
        else:
            lines.append(f"v_add_nc_u32 v15, {off * 4}, v14")
            lines.append(f"buffer_store_dword v{vreg}, v15, s[0:3], 0 offen")
    return lines


def _readfirstlane_vgpr_dst_lines(wave_size):
    """v_readfirstlane_b32 into a VGPR of a mask-holding VGPR, stored back as data. The
    destination is plain data — VGPR writes are never mask-tracked — so every lane must get
    the broadcast wave mask word, not this lane's 0 / ~0 bit (the mask word here is the low
    word in wave64, which is all v2 holds)."""
    is64 = wave_size == 64
    r = {
        "exec": "exec" if is64 else "exec_lo",
        "mov": "s_mov_b64" if is64 else "s_mov_b32",
        "s6": "s[6:7]" if is64 else "s6",
        "s8": "s[8:9]" if is64 else "s8",
        "s10": "s[10:11]" if is64 else "s10",
    }
    return [
        f"{r['mov']} {r['exec']}, -1",
        "v_and_b32 v1, 1, v0",
        f"v_cmp_eq_u32_e64 {r['s6']}, v1, 0",       # even lanes
        f"v_cmp_lt_u32_e64 {r['s8']}, v0, 10",      # lanes < 10
        f"v_cmp_lt_u32_e64 {r['s10']}, v0, 20",     # lanes < 20
        "v_and_or_b32 v2, s6, s8, s10",             # mask VGPR: (even & <10) | <20
        "v_readfirstlane_b32 v3, v2",               # VGPR dst: the broadcast mask word
        "v_mul_lo_u32 v14, v0, 36",
        "buffer_store_dword v3, v14, s[0:3], 0 offen",
    ]


def _vgpr_stores(spv):
    """Return mapping `vgpr_idx -> [rhs_expression, ...]` for constant-index `%vgpr` OpStores."""
    const_uint = {}
    ptr_vgpr = {}
    defs = {}
    stores = {}
    for ln in spv.splitlines():
        s = ln.strip()
        if " = " in s:
            lhs, rhs = s.split(" = ", 1)
            defs[lhs] = rhs
            parts = rhs.split()
            if parts[:2] == ["OpConstant", "%uint"] and len(parts) == 3:
                const_uint[lhs] = int(parts[2])
            elif (
                len(parts) == 4
                and parts[0] == "OpAccessChain"
                and parts[1] == "%_ptr_Function_uint"
                and parts[2] == "%vgpr"
                and parts[3] in const_uint
            ):
                ptr_vgpr[lhs] = const_uint[parts[3]]
        elif s.startswith("OpStore "):
            parts = s.split()
            if len(parts) >= 3 and parts[1] in ptr_vgpr:
                idx = ptr_vgpr[parts[1]]
                stores.setdefault(idx, []).append(defs.get(parts[2], parts[2]))
    return stores


class _VectorBitwiseMixin:
    wave_size = 32

    def test_vector_bitwise_masks_under_partial_exec(self):
        mem = _run_cs(
            f"cs_vec_bitwise_w{self.wave_size}",
            _bitwise_mask_lines(self.wave_size),
            wave_size=self.wave_size,
            emulate=False,
        )
        want_andor = [((i & 1 == 0) and i < 10) or (i & 2 == 0) for i in range(self.wave_size)]
        want_xnor = [(i & 1 == 0) == (i < 10) for i in range(self.wave_size)]
        want_bfi = [(i < 10) if (i & 1 == 0) else (i & 2 == 0) for i in range(self.wave_size)]
        for lane in range(self.wave_size):
            base = lane * 36
            # The mask reads run while EXEC is narrowed to lanes < 10. Hardware SGPRs hold the
            # whole wave's mask word, and the eager gather sees the lanes EXEC has off too, so the
            # bitwise VALU op composes the 32-bit mask words themselves. The VGPR result is that
            # wave-uniform word in every lane (for the low-half SGPRs in wave64 as well).
            andor_word = sum((1 << i) for i, b in enumerate(want_andor[:32]) if b)
            xnor_word = sum((1 << i) for i, b in enumerate(want_xnor[:32]) if b)
            bfi_word = sum((1 << i) for i, b in enumerate(want_bfi[:32]) if b)
            self.assertEqual(mem[base], andor_word, f"lane {lane} raw and_or")
            self.assertEqual(mem[base + 4], xnor_word, f"lane {lane} raw xnor")
            self.assertEqual(mem[base + 8], bfi_word, f"lane {lane} raw bfi")
            # readfirstlane broadcasts lane 0's copy of the composed wave mask word. Mask consumers
            # extract each lane's own bit again: cndmask follows that lane's membership, and bcnt
            # counts the set bits of the composed masks (low plus high halves in wave64).
            self.assertEqual(mem[base + 12], 20 if want_andor[lane] else 10,
                             f"lane {lane} cndmask and_or")
            self.assertEqual(mem[base + 16], 40 if want_xnor[lane] else 30,
                             f"lane {lane} cndmask xnor")
            self.assertEqual(mem[base + 20], 60 if want_bfi[lane] else 50,
                             f"lane {lane} cndmask bfi")
            self.assertEqual(mem[base + 24], sum(want_andor), f"lane {lane} bcnt and_or")
            self.assertEqual(mem[base + 28], sum(want_xnor), f"lane {lane} bcnt xnor")
            self.assertEqual(mem[base + 32], sum(want_bfi), f"lane {lane} bcnt bfi")

    def test_readfirstlane_mask_vgpr_dst_gets_wave_word(self):
        mem = _run_cs(
            f"cs_rfl_vgpr_w{self.wave_size}",
            _readfirstlane_vgpr_dst_lines(self.wave_size),
            wave_size=self.wave_size,
        )
        # (even & lane < 10) | lane < 20 -> lanes 0..19 of the low word, the same in every lane.
        want = sum(1 << i for i in range(32) if ((i & 1) == 0 and i < 10) or i < 20)
        for lane in range(self.wave_size):
            self.assertEqual(mem[lane * 36], want, f"lane {lane} readfirstlane dst")


class TestVectorBitwiseMasksWave32(_VectorBitwiseMixin, unittest.TestCase):
    wave_size = 32


class TestVectorBitwiseMasksWave64Native(_VectorBitwiseMixin, unittest.TestCase):
    wave_size = 64


class TestVectorBitwiseMasksWave64Emulated(unittest.TestCase):
    """Structural SPIR-V checks for `--wave64-emulate` (spirv_mini_interp rejects OpControlBarrier)."""

    def test_vector_bitwise_masks_emulate_spirv_structure(self):
        spv, gen = _translate_cs(
            "cs_vec_bitwise_w64_emu",
            _bitwise_mask_lines(64),
            wave_size=64,
            emulate=True,
        )
        self.assertTrue(gen.wave64 and gen.emulate)
        self.assertEqual(len(gen.mask_writes), 6)
        # Emulation exchanges through %x_buf instead of subgroup ops; FindILsb (GLSL extinst) is fine —
        # the readfirstlane of a mask VGPR now takes the exchange path like any other readfirstlane.
        self.assertNotIn("OpGroupNonUniform", spv)
        self.assertIn("OpControlBarrier", spv)
        self.assertIn("OpStore %x_it", spv)
        stores = _vgpr_stores(spv)
        for vreg in (2, 5):  # v_and_or_b32 -> final OpBitwiseOr
            self.assertTrue(stores[vreg][-1].startswith("OpBitwiseOr %uint "), stores[vreg][-1])
        for vreg in (3, 6):  # v_xnor_b32 -> final OpNot
            self.assertTrue(stores[vreg][-1].startswith("OpNot %uint "), stores[vreg][-1])
        for vreg in (4, 7):  # v_bfi_b32 -> final OpBitwiseOr
            self.assertTrue(stores[vreg][-1].startswith("OpBitwiseOr %uint "), stores[vreg][-1])

    def test_readfirstlane_vgpr_dst_emulate_structure(self):
        spv, gen = _translate_cs(
            "cs_rfl_vgpr_w64_emu",
            _readfirstlane_vgpr_dst_lines(64),
            wave_size=64,
            emulate=True,
        )
        self.assertTrue(gen.wave64 and gen.emulate)
        self.assertNotIn("OpGroupNonUniform", spv)


class TestVectorBitwiseDataControl(unittest.TestCase):
    def test_non_mask_data_stays_exec_gated_wave32(self):
        lines = [
            "s_mov_b32 exec_lo, -1",
            "v_cmp_lt_u32_e64 s8, v0, 16",
            f"v_mov_b32 v2, {SENTINEL:#x}",
            f"v_mov_b32 v3, {SENTINEL:#x}",
            f"v_mov_b32 v4, {SENTINEL:#x}",
            "s_and_saveexec_b32 s6, s8",
            "v_and_or_b32 v2, v0, 1, 2",
            "v_xnor_b32 v3, v0, 3",
            "v_bfi_b32 v4, v0, 0x0f0f0f0f, 0x10101010",
            "s_mov_b32 exec_lo, s6",
            "v_mul_lo_u32 v10, v0, 12",
            "buffer_store_dword v2, v10, s[0:3], 0 offen",
            "v_add_nc_u32 v11, 4, v10",
            "buffer_store_dword v3, v11, s[0:3], 0 offen",
            "v_add_nc_u32 v11, 8, v10",
            "buffer_store_dword v4, v11, s[0:3], 0 offen",
        ]
        mem = _run_cs("cs_vec_bitwise_data_w32", lines, wave_size=32, emulate=False)
        for lane in range(32):
            base = lane * 12
            if lane < 16:
                self.assertEqual(mem[base], (lane & 1) | 2, f"lane {lane} data and_or")
                self.assertEqual(mem[base + 4], (~(lane ^ 3)) & 0xFFFFFFFF, f"lane {lane} data xnor")
                want_bfi = ((lane & 0x0F0F0F0F) | ((~lane) & 0x10101010)) & 0xFFFFFFFF
                self.assertEqual(mem[base + 8], want_bfi, f"lane {lane} data bfi")
            else:
                self.assertEqual(mem[base], SENTINEL, f"inactive lane {lane} and_or")
                self.assertEqual(mem[base + 4], SENTINEL, f"inactive lane {lane} xnor")
                self.assertEqual(mem[base + 8], SENTINEL, f"inactive lane {lane} bfi")


class TestScalarMaskNumberReads(unittest.TestCase):
    """A scalar ALU op that reads a lane mask as a number gets the wave's mask word, not this
    lane's 0 / ~0 bit (MASK_NUMBER_READERS, ported from #1043's NUMBER_READERS). Mask moves and
    mask logic ops keep the bit so the saveexec/restore and mask-copy idioms still work."""

    def _run(self, lines, wave_size=32, emulate=False):
        prologue = ["s_mov_b64 exec, -1"] if wave_size == 64 else []
        body = prologue + lines + [
            "v_lshlrev_b32_e32 v9, 2, v0",
            "buffer_store_dword v4, v9, s[0:3], 0 offen",
            "s_endpgm",
        ]
        mem = _run_cs("cs_scalar_masknum", body, wave_size=wave_size, emulate=emulate)
        return {mem[4 * lane] for lane in range(wave_size) if 4 * lane in mem}

    def test_scalar_add_reads_mask_word(self):
        # wave32: mask of lanes < 10 -> word 0x3FF. wave64: mask of lanes < 43 -> the s10 operand
        # is the low half, all 32 low lanes set -> 0xFFFFFFFF. The scalar op is uniform: every
        # lane's v4 is the same word + 0x1234.
        for wave, lanes, dst, word in ((32, 10, "s10", 0x3FF), (64, 43, "s[10:11]", 0xFFFFFFFF)):
            with self.subTest(wave=wave):
                got = self._run(
                    [f"v_cmp_gt_u32_e64 {dst}, {lanes}, v0",
                     "s_add_u32 s20, s10, 0x1234",
                     "v_mov_b32_e32 v4, s20"], wave_size=wave)
                self.assertEqual(got, {(word + 0x1234) & 0xFFFFFFFF})

    def test_scalar_readers_table(self):
        # A representative slice of MASK_NUMBER_READERS on wave32: mask of lanes < 10 -> 0x3FF.
        setup = ["v_cmp_gt_u32_e64 s10, 10, v0"]
        cases = {
            "s_add_u32 s20, s10, 0x1234": (0x3FF + 0x1234) & 0xFFFFFFFF,
            "s_sub_u32 s20, s10, 0x1234": (0x3FF - 0x1234) & 0xFFFFFFFF,
            "s_mul_i32 s20, s10, 0x1234": (0x3FF * 0x1234) & 0xFFFFFFFF,
            "s_lshl_b32 s20, s10, 3": (0x3FF << 3) & 0xFFFFFFFF,
            "s_lshr_b32 s20, s10, 3": 0x3FF >> 3,
            "s_brev_b32 s20, s10": int(f"{0x3FF:032b}"[::-1], 2),
            "s_min_u32 s20, s10, 0x1234": min(0x3FF, 0x1234),
            "s_max_u32 s20, s10, 0x1234": max(0x3FF, 0x1234),
        }
        for op, want in cases.items():
            with self.subTest(op=op):
                got = self._run(setup + [op, "v_mov_b32_e32 v4, s20"])
                self.assertEqual(got, {want})

    def test_mask_copy_keeps_bit_then_scalar_reads_word(self):
        # s_mov_b32 is not a number reader: the copy stays a mask (per-lane bit), and the scalar
        # add on the copy still gathers the word because lane_masks tracks s11 as a mask.
        got = self._run(["v_cmp_gt_u32_e64 s10, 10, v0", "s_mov_b32 s11, s10",
                         "s_add_u32 s20, s11, 1", "v_mov_b32_e32 v4, s20"])
        self.assertEqual(got, {0x400})

    def test_exec_as_data_gathers_exec_word(self):
        # exec_lo narrowed to lanes < 10 by saveexec: the EXEC operand's own word still gathers —
        # lanes off in EXEC contribute 0 either way, so every driver agrees.
        got = self._run(["v_cmp_gt_u32_e64 s10, 10, v0",
                         "s_and_saveexec_b32 s12, s10",
                         "s_add_u32 s20, exec_lo, 1",
                         "s_mov_b32 exec_lo, s12",
                         "v_mov_b32_e32 v4, s20"])
        self.assertEqual(got, {0x3FF + 1})


class TestMaskDataReadExecGate(unittest.TestCase):
    """A lane-mask data read gathers the wave's mask word even while EXEC is narrowed. EXEC is a
    per-lane value in the translator's model, not SPIR-V control flow; the mask-data reader is kept
    out of lane-local `if (active)` runs and wave_mask's eager store keeps drivers from moving the
    subgroup gather into an EXEC-selected arm. Thus lanes EXEC has off still contribute the bits they
    hold, as hardware's wave-wide mask register does."""

    def _lines(self, tail):
        return [
            "v_and_b32 v1, 1, v0",
            "v_cmp_eq_u32_e64 vcc_lo, v1, 1",     # VCC = odd lanes, 0xAAAAAAAA
            f"v_mov_b32_e32 v2, {SENTINEL:#x}",    # sentinel: lanes EXEC leaves off keep it
            "s_mov_b32 exec_lo, 0x55555555",      # EXEC = even lanes
        ] + tail + [
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
            "s_endpgm",
        ]

    def test_vcc_read_under_narrowed_exec_gathers_mask_word(self):
        lines = self._lines(["v_mov_b32_e32 v2, vcc_lo"])
        spv, _ = _translate_cs("cs_maskdata_narrowed", lines, wave_size=32, emulate=False)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        mem = _run_cs("cs_maskdata_narrowed_run", lines, wave_size=32, emulate=False)
        for lane in range(32):
            # EXEC-active even lanes store the whole odd-lane VCC word; odd lanes are off in EXEC,
            # so the store doesn't run for them.
            if lane & 1:
                self.assertNotIn(lane * 4, mem, f"lane {lane}")
            else:
                self.assertEqual(mem[lane * 4], 0xAAAAAAAA, f"lane {lane}")

    def test_sgpr_read_under_narrowed_exec_gathers_mask_word(self):
        # Gefik23's crosslane_fuzz family: a compare-written SGPR mask read after EXEC narrows.
        lines = [
            "v_and_b32 v1, 1, v0",
            "v_cmp_eq_u32_e64 s10, v1, 1",         # s10 = odd lanes, 0xAAAAAAAA
            f"v_mov_b32_e32 v2, {SENTINEL:#x}",
            "s_mov_b32 exec_lo, 0x55555555",      # EXEC = even lanes
            "v_mov_b32_e32 v2, s10",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv, _ = _translate_cs("cs_maskdata_sgpr_narrowed", lines, wave_size=32, emulate=False)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        mem = _run_cs("cs_maskdata_sgpr_narrowed_run", lines, wave_size=32, emulate=False)
        for lane in range(32):
            if lane & 1:
                self.assertNotIn(lane * 4, mem, f"lane {lane}")
            else:
                self.assertEqual(mem[lane * 4], 0xAAAAAAAA, f"lane {lane}")

    def test_sgpr_read_after_independent_full_exec_restore_gathers(self):
        # ecox982745's independent-SGPR restore: s7 is runtime-all-ones but is not an EXEC alias.
        # The mask word is still the wave-wide value; no EXEC provenance proof is needed.
        lines = [
            "v_mov_b32_e32 v7, -1",
            "v_readfirstlane_b32 s7, v7",
            "v_and_b32 v1, 1, v0",
            "v_cmp_ne_u32_e64 s10, v1, 0",         # odd lanes
            "s_mov_b32 exec_lo, s7",
            "v_mov_b32_e32 v2, s10",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv, _ = _translate_cs("cs_maskdata_independent_exec", lines, wave_size=32, emulate=False)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        mem = _run_cs("cs_maskdata_independent_exec_run", lines, wave_size=32, emulate=False)
        for lane in range(32):
            self.assertEqual(mem[lane * 4], 0xAAAAAAAA, f"lane {lane}")

    def test_vcc_read_after_exec_restore_gathers(self):
        lines = self._lines(["s_mov_b32 exec_lo, -1", "v_mov_b32_e32 v2, vcc_lo"])
        spv, _ = _translate_cs("cs_maskdata_restored", lines, wave_size=32, emulate=False)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        mem = _run_cs("cs_maskdata_restored_run", lines, wave_size=32, emulate=False)
        want = sum(1 << i for i in range(1, 32, 2))     # 0xAAAAAAAA, every lane
        for lane in range(32):
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_exec_read_under_narrowed_exec_still_gathers(self):
        lines = self._lines(["v_mov_b32_e32 v2, exec_lo"])
        spv, _ = _translate_cs("cs_execdata_narrowed", lines, wave_size=32, emulate=False)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        mem = _run_cs("cs_execdata_narrowed_run", lines, wave_size=32, emulate=False)
        for lane in range(32):
            # EXEC's word is 0x55555555; only the even lanes EXEC leaves on store it.
            if lane & 1:
                self.assertNotIn(lane * 4, mem, f"lane {lane}")
            else:
                self.assertEqual(mem[lane * 4], 0x55555555, f"lane {lane}")

    def test_exec_copy_roundtrip_keeps_mask_gather(self):
        # `s_mov_b32 s4, exec_lo` copies EXEC under an unknown narrowing and `s_mov_b32 exec_lo, s4`
        # restores the same value — EXEC is unchanged, so a mask written under that narrowing must
        # still gather. The copy has to carry the narrowing's tag into `alias`: without it the
        # restore minted a fresh tag in _exec_src_ex, mask_exec_sigs coverage failed and the read
        # kept this lane's 0 / ~0 bit where `s_mov_b32 exec_lo, -1` gathered the word (#1397 review).
        lines = [
            "v_mov_b32_e32 v7, -1",
            "v_readfirstlane_b32 s7, v7",        # s7 = 0xffffffff, not a tracked constant
            "s_mov_b32 exec_lo, s7",             # EXEC = unknown narrowing, runtime all-ones
            "v_and_b32 v1, 1, v0",
            "v_cmp_ne_u32_e64 s10, v1, 0",       # odd-lane mask written under that EXEC
            "s_mov_b32 s4, exec_lo",
            "s_mov_b32 exec_lo, s4",
            "v_mov_b32_e32 v2, s10",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv, _ = _translate_cs("cs_maskdata_exec_copy", lines, wave_size=32, emulate=False)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        mem = _run_cs("cs_maskdata_exec_copy_run", lines, wave_size=32, emulate=False)
        want = sum(1 << i for i in range(1, 32, 2))    # 0xAAAAAAAA, every lane
        for lane in range(32):
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")

    def test_exec_copy_roundtrip_keeps_mask_gather_wave64(self):
        # Same round-trip on the exec_lo half of a wave64 EXEC: the VCC mask's low word must still
        # gather while exec_hi stays clean.
        lines = [
            "s_mov_b64 exec, -1",                # wave64 marker (and a clean EXEC to start)
            "v_mov_b32_e32 v7, -1",
            "v_readfirstlane_b32 s7, v7",
            "s_mov_b32 exec_lo, s7",             # exec_lo = unknown narrowing, runtime all-ones
            "v_and_b32 v1, 1, v0",
            "v_cmp_ne_u32_e64 vcc, v1, 0",
            "s_mov_b32 s4, exec_lo",
            "s_mov_b32 exec_lo, s4",
            "v_mov_b32_e32 v2, vcc_lo",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv, _ = _translate_cs("cs_maskdata_exec_copy_w64", lines, wave_size=64, emulate=False)
        self.assertIn("OpGroupNonUniformBitwiseOr", spv)
        mem = _run_cs("cs_maskdata_exec_copy_w64_run", lines, wave_size=64, emulate=False)
        want = sum(1 << i for i in range(1, 32, 2))    # 0xAAAAAAAA, the vcc_lo word
        for lane in range(64):
            self.assertEqual(mem[lane * 4], want, f"lane {lane}")


if __name__ == "__main__":
    unittest.main()
