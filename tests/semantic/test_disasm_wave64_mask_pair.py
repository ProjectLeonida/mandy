"""A wave64 program listed by tools/shader_disasm.py is translated as the program it is.

llvm-mc lists the lane-mask operands of a VALU instruction for the wave size it is told, and wave32 is its default
for the CPUs used here. Listed that way, the compare below writes `s8` where a wave64 writes s[8:9]; the translator
then leaves s9 as it was, and `s_mov_b64 exec, s[8:9]` takes lanes 32..63 from the old word. shader_disasm.disasm()
gets the wave size from the caller (the FSSM header, or the dispatch state of an FSPX or pipeline-library program)
and has to pass it on to llvm-mc, as it already does to the Python decoder.

The program is hand-written, no game data. The Python decoder case needs nothing else; the llvm-mc case is skipped
without an llvm-mc that has the AMDGPU target (LLVM_MC, tools/mc_command.py).

    python -I tests/semantic/test_disasm_wave64_mask_pair.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import shader_disasm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from test_fspx_wave_size import VSHARP, translate  # noqa: E402

# Each instruction with its words as llvm-mc 23.1.5 assembles it (gfx1013, +wavefrontsize64).
PROGRAM = (
    ("s_mov_b32 s9, 0x100001", "ff0389be 01001000"),            # what s9 held before: lanes 32 and 52
    ("v_mov_b32_e32 v5, 7", "87020a7e"),
    ("v_cmp_gt_u32_e64 s[8:9], 40, v0", "0800c4d4 a8000202"),   # lanes 0..39
    ("s_mov_b64 exec, s[8:9]", "0804febe"),
    ("v_mov_b32_e32 v5, 9", "89020a7e"),
    ("s_mov_b64 exec, -1", "c104febe"),
    ("v_lshlrev_b32_e32 v6, 2, v0", "82000c34"),
    ("buffer_store_dword v5, v6, s[0:3], 0 offen", "001070e0 06050080"),
    ("s_endpgm", "000081bf"),
)
CODE = bytes.fromhex("".join(words for _, words in PROGRAM))
EXPECTED = [9] * 40 + [7] * 24
PAIR = "\tv_cmp_gt_u32_e64 s[8:9], 40, v0 "


class Wave64MaskPair(unittest.TestCase):
    def run_listing(self, listing):
        """Translate the listing as a wave64 compute program and run its 64 lanes: the word each one stores."""
        body = [ln.strip() for ln in listing.splitlines() if ln.startswith("\t")]
        self.assertEqual([ln.split()[0] for ln in body], [text.split()[0] for text, _ in PROGRAM])
        spv, wave64 = translate(body[:-1], wave32=False)
        self.assertTrue(wave64)
        memory = {}
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {
            "%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%local_index": i, "%sg_lane": i,
        }, push_constants=VSHARP + [0] * 32, memory=memory)
        return [memory.get(4 * lane) for lane in range(64)]

    def test_python_decoder_listing(self):
        asm, bad, _ = shader_disasm.disasm(CODE, "gfx1013", backend="python", wave32=False)
        self.assertEqual(bad, 0)
        self.assertEqual(self.run_listing(asm), EXPECTED)
        self.assertIn(PAIR, asm)

    def test_llvm_mc_listing(self):
        asm, bad, err = shader_disasm.disasm(CODE, "gfx1013", wave32=False)
        if not asm.strip():
            self.skipTest("needs llvm-mc with the AMDGPU target: " + (err.strip().splitlines() or ["no output"])[0])
        self.assertEqual(bad, 0, err)
        self.assertEqual(self.run_listing(asm), EXPECTED)
        self.assertIn(PAIR, asm)

    def test_wave32_listing_keeps_single_registers(self):
        asm, _, _ = shader_disasm.disasm(CODE, "gfx1013", backend="python", wave32=True)
        self.assertIn("v_cmp_gt_u32_e64 s8, 40, v0", asm)
        asm, _, _ = shader_disasm.disasm(CODE, "gfx1013", wave32=True)
        if asm.strip():     # with an llvm-mc: the default listing is unchanged
            self.assertIn("v_cmp_gt_u32_e64 s8, 40, v0", asm)


if __name__ == "__main__":
    unittest.main()
