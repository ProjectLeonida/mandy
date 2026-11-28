"""Buffer-write metadata (`writes_buffer` and `buffer_write_slots`) tests (Item 3B.7).

Verifies the agreed runtime metadata contract:
  * `meta["writes_buffer"]` is always emitted (`bool`):
      - `False` for read-only buffer shaders, image-store/image-atomic-only shaders, and LDS-only shaders.
      - `True` for any shader with buffer/tbuffer/global/flat/scalar buffer stores or atomics.
  * `meta["buffer_write_slots"]` is always emitted:
      - `[]` when `writes_buffer == False`.
      - Sorted list of integer root/table descriptor offsets when every buffer write is statically traceable.
      - `"dynamic"` when any buffer store or atomic base cannot be statically traced.
"""
import unittest

import support
import rdna2spirv

EPILOGUE = ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]


def translate_meta(name, lines):
    asm = support.translate_lines(name, lines)
    return rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, asm)


class WritesBufferMetaTests(unittest.TestCase):
    def test_read_only_shader_has_writes_buffer_false(self):
        meta = translate_meta("ps_ro_buf", [
            "v_mov_b32 v1, 0",
            "buffer_load_dword v0, v1, s[4:7], 0 offen",
            "global_load_dword v2, v1, s[0:1]",
            "s_load_dword s8, s[0:1], 0",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta["writes_buffer"], False)
        self.assertEqual(meta["buffer_write_slots"], [])

    def test_image_store_and_image_atomic_only_have_writes_buffer_false(self):
        meta_store = translate_meta("ps_img_store_only", [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_store v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta_store["writes_buffer"], False)
        self.assertEqual(meta_store["buffer_write_slots"], [])

        meta_atom = translate_meta("ps_img_atomic_only", [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "image_atomic_add v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta_atom["writes_buffer"], False)
        self.assertEqual(meta_atom["buffer_write_slots"], [])

    def test_buffer_store_root_vsharp_is_traceable(self):
        meta = translate_meta("ps_buf_st_root", [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0",
            "buffer_store_dword v0, v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta["writes_buffer"], True)
        self.assertEqual(meta["buffer_write_slots"], [4])

    def test_buffer_store_table_vsharp_at_const_offsets_is_traceable(self):
        meta = translate_meta("ps_buf_st_table", [
            "s_mov_b32 s1, 0",
            "s_load_dwordx4 s[8:11], s[0:1], 0x80",
            "s_load_dwordx4 s[12:15], s[0:1], 0x40",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0",
            "buffer_store_dword v0, v1, s[8:11], 0 offen",
            "buffer_store_format_x v0, v1, s[12:15], 0 offen",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta["writes_buffer"], True)
        self.assertEqual(meta["buffer_write_slots"], [64, 128])

    def test_buffer_atomic_sets_writes_buffer_true(self):
        meta = translate_meta("ps_buf_atom", [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0",
            "buffer_atomic_add v0, v1, s[4:7], 0 offen glc",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta["writes_buffer"], True)
        self.assertEqual(meta["buffer_write_slots"], [4])

    def test_global_store_and_atomic_root_vs_dynamic(self):
        meta_root = translate_meta("ps_glob_st_root", [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0",
            "global_store_dword v1, v0, s[2:3]",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta_root["writes_buffer"], True)
        self.assertEqual(meta_root["buffer_write_slots"], [2])

        meta_dyn = translate_meta("ps_glob_st_dyn", [
            "v_mov_b32 v0, 0x1000",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 7",
            "global_store_dword v[0:1], v2, off",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta_dyn["writes_buffer"], True)
        self.assertEqual(meta_dyn["buffer_write_slots"], "dynamic")

        meta_gatom = translate_meta("ps_glob_atom_dyn", [
            "v_mov_b32 v0, 0x1000",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 7",
            "global_atomic_add v3, v[0:1], v2, off glc",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta_gatom["writes_buffer"], True)
        self.assertEqual(meta_gatom["buffer_write_slots"], "dynamic")

    def test_dynamic_table_vsharp_store_reports_dynamic_slots(self):
        meta = translate_meta("ps_buf_st_dyn_tab", [
            "s_mov_b32 s1, 0",
            "s_load_dwordx4 s[8:11], s[0:1], s2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0",
            "buffer_store_dword v0, v1, s[8:11], 0 offen",
            "s_waitcnt vmcnt(0)",
        ] + EPILOGUE)
        self.assertIs(meta["writes_buffer"], True)
        self.assertEqual(meta["buffer_write_slots"], "dynamic")

    def test_scalar_atomics_set_writes_buffer_true(self):
        import shader_ir
        import shader_meta
        for line, want_slots in (
            ("s_atomic_add s4, s[0:1], 0", [0]),
            ("s_buffer_atomic_add s8, s[4:7], 0", [4]),
        ):
            with self.subTest(line=line):
                insns = [shader_ir.parse_line(line), shader_ir.parse_line("s_endpgm")]
                info = shader_meta.analyse_instructions([insns], [[]])
                self.assertIs(info["writes_buffer"], True)
                self.assertEqual(info["buffer_write_slots"], want_slots)


if __name__ == "__main__":
    unittest.main()
