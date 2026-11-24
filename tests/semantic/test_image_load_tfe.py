"""TFE/LWE status VGPR on image_load and image_load_mip (docs/design.md "Typed image loads").

With `tfe` or `lwe`, RDNA2 MIMG reads write the fetch status into `vdata[data_vgprs]` after the dmask data
components (2 components per VGPR under `d16`, 1 otherwise). Resident accesses write 0 for active lanes
and preserve the prior value for inactive lanes.
"""
import re
import unittest

import support


def _translate(lines):
    return support.translate_lines(
        "ps_imgload_tfe_unit",
        ["v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 0"] + lines
        + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"],
    )


def _has_status_zero_store(spvasm: str, reg_num: int) -> bool:
    """Return True if the translated SPIR-V writes 0 (%c0) to `%reg_v{reg_num}`."""
    var = f"%reg_v{reg_num}"
    selects = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+)\s*=\s*OpSelect\s+%uint\s+%\w+\s+(%\w+)\s+%\w+", line)
        if m:
            selects[m.group(1)] = m.group(2)
    for line in spvasm.splitlines():
        m = re.match(rf"\s*OpStore\s+{re.escape(var)}\s+(%\w+)", line)
        if m:
            val = selects.get(m.group(1), m.group(1))
            if val == "%c0":
                return True
    return False


class ImageLoadTfeTests(unittest.TestCase):
    def test_image_load_tfe_and_lwe_zero_status_vgpr(self):
        for mod in ("tfe", "lwe"):
            with self.subTest(mod=mod):
                spv = _translate([
                    "v_mov_b32 v8, 0x3e8",
                    f"image_load v[7:8], v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm {mod}",
                    "s_waitcnt vmcnt(0)",
                    "v_mov_b32 v0, v8",
                ])
                self.assertTrue(_has_status_zero_store(spv, 8))

    def test_image_load_d16_tfe_status_offset_packs_halves(self):
        spv_two = _translate([
            "v_mov_b32 v8, 0x3e8",
            "image_load v[7:8], v[1:2], s[16:23] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm d16 tfe",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v0, v8",
        ])
        self.assertTrue(_has_status_zero_store(spv_two, 8))

        spv_three = _translate([
            "v_mov_b32 v9, 0x3e8",
            "image_load v[7:9], v[1:2], s[16:23] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm d16 tfe",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v0, v9",
        ])
        self.assertTrue(_has_status_zero_store(spv_three, 9))

    def test_image_load_coherent_and_mip_tfe_zero_status_vgpr(self):
        spv_glc = _translate([
            "v_mov_b32 v8, 0x3e8",
            "image_load v[7:8], v[1:2], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc tfe",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v0, v8",
        ])
        self.assertTrue(_has_status_zero_store(spv_glc, 8))

        spv_mip = _translate([
            "v_mov_b32 v8, 0x3e8",
            "image_load_mip v[7:8], v[1:3], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm tfe",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v0, v8",
        ])
        self.assertTrue(_has_status_zero_store(spv_mip, 8))

        spv_mip_3d = _translate([
            "v_mov_b32 v4, 0",
            "v_mov_b32 v8, 0x3e8",
            "image_load_mip v[7:8], v[1:4], s[16:23] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm tfe",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v0, v8",
        ])
        self.assertTrue(_has_status_zero_store(spv_mip_3d, 8))


if __name__ == "__main__":
    unittest.main()
