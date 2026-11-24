"""Ordinary GFX10 image_load selects channels of a single fragment (RDNA2 table 42)."""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spirv_mini_interp as si
import support


def translate(arrayed=False, a16=False, d16=False, mask=15, extra="", opcode="image_load"):
    # Read the sample and layer from user data so selecting either is observable at run time.
    if a16:
        coords = ["v_mov_b32 v4, 0x00030002", "v_mov_b32 v5, s0"]
        addr = "v[4:5]"
    else:
        coords = ["v_mov_b32 v4, 2", "v_mov_b32 v5, 3", "v_mov_b32 v6, s0"]
        if arrayed:
            coords.append("v_mov_b32 v7, s1")
        addr = "v[4:7]" if arrayed else "v[4:6]"
    dim = "SQ_RSRC_IMG_2D_MSAA_ARRAY" if arrayed else "SQ_RSRC_IMG_2D_MSAA"
    mods = " ".join(x for x in ("a16" if a16 else "", "d16" if d16 else "", extra) if x)
    count = (mask.bit_count() + 1) // 2 if d16 else mask.bit_count()
    dest = "v8" if count == 1 else f"v[8:{7 + count}]"
    return support.translate_lines("ps_texel_ms", coords + [
        f"{opcode} {dest}, {addr}, s[8:15] dmask:{mask:#x} dim:{dim} {mods}",
        "exp mrt0 v8, v9, v10, v11 done vm", "s_endpgm",
    ])


class MultisampleImageLoadTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(support.rdna2spirv, "MSAA_IMAGE_LOAD", True)
        patch.start()
        self.addCleanup(patch.stop)

    def test_numeric_class_channel_mask_sample_and_array_slice(self):
        for arrayed in (False, True):
            for a16 in (False, True):
                module = translate(arrayed, a16, mask=0xb)
                self.assertIn(" Sample ", module)
                for cls, fmt in enumerate((77, 75, 76)):  # RGBA32 float, uint, sint
                    for sample in range(4):
                        with self.subTest(arrayed=arrayed, a16=a16, cls=cls, sample=sample):
                            layer = 1 if arrayed else 0
                            coord = (2, 3, layer) if arrayed else (2, 3)
                            calls = []

                            def fetch(xy, fragid):
                                calls.append((tuple(xy), fragid))
                                base = 100 * (xy[2] if arrayed else 0) + 10 * fragid
                                values = [base + c + 1 for c in range(4)]
                                if cls == 0:
                                    return tuple(float(x) for x in values)
                                return tuple(x if cls == 1 else -x for x in values)

                            descriptor = [0x1000, fmt << 20, 0, 0, 0, 0, 0, 0]
                            constants = [layer, sample] if arrayed else [sample, 0]
                            if a16 and arrayed:
                                constants = [layer | sample << 16, 0]
                            _, outputs = si.run_workgroup(
                                module, 1, 32, return_outputs=True,
                                push_constants=constants + [0] * 6 + descriptor,
                                resources={"%desc_map": [0, *descriptor, 1],
                                           f"%itx_ms{cls}_{int(arrayed)}": [None, fetch]},
                            )
                            self.assertEqual(calls, [(coord, sample)])
                            chosen = [100 * layer + 10 * sample + c for c in (1, 2, 4)]
                            if cls == 0:
                                chosen = [struct.unpack("<I", struct.pack("<f", x))[0] for x in chosen]
                            elif cls == 2:
                                chosen = [-x & 0xffffffff for x in chosen]
                            self.assertEqual(list(outputs[0]["%out_mrt0"][:3]), chosen)

    def test_target_opt_in_and_unsupported_forms(self):
        with mock.patch.object(support.rdna2spirv, "MSAA_IMAGE_LOAD", False):
            with self.assertRaisesRegex(support.rdna2spirv.Unsupported, "msaa-image-load"):
                translate()
        for flag in ("glc", "tfe", "lwe"):
            with self.subTest(flag=flag), self.assertRaises(support.rdna2spirv.Unsupported):
                translate(extra=flag)
        with self.assertRaises(support.rdna2spirv.Unsupported):
            translate(opcode="image_msaa_load")
        for dest, addr, mask in (("v8", "v[4:5]", "0x1"), ("v8", "v[4:6]", "0xf"),
                                ("v8", "v[4:6]", "0")):
            with self.subTest(dest=dest, addr=addr, mask=mask), self.assertRaises(support.rdna2spirv.Unsupported):
                support.translate_lines("ps_texel_ms_bad", [
                    f"image_load {dest}, {addr}, s[8:15] dmask:{mask} dim:SQ_RSRC_IMG_2D_MSAA",
                    "exp mrt0 v8, v8, v8, v8 done vm", "s_endpgm",
                ])

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "requires SPIRV-Tools")
    def test_packed_results_validate_on_both_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "ms.spv")
            for arrayed in (False, True):
                module = translate(arrayed, a16=True, d16=True)
                subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-", "-o", out],
                               input=module, text=True, check=True)
                subprocess.run(["spirv-val", "--target-env", "vulkan1.2", out], check=True)


if __name__ == "__main__":
    unittest.main()
