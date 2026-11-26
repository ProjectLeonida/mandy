"""tools/vk_lod_bias.py: the mipLodBias measurement tools/vk_subgroup_info.py --lod-bias runs.

No GPU: the verdict is checked from the three texels the compute shader returns (with the LOD 1 control), a Vulkan library without the calls
gives "unknown", and the embedded SPIR-V is checked for its header (and with spirv-val when it is on PATH).
Standard library only; no game files.
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vk_lod_bias  # noqa: E402


class VerdictTest(unittest.TestCase):
    def test_the_texels_decide(self):
        # level 0 holds 0.0 and level 1 holds 1.0; the shader samples LOD 0 with mipLodBias 0 and +1, and LOD 1
        self.assertIs(vk_lod_bias.verdict(0.0, 1.0, 1.0)[0], False)    # the sampler applies the bias
        self.assertIs(vk_lod_bias.verdict(0.0, 0.0, 1.0)[0], True)     # the sampler drops it
        for plain, biased in ((1.0, 1.0), (1.0, 0.0), (0.0, 0.5), (float("nan"), 0.0)):
            with self.subTest(plain=plain, biased=biased):
                answer, reason = vk_lod_bias.verdict(plain, biased, 1.0)
                self.assertIsNone(answer)
                self.assertIn("unexpected", reason)

    def test_the_lod1_control_must_read_one(self):
        # a level 1 that was never cleared also gives a biased 0.0: without the control it would look like a drop
        for level1 in (0.0, 0.5, float("nan")):
            for plain, biased in ((0.0, 0.0), (0.0, 1.0)):
                with self.subTest(level1=level1, plain=plain, biased=biased):
                    answer, reason = vk_lod_bias.verdict(plain, biased, level1)
                    self.assertIsNone(answer)
                    self.assertIn("control read at LOD 1", reason)


class MeasureTest(unittest.TestCase):
    def test_a_library_without_the_calls_is_unknown(self):
        answer, reason = vk_lod_bias.measure(object(), 0x1)
        self.assertIsNone(answer)
        self.assertIn("lacks a call", reason)


class ShaderTest(unittest.TestCase):
    def test_the_words_are_a_spirv_module(self):
        words = vk_lod_bias.SHADER_WORDS
        self.assertEqual(words[0], 0x07230203)            # SPIR-V magic
        self.assertLessEqual(words[1], 0x00010000)        # SPIR-V 1.0, so any Vulkan 1.0 device takes it
        self.assertTrue(all(0 <= w <= 0xFFFFFFFF for w in words))

    @unittest.skipUnless(shutil.which("spirv-val"), "spirv-val not on PATH")
    def test_spirv_val_accepts_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lod_bias.spv")
            with open(path, "wb") as f:
                f.write(struct.pack(f"<{len(vk_lod_bias.SHADER_WORDS)}I", *vk_lod_bias.SHADER_WORDS))
            r = subprocess.run(["spirv-val", "--target-env", "vulkan1.0", path], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
