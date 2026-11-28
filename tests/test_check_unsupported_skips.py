"""check.py reports a test program's exit 77 ("unsupported: ...") as a skip, not a failure, in the three checks that
did not: the texture-constant tests (vk_texture_test), the partial-wave compute tests (vk_compute_test) and
ps_table_fetch, whose skip text was the program's whole output instead of the usual "SKIP (unsupported: ...)".

The outputs below are the ones an Intel Iris Xe with Windows driver 31.0.101.4255 printed, which reports
shaderInt64 = false: check.py counted six of these as failures while run_gpu_tests.py skipped the same tests."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import check

TEXTURE_UNSUPPORTED = ("unsupported: shaderInt64 (a stage declares Int64)\n"
                       "device: Intel(R) Iris(R) Xe Graphics\n"
                       "texel offset limits: -8..7, gather -32..31\n")
TEXTURE_SKIP = "SKIP (unsupported: shaderInt64 (a stage declares Int64))"
COMPUTE_REASON = "unsupported: cs_w64x_partial.spv declares Int64 and the device has no shaderInt64"
COMPUTE_UNSUPPORTED = f"device: Intel(R) Iris(R) Xe Graphics (subgroup 8..32)\n{COMPUTE_REASON}\n"


class TextureConstantUnsupported(unittest.TestCase):
    def run_constant(self, name, code, output):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / f"{name}.spv").touch()
            with patch.object(check, "run", return_value=(code, output)):
                return check.texture_constant_test(tmp, tmp, name)

    def test_exit_77_is_a_skip_that_quotes_the_reason(self):
        for name in check.TEXTURE_CONSTANTS:
            with self.subTest(name=name):
                problem = self.run_constant(name, check.UNSUPPORTED, TEXTURE_UNSUPPORTED)
                self.assertIsInstance(problem, check.Skip)
                self.assertEqual(problem, TEXTURE_SKIP)

    def test_exit_77_without_a_reason_is_still_a_skip(self):
        problem = self.run_constant("ps_gather_wrap", check.UNSUPPORTED, "")
        self.assertIsInstance(problem, check.Skip)
        self.assertEqual(problem, "SKIP (unsupported device)")

    def test_another_exit_code_is_still_a_failure(self):
        problem = self.run_constant("ps_gather_wrap", 1, "vkCreateGraphicsPipelines failed: -3\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertEqual(problem, "exit 1: vkCreateGraphicsPipelines failed: -3")


class PartialWaveUnsupported(unittest.TestCase):
    def run_partial(self, code, output, emulated=False):
        name, want = check.W64X_PARTIAL[0]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / f"{name}.spv").touch()
            with (patch.object(check, "run", return_value=(code, output)),
                  patch.object(check, "W64_EMULATED", emulated), patch.dict(os.environ, {})):
                return check.partial_wave_test(tmp, tmp, name, want)

    def test_exit_77_with_a_reason_is_a_skip(self):
        for emulated in (False, True):
            with self.subTest(emulated=emulated):
                problem = self.run_partial(check.UNSUPPORTED, COMPUTE_UNSUPPORTED, emulated)
                self.assertIsInstance(problem, check.Skip)
                self.assertEqual(problem, f"SKIP ({COMPUTE_REASON})")

    def test_exit_77_without_a_reason_stays_a_failure(self):
        # compute_unsupported's rule: only a 77 that names the missing feature is a skip.
        problem = self.run_partial(check.UNSUPPORTED, "device: gpu (subgroup 8..32)\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertEqual(problem, "exit 77: device: gpu (subgroup 8..32)")

    def test_another_exit_code_is_still_a_failure(self):
        problem = self.run_partial(1, f"device: gpu (subgroup 8..64)\n{COMPUTE_REASON}\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertTrue(problem.startswith("exit 1: "), problem)


class TableFetchUnsupported(unittest.TestCase):
    def run_table_fetch(self, code, output):
        with tempfile.TemporaryDirectory() as tmp, patch.object(check, "run", return_value=(code, output)):
            return check.table_fetch_test(tmp, tmp)

    def test_exit_77_is_a_one_line_skip_that_quotes_the_reason(self):
        problem = self.run_table_fetch(check.UNSUPPORTED, TEXTURE_UNSUPPORTED)
        self.assertIsInstance(problem, check.Skip)
        self.assertEqual(problem, TEXTURE_SKIP)

    def test_exit_77_without_a_reason_is_still_a_skip(self):
        problem = self.run_table_fetch(check.UNSUPPORTED, "")
        self.assertIsInstance(problem, check.Skip)
        self.assertEqual(problem, "SKIP (unsupported device)")

    def test_another_exit_code_is_still_a_failure(self):
        problem = self.run_table_fetch(1, "fetch-sint: 12 wrong pixels\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertEqual(problem, "exit 1: fetch-sint: 12 wrong pixels")


if __name__ == "__main__":
    unittest.main()
