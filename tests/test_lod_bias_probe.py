"""tools/lod_bias_probe.py: tools/vk_subgroup_info.py --lod-bias first, else three vk_sampler_test draws.

No GPU: vk_subgroup_info.py, build.py and vk_sampler_test are replaced by fakes. An answer from vk_subgroup_info.py
is used as it is, with no draw. Without one, the draw answers True only when LOD_BIAS +1 and -1 both read the
no-bias levels, False when both move as check.py expects, and None for a wrong no-bias draw, a mixed answer, a
failed draw or missing test programs.
Standard library only; no game files.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import check  # noqa: E402
import lod_bias_probe  # noqa: E402


def image(level_lz, level_implicit):
    return b"".join(struct.pack("<4f", *check.sampler_texel(level_lz, x, y), *check.sampler_texel(level_implicit, x, y))
                    for y in range(check.SAMPLER_SIZE) for x in range(check.SAMPLER_SIZE))


class FakeRuns:
    """subprocess.run for the probe: build.py writes the two modules, vk_sampler_test the levels `levels(bias)`."""

    def __init__(self, levels, draw_rc=0, info=None):
        self.levels, self.draw_rc, self.info, self.draws = levels, draw_rc, info, 0

    def __call__(self, cmd, **kw):
        if cmd[-1] == "--lod-bias":   # vk_subgroup_info.py --lod-bias
            return subprocess.CompletedProcess(cmd, 0, json.dumps(self.info or {"explicit_lod_bias": None}), "")
        self.draws += 1
        if cmd[-2:] == ["vs_ids", "ps_ssharp"]:
            spirv = Path(cmd[-3]) / "spirv"
            spirv.mkdir(parents=True, exist_ok=True)
            for n in ("vs_ids", "ps_ssharp"):
                (spirv / (n + ".spv")).write_bytes(b"\x03\x02\x23\x07")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        words = [int(w, 16) for w in cmd[4:8]]   # vk_sampler_test vs ps raw <4 S# words> <base> <last>
        bias = (((words[2] & 0x3fff) ^ 0x2000) - 0x2000) / 256
        if not self.draw_rc:
            Path(cmd[3]).write_bytes(image(*self.levels(bias)))
        return subprocess.CompletedProcess(cmd, self.draw_rc, "", "device lost" if self.draw_rc else "")


def applied(bias):      # check.py's expectation: LOD 0 / LOD 1 plus the bias, on two levels
    return (min(max(round(0 + bias), 0), 1), min(max(round(1 + bias), 0), 1))


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.build = tempfile.TemporaryDirectory()
        self.addCleanup(self.build.cleanup)
        Path(self.build.name, "vk_sampler_test" + lod_bias_probe.EXE).write_bytes(b"")

    def run_probe(self, fake):
        with mock.patch.object(lod_bias_probe.subprocess, "run", fake):
            return lod_bias_probe.probe(self.build.name)

    def run_draw(self, fake):
        with mock.patch.object(lod_bias_probe.subprocess, "run", fake):
            return lod_bias_probe.draw_probe(self.build.name)

    def test_a_driver_that_applies_the_bias(self):
        self.assertIs(self.run_probe(FakeRuns(applied))[0], False)

    def test_a_driver_that_drops_the_bias(self):
        answer, reason = self.run_probe(FakeRuns(lambda bias: (0, 1)))
        self.assertIs(answer, True)
        self.assertIn("drops", reason)

    def test_a_mixed_answer_is_unknown(self):
        # +1 applied, -1 dropped: neither rule fits, so the flag must not be chosen from it.
        self.assertIsNone(self.run_probe(FakeRuns(lambda bias: applied(bias) if bias > 0 else (0, 1)))[0])

    def test_a_wrong_no_bias_draw_is_unknown(self):
        self.assertIsNone(self.run_probe(FakeRuns(lambda bias: (1, 0)))[0])

    def test_a_failed_draw_is_unknown(self):
        self.assertIsNone(self.run_probe(FakeRuns(applied, draw_rc=1))[0])

    def test_no_test_programs_is_unknown(self):
        with tempfile.TemporaryDirectory() as empty:
            answer, reason = lod_bias_probe.draw_probe(empty)
        self.assertIsNone(answer)
        self.assertIn("vk_sampler_test", reason)

    def test_a_measured_answer_needs_no_draw(self):
        for answer in (True, False):
            with self.subTest(answer=answer):
                fake = FakeRuns(lambda bias: (1, 0), info={"explicit_lod_bias": answer, "explicit_lod_bias_reason": "m"})
                self.assertEqual(self.run_probe(fake), (answer, "m"))
                self.assertEqual(fake.draws, 0)

    def test_an_error_from_vk_subgroup_info_falls_back_to_the_draw(self):
        fake = FakeRuns(lambda bias: (0, 1), info={"error": "RuntimeError: no Vulkan device"})
        self.assertIs(self.run_probe(fake)[0], True)
        self.assertGreater(fake.draws, 0)

    def test_without_test_programs_only_the_measurement_answers(self):
        fake = FakeRuns(applied)
        with mock.patch.object(lod_bias_probe.subprocess, "run", fake):
            answer, _ = lod_bias_probe.probe(None)
        self.assertIsNone(answer)
        self.assertEqual(fake.draws, 0)

    def test_the_probe_translates_without_the_flag(self):
        seen = {}
        fake = FakeRuns(applied)

        def spy(cmd, **kw):
            if cmd[-2:] == ["vs_ids", "ps_ssharp"]:
                seen["env"] = kw.get("env", {}).get("LEONIDA_EXPLICIT_LOD_BIAS")
            return fake(cmd, **kw)
        self.run_probe(spy)
        self.assertEqual(seen["env"], "0")


if __name__ == "__main__":
    unittest.main()
