"""runtime/test/shaders/fixture_expect.py: expected words declared in a compute fixture's own header.

    python -I tests/test_fixture_expect.py

No GPU: the header parser on synthetic files, the fixtures that use it, and the rule that keeps their words in one
place (a fixture with `; expect:` is not also listed in scripts/run_gpu_tests.py or runtime/test/shaders/check.py).
"""
import os
import re
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
sys.path.insert(0, SHADERS)

import fixture_expect as fe  # noqa: E402


class HeaderTest(unittest.TestCase):
    def fixture(self, text, name="cs_x.asm"):
        d = tempfile.mkdtemp(prefix="leonida expect ")
        self.addCleanup(lambda: [os.remove(os.path.join(d, f)) for f in os.listdir(d)] and os.rmdir(d))
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return d, path

    def test_hex_decimal_continuation_and_groups(self):
        _, path = self.fixture("; what it does\n; expect: 0x10 16, 0xffffffff\n; expect: -1\n; groups: 2x1x1\n"
                               "\nv_mov_b32 v0, 0\n; expect: 99 (after the header: ignored)\n")
        self.assertEqual(fe.header_expect(path), ([16, 16, 0xFFFFFFFF, 0xFFFFFFFF], "2x1x1"))

    def test_no_expect_line_opts_out(self):
        _, path = self.fixture("; Run: vk_compute_test cs_x.spv 1 dump -> out[0..1] = 1 2\nv_mov_b32 v0, 0\n")
        self.assertIsNone(fe.header_expect(path))

    def test_groups_default_to_one(self):
        _, path = self.fixture("; expect: 1 2\ns_endpgm\n")
        self.assertEqual(fe.header_expect(path), ([1, 2], "1"))

    def test_bad_words_are_errors_not_silent_passes(self):
        for header in ("; expect: 0x12 nope\n", "; expect:\n"):
            with self.subTest(header=header):
                _, path = self.fixture(header + "s_endpgm\n")
                with self.assertRaises(fe.FixtureHeaderError):
                    fe.header_expect(path)

    def test_only_compute_fixtures_are_collected(self):
        d, _ = self.fixture("; expect: 1\ns_endpgm\n", name="cs_a.asm")
        with open(os.path.join(d, "ps_a.asm"), "w", encoding="utf-8") as f:
            f.write("; expect: 1\ns_endpgm\n")
        self.assertEqual(fe.expect_fixtures(d), {"cs_a": ([1], "1")})


class RepositoryFixturesTest(unittest.TestCase):
    def test_the_migrated_fixtures_keep_their_words(self):
        found = fe.expect_fixtures(SHADERS)
        self.assertEqual(found["cs_bounds"], ([0x12345678, 0x12345678, 0, 0xEEEEEEEE], "1"))
        self.assertEqual(found["cs_smem_bounds"][0], [0, 0x5EED0010, 0x5EED000C, 0, 0x5EED000C, 0x5EED0010, 0, 0])
        self.assertEqual(found["cs_buf_fmt8"][0], [0x7F, 0x80, 0xFF, 0x01, 0, 0, 0xFF, 1])

    def test_a_header_fixture_has_no_row_in_either_runner(self):
        runners = {rel: open(os.path.join(ROOT, rel), encoding="utf-8").read()
                   for rel in ("scripts/run_gpu_tests.py", "runtime/test/shaders/check.py")}
        for name in fe.expect_fixtures(SHADERS):
            for rel, text in runners.items():
                with self.subTest(fixture=name, runner=rel):
                    self.assertIsNone(re.search(r"[\"']" + re.escape(name) + r"[\"']", text),
                                      f"{name} declares `; expect:` but {rel} still names it")


if __name__ == "__main__":
    unittest.main()
