"""Unit tests for scripts/check_compute_journal_overlay.py using mocked subprocess invocations."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_compute_journal_overlay as cjo  # noqa: E402


class CheckComputeJournalOverlayTests(unittest.TestCase):
    @staticmethod
    def _make_fake_runner(*, ext_hit=1, mutate_local_b=False, skip_at=None, recorded=None):
        def fake_run(cmd, **kwargs):
            cache = Path(kwargs["cwd"])
            label = cache.name
            if recorded is not None:
                recorded.append((label, cmd, kwargs))
            if skip_at == label:
                return subprocess.CompletedProcess(cmd, 77, "", "")
            second = "--second-shader" in cmd
            expected = 14 if second else 10
            hit = ext_hit if label in ("external-A", "local-B") else 0
            pcache = cache / "pipeline_cache.d"
            if label == "cold":
                pcache.mkdir(exist_ok=True)
                (pcache / "cs-0000000000000001.bin").write_bytes(b"A" * 64)
            elif label == "new-B":
                pcache.mkdir(exist_ok=True)
                (pcache / "cs-0000000000000002.bin").write_bytes(b"B" * 64)
            elif label == "local-B" and mutate_local_b:
                (pcache / "cs-0000000000000002.bin").write_bytes(b"C" * 64)
            stdout = (
                "device: Test GPU\n"
                f"compute journal: direct Dispatch(4,1,1) copied both expected words (4,{expected})\n"
            )
            stderr = f"[rt] pipeline-compile trigger=test hit={hit} bytes=64 variant=0 result=0 \n"
            return subprocess.CompletedProcess(cmd, 0, stdout, stderr)

        return fake_run

    def test_four_stage_happy_path_env_scrubbing_and_encoding(self):
        recorded = []
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "d3d12_compute_journal_test.exe"
            exe.write_bytes(b"MZ")
            out, err = io.StringIO(), io.StringIO()
            with (
                mock.patch.dict(os.environ, {"LEONIDA_UNRELATED_VAR": "should_be_scrubbed"}),
                mock.patch.object(cjo.subprocess, "run", side_effect=self._make_fake_runner(recorded=recorded)),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                rc = cjo.main([str(exe), "--device", "1002:744c", "--require-hit"])
        self.assertEqual(rc, 0, err.getvalue())
        self.assertEqual(err.getvalue(), "")
        summary = json.loads(out.getvalue())
        self.assertEqual(summary["device"], "Test GPU")
        self.assertEqual([r[0] for r in recorded], ["cold", "external-A", "new-B", "local-B"])
        for label, _cmd, kwargs in recorded:
            self.assertEqual(kwargs.get("encoding"), "utf-8")
            self.assertEqual(kwargs.get("errors"), "replace")
            env = kwargs["env"]
            self.assertNotIn("LEONIDA_UNRELATED_VAR", env)
            self.assertEqual(env.get("LEONIDA_VK_DEVICE"), "1002:744c")
            if label == "cold":
                self.assertNotIn("LEONIDA_PIPELINE_JOURNAL_SEED_DIR", env)
            else:
                self.assertTrue(env.get("LEONIDA_PIPELINE_JOURNAL_SEED_DIR", "").endswith("pipeline_cache.d"))

    def test_require_hit_mutation_and_skip_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "d3d12_compute_journal_test"

            exe.write_bytes(b"ELF")  # POSIX executable name must also be accepted.
            out, err = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(cjo.subprocess, "run", side_effect=self._make_fake_runner(ext_hit=0)),
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                rc = cjo.main([str(exe), "--require-hit"])
            self.assertEqual(rc, 1)
            self.assertIn("ERROR:", err.getvalue())
            self.assertIn("no application-cache hit", err.getvalue())

            out2, err2 = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(cjo.subprocess, "run", side_effect=self._make_fake_runner(mutate_local_b=True)),
                contextlib.redirect_stdout(out2),
                contextlib.redirect_stderr(err2),
            ):
                rc2 = cjo.main([str(exe)])
            self.assertEqual(rc2, 1)
            self.assertIn("ERROR:", err2.getvalue())
            self.assertIn("changed disposable journal", err2.getvalue())

            out3, err3 = io.StringIO(), io.StringIO()
            with (
                mock.patch.object(cjo.subprocess, "run", side_effect=self._make_fake_runner(skip_at="cold")),
                contextlib.redirect_stdout(out3),
                contextlib.redirect_stderr(err3),
            ):
                rc3 = cjo.main([str(exe)])
            self.assertEqual(rc3, 77)
            self.assertIn("SKIP:", out3.getvalue())
            self.assertEqual(err3.getvalue(), "")

    def test_entries_and_main_reject_bad_paths_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            (cache / "pipeline_cache.d").write_bytes(b"not a dir")
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                cjo.entries(cache)

            # Missing executable, directory named as exe, and wrong binary name all return 2.
            dir_exe = root / "d3d12_compute_journal_test.exe"
            dir_exe.mkdir()
            wrong_exe = root / "other_tool.exe"
            wrong_exe.write_bytes(b"MZ")
            for target in (root / "missing.exe", dir_exe, wrong_exe):
                with self.subTest(target=target.name):
                    out, err = io.StringIO(), io.StringIO()
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        rc = cjo.main([str(target)])
                    self.assertEqual(rc, 2)
                    self.assertEqual(out.getvalue(), "")
                    self.assertIn("ERROR:", err.getvalue())
                    self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
