"""Unit tests for scripts/merge_check.py ref validation, bounds, and git timeout handling.

    python -I tests/test_merge_check_safety.py
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import merge_check as mc  # noqa: E402


class MergeCheckSafetyTests(unittest.TestCase):
    def test_parse_pair_and_cli_reject_invalid_bounds_and_option_injection(self) -> None:
        self.assertEqual(mc.parse_pair("#12=main"), (12, "main"))
        self.assertEqual(mc.parse_pair("12=refs/pull/12/head"), (12, "refs/pull/12/head"))

        bad_argvs = (
            ["--depth", "0", "12=main"],
            ["--depth", "-1", "12=main"],
            ["--min-length", "-1", "12=main"],
            ["0=main"],
            ["##12=main"],
            ["12=--all"],
            ["12=   "],
            ["--main=--exec=bad", "12=main"],
            ["--main", "", "12=main"],
        )
        for argv in bad_argvs:
            with self.subTest(argv=argv):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    mc.main(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_git_timeout_raises_git_error_and_main_returns_2(self) -> None:
        with mock.patch.object(
            mc.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["git"], mc.GIT_TIMEOUT),
        ):
            with self.assertRaises(mc.GitError) as cm:
                mc.git(".", "status")
            self.assertIn("timed out", str(cm.exception))

            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = mc.main(["--repo", ".", "12=main"])
            self.assertEqual(rc, 2)
            self.assertIn("merge_check:", err.getvalue())
            self.assertIn("timed out", err.getvalue())


if __name__ == "__main__":
    unittest.main()
