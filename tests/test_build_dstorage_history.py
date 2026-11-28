"""Subprocess coverage for atomic DirectStorage history JSON writes."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_dstorage_history.py"
TOOLS = ROOT / "tools"
TESTS = ROOT / "tests"
sys.path[:0] = [str(TOOLS), str(TESTS)]
from test_bcpack_history_calib import _encode_bc7_mode6_tiled, _pack_etpf, _write_custom_rds5_pool


_BUILD_WITH_LIMIT = r"""
import pathlib
import resource
import signal
import sys
sys.path.insert(0, sys.argv[1])
import build_dstorage_history as history

class FixedTemporaryDirectory:
    def __init__(self, *args, **kwargs):
        self.path = pathlib.Path(sys.argv[5])
    def __enter__(self):
        return str(self.path)
    def __exit__(self, *args):
        return False

history.tempfile.TemporaryDirectory = FixedTemporaryDirectory
original_refine = history.refine
def constrain_after_refine(docs, votes, history_path):
    original_refine(docs, votes, history_path)
    pathlib.Path(history_path).write_bytes(b"012345678901234567890123456789")
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (128, 128))
history.refine = constrain_after_refine
history.build(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4]))
"""


class BuildDstorageHistoryAtomicTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.history_dir = self.work / "history-temp"
        self.history_dir.mkdir()
        self.pool = self.work / "synthetic.ttpf"
        record = _pack_etpf(64, 64, 1, 1, 98, 4096)
        blocks = (bytes([0x40]) + bytes(15)) * 256
        _write_custom_rds5_pool(
            self.pool,
            [record],
            [(7, 4096, _encode_bc7_mode6_tiled(blocks))],
        )
        self.trainer = self.work / "synthetic-trainer"
        self.trainer.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "sys.stdin.buffer.read()\n"
            "sys.stdout.buffer.write(bytes(2 * 256 * 35 * 16 * 8))\n",
            encoding="utf-8",
        )
        self.trainer.chmod(0o755)

    def tearDown(self):
        self.temp.cleanup()

    def _run_failed_build(self):
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                _BUILD_WITH_LIMIT,
                str(TOOLS),
                str(self.pool),
                str(self.work / "profile.bcph"),
                str(self.trainer),
                str(self.history_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

    def test_failed_write_preserves_existing_report(self):
        report = self.history_dir / "history.json"
        original = b"012345678901234567890123456789"
        self.assertEqual(len(original), 30)

        result = self._run_failed_build()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(report.read_bytes(), original)

    def test_failed_write_removes_temporary_file(self):
        report = self.history_dir / "history.json"

        result = self._run_failed_build()

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("File too large", result.stderr)
        self.assertEqual(sorted(p.name for p in self.history_dir.glob("*.tmp")), [])

    def test_successful_build_writes_valid_json_report(self):
        sidecar = self.work / "profile.bcph"
        report = self.work / "history.json"
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOL),
                str(self.pool),
                "--out",
                str(sidecar),
                "--trainer",
                str(self.trainer),
                "--json-out",
                str(report),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(sidecar.is_file())
        self.assertEqual(sidecar.stat().st_size, 60936)
        docs = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(set(docs), {"bc6h", "bc7"})


if __name__ == "__main__":
    unittest.main()
