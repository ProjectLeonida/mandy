"""Tests for atomic output from gen_gdk_runtime_classes."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "gen_gdk_runtime_classes.py"


def make_docs(root: Path) -> Path:
    """Create the smallest synthetic docs tree accepted by the generator."""
    docs = root / "docs"
    (docs / "COM").mkdir(parents=True)
    return docs


def command(docs: Path, output: Path) -> list[str]:
    return [sys.executable, "-I", str(SCRIPT), str(docs), "--out", str(output)]


def fail_after_bytes(limit: int):
    def set_file_limit() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)

    return set_file_limit


class GenGdkRuntimeClassesTests(unittest.TestCase):
    def test_failed_write_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = make_docs(root)
            output = root / "table.json"
            original = b"previous table must survive\n"
            output.write_bytes(original)

            # Validate that this input reaches the output path and exceeds the limit.
            successful = subprocess.run(command(docs, output), capture_output=True, check=False)
            self.assertEqual(successful.returncode, 0, successful.stderr.decode(errors="replace"))
            self.assertGreater(output.stat().st_size, 64)
            output.write_bytes(original)

            failed = subprocess.run(command(docs, output), capture_output=True, check=False,
                                    preexec_fn=fail_after_bytes(64))
            self.assertNotEqual(failed.returncode, 0, failed.stdout.decode(errors="replace"))
            self.assertEqual(output.read_bytes(), original)

    def test_failed_write_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = make_docs(root)
            output = root / "table.json"
            output.write_bytes(b"existing table")
            before = set(os.listdir(root))

            failed = subprocess.run(command(docs, output), capture_output=True, check=False,
                                    preexec_fn=fail_after_bytes(64))

            self.assertNotEqual(failed.returncode, 0, failed.stdout.decode(errors="replace"))
            self.assertEqual(set(os.listdir(root)), before)

    def test_successful_write_produces_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = make_docs(root)
            output = root / "table.json"

            result = subprocess.run(command(docs, output), capture_output=True, check=False)

            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            with output.open(encoding="utf-8") as stream:
                table = json.load(stream)
            self.assertEqual(table["classes"], {})
            self.assertEqual(table["interfaces"], {})


if __name__ == "__main__":
    unittest.main()
