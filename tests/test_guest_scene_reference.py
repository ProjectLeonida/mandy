"""Fixed reference points for the synthetic guest's renderer and input trace (no GPU)."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


path = Path(__file__).resolve().parents[1] / "scripts" / "run_guest_scene.py"
spec = importlib.util.spec_from_file_location("run_guest_scene", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GuestSceneReferenceTest(unittest.TestCase):
    def test_trace_and_pixel_digest_at_scene_boundaries(self):
        # These records were read back independently on the RTX 5090 and Apple M3 Max.
        anchors = [
            (0, 512, 9, 8, 18, 1, 376597285, 314582078),
            (137, 132, 44, 32, 153, 3, 3593765925, 2667449936),
            (299, 320, 8, 58, 169, 2, 1840220197, 1732923390),
            (300, 64, 8, 57, 177, 2, 1840220197, 1732923390),
            (422, 4, 58, 8, 228, 1, 1662446373, 2331208379),
            (599, 576, 58, 8, 30, 1, 1662446373, 2331208379),
        ]
        steps = runner.trace()
        for frame, buttons, x, y, scene_byte, bias, low, high in anchors:
            with self.subTest(frame=frame):
                self.assertEqual(steps[frame], (buttons, x, y, scene_byte, bias))
                digest, _ = runner.predicted_image(x % 8, bias)
                self.assertEqual(digest, (high << 32) | low)

    def test_pcm_envelope_and_period(self):
        samples = struct.unpack("<11025h", runner.expected_pcm(9000))
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[500:550], (9000,) * 50)
        self.assertEqual(samples[550:600], (-9000,) * 50)
        self.assertEqual(samples[-1], 0)


    def test_redirected_cli_preserves_unicode_paths_and_guest_diagnostic(self):
        # Force strict redirected text streams even on UTF-8 hosts. Invoke the real CLI with a
        # small stand-in loader that produces a diagnostic and exits before rendering.
        entry = (
            "import runpy, sys; "
            "encoding, script = sys.argv[1:3]; del sys.argv[1:3]; "
            "sys.stdout.reconfigure(encoding=encoding, errors='strict'); "
            "sys.stderr.reconfigure(encoding=encoding, errors='strict'); "
            "runpy.run_path(script, run_name='__main__')"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "spirv").mkdir()
            loader = root / "loader.py"
            diagnostic = "guest diagnostic: 測試"
            loader.write_text(f"import os\nos.write(2, {diagnostic.encode()!r})\nraise SystemExit(7)\n",
                              encoding="ascii")
            for encoding in ("cp1252", "utf-8"):
                with self.subTest(encoding=encoding):
                    out = root / f"測試 café {encoding}"
                    result = subprocess.run(
                        [sys.executable, "-I", "-c", entry, encoding, str(path), "--wine", sys.executable,
                         "--loader", str(loader), "--guest", str(loader), "--shaders", str(root), "--out", str(out)],
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    report = out / "report.json"
                    summary = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual(summary["error"], "full: exit code 7")
                    self.assertEqual(summary["runs"]["full"]["outcome"], "exited")
                    self.assertEqual(summary["runs"]["full"]["returncode"], 7)
                    # The runner resolves --out before it writes (run_guest_scene.py:355), so the report path it
                    # prints is the physical one: on Windows the 8.3 name (RUNNER~1) is expanded, and on macOS the
                    # unresolved /var prefix only matched as a suffix of /private/var.
                    self.assertIn(str(report.resolve()).encode(encoding, "backslashreplace"), result.stdout)
                    self.assertIn(diagnostic.encode(encoding, "backslashreplace"), result.stderr)
                    self.assertNotIn(b"UnicodeEncodeError", result.stderr)

    def test_uncreatable_output_reports_clean_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "spirv").mkdir()
            placeholder = root / "placeholder.txt"
            placeholder.write_bytes(b"synthetic input")
            out = placeholder / "output"
            result = subprocess.run(
                [sys.executable, "-I", str(path), "--wine", sys.executable,
                 "--loader", str(placeholder), "--guest", str(placeholder),
                 "--shaders", str(root), "--out", str(out)],
                capture_output=True, encoding="utf-8", errors="replace", timeout=20,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(f"could not create output directory {out.resolve()}", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(placeholder.read_bytes(), b"synthetic input")

    def test_launch_failure_preserves_run_record_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "spirv").mkdir()
            bad_loader = root / "invalid.exe"
            bad_loader.write_bytes(b"synthetic input that is not an executable")
            bad_loader.chmod(0o700)
            out = root / "output"
            cmd = [sys.executable, "-I", str(path), "--loader", str(bad_loader), "--guest", str(bad_loader),
                   "--shaders", str(root), "--out", str(out)]
            loader_index = 0
            if sys.platform != "win32":
                # A wine stub of its own: Session.run puts --wine (and its args) first, so with the loader behind
                # it cmd[0] is the wine path and the loader entry would never be the one compared.
                wine = root / "invalid-wine"
                wine.write_bytes(b"synthetic input that is not an executable")
                wine.chmod(0o700)
                wine_args = ["-a", "path with spaces/wine"]
                cmd += ["--wine", str(wine), *[f"--wine-arg={arg}" for arg in wine_args]]
                loader_index = 1 + len(wine_args)
            # The runner's "could not launch" line carries the OS error text, which Windows localizes and the child
            # prints in the ANSI code page (cp1251 on a Russian host): a strict UTF-8 decode would leave stdout None.
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=20)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertIn("could not launch", report["error"])
            record = json.loads((out / "logs" / "full.json").read_text(encoding="utf-8"))
            self.assertEqual(record, report["runs"]["full"])
            self.assertEqual(record["outcome"], "launch_failed")
            self.assertIsNone(record["returncode"])
            # The runner resolves its loader/guest/shaders paths (main: args.loader.resolve()), and on macOS the
            # temp directory is reached through /var, which is a symlink to /private/var: compare the resolved path.
            self.assertEqual(record["cmd"][loader_index], str(bad_loader.resolve()))
            if sys.platform != "win32":
                self.assertEqual(record["cmd"][1:3], ["-a", "path with spaces/wine"])
            self.assertTrue(record["launch_error"])
            for stream in ("stdout", "stderr"):
                contents = (out / "logs" / f"full.{stream}.log").read_bytes()
                self.assertEqual(record[f"{stream}_sha256"], runner.sha256(contents))

    def test_launch_failure_records_the_resolved_launcher(self):
        """The run record keeps the path that ran, not the spelling the caller typed.

        A symlinked root pins the contract without depending on the host's temporary volume (macOS gets
        /var -> /private/var for free; Windows needs a junction, which an ordinary account cannot create).
        """
        with tempfile.TemporaryDirectory() as temp:
            real = Path(temp) / "real"
            real.mkdir()
            link_root = Path(temp) / "link"
            try:
                link_root.symlink_to(real, target_is_directory=True)
            except (OSError, NotImplementedError) as err:
                self.skipTest(f"symbolic links unavailable: {err}")
            (link_root / "spirv").mkdir()
            real_loader = real / "invalid.exe"
            real_loader.write_bytes(b"synthetic input that is not an executable")
            real_loader.chmod(0o700)
            link_loader = link_root / "invalid.exe"
            self.assertNotEqual(str(link_loader), str(real_loader))
            out = link_root / "output"
            cmd = [sys.executable, "-I", str(path), "--loader", str(link_loader), "--guest", str(link_loader),
                   "--shaders", str(link_root), "--out", str(out)]
            loader_index = 0
            if sys.platform != "win32":
                # Distinct from the loader, so cmd[0] (the wine path) cannot stand in for the loader entry.
                real_wine = real / "invalid-wine"
                real_wine.write_bytes(b"synthetic input that is not an executable")
                real_wine.chmod(0o700)
                cmd += ["--wine", str(link_root / "invalid-wine")]
                loader_index = 1
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=20)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            record = json.loads((out / "logs" / "full.json").read_text(encoding="utf-8"))
            self.assertEqual(record["outcome"], "launch_failed")
            self.assertEqual(record["cmd"][loader_index], str(real_loader.resolve()))

    def test_timeout_has_its_own_run_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(wine=None, loader=root / "loader.exe", guest=root / "guest.exe",
                                      shaders=root, play_audio=False)
            session = runner.Session(args, root)
            session.logs.mkdir()
            with mock.patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("loader", 10)), \
                    self.assertRaisesRegex(runner.Fail, "timed out"):
                session.run("full", "full", root, root)
            self.assertEqual(session.runs["full"]["outcome"], "timed_out")
            self.assertIsNone(session.runs["full"]["returncode"])
            self.assertNotIn("launch_error", session.runs["full"])


if __name__ == "__main__":
    unittest.main()
