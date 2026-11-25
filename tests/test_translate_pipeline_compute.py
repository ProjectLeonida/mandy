"""Two synthetic compute records exercise parallel translation, cache reuse and interrupted publication."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix", "translate_pipeline_compute uses POSIX fcntl/resource/fork")
class ComputePipelineUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts_dir = str(ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import translate_pipeline_compute as tpc  # noqa: E402
        cls.tpc = tpc

    def test_load_compute_names_handles_bom_and_rejects_malformed_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = Path(tmp) / "index.json"
            idx.write_text(
                json.dumps([
                    {"file": "ps_skip.dxbc", "stage": "ps"},
                    {"file": "cs_b.dxbc", "stage": "cs"},
                    {"file": "cs_a.dxbc", "stage": "cs"},
                ]),
                encoding="utf-8-sig",
                newline="\n",
            )
            self.assertEqual(self.tpc.load_compute_names(idx), ["cs_a", "cs_b"])

            # Missing file
            with self.assertRaises(ValueError):
                self.tpc.load_compute_names(Path(tmp) / "missing.json")

            bad_payloads = [
                "not json",
                "{}",
                json.dumps([123]),
                json.dumps([{"stage": 1, "file": "cs_a.dxbc"}]),
                json.dumps([{"stage": "cs", "file": ".dxbc"}]),
                json.dumps([{"stage": "cs", "file": "sub/cs_a.dxbc"}]),
                json.dumps([{"stage": "cs", "file": "sub\\cs_a.dxbc"}]),
                json.dumps([{"stage": "cs", "file": "cs_a.bin"}]),
                json.dumps([{"stage": "cs", "file": None}]),
                json.dumps([{"stage": "cs", "file": "cs_a.dxbc"}, {"stage": "cs", "file": "cs_a.dxbc"}]),
                json.dumps([{"stage": "ps", "file": "ps_only.dxbc"}]),
            ]
            for payload in bad_payloads:
                with self.subTest(payload=payload):
                    idx.write_text(payload, encoding="utf-8", newline="\n")
                    with self.assertRaises(ValueError):
                        self.tpc.load_compute_names(idx)

    def test_build_one_creates_cache_and_sets_published_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "in", root / "out"
            (src / "disasm").mkdir(parents=True)
            (src / "fspx").mkdir()
            out.mkdir()
            name = "cs_mode_test"
            code = struct.pack("<II", 0xBF800000, 0xBF810000)
            state = struct.pack("<7I", 0, 0, 0, 1, 1, 0, 0)
            (src / "fspx" / (name + ".code")).write_bytes(code)
            (src / "fspx" / (name + ".state")).write_bytes(state)
            (src / "fspx" / (name + ".json")).write_text(
                json.dumps({"stage": "cs", "threads": [1, 1, 1], "rsrc2": 0, "wave32": True}),
                encoding="utf-8",
                newline="\n",
            )
            listing = f"; {name}  code 0x0..0x8 (fspx), gfx1013, 0 invalid words\n"
            for asm_line, word in (("s_nop 0", code[:4]), ("s_endpgm", code[4:])):
                listing += f"\t{asm_line} ; encoding: [" + ",".join(hex(b) for b in word) + "]\n"
            (src / "disasm" / (name + ".s")).write_text(listing, encoding="utf-8", newline="\n")

            fake_spv = b"\x03\x02\x23\x07" + bytes(16)

            def fake_assemble(_sa, sv, asm_text="", compact=False):
                Path(sv).write_bytes(fake_spv)
                os.chmod(sv, 0o600)
                return mock.Mock(returncode=0, stderr="")

            with mock.patch.object(self.tpc.translator, "assemble_spirv", side_effect=fake_assemble), \
                    mock.patch.object(self.tpc, "published_mode", return_value=0o644):
                row = self.tpc.build_one(src, out, name, ["settings"])

            target = out / f"{name}_{row['module']}.spv"
            self.assertTrue(target.is_file())
            self.assertEqual(target.stat().st_mode & 0o777, 0o644)

    def test_build_parent_exception_not_masked_by_failing_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "in", root / "out" / "nested"
            src.mkdir()
            with mock.patch.object(self.tpc.os, "fork", return_value=424242), \
                    mock.patch.object(self.tpc.os, "kill") as kill_mock, \
                    mock.patch.object(self.tpc.os, "waitpid", return_value=(424242, 256)) as wait_mock, \
                    mock.patch.object(self.tpc, "build_part", side_effect=RuntimeError("parent boom")):
                with self.assertRaisesRegex(RuntimeError, "parent boom"):
                    self.tpc.build(src, out, ["cs_a", "cs_b"], ["s"], jobs=2)
            kill_mock.assert_called_once_with(424242, signal.SIGTERM)
            wait_mock.assert_called_once_with(424242, 0)

            # When parent succeeds and child exits non-zero, raise RuntimeError.
            with mock.patch.object(self.tpc.os, "fork", return_value=424242), \
                    mock.patch.object(self.tpc.os, "waitpid", return_value=(424242, 256)), \
                    mock.patch.object(self.tpc, "build_part", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "second translation worker failed"):
                    self.tpc.build(src, out, ["cs_a", "cs_b"], ["s"], jobs=2)

    def test_main_validates_index_before_rlimit_and_accepts_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "in", root / "out"
            src.mkdir()
            (src / "index.json").write_text('[{"stage": "cs", "file": ".dxbc"}]', encoding="utf-8", newline="\n")
            fake_bin = root / "fake_tool"
            fake_bin.write_bytes(b"bin")
            err = io.StringIO()
            with mock.patch.object(self.tpc.shutil, "which", return_value=str(fake_bin)), \
                    mock.patch.object(self.tpc.os, "setpriority", side_effect=AssertionError("too early")), \
                    mock.patch.object(self.tpc.resource, "setrlimit", side_effect=AssertionError("too early")), \
                    contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as cm:
                    self.tpc.main([str(src), "--out", str(out)])
            self.assertEqual(cm.exception.code, 2)
            self.assertIn("invalid compute filename", err.getvalue())


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("spirv-val") and shutil.which("spirv-as"),
                     "requires Linux and SPIRV-Tools")
class ComputeBatch(unittest.TestCase):
    def test_resume_and_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, out = root / "input", root / "output"
            (src / "disasm").mkdir(parents=True)
            (src / "fspx").mkdir()
            rows = []
            for i in (0, 1):
                name = f"cs_test_{i}"
                rows.append({"file": name + ".dxbc", "stage": "cs"})
                code = struct.pack("<II", 0xbf800000 | i, 0xbf810000)
                (src / "fspx" / (name + ".code")).write_bytes(code)
                (src / "fspx" / (name + ".state")).write_bytes(struct.pack("<7I", 0, 0, 0, 1, 1, 0, 0))
                (src / "fspx" / (name + ".json")).write_text(
                    json.dumps({"stage": "cs", "threads": [1, 1, 1], "rsrc2": 0, "wave32": True}),
                    encoding="utf-8", newline="\n")
                (src / (name + ".dxbc")).write_bytes(b"DXBC" + bytes(28))
                listing = f"; {name}  code 0x0..0x8 (fspx), gfx1013, 0 invalid words\n"
                for asm, word in ((f"s_nop {i}", code[:4]), ("s_endpgm", code[4:])):
                    listing += f"\t{asm} ; encoding: [" + ",".join(hex(b) for b in word) + "]\n"
                (src / "disasm" / (name + ".s")).write_text(listing, encoding="utf-8", newline="\n")
            (src / "index.json").write_text(json.dumps(rows), encoding="utf-8", newline="\n")
            cmd = [sys.executable, "-I", str(ROOT / "scripts/translate_pipeline_compute.py"),
                   str(src), "--out", str(out), "--wave64-emulate"]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index), 2)
            files = sorted(out.glob("*.spv"))
            self.assertEqual(len(files), 2)
            for p in files:
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), p.stem.rsplit("_", 1)[1])
                os.utime(p, ns=(1000000000, 1000000000))
            (out / "index.json").unlink()  # interruption after checkpointing both shaders
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(index, json.loads((out / "index.json").read_text(encoding="utf-8")))
            self.assertTrue(all(p.stat().st_mtime_ns == 1000000000 for p in files))
            files[0].write_bytes(b"damaged")
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(hashlib.sha256(files[0].read_bytes()).hexdigest(), files[0].stem.rsplit("_", 1)[1])
            self.assertEqual(files[1].stat().st_mtime_ns, 1000000000)
            # A newly rejected input must leave the current index, even when the older file remains readable.
            (src / "disasm/cs_test_0.s").write_text("; synthetic empty listing\n", encoding="utf-8", newline="\n")
            partial = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(partial.returncode, 1, partial.stderr)
            self.assertEqual(len(json.loads((out / "index.json").read_text(encoding="utf-8"))), 1)
            failures = json.loads((out / "failures.json").read_text(encoding="utf-8"))
            rejected = next(m for n, m in index.items() if n.startswith("cs_test_0_"))
            self.assertIn("no instructions", failures[rejected["fspx_code_sha256"]][rejected["fspx_state_sha256"]])


if __name__ == "__main__":
    unittest.main()
