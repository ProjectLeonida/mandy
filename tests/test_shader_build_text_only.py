"""runtime/test/shaders/build.py takes a text-only source's lines as its listing (TEST-BUILD-TEXT-ONLY).

cs_dpp_cmp.asm holds a VOPC compare in DPP form, which llvm-mc has no gfx1013 encoding for. The file says so in a
comment line of its own, and scripts/run_gpu_tests.py passes such a source's instruction lines to the translator
as they are. build.py still sent it to llvm-mc: with no names it stopped at "cs_dpp_cmp.asm did not assemble"
before publishing anything, so check.py reported every test missing and w64_graphics.py found no module.

Checked here: the marker is the same line in both runners, build.py writes the listing run_gpu_tests.py writes,
a marked source never reaches llvm-mc, and an unmarked one still does.

Standard library only; llvm-mc and the translator are replaced by stand-ins. No SPIR-V tools, GPU or game data.

    python -I tests/test_shader_build_text_only.py
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADERS = os.path.join(ROOT, "runtime", "test", "shaders")
BUILD = os.path.join(SHADERS, "build.py")
RUNNER = os.path.join(ROOT, "scripts", "run_gpu_tests.py")


def load_build():
    spec = importlib.util.spec_from_file_location("shaders_build_text_only_under_test", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def marked_fixtures(mark):
    """The .asm fixtures that carry the marker as a line of its own."""
    out = []
    for path in sorted(Path(SHADERS).glob("*.asm")):
        if any(ln.strip() == mark for ln in path.read_text(encoding="utf-8").splitlines()):
            out.append(path.stem)
    return out


class TextOnlySource(unittest.TestCase):
    def setUp(self):
        self.build = load_build()

    def staged(self, names, assembler):
        """translate_tests on `names` with `assembler` in place of llvm-mc and a translator that does nothing.
        Returns the listing text of each name."""
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"LEONIDA_BARY_FALLBACK": "0"}, clear=False), \
                mock.patch.object(self.build.mc_command, "run", assembler), \
                mock.patch.object(self.build.subprocess, "run",
                                  return_value=SimpleNamespace(returncode=0, stdout="", stderr="")):
            self.assertEqual(self.build.translate_tests(Path(tmp), names), 0)
            for n in names:
                self.assertTrue(Path(tmp, "fssm", n + ".fssm").is_file(), n)   # the compute header is still staged
            return {n: Path(tmp, "disasm", n + ".s").read_text(encoding="utf-8") for n in names}

    def test_both_runners_use_the_same_marker_line(self):
        runner = Path(RUNNER).read_text(encoding="utf-8")
        self.assertIn(f'\nTEXT_ONLY_MARK = "{self.build.TEXT_ONLY_MARK}"\n', runner)

    def test_a_fixture_carries_the_marker(self):
        # The case this is for; if the fixture ever assembles and drops the line, the tests below have no subject.
        self.assertIn("cs_dpp_cmp", marked_fixtures(self.build.TEXT_ONLY_MARK))

    def test_listing_is_the_instruction_lines(self):
        raw = (f"; what the test computes\n{self.build.TEXT_ONLY_MARK}\n; group: 32,1,1\n\n"
               "v_mov_b32 v3, 0\n   v_cmp_gt_f32_dpp vcc_lo, v1, v1 dpp8:[4,5,6,7,0,1,2,3]  \ns_endpgm\n")
        self.assertEqual(self.build.text_only_listing("cs_sample", raw),
                         "; cs_sample  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words\n"
                         "\tv_mov_b32 v3, 0\n\tv_cmp_gt_f32_dpp vcc_lo, v1, v1 dpp8:[4,5,6,7,0,1,2,3]\n\ts_endpgm\n")

    def test_only_a_line_of_its_own_marks_a_source(self):
        mark = self.build.TEXT_ONLY_MARK
        for raw in ("v_mov_b32 v3, 0\ns_endpgm\n",                          # no marker
                    f"; see the note: {mark[2:]}\nv_mov_b32 v3, 0\n",       # quoted inside another comment
                    f"{mark} (not yet)\nv_mov_b32 v3, 0\n"):                # the line with something after it
            with self.subTest(raw=raw.splitlines()[0]):
                self.assertIsNone(self.build.text_only_listing("cs_sample", raw))

    def test_a_marked_fixture_never_reaches_llvm_mc(self):
        def refuse(cmd, **kwargs):
            raise AssertionError("llvm-mc was run for a text-only source")
        for name in marked_fixtures(self.build.TEXT_ONLY_MARK):
            with self.subTest(name=name):
                listing = self.staged([name], refuse)[name]
                raw = Path(SHADERS, name + ".asm").read_text(encoding="utf-8")
                code = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith(";")]
                self.assertEqual(listing.splitlines()[1:], ["\t" + ln for ln in code])
                self.assertIn("_dpp ", listing)        # cs_dpp_cmp: the compares llvm-mc refuses are in the listing

    def test_listing_matches_run_gpu_tests(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_gpu_tests   # noqa: E402  (its own copy of build.py and check.py come with it)
        for name in marked_fixtures(self.build.TEXT_ONLY_MARK):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                asm = os.path.join(SHADERS, name + ".asm")
                run_gpu_tests.assemble_to_s("no-llvm-mc-is-needed-for-this", asm, os.path.join(tmp, name + ".s"), name)
                theirs = Path(tmp, name + ".s").read_text(encoding="utf-8")
                ours = self.build.text_only_listing(name, Path(asm).read_text(encoding="utf-8"))
                self.assertEqual(ours, theirs)

    def test_an_unmarked_fixture_is_still_assembled(self):
        calls = []

        def assembler(cmd, **kwargs):
            calls.append((cmd, kwargs.get("input")))
            return SimpleNamespace(returncode=0, stderr="",
                                   stdout="\t.text\n\tv_mov_b32_e32 v0, 0 ; encoding: [0x80,0x02,0x00,0x7e]\n")
        listing = self.staged(["cs_store"], assembler)["cs_store"]
        self.assertEqual(len(calls), 1)
        self.assertIn("-mcpu=gfx1013", calls[0][0])
        self.assertEqual(calls[0][1], Path(SHADERS, "cs_store.asm").read_text(encoding="utf-8"))
        self.assertEqual(listing, "; cs_store  hand-written test\n"
                                  "\tv_mov_b32_e32 v0, 0 ; encoding: [0x80,0x02,0x00,0x7e]\n")

    def test_an_assembler_error_still_stops_the_build(self):
        def assembler(cmd, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="<stdin>:1:1: error: invalid instruction\n")
        with self.assertRaises(SystemExit) as stop:
            self.staged(["cs_store"], assembler)
        self.assertIn("cs_store.asm did not assemble", str(stop.exception))


if __name__ == "__main__":
    unittest.main()
