"""Unit tests for tools/stage_timeline.py on synthetic runtime logs (no game data, no runtime, no GPU).

    python -I tests/test_stage_timeline.py

The logs are written here, line by line, in the format the tool's docstring gives for the runtime's own
`[rt] peek` and `[rt] gameinput` lines; nothing is read from a real boot.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "tools", "stage_timeline.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import stage_timeline as st  # noqa: E402

# A boot from no title state to the end of the loading screen, with three scripted presses.
BOOT = [
    "[rt] peek t=1s: 0x7446510=? 0x98f828c=?",
    "[rt] peek t=2s: 0x7446510=2 0x98f828c=1",
    "[rt] peek t=3s: 0x7446510=3 0x98f828c=1",
    "[rt] peek t=5s: 0x7446510=3 0x98f828c=3",
    "[rt] gameinput: scripted press of button 4 at 5.5 s",
    "[rt] peek t=6s: 0x7446510=3 0x98f828c=d",
    "[rt] peek t=7s: 0x7446510=3 0x98f828c=d",
    "[rt] gameinput: scripted press of button 4 at 8.0 s",
    "[rt] peek t=9s: 0x7446510=3 0x98f828c=0 0x7453ed0=2",
    "[rt] peek t=12s: 0x7446510=3 0x98f828c=0 0x7453ed0=100",
    "[rt] gameinput: scripted press of button 1 at 20.0 s",
]


class _Logs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write_log(self, lines, name="boot.log"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", cwd=self.tmp)


class ReadTest(_Logs):
    def test_reads_peeks_and_presses(self):
        peeks, presses = st.read(self.write_log(BOOT))
        self.assertEqual(len(peeks), 8)
        self.assertEqual(peeks[0], {"t": 1, "game": None, "session": None, "load": None})
        self.assertEqual(peeks[3], {"t": 5, "game": 3, "session": 3, "load": None})
        self.assertEqual(peeks[-1], {"t": 12, "game": 3, "session": 0, "load": 0x100})
        self.assertEqual(presses, [{"t": 5.5, "button": 4}, {"t": 8.0, "button": 4}, {"t": 20.0, "button": 1}])

    def test_session_is_the_low_dword_and_lists_give_their_first_value(self):
        peeks, _ = st.read(self.write_log([
            "[rt] peek t=1s: 0x7446510=3,2 0x98f828c=500000003",
        ]))
        self.assertEqual(peeks, [{"t": 1, "game": 3, "session": 3, "load": None}])

    def test_upper_case_addresses_and_text_around_the_line(self):
        peeks, presses = st.read(self.write_log([
            "12:00:01 [rt] peek t=4s: 0x7446510=3 0x98F828C=D 0x7453ED0=2",
            "noise [rt] gameinput: scripted press of button 7 at 4.25 s trailing",
            "[rt] something else entirely",
        ]))
        self.assertEqual(peeks, [{"t": 4, "game": 3, "session": 0xD, "load": 2}])
        self.assertEqual(presses, [{"t": 4.25, "button": 7}])

    def test_undecodable_bytes_do_not_stop_the_read(self):
        path = os.path.join(self.tmp, "bytes.log")
        with open(path, "wb") as f:
            f.write(b"\xff\xfe garbage\n[rt] peek t=1s: 0x7446510=2 0x98f828c=1\n")
        peeks, _ = st.read(path)
        self.assertEqual(peeks, [{"t": 1, "game": 2, "session": 1, "load": None}])


class TimelineTest(_Logs):
    def test_stage_changes_in_order(self):
        peeks, presses = st.read(self.write_log(BOOT))
        events = st.timeline(peeks, presses)
        self.assertEqual(events, [
            {"t": 1, "stage": "no title state yet"},
            {"t": 2, "stage": "init"},
            {"t": 3, "stage": "boot screens"},
            {"t": 5, "stage": "landing"},
            {"t": 6, "stage": "first-run pages"},
            {"t": 9, "stage": "story load"},
            {"t": 12, "stage": "story load (loading screen ended)"},
        ])

    def test_each_press_is_paired_with_the_next_change(self):
        peeks, presses = st.read(self.write_log(BOOT))
        st.timeline(peeks, presses)
        self.assertEqual([p["next_change"] for p in presses], [
            {"t": 6, "stage": "first-run pages"},
            {"t": 9, "stage": "story load"},
            None,
        ])

    def test_loading_screen_end_needs_the_byte_to_have_been_2_first(self):
        peeks = [{"t": 1, "game": 3, "session": 0, "load": 0}]
        self.assertEqual(st.timeline(peeks, []), [{"t": 1, "stage": "story load"}])

    def test_unknown_values_are_named_by_their_number(self):
        peeks = [
            {"t": 1, "game": 5, "session": None, "load": None},
            {"t": 2, "game": 3, "session": 7, "load": None},
        ]
        self.assertEqual(st.timeline(peeks, []), [
            {"t": 1, "stage": "game state 0x5"},
            {"t": 2, "stage": "session 0x7"},
        ])


class CommandLineTest(_Logs):
    def test_text_report(self):
        r = self.run_tool(self.write_log(BOOT))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("stages (first peek showing each):", r.stdout)
        self.assertIn("story load (loading screen ended)", r.stdout)
        self.assertIn("button 4  -> first-run pages at 6 s", r.stdout)
        self.assertIn("button 1  -> no later change", r.stdout)

    def test_json_report(self):
        r = self.run_tool(self.write_log(BOOT), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(r.stdout)
        self.assertEqual(len(report["stages"]), 7)
        self.assertEqual(report["presses"][1]["next_change"], {"t": 9, "stage": "story load"})

    def test_a_log_without_peek_lines_says_how_to_get_them(self):
        r = self.run_tool(self.write_log(["[rt] nothing to see"]))
        self.assertEqual(r.returncode, 1)
        self.assertIn("LEONIDA_PEEK=", r.stderr)

    def assert_one_error_line(self, r, path):
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        lines = r.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1, r.stderr)
        self.assertTrue(lines[0].startswith("stage_timeline: error: cannot read "), lines[0])
        self.assertIn(path, lines[0])
        self.assertEqual(r.stdout, "")

    def test_a_missing_log_is_one_error_line(self):
        path = os.path.join(self.tmp, "missing.log")
        self.assert_one_error_line(self.run_tool(path), path)

    def test_a_folder_is_one_error_line(self):
        self.assert_one_error_line(self.run_tool(self.tmp), self.tmp)


if __name__ == "__main__":
    unittest.main()
