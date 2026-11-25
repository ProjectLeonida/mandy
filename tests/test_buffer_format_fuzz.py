"""CPU-only tests for tools/buffer_format_fuzz.py's --budget.

    python -I tests/test_buffer_format_fuzz.py

A negative --budget used to be accepted (exit 0) and then silently skipped every random element: `for _ in
range(budget)` just does not run, so -1 and -1000000 produce exactly the run of 0 and the tool still reports "ok".
--self-test was worse: it builds its own patterns and never reads --budget at all, so 0, -100 and 1000000 all
printed the same "self-test: ok" with exit 0. A negative value is a usage error (2), raised before the self-test
runs and before the oracle is opened; 0 stays legal (the sweeps remain, no random extras are added).

No GPU and no game file is used: the hardware path stubs hw_oracle.
"""
import contextlib
import io
import os
import random
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
TOOL = os.path.join(TOOLS, "buffer_format_fuzz.py")
sys.path.insert(0, TOOLS)

import buffer_format_fuzz as bff  # noqa: E402

MESSAGE = "--budget cannot be negative"


def run(*args):
    return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, check=False)


def make_hw_oracle_pkg(opened):
    """A hw_oracle that answers like a missing GPU, remembering whether it was asked at all."""
    pkg = types.ModuleType("hw_oracle")
    runner_mod = types.ModuleType("hw_oracle.runner")
    asm_mod = types.ModuleType("hw_oracle.asm")

    class Unavailable(Exception):
        pass

    def open_oracle(arch=None):
        opened.append(arch)
        raise Unavailable("no GPU here")

    runner_mod.Unavailable = Unavailable
    runner_mod.open_oracle = open_oracle
    asm_mod.family = lambda arch: "gfx11"
    pkg.runner = runner_mod
    pkg.asm = asm_mod
    return {"hw_oracle": pkg, "hw_oracle.runner": runner_mod, "hw_oracle.asm": asm_mod}


def call_main(argv):
    """main() with argv, hw_oracle stubbed: -> (SystemExit code, stderr, stdout)."""
    mods = make_hw_oracle_pkg([])
    out, err = io.StringIO(), io.StringIO()
    with patch.dict(sys.modules, mods):
        with patch.object(sys, "argv", ["buffer_format_fuzz.py", *argv]):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                with unittest.TestCase().assertRaises(SystemExit) as caught:
                    bff.main()
    return caught.exception.code, err.getvalue(), out.getvalue()


class BudgetUsageErrorTests(unittest.TestCase):
    def test_negative_budget_fails_the_self_test_with_2(self):
        proc = run("--self-test", "--budget", "-1")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn(MESSAGE, proc.stderr)
        self.assertNotIn("traceback", proc.stderr.lower())
        self.assertNotIn("self-test: ok", proc.stdout)

    def test_absurdly_negative_budget_fails_too(self):
        proc = run("--self-test", "--budget", "-1000000")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn(MESSAGE, proc.stderr)

    def test_negative_budget_is_refused_before_the_oracle_is_opened(self):
        code, err, out = call_main(["--oracle", "hw", "--budget", "-3"])
        self.assertEqual(code, 2, out + err)
        self.assertIn(MESSAGE, err)
        self.assertNotIn("hardware oracle unavailable", out)   # a usage error, not "no GPU"

    def test_zero_budget_stays_legal(self):
        proc = run("--self-test", "--budget", "0")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("self-test: ok", proc.stdout)


class BudgetValueTests(unittest.TestCase):
    def test_zero_keeps_the_sweeps_and_adds_no_random_elements(self):
        rng = random.Random(1)
        sweeps = len(bff.load_patterns("8_8_8_8_UNORM", rng, 0))
        self.assertGreater(sweeps, 0)
        self.assertEqual(len(bff.load_patterns("8_8_8_8_UNORM", random.Random(1), 0)), sweeps)

    def test_a_negative_budget_is_indistinguishable_from_zero(self):
        """Why -1 must be a usage error rather than "just no random elements"."""
        rng = random.Random(1)
        zero = len(bff.load_patterns("8_8_8_8_UNORM", rng, 0))
        self.assertEqual(len(bff.load_patterns("8_8_8_8_UNORM", random.Random(1), -1)), zero)

    def test_a_budget_adds_two_elements_per_unit(self):
        rng = random.Random(1)
        zero = len(bff.load_patterns("8_8_8_8_UNORM", rng, 0))
        self.assertEqual(len(bff.load_patterns("8_8_8_8_UNORM", random.Random(1), 3)) - zero, 6)


if __name__ == "__main__":
    unittest.main()
