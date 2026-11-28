"""Check how the Wine runner handles CTest fixture metadata with benign child scripts."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_wine_tests.py"
WINDOWS_ROOT = "synthetic-checkout"


class WineFixtureSetupTests(unittest.TestCase):
    def run_fixture(self, setup_status: int, *, fixture: bool = True, disabled: bool = False,
                    dependent: bool = False, duplicate_setup: bool = False,
                    cleanup: bool = False, second_fixture: bool = False, consumer_status: int = 0,
                    jobs: int = 1,
                    ) -> tuple[int, str, list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "launches"
            trace.mkdir()

            def entry(name: str, status: int, properties: dict[str, object], *,
                      test_name: str | None = None) -> dict[str, object]:
                (root / f"{name}.py").write_text(
                    "import os\n"
                    "import time\n"
                    "from pathlib import Path\n"
                    "launch = f'{time.monotonic_ns():020d}-{os.getpid()}.txt'\n"
                    f"(Path(__file__).with_name('launches') / launch).write_text({name!r}, encoding='utf-8')\n"
                    f"raise SystemExit({status})\n",
                    encoding="utf-8", newline="\n",
                )
                return {"name": test_name or name, "command": [f"{WINDOWS_ROOT}/{name}.py"],
                        "properties": [{"name": key, "value": value} for key, value in properties.items()]}

            setup_props: dict[str, object] = {"SKIP_RETURN_CODE": 77, "DISABLED": disabled}
            consumer_props: dict[str, object] = {"DEPENDS": ["setup"]}
            if fixture:
                setup_props["FIXTURES_SETUP"] = ["input"]
                consumer_props["FIXTURES_REQUIRED"] = ["input"]
            setup = entry("setup", setup_status, setup_props)
            consumer = entry("consumer", consumer_status, consumer_props)
            tests = [consumer, setup] if fixture else [setup, consumer]
            if cleanup:
                fixtures = ["input"]
                if second_fixture:
                    fixtures.append("other")
                    tests.extend([
                        entry("other_consumer", 0, {"FIXTURES_REQUIRED": ["other"]}),
                        entry("other_setup", 0, {"FIXTURES_SETUP": ["other"]}),
                    ])
                tests.insert(0, entry("cleanup", 0, {"FIXTURES_CLEANUP": fixtures}))
            if duplicate_setup:
                tests.append(entry("second_setup", 0, setup_props, test_name="setup"))
            if dependent:
                tests.append(entry("dependent", 0, {"DEPENDS": ["consumer"]}))
            listing = root / "ctest.json"
            listing.write_text(json.dumps({"tests": tests}), encoding="utf-8", newline="\n")
            result = subprocess.run(
                [sys.executable, "-I", str(RUNNER), "--tests", str(listing), "--windows-root", WINDOWS_ROOT,
                 "--root", directory, "--require", "consumer", "--jobs", str(jobs),
                 "--wine", f'"{Path(sys.executable).as_posix()}"'],
                capture_output=True, text=True, timeout=15,
            )
            calls = [launch.read_text(encoding="utf-8") for launch in sorted(trace.iterdir())]
            return result.returncode, result.stdout + result.stderr, calls

    def test_fixture_setup_is_run_before_an_earlier_listed_consumer(self) -> None:
        status, output, calls = self.run_fixture(0)
        self.assertEqual(status, 0, output)
        self.assertEqual(calls, ["setup", "consumer"])

    def test_explicit_dependency_still_runs_after_the_fixture_consumer(self) -> None:
        status, output, calls = self.run_fixture(0, dependent=True)
        self.assertEqual(status, 0, output)
        self.assertEqual(calls, ["setup", "consumer", "dependent"])

    def test_failed_setup_blocks_a_required_consumer(self) -> None:
        status, output, calls = self.run_fixture(3)
        self.assertEqual(status, 1, output)
        self.assertEqual(calls, ["setup"])
        self.assertIn("[FAIL] consumer", output)
        self.assertIn("failed fixture setup: setup", output)

    def test_passing_setup_with_same_name_does_not_hide_failure(self) -> None:
        status, output, calls = self.run_fixture(3, duplicate_setup=True)
        self.assertEqual(status, 1, output)
        self.assertEqual(calls, ["setup", "second_setup"])
        self.assertIn("failed fixture setup: setup", output)

    def test_skipped_or_disabled_setup_does_not_block(self) -> None:
        for status, disabled, expected in ((77, False, ["setup", "consumer"]), (3, True, ["consumer"])):
            with self.subTest(disabled=disabled):
                code, output, calls = self.run_fixture(status, disabled=disabled)
                self.assertEqual(code, 0, output)
                self.assertEqual(calls, expected)

    def test_generic_depends_does_not_turn_an_optional_failure_into_a_block(self) -> None:
        status, output, calls = self.run_fixture(3, fixture=False)
        self.assertEqual(status, 0, output)
        self.assertEqual(calls, ["setup", "consumer"])
        self.assertIn("[FAIL] setup", output)
        self.assertIn("[PASS] consumer", output)

    def test_shared_cleanup_waits_for_both_fixtures(self) -> None:
        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                code, output, calls = self.run_fixture(0, cleanup=True, second_fixture=True, jobs=jobs)
                self.assertEqual(code, 0, output)
                self.assertCountEqual(calls, ["setup", "consumer", "other_setup", "other_consumer", "cleanup"])
                self.assertEqual(calls[-1], "cleanup")
                self.assertLess(calls.index("setup"), calls.index("consumer"))
                self.assertLess(calls.index("other_setup"), calls.index("other_consumer"))

    def test_cleanup_runs_after_failed_or_skipped_fixture_work(self) -> None:
        cases = [
            (3, 0, 1, ["setup", "cleanup"]),
            (0, 3, 1, ["setup", "consumer", "cleanup"]),
            (77, 0, 0, ["setup", "consumer", "cleanup"]),
        ]
        for jobs in (1, 2):
            for setup, consumer, expected_code, expected_calls in cases:
                with self.subTest(jobs=jobs, setup=setup, consumer=consumer):
                    code, output, calls = self.run_fixture(
                        setup, cleanup=True, consumer_status=consumer, jobs=jobs,
                    )
                    self.assertEqual(code, expected_code, output)
                    self.assertEqual(calls, expected_calls)
                    self.assertIn("[PASS] cleanup", output)


if __name__ == "__main__":
    unittest.main()
