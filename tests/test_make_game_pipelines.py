"""make_game.py: gta6rt --precompile-pipelines invocation, progress, env and crash handling.

    python -I tests/test_make_game_pipelines.py

Standard library only; no GPU, no Vulkan SDK, no Wine, no game data: uses a small Python script
as a fake gta6rt executed via sys.executable.
"""
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game  # noqa: E402


FAKE_GTA6RT = textwrap.dedent("""\
    import json, os, sys

    state_file = os.environ["FAKE_STATE"]
    plan = json.loads(os.environ["FAKE_PLAN"])
    if os.path.isfile(state_file):
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"calls": []}

    idx = len(state["calls"])
    state["calls"].append({
        "argv": sys.argv[1:],
        "LEONIDA_SHADERS": os.environ.get("LEONIDA_SHADERS"),
        "LEONIDA_PIPELINE_CACHE_DIR": os.environ.get("LEONIDA_PIPELINE_CACHE_DIR"),
        "CUSTOM_PASSTHROUGH": os.environ.get("CUSTOM_PASSTHROUGH"),
        "LEONIDA_DEVICE_FAULT": os.environ.get("LEONIDA_DEVICE_FAULT"),
    })
    with open(state_file, "w", encoding="utf-8", newline="\\n") as f:
        json.dump(state, f)

    step = plan[min(idx, len(plan) - 1)]
    cache_root = os.environ.get("FAKE_CACHE_ROOT", "")
    req_dir = os.path.join(cache_root, "pipeline_cache.d", "requests") if cache_root else ""
    inflight_path = os.path.join(req_dir, "inflight.txt") if req_dir else ""
    crashed_path = os.path.join(req_dir, "crashed.txt") if req_dir else ""
    suspect_path = os.path.join(req_dir, "suspect.txt") if req_dir else ""

    def read_lines(path):
        if not path or not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def write_lines(path, lines):
        if not lines:
            if os.path.isfile(path):
                os.remove(path)
            return
        os.makedirs(req_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\\n") as f:
            f.writelines(line + "\\n" for line in lines)

    # Mirror the runtime's two-strike record (runtime/src/precompile_inflight.h): a line left in inflight.txt becomes
    # a suspect; left again while it is a suspect, it is crashed.
    if inflight_path and os.path.isfile(inflight_path):
        stale = read_lines(inflight_path)
        os.remove(inflight_path)
        suspects, crashed = read_lines(suspect_path), read_lines(crashed_path)
        for line in stale:
            if line in suspects:
                suspects.remove(line)
                crashed.append(line)
            else:
                suspects.append(line)
        write_lines(suspect_path, suspects)
        write_lines(crashed_path, crashed)

    if step.get("leave_inflight") and inflight_path:
        os.makedirs(req_dir, exist_ok=True)
        with open(inflight_path, "w", encoding="utf-8", newline="\\n") as f:
            for name in step["leave_inflight"]:
                f.write(name + "\\n")

    if step.get("write_crashed") and crashed_path:
        os.makedirs(req_dir, exist_ok=True)
        with open(crashed_path, "w", encoding="utf-8", newline="\\n") as f:
            for name in step["write_crashed"]:
                f.write(name + "\\n")

    for line in step.get("stderr", []):
        print(line, file=sys.stderr, flush=True)
    for line in step.get("stdout", []):
        print(line, flush=True)
    if step.get("signal"):  # killed from outside (POSIX): the exit code is the negative signal
        import signal
        os.kill(os.getpid(), getattr(signal, step["signal"]))
    sys.exit(int(step.get("exit", 0)))
""")


# An access violation (0xC0000005): a crash on Windows, and under Wine its low byte (5), also a crash
# (make_game.precompile_exit_kind). A code such as 139 is a crash only off Windows.
CRASH = 0xC0000005


class PipelinesRuntimeHelpersTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name

    def touch(self, *parts):
        path = os.path.join(self.root, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb"):
            pass
        return path

    def test_runtime_exe_search_order_windows(self):
        out = os.path.join(self.root, "out")
        supplied = os.path.join(self.root, "supplied")
        default_exe = self.touch("runtime", "build", "gta6rt.exe")
        with mock.patch.object(make_game, "ROOT", self.root), mock.patch.object(make_game.os, "name", "nt"):
            self.assertEqual(make_game.runtime_exe(out, supplied), default_exe)
            supplied_exe = self.touch("supplied", "gta6rt.exe")
            self.assertEqual(make_game.runtime_exe(out, supplied), supplied_exe)
            out_exe = self.touch("out", "gta6rt.exe")
            self.assertEqual(make_game.runtime_exe(out, supplied), out_exe)

    def test_runtime_exe_search_order_posix(self):
        out = os.path.join(self.root, "out")
        supplied = os.path.join(self.root, "supplied")
        default_exe = self.touch("runtime", "build", "windows", "gta6rt.exe")
        with mock.patch.object(make_game, "ROOT", self.root), mock.patch.object(make_game.os, "name", "posix"):
            self.assertEqual(make_game.runtime_exe(out, supplied), default_exe)
            supplied_exe = self.touch("supplied", "gta6rt.exe")
            self.assertEqual(make_game.runtime_exe(out, supplied), supplied_exe)
            out_exe = self.touch("out", "gta6rt.exe")
            self.assertEqual(make_game.runtime_exe(out, supplied), out_exe)

    def test_precompile_command_windows_and_wine(self):
        # Smallest first always; on Windows the runtime stops with this process (--parent), under Wine it cannot.
        with mock.patch.object(make_game.os, "name", "nt"), mock.patch.object(make_game.os, "getpid", return_value=4242):
            self.assertEqual(
                make_game.precompile_command(r"C:\Port\gta6rt.exe", 6),
                [r"C:\Port\gta6rt.exe", "--precompile-pipelines", "--threads", "6", "--smallest-first",
                 "--parent", "4242"],
            )
            self.assertEqual(
                make_game.precompile_command(r"C:\Port\gta6rt.exe", 6, skip_listed=True),
                [r"C:\Port\gta6rt.exe", "--precompile-pipelines", "--threads", "6", "--smallest-first",
                 "--skip-listed", "--parent", "4242"],
            )
        env = {k: v for k, v in os.environ.items() if k != "WINE"}
        with mock.patch.object(make_game.os, "name", "posix"), mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                make_game.precompile_command("/port/gta6rt.exe", 4),
                ["wine", "/port/gta6rt.exe", "--precompile-pipelines", "--threads", "4", "--smallest-first"],
            )
        with mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch.dict(os.environ, {"WINE": "/opt/proton/files/bin/wine"}):
            self.assertEqual(
                make_game.precompile_command("/port/gta6rt.exe", 4),
                ["/opt/proton/files/bin/wine", "/port/gta6rt.exe", "--precompile-pipelines", "--threads", "4",
                 "--smallest-first"],
            )

    def test_exit_kinds(self):
        kind = make_game.precompile_exit_kind
        with mock.patch.object(make_game.os, "name", "nt"):
            self.assertEqual([kind(c) for c in (0, 1, 2, 3)], ["ok", "device", "failed", "locked"])
            for code in (0xC0000005, 0xC0000409, 0x80000003, -1073741819):  # -1073741819: 0xC0000005 as signed
                self.assertEqual(kind(code), "crash", hex(code))
            for code in (0xC000013A, 4, 255, 0x40010004):  # Ctrl+C, TerminateProcess codes, DBG_TERMINATE_PROCESS
                self.assertEqual(kind(code), "stopped", hex(code))
        with mock.patch.object(make_game.os, "name", "posix"):
            self.assertEqual([kind(c) for c in (0, 1, 2, 3)], ["ok", "device", "failed", "locked"])
            for code in (5, 9, 139, -11, -6):  # Wine's low byte of 0xC0000005 / 0xC0000409; wine itself crashing
                self.assertEqual(kind(code), "crash", code)
            for code in (-1, -2, -9, -15):
                self.assertEqual(kind(code), "stopped", code)

    def test_wine_path_conversion(self):
        with mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch.object(make_game.os.path, "abspath", return_value="/tmp/work/shaders/spirv"):
            self.assertEqual(make_game.wine_path("ignored"), r"Z:\tmp\work\shaders\spirv")
        with mock.patch.object(make_game.os, "name", "nt"), \
                mock.patch.object(make_game.os.path, "abspath", return_value=r"D:\work\shaders\spirv"):
            self.assertEqual(make_game.wine_path("ignored"), r"D:\work\shaders\spirv")


class PipelinesStageTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.game = os.path.join(self.root, "game")
        self.work = os.path.join(self.root, "work")
        self.out = os.path.join(self.root, "out")
        self.spirv = os.path.join(self.work, "shaders", "spirv")
        os.makedirs(self.game)
        os.makedirs(self.spirv)
        os.makedirs(self.out)
        with open(os.path.join(self.spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            f.write('{"cs_a": {}, "cs_b": {}}\n')
        self.fake_exe = os.path.join(self.out, "gta6rt.exe")
        with open(self.fake_exe, "w", encoding="utf-8", newline="\n") as f:
            f.write(FAKE_GTA6RT)
        self.state_file = os.path.join(self.root, "state.json")

    def run_stage(self, plan, *, jobs=4, env_extra=None, os_name=None, runtime_build=None, board=None):
        env = {k: v for k, v in os.environ.items()
               if k not in ("LEONIDA_SHADERS", "LEONIDA_PIPELINE_CACHE_DIR", "LEONIDA_DEVICE_FAULT", "WINE")}
        env["FAKE_STATE"] = self.state_file
        env["FAKE_PLAN"] = json.dumps(plan)
        env["FAKE_CACHE_ROOT"] = self.out
        env["CUSTOM_PASSTHROUGH"] = "keep-me"
        if env_extra:
            env.update(env_extra)
        logs = []
        stderr = io.StringIO()

        real_cmd = make_game.precompile_command

        def fake_cmd(exe, threads, **kw):
            # The real flags, launched through this Python instead of Wine (no --parent off Windows).
            full = real_cmd(exe, threads, **kw)
            return [sys.executable, *full[full.index(exe):]]

        patches = [
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(make_game, "precompile_command", side_effect=fake_cmd),
            mock.patch.object(make_game, "log", logs.append),
            mock.patch("sys.stderr", stderr),
        ]
        if os_name is not None:
            patches.append(mock.patch.object(make_game.os, "name", os_name))
        if board is not None:
            patches.append(mock.patch.object(make_game.leonida_tui, "current_board", return_value=board))

        with context_stack(patches):
            exit_code = None
            try:
                make_game.pipelines(self.work, self.out, jobs, runtime_build=runtime_build)
            except SystemExit as exc:
                exit_code = exc.code
        calls = []
        if os.path.isfile(self.state_file):
            with open(self.state_file, encoding="utf-8") as f:
                calls = json.load(f)["calls"]
        return exit_code, calls, logs, stderr.getvalue()

    def test_invalid_index_shape_preserves_the_previous_crash_report(self):
        # The index is read before the old report is dropped: a stage that cannot start keeps the last one.
        index = os.path.join(self.spirv, "index.json")
        report = os.path.join(self.out, "pipeline_crashes.json")
        for invalid in (["shader"], "shader", 1):
            with self.subTest(index=invalid):
                with open(index, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(invalid, f)
                with open(report, "w", encoding="utf-8", newline="\n") as f:
                    f.write("previous crash report")
                code, calls, _, err = self.run_stage([{"exit": 0}])
                self.assertIsNotNone(code)
                self.assertEqual(calls, [])
                self.assertIn("JSON object", err)
                with open(report, encoding="utf-8") as f:
                    self.assertEqual(f.read(), "previous crash report")

    def test_a_locked_old_crash_report_does_not_prevent_a_clean_run(self):
        report = os.path.join(self.out, "pipeline_crashes.json")
        old_report = '{"modules": [{"name": "old", "spv_sha256": null}]}'
        with open(report, "w", encoding="utf-8", newline="\n") as f:
            f.write(old_report)
        remove = os.remove

        def remove_unlocked(path):
            if os.path.abspath(path) == os.path.abspath(report):
                raise PermissionError("report is locked")
            remove(path)

        with mock.patch.object(make_game.os, "remove", side_effect=remove_unlocked):
            code, calls, logs, err = self.run_stage([{"exit": 0}])
        self.assertIsNone(code, err)
        self.assertEqual(len(calls), 1)
        self.assertTrue(any("cannot remove old pipeline_crashes.json" in line and "may be stale" in line
                            for line in logs), logs)
        with open(report, encoding="utf-8") as f:
            self.assertEqual(f.read(), old_report)

    def test_args_env_progress_and_summary(self):
        plan = [{
            "stderr": [
                "[precompile] 2 compute entries to build (0.1 MB of SPIR-V; 0 filtered out, 0 not buildable)",
                "[precompile] 1/2 cs_a (128 B): compiled and journaled in 0.1 s (VkResult 0) | 1 compiled, 0 from a cache, 0 skipped, 0 failed; 0 s elapsed, ~0 s left by size",
                "[precompile] 2/2 cs_b (128 B): compiled and journaled in 0.1 s (VkResult 0) | 2 compiled, 0 from a cache, 0 skipped, 0 failed; 0 s elapsed, ~0 s left by size",
                "[precompile] done: 2 compiled, 0 from a cache, 0 journaled already, 0 failed, 0 not buildable, in 1 s",
            ],
            "exit": 0,
        }]
        fake_board = mock.Mock()
        fake_board._stop = mock.Mock(is_set=mock.Mock(return_value=False))
        code, calls, logs, err = self.run_stage(plan, jobs=7, board=fake_board)
        self.assertIsNone(code, err)
        self.assertEqual(len(calls), 1)
        # On Windows the runtime also watches this process (--parent).
        parent = ["--parent", str(os.getpid())] if os.name == "nt" else []
        self.assertEqual(calls[0]["argv"], ["--precompile-pipelines", "--threads", "7", "--smallest-first", *parent])
        self.assertEqual(calls[0]["LEONIDA_SHADERS"], make_game.wine_path(self.spirv))
        self.assertEqual(calls[0]["LEONIDA_PIPELINE_CACHE_DIR"], make_game.wine_path(self.out))
        self.assertEqual(calls[0]["CUSTOM_PASSTHROUGH"], "keep-me")
        self.assertEqual(calls[0]["LEONIDA_DEVICE_FAULT"], "1")  # as boot.ps1, so the boot reads this journal
        # Progress parsed 1/2 and 2/2, updated TUI summary, and logged the done summary line.
        self.assertTrue(any("pipelines: 0/2 (0%)" in line for line in logs), logs)
        self.assertTrue(any("pipelines: 2/2 (100%)" in line for line in logs), logs)
        self.assertTrue(any("[precompile] done: 2 compiled" in line for line in logs), logs)
        self.assertIn("pipelines: done", logs)
        fake_board.set_summary.assert_any_call("pipelines: 1/2 (50%)")
        fake_board.set_summary.assert_any_call("pipelines: 2/2 (100%)")

    @unittest.skipIf(os.name == "nt", "exercises POSIX WINE launcher path directly")
    def test_posix_runs_via_wine_env_without_patching_command(self):
        plan = [{
            "stderr": [
                "[precompile] 1/1 cs_a (64 B): compiled and journaled in 0.1 s (VkResult 0)",
                "[precompile] done: 1 compiled, 0 from a cache, 0 journaled already, 0 failed, 0 not buildable, in 0 s",
            ],
            "exit": 0,
        }]
        env = {k: v for k, v in os.environ.items() if k not in ("LEONIDA_SHADERS", "LEONIDA_PIPELINE_CACHE_DIR")}
        env.update({
            "WINE": sys.executable,
            "FAKE_STATE": self.state_file,
            "FAKE_PLAN": json.dumps(plan),
            "FAKE_CACHE_ROOT": self.out,
        })
        logs = []
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(make_game, "log", logs.append):
            make_game.pipelines(self.work, self.out, 3)
        with open(self.state_file, encoding="utf-8") as f:
            calls = json.load(f)["calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["argv"], ["--precompile-pipelines", "--threads", "3", "--smallest-first"])
        self.assertEqual(calls[0]["LEONIDA_SHADERS"], "Z:" + os.path.abspath(self.spirv).replace("/", "\\"))
        self.assertEqual(calls[0]["LEONIDA_PIPELINE_CACHE_DIR"], "Z:" + os.path.abspath(self.out).replace("/", "\\"))

    def test_user_set_shader_and_cache_env_are_preserved_and_logged(self):
        plan = [{"stderr": ["[precompile] done: 0 compiled, 0 from a cache, 0 journaled already, 0 failed, 0 not buildable, in 0 s"], "exit": 0}]
        custom_shaders = r"D:\custom\spirv"
        custom_cache = r"D:\custom\cache"
        code, calls, logs, err = self.run_stage(
            plan,
            env_extra={"LEONIDA_SHADERS": custom_shaders, "LEONIDA_PIPELINE_CACHE_DIR": custom_cache},
        )
        self.assertIsNone(code, err)
        self.assertEqual(calls[0]["LEONIDA_SHADERS"], custom_shaders)
        self.assertEqual(calls[0]["LEONIDA_PIPELINE_CACHE_DIR"], custom_cache)
        self.assertTrue(any("LEONIDA_SHADERS" in line and custom_shaders in line for line in logs), logs)
        self.assertTrue(any("LEONIDA_PIPELINE_CACHE_DIR" in line and custom_cache in line for line in logs), logs)

    def test_user_set_device_fault_is_kept(self):
        # 2 (the fault report without NVIDIA's diagnostics config) and an empty value (off) are passed on as they are.
        for value in ("2", ""):
            code, calls, logs, err = self.run_stage([{"exit": 0}], env_extra={"LEONIDA_DEVICE_FAULT": value})
            self.assertIsNone(code, err)
            self.assertEqual(calls[-1]["LEONIDA_DEVICE_FAULT"], value)
            self.assertTrue(any("LEONIDA_DEVICE_FAULT from environment" in line for line in logs), logs)

    def test_device_fault_matches_the_boot_script(self):
        # The stage's default is what scripts/windows-boot/boot.ps1 boots with, read from the script itself.
        with open(os.path.join(ROOT, "scripts", "windows-boot", "boot.ps1"), encoding="utf-8-sig") as f:
            match = re.search(r'^\s*LEONIDA_DEVICE_FAULT\s*=\s*"([^"]*)"', f.read(), re.M)
        self.assertIsNotNone(match)
        self.assertEqual(make_game.PIPELINE_DEVICE_FAULT, match.group(1))

    def test_exit_2_logs_warning_and_completes_stage(self):
        plan = [{
            "stderr": ["[precompile] done: 1 compiled, 0 from a cache, 0 journaled already, 1 failed, 0 not buildable, in 1 s"],
            "exit": 2,
        }]
        code, calls, logs, err = self.run_stage(plan)
        self.assertIsNone(code, err)
        self.assertEqual(len(calls), 1)
        self.assertTrue(any("WARNING" in line and "exit 2" in line for line in logs), logs)
        self.assertIn("pipelines: done", logs)

    def test_exit_1_fails_stage_without_retrying(self):
        plan = [{"stderr": ["[precompile] failed to create Vulkan device"], "exit": 1}]
        code, calls, logs, err = self.run_stage(plan)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("could not open the Vulkan device or shader library", err)

    def test_stop_from_outside_is_not_a_crash(self):
        # Ctrl+C (STATUS_CONTROL_C_EXIT) or a TerminateProcess code on Windows, a stop signal elsewhere: no restart.
        for os_name, code in (("nt", 0xC000013A), ("nt", 4), ("posix", -15), ("posix", -9)):
            err = io.StringIO()
            with mock.patch.object(make_game, "wine_available", return_value=True), \
                    mock.patch.object(make_game, "runtime_exe", return_value=self.fake_exe), \
                    mock.patch.object(make_game, "_run_precompile", return_value=code) as run_mock, \
                    mock.patch.object(make_game.os, "name", os_name), \
                    mock.patch.object(make_game, "log"), \
                    mock.patch("sys.stderr", err):
                with self.assertRaises(SystemExit) as ctx:
                    make_game.pipelines(self.work, self.out, 2)
            self.assertEqual(ctx.exception.code, 1)
            self.assertEqual(run_mock.call_count, 1, (os_name, code))
            self.assertIn("not a crash", err.getvalue())

    @unittest.skipIf(os.name == "nt", "a POSIX kill signal")
    def test_a_kill_leaves_a_suspect_that_the_next_stage_builds(self):
        # A kill from outside leaves the run's record as a crash would: one death is a suspect, built again, not
        # skipped; the next stage's run builds it and nothing is reported as crashed.
        plan = [{"leave_inflight": ["cs_long k1"], "signal": "SIGKILL"},
                {"stderr": ["[precompile] done: 2 compiled, 0 from a cache, 0 journaled already, 0 failed, 0 not buildable, in 1 s"],
                 "exit": 0}]
        code, calls, logs, err = self.run_stage(plan)
        self.assertEqual(code, 1)
        self.assertIn("not a crash", err)
        req = os.path.join(self.out, "pipeline_cache.d", "requests")
        code, calls, logs, err = self.run_stage(plan)  # the state file says call 1 is done: this is the plan's 2nd
        self.assertIsNone(code, err)
        self.assertEqual(len(calls), 2)
        self.assertFalse(os.path.exists(os.path.join(req, "crashed.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "pipeline_crashes.json")))
        self.assertFalse(any("WARNING" in line for line in logs), logs)

    def test_tui_stop_is_reported_as_a_stop(self):
        board = mock.Mock()
        board._stop = mock.Mock(is_set=mock.Mock(return_value=True))
        err = io.StringIO()
        with mock.patch.object(make_game.leonida_tui, "current_board", return_value=board), \
                mock.patch.object(make_game, "wine_available", return_value=True), \
                mock.patch.object(make_game, "runtime_exe", return_value=self.fake_exe), \
                mock.patch.object(make_game, "_run_precompile", return_value=1) as run_mock, \
                mock.patch.object(make_game, "log"), \
                mock.patch("sys.stderr", err):
            with self.assertRaises(SystemExit):
                make_game.pipelines(self.work, self.out, 2)
        self.assertEqual(run_mock.call_count, 1)
        message = board.write.call_args.args[1]  # fail() writes to the board when there is one
        self.assertIn("pipelines: stopped", message)
        self.assertNotIn("Vulkan device", message)

    def test_skip_compute_env_passes_skip_listed(self):
        code, calls, logs, err = self.run_stage([{"exit": 0}], env_extra={"LEONIDA_SKIP_COMPUTE": "cs_a,cs_b"})
        self.assertIsNone(code, err)
        self.assertIn("--skip-listed", calls[0]["argv"])
        code, calls, logs, err = self.run_stage([{"exit": 0}], env_extra={"LEONIDA_SKIP_COMPUTE": " "})
        self.assertNotIn("--skip-listed", calls[-1]["argv"])

    def test_crash_retries_and_writes_pipeline_crashes_json_from_crashed_txt(self):
        lib_dir = os.path.join(self.spirv, "library")
        os.makedirs(lib_dir)
        blob = b"\x03\x02\x23\x07crash-spv"
        with open(os.path.join(lib_dir, "lib_cs_00042.spv"), "wb") as f:
            f.write(blob)

        plan = [
            # The runtime writes "<name> <key>" (runtime/src/precompile_inflight.h); the stage reports the name.
            # The first death makes it a suspect, built again; the second, by that retry, holds it as crashed.
            {"leave_inflight": ["lib_cs_00042 0123456789abcdef"], "exit": CRASH},
            {"leave_inflight": ["lib_cs_00042 0123456789abcdef"], "exit": CRASH},
            {
                "stderr": ["[precompile] done: 1 compiled, 0 from a cache, 0 journaled already, 1 failed, 0 not buildable, in 1 s"],
                "exit": 2,
            },
        ]
        code, calls, logs, err = self.run_stage(plan)
        self.assertIsNone(code, err)
        self.assertEqual(len(calls), 3)
        restarts = [line for line in logs if "crashed" in line and "while the driver built lib_cs_00042" in line]
        self.assertEqual(len(restarts), 2, logs)
        self.assertTrue(any("lib_cs_00042" in line and "pipeline_crashes.json" in line for line in logs), logs)
        report = os.path.join(self.out, "pipeline_crashes.json")
        with open(report, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc, {"modules": [{"name": "lib_cs_00042", "spv_sha256": hashlib.sha256(blob).hexdigest()}]})
        # One warning for the crashed entry, not a second generic one for the same exit 2; it says how to retry.
        warnings = [line for line in logs if "WARNING" in line]
        self.assertEqual(len(warnings), 1, warnings)
        self.assertIn("--all", warnings[0])

    def test_exit_3_another_precompile_fails_with_its_own_message(self):
        plan = [{"stderr": ["[precompile] another --precompile-pipelines is using X (error 32)"], "exit": 3}]
        code, calls, logs, err = self.run_stage(plan)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertIn("another gta6rt --precompile-pipelines is using this pipeline journal", err)
        self.assertNotIn("Vulkan device", err)

    def test_crashes_outside_a_build_stop_after_retry_limit(self):
        plan = [{"exit": CRASH}]
        with mock.patch.object(make_game, "PIPELINE_CRASH_RETRIES", 2):
            code, calls, logs, err = self.run_stage(plan)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 3)   # initial attempt + 2 restarts
        self.assertIn("kept crashing", err)
        self.assertTrue(any("outside a pipeline build" in line and "(2/2)" in line for line in logs), logs)

    def test_crashes_inside_builds_are_bounded_by_max_restarts(self):
        # A runtime that never skips what it crashed on: the stage still ends, naming what is crashed.
        plan = [{"leave_inflight": ["cs_bad k"], "exit": CRASH}]
        with mock.patch.object(make_game, "PIPELINE_MAX_RESTARTS", 3):
            code, calls, logs, err = self.run_stage(plan)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 4)
        self.assertIn("kept crashing", err)
        self.assertIn("crashed entries: cs_bad", err)
        with open(os.path.join(self.out, "pipeline_crashes.json"), encoding="utf-8") as f:
            self.assertEqual([m["name"] for m in json.load(f)["modules"]], ["cs_bad"])

    def test_clean_run_drops_stale_pipeline_crashes_json(self):
        report = os.path.join(self.out, "pipeline_crashes.json")
        with open(report, "w", encoding="utf-8", newline="\n") as f:
            f.write('{"modules": []}\n')
        code, _, _, err = self.run_stage([{"exit": 0}])
        self.assertIsNone(code, err)
        self.assertFalse(os.path.exists(report))

    def test_main_forwards_jobs_and_runtime_build(self):
        supplied = os.path.join(self.root, "supplied")
        os.makedirs(supplied)
        for name in make_game.RUNTIME_FILES:
            with open(os.path.join(supplied, name), "wb"):
                pass
        argv = ["make_game.py", "--game", self.game, "--work", self.work, "--out", self.out,
                "--stages", "pipelines", "--runtime-build", supplied, "--jobs", "5", "--no-tui"]
        with mock.patch.object(make_game.leonida_tui.os, "cpu_count", return_value=16), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "g.exe"}), \
                mock.patch.object(make_game, "preflight"), \
                mock.patch.object(make_game, "log"), \
                mock.patch.object(make_game, "pipelines") as stage:
            self.assertEqual(make_game.main(), 0)
        stage.assert_called_once_with(
            os.path.abspath(self.work), os.path.abspath(self.out), 5, runtime_build=os.path.abspath(supplied)
        )


class context_stack:
    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.start()

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


if __name__ == "__main__":
    unittest.main()
