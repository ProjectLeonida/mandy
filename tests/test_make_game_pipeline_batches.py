"""Pipeline batching and memory-crash recovery through the real make_game.pipelines orchestration.

The child is a small scripted checker speaking the production CLI/file protocol. Each invocation is a real
process, with synthetic index entries and controlled exits; no GPU, compiler, game data or packages are needed.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_game  # noqa: E402


CHECKER = r'''
import json
import os
from pathlib import Path
import signal
import sys

if sys.argv[1] == "--merge":
    # The stand-in cache is a JSON list of completed names, so a merge is the output's list plus each input's.
    out, *inputs = map(Path, sys.argv[2:])
    control = json.loads(Path(os.environ["SYNTHETIC_MERGE_CONTROL"]).read_text(encoding="utf-8"))
    with Path(os.environ["SYNTHETIC_MERGE_CONTROL"]).with_name("merges.jsonl").open("a", encoding="utf-8",
                                                                                  newline="\n") as f:
        f.write(json.dumps([str(p) for p in inputs]) + "\n")
    if control.get("merge_exit"):
        sys.exit(control["merge_exit"])
    merged = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    for p in inputs:
        merged += json.loads(p.read_text(encoding="utf-8"))
    out.write_text(json.dumps(merged), encoding="utf-8", newline="\n")
    sys.exit(0)
spirv, cache, *args = sys.argv[1:]
spirv, cache = Path(spirv), Path(cache)
config = json.loads((spirv / "control.json").read_text(encoding="utf-8"))
names = sorted(json.loads((spirv / "index.json").read_text(encoding="utf-8")), key=lambda n: n.encode("utf-8"))
def option(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
names_file = option("--names-file")
if names_file:
    names = [n for n in names if n in Path(names_file).read_text(encoding="utf-8").splitlines()]
start = int(option("--start", 0))
count = int(option("--count", len(names)))
selected = names[start:start + count]
skip_path = option("--skip-file")
skipped = Path(skip_path).read_text(encoding="utf-8").splitlines() if skip_path else []
active = [name for name in selected if name not in skipped]
inflight = option("--inflight-file")
def record(event):
    with (spirv / "calls.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(event=event, pid=os.getpid(), selected=selected, active=active, args=args)) + "\n")
record("started")
if config.get("setup_error") or config.get("probe_setup_error") and count == 1 and option("--jobs") == "1":
    print("no Vulkan physical devices found", file=sys.stderr)
    sys.exit(config.get("setup_exit", 2))
if (config.get("interrupt") or config.get("interrupt_probe") and count == 1 and option("--jobs") == "1"
        or config.get("interrupt_at") in active):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.ExitProcess(0xC000013A)
    else:
        os.kill(os.getpid(), signal.SIGTERM)
if config.get("crash_before_work"):
    sys.exit(70)
if config.get("probe_crash_before_work") and count == 1 and option("--jobs") == "1":
    report = config.get("probe_report", "missing")
    if inflight and report != "missing":
        Path(inflight).write_text("unrelated\tpipeline\n" if report == "other" else "",
                                  encoding="utf-8", newline="\n")
    sys.exit(70)
crasher = config.get("crasher")
if crasher in active or len(active) > config.get("max_batch", 1000000):
    if inflight:
        suspects = [crasher] if crasher in active else active[:config.get("suspects", 2)]
        Path(inflight).write_text("".join(n + "\tpipeline\n" for n in suspects), encoding="utf-8", newline="\n")
    sys.exit(70)
# This stand-in's cache contains completed names so the test can detect lost checkpoints and repeated work.
completed = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
cache.write_text(json.dumps(completed + active), encoding="utf-8", newline="\n")
record("completed")
if inflight:
    Path(inflight).write_text("", encoding="utf-8", newline="\n")
if option("--done-file"):
    passed = sorted(n for n in active if n != config.get("invalid"))
    Path(option("--done-file")).write_text("".join(n + "\n" for n in passed), encoding="utf-8", newline="\n")
sys.exit(1 if skipped and set(selected).intersection(skipped) or config.get("invalid") in active else 0)
'''


class PipelineBatchesTest(unittest.TestCase):
    """A checker before --merge: every process loads and saves pipeline_cache.bin (BUILD-1030)."""
    SHARDED = False
    DEVICE = {"vendor": "0x10de", "device": "0x2783", "driver": "0x9b404000", "api": "0x00404165",
              "uuid": "00" * 16, "name": "Synthetic GPU"}

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="pipeline batches ")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.work = self.root / "work"
        self.spirv = self.work / "shaders" / "spirv"
        self.spirv.mkdir(parents=True)
        self.out = self.root / "out"
        self.names = [f"m_{i:02d}" for i in range(7)]
        self.write_index(self.names)
        self.child = self.root / "checker.py"
        self.child.write_text(CHECKER, encoding="utf-8", newline="\n")
        self.commands = []
        self.device = dict(self.DEVICE)
        self.fingerprint = "sources:synthetic"

    def write_index(self, names, meta=None):
        (self.spirv / "index.json").write_text(json.dumps({n: (meta or {}).get(n, {}) for n in reversed(names)}),
                                                encoding="utf-8", newline="\n")
        for n in names:   # the done-list keys a module by its .spv; these bytes only stand for one
            if not (self.spirv / (n + ".spv")).exists():
                (self.spirv / (n + ".spv")).write_bytes(n.encode())

    def invoke(self, control=None, range_support=True):
        (self.spirv / "control.json").write_text(json.dumps(control or {}), encoding="utf-8", newline="\n")
        merge_control = self.root / "merge_control.json"
        merge_control.write_text(json.dumps(control or {}), encoding="utf-8", newline="\n")
        def run(args, hint="", check=True, **kwargs):
            args = [str(a) for a in args]
            self.commands.append(args)
            done = subprocess.run([sys.executable, "-I", str(self.child), *args[1:]],
                                  capture_output=True, text=True, timeout=15,
                                  env=dict(os.environ, SYNTHETIC_MERGE_CONTROL=str(merge_control)))
            if check and done.returncode:
                raise SystemExit(1)
            return done.returncode
        def supports(exe, *options):
            if "--merge" in options:
                return self.SHARDED
            return range_support or "--start" not in options
        logs, err = [], io.StringIO()
        # create=True keeps the regression runnable on the old implementation, which lacks this constant.
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="synthetic-checker"), \
                mock.patch.object(make_game, "checker_supports", side_effect=supports), \
                mock.patch.object(make_game, "checker_device_info", side_effect=lambda exe: dict(self.device)), \
                mock.patch.object(make_game, "checker_fingerprint", side_effect=lambda exe: self.fingerprint), \
                mock.patch.object(make_game, "PIPELINE_BATCH_SIZE", 3, create=True), \
                mock.patch.object(make_game, "run", side_effect=run), \
                mock.patch.object(make_game, "log", side_effect=logs.append), contextlib.redirect_stderr(err):
            try:
                make_game.pipelines(str(self.work), str(self.out), jobs=2)
            except SystemExit as exc:
                return exc.code, logs, err.getvalue()
        return 0, logs, err.getvalue()

    def calls(self, event=None):
        path = self.spirv / "calls.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        return [row for row in rows if event is None or row["event"] == event]

    def test_bounded_processes_cover_the_sorted_index_and_tail(self):
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        completed = self.calls("completed")
        self.assertEqual([r["selected"] for r in completed], [self.names[:3], self.names[3:6], self.names[6:]])
        self.assertEqual(len({r["pid"] for r in completed}), 3)
        self.assertEqual(json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")), self.names)
        self.assertTrue(all(r["args"][r["args"].index("--jobs") + 1] == "2" for r in completed))

    def test_bvh_pipelines_are_added_to_the_runtime_seed(self):
        bvh = self.spirv / "bvh"
        bvh.mkdir()
        (bvh / "index.json").write_text('{"b_0": {}}', encoding="utf-8")
        (bvh / "control.json").write_text("{}", encoding="utf-8")
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")),
                         self.names + ["b_0"])
        rows = [json.loads(line) for line in (bvh / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["selected"] for row in rows if row["event"] == "completed"], [["b_0"]])

    def test_collective_memory_crash_splits_without_skipping_healthy_modules(self):
        code, logs, err = self.invoke({"max_batch": 2})
        self.assertEqual(code, 0, err)
        # Diagnostic one-module probes are also successful; the final bounded work must still cover every name.
        checked = {n for row in self.calls("completed") for n in row["active"]}
        self.assertEqual(checked, set(self.names))
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertTrue(any("smaller" in line for line in logs), logs)

    def test_a_real_crasher_stays_failed_when_later_batches_pass(self):
        code, _, err = self.invoke({"crasher": "m_01"})
        self.assertNotEqual(code, 0)
        self.assertEqual((self.out / "pipeline_crashes.txt").read_text(encoding="utf-8"), "m_01\n")
        self.assertIn("m_01", err)
        checked = {n for row in self.calls("completed") for n in row["active"]}
        self.assertEqual(checked, set(self.names) - {"m_01"})

    def test_one_inflight_module_is_not_blacklisted_when_it_survives_alone(self):
        self.write_index(self.names[:2])   # small corpora need the same protection as multi-batch runs
        code, _, err = self.invoke({"max_batch": 1, "suspects": 1})
        self.assertEqual(code, 0, err)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertEqual({n for row in self.calls("completed") for n in row["active"]}, set(self.names[:2]))

    def test_interrupting_a_diagnostic_probe_stops_before_more_batches(self):
        code, _, _ = self.invoke({"max_batch": 2, "interrupt_probe": True})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)   # the failed batch and the interrupted diagnostic probe
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def test_a_validation_failure_does_not_hide_the_remaining_batches(self):
        code, _, _ = self.invoke({"invalid": "m_01"})
        self.assertNotEqual(code, 0)
        self.assertEqual([n for row in self.calls("completed") for n in row["active"]], self.names)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def test_setup_errors_and_interrupts_stop_immediately(self):
        for control in ({"setup_error": True}, {"interrupt": True}):
            with self.subTest(control=control):
                trace = self.spirv / "calls.jsonl"
                if trace.exists():
                    trace.unlink()
                self.commands.clear()
                code, _, _ = self.invoke(control)
                self.assertNotEqual(code, 0)
                self.assertEqual(len(self.calls()), 1)

    def test_native_setup_exit_one_stops_instead_of_checking_every_range(self):
        code, _, _ = self.invoke({"setup_error": True, "setup_exit": 1})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_native_setup_exit_one_in_a_probe_stops_before_splitting(self):
        code, _, _ = self.invoke({"max_batch": 2, "probe_setup_error": True, "setup_exit": 1})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def assert_inconclusive_probe_stops(self, report):
        code, logs, err = self.invoke({"max_batch": 2, "probe_crash_before_work": True,
                                      "probe_report": report})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)   # failed batch, then the first inconclusive probe
        self.assertEqual(self.calls()[-1]["selected"], [self.names[0]])
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertFalse(any("took vk_pipeline_check down" in line for line in logs), logs)
        self.assertIn("inconclusive", err)

    def test_abnormal_probe_without_a_report_does_not_blacklist_a_module(self):
        self.assert_inconclusive_probe_stops("missing")

    def test_abnormal_probe_with_an_empty_report_does_not_blacklist_a_module(self):
        self.assert_inconclusive_probe_stops("empty")

    def test_abnormal_probe_reporting_another_module_does_not_blacklist_the_selected_one(self):
        self.assert_inconclusive_probe_stops("other")

    def test_a_crash_before_work_cannot_retry_forever_or_claim_success(self):
        code, _, err = self.invoke({"crash_before_work": True})
        self.assertNotEqual(code, 0)
        self.assertLessEqual(len(self.calls()), 4)
        self.assertTrue(err)

    def test_an_older_checker_gets_no_unsupported_range_arguments(self):
        code, _, err = self.invoke(range_support=False)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.commands), 1)
        self.assertNotIn("--start", self.commands[0])
        self.assertNotIn("--count", self.commands[0])

    def test_small_indices_keep_the_single_run(self):
        self.write_index(self.names[:2])
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.commands), 1)


class ShardedPipelineBatchesTest(PipelineBatchesTest):
    """The same recovery with a checker that has --merge (PIPELINES-STAGE-SHARDS-AND-DONE-LIST): each process writes
    its own shard cache, the shards are merged into pipeline_cache.bin once, and a module already built for the
    same GPU, driver and checker is not checked again."""
    SHARDED = True

    def shard_dir(self):
        return self.out / "pipeline_shards"

    def done_list(self):
        return json.loads((self.out / "pipeline_done.json").read_text(encoding="utf-8"))

    def merges(self):
        path = self.root / "merges.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def checked(self, since=0):
        return [n for row in self.calls("completed")[since:] for n in row["active"]]

    def cache(self):
        return json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8"))

    def test_an_older_checker_gets_no_unsupported_range_arguments(self):
        self.skipTest("a checker with --merge also has --names-file")

    def test_small_indices_keep_the_single_run(self):
        self.write_index(self.names[:2])
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual([c[1] for c in self.commands], [str(self.spirv), "--merge"])   # one check, one merge

    def test_each_process_writes_its_own_shard_and_one_merge_makes_the_cache(self):
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        checks = [c for c in self.commands if c[1] != "--merge"]
        shards = [c[2] for c in checks]
        self.assertEqual([os.path.basename(s) for s in shards], ["00001.bin", "00002.bin", "00003.bin"])
        self.assertTrue(all(os.path.dirname(s) == str(self.shard_dir()) for s in shards), shards)
        self.assertFalse(any(str(self.out / "pipeline_cache.bin") in c for c in checks))   # never loaded per batch
        self.assertEqual(len(self.merges()), 1)
        self.assertEqual(self.merges()[0], shards)
        self.assertEqual(self.cache(), self.names)
        self.assertFalse(self.shard_dir().exists(), "the shards are removed after a good merge")
        self.assertEqual(sorted(self.done_list()["modules"]), self.names)
        self.assertTrue(all(r["shard"] is None for r in self.done_list()["modules"].values()))

    def test_the_done_list_is_written_with_lf_line_endings(self):
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertNotIn(b"\r\n", (self.out / "pipeline_done.json").read_bytes())

    def test_a_rerun_skips_every_module_already_built(self):
        self.assertEqual(self.invoke()[0], 0)
        before, commands = len(self.calls("started")), len(self.commands)
        code, logs, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.calls("started")), before, "no checker process for an unchanged corpus")
        self.assertEqual(len(self.commands), commands)
        self.assertEqual(self.cache(), self.names)
        self.assertTrue(any("7 of 7 modules are already built" in line for line in logs), logs)

    def test_only_a_module_whose_spirv_changed_is_built_again(self):
        self.assertEqual(self.invoke()[0], 0)
        since = len(self.calls("completed"))
        (self.spirv / "m_04.spv").write_bytes(b"retranslated")
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(self.checked(since), ["m_04"])
        self.assertEqual(self.cache(), self.names + ["m_04"])   # merged onto what the cache already held
        self.assertEqual(len(self.merges()), 2)

    def test_a_changed_index_entry_is_built_again(self):
        self.assertEqual(self.invoke()[0], 0)
        since = len(self.calls("completed"))
        self.write_index(self.names, meta={"m_02": {"stage": "cs", "push_constant_bytes": 128}})
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(self.checked(since), ["m_02"])

    def test_a_new_driver_or_checker_or_stub_builds_everything_again(self):
        changes = {
            "driver": lambda: self.device.update(driver="0x9b408000"),
            "pipelineCacheUUID": lambda: self.device.update(uuid="11" * 16),
            "GPU": lambda: self.device.update(device="0x2704"),
            "checker": lambda: setattr(self, "fingerprint", "sources:other"),
            "stubs": lambda: ((self.spirv / "stubs").mkdir(exist_ok=True),
                              (self.spirv / "stubs" / "vs_0.spv").write_bytes(b"stub")),
        }
        self.assertEqual(self.invoke()[0], 0)
        cache = str(self.out / "pipeline_cache.bin")
        for what, change in changes.items():
            with self.subTest(what=what):
                since, commands = len(self.calls("completed")), len(self.commands)
                change()
                code, _, err = self.invoke()
                self.assertEqual(code, 0, err)
                self.assertEqual(sorted(self.checked(since)), self.names)
                targets = {c[2] for c in self.commands[commands:] if c[1] != "--merge"}
                if what in ("checker", "stubs"):
                    # Same driver: what the cache holds is still valid, so the checks load it instead of starting
                    # empty shards, and the next change goes to shards again.
                    self.assertEqual(targets, {cache})
                    self.assertFalse(self.done_list()["migrating"])
                else:
                    self.assertNotIn(cache, targets)

    def test_a_new_checker_keeps_leftover_shards_of_the_same_driver(self):
        code, _, _ = self.invoke({"interrupt_at": "m_03"})
        self.assertNotEqual(code, 0)
        self.assertTrue((self.shard_dir() / "00001.bin").exists())
        self.fingerprint = "sources:other"
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        # No pipeline_cache.bin yet, so the new checker's checks go to new shards; the old one is merged with them.
        self.assertEqual([os.path.basename(p) for p in self.merges()[-1]][0], "00001.bin")
        self.assertFalse(self.shard_dir().exists())
        self.assertEqual(sorted(set(self.cache())), self.names)

    def test_an_interrupted_run_keeps_its_shards_and_the_next_run_finishes_the_rest(self):
        code, _, _ = self.invoke({"interrupt_at": "m_03"})
        self.assertNotEqual(code, 0)
        self.assertEqual(self.merges(), [], "a stopped run does not merge")
        self.assertEqual([p.name for p in self.shard_dir().iterdir()], ["00001.bin"])
        self.assertFalse((self.out / "pipeline_cache.bin").exists())
        self.assertEqual(sorted(self.done_list()["modules"]), self.names[:3])
        since = len(self.calls("completed"))
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(self.checked(since), self.names[3:])
        self.assertEqual(sorted(self.cache()), self.names)
        self.assertEqual([os.path.basename(p) for p in self.merges()[0]], ["00001.bin", "00002.bin", "00003.bin"])
        self.assertFalse(self.shard_dir().exists())

    def test_a_failed_merge_keeps_the_shards_for_the_next_run(self):
        code, _, err = self.invoke({"merge_exit": 1})
        self.assertNotEqual(code, 0)
        self.assertIn("merges them again", err)
        self.assertEqual(len(list(self.shard_dir().iterdir())), 3)
        self.assertFalse((self.out / "pipeline_cache.bin").exists())
        before = len(self.calls("started"))
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.calls("started")), before, "nothing is checked again, the shards are merged")
        self.assertEqual(self.cache(), self.names)

    def test_the_bvh_pass_runs_again_only_when_its_shaders_change(self):
        bvh = self.spirv / "bvh"
        bvh.mkdir()
        (bvh / "index.json").write_text('{"b_0": {}}', encoding="utf-8")
        (bvh / "control.json").write_text("{}", encoding="utf-8")
        (bvh / "b_0.spv").write_bytes(b"bvh")
        bvh_runs = lambda: sum(1 for c in self.commands if c[1] == str(bvh))   # noqa: E731
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual((bvh_runs(), len(self.merges())), (1, 1))
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual((bvh_runs(), len(self.merges())), (1, 1), "an unchanged corpus needs no process or merge")
        (bvh / "b_0.spv").write_bytes(b"bvh, rebuilt")
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual((bvh_runs(), len(self.merges())), (2, 2))
        self.assertEqual(self.cache(), self.names + ["b_0", "b_0"])

    def test_a_removed_cache_builds_everything_again(self):
        self.assertEqual(self.invoke()[0], 0)
        (self.out / "pipeline_cache.bin").unlink()
        since = len(self.calls("completed"))
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(self.checked(since), self.names)

    def test_a_crasher_and_a_failed_module_are_not_done(self):
        code, _, _ = self.invoke({"crasher": "m_01", "invalid": "m_05"})
        self.assertNotEqual(code, 0)
        self.assertEqual(sorted(self.done_list()["modules"]), sorted(set(self.names) - {"m_01", "m_05"}))
        self.assertEqual(len(self.merges()), 1, "what was built is still merged after a failure")
        since = len(self.calls("completed"))
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(self.checked(since)), ["m_01", "m_05"])

    def test_the_first_run_over_an_existing_cache_checkpoints_into_it(self):
        # A cache from before the done-list: its pipelines must still compile from it, so the first run keeps
        # the old per-process load and save instead of starting empty shards.
        (self.out).mkdir()
        (self.out / "pipeline_cache.bin").write_text(json.dumps(["old"]), encoding="utf-8", newline="\n")
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        checks = [c for c in self.commands if c[1] != "--merge"]
        self.assertTrue(all(c[2] == str(self.out / "pipeline_cache.bin") for c in checks), checks)
        self.assertEqual(self.merges(), [])
        self.assertEqual(self.cache(), ["old"] + self.names)
        self.assertFalse(self.done_list()["migrating"])
        since = len(self.calls("completed"))
        (self.spirv / "m_06.spv").write_bytes(b"changed")
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(self.checked(since), ["m_06"])
        self.assertEqual(len(self.merges()), 1, "after the first run, shards")


if __name__ == "__main__":
    unittest.main()
