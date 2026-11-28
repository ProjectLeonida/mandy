"""Counts-only corpus watch: counting and report-path safety.

Standard library only, no game data, no GPU. Run with:
    python3 -I tests/test_corpus_watch.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import corpus_watch  # noqa: E402


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f)


class CorpusCounts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = self.tmp.name
        os.makedirs(os.path.join(self.work, "shaders", "dxbc"))
        os.makedirs(os.path.join(self.work, "shaders", "spirv"))

    def test_counts_only_validated_entries_and_per_stage(self):
        write_json(os.path.join(self.work, "shaders", "dxbc", "index.json"), [
            {"stage": "ps"}, {"stage": "ps"}, {"stage": "vs"},
        ])
        write_json(os.path.join(self.work, "shaders", "spirv", "index.json"), {
            "a": {"stage": "ps"}, "b": {"stage": "vs"}, "c": {"stage": "ps", "error": "bad"},
        })
        got = corpus_watch.corpus_counts(self.work)
        self.assertEqual(got["carved"], 3)
        self.assertEqual(got["spirv_entries"], 3)
        self.assertEqual(got["validated"], 2)
        self.assertEqual(got["per_stage"]["ps"], [2, 1])
        self.assertEqual(got["per_stage"]["vs"], [1, 1])

    def test_dict_dxbc_index_and_legacy_layout(self):
        legacy = os.path.join(self.tmp.name, "legacy")
        os.makedirs(os.path.join(legacy, "dxbc"))
        os.makedirs(os.path.join(legacy, "spirv"))
        write_json(os.path.join(legacy, "dxbc", "index.json"), {"x": {"stage": "cs"}})
        write_json(os.path.join(legacy, "spirv", "index.json"), {"x": {"stage": "cs"}})
        got = corpus_watch.corpus_counts(legacy)
        self.assertEqual(got["carved"], 1)
        self.assertEqual(got["validated"], 1)
        self.assertEqual(got["per_stage"]["cs"], [1, 1])

    def test_no_indexes_returns_none(self):
        self.assertIsNone(corpus_watch.corpus_counts(self.work))

    def test_report_paths_must_not_collide(self):
        report = os.path.join(self.work, "report")
        results = os.path.join(self.work, "counts.jsonl")
        self.assertFalse(corpus_watch.outputs_collide(self.work, report, results, None))
        self.assertTrue(corpus_watch.outputs_collide(self.work, report, results, results))
        self.assertTrue(corpus_watch.outputs_collide(self.work, self.work, results, None))

    def test_utf8_bom_indexes_and_malformed_root_types(self):
        dxbc = os.path.join(self.work, "shaders", "dxbc", "index.json")
        spirv = os.path.join(self.work, "shaders", "spirv", "index.json")
        with open(dxbc, "wb") as f:
            f.write(b"\xef\xbb\xbf" + json.dumps([{"stage": "ps"}]).encode("utf-8"))
        with open(spirv, "wb") as f:
            f.write(b"\xef\xbb\xbf" + json.dumps({"a": {"stage": "ps"}}).encode("utf-8"))
        got = corpus_watch.corpus_counts(self.work)
        self.assertEqual((got["carved"], got["validated"]), (1, 1))

        with open(dxbc, "w", encoding="utf-8", newline="\n") as f:
            f.write('"bad"')
        with self.assertRaises(ValueError):
            corpus_watch.corpus_counts(self.work)

        with open(dxbc, "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"stage": "ps"}], f)
        with open(spirv, "w", encoding="utf-8", newline="\n") as f:
            f.write("[]")
        with self.assertRaises(ValueError):
            corpus_watch.corpus_counts(self.work)

    def test_outputs_collide_checks_out_and_game(self):
        game = os.path.join(self.tmp.name, "game")
        out = os.path.join(self.tmp.name, "out")
        report = os.path.join(self.tmp.name, "report")
        results = os.path.join(self.tmp.name, "counts.jsonl")
        self.assertFalse(corpus_watch.outputs_collide(self.work, report, results, None, game=game, out=out))
        self.assertTrue(corpus_watch.outputs_collide(self.work, report, results, None, game=game, out=report))
        self.assertTrue(corpus_watch.outputs_collide(self.work, report, results, None, game=game, out=self.work))
        self.assertTrue(corpus_watch.outputs_collide(self.work, report, game, None, game=game, out=out))

    def test_cli_validation_parent_creation_and_subprocess_encoding(self):
        import contextlib
        import io
        import subprocess
        from unittest import mock

        game = os.path.join(self.tmp.name, "game")
        os.makedirs(game)
        out_dir = os.path.join(self.tmp.name, "port_out")
        report = os.path.join(self.tmp.name, "report")
        existing_dir = os.path.join(self.tmp.name, "existing_dir")
        os.makedirs(existing_dir)
        base_args = [
            "--game", game, "--work", self.work, "--out", out_dir,
            "--report", report, "--results", os.path.join(self.tmp.name, "counts.jsonl"),
        ]
        bad_cases = [
            [*base_args, "--interval", "nan", "--once"],
            [*base_args, "--interval", "inf", "--once"],
            [*base_args, "--interval", "-1", "--once"],
            [*base_args, "--once", "--jobs", "0"],
            [*base_args, "--once", "--jobs", "-2"],
            [*base_args[:-2], "--results", existing_dir, "--once"],
            [*base_args, "--once", "--log", existing_dir + os.sep],
        ]
        for argv in bad_cases:
            with self.subTest(argv=argv[-2:]):
                err = io.StringIO()
                with (
                    mock.patch.object(corpus_watch.shutil, "which") as which,
                    mock.patch.object(corpus_watch.subprocess, "run") as run,
                    contextlib.redirect_stderr(err),
                ):
                    with self.assertRaises(SystemExit) as cm:
                        corpus_watch.main(argv)
                self.assertEqual(cm.exception.code, 2)
                which.assert_not_called()
                run.assert_not_called()

        nested_results = os.path.join(self.tmp.name, "nested", "res", "counts.jsonl")
        nested_log = os.path.join(self.tmp.name, "nested", "logs", "watch.log")
        seen_kwargs = []

        def fake_run(cmd, **kwargs):
            seen_kwargs.append(kwargs)
            return subprocess.CompletedProcess(cmd, 0, "deadbeef\n", "")

        out_io = io.StringIO()
        with (
            mock.patch.object(corpus_watch.shutil, "which", return_value="/usr/bin/tool"),
            mock.patch.object(corpus_watch.subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(out_io),
        ):
            rc = corpus_watch.main([
                "--game", game, "--work", self.work, "--out", out_dir,
                "--report", report, "--results", nested_results, "--log", nested_log, "--once",
            ])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(nested_results))
        self.assertTrue(os.path.isfile(nested_log))
        self.assertGreaterEqual(len(seen_kwargs), 4)
        for kwargs in seen_kwargs:
            self.assertEqual(kwargs.get("encoding"), "utf-8")
            self.assertEqual(kwargs.get("errors"), "replace")


    def test_outputs_collide_rejects_work_corpus_indexes(self):
        import contextlib
        import io

        game = os.path.join(self.tmp.name, "game")
        os.makedirs(game, exist_ok=True)
        out_dir = os.path.join(self.tmp.name, "port_out")
        report = os.path.join(self.tmp.name, "report")
        dxbc_idx = os.path.join(self.work, "shaders", "dxbc", "index.json")
        raw_dxbc = '[{"stage": "vs"}]\n'
        with open(dxbc_idx, "w", encoding="utf-8", newline="\n") as f:
            f.write(raw_dxbc)

        candidates = (
            dxbc_idx,
            os.path.join(self.work, "shaders", "spirv", "index.json"),
            os.path.join(self.work, "dxbc", "index.json"),
            os.path.join(self.work, "spirv", "index.json"),
        )
        for cand in candidates:
            with self.subTest(index_path=cand):
                self.assertTrue(corpus_watch.outputs_collide(self.work, report, cand, None, game=game, out=out_dir))
                self.assertTrue(corpus_watch.outputs_collide(
                    self.work, report, os.path.join(self.tmp.name, "counts.jsonl"), cand, game=game, out=out_dir,
                ))

        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            corpus_watch.main([
                "--game", game, "--work", self.work, "--out", out_dir,
                "--report", report, "--results", dxbc_idx, "--once",
            ])
        self.assertEqual(cm.exception.code, 2)
        with open(dxbc_idx, encoding="utf-8") as f:
            self.assertEqual(f.read(), raw_dxbc)


if __name__ == "__main__":
    unittest.main()
