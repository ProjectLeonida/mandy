"""tools/coord.py: the client for the coordination board (worker/coord).

    python -I tests/test_coord.py

A small local HTTP server stands in for the worker, so nothing leaves the machine. Standard library only; no game
files.
"""
import contextlib
import http.server
import io
import json
import os
import sys
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import coord  # noqa: E402

CLAIM = {"id": "TOOLS-CENSUS", "author": "other", "files": ["tools/census.py"], "n": 3, "body": "claiming: TOOLS-CENSUS - fix it",
         "last": "2026-11-25T00:00:00Z", "done": None, "pr": None, "stale": False}
COMMENT = {"n": 4, "at": "2026-11-26T10:00:00Z", "author": "other", "kind": "note", "id": None, "body": "note: halfway"}


class Fake(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_any(self):
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        Fake.seen.append((self.command, self.path, body, self.headers.get("authorization")))
        if self.command == "POST" and self.headers.get("authorization") != "Bearer sekrit":
            return self.reply(401, {"error": "unauthorized"})
        if self.path.startswith("/check"):
            hit = "census" in self.path
            return self.reply(409 if hit else 200, [dict(CLAIM, why=["files: tools/census.py"])] if hit else [])
        if self.path == "/comments" and self.command == "POST":
            if body.get("id") == "CLASH" and not body.get("force"):
                return self.reply(409, {"error": "overlap", "overlap": [CLAIM]})
            return self.reply(201, {"ok": True, "n": 5, "author": body.get("author") or "gh-login",
                                    "kind": body.get("kind") or body["body"].split(":")[0], "id": body.get("id")})
        if self.path.startswith("/claims"):
            return self.reply(200, [CLAIM])
        if self.path.startswith("/comments"):
            return self.reply(200, [COMMENT])
        return self.reply(404, {"error": "not found"})

    do_GET = do_POST = handle_any


class CoordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.env = {"COORD_URL": f"http://127.0.0.1:{cls.server.server_port}/", "COORD_TOKEN": "sekrit",
                   "COORD_AGENT": "me"}

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def run_cli(self, *argv, env=None):
        Fake.seen.clear()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env or self.env), contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            code = coord.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_check_reports_overlap_and_exits_one(self):
        code, out, _ = self.run_cli("check", "tools/census.py")
        self.assertEqual(code, 1)
        self.assertIn("overlap: files: tools/census.py", out)

    def test_check_without_overlap_exits_zero(self):
        self.assertEqual(self.run_cli("check", "docs/x.md")[:2], (0, "no overlap\n"))

    def test_reading_needs_no_token(self):
        code, out, _ = self.run_cli("read", env={"COORD_URL": self.env["COORD_URL"]})
        self.assertEqual(out, "--- other  2026-11-26 10:00  (#4)\nnote: halfway\n\n")
        self.assertIsNone(Fake.seen[0][3])

    def test_claiming_sends_the_claim(self):
        code, out, _ = self.run_cli("claiming", "DOCS-COORD", "write the docs", "--files", "docs/a.md", "tools/")
        self.assertEqual(code, 0)
        method, path, body, auth = Fake.seen[0]
        self.assertEqual((method, path, auth), ("POST", "/comments", "Bearer sekrit"))
        self.assertEqual(body, {"kind": "claiming", "id": "DOCS-COORD", "text": "write the docs",
                                "files": ["docs/a.md", "tools/"], "force": False, "author": "me"})
        self.assertIn("posted as me: claiming DOCS-COORD", out)

    def test_claim_refused_on_overlap(self):
        code, out, _ = self.run_cli("claiming", "CLASH")
        self.assertEqual(code, 1)
        self.assertIn("not posted", out)
        self.assertEqual(self.run_cli("claiming", "CLASH", "--force")[0], 0)

    def test_the_other_comments_use_the_thread_words(self):
        self.run_cli("finding", "census drops a row", "--body", "tools/census.py:12")
        self.assertEqual(Fake.seen[0][2]["body"], "finding: census drops a row\n\ntools/census.py:12")
        self.run_cli("done", "DOCS-COORD", "--pr", "2151")
        self.assertEqual(Fake.seen[0][2], {"kind": "done", "id": "DOCS-COORD", "pr": 2151, "text": "", "author": "me"})
        self.run_cli("note", "hi")
        self.assertEqual(Fake.seen[0][2]["body"], "note: hi")
        self.run_cli("post", "claiming: X-Y - raw")
        self.assertEqual(Fake.seen[0][2]["body"], "claiming: X-Y - raw")

    def test_board(self):
        code, out, _ = self.run_cli("board", "--json")
        self.assertEqual(json.loads(out)[0]["id"], "TOOLS-CENSUS")

    def test_github_token_sends_no_author(self):
        code, out, _ = self.run_cli("note", "hi", env={"COORD_URL": self.env["COORD_URL"], "GH_TOKEN": "sekrit"})
        self.assertEqual(Fake.seen[0][2], {"body": "note: hi"})
        self.assertIn("posted as gh-login", out)

    def test_bad_token_and_no_token(self):
        code, _, err = self.run_cli("note", "x", env=dict(self.env, COORD_TOKEN="nope"))
        self.assertEqual((code, "HTTP 401" in err), (2, True))
        with mock.patch.object(coord.subprocess, "run", side_effect=OSError):
            code, _, err = self.run_cli("note", "x", env={"COORD_URL": self.env["COORD_URL"]})
        self.assertEqual(code, 2)
        self.assertIn("GitHub token", err)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    unittest.main()
