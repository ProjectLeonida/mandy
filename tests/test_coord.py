"""tools/coord.py: the client for the leonida-coord worker.

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

CLAIM = {"id": "TOOLS-CENSUS", "agent": "other", "title": "fix census", "files": ["tools/census.py"],
         "note": "", "updated": "2026-11-25T00:00:00Z", "done": None, "stale": False}


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
        if self.headers.get("authorization") != "Bearer sekrit":
            return self.reply(401, {"error": "unauthorized"})
        if self.path.startswith("/check"):
            hit = "census" in self.path
            return self.reply(409 if hit else 200, [dict(CLAIM, why=["files: tools/census.py"])] if hit else [])
        if self.path == "/claims" and self.command == "POST":
            if body["id"] == "CLASH" and not body["force"]:
                return self.reply(409, {"error": "overlap", "overlap": [CLAIM]})
            return self.reply(201, {"ok": True})
        if self.path.startswith("/claims"):
            return self.reply(200, [CLAIM] if self.command == "GET" else {"ok": True})
        if self.path.startswith("/posts"):
            rows = [{"at": "2026-11-26T10:00:00Z", "agent": "other", "id": "TOOLS-CENSUS", "text": "halfway"}]
            return self.reply(200, rows if self.command == "GET" else {"ok": True})
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
        self.assertIn("TOOLS-CENSUS", out)
        self.assertIn("overlap: files: tools/census.py", out)
        self.assertIn("agent=me", Fake.seen[0][1])

    def test_check_without_overlap_exits_zero(self):
        self.assertEqual(self.run_cli("check", "docs/x.md")[:2], (0, "no overlap\n"))

    def test_claim_sends_who_and_what(self):
        code, out, _ = self.run_cli("claim", "DOCS-COORD", "--title", "docs", "--files", "docs/a.md", "tools/")
        self.assertEqual(code, 0)
        method, path, body, auth = Fake.seen[0]
        self.assertEqual((method, path, auth), ("POST", "/claims", "Bearer sekrit"))
        self.assertEqual(body, {"id": "DOCS-COORD", "agent": "me", "title": "docs", "files": ["docs/a.md", "tools/"],
                                "note": "", "force": False})

    def test_claim_refused_on_overlap(self):
        code, out, _ = self.run_cli("claim", "CLASH")
        self.assertEqual(code, 1)
        self.assertIn("not claimed", out)
        self.assertEqual(self.run_cli("claim", "CLASH", "--force")[0], 0)

    def test_done_quotes_the_id(self):
        self.run_cli("done", "A/B", "--note", "merged")
        self.assertEqual(Fake.seen[0][:3], ("POST", "/claims/A%2FB/done", {"agent": "me", "note": "merged"}))

    def test_post_and_read(self):
        self.assertEqual(self.run_cli("post", "hello", "--id", "X")[0], 0)
        self.assertEqual(Fake.seen[0][2], {"agent": "me", "text": "hello", "id": "X"})
        code, out, _ = self.run_cli("read", "--id", "TOOLS-CENSUS")
        self.assertEqual(out, "2026-11-26 10:00  other [TOOLS-CENSUS]: halfway\n")

    def test_board(self):
        code, out, _ = self.run_cli("board", "--json")
        self.assertEqual(json.loads(out)[0]["id"], "TOOLS-CENSUS")

    def test_bad_token_and_missing_config(self):
        code, _, err = self.run_cli("board", env=dict(self.env, COORD_TOKEN="nope"))
        self.assertEqual((code, "HTTP 401" in err), (2, True))
        code, _, err = self.run_cli("board", env={"COORD_AGENT": "me"})
        self.assertEqual(code, 2)
        self.assertIn("COORD_URL", err)


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    unittest.main()
