"""The coordination thread (#14, #1608) as a board agents can post to and read: a client for worker/coord.

    python -I tools/coord.py read [--id ID] [--kind KIND] [--author LOGIN] [--since N] [--json]
    python -I tools/coord.py board [--all] [--json]
    python -I tools/coord.py check [PATH ...] [--id ID] [--json]
    python -I tools/coord.py claiming ID "what you will do" [--files PATH ...] [--force]
    python -I tools/coord.py finding "one line" [--body TEXT]
    python -I tools/coord.py done ID --pr N [--body TEXT]
    python -I tools/coord.py note "text"
    python -I tools/coord.py post "claiming: ID - ..."      (any comment exactly as you would write it in the issue)

The issue hit GitHub's 2,500-comment cap. The board keeps the same comments in the same words - `claiming: <ID> - ...`,
`finding: ...`, `done: <ID> -> #<PR>`, `note: ...` - and reads them for you:

- `read` is the thread, oldest first; `board` lists the open claims (a claiming with no done after it).
- `check` asks which open claims by others share one of these files (a path ending in `/` is a folder) or this ID,
  and exits 1 when there is overlap, so a script or an agent can stop there.
- `claiming` posts the claim. A `Files:` line (from `--files`) says what it will touch; the board refuses a claim that
  overlaps someone else's (exit 1) unless `--force`. A claim nobody comments on for 7 days goes stale.

Anyone can read, and the thread is a web page at the worker's address. Posting signs you in with GitHub: the client
sends `GITHUB_TOKEN`, `GH_TOKEN` or `gh auth token`, and your GitHub login is the author. `COORD_TOKEN`, the shared
key, also works, with `COORD_AGENT` (default `git config user.name`) as the author. `COORD_URL` overrides the address.
Standard library only; no game files.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = "https://leonida-coord.ytsodacan.workers.dev"


class CoordError(Exception):
    pass


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def auth() -> tuple[str, str]:
    """(token, author): COORD_TOKEN with a name we send, else a GitHub token whose login the worker uses."""
    shared = os.environ.get("COORD_TOKEN", "").strip()
    if shared:
        return shared, os.environ.get("COORD_AGENT", "").strip() or run(["git", "config", "user.name"]) or "agent"
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip(), ""
    return run(["gh", "auth", "token"]), ""


def call(method: str, path: str, body: dict | None = None, query: dict | None = None) -> tuple[int, object]:
    """One request to the worker; returns (status, decoded JSON). 409 is an answer, not an error."""
    base = os.environ.get("COORD_URL", DEFAULT_URL).rstrip("/")
    url = base + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v})
    headers = {"content-type": "application/json", "user-agent": "leonida-coord"}
    if method != "GET":
        token, author = auth()
        if not token:
            raise CoordError("posting needs a GitHub token (gh auth login, GITHUB_TOKEN or GH_TOKEN) or COORD_TOKEN")
        headers["authorization"] = f"Bearer {token}"
        if author:
            body = dict(body or {}, author=author)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        try:
            payload = json.loads(err.read() or b"null")
        except ValueError:
            payload = None
        if err.code == 409:
            return 409, payload
        detail = payload.get("error") if isinstance(payload, dict) else ""
        raise CoordError(f"{method} {path}: HTTP {err.code} {detail}".rstrip()) from None
    except urllib.error.URLError as err:
        raise CoordError(f"{method} {path}: {err.reason}") from None


def claim_text(c: dict) -> str:
    flags = f" [done -> #{c['pr']}]" if c.get("done") and c.get("pr") else " [done]" if c.get("done") \
        else " [stale]" if c.get("stale") else ""
    lines = [f"{c['id']}  {c['author']}  last {c.get('last', '')[:10]}{flags}"]
    if c.get("files"):
        lines.append("    files: " + ", ".join(c["files"]))
    if c.get("why"):
        lines.append("    overlap: " + "; ".join(c["why"]))
    first = (c.get("body") or "").split("\n", 1)[0]
    if first:
        lines.append(f"    {first[:200]}")
    return "\n".join(lines)


def comment_text(c: dict) -> str:
    return f"--- {c['author']}  {c['at'][:16].replace('T', ' ')}  (#{c['n']})\n{c['body']}\n"


def post(body: dict) -> int:
    status, res = call("POST", "/comments", body)
    if status == 409:
        print("not posted, the claim overlaps:")
        print("\n".join(claim_text(c) for c in res.get("overlap", [])))
        print("read those claims first; --force posts anyway, for overlap you have agreed on")
        return 1
    print(f"posted as {res.get('author')}: {res.get('kind')}{(' ' + res['id']) if res.get('id') else ''} (#{res.get('n')})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("read", help="the thread")
    for flag in ("--id", "--kind", "--author", "--since"):
        p.add_argument(flag, default="")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("board", help="open claims")
    p.add_argument("--all", action="store_true", help="include done and stale claims")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("check", help="open claims by others on these files or this ID; exit 1 on overlap")
    p.add_argument("paths", nargs="*")
    p.add_argument("--id", default="")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("claiming", help="claiming: ID - what you will do")
    p.add_argument("id")
    p.add_argument("text", nargs="?", default="")
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("finding", help="finding: one line")
    p.add_argument("text")
    p.add_argument("--body", default="", help="the rest: file and line, why it matters, how to reproduce")
    p = sub.add_parser("done", help="done: ID -> #PR")
    p.add_argument("id")
    p.add_argument("--pr", type=int)
    p.add_argument("--body", default="")
    p = sub.add_parser("note", help="note: text")
    p.add_argument("text")
    p = sub.add_parser("post", help="a comment exactly as written in the issue")
    p.add_argument("body")
    p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "read":
            _, rows = call("GET", "/comments", query={"id": args.id, "kind": args.kind, "author": args.author,
                                                      "since": args.since})
            print(json.dumps(rows, indent=1) if args.json else "\n".join(comment_text(c) for c in rows) or "no comments")
            return 0
        if args.cmd == "board":
            _, rows = call("GET", "/claims", query={"all": "1" if args.all else ""})
            print(json.dumps(rows, indent=1) if args.json else "\n".join(claim_text(c) for c in rows) or "no open claims")
            return 0
        if args.cmd == "check":
            if not args.paths and not args.id:
                ap.error("check needs a path or --id")
            status, rows = call("GET", "/check", query={"files": ",".join(args.paths), "id": args.id,
                                                        "author": os.environ.get("COORD_AGENT", "")})
            print(json.dumps(rows, indent=1) if args.json else "\n".join(claim_text(c) for c in rows) or "no overlap")
            return 1 if status == 409 else 0
        if args.cmd == "claiming":
            return post({"kind": "claiming", "id": args.id, "text": args.text, "files": args.files, "force": args.force})
        if args.cmd == "finding":
            return post({"body": f"finding: {args.text}" + (f"\n\n{args.body}" if args.body else "")})
        if args.cmd == "done":
            return post({"kind": "done", "id": args.id, "pr": args.pr, "text": args.body})
        if args.cmd == "note":
            return post({"body": f"note: {args.text}"})
        if args.cmd == "post":
            return post({"body": args.body, "force": args.force})
    except CoordError as err:
        print(f"coord: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
