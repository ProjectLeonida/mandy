"""Who works on what: a client for the leonida-coord Cloudflare Worker (worker/coord).

    python -I tools/coord.py board [--all] [--json]
    python -I tools/coord.py check [PATH ...] [--id ID] [--json]
    python -I tools/coord.py claim ID [--title TEXT] [--files PATH ...] [--note TEXT] [--force]
    python -I tools/coord.py done ID [--note TEXT]
    python -I tools/coord.py post TEXT [--id ID]
    python -I tools/coord.py read [--id ID] [--since ISO] [--json]

The coordination issue hit GitHub's 2,500-comment cap, and a claim there never said which files a change touches.
The worker keeps claims (ID, agent, files, note) and a message feed:

- `check` asks which open claims share one of these files (a path ending in `/` is a folder) or this ID, and exits 1
  when there is overlap, so a script or an agent can stop there.
- `claim` records the work. The worker refuses it on overlap (exit 1) unless `--force`.
- `done` releases it, finished or given up. A claim with no claim, post or done for 7 days goes stale and stops
  blocking others.
- `post` and `read` are the feed: questions, handoffs, "I'm halfway through X". A post with `--id` keeps that claim
  fresh.

`COORD_URL` overrides the worker's address. Sign in with GitHub: the client sends `GITHUB_TOKEN`, `GH_TOKEN` or
`gh auth token`, and the worker names you by that GitHub login. Or set `COORD_TOKEN`, the worker's shared key, and
name yourself with `COORD_AGENT` (default `git config user.name`). Standard library only; no game files.
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


def agent_name() -> str:
    name = os.environ.get("COORD_AGENT", "").strip()
    if name:
        return name
    try:
        name = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or os.environ.get("USER", "") or "agent"


def auth_token() -> str:
    """COORD_TOKEN if set, else a GitHub token: GITHUB_TOKEN, GH_TOKEN or `gh auth token`."""
    for name in ("COORD_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def call(method: str, path: str, body: dict | None = None, query: dict | None = None) -> tuple[int, object]:
    """One request to the worker; returns (status, decoded JSON). 409 is an answer, not an error."""
    base = os.environ.get("COORD_URL", DEFAULT_URL).rstrip("/")
    token = auth_token()
    if not base or not token:
        raise CoordError("set COORD_URL, and COORD_TOKEN or a GitHub token (see docs/coordination.md)")
    url = base + path
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "authorization": f"Bearer {token}", "content-type": "application/json", "user-agent": "leonida-coord"})
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


def claim_line(c: dict) -> str:
    flags = " [done]" if c.get("done") else " [stale]" if c.get("stale") else ""
    head = f"{c['id']}  {c['agent']}  {c.get('updated', '')[:10]}{flags}"
    lines = [head]
    if c.get("title"):
        lines.append(f"    {c['title']}")
    if c.get("files"):
        lines.append("    files: " + ", ".join(c["files"]))
    if c.get("why"):
        lines.append("    overlap: " + "; ".join(c["why"]))
    if c.get("note"):
        lines.append(f"    note: {c['note']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("board", help="list open claims")
    p.add_argument("--all", action="store_true", help="include done and stale claims")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("check", help="open claims that share these files or this ID; exit 1 on overlap")
    p.add_argument("paths", nargs="*")
    p.add_argument("--id", default="")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("claim", help="claim work")
    p.add_argument("id")
    p.add_argument("--title", default="")
    p.add_argument("--files", nargs="*", default=[])
    p.add_argument("--note", default="")
    p.add_argument("--force", action="store_true", help="claim despite overlap you have agreed on")
    p = sub.add_parser("done", help="release a claim")
    p.add_argument("id")
    p.add_argument("--note", default="")
    p = sub.add_parser("post", help="post to the feed")
    p.add_argument("text")
    p.add_argument("--id", default="")
    p = sub.add_parser("read", help="read the feed")
    p.add_argument("--id", default="")
    p.add_argument("--since", default="")
    p.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    me = agent_name()

    try:
        if args.cmd == "board":
            _, rows = call("GET", "/claims", query={"all": "1" if args.all else ""})
            if args.json:
                print(json.dumps(rows, indent=1))
            else:
                print("\n".join(claim_line(c) for c in rows) or "no open claims")
            return 0
        if args.cmd == "check":
            if not args.paths and not args.id:
                ap.error("check needs a path or --id")
            status, rows = call("GET", "/check", query={"files": ",".join(args.paths), "id": args.id, "agent": me})
            if args.json:
                print(json.dumps(rows, indent=1))
            else:
                print("\n".join(claim_line(c) for c in rows) or "no overlap")
            return 1 if status == 409 else 0
        if args.cmd == "claim":
            status, res = call("POST", "/claims", {"id": args.id, "agent": me, "title": args.title,
                                                   "files": args.files, "note": args.note, "force": args.force})
            if status == 409:
                print(f"not claimed, {args.id} overlaps:")
                print("\n".join(claim_line(c) for c in res.get("overlap", [])))
                return 1
            print(f"claimed {args.id} as {res.get('agent') or me}")
            return 0
        if args.cmd == "done":
            call("POST", f"/claims/{urllib.parse.quote(args.id, safe='')}/done", {"agent": me, "note": args.note})
            print(f"released {args.id}")
            return 0
        if args.cmd == "post":
            call("POST", "/posts", {"agent": me, "text": args.text, "id": args.id})
            return 0
        if args.cmd == "read":
            _, rows = call("GET", "/posts", query={"id": args.id, "since": args.since})
            if args.json:
                print(json.dumps(rows, indent=1))
            else:
                for r in rows:
                    about = f" [{r['id']}]" if r.get("id") else ""
                    print(f"{r['at'][:16].replace('T', ' ')}  {r['agent']}{about}: {r['text']}")
            return 0
    except CoordError as err:
        print(f"coord: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
