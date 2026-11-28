#!/usr/bin/env python3
"""Export public GitHub issue and pull-request metadata without cloning a repository.

Only public GitHub REST endpoints are queried. No repository source or assets are fetched.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
FIELDS = ("id", "number", "html_url", "state", "title", "body", "created_at", "updated_at", "closed_at")
COMMENT_FIELDS = ("id", "html_url", "created_at", "updated_at", "body")
REVIEW_FIELDS = ("id", "html_url", "state", "submitted_at", "body")


def select(record, fields):
    result = {key: record.get(key) for key in fields}
    user = record.get("user")
    result["author"] = user.get("login") if isinstance(user, dict) else None
    return result


def fetch_json(url, token=None, retries=3):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "leonida-public-metadata-export",
               "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = "Bearer " + token
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            remaining = error.headers.get("X-RateLimit-Remaining")
            reset = error.headers.get("X-RateLimit-Reset")
            retry_after = error.headers.get("Retry-After")
            limited = error.code in (429, 502, 503, 504) or (error.code == 403 and remaining == "0")
            if not limited or attempt == retries:
                raise
            if retry_after:
                delay = max(1, min(60, int(float(retry_after))))
            elif remaining == "0" and reset:
                delay = max(1, min(60, int(reset) - int(time.time()) + 1))
            else:
                delay = min(2 ** attempt, 30)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def pages(endpoint, token=None):
    page = 1
    while True:
        separator = "&" if "?" in endpoint else "?"
        url = f"{API}{endpoint}{separator}per_page=100&page={page}"
        records = fetch_json(url, token)
        if not isinstance(records, list):
            raise ValueError(f"Expected list at {url}")
        yield from records
        if len(records) < 100:
            break
        page += 1


def export(repo, token=None):
    owner, name = repo.split("/", 1)
    if not owner or not name or "/" in name:
        raise ValueError("Repository must be owner/name")
    base = "/repos/" + urllib.parse.quote(owner, safe="") + "/" + urllib.parse.quote(name, safe="")
    issues, pulls = [], []
    for item in pages(base + "/issues?state=all", token):
        number = item["number"]
        entry = select(item, FIELDS)
        entry["comments"] = [select(c, COMMENT_FIELDS) for c in pages(base + f"/issues/{number}/comments", token)]
        if "pull_request" in item:
            entry["reviews"] = [select(r, REVIEW_FIELDS) for r in pages(base + f"/pulls/{number}/reviews", token)]
            pulls.append(entry)
        else:
            issues.append(entry)
    for collection in (issues, pulls):
        if len({item["id"] for item in collection}) != len(collection):
            raise ValueError("Duplicate record IDs detected; export incomplete")
    return {"repository": repo, "issues": issues, "pull_requests": pulls,
            "counts": {"issues": len(issues), "pull_requests": len(pulls),
                       "comments": sum(len(i["comments"]) for i in issues + pulls),
                       "reviews": sum(len(i.get("reviews", [])) for i in pulls)},
            "scope": "public GitHub API; PR issue comments and PR reviews; no private or deleted records"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", help="owner/name")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        # Unauthenticated requests prevent exporting a private repository using a
        # token inherited from the environment.
        data = export(args.repository)
        # Atomic replacement: API errors never leave a partial output presented as complete.
        target = args.output
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".tmp")
        try:
            with temp.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        print(json.dumps(data["counts"], sort_keys=True))
    except (ValueError, OSError, urllib.error.URLError, KeyError) as error:
        print(f"Export failed (no complete output written): {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
