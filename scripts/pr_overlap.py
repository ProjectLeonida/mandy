"""Which open pull requests change the files you are about to change, and does your branch merge with them?

    python -I scripts/pr_overlap.py [PATH ...] [--base REF] [--head REF] [--remote NAME] [--no-fetch] [--json]

GitHub's search does not find pull requests by the path they change, and with more than a thousand of them open,
reading them is not an option. This asks git instead, with no API call and no token:

- The open pull requests are the ones the remote has a `refs/pull/N/merge` for: GitHub keeps that ref while a pull
  request is open and removes it when it is merged or closed. One `git ls-remote` lists them with their heads.
- The heads are fetched into `refs/pr-overlap/N`, a namespace of this tool. Refs of pull requests that are no longer
  open are deleted, so the namespace is the open set as of the last run, and `--no-fetch` works from it offline.
- A pull request's files are the ones that differ between its head and its merge base with `--base` (default
  `origin/main`), so what `main` changed since the pull request branched is not counted as the pull request's.

With PATHs (files or folders, relative to the current folder inside the repository, spelled as git spells them) it
lists the open pull requests that change any of them: the check to run before starting on a file.

With no PATH it takes your own work: the files `--head` (default HEAD) changes against its merge base with `--base`
and, for HEAD, the files modified or untracked in the working tree. For each pull request that changes one of them
it says how `--head` stands to it:

- `clean` or `conflict`: what `git merge-tree` says about merging the two (git 2.38 or newer; committed work only),
  with the files that conflict. These are the pull requests the template's `Depends on:` line asks for.
- `same`: the pull request's head is `--head`; it is this branch's own pull request.
- `included`: the pull request's head is an ancestor of `--head`: the branch is built on it, or it is this
  branch's own pull request with commits not pushed yet.
- `unknown`: git could not try the merge (an older git, or no common history).

A pull request opened seconds ago can be missing: GitHub writes its merge ref a moment after it is opened. Titles
and authors' logins are not in git, so a pull request is shown with the author name, date and subject of its head
commit. A shallow clone has no merge base for most pull requests: they are counted as skipped, and the report says
that `git fetch --unshallow` is what it needs.

Exit 0 when it ran, whatever it found; 2 on a usage or git error. It writes refs under `refs/pr-overlap/` and
nothing else. Standard library only; needs git 2.29 or newer for `fetch --stdin`. No game files.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

NAMESPACE = "refs/pr-overlap/"
PULL_REF = re.compile(r"refs/pull/(\d+)/(head|merge)")
SHOWN_FILES = 6


class GitError(Exception):
    pass


def git(repo, *args, check=True, stdin=None):
    try:
        r = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                           input=None if stdin is None else stdin.encode("utf-8"))
    except OSError as e:  # git is not on PATH, or cannot be executed: a usage error (exit 2), not a traceback
        raise GitError(f"cannot run git: {e.strerror or e}") from None
    r.stdout = r.stdout.decode("utf-8", "replace")
    r.stderr = r.stderr.decode("utf-8", "replace")
    if check and r.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {r.stderr.strip() or r.returncode}")
    return r


def commit(repo, ref):
    """The commit `ref` names, or '' when it names none."""
    return git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False).stdout.strip()


def names(text):
    """The paths of a NUL-separated list, as `-z` prints them."""
    return [p for p in text.split("\0") if p]


def default_remote(repo, base):
    """The remote `base` is a branch of (`origin/main` -> `origin`), or `origin`."""
    name, slash, _ = base.partition("/")
    return name if slash and name in git(repo, "remote").stdout.split() else "origin"


def remote_pull_requests(repo, remote):
    """({number: head sha} of the open pull requests, how many pull request heads the remote has at all)."""
    heads, is_open = {}, set()
    for line in git(repo, "ls-remote", remote, "refs/pull/*/head", "refs/pull/*/merge").stdout.splitlines():
        sha, _, ref = line.partition("\t")
        m = PULL_REF.fullmatch(ref.strip())
        if not m:
            continue
        if m.group(2) == "head":
            heads[int(m.group(1))] = sha
        else:
            is_open.add(int(m.group(1)))
    return {n: heads[n] for n in sorted(is_open) if n in heads}, len(heads)


def local_pull_requests(repo):
    """{number: head sha} of the refs this tool keeps."""
    out = git(repo, "for-each-ref", "--format=%(objectname) %(refname)", NAMESPACE).stdout
    found = {}
    for line in out.splitlines():
        sha, _, ref = line.partition(" ")
        number = ref[len(NAMESPACE):]
        if number.isdigit():
            found[int(number)] = sha
    return dict(sorted(found.items()))


def sync(repo, remote):
    """Make refs/pr-overlap/ the remote's open pull requests. Returns (open, fetched, removed, heads on the remote)."""
    wanted, total = remote_pull_requests(repo, remote)
    have = local_pull_requests(repo)
    new = [n for n, sha in wanted.items() if have.get(n) != sha]
    stale = [n for n in have if n not in wanted]
    if new:
        if len(new) > 50:
            print(f"pr_overlap: fetching {len(new)} pull request heads from {remote}", file=sys.stderr)
        # No tags, no submodules and no FETCH_HEAD: the fetch is for this namespace alone.
        git(repo, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", "--no-write-fetch-head", "--stdin",
            remote, stdin="".join(f"+refs/pull/{n}/head:{NAMESPACE}{n}\n" for n in new))
    if stale:
        git(repo, "update-ref", "--stdin", stdin="".join(f"delete {NAMESPACE}{n}\n" for n in stale))
    return local_pull_requests(repo), len(new), len(stale), total


def changed_files(repo, base, ref):
    """The files `ref` changes against its merge base with `base`; None when the two share no history. A rename
    counts as both of its paths: whoever works on the old name is affected as well."""
    r = git(repo, "diff", "--name-only", "--no-renames", "-z", f"{base}...{ref}", check=False)
    return names(r.stdout) if r.returncode == 0 else None


def own_files(repo, base, head):
    files = changed_files(repo, base, head)
    if files is None:
        raise GitError(f"{head} and {base} have no merge base")
    work = []
    if head == "HEAD":
        work = names(git(repo, "diff", "--name-only", "--no-renames", "-z", "HEAD").stdout)
        work += names(git(repo, "ls-files", "--others", "--exclude-standard", "-z").stdout)
    return sorted(set(files) | set(work)), sorted(set(work) - set(files))


def repo_paths(top, paths):
    """PATH arguments as repository paths with forward slashes; '' is the whole repository."""
    top = os.path.realpath(top)
    cwd = os.path.realpath(os.getcwd())
    try:
        inside = os.path.normcase(os.path.commonpath([top, cwd])) == os.path.normcase(top)
    except ValueError:  # another drive
        inside = False
    found = []
    for path in paths:
        full = os.path.realpath(os.path.join(cwd if inside else top, path))
        try:
            rel = os.path.relpath(full, top)
        except ValueError:
            rel = os.pardir
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            raise GitError(f"{path} is not inside the repository {top}")
        found.append("" if rel == os.curdir else rel.replace(os.sep, "/"))
    return found


def touches(path, targets):
    return any(not t or path == t or path.startswith(t + "/") for t in targets)


def is_ancestor(repo, old, new):
    return git(repo, "merge-base", "--is-ancestor", old, new, check=False).returncode == 0


def standing(repo, head_sha, pr_sha):
    """How the branch at head_sha stands to a pull request head: (word, conflicting files)."""
    if pr_sha == head_sha:
        return "same", []
    if is_ancestor(repo, pr_sha, head_sha):
        return "included", []
    r = git(repo, "merge-tree", "--write-tree", "--name-only", "--no-messages", "-z", head_sha, pr_sha, check=False)
    if r.returncode == 0:
        return "clean", []
    if r.returncode == 1:
        return "conflict", names(r.stdout)[1:]  # the first field is the tree of the conflicted merge
    return "unknown", []


def describe(repo, sha):
    author, date, subject = (git(repo, "log", "-1", "--format=%an%x00%as%x00%s", sha).stdout.rstrip("\n")
                             .split("\0") + ["", "", ""])[:3]
    return {"author": author, "date": date, "subject": subject}


def survey(repo, base, pulls, targets, head_sha, jobs):
    """The open pull requests that change a target path, with how head_sha stands to each (None: paths only)."""
    def one(item):
        number, sha = item
        files = changed_files(repo, base, sha)
        if files is None:
            return {"number": number, "head": sha, "unrelated": True}
        shared = [p for p in files if touches(p, targets)]
        if not shared:
            return None
        found = {"number": number, "head": sha, **describe(repo, sha), "files": shared, "changed_files": len(files)}
        if head_sha:
            found["merge"], found["conflicts"] = standing(repo, head_sha, sha)
        return found

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = [r for r in pool.map(one, pulls.items()) if r]
    return [r for r in results if not r.get("unrelated")], sum(1 for r in results if r.get("unrelated"))


ORDER = {"conflict": 0, "unknown": 1, "clean": 2, "included": 3, "same": 4, None: 5}


def listing(paths, limit=SHOWN_FILES):
    return ", ".join(paths[:limit]) + (f" and {len(paths) - limit} more" if len(paths) > limit else "")


def print_report(report):
    found, by = report["pull_requests"], {}
    for pr in found:
        by.setdefault(pr.get("merge"), []).append(f"#{pr['number']}")
    print(f"{report['open']} open pull request(s) on {report['remote']}"
          + (" as of the last fetch" if report["fetched"] is None else
             f" ({report['fetched']} head(s) fetched, {report['removed']} no longer open)"))
    if report["mode"] == "paths":
        print(f"paths: {listing([p or '(the whole repository)' for p in report['paths']])}")
    else:
        print(f"{len(report['files'])} file(s) changed by {report['head']} against {report['base']}"
              + (f", {len(report['uncommitted'])} of them only in the working tree" if report["uncommitted"] else "")
              + (f": {listing(report['files'])}" if report["files"] else ""))
    if report["unrelated"]:
        print(f"{report['unrelated']} pull request(s) share no history with {report['base']} and were skipped"
              + (": this clone is shallow, and `git fetch --unshallow` brings the history a merge base needs"
                 if report["shallow"] else ""))
    print()
    for pr in found:
        word = f"{pr['merge']:<9}" if report["mode"] == "branch" else ""
        print(f"#{pr['number']:<6}{word}{pr['date']}  {pr['author']}: {pr['subject']}")
        print(f"         {listing(pr['files'])}" + (f"  ({pr['changed_files']} files in all)"
                                                    if pr["changed_files"] > len(pr["files"]) else ""))
        if pr.get("conflicts"):
            print(f"         conflicts in: {listing(pr['conflicts'])}")
    if found:
        print()
    if report["mode"] == "paths":
        print(f"{len(found)} open pull request(s) change these paths.")
        return
    if not report["files"]:
        print(f"Nothing here differs from {report['base']}. Name the files or folders you plan to change to see "
              "which open pull requests change them.")
        return
    others = [pr for pr in found if pr["merge"] not in ("same", "included")]
    print(f"{len(others)} other open pull request(s) change these files"
          + "".join(f"; {len(by[word])} {text}: {', '.join(by[word])}" for word, text in (
              ("conflict", "conflict with this branch"), ("clean", "merge cleanly with it"),
              ("unknown", "could not be merged for the check")) if word in by) + ".")
    for word, text in (("same", "is this branch's own pull request"),
                       ("included", "is already in this branch (it builds on it, or it is its own pull request)")):
        for number in by.get(word, []):
            print(f"{number} {text}.")
    if report["uncommitted"] and others:
        print("The merge check covers committed work only; the working tree's changes are not in it.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("paths", nargs="*", metavar="PATH",
                    help="files or folders to look up; with none, the files of --head and of the working tree")
    ap.add_argument("--base", default="origin/main", help="the branch pull requests merge into (default origin/main)")
    ap.add_argument("--head", default="HEAD", help="your branch, when no PATH is given (default HEAD)")
    ap.add_argument("--remote", help="remote name or URL that has the pull requests (default: the remote of --base)")
    ap.add_argument("--repo", default=".", help="repository folder (default: the current one)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="do not contact the remote: use the heads the last run fetched")
    ap.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 4), help="git processes at once")
    ap.add_argument("--json", action="store_true", help="one JSON report instead of text")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    try:
        top = git(args.repo, "rev-parse", "--show-toplevel").stdout.strip()
        if not commit(top, args.base):
            raise GitError(f"--base {args.base} is not a commit here (git fetch first?)")
        remote = args.remote or default_remote(top, args.base)
        report = {"remote": remote, "base": args.base, "fetched": None, "removed": None, "open": 0}
        # What to look up comes first: a mistyped path or branch should not cost a fetch.
        if args.paths:
            report.update(mode="paths", paths=repo_paths(top, args.paths))
            targets, head_sha = report["paths"], None
        else:
            head_sha = commit(top, args.head)
            if not head_sha:
                raise GitError(f"--head {args.head} is not a commit here")
            files, uncommitted = own_files(top, args.base, args.head)
            report.update(mode="branch", head=args.head, head_sha=head_sha, files=files, uncommitted=uncommitted)
            targets = files
        if args.no_fetch:
            pulls = local_pull_requests(top)
            if not pulls:
                raise GitError("no pull request heads here yet: run once without --no-fetch")
        else:
            pulls, report["fetched"], report["removed"], total = sync(top, remote)
            if not total:
                print(f"pr_overlap: {remote} has no pull request refs. Is it a fork? Name the repository the pull "
                      "requests are opened against with --remote (a remote name or its URL).", file=sys.stderr)
        report["open"] = len(pulls)
        found, report["unrelated"] = survey(top, args.base, pulls, targets, head_sha, args.jobs) if targets else ([], 0)
        report["pull_requests"] = sorted(found, key=lambda pr: (ORDER[pr.get("merge")], pr["number"]))
        # A shallow clone (the usual checkout of a CI job or an agent's sandbox) has no merge base to offer.
        report["shallow"] = bool(report["unrelated"]) and (
            git(top, "rev-parse", "--is-shallow-repository", check=False).stdout.strip() == "true")
    except GitError as e:
        print(f"pr_overlap: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
