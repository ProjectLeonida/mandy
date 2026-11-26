"""Which tests exercise the files you changed, and the commands that run them.

    python -I scripts/tests_for.py [PATH ...] [--base REF] [--head REF] [--all] [--json]

A reviewer runs "the pull request's own tests" and a contributor the tests of what they changed, but tests are
found by name here, and a helper that many tools import has no test of its name. This reads the repository and
answers for each PATH (files or folders), or with none for the files your branch changes: what `--head` (default
HEAD) changes against its merge base with `--base` (default origin/main) and, for HEAD, what is modified or
untracked in the working tree.

For every such file it lists the tests under `tests/` that reach it, each for one reason:

- `names it`: the test imports the file as a module or names its path (`"tools/x.py"`, `ROOT, "viewer",
  "page.html"`, `ROOT / "viewer" / "page.html"`);
- `imports`: a module the test imports reaches the file through imports, with the module it goes through;
- `folder`: the test names the folder the file is in and neither imports from it nor names a file in it, so
  it probably reads what is there (`ROOT / "viewer"`, then every `*.html`); for a Markdown file these are
  `tests/test_docs_*.py` and `tests/test_i18n_index.py`, which CI runs for every documentation change;
- `fixture`: the file is under a `fixtures` folder of the tests and a test beside that folder quotes its file
  name or its stem (`"ps_case"` for `tests/semantic/fixtures/ps_case.asm`);
- `ctest`: under `runtime/`, the ctest tests whose program is built from the file according to
  `runtime/CMakeLists.txt`: as a source, through a header that a source includes, or through a library the
  program links. `foreach` loops and `set()` lists are followed; a target that a function or a generator
  expression builds is not, so take this list as the least there is to run.

Then the commands, split the way CI splits the tests: `python -I` for a standalone file, pytest for a file that
imports pytest or has no `__main__` block, the semantic runner for `tests/semantic`, `node --test` and
`ctest -R`. Tests that `names it` and changed tests always get their commands; the wider reasons do when there
are at most 20 of them or with `--all`.

It reads files and asks git for the changed ones; it runs no test and writes nothing. What a test reaches is read
from its text (imports and quoted paths), so a path that is built at run time is not seen. Exit 0, or 2 on a usage
or git error. Standard library only; no game files.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_DIRS = ("tools", "scripts", "viewer", "tests", "tests/semantic", "")
IMPORT = re.compile(r"^[ \t]*(?:import[ \t]+([\w.]+(?:[ \t]*,[ \t]*[\w.]+)*)|from[ \t]+([\w.]+)[ \t]+import\b)", re.M)
LITERAL = re.compile(r'''"([^"\\\n]*)"|'([^'\\\n]*)\'''')
# A folder named alone ("docs") is a path of this repository only next to its root: ROOT, REPO_ROOT, __dirname.
# Upper case only: `root / "tools"` is usually a temporary checkout, and "report" is no repository.
PATH_CONTEXT = re.compile(r"\b(?:[A-Z]+_)?(?:ROOT|REPO)(?:_[A-Z]+)?\b|__dirname|__file__")
PYTEST_MARKERS = re.compile(r"^import pytest|^ *HAVE_NUMPY =", re.M)   # PYTEST_FILE_MARKERS of ci.yml
SOURCE = re.compile(r"\.(?:c|cc|cpp|cxx|h|hpp|inc|asm|rc|def)$", re.I)
INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\n]+)"', re.M)
COMMAND = re.compile(r"(?m)^[ \t]*([A-Za-z_]\w*)[ \t]*\(")
WIDE_LIMIT = 20
REASONS = ("names it", "imports", "folder", "fixture")


class GitError(Exception):
    pass


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def listing(root, folder):
    try:
        return sorted(os.listdir(os.path.join(root, folder)))
    except OSError:
        return []


class Repository:
    """The modules and tests of a checkout, and what each one names."""

    def __init__(self, root=ROOT):
        self.root = root
        entries = [name for name in listing(root, "") if name != ".git"]
        self.top = {name for name in entries if os.path.isdir(os.path.join(root, name))}
        # Every path this looks for starts with a top-level folder or is a file beside them, so a line that
        # holds none of those names is not looked at twice.
        self.hint = re.compile("|".join(re.escape(name) for name in sorted(entries, key=len, reverse=True)
                                        if name in self.top or "." in name) or r"(?!)")
        self.modules = defaultdict(list)        # module name -> its files, in MODULE_DIRS order
        for folder in MODULE_DIRS:
            for name in listing(root, folder):
                if name.endswith(".py"):
                    self.modules[name[:-3]].append(f"{folder}/{name}" if folder else name)
        self.module_files = sorted({path for paths in self.modules.values() for path in paths})
        self.tests =[f"tests/{name}" for name in listing(root, "tests")
                      if re.fullmatch(r"test_.*\.(?:py|cjs|mjs)", name)]
        for base, _, names in os.walk(os.path.join(root, "tests", "semantic")):
            rel = os.path.relpath(base, root).replace(os.sep, "/")
            self.tests += [f"{rel}/{name}" for name in sorted(names) if re.fullmatch(r"test.*\.py", name)]
        self.tests.sort()
        self.names, self.folders, self.listings = {}, {}, {}
        for path in sorted({*self.module_files, *self.tests}):
            self.names[path], self.folders[path] = self.references(path, read(os.path.join(root, path)))

    def kind(self, path):
        """`dir`, `file` or None for a repository path spelled exactly as on disk. Windows and macOS would open
        `Docs/World.md` too, and the answer must not depend on where the tool runs."""
        folder = ""
        for part in path.split("/"):
            if folder not in self.listings:
                self.listings[folder] = set(listing(self.root, folder))
            if part not in self.listings[folder]:
                return None
            folder = f"{folder}/{part}" if folder else part
        return "dir" if os.path.isdir(os.path.join(self.root, path)) else "file"

    def module(self, name, importer):
        """The file `import name` loads in `importer`: the one beside it, as Python looks there first, else the
        first in MODULE_DIRS order."""
        found = self.modules.get(name, ())
        beside = f"{os.path.dirname(importer)}/{name}.py".lstrip("/")
        return beside if beside in found else found[0] if found else None

    def references(self, path, text):
        """(files, folders) of the repository that `text` names: imported modules, quoted paths and paths joined
        from quoted segments. A folder on a `sys.path` line is where modules come from, not something read."""
        files, folders = set(), set()

        def add(found, kind, line):
            if kind == "file":
                files.add(found)
            elif "sys.path" not in line:
                folders.add(found)

        if path.endswith(".py"):
            for m in IMPORT.finditer(text):
                for name in re.split(r"[ \t]*,[ \t]*", m.group(1) or m.group(2)):
                    module = self.module(name, path)
                    if module:
                        files.add(module)
        for line in text.splitlines():
            if not self.hint.search(line):
                continue
            run, end = [], None
            for m in [*LITERAL.finditer(line), None]:
                segment = (m.group(1) if m.group(1) is not None else m.group(2)).strip("/") if m else None
                joined = m and end is not None and re.fullmatch(r"\s*[,/]\s*", line[end:m.start()])
                if run and not joined:
                    self.resolve(run, line, add)
                    run = []
                if m and segment:
                    run.append(segment)
                if m:
                    end = m.end()
        return files, folders

    def resolve(self, run, line, add):
        """Report the repository paths in a run of quoted segments that are joined by `,` or `/`: the longest
        join that exists, starting at each segment in turn. A path starts with a top-level folder, or is a file
        beside it (`make_game.py`). A top-level folder named alone counts only on a line that names the
        repository root, since `os.path.join(tmp, "game")` is somebody's temporary folder."""
        i = 0
        while i < len(run):
            for j in range(len(run) - 1, i - 1, -1):
                candidate = os.path.normpath("/".join(run[i:j + 1])).replace(os.sep, "/")
                kind = None if candidate.startswith("..") or os.path.isabs(candidate) else self.kind(candidate)
                if kind is None:
                    continue
                if "/" in candidate:
                    known = True
                elif kind == "dir":
                    known = PATH_CONTEXT.search(line) is not None
                else:
                    known = "." in candidate
                if known:
                    add(candidate, kind, line)
                    i = j
                    break
            i += 1

    def reached(self, test):
        """{file: the module it is reached through, or None when the test names it} for one test."""
        seen = {name: None for name in self.names[test]}
        todo = deque((name, name) for name in sorted(self.names[test]))
        while todo:
            name, through = todo.popleft()
            for inner in sorted(self.names.get(name, ())):
                if inner not in seen and inner != test:
                    seen[inner] = through
                    todo.append((inner, through))
        return seen


def runner_of(root, test):
    """python, pytest, semantic or node: how CI runs a test file."""
    if test.endswith((".cjs", ".mjs")):
        return "node"
    if test.startswith("tests/semantic/"):
        return "semantic"
    text = read(os.path.join(root, test))
    return "pytest" if PYTEST_MARKERS.search(text) or "__main__" not in text else "python"


def cmake_commands(text):
    """[(command, [argument])] of a CMake file, comments dropped."""
    text = re.sub(r"(?m)(^|\s)#.*$", r"\1", text)
    found = []
    for m in COMMAND.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += (text[i] == "(") - (text[i] == ")")
            i += 1
        found.append((m.group(1).lower(), [a.strip('"') for a in re.findall(r'"[^"]*"|[^\s()"]+', text[m.end():i - 1])]))
    return found


class Runtime:
    """What runtime/CMakeLists.txt builds: which program each source goes into and which ctest runs it."""

    def __init__(self, root=ROOT):
        self.root = root
        self.sources, self.links, self.tests = defaultdict(set), defaultdict(set), []
        self.variables = {"CMAKE_CURRENT_SOURCE_DIR": ["."], "CMAKE_SOURCE_DIR": ["."]}
        self.interpret(cmake_commands(read(os.path.join(root, "runtime", "CMakeLists.txt"))))
        self.includers = defaultdict(set)
        for base, _, names in os.walk(os.path.join(root, "runtime")):
            for name in names:
                if SOURCE.search(name):
                    path = os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/")
                    for inc in INCLUDE.findall(read(os.path.join(base, name))):
                        target = self.include(path, inc)
                        if target:
                            self.includers[target].add(path)

    def include(self, includer, name):
        for folder in (os.path.dirname(includer), "runtime/src", "runtime/shims", "runtime/test", "runtime"):
            candidate = os.path.normpath(os.path.join(folder, name)).replace(os.sep, "/")
            if os.path.isfile(os.path.join(self.root, candidate)):
                return candidate
        return None

    def expand(self, args):
        out = []
        for arg in args:
            whole = re.fullmatch(r"\$\{(\w+)\}", arg)
            if whole:
                out += self.variables.get(whole.group(1), [])
            else:
                out.append(re.sub(r"\$\{(\w+)\}", lambda m: (self.variables.get(m.group(1)) or [m.group(0)])[0]
                                  if len(self.variables.get(m.group(1), [])) <= 1 else m.group(0), arg))
        return out

    def interpret(self, commands):
        i = 0
        while i < len(commands):
            name, args = commands[i]
            i += 1
            if name == "foreach" and args:
                depth, j = 1, i
                while j < len(commands) and depth:
                    depth += (commands[j][0] == "foreach") - (commands[j][0] == "endforeach")
                    j += 1
                body, i = commands[i:j - 1], j
                items = self.expand(args[1:])
                if items[:2] == ["IN", "ITEMS"]:
                    items = items[2:]
                elif items[:2] == ["IN", "LISTS"]:
                    items = [value for name_ in items[2:] for value in self.variables.get(name_, [])]
                elif items[:1] == ["RANGE"]:
                    items = []
                saved = self.variables.get(args[0])
                for item in items:
                    self.variables[args[0]] = [item]
                    self.interpret(body)
                if saved is None:
                    self.variables.pop(args[0], None)
                else:
                    self.variables[args[0]] = saved
                continue
            args = self.expand(args)
            if name == "set" and args:
                self.variables[args[0]] = [a for a in args[1:] if a not in ("PARENT_SCOPE", "CACHE")]
            elif name == "list" and len(args) > 2 and args[0] == "APPEND":
                self.variables.setdefault(args[1], []).extend(args[2:])
            elif name in ("add_executable", "add_library", "target_sources") and args:
                self.sources[args[0]] |= {self.path(a) for a in args[1:] if SOURCE.search(a)}
            elif name == "target_link_libraries" and args:
                for library in args[1:]:
                    self.links[library].add(args[0])
            elif name == "add_test" and "NAME" in args and "COMMAND" in args:
                command = args[args.index("COMMAND") + 1:]
                programs = {re.sub(r"^\$<TARGET_FILE:(\w+)>$", r"\1", a) for a in command}
                self.tests.append((args[args.index("NAME") + 1], programs,
                                   {self.path(a) for a in command if a.endswith(".py")}))

    def path(self, name):
        return os.path.normpath(os.path.join("runtime", name)).replace(os.sep, "/")

    def tests_for(self, path):
        """The ctest tests whose program is built from `path`, in CMakeLists order."""
        files, todo = set(), [path]
        while todo:
            name = todo.pop()
            if name not in files:
                files.add(name)
                todo += self.includers.get(name, ())
        targets, todo = set(), [t for t, sources in self.sources.items() if sources & files]
        while todo:
            name = todo.pop()
            if name not in targets:
                targets.add(name)
                todo += self.links.get(name, ())
        found = []
        for name, programs, scripts in self.tests:
            if (programs & targets or path in scripts) and name not in found and "${" not in name:
                found.append(name)
        return found


def changed_files(root, base, head):
    def git(*args):
        try:
            r = subprocess.run(["git", "-C", root, *args], capture_output=True)
        except OSError as e:
            raise GitError(f"cannot run git: {e.strerror or e}") from None
        if r.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip() or r.returncode}")
        return [p for p in r.stdout.decode("utf-8", "replace").split("\0") if p]

    files = git("diff", "--name-only", "--no-renames", "-z", f"{base}...{head}")
    if head == "HEAD":
        files += git("diff", "--name-only", "--no-renames", "-z", "HEAD")
        files += git("ls-files", "--others", "--exclude-standard", "-z")
    return sorted(set(files))


def expand_targets(root, paths):
    """PATH arguments as repository files: a folder stands for the files in it."""
    top, files = os.path.realpath(root), []
    cwd = os.path.realpath(os.getcwd())
    try:
        inside = os.path.normcase(os.path.commonpath([top, cwd])) == os.path.normcase(top)
    except ValueError:
        inside = False
    for path in paths:
        full = os.path.realpath(os.path.join(cwd if inside else top, path))
        try:
            rel = os.path.relpath(full, top)
        except ValueError:
            rel = os.pardir
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            raise GitError(f"{path} is not inside the repository {top}")
        if os.path.isdir(full):
            for base, dirs, names in os.walk(full):
                dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
                files += [os.path.relpath(os.path.join(base, n), top).replace(os.sep, "/") for n in sorted(names)]
        else:
            files.append(rel.replace(os.sep, "/"))
    return sorted(set(files))


def survey(root, files):
    """For each file: {"file", "is_test", "tests": [{"test", "reason", "through"}], "ctest": [names]}."""
    repo = Repository(root)
    reached = {test: repo.reached(test) for test in repo.tests}
    docs = [t for t in repo.tests if re.fullmatch(r"tests/test_(?:docs_.*|i18n_index)\.py", t)]
    runtime = None
    report = []
    for path in files:
        entry = {"file": path, "is_test": path in repo.tests, "tests": [], "ctest": []}
        # tests/semantic/fixtures/disasm/x.s belongs to the tests in tests/semantic.
        beside = path.split("/fixtures/")[0] if path.startswith("tests/") and "/fixtures/" in path else None
        stem, suffix = os.path.splitext(os.path.basename(path))
        quoted = re.compile(rf"""["']{re.escape(stem)}(?:{re.escape(suffix)})?["']""")
        for test in repo.tests:
            if test == path:
                continue
            if path in reached[test]:
                through = reached[test][path]
                entry["tests"].append({"test": test, "reason": "imports" if through else "names it",
                                       "through": through})
            elif (path.endswith(".md") and test in docs) or (
                    os.path.dirname(path) in repo.folders[test]
                    # A folder the test gets modules or named files from is not one it reads as a whole.
                    and not any(os.path.dirname(f) == os.path.dirname(path) for f in reached[test])):
                entry["tests"].append({"test": test, "reason": "folder", "through": None})
            elif os.path.dirname(test) == beside and quoted.search(read(os.path.join(root, test))):
                entry["tests"].append({"test": test, "reason": "fixture", "through": None})
        if path.startswith("runtime/"):
            runtime = runtime or Runtime(root)
            entry["ctest"] = runtime.tests_for(path)
        entry["tests"].sort(key=lambda t: (REASONS.index(t["reason"]), t["test"]))
        report.append(entry)
    return report


def commands(root, report, everything):
    """The commands that run the tests of a report, and how many wider tests were left out of them."""
    chosen, wide, ctest = [], [], []
    for entry in report:
        if entry["is_test"]:
            chosen.append(entry["file"])
        chosen += [t["test"] for t in entry["tests"] if t["reason"] == "names it"]
        wide += [t["test"] for t in entry["tests"] if t["reason"] != "names it"]
        ctest += [name for name in entry["ctest"] if name not in ctest]
    chosen = sorted(set(chosen))
    wide = sorted(set(wide) - set(chosen))
    left_out = 0 if everything or len(wide) <= WIDE_LIMIT else len(wide)
    tests = chosen if left_out else sorted(set(chosen) | set(wide))
    by = defaultdict(list)
    for test in tests:
        by[runner_of(root, test)].append(test)
    lines = [f"python -I {test}" for test in by["python"]]
    if by["pytest"]:
        lines.append("python -m pytest " + " ".join(by["pytest"]) + "   # needs tests/requirements.txt")
    if len(by["semantic"]) > WIDE_LIMIT:
        lines.append(f"python -I scripts/run_semantic_tests.py tests/semantic   # {len(by['semantic'])} of its files")
    else:
        lines += [f"python -I scripts/run_semantic_tests.py tests/semantic -p {os.path.basename(test)}"
                  for test in by["semantic"]]
    if by["node"]:
        lines.append("node --test " + " ".join(by["node"]))
    if len(ctest) > 2 * WIDE_LIMIT:
        lines.append(f"ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure   # {len(ctest)} of its tests")
    elif ctest:
        lines.append('ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure -R "^('
                     + "|".join(ctest) + ')$"')
    return lines, left_out


def listed(names, limit=8):
    return ", ".join(names[:limit]) + (f" and {len(names) - limit} more" if len(names) > limit else "")


def print_report(report, lines, left_out):
    for entry in report:
        print(entry["file"] + ("  (a test)" if entry["is_test"] else ""))
        for reason in REASONS:
            tests = [t for t in entry["tests"] if t["reason"] == reason]
            if not tests:
                continue
            shown = [t["test"] + (f" (through {t['through']})" if t["through"] else "") for t in tests]
            print(f"  {reason} ({len(tests)}): {listed(shown)}")
        if entry["ctest"]:
            print(f"  ctest ({len(entry['ctest'])}): {listed(entry['ctest'])}")
        if not entry["tests"] and not entry["ctest"] and not entry["is_test"]:
            print("  no test names it")
    print()
    if lines:
        print("Run:")
        for line in lines:
            print("  " + line)
    else:
        print("Nothing to run was found for these files.")
    if left_out:
        print(f"{left_out} more test(s) reach these files through imports or folders; --all adds their commands.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("paths", nargs="*", metavar="PATH",
                    help="files or folders; with none, the files of --head and of the working tree")
    ap.add_argument("--base", default="origin/main", help="the branch to compare with (default origin/main)")
    ap.add_argument("--head", default="HEAD", help="your branch, when no PATH is given (default HEAD)")
    ap.add_argument("--all", action="store_true", help="commands for every test found, however many")
    ap.add_argument("--json", action="store_true", help="one JSON report instead of text")
    ap.add_argument("--repo", default=ROOT, help="repository folder (default: the one this script is in)")
    args = ap.parse_args(argv)
    try:
        files = expand_targets(args.repo, args.paths) if args.paths else changed_files(args.repo, args.base, args.head)
    except GitError as e:
        print(f"tests_for: {e}", file=sys.stderr)
        return 2
    report = survey(args.repo, files)
    lines, left_out = commands(args.repo, report, args.all)
    if args.json:
        print(json.dumps({"files": report, "commands": lines, "left_out": left_out}, indent=2))
    elif not files:
        print("No changed files: name the files or folders to look up.")
    else:
        print_report(report, lines, left_out)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
