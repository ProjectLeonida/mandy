"""One overview of the GPU test records in gpu-test-results/: what ran where, and which fixtures fail on what.

    python -I tools/gpu_results.py [RECORD_OR_FOLDER ...] [--markdown | --json | --check] [--main REF | --no-git]

A record is a text file of `Key: value` fields, each starting in the first column and continued by indented
lines (gpu-test-results/README.md has the fields). Nothing else reads them, so which fixtures fail on which GPUs
and drivers is only in their prose. This reads every record (the repository's folder when none is named) and
prints:

- a row per record: GPU, driver, how far its commit is behind `--main`, the native and the wave64-emulated run of
  `run_gpu_tests.py` as passed/total with skips and failures, and the fixtures it lists as failed;
- each failing fixture with the records and vendors it fails on. A fixture that fails across vendors points at
  the fixture or the translator, one that fails on a single driver at that driver. With git at hand it also says
  where it was last seen failing and how many records taken at newer commits do not list it, since many records
  name failures that `main` has fixed since;
- the records it could not count, and why.

`--markdown` prints the record table for a document, `--json` everything as one object. `--check` only verifies
the records: it names each one that lacks GPU, Driver, OS or Commit, or whose Commit has no hash, and exits 1 when
there is one, which is the check for a pull request that adds a record.

What is read from the prose is a convention, not a grammar: a run is a line with `N/M passed`, labelled by what
stands before it (`native:`, `LEONIDA_W64_EMULATE=1:`) or after it in parentheses; a failed fixture is a fixture
name (`cs_`, `ps_`, `vs_`, `ms_`, `gs_`...) that a line of the `Failed:` field starts with, alone or as a list.
A record that words these differently is still shown, with what could be read.

Exit 0, or 1 when `--check` finds a record to fix, 2 on a usage error. Reads the records and asks git for commit
distances; writes nothing. Standard library only; no game files.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FOLDER = os.path.join(ROOT, "gpu-test-results")
REQUIRED = ("GPU", "Driver", "OS", "Commit")
# A key has no comma: a sentence that happens to hold a colon is not a field.
FIELD = re.compile(r"^([A-Za-z][^:,\n]{0,60}?):(?:[ \t]+|$)")
COUNTS = re.compile(r"(\d+)\s*/\s*(\d+)\s+passed(?:\s*,\s*(\d+)\s+skipped)?(?:\s*,\s*(\d+)\s+failed)?")
FIXTURE = r"(?:cs|ps|vs|ms|gs|hs|ds)_[a-z0-9_]+"
FAILED_LEAD = re.compile(rf"^(?:[-*]\s+)?({FIXTURE}(?:\s*(?:,|/|&|and|,\s*and)\s*{FIXTURE})*)")
ENTRY_END = re.compile(r"\s*(?:\([^)]*\))?\s*:")
HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
VENDORS = (("NVIDIA", ("nvidia", "geforce", "quadro", "tesla")), ("AMD", ("amd", "radeon")),
           ("Intel", ("intel",)), ("Apple", ("apple",)), ("software", ("llvmpipe", "lavapipe", "swiftshader")))


def parse_fields(text):
    """[(key, value)] in file order. A field starts in the first column; indented lines, blank lines and lines
    that are no field continue the one before."""
    fields = []
    for line in text.splitlines():
        m = FIELD.match(line)
        if m:
            fields.append([m.group(1).strip(), line[m.end():].strip()])
        elif fields:
            fields[-1][1] += "\n" + line.strip()
    return [(key, value.strip()) for key, value in fields]


def outside_parentheses(text, stop=","):
    """`text` up to the first `stop` that is not inside parentheses."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == stop and depth == 0:
            return text[:i]
    return text


def short(text, keep_parenthesis_without_digit=False):
    """The first line of a field without its trailing detail: `NVIDIA 617.42 (proprietary), Vulkan ...` ->
    `NVIDIA 617.42`. A driver named without a version keeps its parenthesis: `RADV (Mesa 26.3.1)`."""
    head = outside_parentheses(text.split("\n", 1)[0]).strip()
    name, paren, rest = head.partition(" (")
    if paren and keep_parenthesis_without_digit and not any(ch.isdigit() for ch in name):
        return f"{name} ({rest.split(')', 1)[0]})"
    return name.strip()


def gpu_name(text):
    """`NVIDIA GeForce RTX 4070 (Ada / AD104, PCI ...), subgroup 32..32` -> (`NVIDIA GeForce RTX 4070`, `Ada`)."""
    head = text.split("\n", 1)[0]
    name, paren, rest = head.partition(" (")
    family = re.split(r"[,/)]", rest, maxsplit=1)[0].strip() if paren else ""
    return outside_parentheses(name).strip() if not paren else name.strip(), family


def vendor_of(gpu):
    low = gpu.lower()
    for vendor, words in VENDORS:
        if any(word in low for word in words):
            return vendor
    return "other"


def mode_of(label):
    """native, emulated, both (one line for the two runs) or the label itself for any other run."""
    low = label.lower()
    emulated = "w64_emulate" in low or "emulated" in low
    native = re.search(r"\bnative\b", low) is not None
    if emulated and native:
        return "both"
    if emulated:
        return "emulated" if re.fullmatch(r"\W*(leonida_w64_emulate=1|emulated)\W*", low) else label
    if native:
        return "native" if re.fullmatch(r"\W*native\W*", low) else label
    return "native" if not label else label


def parse_runs(fields):
    """The run_gpu_tests.py runs of a record: [{mode, label, passed, total, skipped, failed}]."""
    runs = []
    for key, value in fields:
        if not key.startswith("run_gpu_tests.py"):
            continue
        key_label = key[len("run_gpu_tests.py"):].strip(" ()")
        for line in value.splitlines():
            m = COUNTS.search(line)
            if not m:
                continue
            before = line[:m.start()].split(";")[-1].strip().rstrip(":").strip()
            after = re.match(r"\s*[.;]?\s*\(([^)]*)\)", line[m.end():])
            label = before or key_label or (after.group(1).strip() if after else "")
            if after and not before and not key_label and mode_of(label) not in ("native", "emulated", "both"):
                label = ""   # a remark in parentheses, such as "(exit 1)", is not the name of a run
            run = {"label": label, "passed": int(m.group(1)), "total": int(m.group(2)),
                   "skipped": None if m.group(3) is None else int(m.group(3)),
                   "failed": None if m.group(4) is None else int(m.group(4))}
            mode = mode_of(label)
            for one in (("native", "emulated") if mode == "both" else (mode,)):
                runs.append({"mode": one, **run})
    return runs


def parse_failed(fields):
    """The fixtures the Failed field names. The field opens with them (`cs_a, cs_b (both modes): ...`); a later
    line adds more only as an entry of its own, `ps_c: ...` or `ps_c (native): ...`, because a wrapped sentence
    can start with a fixture that passes (`cs_d and cs_e pass with #N applied`)."""
    names = []
    for key, value in fields:
        if key != "Failed" or value.lower().startswith("none"):
            continue
        for number, line in enumerate(value.splitlines()):
            line = line.strip()
            m = FAILED_LEAD.match(line)
            if not m or (number and not ENTRY_END.match(line[m.end():])):
                continue
            for name in re.findall(FIXTURE, m.group(1)):
                if name not in names:
                    names.append(name)
    return names


def parse_record(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        fields = parse_fields(f.read())
    first = {}
    for key, value in fields:
        first.setdefault(key, value)
    gpu, family = gpu_name(first.get("GPU", ""))
    commit = HASH.search(first.get("Commit", ""))
    subgroup = re.search(r"(\d+)\s*\.\.\s*(\d+)", first.get("Subgroup", "") + "\n" + first.get("GPU", ""))
    runs = parse_runs(fields)
    record = {
        "file": os.path.basename(path),
        "gpu": gpu, "family": family, "vendor": vendor_of(first.get("GPU", "")),
        "driver": short(first.get("Driver", ""), keep_parenthesis_without_digit=True),
        "os": short(first.get("OS", "")),
        "commit": commit.group(0) if commit else None, "behind": None,
        "subgroup": [int(subgroup.group(1)), int(subgroup.group(2))] if subgroup else None,
        "runs": runs, "failed": parse_failed(fields),
        "problems": [f"no {key} field" for key in REQUIRED if not first.get(key)],
        "notes": [],
    }
    if first.get("Commit") and not commit:
        record["problems"].append("the Commit field has no commit hash")
    if not runs:
        record["notes"].append("no run_gpu_tests.py line with `N/M passed`")
    most = max((run["failed"] or 0 for run in runs), default=0)
    if most and len(record["failed"]) < most:
        record["notes"].append(f"{most} failed in a run, {len(record['failed'])} fixture name(s) read from Failed")
    return record


def find_records(targets):
    paths = []
    for target in targets:
        if os.path.isdir(target):
            paths += [os.path.join(target, name) for name in sorted(os.listdir(target))
                      if name.lower().endswith(".txt")]
        elif os.path.isfile(target):
            paths.append(target)
        else:
            raise OSError(f"{target}: no such record or folder")
    return paths


def commits_behind(records, main, repo=ROOT):
    """Fill `behind`: how many commits on `main` a record's commit lacks. Left None where git cannot tell."""
    def git(*args):
        try:
            r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
        except OSError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    if git("rev-parse", "--verify", "--quiet", f"{main}^{{commit}}") is None:
        return False
    for record in records:
        if record["commit"]:     # a hash this repository does not have leaves it None: rev-list fails
            count = git("rev-list", "--count", f"{record['commit']}..{main}")
            record["behind"] = int(count) if count and count.isdigit() else None
    return True


def run_of(record, mode):
    return next((run for run in record["runs"] if run["mode"] == mode), None)


def fixtures(records):
    """Each failing fixture: the records that list it, by vendor, and what newer records say."""
    counted = [r for r in records if r["runs"]]
    table = []
    for name in sorted({name for r in records for name in r["failed"]}):
        on = [r for r in records if name in r["failed"]]
        aged = [r["behind"] for r in on if r["behind"] is not None]
        entry = {"fixture": name, "records": [r["file"] for r in on],
                 "vendors": dict(Counter(r["vendor"] for r in on).most_common()),
                 "last_behind": min(aged) if aged else None, "last_record": None, "newer_without": None}
        if aged:
            entry["last_record"] = next(r["file"] for r in on if r["behind"] == entry["last_behind"])
            # Strictly newer than the newest record that lists it, so none of these lists it.
            entry["newer_without"] = sum(1 for r in counted if r["behind"] is not None
                                         and r["behind"] < entry["last_behind"])
        table.append(entry)
    return sorted(table, key=lambda e: (-len(e["records"]), e["fixture"]))


def overview(records, main):
    return {"records": records, "main": main,
            "vendors": dict(Counter(r["vendor"] for r in records).most_common()),
            "fixtures": fixtures(records)}


def show_run(run):
    if not run:
        return "-"
    return (f"{run['passed']}/{run['total']}" + ("" if run["skipped"] is None else f" {run['skipped']}s")
            + ("" if run["failed"] is None else f" {run['failed']}f"))


def print_text(report):
    records = report["records"]
    print(f"{len(records)} record(s): " + ", ".join(f"{v} {n}" for v, n in report["vendors"].items()))
    if report["main"]:
        print(f"behind: commits of {report['main']} that the record's commit does not have; ? when git cannot tell")
    print("runs: passed/total, s skipped, f failed\n")
    rows = [("record", "GPU", "driver", "behind", "native", "emulated", "failed")]
    for r in sorted(records, key=lambda r: (r["vendor"], r["gpu"], r["file"])):
        rows.append((r["file"], r["gpu"] + (f" ({r['family']})" if r["family"] else ""), r["driver"],
                     "?" if r["behind"] is None else str(r["behind"]), show_run(run_of(r, "native")),
                     show_run(run_of(r, "emulated")), ", ".join(r["failed"]) or "-"))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[-1])
    if report["fixtures"]:
        print("\nFixtures that fail")
    for e in report["fixtures"]:
        line = (f"  {e['fixture']}: {len(e['records'])} record(s), "
                + ", ".join(f"{vendor} {n}" for vendor, n in e["vendors"].items()))
        if e["last_behind"] is not None:
            line += (f"; last in {e['last_record']} ({e['last_behind']} behind), "
                     f"{e['newer_without']} newer record(s) do not list it")
        print(line)
        print(textwrap.fill(", ".join(e["records"]), width=110, initial_indent="      ", subsequent_indent="      ",
                            break_long_words=False, break_on_hyphens=False))
    other = [(r["file"], run) for r in records for run in r["runs"] if run["mode"] not in ("native", "emulated")]
    if other:
        print("\nOther runs")
        for name, run in other:
            print(f"  {name}: {run['label']}: {show_run(run)}")
    noted = [(r["file"], text) for r in records for text in r["problems"] + r["notes"]]
    if noted:
        print("\nNot read in full")
        for name, text in noted:
            print(f"  {name}: {text}")


def print_markdown(report, folder_url):
    print("| Record | GPU | Driver | OS | Commit | Native | Emulated | Failed |")
    print("|---|---|---|---|---|---|---|---|")
    cell = lambda text: text.replace("|", "\\|")  # noqa: E731
    for r in sorted(report["records"], key=lambda r: (r["vendor"], r["gpu"], r["file"])):
        print("| " + " | ".join((
            f"[{cell(r['file'])}]({folder_url}/{r['file']})",
            cell(r["gpu"] + (f" ({r['family']})" if r["family"] else "")), cell(r["driver"]), cell(r["os"]),
            f"`{r['commit'][:9]}`" if r["commit"] else "?", show_run(run_of(r, "native")),
            show_run(run_of(r, "emulated")), ", ".join(f"`{name}`" for name in r["failed"]) or "none")) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("targets", nargs="*", metavar="RECORD_OR_FOLDER",
                    help="records or folders of *.txt records (default: the repository's gpu-test-results)")
    form = ap.add_mutually_exclusive_group()
    form.add_argument("--markdown", action="store_true", help="the record table as Markdown")
    form.add_argument("--json", action="store_true", help="everything as one JSON object")
    form.add_argument("--check", action="store_true",
                      help="only verify the records; exit 1 when one lacks a required field")
    ap.add_argument("--main", default="origin/main", help="the branch commit distances are measured to (origin/main)")
    ap.add_argument("--no-git", action="store_true", help="do not ask git how far behind each record's commit is")
    ap.add_argument("--folder-url", default="../gpu-test-results",
                    help="link prefix of the records in --markdown (default ../gpu-test-results, as seen from docs/)")
    args = ap.parse_args(argv)
    try:
        records = [parse_record(path) for path in find_records(args.targets or [DEFAULT_FOLDER])]
    except OSError as e:
        print(f"gpu_results: {e}", file=sys.stderr)
        return 2
    if args.check:
        bad = [(r["file"], text) for r in records for text in r["problems"]]
        for name, text in bad:
            print(f"{name}: {text}")
        for r in records:
            for text in r["notes"]:
                print(f"{r['file']}: note: {text}")
        print(f"{len(records)} record(s) checked, {len({name for name, _ in bad})} to fix")
        return 1 if bad else 0
    measured = not args.no_git and commits_behind(records, args.main)
    report = overview(records, args.main if measured else None)
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.markdown:
        print_markdown(report, args.folder_url.rstrip("/"))
    else:
        print_text(report)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
