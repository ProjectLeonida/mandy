"""One page of counts for the whole shader chain of a work folder: tools, disassembly and translation per family, the
contract with the runtime and, from a run's log, what the runtime made of it. One block per link, each with a verdict
(OK / WARN / FAIL) and its most likely cause, then the weakest link.

    python -I tools/chain_check.py <work folder> [--log RUN.log] [--no-probe] [--json] [--strict] [--show-paths]

The links fail quietly one by one: an llvm-mc whose gfx1013 decoder rejects words leaves programs untranslated
without a warning (#1923), a translation made by older translator sources keeps loading after main changed the
contract (#2098, "descriptor_table": "tagged"), and the runtime's slot-67 refusals and failed CreatePipelineState
calls sit in a log of a million lines. This page puts the counts side by side.

Reads, in the work folder (make_game.py's layout):

* shaders/<family>/index.json (and dxbc/fssm_index.json): what was unpacked, with `invalid_words` where the
  unpacker records them; shaders/<family>/disasm/*.s: the listings;
* shaders/spirv[/<family>]/index.json and the .spv beside it: what was translated, with each module's
  `descriptor_table`; translation_sources.sha256 and <family>_build.json: the llvm-mc and the translator sources
  the translation was made with; shaders/dxbc/disasm.done: the llvm-mc of the game's disassembly;
* logs/shaders.log: the translator's own refusal reasons of the latest run per family (optional);
* --log: a gta6rt stderr log (scripts/windows-boot/boot.ps1's <Name>.log), with <Name>.stdout.log and
  <Name>.digest.txt beside it when they are there.

Verdicts (also in --help):

* tools: FAIL with no llvm-mc record or one without the AMDGPU (amdgcn) target; WARN for an llvm-mc older than
  GOOD_LLVM_MAJOR (23, the version #1923 measured reading the library), for families made with different llvm-mc
  versions or with another one than LLVM_MC names now, for a translation made by other translator sources than this
  checkout's (make_game.py's translation_sources_digest()), and for no spirv-val on PATH.
* a family: OK when every unpacked program has a module; WARN when at most FAIL_FRACTION (1 %) of them have none
  (99 of 100 is WARN, 197 of 200 is FAIL) or when there are more modules than programs; FAIL above that, with no
  readable index, or with no translated index (named "in progress or interrupted" when shard files are there); n/a
  when the work folder does not hold that family. An index entry whose .spv is missing does not count as translated.
* contract: WARN for mixed tagged and plain modules, for only plain modules when the translator's default is tagged
  (#2098), and for no fspx_compute/ while library programs are untranslated.
* runtime: FAIL for a device loss or fault, a failed device slot call, no translated-shaders line or a non-zero exit
  code; WARN for "device-fault: submitted dispatch" lines alone (a stall prints them too), for failed or unsupported
  D3D12 calls and for a loaded module count other than the work folder's. The log may be UTF-8 or UTF-16 with or
  without a BOM (detected from its zero bytes).

The weakest link is the first block, in chain order, with the worst verdict. The exit code is 0 whatever the
verdicts (this is a report); --strict exits 1 when a block is FAIL. Bad arguments exit 2.

It prints counts, verdicts, versions, short digests and the translator's and runtime's own fixed words, never a
program name, a path or a log line (AGENTS.md, "No game data"). Each free-text field is closed against the sources:
a refusal reason is printed only when it matches one of the messages rdna2spirv.py raises (read from its source),
each variable part kept only when it is an instruction of rdna2_isa_tables.py (encoding suffix _dpp, _sdwa, _e32 or
_e64 removed) or a built-in exception name and printed as "#" otherwise, and the whole reason is "other" when it
matches no message; a D3D12 trace method is printed only when it is a string literal of runtime/shims or
runtime/src (or Interface::Method built from them), else "(other)"; boot.ps1's ending only in its three forms, else
"other". The work folder and log paths appear only with --show-paths.
Standard library only. It runs nothing heavy: with probing on (the default) it asks llvm-mc and spirv-val for
--version, and git for the current commit; --no-probe runs nothing. A partial or empty work folder gives n/a blocks.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

OK, WARN, FAIL, NA = "OK", "WARN", "FAIL", "n/a"
RANK = {NA: -1, OK: 0, WARN: 1, FAIL: 2}
# The llvm-mc major whose gfx1013 decoder reads the pipeline library (#1923 measured 21.1.8 against 23.1.5).
GOOD_LLVM_MAJOR = 23
# A family with untranslated programs is WARN up to this fraction of its programs, FAIL above it.
FAIL_FRACTION = 0.01
VERDICTS = (f"Verdicts: a family is OK when every program has a module, WARN up to {FAIL_FRACTION:.0%} untranslated, "
            f"FAIL above; tools are WARN for llvm-mc older than {GOOD_LLVM_MAJOR} or a stale translator digest, FAIL "
            "with no llvm-mc record or no AMDGPU target; the runtime is FAIL for a failed device slot call, a device "
            "loss or a non-zero exit. Exit code 0 (a report), 1 with --strict when a block is FAIL, 2 for bad "
            "arguments.")
# (key, label, folder under shaders/, folder under shaders/spirv/, build settings file)
FAMILIES = (
    ("game", "game programs (shaders.rpf)", "dxbc", "", None),
    ("library", "pipeline library compute", "library", "library", "library_build.json"),
    ("fspx", "FidelityFX FSPX", "fspx", "fspx", "fspx_build.json"),
    ("bvh", "BVH builder", "bvh", "bvh", "bvh_build.json"),
    ("exe", "executable", "exe", "exe", "exe_build.json"),
)
# The sub-folders the runtime's ShaderLibrary::load() reads (runtime/src/shader_library.h).
RUNTIME_FOLDERS = ("fspx", "fspx_compute", "fspx_gfx", "bvh", "exe", "library")

CMD = re.compile(r"rdna2spirv\.py\s+(\S+)")
SUMMARY = re.compile(r"^\s*(\d+) shaders: (\d+) translated, (\d+) ")
REASON = re.compile(r"^\s*(\d+)\s{2,}(\S.{0,300}?)\s*$")   # free text here: normalize_reason() closes it
LLVM_VERSION = re.compile(r"LLVM version (\d+)(?:\.(\d+))?(?:\.(\d+))?")
SPIRV_VAL_VERSION = re.compile(r"(v\d{4}\.\d+)")
DEFAULT_TABLE = re.compile(r'"--descriptor-table",\s*choices=DESCRIPTOR_TABLES,\s*default="(\w+)"')

LOADED = re.compile(r"\[rt\] d3d12: (\d+) translated shaders \((\d+) fspx, (\d+) fspx_compute, (\d+) fspx_gfx, "
                    r"(\d+) bvh, (\d+) exe\)")
TABLE_ALL = re.compile(r"\[rt\] d3d12: descriptor table: (plain|tagged) \(all (\d+) modules\)")
TABLE_MIX = re.compile(r"\[rt\] d3d12: descriptor table: (\d+) tagged / (\d+) plain modules; using (plain|tagged)")
TABLE_FORCED = re.compile(r"\[rt\] d3d12: descriptor table: (plain|tagged) \(LEONIDA_DESCRIPTOR_TABLE; (\d+) tagged / "
                          r"(\d+) plain modules\)")
SLOT_FAIL = re.compile(r"\[rt\] d3d12_xs: slot (\d+) call \d+ failed with (0x[0-9a-fA-F]{8})")
NO_COMPUTE = re.compile(r"\[rt\] d3d12: no translated compute shader for ")
BUILD_START = re.compile(r"\[rt\] d3d12: compute pipeline \S+ \(\d+ B SPIR-V, [^)]*\) build starts")
BUILD_SLOW = re.compile(r"\[rt\] d3d12: compute pipeline \S+ took [0-9.]+ s to build")
TRACE = re.compile(r"\[d3d12-trace\] #\d+ (.+) (failed|unsupported|refused|partial)\s*$")
GATE = re.compile(r"\[rt\] submit-lock: thread \d+ is waiting at the compile gate")
STALL = re.compile(r"\[rt\] device-fault: submitted dispatch ")   # also printed for a plain stall
FAULT = re.compile(r"\[rt\] device-fault: (?!enabled|submitted dispatch )")
LOST = re.compile(r"VK_ERROR_DEVICE_LOST")
EXITED = re.compile(r"game exited with code (-?\d+)")
# A trace method is printed only when it is a string literal of the runtime's sources, or Interface::Method /
# Interface::Slot<N> built from one (d3d12_xs_device.cpp, child_trace_name()); anything else is "(other)".
COMPOSED_METHOD = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z_][A-Za-z0-9_]*)$")
C_STRING = re.compile(r'"((?:[^"\\\n]|\\.){1,120})"')
RUNTIME_SOURCE_DIRS = ("runtime/shims", "runtime/src")
# A variable part of a refusal reason is kept only when it is an instruction of tools/rdna2_isa_tables.py (with a
# _dpp, _sdwa, _e32 or _e64 encoding suffix removed first) or a built-in exception name; else it is "#".
MNEMONIC = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
ISA_NAME = re.compile(r"'([a-z][a-z0-9]*(?:_[a-z0-9]+)+)'")
ISA_SUFFIX = re.compile(r"_(?:dpp|sdwa|e32|e64)$")
UNSUPPORTED_CALL = re.compile(r"(?<![\w.])Unsupported\(")
# The three verdicts scripts/windows-boot/boot.ps1 writes on its digest's first line; anything else is "other".
ENDINGS = (re.compile(r"^ran its full time limit \(\d+ s\), stopped$"),
           re.compile(r"^exited by itself after \d+ s, exit code (?:unknown|(-?\d+) \(0x[0-9a-f]{8}\))$"),
           re.compile(r"^digest of an earlier run \(-DigestOnly\)$"))


# ------------------------------------------------------------------------------------------------ reading helpers
def load_json(path: str):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def log_encoding(path: str) -> str:
    """UTF-8, or UTF-16 with a BOM (PowerShell 5.1 redirection) or without one: ASCII text in UTF-16 has a zero in
    every other byte, little-endian at the odd offsets."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return "utf-16"
    if head.count(0) * 4 >= len(head) > 0:
        return "utf-16-le" if head[1::2].count(0) >= head[0::2].count(0) else "utf-16-be"
    return "utf-8"


def log_lines(path: str):
    """The lines of a log in the encoding log_encoding() finds."""
    with open(path, encoding=log_encoding(path), errors="replace") as f:
        yield from f


def _safe_lines(path: str):
    """log_lines(), ending quietly at an unreadable file instead of raising."""
    try:
        yield from log_lines(path)
    except (OSError, UnicodeError):
        return


def listing_count(folder: str) -> int | None:
    try:
        with os.scandir(folder) as entries:
            return sum(1 for e in entries if e.name.endswith(".s") and e.is_file() and e.stat().st_size > 0)
    except OSError:
        return None


def shard_parts(spirv: str) -> int:
    """The shard plans and index parts make_game.py's translate() leaves while a translation runs or once it stopped."""
    try:
        with os.scandir(spirv) as entries:
            return sum(1 for e in entries if re.fullmatch(r"(?:shard\d+|index\.part\d+)\.json", e.name))
    except OSError:
        return 0


def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def short(digest: str | None) -> str:
    return digest[:12] if isinstance(digest, str) and digest else "none"


# ------------------------------------------------------------------------------------------------ repository side
def repo_translation_digest(root: str = ROOT) -> str | None:
    """make_game.py's translation_sources_digest(), computed without importing make_game.py: the TRANSLATION_SOURCES
    tuple is read from its source, then SHA-256 over each file's name, length and bytes (sources_digest())."""
    try:
        with open(os.path.join(root, "make_game.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return None
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TRANSLATION_SOURCES" for t in node.targets):
            names = ast.literal_eval(node.value)
    if not names:
        return None
    h = hashlib.sha256(b"")
    try:
        for name in names:
            with open(os.path.join(root, "tools", name), "rb") as f:
                data = f.read()
            h.update(f"{name}\0{len(data)}\0".encode())
            h.update(data)
    except OSError:
        return None
    return h.hexdigest()


def translator_default_table(root: str = ROOT) -> str | None:
    text = read_text(os.path.join(root, "tools", "rdna2spirv.py")) or ""
    m = DEFAULT_TABLE.search(text)
    return m.group(1) if m else None


def run_quiet(command: list[str], timeout: int = 20) -> str | None:
    try:
        r = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return (r.stdout + r.stderr).strip() if r.returncode == 0 else None


def parse_llvm(text: str | None) -> dict:
    """Version and AMDGPU target from an llvm-mc --version text, or from make_game.py's signature (a JSON pair)."""
    command = None
    if text and text.startswith("["):
        try:
            command, text = json.loads(text)
        except (ValueError, TypeError):
            pass
    m = LLVM_VERSION.search(text or "")
    program = os.path.basename(str(command[-1])) if isinstance(command, list) and command else None
    via = "wsl" if isinstance(command, list) and any("wsl" in str(c).lower() for c in command[:1]) else None
    return {"version": ".".join(g for g in m.groups() if g) if m else None,
            "major": int(m.group(1)) if m else None,
            "amdgpu": ("amdgcn" in text) if text else None,
            "program": program, "via": via}


# ------------------------------------------------------------------------------------------------ refusal reasons
def _pattern(node) -> list:
    """An Unsupported(...) message as a list of literal strings, with None for each variable part."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [v.value if isinstance(v, ast.Constant) else None for v in node.values]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _pattern(node.left) + _pattern(node.right)
    return [None]


def _unsupported_args(source: str):
    """The first argument of each Unsupported(...) call in source. Each call is parsed on its own (the shortest text
    up to a closing parenthesis that parses), since parsing the whole 20,000-line translator takes seconds."""
    for m in UNSUPPORTED_CALL.finditer(source):
        start = m.end()
        end = start
        while (end := source.find(")", end + 1 if end > start else start)) != -1 and end - start < 2000:
            try:
                call = ast.parse("f(" + source[start:end] + ")", mode="eval").body
            except SyntaxError:
                continue
            if isinstance(call, ast.Call) and call.args:
                yield call.args[0]
            break


@functools.lru_cache(maxsize=4)
def reason_templates(root: str = ROOT) -> tuple:
    """The refusal messages rdna2spirv.py counts, as (regex, parts) pairs with the most literal text first, plus its
    "translator error <exception>" count. Empty when the translator's source cannot be read."""
    try:
        with open(os.path.join(root, "tools", "rdna2spirv.py"), encoding="utf-8") as f:
            source = f.read()
    except (OSError, ValueError):
        return ()
    found = {("translator error ", None)}
    for arg in _unsupported_args(source):
        merged = []
        for part in _pattern(arg):   # adjacent variable parts are one
            if not (part is None and merged and merged[-1] is None):
                merged.append(part)
        found.add(tuple(merged))
    templates = [(re.compile("^" + "".join("(.+?)" if p is None else re.escape(p) for p in parts) + "$"), parts)
                 for parts in found]
    templates.sort(key=lambda t: (-sum(len(p) for p in t[1] if p), t[0].pattern))
    return tuple(templates)


@functools.lru_cache(maxsize=4)
def isa_names(root: str = ROOT) -> frozenset:
    """The instruction names of tools/rdna2_isa_tables.py (empty when it cannot be read)."""
    try:
        with open(os.path.join(root, "tools", "rdna2_isa_tables.py"), encoding="utf-8") as f:
            return frozenset(ISA_NAME.findall(f.read()))
    except (OSError, ValueError):
        return frozenset()


def known_value(value: str) -> bool:
    """A variable part that may be printed: an ISA instruction of the decoder's tables or a built-in exception."""
    if MNEMONIC.match(value) and ISA_SUFFIX.sub("", value) in isa_names():
        return True
    builtin = getattr(builtins, value, None)
    return isinstance(builtin, type) and issubclass(builtin, BaseException)


@functools.lru_cache(maxsize=4)
def runtime_literals(root: str = ROOT) -> frozenset:
    """Every short C string literal in the runtime's sources: the closed set a trace method name must come from."""
    found = set()
    for folder in RUNTIME_SOURCE_DIRS:
        for dirpath, _, files in os.walk(os.path.join(root, folder)):
            for name in files:
                if name.endswith((".cpp", ".h", ".hpp", ".inl")):
                    try:
                        with open(os.path.join(dirpath, name), encoding="utf-8", errors="replace") as f:
                            found.update(C_STRING.findall(f.read()))
                    except OSError:
                        continue
    return frozenset(found)


def trace_method(method: str) -> str:
    """The method of a [d3d12-trace] line when the runtime's sources name it, else "(other)"."""
    literals = runtime_literals()
    if method in literals:
        return method
    m = COMPOSED_METHOD.match(method)
    if m and m.group(1) in literals and (m.group(2) in literals or re.fullmatch(r"Slot\d{1,3}", m.group(2))):
        return method
    return "(other)"


def normalize_reason(reason: str, templates: list) -> str:
    """The reason when it is one of the translator's messages, each variable part kept only when known_value() (else
    "#"); "other" for anything else."""
    for regex, parts in templates:
        m = regex.match(reason)
        if not m:
            continue
        values = iter(m.groups())
        out = []
        for part in parts:
            if part is None:
                value = next(values)
                part = value if known_value(value) else "#"
            out.append(part)
        text = "".join(out)
        # A message that is only a variable part (Unsupported(op)) says nothing once that part is masked.
        return "other" if not text.strip("# ") else text
    return "other"


# ------------------------------------------------------------------------------------------------ the build log
def family_of(disasm_path: str) -> str | None:
    parts = [p for p in re.split(r"[\\/]+", disasm_path.lower()) if p]
    parent = parts[-2] if len(parts) >= 2 else ""
    return {"dxbc": "game", "library": "library", "fspx": "fspx", "bvh": "bvh", "exe": "exe"}.get(parent)


def parse_build_log(path: str, templates: list | None = None) -> dict:
    """The latest translator run of each family in logs/shaders.log: programs, translated, and refusals by reason.
    A run starts at an rdna2spirv.py command line (not --merge) and its shards' summaries follow it."""
    runs: dict = {}
    current = None
    in_reasons = False
    try:
        lines = log_lines(path)
        for line in lines:
            m = CMD.search(line)
            if m and "--merge" not in line:
                current = family_of(m.group(1))
                if current:
                    runs[current] = {"programs": 0, "translated": 0, "valid": 0, "reasons": Counter()}
                in_reasons = False
                continue
            m = SUMMARY.match(line)
            if m and current:
                run = runs[current]
                run["programs"] += int(m.group(1))
                run["translated"] += int(m.group(2))
                run["valid"] += int(m.group(3))
                in_reasons = True
                continue
            if in_reasons and current:
                m = REASON.match(line)
                if m:
                    reason = normalize_reason(m.group(2), templates or [])
                    runs[current]["reasons"][reason] += int(m.group(1))
                    continue
                in_reasons = False
    except (OSError, UnicodeError):
        return {}
    return runs


# ------------------------------------------------------------------------------------------------ the blocks
def tools_block(work: str, probe: bool) -> dict:
    sh = os.path.join(work, "shaders")
    records = {}
    done = load_json(os.path.join(sh, "dxbc", "disasm.done"))
    if isinstance(done, dict) and isinstance(done.get("llvm"), str):
        records["game"] = parse_llvm(done["llvm"])
    translator = {"game": read_text(os.path.join(sh, "spirv", "translation_sources.sha256"))}
    for key, _, _, spirv_sub, settings in FAMILIES:
        if not settings:
            continue
        s = load_json(os.path.join(sh, "spirv", spirv_sub, settings))
        if isinstance(s, dict):
            if isinstance(s.get("llvm_mc"), str):
                records[key] = parse_llvm(s["llvm_mc"])
            translator[key] = s.get("translator")
    versions = Counter(r["version"] or "?" for r in records.values())
    repo = repo_translation_digest()
    translator = {k: v for k, v in translator.items() if isinstance(v, str) and v}
    local = Counter(short(v) for v in translator.values())
    stale = sorted(k for k, v in translator.items() if repo and v != repo)
    current = spirv_val = head = None
    if probe:
        sys.path.insert(0, HERE)
        try:
            from mc_command import mc_command
            command = mc_command()
            command[0] = shutil.which(command[0]) or command[0]
            text = run_quiet([*command, "--version"])
            current = parse_llvm(text) if text else {"version": None}
        except (Exception, SystemExit):  # noqa: BLE001 - a broken LLVM_MC is reported, not raised
            current = {"version": None}
        exe = shutil.which("spirv-val")
        text = run_quiet([exe, "--version"]) if exe else None
        m = SPIRV_VAL_VERSION.search(text or "")
        spirv_val = (m.group(1) if m else "found, version unread") if exe else "not found"
        head = run_quiet(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"])
    verdict, causes = OK, []
    majors = {r["major"] for r in records.values()}
    if not records and not translator:
        verdict, causes = NA, ["nothing disassembled or translated in this work folder"]
    elif not records:
        verdict, causes = FAIL, ["no llvm-mc record in the work folder (disassembly never finished)"]
    elif any(r["amdgpu"] is False for r in records.values()):
        verdict, causes = FAIL, ["a recorded llvm-mc has no AMDGPU (amdgcn) target"]
    else:
        old = {k: r["version"] for k, r in records.items() if r["major"] is not None and r["major"] < GOOD_LLVM_MAJOR}
        if old:
            verdict = WARN
            causes.append(f"llvm-mc {'/'.join(sorted(set(old.values())))} (used for {', '.join(sorted(old))}) "
                          f"predates {GOOD_LLVM_MAJOR}: its gfx1013 decoder rejects words the library needs (#1923)")
        if len(versions) > 1:
            verdict = WARN
            causes.append("families were disassembled with different llvm-mc versions")
        if current and current.get("version") and current["version"] not in versions:
            verdict = WARN
            causes.append(f"llvm-mc now on LLVM_MC is {current['version']}: the folder was made with another")
    if stale and verdict != NA:
        verdict = max(verdict, WARN, key=RANK.get)
        causes.append(f"translation of {', '.join(stale)} made by other translator sources than this checkout's "
                      "(make_game.py shaders translates again)")
    if spirv_val == "not found" and verdict != NA:
        verdict = max(verdict, WARN, key=RANK.get)
        causes.append("spirv-val not on PATH")
    return {"name": "tools", "verdict": verdict, "cause": "; ".join(causes) or "-",
            "_majors": {k: r["major"] for k, r in records.items()},
            "llvm_mc_recorded": dict(versions), "llvm_mc_records": len(records),
            "llvm_mc_amdgpu": (False if any(r["amdgpu"] is False for r in records.values()) else
                               True if records and all(r["amdgpu"] for r in records.values()) else None),
            "llvm_mc_via": sorted({r["via"] or "native" for r in records.values()}),
            "llvm_mc_current": (current or {}).get("version") if probe else NA,
            "spirv_val": spirv_val if probe else NA,
            "translator_local": dict(local), "translator_repo": short(repo), "repo_head": head,
            "translator_stale": stale}


def family_block(work: str, key: str, label: str, folder: str, spirv_sub: str, build: dict, llvm_major) -> dict:
    sh = os.path.join(work, "shaders")
    unpacked = os.path.join(sh, folder)
    block = {"name": key, "label": label}
    if not os.path.isdir(unpacked):
        block.update(verdict=NA, cause="not unpacked in this work folder")
        return block
    index = load_json(os.path.join(unpacked, "fssm_index.json" if key == "game" else "index.json"))
    expected = len(index) if isinstance(index, (list, dict)) else None
    rejected = None
    if isinstance(index, list) and index and isinstance(index[0], dict) and "invalid_words" in index[0]:
        rejected = sum(1 for e in index if isinstance(e, dict) and isinstance(e.get("invalid_words"), int)
                       and e["invalid_words"] > 0)
    listings = listing_count(os.path.join(unpacked, "disasm"))
    spirv = os.path.join(sh, "spirv", spirv_sub) if spirv_sub else os.path.join(sh, "spirv")
    translated_index = load_json(os.path.join(spirv, "index.json"))
    translated = no_module = None
    tables = Counter()
    if isinstance(translated_index, dict):
        present = [n for n in translated_index if n and os.path.isfile(os.path.join(spirv, n + ".spv"))]
        translated, no_module = len(present), len(translated_index) - len(present)
        tables = Counter(m["descriptor_table"] if isinstance(m, dict) and m.get("descriptor_table") == "tagged"
                         else "plain" for m in translated_index.values())
    run = build.get(key)
    reasons = run["reasons"].most_common(5) if run else []
    block.update(expected=expected, listings=listings, listings_with_rejected_words=rejected,
                 translated=translated, modules_missing=no_module, tables=dict(tables),
                 reasons=[[n, r] for r, n in reasons],
                 log_run=({"programs": run["programs"], "translated": run["translated"]} if run else None))
    if expected is None:
        block.update(verdict=FAIL, cause="no readable unpack index")
    elif translated is None:
        parts = shard_parts(spirv)
        block.update(verdict=FAIL, cause=f"no translated index: translation in progress or interrupted ({parts} shard "
                                         "files)" if parts else "no translated index (translation never finished)")
    else:
        failed = max(expected - translated, 0)
        block.update(failed=failed, percent=pct(translated, expected))
        if translated > expected:
            block.update(verdict=WARN, cause=f"{translated - expected} more modules than unpacked programs "
                                             "(a translation from another unpack?)")
        elif failed == 0 and not no_module:
            block.update(verdict=OK, cause="-")
        else:
            block["verdict"] = WARN if failed <= expected * FAIL_FRACTION and not no_module else FAIL
            top = reasons[0][0] if reasons else None
            if top and "undecodable" in top and llvm_major is not None and llvm_major < GOOD_LLVM_MAJOR:
                cause = f"llvm-mc {llvm_major}.x leaves undecodable words ({reasons[0][1]} refusals; #1923)"
            elif top:
                cause = f"translator refusal: {top} ({reasons[0][1]})"
            elif no_module:
                cause = f"{no_module} index entries without their .spv"
            else:
                cause = "untranslated, no reason in logs/shaders.log"
            block["cause"] = cause
    return block


def contract_block(work: str, families: list[dict]) -> dict:
    spirv = os.path.join(work, "shaders", "spirv")
    default = translator_default_table()
    tables = Counter()
    for f in families:
        tables.update(f.get("tables") or {})
    folders = {name: os.path.isfile(os.path.join(spirv, name, "index.json")) for name in RUNTIME_FOLDERS}
    library_failed = next((f.get("failed") or 0 for f in families if f["name"] == "library"), 0)
    verdict, causes = (OK, []) if tables else (NA, ["no translated modules"])
    if tables.get("tagged") and tables.get("plain"):
        verdict = WARN
        causes.append("mixed tagged and plain modules: the runtime picks one table for all")
    elif default == "tagged" and tables and not tables.get("tagged"):
        verdict = WARN
        causes.append("every module is plain while this checkout's translator writes tagged (#2098): "
                      "the runtime falls back to the plain table without a word")
    if tables and not folders["fspx_compute"] and library_failed:
        verdict = max(verdict, WARN, key=RANK.get)
        causes.append(f"no fspx_compute/ to cover the {library_failed} untranslated library programs")
    return {"name": "contract", "verdict": verdict, "cause": "; ".join(causes) or "-",
            "translator_default_table": default, "modules": dict(tables),
            "folders_present": [n for n, ok in folders.items() if ok],
            "folders_absent": [n for n, ok in folders.items() if not ok]}


def runtime_block(path: str, families: list[dict]) -> dict:
    loaded = table = None
    slots, trace = Counter(), Counter()
    counts = Counter()
    exit_code = None
    try:
        encoding = log_encoding(path)
    except OSError:
        encoding = None
    for line in _safe_lines(path):
        if "[rt]" not in line and "[d3d12-trace]" not in line and "VK_ERROR" not in line and "exited" not in line:
            continue
        m = TRACE.search(line)
        if m:
            trace[f"{trace_method(m.group(1))} {m.group(2)}"] += 1
            continue
        if loaded is None and (m := LOADED.search(line)):
            loaded = dict(zip(("total", "fspx", "fspx_compute", "fspx_gfx", "bvh", "exe"), map(int, m.groups())))
        elif m := SLOT_FAIL.search(line):
            slots[f"slot {m.group(1)} {m.group(2).lower()}"] += 1
        elif NO_COMPUTE.search(line):
            counts["no_translated_compute"] += 1
        elif BUILD_START.search(line):
            counts["compute_builds"] += 1
        elif BUILD_SLOW.search(line):
            counts["slow_compute_builds"] += 1
        elif GATE.search(line):
            counts["compile_gate_waits"] += 1
        elif STALL.search(line):
            counts["stalled_dispatch_lines"] += 1
        elif FAULT.search(line):
            counts["device_fault_lines"] += 1
        elif m := TABLE_ALL.search(line):
            table = m.group(1)
        elif m := TABLE_MIX.search(line):
            table = f"{m.group(3)} ({m.group(1)} tagged / {m.group(2)} plain)"
        elif m := TABLE_FORCED.search(line):
            table = f"{m.group(1)} forced by LEONIDA_DESCRIPTOR_TABLE ({m.group(2)} tagged / {m.group(3)} plain)"
        if LOST.search(line):
            counts["device_lost_lines"] += 1
        if m := EXITED.search(line):
            exit_code = int(m.group(1))
    stem = path[:-4] if path.lower().endswith(".log") else path
    if exit_code is None and os.path.isfile(stem + ".stdout.log"):
        for line in _safe_lines(stem + ".stdout.log"):
            if m := EXITED.search(line):
                exit_code = int(m.group(1))
    digest = read_text(stem + ".digest.txt")
    ending = None
    if digest:
        first = digest.splitlines()[0]
        ending = first.split(":", 1)[1].strip() if first.startswith("== boot") and ":" in first else None
        if ending:
            matched = next((m for m in (e.match(ending) for e in ENDINGS) if m), None)
            if not matched:
                ending = "other"
            elif exit_code is None and matched.groups() and matched.group(1) is not None:
                exit_code = int(matched.group(1))
    local = sum(f.get("translated") or 0 for f in families)
    library = next((f for f in families if f["name"] == "library"), {})
    verdict, causes = OK, []
    if counts["device_lost_lines"] or counts["device_fault_lines"]:
        verdict = FAIL
    if counts["device_lost_lines"] or counts["device_fault_lines"] or counts["stalled_dispatch_lines"]:
        verdict = max(verdict, WARN, key=RANK.get)
        causes.append(f"device lost, faulted or stalled (lost {counts['device_lost_lines']}, fault "
                      f"{counts['device_fault_lines']}, stalled dispatch {counts['stalled_dispatch_lines']} lines)")
    if slots:
        verdict = FAIL
        worst, n = slots.most_common(1)[0]
        hint = ""
        if worst.startswith("slot 67") and library.get("failed"):
            hint = f", against {library['failed']} untranslated library programs"
        causes.append(f"{sum(slots.values())} device slot calls failed ({worst}: {n}{hint})")
    failed_pso = sum(n for k, n in trace.items() if k.endswith((" failed", " unsupported", " refused")))
    if failed_pso:
        verdict = max(verdict, WARN, key=RANK.get)
        causes.append(f"{failed_pso} D3D12 calls failed or unsupported")
    if loaded is None:
        verdict = FAIL
        causes.append("no translated-shaders line (Vulkan device not reached?)" if encoding else
                      "the log could not be read")
    elif loaded["total"] != local:
        verdict = max(verdict, WARN, key=RANK.get)
        causes.append(f"loaded {loaded['total']} modules, the work folder holds {local} (another folder or build?)")
    if exit_code not in (None, 0):
        verdict = FAIL
        causes.append(f"exit code {exit_code}")
    return {"name": "runtime", "verdict": verdict, "cause": "; ".join(causes) or "-", "loaded": loaded,
            "descriptor_table": table, "slot_failures": dict(slots.most_common(5)),
            "d3d12_problems": dict(trace.most_common(5)), "exit_code": exit_code, "ending": ending,
            "log_encoding": encoding,
            **{k: counts[k] for k in ("no_translated_compute", "compute_builds", "slow_compute_builds",
                                      "compile_gate_waits", "device_fault_lines", "device_lost_lines",
                                      "stalled_dispatch_lines")}}


def check(work: str, log: str | None = None, probe: bool = True, show_paths: bool = False) -> dict:
    tools = tools_block(work, probe)
    majors = tools.pop("_majors")   # the llvm-mc major each family was disassembled with
    build = parse_build_log(os.path.join(work, "logs", "shaders.log"), reason_templates())
    families = [family_block(work, key, label, folder, sub, build, majors.get(key))
                for key, label, folder, sub, _ in FAMILIES]
    blocks = [tools, *families, contract_block(work, families)]
    if log:
        blocks.append(runtime_block(log, families))
    worst = max(RANK[b["verdict"]] for b in blocks)
    weakest = next(b for b in blocks if RANK[b["verdict"]] == worst)
    report = {"blocks": blocks,
              "weakest": ({"name": weakest["name"], "verdict": weakest["verdict"], "cause": weakest["cause"]}
                          if worst >= 0 else {"name": "none", "verdict": NA, "cause": "nothing to check"})}
    if show_paths:
        report.update(work=work, log=log)
    return report


# ------------------------------------------------------------------------------------------------ text output
def fmt_counter(d: dict, joiner: str = "") -> str:
    sep = f" {joiner} " if joiner else " "
    return ", ".join(f"{k}{sep}{v}" for k, v in d.items()) or "none"


def render(report: dict) -> str:
    out = ["chain check" + (f" of {report['work']}" if report.get("work") else "")
           + (f" with log {report['log']}" if report.get("log") else "")]
    for b in report["blocks"]:
        out.append(f"[{b['verdict']}] {b['name']}" + (f"  {b['label']}" if b.get("label") else ""))
        if b["verdict"] == NA:
            pass
        elif b["name"] == "tools":
            recorded = fmt_counter(b["llvm_mc_recorded"], "in")
            amdgcn = {True: "yes", False: "NO", None: "n/a"}[b["llvm_mc_amdgpu"]]
            out.append(f"  llvm-mc     recorded {recorded} ({b['llvm_mc_records']} records, "
                       f"amdgcn {amdgcn}, via {'/'.join(b['llvm_mc_via']) or 'n/a'}); "
                       f"LLVM_MC now {b['llvm_mc_current'] or 'unreadable'}")
            out.append(f"  spirv-val   {b['spirv_val']}")
            local = fmt_counter(b["translator_local"], "in")
            out.append(f"  translator  local {local}; checkout {b['translator_repo']}"
                       f"{' at ' + b['repo_head'] if b['repo_head'] else ''}; stale: "
                       f"{', '.join(b['translator_stale']) or 'none'}")
        elif b["name"] == "contract":
            out.append(f"  modules     {fmt_counter(b['modules'])}; translator default {b['translator_default_table']}")
            out.append(f"  folders     present {', '.join(b['folders_present']) or 'none'}; absent "
                       f"{', '.join(b['folders_absent']) or 'none'}")
        elif b["name"] == "runtime":
            ld = b["loaded"]
            out.append("  loaded      " + (fmt_counter(ld) if ld else "no translated-shaders line")
                       + f"; descriptor table {b['descriptor_table'] or '?'}")
            out.append(f"  slot fails  {fmt_counter(b['slot_failures'])}; no translated compute shader "
                       f"{b['no_translated_compute']}")
            out.append(f"  d3d12       {fmt_counter(b['d3d12_problems'])}")
            out.append(f"  pipelines   compute builds {b['compute_builds']} (slow {b['slow_compute_builds']}); "
                       f"compile gate waits {b['compile_gate_waits']}")
            out.append(f"  device      fault lines {b['device_fault_lines']}, device lost lines "
                       f"{b['device_lost_lines']}, stalled dispatch lines {b['stalled_dispatch_lines']}; exit code "
                       f"{'none logged' if b['exit_code'] is None else b['exit_code']}"
                       + (f"; boot.ps1: {b['ending']}" if b['ending'] else ""))
        else:
            def n(key):
                return "none" if b.get(key) is None else b[key]
            rej = b.get("listings_with_rejected_words")
            out.append(f"  expected {n('expected')}  listings {n('listings')}"
                       + (f" ({rej} with rejected words)" if rej is not None else "")
                       + f"  translated {n('translated')} ({b.get('percent', '-')})  failed {b.get('failed', '-')}"
                       + (f"  index without .spv {b['modules_missing']}" if b.get("modules_missing") else ""))
            if b.get("reasons"):
                out.append("  reasons     " + "; ".join(f"{n} {r}" for n, r in b["reasons"]))
        out.append(f"  cause: {b['cause']}")
    w = report["weakest"]
    out.append(f"weakest link: {w['name']} [{w['verdict']}]: {w['cause']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One page of counts for the shader chain of a work folder.",
                                 epilog=VERDICTS)
    ap.add_argument("work", help="make_game.py's work folder (holds shaders/ and logs/)")
    ap.add_argument("--log", help="a gta6rt stderr log of one run (scripts/windows-boot/boot.ps1's <Name>.log)")
    ap.add_argument("--no-probe", action="store_true", help="run nothing: no llvm-mc, spirv-val or git --version")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a block is FAIL (default: 0, a report)")
    ap.add_argument("--show-paths", action="store_true",
                    help="name the work folder and the log in the report (not for publishing)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.work):
        ap.error(f"no work folder at {a.work}")
    if a.log and not os.path.isfile(a.log):
        ap.error(f"no log at {a.log}")
    report = check(a.work, a.log, probe=not a.no_probe, show_paths=a.show_paths)
    print(json.dumps(report, indent=1) if a.json else render(report))
    return 1 if a.strict and any(b["verdict"] == FAIL for b in report["blocks"]) else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    sys.exit(main())
