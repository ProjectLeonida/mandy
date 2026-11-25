r"""Run the translator regression suite (tests/semantic) one process per test file, several files at a time.

    python -I scripts/run_semantic_tests.py tests/semantic          # one file per core
    python -I scripts/run_semantic_tests.py tests/semantic -p 'test_image_*.py'  # a matching subset
    python -I scripts/run_semantic_tests.py tests/semantic -j 4     # four at a time
    python -I scripts/run_semantic_tests.py tests/semantic -v       # also print the output of the files that pass
    python -I scripts/run_semantic_tests.py tests/semantic --timeout 300  # optional subprocess time limit

`-p` / `--pattern` selects test filenames using unittest discover's glob syntax (default `test*.py`). Quote the
pattern so the shell does not expand it. Only Python module filenames are selected, including inside packages;
the pattern matches their basenames, not directory paths. Package `__init__.py` modules are always included,
because discover imports them and runs any TestCase they define regardless of the pattern. No matching files is an
error in serial and parallel runs.

It runs the files `python -m unittest discover -s tests/semantic` finds (`test*.py` by default, and packages below it), each in its own process from that folder, so a file sees the same import path as under discover.

The files can run side by side because each one writes only into its own temporary folder; a new test must do the same. With -j 1,
or on a single core, it runs that discover command itself. Prints the output of every failing file and one summary
line, and exits 1 if a file fails, none is found, or no test ran. Standard library only.

The count of tests comes from unittest's own result object, not from its output: a module that writes a line shaped like unittest's summary to stderr - a nested runner, or an atexit handler that runs after unittest's own summary - cannot change it.

In a parallel run the processes share the modules they compile through a temporary folder that is removed when the
run ends, so the translator is compiled once per run instead of once per file (BYTECODE_CACHE_PROGRAM has the
reason). Nothing is written into the tree for it, and a file's own `sys.dont_write_bytecode` still keeps
`__pycache__` out of it.

`--timeout SECONDS` limits each module subprocess in parallel mode, or the entire discover subprocess with
`-j 1`. The value must be positive and at most 86400 seconds (24 hours), keeping it within platform wait limits.
A timeout fails the run and reports any output already produced; other parallel modules keep running.
The default has no time limit. This stops the direct test subprocess, not any descendants it may have started.
"""
import argparse
import concurrent.futures
import fnmatch
import json
import locale
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

MAX_TIMEOUT_SECONDS = 24 * 60 * 60

# The child processes running right now: on SIGINT / Ctrl+C, the main thread terminates them so no orphans linger.
_active_processes = set()
_processes_lock = threading.Lock()


def _register_process(proc):
    with _processes_lock:
        _active_processes.add(proc)


def _unregister_process(proc):
    with _processes_lock:
        _active_processes.discard(proc)


def _kill_process_tree(proc):
    if proc is None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass


def _terminate_active_processes():
    with _processes_lock:
        procs = list(_active_processes)
    for p in procs:
        try:
            p.terminate()
        except OSError:
            pass
    for p in procs:
        _kill_process_tree(p)
        try:
            p.wait(timeout=0.2)
        except (subprocess.TimeoutExpired, OSError):
            pass

# Where the children find the bytecode another child has already compiled (the folder is the run's own, see main()).
BYTECODE_CACHE_VARIABLE = "LEONIDA_BYTECODE_CACHE"

# Installed in every child before it imports its test module. A source module whose own bytecode file is missing or
# stale, or was never written because the importing file sets `sys.dont_write_bytecode` (most of this suite does, to
# keep `__pycache__` out of the tree), is compiled by the first child that needs it and read back by all the others.
#
# It matters because CPython tokenizes a source *string* in time quadratic in its f-string replacement fields: since
# 3.12 every `{` copies the rest of the buffer (`update_fstring_expr` in 3.12's Parser/tokenizer.c). Importing
# tools/rdna2spirv.py, 1.2 MB with some 4,800 f-strings, takes 7.2 s on 3.14.7 and 7.5 s on 3.13.13 (Windows 11; 0.1 s
# of it is code generation) against under 0.1 s from bytecode, and each file of the suite paid that. Run as a script
# the same file starts in 0.4 s, because a file is tokenized line by line.
#
# `source_to_code` is what the import system calls when it has no valid bytecode file for a module, so a module whose
# `__pycache__` entry is current never comes here. An entry is named after the path and the bytes of the source it was
# compiled from, so a source that changes during a run, or a second copy of tools/, gets one of its own and no entry is
# ever stale; an entry that cannot be read or written just means compiling, as before.
BYTECODE_CACHE_PROGRAM = (
    "import importlib.machinery, importlib.util, marshal, os, sys\n"
    "class SharedBytecodeLoader(importlib.machinery.SourceFileLoader):\n"
    "    def source_to_code(self, data, path, **options):\n"
    "        if options or not isinstance(data, bytes):\n"
    "            return super().source_to_code(data, path, **options)\n"
    "        key = importlib.util.source_hash(os.fsencode(path) + b'\\0' + data).hex()\n"
    "        entry = os.path.join(SHARED_BYTECODE, '%s-%d-%s.marshal' % (\n"
    "            self.name.rpartition('.')[2], sys.flags.optimize, key))\n"
    "        try:\n"
    "            with open(entry, 'rb') as handle:\n"
    "                code = marshal.load(handle)\n"
    "        except (OSError, EOFError, ValueError, TypeError):\n"
    "            code = None\n"
    "        if isinstance(code, type(compile('', path, 'exec'))):\n"
    "            return code\n"
    "        code = super().source_to_code(data, path)\n"
    "        mine = '%s.%d' % (entry, os.getpid())\n"
    "        try:\n"
    "            with open(mine, 'wb') as handle:\n"
    "                marshal.dump(code, handle)\n"
    "            os.replace(mine, entry)\n"
    "        except OSError:\n"
    "            try:\n"
    "                os.remove(mine)\n"
    "            except OSError:\n"
    "                pass\n"
    "        return code\n"
    "class SharedBytecodeFinder:\n"
    "    @staticmethod\n"
    "    def find_spec(name, path=None, target=None):\n"
    "        spec = importlib.machinery.PathFinder.find_spec(name, path, target)\n"
    "        if spec is not None and type(spec.loader) is importlib.machinery.SourceFileLoader:\n"
    "            spec.loader = SharedBytecodeLoader(spec.loader.name, spec.loader.path)\n"
    "        return spec\n"
    f"SHARED_BYTECODE = os.environ.get({BYTECODE_CACHE_VARIABLE!r})\n"
    # Ahead of the path finder only: built-in and frozen modules keep their own finders.
    "if SHARED_BYTECODE and importlib.machinery.PathFinder in sys.meta_path:\n"
    "    sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), SharedBytecodeFinder)\n")

# The child runs the module through unittest's own program and writes the result to a file, so the count of tests
# comes from the result object and nothing the module writes can change it. The exit codes are unittest's own: 1
# when a test failed, 5 when nothing ran at all, 0 otherwise. The module is left unset, as `python -m unittest` leaves it, so
# the name is imported and not looked up as an attribute of the program itself.
CHILD_PROGRAM = BYTECODE_CACHE_PROGRAM + (
    "import json, os, sys, unittest\n"
    "program = unittest.TestProgram(module=None, argv=['unittest'] + sys.argv[1:], exit=False)\n"
    "result = program.result\n"
    "with open(os.environ['LEONIDA_TEST_COUNT'], 'w', encoding='utf-8') as handle:\n"
    "    json.dump({'testsRun': result.testsRun}, handle)\n"
    "if not result.wasSuccessful():\n"
    "    sys.exit(1)\n"
    "elif result.testsRun == 0 and len(result.skipped) == 0:\n"
    "    sys.exit(5)\n"
    "sys.exit(0)\n")


def module_path(directory, module):
    """Source path of a dotted module name, including a package's `__init__.py`."""
    parts = module.split(".")
    if parts[-1] == "__init__":
        return os.path.join(directory, *parts[:-1], "__init__.py")
    return os.path.join(directory, *parts) + ".py"


def test_modules(directory, pattern="test*.py"):
    """Dotted names of the test modules unittest discover finds under `directory` (it enters packages only)."""
    names = []
    for base, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if os.path.isfile(os.path.join(base, d, "__init__.py")))
        prefix = os.path.relpath(base, directory).replace(os.sep, ".")
        # Discover imports every package below its start folder even when no filename in it matches the pattern,
        # and loads TestCase classes defined in the package initializer. Parallel runs must see those too.
        if prefix != "." and "__init__.py" in files:
            init = f"{prefix}.__init__"
            if init not in names:
                names.append(init)
        for name in sorted(files):
            if re.fullmatch(r"[_a-z]\w*\.py", name, re.IGNORECASE) and fnmatch.fnmatch(name, pattern):
                module = name[:-3] if prefix == "." else f"{prefix}.{name[:-3]}"
                if module not in names:
                    names.append(module)
    return names


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"timeout must be a positive finite number of seconds no greater than {MAX_TIMEOUT_SECONDS} (24 hours)")
    return seconds


def read_count(path):
    """The number of tests the child ran, as unittest's own result object reported it."""
    try:
        with open(path, encoding="utf-8") as handle:
            return int(json.load(handle)["testsRun"])
    except (OSError, ValueError, KeyError, TypeError):
        # The child wrote no result: it did not finish, so no test is known to have run.
        return 0


def run_module(directory, module, verbose, timeout=None, bytecode_cache=None):
    """(return code, tests run, output) of `python -m unittest <target>` run from `directory`.

    `bytecode_cache` is the folder this child shares compiled modules through (BYTECODE_CACHE_PROGRAM); without one
    it compiles what it imports itself."""
    # `python -m unittest pkg.__init__` imports pkg and then runs the same file again as pkg.__init__, so a
    # registration in the initializer happens twice. Running the package itself imports it once; module_path() is
    # still the initializer's path, so the jobs stay sorted by the file they run.
    target = module[:-len(".__init__")] if module.endswith(".__init__") else module
    cmd = [sys.executable, "-c", CHILD_PROGRAM, *(["-v"] if verbose else []), target]
    with tempfile.TemporaryDirectory() as scratch:
        count_path = os.path.join(scratch, "count.json")
        env = dict(os.environ, LEONIDA_TEST_COUNT=count_path)
        # Never the folder of a runner further up (this runner started from a test of another run): it is not ours
        # to fill, and it is removed when that run ends.
        env.pop(BYTECODE_CACHE_VARIABLE, None)
        if bytecode_cache is not None:
            env[BYTECODE_CACHE_VARIABLE] = bytecode_cache
        proc = subprocess.Popen(cmd, cwd=directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors="replace")
        _register_process(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = proc.communicate()
            # TimeoutExpired retains bytes even when text=True.
            parts = [chunk.decode(locale.getpreferredencoding(False), errors="replace") if isinstance(chunk, bytes)
                     else chunk for chunk in (stdout, stderr) if chunk]
            out = "".join(parts)
            return 1, 0, out + f"\nTIMED OUT after {timeout:g} s: {module}\n"
        except BaseException:
            _kill_process_tree(proc)
            try:
                proc.wait(timeout=0.5)
            except (subprocess.TimeoutExpired, OSError):
                pass
            raise
        finally:
            _unregister_process(proc)
        count = read_count(count_path)
    out = stdout + stderr
    # Discover imports every package initializer, but an empty one has no tests to run. It is included in the
    # module list so parallel runs see TestCase classes when there are any; without them, it is not a failure.
    if proc.returncode == 5 and count == 0 and module.endswith(".__init__") and "NO TESTS RAN" in stderr:
        return 0, 0, out
    return proc.returncode, count, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="the suite's folder, tests/semantic")
    ap.add_argument("-p", "--pattern", default="test*.py", help="test filename glob (default: test*.py)")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="files at a time (default: cores)")
    ap.add_argument("-v", "--verbose", action="store_true", help="unittest -v, and print passing files' output")
    ap.add_argument("--timeout", type=positive_seconds, default=None, metavar="SECONDS",
                    help=f"positive finite time limit up to {MAX_TIMEOUT_SECONDS} seconds (24 hours) per module "
                         "subprocess, or for the whole discover subprocess with -j 1 (default: no limit)")
    a = ap.parse_args(argv)
    directory = os.path.abspath(a.directory)
    modules = test_modules(directory, a.pattern)
    if not modules:
        print(f"no {a.pattern} files in {directory}")
        return 1
    if a.jobs <= 1:
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", directory,
               "-p", a.pattern, *(["-v"] if a.verbose else [])]
        proc = subprocess.Popen(cmd)
        _register_process(proc)
        start_time = time.monotonic()
        try:
            while True:
                try:
                    return proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    if a.timeout is not None and time.monotonic() - start_time >= a.timeout:
                        raise
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            print(f"TIMED OUT after {a.timeout:g} s: unittest discover in {directory}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            _kill_process_tree(proc)
            try:
                proc.wait(timeout=0.5)
            except (subprocess.TimeoutExpired, OSError):
                pass
            print("interrupted", file=sys.stderr)
            return 130
        except BaseException:
            _kill_process_tree(proc)
            raise
        finally:
            _unregister_process(proc)
    # Largest files first: they tend to run longest, and one started last would set the wall time alone.
    modules.sort(key=lambda m: -os.path.getsize(module_path(directory, m)))
    # One folder of compiled modules for the whole run, outside the tree and gone when the run ends (a file a
    # stopped child still holds open on Windows must not turn a finished run into a traceback).
    with tempfile.TemporaryDirectory(prefix="leonida-bytecode-", ignore_cleanup_errors=True) as bytecode_cache:
        return run_parallel(directory, modules, a, bytecode_cache)


def run_parallel(directory, modules, a, bytecode_cache):
    """Exit code of the run: the modules, `a.jobs` at a time, each in its own process."""
    start, total, failed = time.monotonic(), 0, []
    pool = concurrent.futures.ThreadPoolExecutor(a.jobs)
    jobs = {}
    try:
        for m in modules:
            jobs[pool.submit(run_module, directory, m, a.verbose, a.timeout, bytecode_cache)] = m
        pending = set(jobs)
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED)
            for job in done:
                pending.remove(job)
                code, ran, out = job.result()
                total += ran
                if code:
                    failed.append(jobs[job])
                    print(f"==== FAILED: {jobs[job]}\n{out}", flush=True)
                elif a.verbose:
                    print(f"==== {jobs[job]}\n{out}", flush=True)
    except KeyboardInterrupt:
        for job in jobs:
            job.cancel()
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
        _terminate_active_processes()
        print("interrupted", file=sys.stderr)
        return 130
    except BaseException:
        for job in jobs:
            job.cancel()
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
        _terminate_active_processes()
        raise
    else:
        pool.shutdown(wait=True)
    took = time.monotonic() - start
    summary = f"{len(modules)} files, {total} tests, {len(failed)} files failed in {took:.0f} s with {a.jobs} jobs"
    print(summary + (": " + ", ".join(sorted(failed)) if failed else ""))
    if total == 0:
        # An empty package initializer is a job that legitimately runs no tests; a run where every job did is not a
        # pass (discover, which -j 1 forwards to, exits 5 for it).
        print(f"no tests ran in {directory}")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
