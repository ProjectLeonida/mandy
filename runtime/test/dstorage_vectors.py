"""DirectStorage request vectors from independent synthetic GDK goldens."""
import argparse
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zlib


def write(out):
    source = Path(__file__).resolve().parents[2] / "tests/fixtures/bcpack-standalone.txt"
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        codec, name, packed, expected = line.split()
        if not name.endswith("-request"):
            continue
        raw, plain = bytes.fromhex(packed), bytes.fromhex(expected)
        assert len(raw) < len(plain) and len(zlib.compress(raw)) < len(plain)
        for label, data in (("raw", raw), ("plain", plain), ("wrapped", zlib.compress(raw))):
            print(f"static constexpr uint8_t {label}{codec}[] = {{", file=out)
            for i in range(0, len(data), 24):
                print("    " + ", ".join(map(str, data[i:i+24])) + ",", file=out)
            print("};", file=out)
    print("static constexpr Vector vectors[] = {", file=out)
    for codec in range(1, 8):
        print(f"    {{{codec}, raw{codec}, plain{codec}, wrapped{codec}}},", file=out)
    print("};", file=out)


def _validate_output_path(raw: str, parser: argparse.ArgumentParser) -> Path:
    if not raw.strip():
        parser.error("output path must not be empty")
    if raw.endswith(("/", "\\")) or (os.sep and raw.endswith(os.sep)):
        parser.error(f"output must be a file path, not a directory: {raw}")
    for cand in (Path(os.path.abspath(raw)), Path(os.path.realpath(raw))):
        if (cand.exists() or cand.is_symlink()) and cand.is_dir():
            parser.error(f"output path is a directory: {raw}")
        anc = cand.parent
        while str(anc) and not (anc.exists() or anc.is_symlink()):
            if anc.parent == anc:
                break
            anc = anc.parent
        if str(anc) and not anc.is_dir():
            parser.error(f"output parent is not a directory: {anc}")
    return Path(raw)


def write_file_atomic(path: Path | str, text: str) -> None:
    """Atomically write UTF-8 LF text to `path`, preserving existing content on failure."""
    dest = Path(os.path.realpath(path))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".dstorage-vectors-", suffix=".tmp")
    try:
        try:
            f = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(fd)
            raise
        with f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if os.name == "posix":
            try:
                shutil.copymode(dest, tmp)
            except FileNotFoundError:
                pass
        for delay in (0.01, 0.02, 0.04) if os.name == "nt" else ():
            try:
                os.replace(tmp, dest)
                return
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33):
                    raise
                time.sleep(delay)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="header file to write")
    args = parser.parse_args(argv)
    out_path = _validate_output_path(args.output, parser)
    buf = io.StringIO()
    write(buf)
    try:
        write_file_atomic(out_path, buf.getvalue())
    except OSError as exc:
        print(f"ERROR: cannot write {out_path}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
