"""Convert one exported cue_regions.json to per-channel Audacity label tracks."""

import argparse
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

MAX_JSON_DEPTH = 64


def _check_json_depth(text: str, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Fail fast before json.loads; CPython's C stack limit varies across OSes."""
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
            if depth > max_depth:
                raise ValueError("cue sheet JSON nesting is too deep")
        elif ch in "}]" and depth > 0:
            depth -= 1


def convert(sheet, output):
    text = sheet.read_text(encoding="utf-8")
    _check_json_depth(text)
    try:
        data = json.loads(text)
    except RecursionError:
        raise ValueError("cue sheet JSON nesting is too deep") from None
    if (not isinstance(data, dict) or data.get("sample_rate") != 48000
            or not isinstance(data.get("regions"), list)):
        raise ValueError("expected a 48 kHz cue sheet with a regions list")

    tracks = {}
    for ordinal, row in enumerate(data["regions"], 1):
        if not isinstance(row, dict):
            raise ValueError(f"invalid cue at row {ordinal}")
        channel = row.get("file")
        cue = row.get("name_hash")
        start = row.get("start_sample")
        end = row.get("end_sample")
        if (not isinstance(channel, str) or not re.fullmatch(r"[0-9A-F]{8}\.ogg", channel)
                or not isinstance(cue, str) or not re.fullmatch(r"[0-9A-F]{8}", cue)
                or type(start) is not int or type(end) is not int or not 0 <= start <= end <= 0xffffffff):
            raise ValueError(f"invalid channel, cue hash or sample interval at row {ordinal}")
        tracks.setdefault(channel, []).append(
            f"{start / 48000:.9f}\t{end / 48000:.9f}\t{cue} #{ordinal}\n")

    # Publish a whole bank together, including an empty bank.
    if output.exists():
        raise ValueError("choose a new output directory for this bank")
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "labels"
        staged.mkdir()
        for channel, lines in tracks.items():
            (staged / Path(channel).with_suffix(".txt")).write_text(
                "".join(lines), encoding="utf-8", newline="\n")
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", type=Path, help="one bank's exported cue_regions.json")
    parser.add_argument("output", type=Path, help="new directory for this bank's label tracks")
    args = parser.parse_args()
    try:
        convert(args.sheet, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    main()
