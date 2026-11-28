"""Read shader names from a completed carve or unpack index and check disassembly completeness.

An index.json saved with a UTF-8 BOM (Windows PowerShell 5.1, old Notepad) is read, not refused.
"""

import json
import ntpath
import os
import posixpath
import re

LLVM_MC_FOOTER = "\n; --- llvm-mc messages ---\n"
COMPLETE_ENDPGM = re.compile(r"^\s*s_endpgm(?:\s*;\s*encoding:\s*\[[^\]]+\])?\s*$")
COMPLETE_ENDPGM_ENCODED = re.compile(r"^\s*s_endpgm\s*;\s*encoding:\s*\[[^\]]+\]\s*$")


def listing_is_complete(text: str, *, require_encoding: bool | None = None) -> bool:
    """Whether disassembly text ends cleanly with a complete trailing s_endpgm instruction."""
    if not text or not text.endswith("\n"):
        return False
    code_part = text.split(LLVM_MC_FOOTER, 1)[0]
    if require_encoding is None:
        require_encoding = "; encoding: [" in code_part
    last_code_line = None
    for line in reversed(code_part.splitlines()):
        stripped = line.lstrip()
        if stripped and not stripped.startswith(";"):
            last_code_line = line
            break
    if last_code_line is None:
        return False
    pattern = COMPLETE_ENDPGM_ENCODED if require_encoding else COMPLETE_ENDPGM
    return pattern.match(last_code_line) is not None


def listing_file_is_complete(path: str, tail_bytes: int = 4096) -> bool:
    """Fast check that an existing .s disassembly file is non-empty and ends cleanly on s_endpgm."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return False
            if size <= tail_bytes:
                f.seek(0)
                data = f.read()
                if not data.endswith(b"\n"):
                    return False
                return listing_is_complete(data.decode("utf-8", errors="replace"))
            f.seek(size - tail_bytes)
            tail_data = f.read()
            if not tail_data.endswith(b"\n"):
                return False
            _, _, tail_text = tail_data.decode("utf-8", errors="replace").partition("\n")
            code_tail = tail_text.split(LLVM_MC_FOOTER, 1)[0]
            has_code_line = any(
                line.lstrip() and not line.lstrip().startswith(";")
                for line in code_tail.splitlines()
            )
            if not has_code_line:
                f.seek(0)
                return listing_is_complete(f.read().decode("utf-8", errors="replace"))
            require_encoding = True if "; encoding: [" in code_tail else None
            if require_encoding is None:
                f.seek(0)
                head_text = f.read(tail_bytes).decode("utf-8", errors="replace")
                require_encoding = "; encoding: [" in head_text
            return listing_is_complete(tail_text, require_encoding=require_encoding)
    except OSError:
        return False


def _stem_rows(path: str, field: str, suffix: str = "") -> list[tuple[str, dict]]:
    """Each entry of the index as (validated stem, row)."""
    try:
        # utf-8-sig: PowerShell 5.1 and old Notepad write JSON with a UTF-8 BOM.
        with open(path, encoding="utf-8-sig") as f:
            rows = json.load(f)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: invalid UTF-8: {exc}") from exc
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected an array of shader entries")
    out = []
    for row in rows:
        value = row.get(field) if isinstance(row, dict) else None
        if (
            not isinstance(value, str)
            or not value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or ":" in value
            or "/" in value
            or "\\" in value
            or ntpath.splitdrive(value)[0]
            or ntpath.basename(value) != value
            or posixpath.basename(value) != value
        ):
            raise ValueError(f"{path}: expected a shader basename in {field!r}")
        if suffix and (not value.endswith(suffix) or len(value) == len(suffix)):
            raise ValueError(f"{path}: expected a {suffix} filename in {field!r}")
        stem = value.removesuffix(suffix)
        if not stem or stem in (".", ".."):
            raise ValueError(f"{path}: expected a shader basename in {field!r}")
        out.append((stem, row))
    return out


def shader_stems(path: str, field: str, suffix: str = "") -> set[str]:
    return {stem for stem, _ in _stem_rows(path, field, suffix)}


def shader_stages(path: str, field: str, suffix: str = "") -> dict[str, str | None]:
    """The stage the index records for each shader, by stem. None where an entry has no stage string: the stage is
    then unknown, not absent, and the caller says so."""
    stages = {}
    for stem, row in _stem_rows(path, field, suffix):
        stage = row.get("stage")
        stages[stem] = stage if isinstance(stage, str) and stage else None
    return stages
