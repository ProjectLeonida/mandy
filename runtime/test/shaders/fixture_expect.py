"""Expected words a compute fixture declares in its own header, read by both GPU runners.

A fixture opts in with header lines in its leading comment block:

    ; expect: 0x12345678 0x12345678 0 0xeeeeeeee
    ; groups: 1

`expect:` lists the first words `vk_compute_test <name>.spv <groups> dump` must print (hex or decimal; the line may be
repeated to continue the list); `groups:` is optional and defaults to 1. Such a fixture needs no row in
`scripts/run_gpu_tests.py` or `check.py`: both find it here, so two PRs that add unrelated fixtures no longer edit the
same shared lists. It must run on any subgroup width (one thread, or per-lane results); fixtures that need a 64-wide
subgroup, a texture or a custom check keep their rows in the runners. Standard library only.
"""
import os
import re

_EXPECT = re.compile(r"^;\s*expect:\s*(.*?)\s*$")
_GROUPS = re.compile(r"^;\s*groups:\s*(\S+)\s*$")


class FixtureHeaderError(ValueError):
    pass


def header_expect(path):
    """(words, groups) from the fixture's leading `;` comment block, or None when it declares no `expect:`."""
    words, groups, seen = [], "1", False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s.startswith(";"):
                break  # the header ends at the first instruction
            if m := _EXPECT.match(s):
                seen = True
                for token in m.group(1).replace(",", " ").split():
                    try:
                        words.append(int(token, 0) & 0xFFFFFFFF)
                    except ValueError:
                        raise FixtureHeaderError(f"{path}: expect word {token!r} is not a number") from None
            elif m := _GROUPS.match(s):
                groups = m.group(1)
    if not seen:
        return None
    if not words:
        raise FixtureHeaderError(f"{path}: 'expect:' lists no words")
    return words, groups


def expect_fixtures(directory):
    """{name: (words, groups)} for every `cs_*.asm` in `directory` that declares `expect:`, in name order."""
    found = {}
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("cs_") and fname.endswith(".asm"):
            got = header_expect(os.path.join(directory, fname))
            if got:
                found[fname[:-4]] = got
    return found
