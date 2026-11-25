#!/usr/bin/env python3
"""Fill fspx_state_sha256 in a pipeline-library translation index from its .state sidecars.

A library program translated before make_game.py emitted its kind-0 sidecar has an index entry with the
fspx_code_sha256 but no fspx_state_sha256. The runtime then refuses the program when the game creates it with a kind-0
description (docs/fspx.md, "Build and run-time lookup"): every entry for that code is filtered out by the state key, so
the pipeline becomes a stub. The sidecar is the state the translation was made from, so recording its SHA-256 makes the
entry match again.

The tool is for an existing work folder whose library index predates the sidecars (the programs no longer translate in
the current decoder, so a retranslation would drop them). A fresh translation already records the key
(tools/shader_meta.py); this only fills entries that lack it and leaves every other entry alone.

    python -I tools/fill_fspx_library_states.py <work>/shaders/spirv/library/index.json <work>/shaders/library/fspx
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile


def fill_states(index, sidecar_dir):
    """Add fspx_state_sha256 to every FSPX entry that lacks it and has a sidecar; return (added, without_sidecar)."""
    added = without_sidecar = 0
    for name, meta in index.items():
        if not isinstance(meta, dict) or "fspx_code_sha256" not in meta or meta.get("fspx_state_sha256"):
            continue
        path = os.path.join(sidecar_dir, name + ".state")
        try:
            with open(path, "rb") as state:
                digest = hashlib.sha256(state.read()).hexdigest()
        except OSError:
            without_sidecar += 1
            continue
        meta["fspx_state_sha256"] = digest
        added += 1
    return added, without_sidecar


def publish(path, document):
    """Write the index atomically, with the translator's formatting and the tool conventions' line endings."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    handle, temporary = tempfile.mkstemp(prefix=".index-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as sink:
            json.dump(document, sink, indent=0, sort_keys=True)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("index", help="<work>/shaders/spirv/library/index.json")
    parser.add_argument("sidecars", help="<work>/shaders/library/fspx (the <name>.state files)")
    args = parser.parse_args(argv)
    try:
        with open(args.index, encoding="utf-8") as source:
            index = json.load(source)
    except (OSError, ValueError) as error:
        print(f"cannot read {args.index}: {error}", file=sys.stderr)
        return 1
    if not isinstance(index, dict):
        print(f"{args.index} is not a JSON object", file=sys.stderr)
        return 1
    added, without_sidecar = fill_states(index, args.sidecars)
    publish(args.index, index)
    print(f"filled fspx_state_sha256 for {added} entries; {without_sidecar} had no .state sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
