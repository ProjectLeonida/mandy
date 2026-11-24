r"""Write the BC6H/BC7 BCPack history sidecar the runtime decoder reads (stdlib only).

The built-in BC6H and BC7 history of BCPack is not in the repository: tools/bcpack_tables_bptc.py knows none of it.
A user who has calibrated it from their own copy (tools/bcpack_history_calib.py, the `--history` JSON of
tools/bcpack.py) can hand it to the runtime decoder (runtime/src/bcpack.h, `read_history_sidecar`) through this file:
the history JSON is read by the same loaders tools/bcpack.py uses (`load_bc6h`, `load_bc7`, so a file, a folder
holding bc6h.json / bc7.json, or a dict all work, with the same merging and mid-range fill), and written as fixed
little-endian tables the C++ reads without a JSON parser.

The layout is "BCPHIST1" (8 bytes), then 256 BC6H entries of 70 bytes:
    12 x uint16  endpoint history (3 channels x w, x, y, z)
    16 x uint8   index history
    12 x uint8   endpoint confirmed bits (3 channels x 4 quantities)
    16 x uint8   index confirmed bits
    2  x uint8   endpoint class, index class (0 unknown, 1 fitted, 2 last resort, 3 confirmed)
then 256 BC7 entries of 168 bytes: endpoints, endpoint bits, endpoint confirmed bits (24 bytes each), indices, index
bits, index confirmed bits (32 bytes each): 60,936 bytes in all. It is the profile format of the integration branch
in #984 (tools/gen_bcpack_runtime_tables.py --profile), byte for byte, so one file serves both.

The sidecar is made from the user's own data: keep it on their machine, never commit or post it.

Usage:
    python -I tools/bcpack_history_sidecar.py <history.json | folder> <out.bin>
"""
import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcpack_tables_bptc as bptc  # noqa: E402

MAGIC = b"BCPHIST1"
BC6H_ENTRY = 70
BC7_ENTRY = 168
SIZE = len(MAGIC) + 256 * (BC6H_ENTRY + BC7_ENTRY)


def render(history):
    """The sidecar bytes for `history` (a JSON path, a folder or a dict, as tools/bcpack.py takes it)."""
    ep, ix, ec, ic, ep_cb, ix_cb = bptc.load_bc6h(history, with_confirmed_bits=True)
    b7 = bptc.load_bc7(history)
    out = bytearray(MAGIC)
    for i in range(256):
        out += struct.pack("<12H", *(v & 0xFFFF for channel in ep[i] for v in channel))
        out += bytes(v & 0xFF for v in ix[i])
        out += bytes(v for channel in ep_cb[i] for v in channel)
        out += bytes(ix_cb[i])
        out += bytes((ec[i], ic[i]))
    for i in range(256):
        for key in ("history_endpoints", "history_endpoint_bits", "history_endpoint_confirmed_bits",
                    "history_indices", "history_index_bits", "history_index_confirmed_bits"):
            out += bytes(v & 0xFF for v in b7[key][i])
    if len(out) != SIZE:
        raise SystemExit(f"internal error: wrote {len(out)} bytes, expected {SIZE}")
    return bytes(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("history", help="the calibrated history: a JSON file or a folder with bc6h.json / bc7.json")
    ap.add_argument("out", help="the sidecar to write (keep it on this machine)")
    args = ap.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.history):
        print("ERROR: the sidecar would overwrite its own input")
        return 2
    try:
        data = render(args.history)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}")
        return 2
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"wrote {args.out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
