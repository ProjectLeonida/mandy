r"""Write runtime/test/bcpack_vectors.h: BCPack chunks with what tools/bcpack.py makes of them (stdlib only).

bcpack_test (runtime/test/bcpack_test.cpp) decodes every chunk here with the runtime's decoder (src/bcpack.cpp) and
compares with the Python decoder's answer, so the C++ port is checked against the reference without Python at test
time. Every chunk is synthetic: tests/test_bcpack.py's encoder over random lanes (no game files or texture bytes).

- Lane cases: random lanes of all six modes, some corrupted, cut short or with a byte appended, as
  tests/test_bcpack_native.py checks the Rust port. Expected: the lanes `_decode_lanes_raw` returns (with nothing
  but zeros after the bytes it used), or a refusal.
- Block cases: modes 1, 3, 4 and 5 with history distances that point at no block, at an earlier block or at a
  built-in entry (confirmed or not), and now and then a non-zero lane 1. Expected: `bcpack.decode`'s strict output
  (the default), or the error class it raises.

The pure-Python decoder is the reference (LEONIDA_BCPACK_NATIVE=0), so the Rust library does not take part.

Usage:
    python -I runtime/test/bcpack_vectors.py            # rewrite runtime/test/bcpack_vectors.h
    python -I runtime/test/bcpack_vectors.py --check    # exit 1 if the header is not what this writes
"""
import argparse
import os
import random
import sys

os.environ["LEONIDA_BCPACK_NATIVE"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import bcpack  # noqa: E402
import test_bcpack as synth  # noqa: E402

OUT = os.path.join(HERE, "bcpack_vectors.h")
SEED = 20261124

# runtime/src/bcpack.h Status, in declaration order.
OK, UNSUPPORTED, BAD_SIZE, BAD_STREAM, TRAILING, LANE1, BAD_SYMBOL, UNCONFIRMED = range(8)


def _lanes_reference(raw, codec, nsym):
    """`_decode_lanes_raw`'s lanes for `raw`, or None where it raises or leaves non-zero bytes."""
    try:
        lanes, used = bcpack._decode_lanes_raw(memoryview(raw), bcpack.LANE_WIDTHS[codec], nsym)
    except (bcpack.BcpackError, ZeroDivisionError, IndexError):
        return None
    return None if any(raw[used:]) else lanes


def _corrupt(rng, raw):
    raw = bytearray(raw)
    kind = rng.random()
    if kind < 0.25:
        for _ in range(rng.randint(1, 3)):
            raw[rng.randrange(len(raw))] = rng.randrange(256)
    elif kind < 0.35:
        del raw[rng.randrange(len(raw)):]
    elif kind < 0.42:
        raw += bytes([rng.randrange(1, 256)])
    return bytes(raw)


def _random_lanes(rng, codec, nsym):
    lanes = []
    for w in bcpack.LANE_WIDTHS[codec]:
        rate = rng.random() + 0.05
        lanes.append([min((1 << w) - 1, int(rng.expovariate(rate))) if rng.random() < 0.9
                      else rng.randrange(1 << w) for _ in range(nsym)])
    return lanes


def lane_cases(rng, count):
    out = []
    for _ in range(count):
        codec = rng.choice(sorted(bcpack.LANE_WIDTHS))
        nsym = rng.choice([1, 2, 7, 16]) * bcpack.BLOCK_BYTES[codec] // bcpack.ELEMENT_BYTES
        lanes = _random_lanes(rng, codec, nsym)
        q = rng.randrange(16) if rng.random() < 0.7 else 0
        raw = _corrupt(rng, synth.encode_lanes(lanes, codec, q=q))
        out.append((codec, nsym, raw, _lanes_reference(raw, codec, nsym)))
    return out


def _distance_code(rng, i):
    """A history distance code for block i: none, an earlier block, or a built-in entry (d > i)."""
    r = rng.random()
    if r < 0.5:
        d = 0
    elif r < 0.96:
        d = rng.randint(1, min(i, 255)) if i > 0 else 0
    else:
        d = rng.randint(i + 1, 255) if i < 255 else 0
    # The 8-bit code dist() and dist8() read back as d: even 2d up to 127, odd 2 * (256 - d) - 1 above.
    return 2 * d if d <= 127 else 2 * (256 - d) - 1


def block_cases(rng, count):
    out = []
    for _ in range(count):
        codec = rng.choice([1, 3, 4, 5])
        nblocks = rng.choice([1, 3, 8, 16])
        two = bcpack.BLOCK_BYTES[codec] == 16
        nsym = nblocks * (2 if two else 1)
        lanes = _random_lanes(rng, codec, nsym)
        lanes[1] = [0] * nsym if rng.random() > 0.06 else [rng.randrange(1, 16)] + [0] * (nsym - 1)
        for i in range(nblocks):
            code = _distance_code(rng, i)
            if two:  # lane 0 holds the code's even bits in element a and its odd bits in element b (`_inter8`)
                lanes[0][2 * i] = sum(((code >> (2 * k)) & 1) << k for k in range(4))
                lanes[0][2 * i + 1] = sum(((code >> (2 * k + 1)) & 1) << k for k in range(4))
            else:
                lanes[0][i] = code
        raw = synth.encode_lanes(lanes, codec, q=rng.randrange(16) if rng.random() < 0.7 else 0)
        if rng.random() < 0.12:
            raw = _corrupt(rng, raw)
        size = nblocks * bcpack.BLOCK_BYTES[codec]
        try:
            expected, status = bcpack.decode(raw, codec, size), OK
        except bcpack.BcpackProvisionalError:
            expected, status = None, UNCONFIRMED
        except bcpack.BcpackError as e:
            text = str(e)
            expected = None
            # "lane 10 ... input ran out" is a stream error: match the rebuilders' own message, not "lane 1".
            status = LANE1 if "lane 1 is non-zero" in text else TRAILING if "left over" in text else BAD_STREAM
        except (ZeroDivisionError, IndexError):
            expected, status = None, BAD_STREAM
        out.append((codec, raw, size, status, expected))
    return out


def _bytes(name, data, ctype="std::uint8_t"):
    body = ", ".join(f"0x{b:02x}" for b in data) if data else "0"
    return f"inline constexpr {ctype} {name}[] = {{{body}}};"


def render():
    rng = random.Random(SEED)
    lanes = lane_cases(rng, 120)
    blocks = block_cases(rng, 90)
    lines = [
        "// Generated by runtime/test/bcpack_vectors.py. Do not edit by hand (tests/test_bcpack_vectors.py checks it).",
        "// Synthetic BCPack chunks (tests/test_bcpack.py's encoder over random lanes; no game data) and what the",
        "// Python reference decoder, tools/bcpack.py, makes of them.",
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace bcpack_vectors {",
        "",
        "struct LaneCase {",
        "    std::uint32_t mode;",
        "    std::size_t count;  // symbols per lane",
        "    const std::uint8_t* raw;",
        "    std::size_t raw_size;",
        "    bool ok;                       // the Python lane model accepts the stream",
        "    const std::uint16_t* symbols;  // lane-major, count per lane, when ok",
        "};",
        "",
        "struct BlockCase {",
        "    std::uint32_t mode;",
        "    const std::uint8_t* raw;",
        "    std::size_t raw_size;",
        "    std::size_t out_size;",
        "    int status;                 // leonida::bcpack::Status, as an int",
        "    const std::uint8_t* blocks;  // out_size bytes when status is 0",
        "};",
        "",
    ]
    for k, (codec, nsym, raw, ref) in enumerate(lanes):
        lines.append(_bytes(f"lane_raw_{k}", raw))
        flat = [s for lane in ref for s in lane] if ref is not None else []
        body = ", ".join(str(s) for s in flat) if flat else "0"
        lines.append(f"inline constexpr std::uint16_t lane_sym_{k}[] = {{{body}}};")
    lines += ["", f"inline constexpr LaneCase kLaneCases[] = {{"]
    for k, (codec, nsym, raw, ref) in enumerate(lanes):
        lines.append(f"    {{{codec}, {nsym}, lane_raw_{k}, {len(raw)}, {'true' if ref is not None else 'false'}, "
                     f"lane_sym_{k}}},")
    lines += ["};", ""]
    for k, (codec, raw, size, status, expected) in enumerate(blocks):
        lines.append(_bytes(f"block_raw_{k}", raw))
        lines.append(_bytes(f"block_out_{k}", expected or b""))
    lines += ["", "inline constexpr BlockCase kBlockCases[] = {"]
    for k, (codec, raw, size, status, expected) in enumerate(blocks):
        lines.append(f"    {{{codec}, block_raw_{k}, {len(raw)}, {size}, {status}, block_out_{k}}},")
    lines += ["};", "", "}  // namespace bcpack_vectors", ""]
    return "\n".join(lines), lanes, blocks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare instead of writing")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    text, lanes, blocks = render()
    if args.check:
        try:
            with open(args.out, encoding="utf-8", newline="") as f:
                current = f.read()
        except OSError as e:
            print(f"ERROR: cannot read {args.out}: {e}")
            return 1
        if current != text:
            print(f"ERROR: {args.out} is out of date; run python -I runtime/test/bcpack_vectors.py")
            return 1
        print("bcpack_vectors.h is up to date")
        return 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    accepted = sum(ref is not None for *_, ref in lanes)
    by_status = {}
    for *_, status, _ in blocks:
        by_status[status] = by_status.get(status, 0) + 1
    print(f"wrote {args.out}: {len(lanes)} lane cases ({accepted} accepted), {len(blocks)} block cases "
          f"(by status {dict(sorted(by_status.items()))})")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
