"""Standalone BC1-BC7 BCPack decoder (Python standard library only).

The rANS lanes feed fixed dictionaries and exact block reconstruction from
bcpack_standalone.py. No calibration files or provisional predictors are used.
The legacy quality/history keywords remain accepted for caller compatibility.
See docs/bcpack-standalone.md for provenance and independent verification.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bcpack_standalone import TABLES

# Probability tables contain only fixed rANS frequencies, indexed by width and shape.
_PROB_TABLES = {(w, q): TABLES["FREQ_TABLE"][w-1][q][:w+1]
                for w in range(1, 9) for q in range(16)}

# rANS and lane-array constants
RANS_L = 1 << 24
MAX_WIDTH = 10
PREFETCH_BUDGET = 8
MAX_VIRTUAL_ZEROS = 8

CODEC_NAMES = {
    1: "BC1",
    2: "BC2",
    3: "BC3",
    4: "BC4",
    5: "BC5",
    6: "BC6H",
    7: "BC7",
}

# Bytes represented by one lane symbol-tuple (BC3/BC5/BC6H/BC7 use 2 elements per 16-byte block).
ELEMENT_BYTES = 8

BLOCK_BYTES = {
    1: 8,
    2: 16,
    3: 16,
    4: 8,
    5: 16,
    6: 16,
    7: 16,
}

# Default per-lane symbol widths (bits) for each DSTORAGE_BCPACK_MODE.
LANE_WIDTHS = {
    2: (4,)*10 + (5,6,5,8,8),
    1: (8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8),
    3: (4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8),
    4: (8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    5: (4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6),
    6: (4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5),
    7: (4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5),
}

ALL_CODECS = frozenset(CODEC_NAMES)
SUPPORTED_REBUILD_CODECS = frozenset(range(1, 8))


class BcpackError(ValueError):
    """Raised when a BCPack stream fails structural or rANS validation."""


class BcpackProvisionalError(BcpackError):
    """Raised when strict decoding encounters an unconfirmed or unknown history entry."""


class BcpackUnsupportedCodec(NotImplementedError):
    """Raised when a BCPack codec has no block rebuilder yet."""


class _Model:
    """16-bit rANS banded frequency model for one (symbol_width, q) pair."""

    __slots__ = ("w", "freq", "base")

    def __init__(self, w, freq):
        self.w = w
        self.freq = freq
        base = []
        acc = 0
        for p, f in enumerate(freq):
            base.append(acc)
            acc += f * (1 if p == 0 else 1 << (p - 1))
        if acc != 65536:
            raise BcpackError(f"probability table ({w}) does not sum to 65536 (got {acc})")
        self.base = tuple(base)

    def lookup(self, cf):
        p = self.w
        base = self.base
        while p > 0 and cf < base[p]:
            p -= 1
        return 0 if p == 0 else (1 << (p - 1)) + (cf - base[p]) // self.freq[p]

    def advance(self, s, x):
        p = s.bit_length()
        f = self.freq[p]
        c = self.base[p] + f * (s - (0 if p == 0 else 1 << (p - 1)))
        if c == 0:
            f, c = f - 1, 1
        return f * (x >> 16) + (x & 0xFFFF) - c


_MODEL_CACHE = {}


def _get_model(w, q):
    key = (w, q)
    m = _MODEL_CACHE.get(key)
    if m is None:
        freq = _PROB_TABLES.get(key)
        if freq is None:
            return None
        m = _Model(w, freq)
        _MODEL_CACHE[key] = m
    return m


def _decode_lanes_raw(raw, widths, nsym):
    """Decode the interleaved rANS lane array into one symbol list per lane.

    Returns `(lanes, bytes_used)`. Raises `BcpackError` if any fragment fails
    to land at the exact rANS lower bound `RANS_L`, runs out of input, or
    produces an out-of-range symbol.
    """
    nl = len(widths)
    HDR0, HDR1, HDR2, PROC = 0, 1, 2, 3
    L = RANS_L
    ctrl = [HDR0] * nl
    out_rem = [nsym] * nl
    w = [0] * nl
    q = [0] * nl
    h3 = [0] * nl
    flush = [0] * nl
    done = [nsym <= 0] * nl
    full = [False] * nl
    held = [False] * nl
    vz = [0] * nl
    inbuf = [0] * nl
    sym = [0] * nl
    in_rem = [0] * nl
    p0 = [0] * nl
    p1 = [0] * nl
    model = [None] * nl
    slotfull = [False] * nl
    slotval = [0] * nl
    stall = [False] * nl
    out = [[] for _ in range(nl)]
    pos = 0
    nraw = len(raw)

    def fail(msg, lane):
        raise BcpackError(
            f"lane {lane} (width {w[lane]}, q {q[lane]}) at input byte {pos}: {msg}"
        )

    while True:
        if all(done[i] and not slotfull[i] for i in range(nl)):
            return out, pos

        # Phase 1: state update, prefetch arbitration (at most 8 bytes/cycle),
        # and symbol release for non-stalled lanes.
        left = PREFETCH_BUDGET
        for i in range(nl):
            if done[i]:
                stall[i] = True
                continue
            if held[i] and slotfull[i]:
                stall[i] = True
                continue
            stall[i] = False
            if ctrl[i] == PROC:
                p0[i] = model[i].advance(sym[i], p1[i]) if held[i] else p1[i]
            if not full[i] and (ctrl[i] != PROC or in_rem[i] > 0):
                if left == 0:
                    stall[i] = True
                    continue
                if pos >= nraw:
                    fail("input ran out", i)
                inbuf[i] = raw[pos]
                pos += 1
                full[i] = True
                left -= 1
            if held[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False

        # Phase 2: header parsing, byte merge, virtual zero fill on flush, and symbol lookup.
        for i in range(nl):
            if done[i] or stall[i]:
                continue
            c = ctrl[i]
            if c == HDR0:
                if not full[i]:
                    continue
                h = inbuf[i]
                q[i] = h & 15
                flush[i] = (h >> 4) & 1
                h3[i] = (h >> 5) & 1
                w[i] = widths[i] - (h >> 6)
                full[i] = False
                ctrl[i] = HDR1
                vz[i] = 0
            elif c == HDR1:
                if not full[i]:
                    continue
                full[i] = False
                wi = w[i]
                if not 1 <= wi <= MAX_WIDTH:
                    fail("symbol width out of range", i)
                m = _get_model(wi, q[i])
                if m is None:
                    fail("probability table for this width and q is not known", i)
                model[i] = m
                if h3[i]:
                    in_rem[i] = inbuf[i]
                    ctrl[i] = HDR2
                else:
                    in_rem[i] = inbuf[i] + 1
                    ctrl[i] = PROC
            elif c == HDR2:
                if not full[i]:
                    continue
                in_rem[i] = ((inbuf[i] << 8) | (in_rem[i] & 0xFF)) + 257
                ctrl[i] = PROC
                full[i] = False
            else:
                st = p0[i]
                if st < L:
                    if in_rem[i] > 0:
                        if full[i]:
                            st = (st << 8) | inbuf[i]
                            in_rem[i] -= 1
                            full[i] = False
                            if in_rem[i] == 0 and flush[i]:
                                while st < L and st != 0 and vz[i] < MAX_VIRTUAL_ZEROS:
                                    st <<= 8
                                    vz[i] += 1
                    elif flush[i]:
                        if st == 0 or vz[i] >= MAX_VIRTUAL_ZEROS:
                            fail("fragment did not end at its home state", i)
                        while st < L and st != 0 and vz[i] < MAX_VIRTUAL_ZEROS:
                            st <<= 8
                            vz[i] += 1
                    else:
                        p1[i] = st
                        if out_rem[i] > 0:
                            ctrl[i] = HDR0
                        else:
                            done[i] = True
                        continue
                nxt = st > L and out_rem[i] > 0
                if nxt:
                    s = model[i].lookup(st & 0xFFFF)
                    sym[i] = s
                    out_rem[i] -= 1
                    if s >> w[i]:
                        fail("symbol outside its width", i)
                held[i] = nxt
                p1[i] = st
                if in_rem[i] == 0:
                    if flush[i]:
                        if st == L:
                            if out_rem[i] > 0:
                                ctrl[i] = HDR0
                                p0[i] = 0
                                p1[i] = 0
                            else:
                                done[i] = True
                        elif st > L and not nxt:
                            fail("fragment did not end at its home state", i)
                    elif out_rem[i] == 0 and not nxt:
                        done[i] = True
                    elif st == L:
                        fail("fragment did not end at its home state", i)
                elif st == L or (out_rem[i] == 0 and not nxt and st > L):
                    fail("fragment did not end at its home state", i)

        # Phase 3: emit one synchronized element when every active lane has filled its slot.
        ready = True
        anyfull = False
        for i in range(nl):
            if slotfull[i]:
                anyfull = True
            elif not (done[i] and not held[i]):
                ready = False
                break
        if ready and anyfull:
            for i in range(nl):
                if slotfull[i]:
                    out[i].append(slotval[i])
                    slotfull[i] = False
        for i in range(nl):
            if done[i] and held[i] and not slotfull[i]:
                slotfull[i] = True
                slotval[i] = sym[i]
                held[i] = False


_NATIVE = []


def _native_unpack():
    """`(unpack_model, tables)` of the optional Rust lane decoder, or None.

    `tools/bcpack_calibration` (built with Cargo, see its README.md) exports
    `unpack_model`, a port of `_decode_lanes_raw` that is fed this module's
    probability tables, so it accepts the same streams and gives the same
    symbols. Set `LEONIDA_BCPACK_NATIVE=0` to use Python only.
    """
    if not _NATIVE:
        found = None
        if os.environ.get("LEONIDA_BCPACK_NATIVE", "1") != "0":
            import ctypes
            name = {"win32": "bcpack_calibration.dll",
                    "darwin": "libbcpack_calibration.dylib"}.get(sys.platform, "libbcpack_calibration.so")
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "bcpack_calibration", "target", "release", name)
            if os.path.isfile(path):
                try:
                    lib = ctypes.CDLL(path)
                    fn = lib.unpack_model
                except (OSError, AttributeError):
                    fn = None
                if fn is not None:
                    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
                                   ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
                    fn.restype = ctypes.c_ssize_t
                    side = MAX_WIDTH + 1
                    flat = [-1] * (side * 16 * side)
                    for (w, q), freq in _PROB_TABLES.items():
                        if 0 <= w <= MAX_WIDTH and 0 <= q < 16 and len(freq) == w + 1:
                            flat[(w * 16 + q) * side:(w * 16 + q) * side + w + 1] = freq
                    found = fn, (ctypes.c_int32 * len(flat))(*flat)
        _NATIVE.append(found)
    return _NATIVE[0]


def _decode_lanes_native(raw, widths, nsym):
    """`(lanes, bytes_used)` from the Rust port of `_decode_lanes_raw`, or None.

    None means the library is not built or the stream is refused (or is one
    the port leaves to Python); `decode_lanes` then runs the Python model,
    which raises its usual error. `tests/test_bcpack_native.py` checks that the
    two agree.
    """
    native = _native_unpack()
    if native is None or nsym <= 0:
        return None
    import ctypes
    fn, tables = native
    nl = len(widths)
    data = bytes(raw)
    out = (ctypes.c_uint16 * (nsym * nl))()
    used = fn(data, len(data), bytes(widths), nl, nsym, tables, len(tables), out)
    if used < 0:
        return None
    flat = memoryview(out).cast("B").cast("H")
    return [flat[i * nsym:(i + 1) * nsym].tolist() for i in range(nl)], used


def decode_lanes(raw, codec, out_size):
    """Entropy layer: decode `raw` into per-lane symbol lists and bytes consumed.

    Works for all seven codecs (`1` through `7`). Trailing non-zero
    bytes after the last lane finishes are rejected. Uses the Rust decoder when
    it is built (`_native_unpack`) and Python for the streams it refuses.
    """
    if codec not in LANE_WIDTHS:
        raise BcpackError(f"unsupported BCPack codec id: {codec}")
    if out_size < 0 or out_size % BLOCK_BYTES[codec] != 0:
        raise BcpackError(
            f"decoded size {out_size} is not a whole number of {CODEC_NAMES[codec]} blocks"
        )
    nsym = out_size // ELEMENT_BYTES
    fast = _decode_lanes_native(raw, LANE_WIDTHS[codec], nsym)
    if fast is not None:
        if any(memoryview(raw)[fast[1]:]):
            raise BcpackError("non-zero data left over after the last lane finished")
        return fast
    data = memoryview(raw)
    lanes, used = _decode_lanes_raw(data, LANE_WIDTHS[codec], nsym)
    if any(data[used:]):
        raise BcpackError("non-zero data left over after the last lane finished")
    return lanes, used


# BC6H ------------------------------------------------------------------------

# Mode index (1..14, where mode = lane15[2*i] + 1) -> D3D11/Khronos mode bits.
_BC6H_CODE = (
    0, 0x00, 0x01, 0x02, 0x06, 0x0A, 0x0E, 0x12,
    0x16, 0x1A, 0x1E, 0x03, 0x07, 0x0B, 0x0F,
)

# Mode code -> (transformed, base_bits, (dr, dg, db), regions).
_BC6H_INFO = {
    0x00: (1, 10, (5, 5, 5), 2),
    0x01: (1, 7, (6, 6, 6), 2),
    0x02: (1, 11, (5, 4, 4), 2),
    0x06: (1, 11, (4, 5, 4), 2),
    0x0A: (1, 11, (4, 4, 5), 2),
    0x0E: (1, 9, (5, 5, 5), 2),
    0x12: (1, 8, (6, 5, 5), 2),
    0x16: (1, 8, (5, 6, 5), 2),
    0x1A: (1, 8, (5, 5, 6), 2),
    0x1E: (0, 6, (6, 6, 6), 2),
    0x03: (0, 10, (10, 10, 10), 1),
    0x07: (1, 11, (9, 9, 9), 1),
    0x0B: (1, 12, (8, 8, 8), 1),
    0x0F: (1, 16, (4, 4, 4), 1),
}

# Standard D3D11/Khronos BC6H header bit layouts (LSB-first after the 2- or 5-bit mode).
_BC6H_LAYOUT = {
    0x00: "gy4 by4 bz4 rw10 gw10 bw10 rx5 gz4 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x01: "gy5 gz4 gz5 rw7 bz0 bz1 by4 gw7 by5 bz2 gy4 bw7 bz3 bz5 bz4 rx6 gy4_ gx6 gz4_ bx6 by4_ ry6 rz6 d",
    0x02: "rw10 gw10 bw10 rx5 rw:10 gy4_ gx4 gw:10 bz0 gz4_ bx4 bw:10 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x06: "rw10 gw10 bw10 rx4 rw:10 gz4 gy4_ gx5 gw:10 gz4_ bx4 bw:10 bz1 by4_ ry4 bz0 bz2 rz4 gy4 bz3 d",
    0x0A: "rw10 gw10 bw10 rx4 rw:10 by4 gy4_ gx4 gw:10 bz0 gz4_ bx5 bw:10 by4_ ry4 bz1 bz2 rz4 bz4 bz3 d",
    0x0E: "rw9 by4 gw9 gy4 bw9 bz4 rx5 gz4 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x12: "rw8 gz4 by4 gw8 bz2 gy4 bw8 bz3 bz4 rx6 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry6 rz6 d",
    0x16: "rw8 bz0 by4 gw8 gy5 gy4 bw8 gz5 bz4 rx5 gz4 gy4_ gx6 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d",
    0x1A: "rw8 bz1 by4 gw8 by5 gy4 bw8 bz5 bz4 rx5 gz4 gy4_ gx5 bz0 gz4_ bx6 by4_ ry5 bz2 rz5 bz3 d",
    0x1E: "rw6 gz4 bz0 bz1 by4 gw6 gy5 by5 bz2 gy4 bw6 gz5 bz3 bz5 bz4 rx6 gy4_ gx6 gz4_ bx6 by4_ ry6 rz6 d",
    0x03: "rw10 gw10 bw10 rx10 gx10 bx10",
    0x07: "rw10 gw10 bw10 rx9 rw:10 gx9 gw:10 bx9 bw:10",
    0x0B: "rw10 gw10 bw10 rx8 rw:11-10 gx8 gw:11-10 bx8 bw:11-10",
    0x0F: "rw10 gw10 bw10 rx4 rw:15-10 gx4 gw:15-10 bx4 bw:15-10",
}

_BC6H_FIELD_MAP = {
    "rw": (0, 0), "gw": (1, 0), "bw": (2, 0),
    "rx": (0, 1), "gx": (1, 1), "bx": (2, 1),
    "ry": (0, 2), "gy": (1, 2), "by": (2, 2),
    "rz": (0, 3), "gz": (1, 3), "bz": (2, 3),
}

_BC6H_NONE_E = ((0x8000, 0x200, 0x20, 0x20),) * 3
_BC6H_NONE_I = (8,) * 16
_BC6H_PLANS = {}


def _bc6h_plan(code):
    """Cached `(plan, header_bits)` for a BC6H mode code.

    Each plan item is `(channel, endpoint_slot, bit_index, stream_pos)` with
    `channel == -1` for the 5-bit partition field `d`.
    """
    cached = _BC6H_PLANS.get(code)
    if cached is not None:
        return cached
    pos = 2 if code < 2 else 5
    plan = []
    for token in _BC6H_LAYOUT[code].split():
        if token == "d":
            for b in range(5):
                plan.append((-1, -1, b, pos + b))
            pos += 5
        elif ":" in token:
            name, span = token.split(":")
            c, q = _BC6H_FIELD_MAP[name]
            hi, lo = (int(x) for x in span.split("-")) if "-" in span else (int(span), int(span))
            for b in range(hi, lo - 1, -1):
                plan.append((c, q, b, pos))
                pos += 1
        elif token.endswith("_"):
            name, n = token[:2], int(token[2:-1])
            c, q = _BC6H_FIELD_MAP[name]
            for b in range(n):
                plan.append((c, q, b, pos + b))
            pos += n
        else:
            name, n = token[:2], int(token[2:])
            c, q = _BC6H_FIELD_MAP[name]
            if name[1] in "wx" or name in ("ry", "rz"):
                for b in range(n):
                    plan.append((c, q, b, pos + b))
                pos += n
            else:
                plan.append((c, q, n, pos))
                pos += 1
    res = (tuple(plan), pos)
    _BC6H_PLANS[code] = res
    return res


def _bc6h_stream_w(mode, w):
    """Map BCPack's base endpoint `w` into the D3D11/Khronos BC6H bit order."""
    if mode == 13:
        return (w & 0x3FF) | (((w >> 10) & 1) << 11) | (((w >> 11) & 1) << 10)
    if mode == 14:
        out = w & 0x3FF
        for b in range(10, 16):
            out |= ((w >> b) & 1) << (25 - b)
        return out
    return w


def decode_with_info(raw, codec, out_size, *, allow_provisional=False, history=None):
    """Exact standalone decode; legacy profile/quality options are ignored.

    Every predictor is a fixed format constant. This entry point never uses
    fitted history, per-asset overrides or the old native block rebuilder.
    """
    from bcpack_standalone import rebuild
    if codec not in SUPPORTED_REBUILD_CODECS:
        raise BcpackUnsupportedCodec(f"unsupported BCPack codec id: {codec}")
    lanes, _ = decode_lanes(raw, codec, out_size)
    try:
        data = rebuild(lanes, codec, out_size // BLOCK_BYTES[codec])
    except (ValueError, IndexError) as exc:
        raise BcpackError(str(exc)) from exc
    return data, set()


def decode(raw, codec, out_size, *, allow_provisional=False, history=None):
    """Decode one BCPack chunk into `out_size` bytes of standard BCn blocks."""
    data, _ = decode_with_info(
        raw, codec, out_size, allow_provisional=allow_provisional, history=history
    )
    return data
