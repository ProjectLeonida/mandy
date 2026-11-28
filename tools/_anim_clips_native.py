"""ctypes adapter for the optional animation sequence decoder."""
import collections
import os
import struct
import sys

_NATIVE = []
_HEADER = struct.Struct("<HH?11H2I")
_TRACK = struct.Struct("<HBBd")
_ROWS = (None, *(struct.Struct("<" + "d" * width) for width in range(1, 5)))
_QUATERNION_STATS = {
    3: ("quaternion_rebuilt", "quaternion_rebuilt_worst_1e6"),
    4: ("quaternion_type8_four_stored", "quaternion_type8_four_stored_worst_1e6"),
    5: ("quaternion_four_stored", "quaternion_four_stored_worst_1e6"),
}

# Detect the CPython sum behavior so a release build works with both 3.11 and 3.12+.
_COMPENSATED_SUM = sum((1.0, 2.0 ** -53, 2.0 ** -53)) != 1.0


def native():
    """Return the versioned sequence decoder and its free function, if built."""
    disabled = "0" in (os.environ.get("LEONIDA_NATIVE", "1"), os.environ.get("LEONIDA_ANIM_NATIVE", "1"))
    if disabled or sys.implementation.name != "cpython":
        return None
    if not _NATIVE:
        name = {"win32": "anim_clips_native.dll", "darwin": "libanim_clips_native.dylib"}.get(
            sys.platform, "libanim_clips_native.so")
        # The workspace build (tools/native) first, then a build of the crate on its own.
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (os.path.join(here, "native", "target", "release", name),
                     os.path.join(here, "anim_clips_native", "target", "release", name)):
            if os.path.isfile(path):
                break
        found = None
        if os.path.isfile(path):
            try:
                import ctypes

                lib = ctypes.CDLL(path)
                decode = lib.anim_decode_sequence_v1
                free = lib.anim_free_sequence_v1
            except (ImportError, OSError, AttributeError):
                pass
            else:
                decode.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                   ctypes.c_uint8, ctypes.POINTER(ctypes.c_size_t))
                decode.restype = ctypes.c_void_p
                free.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
                free.restype = None
                found = (ctypes, decode, free)
        _NATIVE.append(found)
    return _NATIVE[0]


def _unpack(data, stats, channel_names):
    nf, ntracks, quantize_bits_match, *counts, boundaries, closed = _HEADER.unpack_from(data)
    out, metrics = {}, []
    pos = _HEADER.size
    for _ in range(ntracks):
        track, kind, width, worst = _TRACK.unpack_from(data, pos)
        pos += _TRACK.size
        row = _ROWS[width]
        if kind in (1, 2):  # A whole static track repeats the same tuple.
            out[track] = [row.unpack_from(data, pos)] * nf
            pos += row.size
        else:
            end = pos + row.size * nf
            out[track] = list(row.iter_unpack(data[pos:end]))
            pos = end
        if stats is not None and kind not in (0, 2):
            metrics.append((kind, worst))

    if stats is not None:
        if boundaries:
            stats["delta2_boundaries"] += boundaries
            stats["delta2_boundaries_closed"] += closed
        stats["quantize_bits_match"] += quantize_bits_match
        for t, count in enumerate(counts):
            if count:
                stats[f"channel:{t:02d}_{channel_names[t]}"] += count
        for kind, worst in metrics:
            if kind == 1:
                stats["quaternion_static"] += 1
                stats["quaternion_static_xyz_over_unit"] += int(worst)
            else:
                key, worst_key = _QUATERNION_STATS[kind]
                stats[key] += 1
                stats[worst_key] = max(stats[worst_key], round(worst * 1e6))
    return nf, out


def decode_sequence(data, at, anim, stats, channel_names):
    """Return assembled per-track rows, or defer to Python without updating stats."""
    decoder = native()
    if decoder is None or type(data) is not bytes or type(at) is not int or not 0 <= at < len(data):
        return None
    tracks = anim.get("tracks") if type(anim) is dict else None
    if type(tracks) not in (list, tuple):
        return None
    # Other mappings can observe each update or raise from one. Keep the reference ordering for them.
    if stats is not None and (type(stats) is not collections.Counter
                              or any(type(value) is not int for value in stats.values())):
        return None
    ctypes, decode, free = decoder
    length = ctypes.c_size_t()
    try:
        ptr = decode(data, len(data), at, len(tracks), _COMPENSATED_SUM, ctypes.byref(length))
    except (OSError, ctypes.ArgumentError):
        return None
    if not ptr:
        return None
    try:
        packet = (ctypes.c_char * length.value).from_address(ptr)
        return _unpack(memoryview(packet), stats, channel_names)
    finally:
        free(ptr, length.value)
