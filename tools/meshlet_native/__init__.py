"""Optional ctypes bindings for the meshlet page kernels."""

import ctypes
import functools
import os
import struct
import sys


_COMPENSATED = int(sys.version_info >= (3, 12))
_UINT32 = ctypes.c_uint32
_SIZE = ctypes.c_size_t
_SOURCE = (ctypes.c_char_p, _SIZE)


class _Block(ctypes.Structure):
    _fields_ = [
        ("header", _SIZE),
        ("vertices", _SIZE),
        ("position_at", _SIZE),
        ("normal_at", _SIZE),
        ("normal_mode", _UINT32),
    ]


_GROUP_KEYS = (
    ("faces_4240_", ("shared_edges", "agree", "opposed")),
    ("centres_4240_", ("meshlets", "default_words", "coarse_steps", "axes", "axes_exact", "axes_within_step",
                       "radius_covers", "radius_covers_within_rounding", "radius_tight")),
    ("cones_4240_", ("meshlets", "cones", "axis_within_25_degrees", "faces_inside", "faces_inside_within_0_05")),
    ("normals_4240_", ("vertices", "within_8_degrees", "within_25_degrees", "opposed")),
)
_RESULT_COUNT = sum(len(keys) for _, keys in _GROUP_KEYS)


@functools.lru_cache(maxsize=1)
def _load():
    if sys.implementation.name != "cpython" or ctypes.sizeof(_SIZE) != 8:
        return None
    if sys.platform == "win32":
        name = "meshlet_native.dll"
    elif sys.platform == "darwin":
        name = "libmeshlet_native.dylib"
    else:
        name = "libmeshlet_native.so"
    # The workspace build (tools/native) first, then a build of the crate on its own.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(os.path.dirname(here), "native", "target", "release", name),
                 os.path.join(here, "target", "release", name)):
        if os.path.isfile(path):
            break
    try:
        lib = ctypes.CDLL(path)
        lib.meshlet_validate_v1.argtypes = (*_SOURCE, _SIZE, ctypes.POINTER(_UINT32))
        lib.meshlet_geometry_v1.argtypes = (*_SOURCE, _SIZE, ctypes.POINTER(_Block), _UINT32, _UINT32,
                                             ctypes.POINTER(_UINT32))
        lib.meshlet_decode_v1.argtypes = (*_SOURCE, _SIZE, _SIZE, _SIZE, _UINT32, _UINT32,
                                         ctypes.POINTER(ctypes.c_double))
        lib.meshlet_block_v1.argtypes = (*_SOURCE, ctypes.POINTER(_Block), ctypes.POINTER(_SIZE), _SIZE, _UINT32,
                                        ctypes.POINTER(_SIZE))
        for func in (lib.meshlet_validate_v1, lib.meshlet_geometry_v1, lib.meshlet_decode_v1, lib.meshlet_block_v1):
            func.restype = ctypes.c_int
    except (AttributeError, OSError):
        return None
    return lib


def _library(data):
    # ctypes borrows these immutable bytes without copying, including embedded NULs.
    if type(data) is bytes and "0" not in (os.environ.get("LEONIDA_NATIVE"), os.environ.get("LEONIDA_MESHLET_NATIVE")):
        return _load()
    return None


def _call(func, data, out, *args):
    try:
        return func(data, len(data), *args, out) == 0
    except (ctypes.ArgumentError, MemoryError, OSError, OverflowError, TypeError, ValueError):
        return False


def validate_group(data, at):
    lib = _library(data)
    if lib is None or type(at) is not int or not 0 <= at <= len(data) - 12:
        return None
    out = (_UINT32 * 3)()
    if _call(lib.meshlet_validate_v1, data, out, at):
        return tuple(out)
    return None


def parse_legacy_group(data, at, head):
    if not 0x80000001 <= head <= 0x80000800:
        return None
    stats = validate_group(data, at)
    if stats is None:
        return None
    meshlets = head & 0x7fffffff
    triangles, refs = struct.unpack_from("<2I", data, at + 4)
    full, span, unique = stats
    end = at + 12 + 16 * meshlets + 4 * triangles + 2 * refs
    return {"kind": "group", "offset": at, "meshlets": meshlets, "triangles": triangles,
            "vertex_refs": refs, "vertex_span": span, "unique_refs": unique, "full_meshlets": full}, end


def _layout(block):
    if type(block) is not dict or type(block.get("slots")) is not list:
        return None
    at, n = block.get("offset"), block.get("vertices")
    if type(at) is not int or type(n) is not int or min(at, n) < 0:
        return None
    for row in block["slots"]:
        if type(row) not in (list, tuple) or len(row) != 3 or any(type(v) is not int for v in row):
            return None
        _, tag, off = row
        if not 0 <= tag <= 255 or off < 0:
            return None
    return at, n, block["slots"]


def _rows(lib, data, at, start, n, mode):
    stride = 8 if mode == 0 else 4
    if not 0 <= start <= len(data) or n > (len(data) - start) // stride:
        return None
    width = 4 if mode == 8 else 3
    try:
        out = (ctypes.c_double * (n * width))()
        if _call(lib.meshlet_decode_v1, data, out, at, start, n, mode, _COMPENSATED):
            rows = struct.iter_unpack(f"={width}d", out)
            return [(x, y, z, int(sign)) for x, y, z, sign in rows] if mode == 8 else list(rows)
    except (MemoryError, OverflowError):
        pass
    return None


def positions(data, block):
    lib = _library(data)
    layout = _layout(block) if lib is not None else None
    if layout is not None:
        at, n, slots = layout
        tag, off = {i: (tag, off) for i, tag, off in slots}.get(0, (None, 0))
        if tag == 0x42 and at <= len(data) - 0x7C:
            return _rows(lib, data, at, at + off, n, 0)
    return None


def _normal_mode(tag):
    return tag & 15 if (tag & 0xEF) in (0x21, 0x27) else 0


def normals(data, block):
    lib = _library(data)
    layout = _layout(block) if lib is not None else None
    if layout is not None:
        at, n, slots = layout
        tag, off = {i: (tag, off) for i, tag, off in slots}.get(2, (0, 0))
        mode = _normal_mode(tag)
        if mode:
            return _rows(lib, data, 0, at + off, n, mode)
    return None


def tangents(data, block):
    lib = _library(data)
    layout = _layout(block) if lib is not None else None
    if layout is None:
        return None
    at, n, slots = layout
    result = {}
    for slot, tag, off in slots:
        if slot != 2 and _normal_mode(tag) == 7:
            rows = _rows(lib, data, 0, at + off, n, 8)
            if rows is None:
                return None
            result[slot] = rows
    return result


def _descriptor(data, block, normals):
    layout = _layout(block)
    if layout is None:
        return None
    at, n, rows = layout
    slots = {i: (tag, off) for i, tag, off in rows}
    pos_tag, pos_off = slots.get(0, (0, 0))
    norm_tag, norm_off = slots.get(2, (0, 0))
    normal_mode = _normal_mode(norm_tag) if normals else 0
    if pos_tag != 0x42 or at + pos_off > len(data) or (normal_mode and at + norm_off > len(data)):
        return None
    return _Block(at, n, at + pos_off, at + norm_off if normal_mode else 0, normal_mode)


def block_checks(data, block, normals):
    lib = _library(data)
    if lib is None or type(normals) is not bool:
        return None
    descriptor = _descriptor(data, block, normals)
    if descriptor is None:
        return None
    starts = []
    if descriptor.normal_mode:
        # Parsed slot numbers are unique, and the Python decoder accepts any
        # four-byte type-7 slot other than the normal in slot 2.
        starts = [descriptor.header + off for slot, tag, off in block["slots"]
                  if slot != 2 and _normal_mode(tag) == 7]
        if any(start > len(data) for start in starts):
            return None
    offsets = (_SIZE * len(starts))(*starts)
    out = (_SIZE * 7)()
    if not _call(lib.meshlet_block_v1, data, out, ctypes.byref(descriptor), offsets, len(starts), _COMPENSATED):
        return None
    keys = ("positions_4240", "positions_4240_finite", "positions_4240_within_300m")
    if descriptor.normal_mode:
        keys += tuple("tangents_4240_" + key for key in ("vertices", "perpendicular", "sign_plus", "sign_minus"))
    return dict(zip(keys, out))


def geometry(data, group, block, faces, centres, normals, cones):
    lib = _library(data)
    if lib is None or not all(type(flag) is bool for flag in (faces, centres, normals, cones)):
        return None
    descriptor = _descriptor(data, block, normals)
    if descriptor is None:
        return None
    flags = faces | (centres << 1) | (normals << 2) | (cones << 3)
    out = (_UINT32 * _RESULT_COUNT)()
    if not _call(lib.meshlet_geometry_v1, data, out, group["offset"], ctypes.byref(descriptor), flags, _COMPENSATED):
        return None
    result = {}
    start = 0
    for enabled, (prefix, keys) in zip((faces, centres, cones, bool(descriptor.normal_mode)), _GROUP_KEYS):
        if enabled:
            result.update((prefix + key, value) for key, value in zip(keys, out[start:start + len(keys)]))
        start += len(keys)
    return result
