"""Optional ctypes binding for the Rust rehydrator."""

import ctypes
import functools
import os
import sys


class _Geometry(ctypes.Structure):
    _fields_ = [
        ("vertices", ctypes.c_char_p),
        ("vertices_len", ctypes.c_size_t),
        ("indices", ctypes.c_char_p),
        ("indices_len", ctypes.c_size_t),
        ("stride", ctypes.c_size_t),
        ("index_width", ctypes.c_size_t),
        ("transform", ctypes.POINTER(ctypes.c_double)),
    ]


@functools.lru_cache(maxsize=1)
def _load():
    if ctypes.sizeof(ctypes.c_size_t) != 8:
        return None
    if sys.platform == "win32":
        name = "bvh_native.dll"
    elif sys.platform == "darwin":
        name = "libbvh_native.dylib"
    else:
        name = "libbvh_native.so"
    # The workspace build (tools/native) first, then a build of the crate on its own.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(os.path.dirname(here), "native", "target", "release", name),
                 os.path.join(here, "target", "release", name)):
        if os.path.isfile(path):
            break
    try:
        decode = ctypes.CDLL(path).bvh_rehydrate_v1
    except (AttributeError, OSError):
        return None
    decode.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(_Geometry), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte),
    ]
    decode.restype = ctypes.c_int
    return decode


def _descriptor(geometry, geometry_class):
    # Subclasses, overridden methods, and mutable/custom buffers use the Python implementation.
    if type(geometry) is not geometry_class or "triangle" in vars(geometry) or "vertex" in vars(geometry):
        return None
    vertices, indices, stride, index16, matrix = (geometry.vertices, geometry.indices, geometry.stride,
                                                  geometry.index16, geometry.transform)
    if type(vertices) is not bytes or (indices is not None and type(indices) is not bytes):
        return None
    if type(stride) is not int or not 12 <= stride <= sys.maxsize or type(index16) is not bool:
        return None
    transform = None
    if matrix is not None:
        if type(matrix) not in (list, tuple) or len(matrix) != 3:
            return None
        if any(type(row) not in (list, tuple) or len(row) != 4 for row in matrix):
            return None
        values = [value for row in matrix for value in row]
        if any(type(value) not in (float, int) for value in values):
            return None
        transform = (ctypes.c_double * 12)(*values)
    return _Geometry(vertices, len(vertices), indices, len(indices) if indices is not None else 0,
                     stride, (2 if index16 else 4) if indices is not None else 0, transform)


def try_rehydrate(src, kept, geometries, frame, geometry_class):
    """Return a complete bytearray on success. A refusal leaves the Python inputs untouched."""
    if "0" in (os.environ.get("LEONIDA_NATIVE"), os.environ.get("LEONIDA_BVH_NATIVE")):
        return None
    decode = _load()
    if decode is None:
        return None
    try:
        descriptors = [_descriptor(geometry, geometry_class) for geometry in geometries]
        if any(descriptor is None for descriptor in descriptors):
            return None
        native_geometries = (_Geometry * len(descriptors))(*descriptors)
        native_frame = None
        if frame.on:
            rotation = tuple(component for row in frame.r for component in row)
            native_frame = (ctypes.c_float * 15)(*rotation, *frame.t, *frame.s)
        dst = bytearray(len(src))
        address = (ctypes.c_ubyte * len(dst)).from_buffer(dst)
        if decode(src, len(src), kept, native_geometries, len(descriptors), native_frame, address) == 0:
            return dst
    except (AttributeError, ctypes.ArgumentError, MemoryError, OSError, OverflowError, TypeError, ValueError):
        # Retry through Python for its regular result or exception.
        pass
    return None
