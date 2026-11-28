"""Optional ctypes binding for the Rust GLB packing kernels (``tools/glb_native``).

``try_pack`` and ``try_bounds`` return ``None`` whenever the library is missing or cannot be
loaded, this interpreter's object layout or ``math.hypot`` differ from what the library expects,
the library declines the input, or ``LEONIDA_GLB_NATIVE=0`` is set. ``glb_writer`` then runs its
Python code, which gives the result or raises the error.

The library reads ``pack_primitive``'s lists in place through CPython API functions passed to it
here, called through ``ctypes.PyDLL`` so the GIL stays held. It only reads exact ``list``,
``tuple``, ``float`` and ``int`` objects; anything else (bools, subclasses, strings, other numbers)
is declined and left to Python.
"""

import ctypes
import functools
import math
import os
import random
import struct
import sys

_SEQUENCES = (list, tuple)
_FUNCTIONS = (
    ("list_size", "PyList_Size"),
    ("list_get", "PyList_GetItem"),
    ("tuple_size", "PyTuple_Size"),
    ("tuple_get", "PyTuple_GetItem"),
    ("float_as_double", "PyFloat_AsDouble"),
    ("long_as_double", "PyLong_AsDouble"),
    ("long_as_long_long_and_overflow", "PyLong_AsLongLongAndOverflow"),
    ("err_occurred", "PyErr_Occurred"),
    ("err_clear", "PyErr_Clear"),
)
_TYPES = (("float_type", float), ("long_type", int), ("list_type", list), ("tuple_type", tuple))


class _PyApi(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name, _ in _FUNCTIONS + _TYPES] + [
        ("direct", ctypes.c_size_t), ("tuple_items", ctypes.c_size_t),
    ]


def _library_name():
    if sys.platform == "win32":
        return "glb_native.dll"
    if sys.platform == "darwin":
        return "libglb_native.dylib"
    return "libglb_native.so"


def _hypot_agrees(hypot3):
    """The library reproduces this interpreter's 3-argument ``math.hypot`` on a fixed sample."""
    rng = random.Random(0x61B)
    for _ in range(4096):
        values = []
        for _ in range(3):
            kind = rng.random()
            if kind < 0.1:
                values.append(0.0)
            elif kind < 0.4:
                values.append(rng.uniform(-1.0, 1.0) * 2.0 ** rng.randint(-1074, 1023))
            else:
                values.append(rng.uniform(-1e4, 1e4))
        expected = math.hypot(*values)
        if math.isfinite(expected) and hypot3(*values) != expected:
            return False
    return True


@functools.lru_cache(maxsize=1)
def _load():
    """(library, API table), or None when the native path cannot be used here."""
    if ctypes.sizeof(ctypes.c_size_t) != 8 or sys.implementation.name != "cpython":
        return None
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        return None
    # The workspace build (tools/native) first, then a build of the crate on its own.
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(os.path.dirname(here), "native", "target", "release", _library_name()),
                 os.path.join(here, "target", "release", _library_name())):
        if os.path.isfile(path):
            break
    try:
        lib = ctypes.PyDLL(path)
        type_of, hypot3 = lib.glb_type_of_v1, lib.glb_hypot3_v1
        pack, bounds = lib.glb_pack_obj_v1, lib.glb_bounds_obj_v1
        probe, float_probe = lib.glb_layout_probe_v1, lib.glb_float_probe_v1
        api = _PyApi(**{field: ctypes.cast(getattr(ctypes.pythonapi, name), ctypes.c_void_p).value
                        for field, name in _FUNCTIONS},
                     **{field: id(kind) for field, kind in _TYPES})
    except (AttributeError, OSError):
        return None
    type_of.argtypes = [ctypes.py_object]
    type_of.restype = ctypes.c_size_t
    hypot3.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
    hypot3.restype = ctypes.c_double
    pack.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    pack.restype = ctypes.c_int
    bounds.argtypes = [ctypes.c_void_p] * 4
    bounds.restype = ctypes.c_int
    probe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    probe.restype = ctypes.c_ssize_t
    float_probe.argtypes = [ctypes.py_object]
    float_probe.restype = ctypes.c_uint64
    # Py_TYPE as the library reads it must be the interpreter's type for every kind it accepts.
    if any(type_of(sample) != id(type(sample)) for sample in (0.5, 7, [], (), True, 2 ** 70)):
        return None
    if not _hypot_agrees(hypot3):
        return None
    api.tuple_items = tuple.__basicsize__
    if _layout_agrees(lib, api):
        api.direct = 1
    return lib, api


def _layout_agrees(lib, api):
    """The library's in-place reads of lists, tuples and floats match ``len``, ``id`` and the
    values; otherwise it keeps reading through the C-API functions. Lengths are checked before
    item addresses, and no address is followed until it has matched."""
    grown = [0.25]
    for k in range(40):
        grown.append(k * -1.5)
    samples = ([0.5, -3.25, 1e300], (0.5, 2.0, -0.0, 7.5, 5e-324), grown, [], ())
    for sample in samples:
        words = (ctypes.c_uint64 * (2 + len(sample)))()
        if lib.glb_layout_probe_v1(ctypes.addressof(api), id(sample), ctypes.addressof(words), len(words)) < 0:
            return False
        if list(words)[:1 + len(sample)] != [len(sample)] + [id(item) for item in sample]:
            return False
    for value in (0.5, -0.0, 1e300, 5e-324, math.inf, -123.456):
        if lib.glb_float_probe_v1(value) != struct.unpack("<Q", struct.pack("<d", value))[0]:
            return False
    return True


def _enabled():
    return "0" not in (os.environ.get("LEONIDA_NATIVE"), os.environ.get("LEONIDA_GLB_NATIVE"))


def try_pack(prim):
    """``pack_primitive(prim)`` for a dict whose ``positions`` and ``indices`` are lists, or ``None``.

    The lists are read in place (the GIL is held for the whole call), so ``pack_primitive``'s
    copies of them are not needed.
    """
    if not _enabled() or type(prim) is not dict:
        return None
    positions = prim.get("positions")
    indices = prim.get("indices")
    if type(positions) is not list or type(indices) is not list:
        return None
    loaded = _load()
    if loaded is None:
        return None
    lib, api = loaded
    try:
        vertices = len(positions)
        index_count = len(indices)
        if not vertices or not index_count:
            return None
        normals_in = prim.get("normals")
        if normals_in is None or len(normals_in) == 0:
            normals_in = None
        elif type(normals_in) not in _SEQUENCES or len(normals_in) != vertices:
            return None
        uvs_in = prim.get("uvs")
        if uvs_in is None or len(uvs_in) == 0:
            uvs_in = None
        elif type(uvs_in) not in _SEQUENCES or len(uvs_in) != vertices:
            return None
        index_width = 2 if vertices <= 65535 else 4
        pos_len = 12 * vertices
        uv_len = 8 * vertices if uvs_in is not None else 0
        out = bytearray(2 * pos_len + uv_len + index_width * index_count)
        out_c = (ctypes.c_char * len(out)).from_buffer(out)
        box = (ctypes.c_double * 6)()
        rc = lib.glb_pack_obj_v1(
            ctypes.addressof(api), id(positions), id(indices),
            None if normals_in is None else id(normals_in), None if uvs_in is None else id(uvs_in),
            ctypes.addressof(out_c), len(out), ctypes.addressof(box),
        )
        del out_c
    except (AttributeError, ctypes.ArgumentError, MemoryError, OSError, OverflowError, TypeError, ValueError):
        return None
    if rc != 0:
        return None
    material = prim.get("material", "")
    view = memoryview(out)
    b = 2 * pos_len
    c = b + uv_len
    return {
        "positions": view[:pos_len].tobytes(),
        "normals": view[pos_len:b].tobytes(),
        "uvs": view[b:c].tobytes(),
        "indices": view[c:].tobytes(),
        "index_type": 5123 if index_width == 2 else 5125,
        "vertices": vertices,
        "index_count": index_count,
        "min": [box[0], box[1], box[2]],
        "max": [box[3], box[4], box[5]],
        "material": material if isinstance(material, (str, int)) and not isinstance(material, bool) else "",
        "material_index": prim.get("material_index"),
    }


NO_BOUNDS = object()


def try_bounds(primitives):
    """``compute_bounds``'s list (``NO_BOUNDS`` for its ``None``), or ``None`` to run Python."""
    if type(primitives) not in _SEQUENCES:
        return None  # a one-shot iterable would be half consumed when a later primitive is declined
    if not _enabled():
        return None
    loaded = _load()
    if loaded is None:
        return None
    lib, api = loaded
    box = (ctypes.c_double * 6)(math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf)
    found = ctypes.c_int(0)
    try:
        for prim in primitives:
            positions = prim.get("positions") or ()
            if type(positions) not in _SEQUENCES:
                return None
            if positions and lib.glb_bounds_obj_v1(ctypes.addressof(api), id(positions),
                                                   ctypes.addressof(box), ctypes.addressof(found)) != 0:
                return None
    except (AttributeError, ctypes.ArgumentError, MemoryError, OSError, OverflowError, TypeError, ValueError):
        return None
    if not found.value:
        return NO_BOUNDS
    return [0.0 if v == 0.0 else v for v in box]
