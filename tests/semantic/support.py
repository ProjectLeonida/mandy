"""Shared helpers for the translator tests (standard library only)."""
import gc
import importlib.machinery
import os
import struct
import sys
import tempfile
import warnings

import piecewise_parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# The mutation tests point this at a sabotaged copy of tools/; normally it is the repository's own.
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
FIXTURE_DIR = os.path.join(HERE, "fixtures", "disasm")


class PiecewiseLoader(importlib.machinery.SourceFileLoader):
    """Compiles a module from the tree piecewise_parse.py builds: the same code object as the usual loader's,
    without the 6 s CPython 3.12 to 3.14.7 spends on rdna2spirv.py's f-strings when it parses the file as one
    string. A valid .pyc is still used first, and a source the pieces cannot give goes to the usual compile."""

    def source_to_code(self, data, path, *args, **kwargs):
        try:
            tree = piecewise_parse.parse_pieces(data, path)
            if tree is not None:
                return super().source_to_code(tree, path, *args, **kwargs)
        except Exception:       # nothing to report from here: the usual compile says what is wrong, in its words
            pass
        return super().source_to_code(data, path, *args, **kwargs)


class ToolsFinder:
    """Hands the modules of TOOLS_DIR to PiecewiseLoader. It asks PathFinder, so sys.path decides which file a
    name means exactly as it does without this finder; a name found anywhere else, or one that is not a plain
    source file, is left to the finders after it."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if path is not None or not os.path.isfile(os.path.join(TOOLS_DIR, name + ".py")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path, target)
        loader = getattr(spec, "loader", None)
        if type(loader) is not importlib.machinery.SourceFileLoader or not _same_folder(loader.path, TOOLS_DIR):
            return None
        spec.loader = PiecewiseLoader(loader.name, loader.path)
        return spec


def _same_folder(path, folder):
    return os.path.normcase(os.path.dirname(os.path.abspath(path))) == os.path.normcase(os.path.abspath(folder))


sys.dont_write_bytecode = True      # never drop __pycache__ into the tree under test
sys.path.insert(0, TOOLS_DIR)
# Only where the tokenizer has that cost; just before PathFinder, which would otherwise load these modules.
if piecewise_parse.needed() and importlib.machinery.PathFinder in sys.meta_path:
    sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), ToolsFinder)
import rdna2spirv  # noqa: E402  (path set above)
from spirv_mini_interp import run_fragment  # noqa: E402

# translate() reads <dir>/<name>.s and, for pixel shaders, an input signature from <dir>/../<name>.dxbc.
# A DXBC container with zero parts (magic, 24 zero bytes, part count 0, 4 more bytes) declares no inputs.
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def translate_fixture(name, dxbc=STUB_DXBC, fssm=None):
    """SPIR-V assembly text for tests/semantic/fixtures/disasm/<name>.s, translated by the translator under test.

    `dxbc` overrides the empty input signature; pixel shaders that read attributes need a real ISG1 part.
    `fssm` supplies the compute-shader header blob (RSRC2 and group size) for cs_* fixtures.
    """
    src = os.path.join(FIXTURE_DIR, name + ".s")
    if not os.path.isfile(src):
        raise AssertionError(f"missing fixture {src}")        # a missing fixture is a failure, never a skip
    with open(src, "rb") as f:
        return translate_listing(name, f.read(), dxbc, fssm)


def translate_lines(name, lines, dxbc=STUB_DXBC, fssm=None):
    """Like translate_fixture for a hand-written program given as instruction lines (no fixture file)."""
    text = f"; {name}  hand-written test program\n" + "".join("\t" + ln.strip() + "\n" for ln in lines)
    return translate_listing(name, text.encode("utf-8"), dxbc, fssm)


def translate_listing(name, data, dxbc=STUB_DXBC, fssm=None):
    """SPIR-V assembly text for a listing in the translator's input format (bytes of a .s file)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "disasm", name + ".s"), "wb") as f:
            f.write(data)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(dxbc)
        if fssm is not None:
            os.mkdir(os.path.join(tmp, "fssm"))
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(fssm)
        with warnings.catch_warnings():
            # translate() leaves its .dxbc handle to the garbage collector; harmless, and not ours to change here.
            warnings.simplefilter("ignore", ResourceWarning)
            result = rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))
            gc.collect()        # release the handle before the temporary directory is removed (Windows)
        return result


def mrt0_words(spvasm, **run_options):
    """Run the translated pixel shader once; return the four 32-bit words it writes to mrt0."""
    out = run_fragment(spvasm, **run_options)
    if list(out) != ["%out_mrt0"]:
        raise AssertionError(f"expected exactly one output, %out_mrt0; got {sorted(out)}")
    return list(out["%out_mrt0"])


def varyings(spvasm, attrs, bary):
    """Input values for a translated PS, given {attribute n: three per-vertex vec4s of the producer's raw 32-bit
    words} and the barycentric weights `bary` of vertices 0..2, as Vulkan hands them over: a PerVertexKHR input gets
    the three vertices, a Flat one vertex 0, any other the words read as f32 and interpolated; %bary gets `bary`."""
    f = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]    # noqa: E731
    out = {"%bary": tuple(bary)}
    for n, verts in attrs.items():
        name, fv = f"%in_attr{n}", [tuple(f(w) for w in v) for v in verts]
        if f"OpDecorate {name} PerVertexKHR" in spvasm:
            out[name] = tuple(fv)
        elif f"OpDecorate {name} Flat" in spvasm:
            out[name] = fv[0]
        else:
            out[name] = tuple(sum(b * v[c] for b, v in zip(bary, fv)) for c in range(4))
    return out
