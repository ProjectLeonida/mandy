"""Measure how a GPU interpolates a packed 16-bit varying and which arithmetic model explains it (SHADER-468).

A 16-bit varying shares its 32-bit param channel with another one, so a pixel shader reads it with
`v_interp_p1ll_f16` / `v_interp_p2_f16` (docs/design.md, "16-bit interpolation"). What the SPI writes into LDS for
such a channel, and how p2 narrows the result, had never been measured on hardware. This tool draws a hand-written
16-bit varying (`f16vec2` in and out, no game files) through `runtime/build/vk_harness`, reads the interpolated
values back at every pixel centre together with the hardware's own barycentrics, and compares each one with four
models:

  rdna2           P10/P20 are per-half f16 differences (round to nearest even), the weighted sum is f32, and p2
                  narrows the result toward zero
  translator_rne  the f32 difference of the halves, weighted in f32, narrowed to nearest even: the translation
                  before SHADER-468
  translator_rtz  the same with the p2 narrow toward zero
  f16_acc         f16 deltas, accumulated with f16 fma instead of an f32 sum

Measured on the RDNA2 iGPU of a Ryzen 5 7600 (gfx1036, AMD Windows driver 24.10.24, Vulkan 1.3.280): `rdna2` explains
98,274 of 98,304 samples over the 12 draws below (six value sets, an orthographic and a perspective triangle) and the
other three fall short: `translator_rtz` 89,993, `f16_acc` 86,919, `translator_rne` 62,226. The 30 misses are two
corners: 26 in the f16 subnormal range (the exact value is a few 2**-24, where the hardware gives the smallest
subnormal and a plain toward-zero narrow gives zero) and 4 at an exact f16 grid boundary in the perspective draw,
where the vertex-0 weight is the SPI's own f32 value and not 1 - i - j. `VKH_SHADER_INFO=1` on `vk_harness` prints
the driver's ISA: it runs `v_interp_p1ll_f16` / `v_interp_p2_f16` inside `s_round_mode 0x000c` (MODE.FP16_ROUND
toward zero), which is why the narrow is toward zero (LLVM's SIModeRegister: "f16 interpolation instructions need
double precision round to zero").

    python -I tools/f16_interp_probe.py                     # every draw on the default device
    python -I tools/f16_interp_probe.py --gpu amd --isa      # AMD iGPU, and print the fragment ISA
    python -I tools/f16_interp_probe.py --draw 3             # one draw, verbose

Needs `glslangValidator` (Vulkan SDK) and `vk_harness` from `runtime/build`; exits 77 when either is missing or the
device has no 16-bit shader input/output, like the rest of the GPU tests.
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
UNSUPPORTED = 77

VERT = """#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(location = 0) smooth out f16vec2 vary;
void main() {
    vec4 p[3] = vec4[3](@P0@, @P1@, @P2@);
    uint w[3] = uint[3](@W0@, @W1@, @W2@);
    gl_Position = p[gl_VertexIndex];
    vary = unpackFloat2x16(w[gl_VertexIndex]);
}
"""

FRAG = """#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) smooth in f16vec2 vary;
layout(location = 0) out vec4 out_color;
void main() {
    out_color = vec4(vec2(vary), gl_BaryCoordEXT.y, gl_BaryCoordEXT.z);
}
"""

# The screen-space triangle is the whole viewport both times. Vertex 2 and 1 are outside it, so every pixel centre is
# covered; the second one gives the vertices w = 1, 2, 4, which makes the weights non-exact in f16.
ORTHO = ("vec4(-1.0, -1.0, 0.0, 1.0)", "vec4(3.0, -1.0, 0.0, 1.0)", "vec4(-1.0, 3.0, 0.0, 1.0)")
PERSPECTIVE = ("vec4(-1.0, -1.0, 0.0, 1.0)", "vec4(6.0, -2.0, 0.0, 2.0)", "vec4(-4.0, 12.0, 0.0, 4.0)")

# (name, low halves of vertices 0..2, high halves of vertices 0..2): hand-written f16 values.
DRAWS = [
    ("weights", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ("delta rounds", (1.0, 65504.0, 1.0), (0.0, 1.0, 0.0)),
    ("delta overflows f16", (-65504.0, 1000.0, -65504.0), (0.0, 1.0, 0.0)),
    ("small values", (0.5, 1.5, 2.5), (-4.0, 8.0, 16.0)),
    ("exact grid", (300.0, 300.5, 300.25), (100.0, 101.0, 99.0)),
    ("large values", (1000.0, 1001.0, 1000.0), (3.0, -7.0, 11.0)),
]

GPU_ENV = {
    "amd": {"VK_LOADER_DRIVERS_SELECT": "*amd-vulkan*", "LEONIDA_VK_DEVICE": "radeon"},
    "nv": {"VK_LOADER_DRIVERS_SELECT": "*nv-vk*", "LEONIDA_VK_DEVICE": "4070"},
}

INF = float("inf")


# ---- exact binary16 -------------------------------------------------------------------------------------------
def _round_half_even(n, d):
    q, r = divmod(n, d)
    return q + (2 * r > d or (2 * r == d and q & 1))


def _special_bits(x):
    return 0x7E00 if x != x else 0x7C00 | (0x8000 if x < 0 else 0)


def _bits(q, rtz=False):
    """f16 bits of an exact Fraction, round to nearest even or toward zero"""
    if q == 0:
        return 0
    sign = 0x8000 if q < 0 else 0
    a = abs(q)
    num, den = a.numerator, a.denominator
    e = num.bit_length() - den.bit_length()
    while Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    if e < -14:                                    # subnormal: a / 2**-24
        n, d = num << 24, den
        return sign | min((n // d) if rtz else _round_half_even(n, d), 1024)
    if e > 15:
        # finite overflow: rtz saturates at the largest f16 (0x7BFF), as Gen.f16_bits(rtz=True) does
        return sign | (0x7BFF if rtz else 0x7C00)
    n, d = (num << (10 - e), den) if e <= 10 else (num, den << (e - 10))
    m = (n // d) if rtz else _round_half_even(n, d)
    if m >= 2048:
        m >>= 1
        e += 1
        if e > 15:
            return sign | (0x7BFF if rtz else 0x7C00)
    return sign | ((e + 15) << 10) | (m - 1024)


def h_bits(x, rtz=False):
    """f16 bits of a Python float (inf and NaN keep their own bits)"""
    if x != x or abs(x) == INF:
        return _special_bits(x)
    return _bits(Fraction(x), rtz)


def h_value(b):
    """the exact float an f16 holds"""
    s = -1.0 if b & 0x8000 else 1.0
    e, m = (b >> 10) & 0x1F, b & 0x3FF
    if e == 0:
        return s * m / 2 ** 24
    if e == 31:
        return s * INF if m == 0 else float("nan")
    return s * (1024 + m) / 2 ** 10 * 2 ** (e - 15)


def f16_sub(a, b):
    """f16(a - b) with round to nearest even, the SPI's per-half P10/P20 word"""
    if a != a or b != b or abs(a) == INF or abs(b) == INF:
        return h_value(_special_bits(a - b))
    return h_value(_bits(Fraction(a) - Fraction(b)))


def narrow(q, rtz):
    """f16 bits of an exact value, toward zero or to nearest even"""
    if not (q == q) or abs(q) == INF:
        return _special_bits(q)
    return _bits(q, rtz)


# ---- the models -----------------------------------------------------------------------------------------------
def sample(models, i, j, h):
    """every model's f16 bits for one half of the channel at weights i, j"""
    out = {}
    for name, (delta, acc, mode) in models.items():
        rtz = mode == "rtz"
        d1, d2 = (f16_sub(h[1], h[0]), f16_sub(h[2], h[0])) if delta else (h[1] - h[0], h[2] - h[0])
        values = (i, j, h[0], d1, d2)
        if all(v == v and abs(v) != INF for v in values):
            if acc == "f32":
                out[name] = narrow(Fraction(h[0]) + Fraction(i) * Fraction(d1) + Fraction(j) * Fraction(d2), rtz)
                continue
            values = (h[0], h_value(_bits(Fraction(i) * Fraction(d1) + Fraction(h[0]), rtz)), d2, j)
            out[name] = narrow(Fraction(j) * Fraction(values[2]) + Fraction(values[1]), rtz)
            continue
        # an inf or NaN half: the same arithmetic in floats, whose inf/NaN rules are the f16 ones
        if acc == "f32":
            out[name] = h_bits(h[0] + i * d1 + j * d2, rtz)
        else:
            out[name] = h_bits(j * d2 + h_value(h_bits(i * d1 + h[0], rtz)), rtz)
    return out


MODELS = {
    "rdna2": (True, "f32", "rtz"),
    "translator_rne": (False, "f32", "rne"),
    "translator_rtz": (False, "f32", "rtz"),
    "f16_acc": (True, "f16", "rtz"),
}


def packed_word(lo, hi):
    return h_bits(lo) | (h_bits(hi) << 16)


# ---- running --------------------------------------------------------------------------------------------------
def find_glslang():
    env = os.environ.get("VULKAN_SDK")
    if env:
        for sub in ("Bin", "bin"):
            p = os.path.join(env, sub, "glslangValidator.exe" if os.name == "nt" else "glslangValidator")
            if os.path.isfile(p):
                return p
    return shutil.which("glslangValidator")


def build_pair(work, words, geometry):
    src = os.path.join(work, "vs.vert")
    frag = os.path.join(work, "ps.frag")
    text = VERT
    for key, value in (("@P0@", geometry[0]), ("@P1@", geometry[1]), ("@P2@", geometry[2]),
                       ("@W0@", "0x%08xu" % words[0]), ("@W1@", "0x%08xu" % words[1]),
                       ("@W2@", "0x%08xu" % words[2])):
        text = text.replace(key, value)
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with open(frag, "w", encoding="utf-8", newline="\n") as f:
        f.write(FRAG)
    glslang = find_glslang()
    out = []
    for path in (src, frag):
        spv = os.path.splitext(path)[0] + ".spv"
        r = subprocess.run([glslang, "-V", "--target-env", "vulkan1.3", "-o", spv, path],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"glslangValidator failed on {path}:\n{r.stdout}{r.stderr}")
        out.append(spv)
    return out


def run_draw(harness, vs, fs, gpu, isa, size=64):
    env = dict(os.environ)
    env.update(GPU_ENV.get(gpu, {}))
    env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"
    env["VKH_PC_BYTES"] = "4"   # the fixtures use no push constants (range size must be a multiple of 4)
    if isa:
        env["VKH_SHADER_INFO"] = "1"
    with tempfile.TemporaryDirectory(prefix="leonida-f16-interp-") as tmp:
        raw = os.path.join(tmp, "pixels.raw")
        r = subprocess.run([harness, vs, fs, raw, str(size), str(size)], env=env, capture_output=True, text=True)
        if r.returncode == UNSUPPORTED:
            raise SystemExit(f"SKIP: {r.stderr.strip()}")
        if r.returncode or not os.path.isfile(raw):
            raise SystemExit(f"vk_harness failed ({r.returncode}):\n{r.stdout}{r.stderr}")
        with open(raw, "rb") as f:
            data = f.read()
    if isa:
        print("\n".join(line for line in r.stdout.splitlines() if line.strip()))
    if len(data) != size * size * 16:
        raise SystemExit(f"readback is {len(data)} bytes, expected {size * size * 16}")
    return list(struct.iter_unpack("<4f", data)), r.stderr


def compare(pixels, lows, highs, counts, fails):
    for lo, hi, i, j in ((p[0], p[1], p[2], p[3]) for p in pixels):
        if i != i or j != j:
            continue
        for measured, h in ((lo, lows), (hi, highs)):
            for name, bits in sample(MODELS, i, j, h).items():
                counts[name][1] += 1
                if bits == h_bits(measured):
                    counts[name][0] += 1
                elif len(fails[name]) < 3:
                    fails[name].append((round(i, 7), round(j, 7), h, measured, h_value(bits)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tools", default=os.path.join(ROOT, "runtime", "build"), help="directory with vk_harness")
    ap.add_argument("--gpu", default="", help="amd or nv: the loader filter and LEONIDA_VK_DEVICE to use")
    ap.add_argument("--draw", type=int, default=None, help="run only this draw (0-based)")
    ap.add_argument("--isa", action="store_true", help="print the driver's fragment disassembly")
    args = ap.parse_args()
    harness = os.path.join(args.tools, "vk_harness" + (".exe" if os.name == "nt" else ""))
    if not os.path.isfile(harness):
        print(f"SKIP: no vk_harness in {args.tools} (build the runtime first)", file=sys.stderr)
        return UNSUPPORTED
    if not find_glslang():
        print("SKIP: no glslangValidator (set VULKAN_SDK or put it on PATH)", file=sys.stderr)
        return UNSUPPORTED
    draws = [(n, lo, hi, g) for n, lo, hi in DRAWS for g in ("ortho", "perspective")]
    if args.draw is not None:
        draws = [draws[args.draw]]
    counts = {name: [0, 0] for name in MODELS}
    fails = {name: [] for name in MODELS}
    device = ""
    with tempfile.TemporaryDirectory(prefix="leonida-f16-interp-") as work:
        for index, (name, lows, highs, geometry) in enumerate(draws):
            words = [packed_word(lows[k], highs[k]) for k in range(3)]
            vs, fs = build_pair(work, words, ORTHO if geometry == "ortho" else PERSPECTIVE)
            pixels, err = run_draw(harness, vs, fs, args.gpu, args.isa and index == 0)
            if not device:
                device = next((line.split("device:", 1)[1].strip() for line in err.splitlines() if "device:" in line), "")
            lv = [h_value(h_bits(x)) for x in lows]
            hv = [h_value(h_bits(x)) for x in highs]
            compare(pixels, lv, hv, counts, fails)
            if args.draw is not None:
                got = {model: counts[model][0] for model in MODELS}
                print(f"draw {index} ({name}, {geometry}): " + ", ".join(f"{m} {n}" for m, n in got.items()))
    total = counts["rdna2"][1]
    print(f"f16_interp_probe: {device or 'device'} | {total} samples over {len(draws)} draws")
    for model in MODELS:
        print(f"  {model:16s} {counts[model][0]:6d}/{total}")
    best = max(MODELS, key=lambda m: counts[m][0])
    for f in fails[best][:4]:
        print(f"  {best} misses: i={f[0]} j={f[1]} halves={f[2]} measured={f[3]!r} model={f[4]!r}")
    explained = counts["rdna2"][0] == total
    print(f"  -> {'every sample is the RDNA2 model' if explained else f'the RDNA2 model explains {counts["rdna2"][0]}'}")
    return 0 if explained else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
