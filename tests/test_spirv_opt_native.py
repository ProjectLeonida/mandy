"""The optional native spirv_opt (tools/spirv_opt_native) against the Python reference in tools/spirv_opt.py.

Every input here is synthetic: random modules from the generator below, hand-written edge cases, and the
translator's output for the hand-written tests/semantic fixtures. The comparisons skip when the library is not
built (cargo build --release --manifest-path tools/spirv_opt_native/Cargo.toml); the fallback tests always run.
"""
import os
import random
import subprocess
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import spirv_opt  # noqa: E402

NATIVE = spirv_opt._native_optimize("", 4) is not None


def _python(text, max_passes=4):
    """spirv_opt.optimize with the native path switched off, as (result, None) or (None, exception repr)."""
    saved = list(spirv_opt._NATIVE)
    spirv_opt._NATIVE[:] = [None]
    try:
        return spirv_opt.optimize(text, max_passes), None
    except Exception as exc:  # noqa: BLE001 - the error itself is what is compared
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        spirv_opt._NATIVE[:] = saved


HANG = "hang"


def _compare_guarded(text, max_passes=4):
    """(hooked, python) results compared in a child process with a time limit, or HANG.

    For inputs the port declines: the Python optimizer loops forever on some alias cycles, so this cannot run in
    the test process. Both sides run in the same child, so they share one string-hash seed.
    """
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import test_spirv_opt_native as T; "
            "t = sys.stdin.buffer.read().decode('utf-8'); n = int(sys.argv[2]); "
            "sys.stdout.buffer.write(repr((T._hooked(t, n), T._python(t, n))).encode())")
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code, os.path.dirname(os.path.abspath(__file__)),
                            str(int(max_passes))], input=text.encode("utf-8"), capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        return HANG
    import ast
    return ast.literal_eval(p.stdout.decode("utf-8"))


def _hooked(text, max_passes=4):
    try:
        return spirv_opt.optimize(text, max_passes), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------------------------------------------
# Random module generator: every opcode, operand shape and table the optimizer reads.

INT_TYPES = [("%uint", 32, 0), ("%int", 32, 1), ("%ulong", 64, 0), ("%long", 64, 1), ("%ushort", 16, 0),
             ("%uchar", 8, 0)]
BIN_OPS = ["OpIAdd", "OpISub", "OpIMul", "OpUDiv", "OpUMod", "OpSDiv", "OpBitwiseAnd", "OpBitwiseOr",
           "OpBitwiseXor", "OpShiftLeftLogical", "OpShiftRightLogical", "OpShiftRightArithmetic", "OpFAdd",
           "OpFMul"]
CMP_OPS = ["OpIEqual", "OpINotEqual", "OpULessThan", "OpULessThanEqual", "OpUGreaterThan", "OpUGreaterThanEqual",
           "OpSLessThan", "OpSLessThanEqual", "OpSGreaterThan", "OpSGreaterThanEqual"]
LITS = ["0", "1", "2", "3", "5", "7", "8", "31", "32", "63", "64", "255", "0x0", "0xff", "0xFFFFFFFF", "-1", "-2",
        "4294967295", "4294967296", "0x8000000000000000", "1_0", "0x_1f", "010", "00", "0b101", "0o17", "1.0",
        "18446744073709551615", "-9223372036854775808", "0.0", "+3", "65535", "0x10000000000000003"]


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.n = 0

    def new(self, stem="v"):
        self.n += 1
        return f"%{stem}{self.n}"

    def module(self):
        r = self.rng
        h = ["; SPIR-V", "; Generator: synthetic", "               OpCapability Shader",
             '          %glsl = OpExtInstImport "GLSL.std.450"', "               OpMemoryModel Logical GLSL450",
             '               OpEntryPoint Fragment %main "main" %in0 %out0', "               OpExecutionMode %main OriginUpperLeft"]
        self.decorate_later = []
        h.append("       %void = OpTypeVoid")
        h.append("         %fn = OpTypeFunction %void")
        h.append("       %bool = OpTypeBool")
        h.append("      %float = OpTypeFloat 32")
        for name, w, s in INT_TYPES:
            wtxt = r.choice([str(w)] * 6 + ["0" + str(w), "+" + str(w), str(w)[0] + "_" + str(w)[1:]])
            h.append(f"{name:>12} = OpTypeInt {wtxt} {s}")
        if r.random() < 0.1:
            h.append("      %odd = OpTypeInt 24 0")
        h.append("       %true = OpConstantTrue %bool")
        h.append("      %false = OpConstantFalse %bool")
        if r.random() < 0.03:
            h.append("      %true2 = OpConstantTrue %bool")
        self.consts = {}
        for name, _w, _s in INT_TYPES:
            ids = []
            for lit in r.sample(LITS, r.randint(4, 12)):
                cid = self.new("c")
                h.append(f"{cid:>12} = OpConstant {name} {lit}")
                ids.append(cid)
            self.consts[name] = ids
        self.fconsts = []
        for lit in ("0.0", "1.0", "0", "2.5"):
            cid = self.new("f")
            h.append(f"{cid:>12} = OpConstant %float {lit}")
            self.fconsts.append(cid)
        h.append("     %u4 = OpConstant %uint 4")
        h.append("    %arr = OpTypeArray %uint %u4")
        h.append("   %arrf = OpTypeArray %float %u4")
        h.append("   %parr = OpTypePointer Function %arr")
        h.append("  %parrf = OpTypePointer Function %arrf")
        h.append("     %pu = OpTypePointer Function %uint")
        h.append("     %pf = OpTypePointer Function %float")
        h.append("    %pbo = OpTypePointer Function %bool")
        h.append("   %null = OpConstantNull %arr")
        h.append("  %nullf = OpConstantNull %arrf")
        h.append("   %comp = OpConstantComposite %arr " + " ".join(r.choice(self.consts["%uint"]) for _ in range(4)))
        h.append("    %ipu = OpTypePointer Input %uint")
        h.append("    %opu = OpTypePointer Output %uint")
        h.append("    %ppu = OpTypePointer PushConstant %uint")
        h.append("    %in0 = OpVariable %ipu Input")
        h.append("     %pc = OpVariable %ppu PushConstant")
        h.append("   %out0 = OpVariable %opu Output")
        body = []
        nfun = 1 if r.random() < 0.85 else 2
        for k in range(nfun):
            body += self.function("%main" if k == 0 else self.new("fn"))
            if r.random() < 0.2:
                body.append("%stray = OpTypeInt 32 0")
        decos = []
        for t in r.sample(self.decorate_later, min(len(self.decorate_later), r.randint(0, 6))):
            kind = r.random()
            if kind < 0.5:
                decos.append(f"               OpDecorate {t} RelaxedPrecision")
            elif kind < 0.7:
                decos.append(f'               OpName {t} "n{t[1:]}"')
            elif kind < 0.8:
                decos.append(f'               OpDecorateString {t} UserSemantic "x"')
            else:
                decos.append(f"               OpDecorateId {t} UniformId %u4")
        if len(self.decorate_later) >= 2 and r.random() < 0.3:
            g = r.sample(self.decorate_later, 2)
            decos.append(f"               OpGroupDecorate %grp {g[0]} {g[1]} ; grouped")
        lines = h[:5] + decos + h[5:] + body
        if r.random() < 0.3:
            lines.insert(r.randrange(len(lines)), "")
        if r.random() < 0.3:
            lines.insert(r.randrange(len(lines)), "   ; a comment line")
        sep = r.choice(["\n"] * 8 + ["\r\n", "\r"])
        text = sep.join(lines)
        if r.random() < 0.8:
            text += sep
        if r.random() < 0.05:
            text = text.replace("         ", "\t", 3)
        return text

    def function(self, fname):
        r = self.rng
        out = [f"{fname:>12} = OpFunction %void None %fn"]
        nblocks = r.randint(1, 7)
        labels = [self.new("L") for _ in range(nblocks)]
        if r.random() < 0.1:
            labels.append(self.new("dead"))   # unreachable block
        out.append(f"{labels[0]:>12} = OpLabel")
        vars_ = []
        for _ in range(r.randint(1, 5)):
            v = self.new("var")
            kind = r.random()
            if kind < 0.35:
                out.append(f"{v:>12} = OpVariable %parr Function %null")
                vars_.append((v, "arr"))
            elif kind < 0.45:
                out.append(f"{v:>12} = OpVariable %parrf Function %nullf")
                vars_.append((v, "arrf"))
            elif kind < 0.6:
                out.append(f"{v:>12} = OpVariable %parr Function")
                vars_.append((v, "arr"))
            elif kind < 0.75:
                out.append(f"{v:>12} = OpVariable %pu Function {r.choice(self.consts['%uint'])}")
                vars_.append((v, "u"))
            elif kind < 0.85:
                out.append(f"{v:>12} = OpVariable %pbo Function %true")
                vars_.append((v, "b"))
            else:
                out.append(f"{v:>12} = OpVariable %pu Function")
                vars_.append((v, "u"))
            self.decorate_later.append(v)
        self.vals = {"%uint": list(self.consts["%uint"]), "%bool": ["%true", "%false"], "%float": list(self.fconsts)}
        for name, _w, _s in INT_TYPES:
            self.vals.setdefault(name, list(self.consts[name]))
        for bi, lab in enumerate(labels):
            if bi:
                out.append(f"{lab:>12} = OpLabel")
            for _ in range(r.randint(2, 18)):
                out += self.inst(vars_, labels[:bi + 1])
            # terminator
            t = r.random()
            later = labels[bi + 1:nblocks] or [None]
            if bi >= nblocks - 1 or later == [None]:
                if r.random() < 0.15 and bi > 0:
                    out.append(f"               OpBranch {r.choice(labels[:bi + 1])}")
                else:
                    out.append("               OpReturn")
            elif t < 0.4:
                out.append(f"               OpBranch {r.choice(later)}")
            elif t < 0.8:
                c = r.choice(self.vals["%bool"])
                out.append(f"               OpBranchConditional {c} {r.choice(later)} {r.choice(later + labels[:bi])}")
            else:
                sel = r.choice(self.vals["%uint"])
                cases = " ".join(f"{k} {r.choice(later)}" for k in range(r.randint(0, 3)))
                out.append(f"               OpSwitch {sel} {r.choice(later)} {cases}".rstrip())
        out.append("               OpFunctionEnd")
        return out

    def pick(self, ty):
        return self.rng.choice(self.vals[ty])

    def define(self, ty, rid):
        self.vals.setdefault(ty, []).append(rid)
        if self.rng.random() < 0.05:
            self.decorate_later.append(rid)

    def inst(self, vars_, seen_labels):
        r = self.rng
        k = r.random()
        rid = self.new()
        out = []
        if k < 0.25 and vars_:
            v, kind = r.choice(vars_)
            if kind in ("arr", "arrf"):
                elem_ty, pty = ("%uint", "%pu") if kind == "arr" else ("%float", "%pf")
                if r.random() < 0.15:
                    idx = self.pick("%uint")  # may be dynamic (a loaded value) or constant
                else:
                    idx = r.choice(self.consts["%uint"][:4] + self.consts["%int"][:2])
                out.append(f"{rid:>12} = OpAccessChain {pty} {v} {idx}")
                ptr = rid
                if r.random() < 0.05:
                    ptr = v      # whole-array access: mixed depths
                    elem_ty = "%arr"
            else:
                ptr, elem_ty = v, ("%uint" if kind == "u" else "%bool")
            if r.random() < 0.5 or elem_ty == "%arr":
                lid = self.new()
                vol = " Volatile" if r.random() < 0.03 else ""
                out.append(f"{lid:>12} = OpLoad {elem_ty} {ptr}{vol}")
                if elem_ty != "%arr":
                    self.define(elem_ty, lid)
            else:
                val = self.pick(elem_ty) if elem_ty in self.vals else self.pick("%uint")
                vol = " Volatile" if r.random() < 0.03 else ""
                out.append(f"               OpStore {ptr} {val}{vol}")
            return out
        if k < 0.30:
            src = r.choice(["%in0", "%pc"])
            out.append(f"{rid:>12} = OpLoad %uint {src}")
            self.define("%uint", rid)
            return out
        if k < 0.55:
            name, _w, _s = r.choice(INT_TYPES[:3])
            op = r.choice(BIN_OPS)
            a, b = self.pick(name), self.pick(name)
            if r.random() < 0.15:
                b = a
            out.append(f"{rid:>12} = {op} {name} {a} {b}")
            self.define(name, rid)
            return out
        if k < 0.63:
            name, _w, _s = r.choice(INT_TYPES[:3])
            op = r.choice(CMP_OPS)
            a, b = self.pick(name), self.pick(name)
            if r.random() < 0.2:
                b = a
            out.append(f"{rid:>12} = {op} %bool {a} {b}")
            self.define("%bool", rid)
            return out
        if k < 0.68:
            c = self.pick("%bool")
            x, y = self.pick("%uint"), self.pick("%uint")
            if r.random() < 0.3:
                x, y = r.choice(self.consts["%uint"]), r.choice(self.consts["%uint"])
            if r.random() < 0.2:
                out.append(f"{rid:>12} = OpSelect %bool {c} %true %false")
                self.define("%bool", rid)
            else:
                out.append(f"{rid:>12} = OpSelect %uint {c} {x} {y}")
                self.define("%uint", rid)
            return out
        if k < 0.73:
            op = r.choice(["OpLogicalNot", "OpLogicalAnd", "OpLogicalOr"])
            if op == "OpLogicalNot":
                out.append(f"{rid:>12} = {op} %bool {self.pick('%bool')}")
            else:
                out.append(f"{rid:>12} = {op} %bool {self.pick('%bool')} {self.pick('%bool')}")
            self.define("%bool", rid)
            return out
        if k < 0.77:
            out.append(f"{rid:>12} = OpCopyObject %uint {self.pick('%uint')}")
            self.define("%uint", rid)
            return out
        if k < 0.81:
            src = self.pick("%uint")
            out.append(f"{rid:>12} = OpBitcast %float {src}")
            self.define("%float", rid)
            if r.random() < 0.6:
                r2 = self.new()
                out.append(f"{r2:>12} = OpBitcast %uint {rid}")
                self.define("%uint", r2)
            return out
        if k < 0.84:
            elems = [self.pick("%uint") for _ in range(r.randint(2, 4))]
            out.append(f"{rid:>12} = OpCompositeConstruct %v4 {' '.join(elems)}")
            r2 = self.new()
            idx = r.choice(["0", "1", "3", "7", "01", "x"])
            out.append(f"{r2:>12} = OpCompositeExtract %uint {rid} {idx}")
            self.define("%uint", r2)
            return out
        if k < 0.87:
            op = r.choice(["FAbs", "Sqrt", "UMin", "FMax", "Frexp"])
            if op in ("UMin", "FMax"):
                out.append(f"{rid:>12} = OpExtInst %uint %glsl {op} {self.pick('%uint')} {self.pick('%uint')}")
            else:
                out.append(f"{rid:>12} = OpExtInst %float %glsl {op} {self.pick('%float')}")
            self.define("%uint" if op in ("UMin", "FMax") else "%float", rid)
            return out
        if k < 0.90:
            op = r.choice(["OpNot", "OpUConvert"])
            name = r.choice(["%uint", "%ulong", "%ushort", "%uchar"])
            out.append(f"{rid:>12} = {op} {name} {self.pick(r.choice(['%uint', '%ulong']))}")
            self.define(name, rid)
            return out
        if k < 0.93:
            base = self.pick("%uint")
            off, cnt = r.choice(self.consts["%uint"]), r.choice(self.consts["%uint"])
            out.append(f"{rid:>12} = OpBitFieldUExtract %uint {base} {off} {cnt}")
            self.define("%uint", rid)
            return out
        if k < 0.96 and len(seen_labels) > 1:
            pairs = " ".join(f"{self.pick('%uint')} {lab}" for lab in r.sample(seen_labels, min(len(seen_labels), 2)))
            if r.random() < 0.3:
                pairs += f" {rid} {seen_labels[0]}"
            out.append(f"{rid:>12} = OpPhi %uint {pairs}")
            self.define("%uint", rid)
            return out
        if k < 0.98:
            out.append(f"               OpStore %out0 {self.pick('%uint')}")
            return out
        out.append(f"{rid:>12} = OpFunctionCall %void %helper {self.pick('%uint')} ; call")
        return out


def _random_modules(seed, count):
    rng = random.Random(seed)
    for _ in range(count):
        yield Gen(rng).module()


def _fixture_modules():
    """The translator's output for the hand-written tests/semantic fixtures (no game data)."""
    if SEMANTIC not in sys.path:
        sys.path.insert(0, SEMANTIC)
    import support  # noqa: E402
    out = []
    for fn in sorted(os.listdir(support.FIXTURE_DIR)):
        if not fn.endswith(".s"):
            continue
        try:
            out.append(support.translate_fixture(fn[:-2]))
        except Exception:  # noqa: BLE001 - fixtures that need a header blob are simply left out
            continue
    return out


EDGE_CASES = [
    "",
    "\n",
    "OpCapability Shader",
    "%f = OpFunction %v None %fn\n%l = OpLabel\nOpReturn\nOpFunctionEnd",
    "%f = OpFunction %v None %fn\r\n%l = OpLabel\r\n  OpReturn ; c\r\nOpFunctionEnd\r\n",
    "%f = OpFunction %v None %fn\n%l = OpLabel\n%a = OpUndef\nOpReturn\nOpFunctionEnd\n",       # IndexError
    "%f = OpFunction %v None %fn\n%l = OpLabel\n%a = OpCopyObject %t %a\nOpStore %p %a\nOpReturn\nOpFunctionEnd\n",
    "%f = OpFunction %v None %fn\n%l = OpLabel\nOpReturn\n",                                       # no end
    "%f = OpFunction %v None %fn\n%l = OpLabel\nOpName %x \"café\"\nOpReturn\nOpFunctionEnd\n",   # not ASCII
    "%f = OpFunction %v None %fn\n%l = OpLabel\x0cOpReturn\nOpFunctionEnd\n",                     # form feed
    "%uint = OpTypeInt 32 0\n%c = OpConstant %uint " + "9" * 5000 + "\n%f = OpFunction %v None %fn\n%l = OpLabel\n"
    "OpReturn\nOpFunctionEnd\n",                                                                    # int digit limit
    "%t = OpConstantTrue %b\n%t2 = OpConstantTrue %b\n%f = OpFunction %v None %fn\n%l = OpLabel\n"
    "%x = OpLogicalNot %b %t2\nOpStore %o %x\nOpReturn\nOpFunctionEnd\n",                         # set order
    "%f = OpFunction %v None %fn\nOpReturn\nOpFunctionEnd\n%g = OpFunction %v None %fn\n%l = OpLabel\n"
    "%a = OpIAdd %u %x %y\n%b = OpIAdd %u %y %x\nOpStore %o %b\nOpReturn\nOpFunctionEnd\n",
]


@unittest.skipUnless(NATIVE, "tools/spirv_opt_native is not built")
class NativeParity(unittest.TestCase):
    def check(self, text, max_passes=4):
        """Native and Python agree; where the port declines, optimize() gives exactly the Python result."""
        if spirv_opt._native_optimize(text, max_passes) is None:
            res = _compare_guarded(text, max_passes)
            if res == HANG:
                return res      # Python never returns here; the port declined instead of guessing
            self.assertEqual(res[0], res[1], msg=text[:4000])
            return res[1]
        want = _python(text, max_passes)
        got = _hooked(text, max_passes)
        self.assertEqual(got, want, msg=text[:4000])
        return want

    def test_edge_cases(self):
        for text in EDGE_CASES[:6] + EDGE_CASES[7:]:
            with self.subTest(text=text[:80]):
                self.check(text)

    def test_declines_where_python_raises_or_differs(self):
        # [6] is an alias cycle where the Python resolve() loops forever.
        for text in (EDGE_CASES[5], EDGE_CASES[6], EDGE_CASES[8], EDGE_CASES[9], EDGE_CASES[10], EDGE_CASES[11]):
            self.assertIsNone(spirv_opt._native_optimize(text, 4))

    def test_random_modules(self):
        n = declined = hangs = 0
        for text in _random_modules(0x5EED, 1500):
            hangs += self.check(text) == HANG
            n += 1
            declined += spirv_opt._native_optimize(text, 4) is None
        self.assertEqual(n, 1500)
        self.assertLess(declined, n // 10, "the port declines too often")
        print(f"\n{n} random modules: {n - declined} native, {declined} declined ({hangs} where Python hangs)",
              file=sys.stderr)

    def test_random_modules_pass_counts(self):
        for k, text in enumerate(_random_modules(0xC0FFEE, 200)):
            self.check(text, max_passes=(k % 7) - 1)
        self.check(EDGE_CASES[3], True)
        self.check(EDGE_CASES[3], 10 ** 30)

    def test_translated_fixtures(self):
        mods = _fixture_modules()
        self.assertGreater(len(mods), 5)
        for text in mods:
            want = self.check(text)
            self.assertIsNotNone(spirv_opt._native_optimize(text, 4))
            self.assertIsNotNone(want[0])

    def test_threads(self):
        mods = [t for t in _random_modules(7, 60) if spirv_opt._native_optimize(t, 4) is not None]
        want = [_python(t)[0] for t in mods]
        errors = []

        def work():
            for t, w in zip(mods, want):
                if spirv_opt.optimize(t) != w:
                    errors.append(t)
        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_benchmark(self):
        mods = [t for t in _fixture_modules() + list(_random_modules(99, 200))
                if spirv_opt._native_optimize(t, 4) is not None]
        t0 = time.perf_counter()
        ref = [_python(t)[0] for t in mods]
        t_py = time.perf_counter() - t0
        t0 = time.perf_counter()
        got = [spirv_opt.optimize(t) for t in mods]
        t_nat = time.perf_counter() - t0
        self.assertEqual(got, ref)
        mb = sum(len(t) for t in mods) / 1e6
        print(f"\nspirv_opt over {len(mods)} synthetic modules ({mb:.1f} MB): Python {t_py * 1e3:.0f} ms, "
              f"native {t_nat * 1e3:.1f} ms, {t_py / t_nat:.1f}x", file=sys.stderr)


class Fallback(unittest.TestCase):
    def test_env_opt_out(self):
        old = os.environ.get("LEONIDA_SPIRV_OPT_NATIVE")
        os.environ["LEONIDA_SPIRV_OPT_NATIVE"] = "0"
        try:
            self.assertIsNone(spirv_opt._load_native())
        finally:
            if old is None:
                del os.environ["LEONIDA_SPIRV_OPT_NATIVE"]
            else:
                os.environ["LEONIDA_SPIRV_OPT_NATIVE"] = old

    def test_missing_library(self):
        real = os.path.isfile
        try:
            os.path.isfile = lambda p: False if "spirv_opt_native" in p else real(p)
            self.assertIsNone(spirv_opt._load_native())
        finally:
            os.path.isfile = real

    def test_declined_and_disabled_calls_give_python_result(self):
        saved = list(spirv_opt._NATIVE)
        try:
            for fake in (None, lambda text, passes: None):
                spirv_opt._NATIVE[:] = [fake]
                for text in [t for t in _random_modules(3, 20) if NATIVE and spirv_opt._native_optimize(t, 4)
                             is not None] + EDGE_CASES[:5]:
                    self.assertEqual(_hooked(text), _python(text))
        finally:
            spirv_opt._NATIVE[:] = saved

    def test_non_str_and_odd_pass_counts_stay_in_python(self):
        self.assertIsNone(spirv_opt._native_optimize(b"OpCapability Shader", 4))
        self.assertIsNone(spirv_opt._native_optimize("", 4.0))
        self.assertIsNone(spirv_opt._native_optimize("", 1 << 70))

    def test_int_subclass_pass_count_matches_python(self):
        class Passes(int):
            def __int__(self):
                return 0

        # range() uses __index__, so Python runs 3 passes; int(Passes(3)) would give the port 0.
        texts = [t for t in _random_modules(7, 20) if _python(t, 3) != _python(t, 0)]
        self.assertTrue(texts)
        for text in texts:
            self.assertIsNone(spirv_opt._native_optimize(text, Passes(3)))
            self.assertEqual(_hooked(text, Passes(3)), _python(text, Passes(3)))


if __name__ == "__main__":
    unittest.main()
