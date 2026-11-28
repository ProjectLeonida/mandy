"""The optional Rust SPIR-V assembler (tools/spirv_asm_native) against the Python one in tools/spirv_asm.py.

Hand-written and translated synthetic modules only (no game data). The comparisons skip when the library is not
built:

    cargo build --release --locked --offline --manifest-path tools/spirv_asm_native/Cargo.toml
    python -I tests/test_spirv_asm_native.py

The native path either gives the bytes of the Python assembler or declines (None), and `assemble()` then runs in
Python; an input Python refuses is always declined, so errors keep their Python message.
"""
import glob
import os
import random
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.join(ROOT, "tests", "semantic")
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spirv_asm  # noqa: E402

NATIVE = spirv_asm._native_assemble("", "vulkan1.2") is not None


def python_assemble(text, target_env="vulkan1.2"):
    with mock.patch.object(spirv_asm, "_NATIVE", [None]):
        return spirv_asm.assemble(text, target_env=target_env)


HEADER = (
    "OpCapability Shader\n"
    "OpCapability Int64\n"
    "OpCapability Int16\n"
    "OpCapability Int8\n"
    "OpCapability Float16\n"
    "OpCapability Float64\n"
    '%glsl = OpExtInstImport "GLSL.std.450"\n'
    "OpMemoryModel Logical GLSL450\n"
    'OpEntryPoint GLCompute %main "main"\n'
    "OpExecutionMode %main LocalSize 8 4 1\n"
    "%void = OpTypeVoid\n"
    "%fn = OpTypeFunction %void\n"
    "%u8 = OpTypeInt 8 0\n%i8 = OpTypeInt 8 1\n"
    "%u16 = OpTypeInt 16 0\n%i16 = OpTypeInt 16 1\n"
    "%u32 = OpTypeInt 32 0\n%i32 = OpTypeInt 32 1\n"
    "%u64 = OpTypeInt 64 0\n%i64 = OpTypeInt 64 1\n"
    "%f16 = OpTypeFloat 16\n%f32 = OpTypeFloat 32\n%f64 = OpTypeFloat 64\n"
)
TYPES = ("%u8", "%i8", "%u16", "%i16", "%u32", "%i32", "%u64", "%i64", "%f16", "%f32", "%f64")

MODULE = HEADER + (
    "; a comment line\n"
    "\t\tOpDecorate %buf DescriptorSet 0 ; inline comment\r\n"
    "OpDecorate %buf Binding 3\r"
    'OpName %main "main; \\"quoted\\" \\\\ tail"\n'
    'OpName %buf "abcd"\n'
    '%src = OpString "x"\n'
    "%ptr = OpTypePointer Function %u32\n"
    "%c0 = OpConstant %u32 0\n%c1 = OpConstant %u32 1\n"
    "%c64 = OpConstant %u64 0x1122334455667788\n"
    "%cneg = OpConstant %i16 -2\n%chex = OpConstant %i16 0xFFFF\n"
    "%h = OpConstant %f16 0.333\n%fx = OpConstant %f32 0x1.8p+0\n%d = OpConstant %f64 -3.5e-7\n"
    "%inf = OpConstant %f32 0x1p+128\n%nan = OpConstant %f32 -0x1.8p+128\n"
    "%sc = OpSpecConstant %u32 7\n"
    "%sco = OpSpecConstantOp %u32 IAdd %sc %c1\n"
    "%t1240_e3 = OpConstant %u32 4294967295\n"
    "%cbf800000 = OpConstant %f32 -1.0\n"
    "%main = OpFunction %void None %fn\n"
    "%entry = OpLabel\n"
    "%buf = OpVariable %ptr Function\n"
    "OpStore %buf %c1 Nontemporal|Aligned 4\n"
    "%ld = OpLoad %u32 %buf Aligned|Volatile|0x20 4\n"
    "%abs = OpExtInst %f32 %glsl FAbs %fx\n"
    "%raw = OpExtInst %f32 %glsl 4 %abs\n"
    "%wide = OpIAdd %u64 %c64 %c64\n"
    "OpSelectionMerge %merge None\n"
    "OpSwitch %wide %def 0 %a 0x100000001 %b\n"
    "%a = OpLabel\nOpBranch %merge\n"
    "%b = OpLabel\nOpBranch %merge\n"
    "%def = OpLabel\nOpBranch %merge\n"
    "%merge = OpLabel\n"
    "%phi = OpPhi %u32 %c0 %a %c1 %b %ld %def\n"
    "OpReturn\n"
    "OpFunctionEnd\n"
)


@unittest.skipUnless(NATIVE, "tools/spirv_asm_native is not built")
class NativeMatchesPython(unittest.TestCase):
    def assert_same(self, text, target_env="vulkan1.2"):
        native = spirv_asm._native_assemble(text, target_env)
        self.assertIsNotNone(native, "the native assembler declined")
        self.assertEqual(native, python_assemble(text, target_env))

    def test_hand_written_module(self):
        for env in sorted(spirv_asm.TARGET_ENV_VERSION):
            with self.subTest(env=env):
                self.assert_same(MODULE, env)

    def test_empty_and_comment_only_text(self):
        for text in ("", "\n", "; nothing\r\n\r\n", "   \t\n"):
            with self.subTest(text=text):
                self.assert_same(text)

    def test_translated_fixtures(self):
        # The hand-written translator fixtures that need no compute header, as rdna2spirv emits them.
        import support
        count = 0
        for path in sorted(glob.glob(os.path.join(support.FIXTURE_DIR, "*.s"))):
            name = os.path.basename(path)[:-2]
            if name.startswith("cs_"):
                continue
            try:
                text = support.translate_fixture(name)
            except Exception:       # a fixture the translator itself refuses is not this test's business
                continue
            with self.subTest(name=name):
                self.assert_same(text)
            count += 1
        self.assertGreater(count, 20)

    def test_many_ids_and_prefixes(self):
        # Numbered names (%t<n>) share per-prefix tables; mixing spellings must still give one id each.
        rng = random.Random(7)
        names = ["%t0", "%t00", "%t1", "%t01", "%t10", "%t99999", "%t100000", "%x_1", "%x_01", "%a1b2",
                 "%abcdefgh1", "%abcdefghi1", "%seventeen_chars_x", "%n"]
        names += [f"%p{i}_e{rng.randrange(40)}" for i in range(60)]
        names += [f"%c{rng.randrange(1 << 32):x}" for _ in range(60)]
        lines = [HEADER]
        for i in range(400):
            a, b = rng.choice(names), rng.choice(names)
            lines.append(f"%r{i} = OpIAdd %u32 {a} {b}\n")
        self.assert_same("".join(lines))

    def test_random_literals(self):
        # Each literal either assembles to the Python bytes or is declined; never a different module.
        rng = random.Random(11)
        pieces = ["0", "1", "-1", "+1", "0x", "0X", "1.5", "-0.0", "e", "E", "+", "-", "p", "P", ".", "_",
                  "00", "7", "9", "f", "F", "a", "inf", "nan", "1e40", "65504", "65520", "x"]
        accepted = 0
        for _ in range(3000):
            tok = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 5)))
            text = f"{HEADER}%k = OpConstant {rng.choice(TYPES)} {tok}\n"
            try:
                expected = python_assemble(text)
            except spirv_asm.SpirvAsmError:
                expected = None
            with self.subTest(tok=tok):
                native = spirv_asm._native_assemble(text, "vulkan1.2")
                if native is not None:
                    self.assertEqual(native, expected)
                    accepted += 1
        self.assertGreater(accepted, 150)

    def test_numeric_literals_of_every_width(self):
        rng = random.Random(3)
        for _ in range(1500):
            ty = rng.choice(TYPES)
            if ty[1] == "f":
                tok = rng.choice([repr(rng.uniform(-1e6, 1e6)), f"{rng.uniform(-70000, 70000):.6g}",
                                  repr(rng.uniform(-1e-5, 1e-5)), f"{rng.getrandbits(52) / 2.0 ** rng.randint(1, 80):e}",
                                  float.hex(rng.uniform(-100, 100)), f"0x1.{rng.getrandbits(8):02x}p{rng.randint(-20, 20)}"])
            else:
                bits = int(ty[2:])
                v = rng.randrange(-(1 << bits), 1 << bits)
                tok = rng.choice([str(v), hex(v), hex(v & ((1 << bits) - 1))])
            text = f"{HEADER}%k = OpConstant {ty} {tok}\n"
            try:
                expected = python_assemble(text)
            except spirv_asm.SpirvAsmError:
                expected = None
            with self.subTest(ty=ty, tok=tok):
                native = spirv_asm._native_assemble(text, "vulkan1.2")
                if expected is not None and not tok.startswith(("0x", "-0x")):
                    self.assertIsNotNone(native, "a plain literal was declined")
                if native is not None:
                    self.assertEqual(native, expected)

    def test_errors_are_declined_and_keep_the_python_message(self):
        bad = [
            "%t = OpTypeInt 32 0\n%t = OpTypeFloat 32\n",
            "OpBogus %x\n",
            "OpCapability Shader extra\n",
            "%x = OpCapability Shader\n",
            "OpName %bad-id \"x\"\n",
            'OpName %x "unterminated\n',
            HEADER + "%k = OpConstant %u32 -1\n",
            HEADER + "%k = OpConstant %u8 256\n",
            HEADER + "%k = OpConstant %i16 +0xFFFF\n",
            HEADER + "%k = OpConstant %f32 1.0.0\n",
            HEADER + "%k = OpConstant %void 1\n",
            HEADER + "OpSwitch %later %d 1 %c\n%later = OpIAdd %u32 %u32 %u32\n",
            HEADER + "%k = OpLoad %u32 %p Aligned\n",
            HEADER + "%k = OpLoad %u32 %p Bogus|Volatile\n",
        ]
        for text in bad:
            with self.subTest(text=text[-60:]):
                self.assertIsNone(spirv_asm._native_assemble(text, "vulkan1.2"))
                with self.assertRaises(spirv_asm.SpirvAsmError) as python_error:
                    python_assemble(text)
                with self.assertRaises(spirv_asm.SpirvAsmError) as hooked_error:
                    spirv_asm.assemble(text)
                self.assertEqual(str(hooked_error.exception), str(python_error.exception))

    def test_declined_spellings_still_assemble_in_python(self):
        # Valid for the Python assembler but left to it by the port: same bytes through assemble().
        cases = [
            HEADER + 'OpName %main "café"\n',
            HEADER + "OpName %main \"x\"\n",
            HEADER + "%k = OpConstant %u32 1_000\n",
            HEADER + "%k = OpConstant %f32 1_0.5\n",
            HEADER + "%k = OpConstant %f32 inf\n",
            HEADER + "%k = OpConstant %f32 0x1.000001p-140\n",
            HEADER + "%k = OpConstant %f32 0x1.0000018p+0\n",
            HEADER + "%k\x0b= OpConstant %u32 1\n",
            "%w = OpTypeInt 24 0\n%k = OpConstant %w 5\n",
        ]
        for text in cases:
            with self.subTest(text=text[-40:]):
                self.assertIsNone(spirv_asm._native_assemble(text, "vulkan1.2"))
                try:
                    expected = python_assemble(text)
                except spirv_asm.SpirvAsmError as exc:
                    with self.assertRaisesRegex(spirv_asm.SpirvAsmError, "^" + __import__("re").escape(str(exc))):
                        spirv_asm.assemble(text)
                else:
                    self.assertEqual(spirv_asm.assemble(text), expected)

    def test_assemble_uses_the_native_path(self):
        expected = python_assemble(MODULE)
        with mock.patch.object(spirv_asm, "_tokenize_line", side_effect=AssertionError("Python path used")):
            self.assertEqual(spirv_asm.assemble(MODULE), expected)

    def test_unknown_target_env_is_refused_in_python(self):
        self.assertIsNone(spirv_asm._native_assemble(MODULE, "vulkan9"))
        with self.assertRaisesRegex(spirv_asm.SpirvAsmError, "unsupported target environment"):
            spirv_asm.assemble(MODULE, target_env="vulkan9")

    def test_threads_get_their_own_results(self):
        from concurrent.futures import ThreadPoolExecutor
        texts = [MODULE.replace("LocalSize 8 4 1", f"LocalSize {i + 1} 1 1") for i in range(32)]
        expected = [python_assemble(t) for t in texts]
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda t: spirv_asm._native_assemble(t, "vulkan1.2"), texts * 4))
        self.assertEqual(results, expected * 4)

    def test_deeply_nested_spec_constant_op_does_not_overflow_the_stack(self):
        # Each `SpecConstantOp` token nests one more OpSpecConstantOp. The native assembler declines past a
        # fixed depth, so Python decides: the same bytes at depth 100, its own RecursionError at depth 1100,
        # and no crash of the interpreter in either case.
        def nested(depth):
            return ("%u = OpTypeInt 32 0\n%a = OpConstant %u 1\n"
                    "%b = OpSpecConstantOp %u " + "SpecConstantOp " * depth + "IAdd %a %a\n")
        shallow = nested(10)
        self.assertEqual(spirv_asm._native_assemble(shallow, "vulkan1.2"), python_assemble(shallow))
        for depth in (65, 100):
            with self.subTest(depth=depth):
                text = nested(depth)
                self.assertIsNone(spirv_asm._native_assemble(text, "vulkan1.2"))
                self.assertEqual(spirv_asm.assemble(text), python_assemble(text))
        deep = nested(1100)
        self.assertIsNone(spirv_asm._native_assemble(deep, "vulkan1.2"))
        with self.assertRaises(RecursionError):
            python_assemble(deep)
        with self.assertRaises(RecursionError):
            spirv_asm.assemble(deep)


class Fallback(unittest.TestCase):
    def test_disabled_by_environment(self):
        # A fresh interpreter with LEONIDA_SPIRV_ASM_NATIVE=0 loads no library and assembles in Python.
        code = (
            "import sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import spirv_asm\n"
            "assert spirv_asm._native_assemble('', 'vulkan1.2') is None\n"
            "sys.stdout.write(spirv_asm.assemble(sys.stdin.read()).hex())\n"
        )
        env = dict(os.environ, LEONIDA_SPIRV_ASM_NATIVE="0")
        result = subprocess.run([sys.executable, "-I", "-c", code, TOOLS], input=MODULE, env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(bytes.fromhex(result.stdout), python_assemble(MODULE))

    def test_missing_library_falls_back(self):
        with mock.patch.object(spirv_asm, "_NATIVE", []), \
                mock.patch.object(spirv_asm.os.path, "isfile", return_value=False):
            self.assertIsNone(spirv_asm._native_assemble(MODULE, "vulkan1.2"))
            self.assertEqual(spirv_asm.assemble(MODULE), python_assemble(MODULE))

    def test_declined_call_falls_back(self):
        with mock.patch.object(spirv_asm, "_NATIVE", [lambda text, version: None]):
            self.assertEqual(spirv_asm.assemble(MODULE), python_assemble(MODULE))
            with self.assertRaisesRegex(spirv_asm.SpirvAsmError, "unknown opcode"):
                spirv_asm.assemble("OpBogus\n")


if __name__ == "__main__":
    unittest.main()
