"""Regression tests for short-form XCHG (0x91..0x97) and MOVNTI (0F C3) in tools/x64_recomp.py.

Usage:
    python -I tests/test_x64_recomp_xchg_movnti.py
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import x64_recomp as xr  # noqa: E402

CAN_EXEC = bool(getattr(xr, "CAN_RUN_MACHO", False) and hasattr(xr, "run_result"))


def _mov_reg_imm64(reg: int, val: int) -> bytes:
    """Emit `mov r64, imm64` (`48/49 B8+rd id`)."""
    rex = 0x48 | (0x01 if reg >= 8 else 0x00)
    return bytes([rex, 0xB8 + (reg & 7)]) + struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF)


def _run_rax64(body: bytes) -> int:
    """Execute `body` via Mach-O runner and return the full 64-bit guest RAX."""
    # mov edi, eax; ret
    lo, ft_lo = xr.run_result(body + b"\x89\xc7\xc3", 0x46000000)
    assert not ft_lo.stats["unhandled"], ft_lo.stats["unhandled"]
    # shr rax, 32; mov edi, eax; ret
    hi, ft_hi = xr.run_result(body + b"\x48\xc1\xe8\x20\x89\xc7\xc3", 0x46000000)
    assert not ft_hi.stats["unhandled"], ft_hi.stats["unhandled"]
    return ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)


class TestShortXchgAndMovntiDecode(unittest.TestCase):
    def test_short_xchg_32bit_operands(self) -> None:
        for i in range(1, 8):
            ins = xr.decode_one(bytes([0x90 + i]), 0, 1, 0x1000)
            self.assertEqual(ins.mnem, "xchg")
            self.assertEqual(ins.kind, "xchg")
            self.assertEqual(ins.size, 1)
            self.assertEqual(ins.ops, (("r", i, 4, False), ("r", 0, 4)))

    def test_short_xchg_16bit_operands(self) -> None:
        for i in range(1, 8):
            ins = xr.decode_one(bytes([0x66, 0x90 + i]), 0, 2, 0x1000)
            self.assertEqual(ins.mnem, "xchg")
            self.assertEqual(ins.ops, (("r", i, 2, False), ("r", 0, 2)))

    def test_short_xchg_64bit_operands(self) -> None:
        for i in range(1, 8):
            ins = xr.decode_one(bytes([0x48, 0x90 + i]), 0, 2, 0x1000)
            self.assertEqual(ins.mnem, "xchg")
            self.assertEqual(ins.ops, (("r", i, 8, False), ("r", 0, 8)))

    def test_short_xchg_rexb_keeps_rax_unextended(self) -> None:
        # 41 91: xchg r9d, eax
        ins32 = xr.decode_one(bytes.fromhex("4191"), 0, 2, 0x1000)
        self.assertEqual(ins32.ops, (("r", 9, 4, False), ("r", 0, 4)))
        # 66 41 92: xchg r10w, ax
        ins16 = xr.decode_one(bytes.fromhex("664192"), 0, 3, 0x1000)
        self.assertEqual(ins16.ops, (("r", 10, 2, False), ("r", 0, 2)))
        # 49 97: xchg r15, rax
        ins64 = xr.decode_one(bytes.fromhex("4997"), 0, 2, 0x1000)
        self.assertEqual(ins64.ops, (("r", 15, 8, False), ("r", 0, 8)))

    def test_movnti_decode_widths_and_66_ignored(self) -> None:
        cases = [
            ("0fc307", 4, 7, 0),       # movnti dword ptr [rdi], eax
            ("660fc307", 4, 7, 0),     # 66 ignored on movnti -> still 32-bit
            ("480fc307", 8, 7, 0),     # movnti qword ptr [rdi], rax
            ("66480fc307", 8, 7, 0),   # REX.W + 66 -> 64-bit
            ("450fc308", 4, 8, 9),     # movnti dword ptr [r8], r9d
            ("4d0fc311", 8, 9, 10),    # movnti qword ptr [r9], r10
        ]
        for hex_bytes, exp_sz, exp_base, exp_reg in cases:
            buf = bytes.fromhex(hex_bytes)
            ins = xr.decode_one(buf, 0, len(buf), 0x1000)
            self.assertEqual(ins.mnem, "movnti", hex_bytes)
            self.assertEqual(ins.kind, "mov", hex_bytes)
            self.assertEqual(ins.size, len(buf), hex_bytes)
            dst, src = ins.ops
            self.assertEqual(dst[0], "m", hex_bytes)
            self.assertEqual(dst[1]["base"], exp_base, hex_bytes)
            self.assertEqual(dst[1]["size"], exp_sz, hex_bytes)
            self.assertEqual(src[:3], ("r", exp_reg, exp_sz), hex_bytes)

    def test_movnti_translates_memory_forms(self) -> None:
        # Valid 32-bit, 66-prefixed 32-bit, and 64-bit movnti translate cleanly with zero unhandled insns.
        code = bytes.fromhex("0fc307480fc307660fc30766480fc307c3")
        ft = xr.FunctionTranslator(list(xr.disasm(code, 0x1000)), xr.Asm())
        ft.translate()
        self.assertEqual(ft.stats["unhandled"], [])


@unittest.skipUnless(CAN_EXEC, "native arm64 Mach-O runner unavailable")
class TestShortXchgAndMovntiExec(unittest.TestCase):
    def test_exec_short_xchg_32_swaps_and_zero_extends(self) -> None:
        setup = (
            _mov_reg_imm64(0, 0xAAAABBBB11112222)
            + _mov_reg_imm64(1, 0xCCCCDDDD33334444)
            + b"\x91"  # xchg ecx, eax
        )
        self.assertEqual(_run_rax64(setup), 0x0000000033334444)
        # mov rax, rcx (48 89 C8)
        self.assertEqual(_run_rax64(setup + b"\x48\x89\xc8"), 0x0000000011112222)

    def test_exec_short_xchg_16_preserves_upper_48_bits(self) -> None:
        setup = (
            _mov_reg_imm64(0, 0x112233445566AABB)
            + _mov_reg_imm64(2, 0x998877665544CCDD)
            + b"\x66\x92"  # xchg dx, ax
        )
        self.assertEqual(_run_rax64(setup), 0x112233445566CCDD)
        # mov rax, rdx (48 89 D0)
        self.assertEqual(_run_rax64(setup + b"\x48\x89\xd0"), 0x998877665544AABB)

    def test_exec_short_xchg_64_swaps_full_width(self) -> None:
        setup = (
            _mov_reg_imm64(0, 0x0123456789ABCDEF)
            + _mov_reg_imm64(3, 0xFEDCBA9876543210)
            + b"\x48\x93"  # xchg rbx, rax
        )
        self.assertEqual(_run_rax64(setup), 0xFEDCBA9876543210)
        # mov rax, rbx (48 89 D8)
        self.assertEqual(_run_rax64(setup + b"\x48\x89\xd8"), 0x0123456789ABCDEF)

    def test_exec_short_xchg_rexb_registers(self) -> None:
        # 41 91: xchg r9d, eax
        s32 = _mov_reg_imm64(0, 0xFFFF000012345678) + _mov_reg_imm64(9, 0xEEEE000087654321) + b"\x41\x91"
        self.assertEqual(_run_rax64(s32), 0x0000000087654321)
        # mov rax, r9 (4C 89 C8)
        self.assertEqual(_run_rax64(s32 + b"\x4c\x89\xc8"), 0x0000000012345678)

        # 66 41 92: xchg r10w, ax
        s16 = _mov_reg_imm64(0, 0xABCDEF0123451111) + _mov_reg_imm64(10, 0x5566778899AA2222) + b"\x66\x41\x92"
        self.assertEqual(_run_rax64(s16), 0xABCDEF0123452222)
        # mov rax, r10 (4C 89 D0)
        self.assertEqual(_run_rax64(s16 + b"\x4c\x89\xd0"), 0x5566778899AA1111)

        # 49 97: xchg r15, rax
        s64 = _mov_reg_imm64(0, 0xCAFEBABE11223344) + _mov_reg_imm64(15, 0xDEADBEEF55667788) + b"\x49\x97"
        self.assertEqual(_run_rax64(s64), 0xDEADBEEF55667788)
        # mov rax, r15 (4C 89 F8)
        self.assertEqual(_run_rax64(s64 + b"\x4c\x89\xf8"), 0xCAFEBABE11223344)

    def test_exec_movnti_32_64_and_66_ignored(self) -> None:
        # Allocate 8-byte stack slot initialized with 0xDEADBEEFCAFEBABE:
        # sub rsp, 8 (48 83 EC 08)
        # mov rcx, 0xDEADBEEFCAFEBABE; mov [rsp], rcx (48 89 0C 24)
        prologue = (
            b"\x48\x83\xec\x08"
            + _mov_reg_imm64(1, 0xDEADBEEFCAFEBABE)
            + b"\x48\x89\x0c\x24"
        )
        # Read back [rsp] into rax and restore rsp: mov rax, [rsp] (48 8B 04 24); add rsp, 8 (48 83 C4 08)
        epilogue = b"\x48\x8b\x04\x24\x48\x83\xc4\x08"

        # 1. 0F C3 04 24: movnti dword ptr [rsp], eax (writes low 32 bits only)
        code32 = prologue + _mov_reg_imm64(0, 0x1111222233334444) + b"\x0f\xc3\x04\x24" + epilogue
        self.assertEqual(_run_rax64(code32), 0xDEADBEEF33334444)

        # 2. 66 0F C3 04 24: 66 prefix ignored on movnti -> still writes full 32 bits (not 16)
        code66 = prologue + _mov_reg_imm64(0, 0x1111222255667788) + b"\x66\x0f\xc3\x04\x24" + epilogue
        self.assertEqual(_run_rax64(code66), 0xDEADBEEF55667788)

        # 3. 48 0F C3 04 24: movnti qword ptr [rsp], rax (writes all 64 bits)
        code64 = prologue + _mov_reg_imm64(0, 0x0123456789ABCDEF) + b"\x48\x0f\xc3\x04\x24" + epilogue
        self.assertEqual(_run_rax64(code64), 0x0123456789ABCDEF)

        # 4. 66 48 0F C3 04 24: REX.W + 66 -> writes all 64 bits
        code66w = prologue + _mov_reg_imm64(0, 0x89ABCDEF01234567) + b"\x66\x48\x0f\xc3\x04\x24" + epilogue
        self.assertEqual(_run_rax64(code66w), 0x89ABCDEF01234567)


if __name__ == "__main__":
    unittest.main()
