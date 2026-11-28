"""Unit tests for x64 .xdata UNWIND_INFO / UNWIND_CODE decoding in tools/recomp_census.py.

Builds synthetic PE32+ images byte-by-byte with hand-constructed .pdata and .xdata
records (no game files, standard library only).

    python -I tests/test_recomp_census_unwind.py
"""
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import pe_inspect  # noqa: E402
import recomp_census as rc  # noqa: E402

TEXT_RVA = 0x1000
TEXT_RAW = 0x200
TEXT_SIZE = 0x4000
IMAGE_BASE = 0x140000000


def unwind_header(version, flags, prolog_size, code_count, frame_reg=0, frame_off_scaled=0):
    b0 = (version & 0x7) | ((flags & 0x1F) << 3)
    b3 = (frame_reg & 0xF) | ((frame_off_scaled & 0xF) << 4)
    return bytes([b0, prolog_size & 0xFF, code_count & 0xFF, b3])


def ucode(code_offset, op, info=0):
    return bytes([code_offset & 0xFF, (op & 0xF) | ((info & 0xF) << 4)])


def u16(val):
    return struct.pack("<H", val)


def u32(val):
    return struct.pack("<I", val)


def build_pe_with_xdata(pdata_list, xdata_blobs, code_blobs=None):
    """Build a minimal PE32+ image placing code, .pdata, and .xdata blobs in .text."""
    opt_size = 240
    buf = bytearray(TEXT_RAW + TEXT_SIZE)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, 0x40)
    buf[0x40:0x44] = b"PE\0\0"
    struct.pack_into("<HHIIIHH", buf, 0x44, 0x8664, 1, 0, 0, 0, opt_size, 0x0022)
    opt = 0x58
    struct.pack_into("<H", buf, opt, 0x20B)
    struct.pack_into("<I", buf, opt + 20, TEXT_RVA)
    struct.pack_into("<Q", buf, opt + 24, IMAGE_BASE)
    struct.pack_into("<I", buf, opt + 56, 0x6000)
    struct.pack_into("<I", buf, opt + 60, TEXT_RAW)
    struct.pack_into("<I", buf, opt + 108, 16)
    pdata_rva = 0x2000
    struct.pack_into("<II", buf, opt + 112 + 3 * 8, pdata_rva, len(pdata_list) * 12)
    sec = opt + opt_size
    buf[sec:sec + 8] = b".text\0\0\0"
    struct.pack_into("<IIII", buf, sec + 8, TEXT_SIZE, TEXT_RVA, TEXT_SIZE, TEXT_RAW)

    def place(rva, blob):
        at = TEXT_RAW + (rva - TEXT_RVA)
        buf[at:at + len(blob)] = blob

    for rva, blob in (code_blobs or {}).items():
        place(rva, blob)
    for i, (b, e, u) in enumerate(pdata_list):
        place(pdata_rva + i * 12, struct.pack("<III", b, e, u))
    for rva, blob in xdata_blobs.items():
        place(rva, blob)
    return bytes(buf)


class UnwindHeaderBugfixTests(unittest.TestCase):
    """Regression tests for UNW_FLAG_CHAININFO=0x4 and byte-2 CountOfCodes."""

    def test_unwind_flags_reads_count_of_codes_from_byte_2_and_chained_from_bit_4(self):
        # prolog_size=0x18 (byte 1), code_count=3 (byte 2), flags=UNW_FLAG_CHAININFO (0x4)
        blob = unwind_header(version=1, flags=0x4, prolog_size=0x18, code_count=3)
        img = pe_inspect.PeImage(build_pe_with_xdata([], {0x3000: blob}))
        ver, flags, count = rc.unwind_flags(img, 0x3000)
        self.assertEqual(ver, 1)
        self.assertEqual(flags, ["CHAINED"])
        self.assertEqual(count, 3)

    def test_unwind_flags_ehandler_and_uhandler_combined(self):
        blob = unwind_header(version=2, flags=0x3, prolog_size=0x0C, code_count=0)
        img = pe_inspect.PeImage(build_pe_with_xdata([], {0x3000: blob}))
        ver, flags, count = rc.unwind_flags(img, 0x3000)
        self.assertEqual((ver, flags, count), (2, ["EHANDLER", "UHANDLER"], 0))

    def test_bit_8_is_not_chained(self):
        blob = unwind_header(version=1, flags=0x8, prolog_size=4, code_count=0)
        img = pe_inspect.PeImage(build_pe_with_xdata([], {0x3000: blob}))
        _ver, flags, _count = rc.unwind_flags(img, 0x3000)
        self.assertEqual(flags, [])
        parsed = rc.parse_unwind_info(img, 0x3000)
        self.assertIsNotNone(parsed["error"])
        self.assertIn("unknown unwind flags", parsed["error"])


class UnwindOpDecodeTests(unittest.TestCase):
    """Decode every x64 UNWIND_CODE opcode (0..10), padding, handlers, and chains."""

    def test_all_prologue_register_and_alloc_ops_with_odd_slot_padding(self):
        # 15 slots (odd -> padded to 16 slots = 32 bytes before handler RVA):
        #  1: PUSH_NONVOL rbx (1 slot) -> +8
        #  2: ALLOC_SMALL info=3 -> 32 bytes (1 slot)
        #  3: ALLOC_LARGE info=0, 0x20 -> 0x100 bytes (2 slots)
        #  4: ALLOC_LARGE info=1, 0x0002_0040 (3 slots)
        #  5: SET_FPREG rbp, FrameOffset=2*16=32 (1 slot)
        #  6: SAVE_NONVOL rsi, offset=4*8=32 (2 slots)
        #  7: SAVE_NONVOL_FAR r12, offset=0x0001_0020 (3 slots)
        #  8: SAVE_XMM128 xmm6, offset=3*16=48 (2 slots)
        codes = b"".join([
            ucode(0x02, 0, 3),                              # PUSH_NONVOL rbx (1)
            ucode(0x06, 2, 3),                              # ALLOC_SMALL 32 (1)
            ucode(0x0D, 1, 0) + u16(0x20),                  # ALLOC_LARGE 256 (2)
            ucode(0x14, 1, 1) + u32(0x00020040),            # ALLOC_LARGE 0x20040 (3)
            ucode(0x18, 3, 0),                              # SET_FPREG rbp (1)
            ucode(0x1D, 4, 6) + u16(4),                     # SAVE_NONVOL rsi @ 32 (2)
            ucode(0x24, 5, 12) + u32(0x00010020),           # SAVE_NONVOL_FAR r12 @ 0x10020 (3)
            ucode(0x2C, 8, 6) + u16(3),                     # SAVE_XMM128 xmm6 @ 48 (2)
        ])
        self.assertEqual(len(codes), 15 * 2)
        handler_rva = 0x1900
        # 15 slots -> 2 bytes padding before handler_rva
        blob = unwind_header(1, rc.UNW_FLAG_EHANDLER, 0x30, 15, frame_reg=5, frame_off_scaled=2) + codes + b"\x00\x00" + u32(handler_rva)
        img = pe_inspect.PeImage(build_pe_with_xdata([], {0x3000: blob}))
        info = rc.parse_unwind_info(img, 0x3000)

        self.assertIsNone(info["error"])
        self.assertEqual(info["version"], 1)
        self.assertEqual(info["flags"], ["EHANDLER"])
        self.assertEqual(info["prolog_size"], 0x30)
        self.assertEqual(info["code_count"], 15)
        self.assertEqual(info["frame_register"], "rbp")
        self.assertEqual(info["frame_offset"], 32)
        self.assertEqual(info["stack_alloc"], 8 + 32 + 256 + 0x20040)
        self.assertEqual(info["saved_gprs"], ["rbx", "rsi", "r12"])
        self.assertEqual(info["saved_xmms"], ["xmm6"])
        self.assertEqual(info["handler_rva"], handler_rva)
        self.assertEqual(len(info["ops"]), 8)
        self.assertEqual(info["ops"][5], {"op": "UWOP_SAVE_NONVOL", "code_offset": 0x1D, "reg": "rsi", "offset": 32})
        self.assertEqual(info["ops"][6], {"op": "UWOP_SAVE_NONVOL_FAR", "code_offset": 0x24, "reg": "r12", "offset": 0x10020})
        self.assertEqual(info["ops"][7], {"op": "UWOP_SAVE_XMM128", "code_offset": 0x2C, "reg": "xmm6", "offset": 48})

    def test_v2_epilog_spare_xmm_far_and_machframe(self):
        # 10 slots (even -> 0 padding bytes before UHANDLER):
        #  2 UWOP_EPILOG (1st at_end=True len=5; 2nd offset=0x120) -> 2 slots
        #  1 UWOP_EPILOG padding (offset=0) -> 1 slot
        #  1 UWOP_SPARE_CODE -> 3 slots
        #  1 UWOP_SAVE_XMM128_FAR xmm15 @ 0x30000 -> 3 slots
        #  1 UWOP_PUSH_MACHFRAME errcode=1 -> 1 slot
        codes = b"".join([
            ucode(0x05, 6, 1),                              # EPILOG at_end=True, length=5 (1)
            ucode(0x20, 6, 1),                              # EPILOG offset=0x120 (1)
            ucode(0x00, 6, 0),                              # EPILOG padding (1)
            ucode(0x00, 7, 0) + u16(0) + u16(0),            # SPARE_CODE (3)
            ucode(0x10, 9, 15) + u32(0x30000),              # SAVE_XMM128_FAR xmm15 @ 0x30000 (3)
            ucode(0x00, 10, 1),                             # PUSH_MACHFRAME errcode=yes (1)
        ])
        self.assertEqual(len(codes), 10 * 2)
        blob = unwind_header(2, rc.UNW_FLAG_UHANDLER, 0x14, 10) + codes + u32(0x1A00)
        img = pe_inspect.PeImage(build_pe_with_xdata([], {0x3000: blob}))
        info = rc.parse_unwind_info(img, 0x3000)

        self.assertIsNone(info["error"])
        self.assertEqual(info["version"], 2)
        self.assertEqual(info["epilog_count"], 2)  # at_end + offset 0x120; padding slot excluded
        self.assertTrue(info["ops"][0]["at_end"])
        self.assertEqual(info["ops"][0]["length"], 5)
        self.assertEqual(info["ops"][1]["epilog_offset"], 0x120)
        self.assertTrue(info["ops"][2].get("padding"))
        self.assertEqual(info["ops"][3]["op"], "UWOP_SPARE_CODE")
        self.assertEqual(info["ops"][4], {"op": "UWOP_SAVE_XMM128_FAR", "code_offset": 0x10, "reg": "xmm15", "offset": 0x30000})
        self.assertEqual(info["saved_xmms"], ["xmm15"])
        self.assertTrue(info["has_machframe"])
        self.assertTrue(info["ops"][5]["has_error_code"])
        self.assertEqual(info["handler_rva"], 0x1A00)

    def test_multi_hop_chained_unwind_aggregates_frame_and_handler(self):
        # Secondary (cold) entry @ 0x3000 -> chains to middle @ 0x3040 -> chains to primary @ 0x3080 (with EHANDLER)
        # Cold: 1 odd slot (ALLOC_SMALL 16B) + 2B pad + chained RUNTIME_FUNCTION -> 0x3040
        cold = (unwind_header(1, rc.UNW_FLAG_CHAININFO, 0x04, 1)
                + ucode(0x04, 2, 1) + b"\x00\x00"
                + struct.pack("<III", 0x1000, 0x1020, 0x3040))
        # Middle: 2 even slots (PUSH_NONVOL rdi, PUSH_NONVOL rbx) + chained RUNTIME_FUNCTION -> 0x3080
        mid = (unwind_header(1, rc.UNW_FLAG_CHAININFO, 0x04, 2)
               + ucode(0x02, 0, 7) + ucode(0x01, 0, 3)
               + struct.pack("<III", 0x1000, 0x1020, 0x3080))
        # Primary: EHANDLER @ 0x1F00, SET_FPREG rbp + SAVE_XMM128 xmm7 + PUSH_NONVOL rbx (deduped)
        prim_codes = ucode(0x08, 3, 0) + ucode(0x06, 8, 7) + u16(1) + ucode(0x02, 0, 3)  # 4 slots (even)
        prim = unwind_header(1, rc.UNW_FLAG_EHANDLER, 0x08, 4, frame_reg=5, frame_off_scaled=1) + prim_codes + u32(0x1F00)

        img = pe_inspect.PeImage(build_pe_with_xdata([], {0x3000: cold, 0x3040: mid, 0x3080: prim}))
        info = rc.parse_unwind_info(img, 0x3000)

        self.assertIsNone(info["error"])
        self.assertEqual(info["flags"], ["CHAINED"])
        self.assertEqual(info["chain_depth"], 2)
        self.assertEqual(info["chained_rvas"], [0x3040, 0x3080])
        self.assertEqual(info["handler_rva"], 0x1F00)
        self.assertEqual(info["frame_register"], "rbp")
        self.assertEqual(info["frame_offset"], 16)
        self.assertEqual(info["stack_alloc"], 16 + 8 + 8 + 8)
        self.assertEqual(info["saved_gprs"], ["rdi", "rbx"])
        self.assertEqual(info["saved_xmms"], ["xmm7"])


class UnwindMalformedTests(unittest.TestCase):
    """Malformed, truncated, and cyclic .xdata records fail cleanly without raising."""

    def parse_single(self, blobs, rva=0x3000):
        img = pe_inspect.PeImage(build_pe_with_xdata([], blobs))
        return rc.parse_unwind_info(img, rva)

    def test_unmapped_rva(self):
        info = self.parse_single({}, rva=0x9000)
        self.assertIn("unmapped unwind header", info["error"])

    def test_unsupported_versions(self):
        for ver in (0, 3):
            with self.subTest(version=ver):
                info = self.parse_single({0x3000: unwind_header(ver, 0, 0, 0)})
                self.assertIn("unsupported unwind version", info["error"])

    def test_chained_with_handler_flag_rejected(self):
        for flags in (0x5, 0x6, 0x7):
            with self.subTest(flags=flags):
                info = self.parse_single({0x3000: unwind_header(1, flags, 0, 0)})
                self.assertIn("CHAINED combined with EHANDLER/UHANDLER", info["error"])

    def test_truncated_multislot_ops(self):
        cases = [
            ("alloc_large_0", unwind_header(1, 0, 4, 1) + ucode(4, 1, 0)),
            ("alloc_large_1", unwind_header(1, 0, 4, 2) + ucode(4, 1, 1) + u16(0)),
            ("save_nonvol", unwind_header(1, 0, 4, 1) + ucode(4, 4, 3)),
            ("save_nonvol_far", unwind_header(1, 0, 4, 2) + ucode(4, 5, 3) + u16(0)),
            ("spare_code", unwind_header(2, 0, 4, 2) + ucode(4, 7, 0) + u16(0)),
            ("save_xmm128", unwind_header(1, 0, 4, 1) + ucode(4, 8, 6)),
            ("save_xmm128_far", unwind_header(1, 0, 4, 2) + ucode(4, 9, 6) + u16(0)),
        ]
        for label, blob in cases:
            with self.subTest(label=label):
                info = self.parse_single({0x3000: blob})
                self.assertIn("truncated", info["error"])

    def test_invalid_alloc_large_info_and_unknown_op(self):
        info = self.parse_single({0x3000: unwind_header(1, 0, 4, 1) + ucode(4, 1, 2)})
        self.assertIn("invalid UWOP_ALLOC_LARGE info", info["error"])

        info2 = self.parse_single({0x3000: unwind_header(1, 0, 4, 1) + ucode(4, 11, 0)})
        self.assertIn("unknown unwind op 11", info2["error"])

    def test_set_fpreg_with_zero_frame_register(self):
        info = self.parse_single({0x3000: unwind_header(1, 0, 4, 1, frame_reg=0) + ucode(4, 3, 0)})
        self.assertIn("FrameRegister=0", info["error"])

    def test_epilog_in_version_1_rejected(self):
        info = self.parse_single({0x3000: unwind_header(1, 0, 4, 1) + ucode(4, 6, 1)})
        self.assertIn("version 2", info["error"])

    def test_truncated_codes_and_trailing_data_at_section_end(self):
        # .text ends at RVA 0x5000 (TEXT_RVA 0x1000 + TEXT_SIZE 0x4000).
        # Place header at 0x4FFC (only 4 bytes left) claiming 2 codes -> truncated codes
        info = self.parse_single({0x4FFC: unwind_header(1, 0, 4, 2)}, rva=0x4FFC)
        self.assertIn("truncated unwind codes", info["error"])

        # Place EHANDLER header at 0x4FFC with 0 codes -> handler RVA at 0x5000 is outside section
        info2 = self.parse_single({0x4FFC: unwind_header(1, rc.UNW_FLAG_EHANDLER, 0, 0)}, rva=0x4FFC)
        self.assertIn("truncated exception handler RVA", info2["error"])

        # Place CHAINED header at 0x4FF8 with 0 codes -> chained RUNTIME_FUNCTION needs 12B, only 4B left
        info3 = self.parse_single({0x4FF8: unwind_header(1, rc.UNW_FLAG_CHAININFO, 0, 0)}, rva=0x4FF8)
        self.assertIn("truncated chained RUNTIME_FUNCTION", info3["error"])

    def test_chained_cycle_detected(self):
        # 0x3000 -> 0x3020 -> 0x3000
        a = unwind_header(1, rc.UNW_FLAG_CHAININFO, 0, 0) + struct.pack("<III", 0x1000, 0x1004, 0x3020)
        b = unwind_header(1, rc.UNW_FLAG_CHAININFO, 0, 0) + struct.pack("<III", 0x1000, 0x1004, 0x3000)
        info = self.parse_single({0x3000: a, 0x3020: b})
        self.assertIn("cycle", info["error"])

    def test_excessive_chain_depth_bounded(self):
        blobs = {}
        for i in range(rc.MAX_UNWIND_CHAIN + 2):
            rva = 0x3000 + i * 16
            nxt = 0x3000 + (i + 1) * 16
            blobs[rva] = unwind_header(1, rc.UNW_FLAG_CHAININFO, 0, 0) + struct.pack("<III", 0x1000, 0x1004, nxt)
        info = self.parse_single(blobs, rva=0x3000)
        self.assertIn("depth exceeds", info["error"])


class CensusUnwindIntegrationTests(unittest.TestCase):
    """End-to-end census() + to_markdown() over a multi-function PE with rich .xdata."""

    def test_census_and_markdown_report_unwind_breakdown(self):
        code = {
            0x1000: bytes.fromhex("c3"),
            0x1010: bytes.fromhex("c3"),
            0x1020: bytes.fromhex("c3"),
            0x1030: bytes.fromhex("c3"),
        }
        # fn1: v1 EHANDLER, FP=rbp, PUSH rbx, ALLOC_SMALL 40B, SAVE_XMM128 xmm6
        u1 = (unwind_header(1, rc.UNW_FLAG_EHANDLER, 0x10, 5, frame_reg=5, frame_off_scaled=1)
              + ucode(0x02, 0, 3) + ucode(0x06, 2, 4) + ucode(0x08, 3, 0)
              + ucode(0x0C, 8, 6) + u16(1) + b"\x00\x00" + u32(0x1800))
        # fn2: v1 CHAINED -> u1 (inherits EHANDLER 0x1800, FP=rbp, xmm6, adds ALLOC_SMALL 16B)
        u2 = (unwind_header(1, rc.UNW_FLAG_CHAININFO, 0x04, 1)
              + ucode(0x04, 2, 1) + b"\x00\x00"
              + struct.pack("<III", 0x1000, 0x1001, 0x3000))
        # fn3: v2 with UWOP_EPILOG + UWOP_PUSH_MACHFRAME
        u3 = unwind_header(2, 0, 0x04, 2) + ucode(0x02, 6, 1) + ucode(0x00, 10, 0)
        # fn4: malformed (version 3)
        u4 = unwind_header(3, 0, 0, 0)

        pdata = [
            (0x1000, 0x1001, 0x3000),
            (0x1010, 0x1011, 0x3040),
            (0x1020, 0x1021, 0x3080),
            (0x1030, 0x1031, 0x30C0),
        ]
        pe_bytes = build_pe_with_xdata(pdata, {0x3000: u1, 0x3040: u2, 0x3080: u3, 0x30C0: u4}, code)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "xdata_fixture.exe")
            with open(path, "wb") as f:
                f.write(pe_bytes)
            report = rc.census(path)

        self.assertEqual(report["unwind_flags"], {"EHANDLER": 1, "CHAINED": 1, "none": 2})
        uw = report["unwind"]
        self.assertEqual(uw["parsed"], 3)
        self.assertEqual(uw["malformed"], 1)
        self.assertEqual(uw["versions"], {"1": 2, "2": 1, "3": 1})
        self.assertEqual(uw["with_frame_pointer"], 2)
        self.assertEqual(uw["with_saved_xmm"], 2)
        self.assertEqual(uw["with_handler"], 2)
        self.assertEqual(uw["unique_handlers"], 1)
        self.assertEqual(uw["with_machframe"], 1)
        self.assertEqual(uw["chained"], 1)
        self.assertEqual(uw["max_chain_depth"], 1)
        self.assertEqual(uw["max_stack_alloc"], 16 + 8 + 40)
        self.assertEqual(uw["saved_gprs"], {"rbx": 2})
        self.assertEqual(uw["saved_xmms"], {"xmm6": 2})
        self.assertIn("unsupported unwind version 3", uw["errors"])

        ops_dict = dict(uw["ops"])
        self.assertEqual(ops_dict["UWOP_ALLOC_SMALL"], 3)
        self.assertEqual(ops_dict["UWOP_PUSH_NONVOL"], 2)
        self.assertEqual(ops_dict["UWOP_SAVE_XMM128"], 2)
        self.assertEqual(ops_dict["UWOP_EPILOG"], 1)
        self.assertEqual(ops_dict["UWOP_PUSH_MACHFRAME"], 1)

        md = rc.to_markdown(report)
        self.assertIn("## Unwind (.xdata) details", md)
        self.assertIn("parsed: 3; malformed: 1; chained: 1 (max depth 1)", md)
        self.assertIn("| UWOP_ALLOC_SMALL | 3 |", md)
        self.assertIn("- CHAINED: 1", md)


if __name__ == "__main__":
    unittest.main()
