"""Unit tests for tools/pe_shader_containers.py, on synthetic containers (no game data, no PE needed).

The containers are built by hand here: a u32 offset table at +32, and each chunk beginning with its fourcc
and a u32 size. The scan's job is to accept exactly that shape and to reject the `DXBC` immediates and error
strings that occur in ordinary code.
"""

import contextlib
import hashlib
import io
import json
import os
import random
import struct
import sys
import tempfile
import unittest
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import pe_shader_containers as psc


def container(parts) -> bytes:
    """Assemble a DXBC container from (fourcc, payload) pairs."""
    header = 32 + 4 * len(parts)
    offsets, body, end = [], b"", header
    for fourcc, payload in parts:
        offsets.append(header + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
        end = header + len(body)
        body += bytes(
            (-len(body)) % 4
        )  # chunks are 4-byte aligned; nothing follows the last one
    total = end
    out = b"DXBC" + bytes(16) + struct.pack("<III", 1, total, len(parts))
    out += b"".join(struct.pack("<I", o) for o in offsets)
    return out + body[: total - header]


def xdat(entries, base: int) -> bytes:
    """An XDAT directory whose payload offsets are relative to the DXIL part, as the package has it."""
    head = base + 12 + 12 * len(entries)
    directory, payload = b"", b""
    for tag, value in entries:
        directory += struct.pack("<III", tag, head + len(payload), len(value))
        payload += value
    return b"XDAT" + struct.pack("<II", len(entries), 0) + directory + payload


def dxil_payload(name: str, xbch: bool = True, bitcode: bool = True) -> bytes:
    entries = [(0x12, bytes(16))]
    if xbch:
        entries.append((0x0D, b"XBCH" + bytes(20) + b"payload"))
    if bitcode:
        entries.append((0x07, b"BC\xc0\xde" + bytes(8)))
    prefix = (
        b"e\x00\x05\x00"
        + struct.pack("<I", 24)
        + b"DXIL"
        + b"\x05\x01\x00\x00"
        + struct.pack("<I", 0x70)
        + struct.pack("<I", 0x668)
    )
    return prefix + xdat(entries, len(prefix))


def ildn(name: str) -> bytes:
    path = (
        "X:\\americas\\src\\dev\\rage\\3rdparty\\Microsoft\\XboxBvhBuilder\\Build\\Release\\Output\\"
        f"XboxBvhBuilder\\{name}.pdb"
    ).encode()
    return b"\x00\x00" + struct.pack("<H", len(path)) + path


def packed(dxbc: bytes, trailing: bytes = b"", stored=None, size=None, level: int = -1) -> bytes:
    """A compressed container as the image stores it: the stored size, the constant 0x000DDCE7, the stored size
    again, the inflated size and a raw deflate stream. Both stored sizes count from the constant."""
    compressor = zlib.compressobj(level, wbits=-15)
    stream = compressor.compress(dxbc) + compressor.flush() + trailing
    first, second = stored or (12 + len(stream),) * 2
    return struct.pack("<IIII", first, 0xDDCE7, second, len(dxbc) if size is None else size) + stream


class PackedContainerTests(unittest.TestCase):
    """A console program stored compressed is found, described by its inflated bytes and extracted inflated."""

    def setUp(self):
        self.native = container([(b"DXIL", dxil_payload("packed")), (b"ILDN", ildn("PackedPS"))])
        self.plain = container([(b"DXIL", dxil_payload("plain")), (b"ILDN", ildn("PlainCS"))])

    def test_packed_program_is_listed_after_the_plain_containers(self):
        record = packed(self.native)
        image = bytes(7) + record + bytes(5) + self.plain
        records, rejected = psc.inspect(image, "synthetic.exe")
        self.assertEqual(rejected, [])
        self.assertEqual([r["name"] for r in records], ["PlainCS", "PackedPS"])
        plain, found = records
        self.assertEqual((plain["offset"], plain["size"]), (7 + len(record) + 5, len(self.plain)))
        self.assertNotIn("stored_size", plain)
        # offset is that of the constant; the sizes before and after it count from there
        self.assertEqual((found["offset"], found["stored_size"]), (7 + 4, len(record) - 4))
        self.assertEqual((found["size"], found["sha256"]), (len(self.native), hashlib.sha256(self.native).hexdigest()))
        self.assertTrue(found["has_xbch"])
        self.assertEqual([c["fourcc"] for c in found["chunks"]], ["DXIL", "ILDN"])

    def test_only_a_whole_console_container_counts(self):
        other = container([(b"SFI0", bytes(8))])                       # a container, but no console program
        good = packed(self.native)
        n = len(good) - 4
        cases = {
            "no console program": packed(other),
            "bytes after the stream": packed(self.native, trailing=b"junk"),
            "cut short": good[:-1],
            "first stored size differs": packed(self.native, stored=(n + 1, n)),
            "second stored size differs": packed(self.native, stored=(n, n - 1)),
            "inflated size too small": packed(self.native, size=len(self.native) - 1),
            "inflated size too large": packed(self.native, size=len(self.native) + 1),
            "stored size of the header alone": struct.pack("<IIII", 12, 0xDDCE7, 12, len(self.native)),
            "not a deflate stream": struct.pack("<IIII", 20, 0xDDCE7, 20, 64) + b"\xff" * 8,
            "not a container": packed(b"x" * 64),
            "two containers": packed(self.native + self.native),
            "container and padding": packed(self.native + bytes(4)),
            "constant at the start of the file": good[4:],
        }
        for name, record in cases.items():
            with self.subTest(name):
                records, _ = psc.inspect(record + self.plain, "synthetic.exe")
                self.assertEqual([r["name"] for r in records], ["PlainCS"])
        records, _ = psc.inspect(good + self.plain, "synthetic.exe")    # the control
        self.assertEqual([r["name"] for r in records], ["PlainCS", "PackedPS"])

    def test_stored_blocks_list_the_container_once(self):
        # Level 0 writes stored blocks, so the container sits in the stream as it is and the plain scan finds it.
        record = packed(self.native, level=0)
        self.assertIn(self.native, record)
        records, _ = psc.inspect(record + self.plain, "synthetic.exe")
        self.assertEqual([(r["name"], "stored_size" in r) for r in records], [("PackedPS", False), ("PlainCS", False)])

    def test_extract_writes_the_inflated_container(self):
        image = packed(self.native) + self.plain + packed(self.native, trailing=b"junk")
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "synthetic.exe")
            with open(src, "wb") as f:
                f.write(image)
            out, report = os.path.join(tmp, "out"), os.path.join(tmp, "report.json")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(psc.main([src, "--extract", out, "--json", report, "--verbose"]), 0)
            self.assertIn("2 container(s)", stdout.getvalue())
            with open(os.path.join(out, "index.json"), encoding="utf-8") as f:
                self.assertEqual([e["file"] for e in json.load(f)], ["PlainCS.dxbc", "PackedPS.dxbc"])
            for file, expected in (("PlainCS.dxbc", self.plain), ("PackedPS.dxbc", self.native)):
                with open(os.path.join(out, file), "rb") as f:
                    self.assertEqual(f.read(), expected)
            with open(report, encoding="utf-8") as f:
                rows = json.load(f)
            first = len(packed(self.native))
            self.assertEqual([(r["offset"], r.get("stored_size"), r["size"]) for r in rows],
                             [(first, None, len(self.plain)), (4, first - 4, len(self.native))])

    def test_random_headers_raise_nothing(self):
        rng = random.Random(0xDDCE7)
        good = packed(self.native)
        for _ in range(2000):
            data = bytearray(rng.randbytes(rng.randrange(0, 300)))
            if rng.random() < 0.5:
                data += good
                for _ in range(rng.randrange(1, 4)):                    # damage a few bytes of a real record
                    data[rng.randrange(len(data) - len(good), len(data))] = rng.randrange(256)
            elif len(data) >= 16:
                at = rng.randrange(0, len(data) - 15)
                n = rng.randrange(0, 64)
                data[at : at + 16] = struct.pack("<IIII", n, 0xDDCE7, n, rng.randrange(0, 200))
            try:
                psc.inspect(bytes(data), "fuzz.exe")
            except Exception as e:  # noqa: BLE001 - any exception is the failure being asserted
                self.fail(f"inspect raised {type(e).__name__}: {e}")


class ContainerTests(unittest.TestCase):
    def test_packed_native_programs_keep_plain_indices_and_extract_inflated_bytes(self):
        def pack(dxbc, trailing=b""):
            compressor = zlib.compressobj(wbits=-15)
            payload = compressor.compress(dxbc) + compressor.flush() + trailing
            stored = 12 + len(payload)
            return struct.pack("<IIII", stored, 0xDDCE7, stored, len(dxbc)) + payload

        native = container([(b"DXIL", dxil_payload("packed"))])
        plain = container([(b"SFI0", bytes(8))])
        packed = pack(native)
        image = packed + plain + pack(plain) + pack(native, b"junk") + packed[:-1]
        records, _ = psc.inspect(image, "synthetic.exe")
        self.assertEqual([r["offset"] for r in records], [len(packed), 4])
        self.assertEqual(records[1]["stored_size"], len(packed) - 4)
        with tempfile.TemporaryDirectory() as tmp:
            entries = psc.extract(image, records, tmp)
            for entry, expected in zip(entries, (plain, native)):
                with open(os.path.join(tmp, entry["file"]), "rb") as f:
                    self.assertEqual(f.read(), expected)

    def test_scan_reads_one_container(self):
        data = container(
            [
                (b"SFI0", bytes(8)),
                (b"ILDN", ildn("BitonicPreSort64CS")),
                (b"DXIL", dxil_payload("BitonicPreSort64CS")),
            ]
        )
        found, rejected = psc.inspect(b"MZ" + bytes(64) + data, "fake.dll")
        self.assertEqual(rejected, [])
        self.assertEqual(len(found), 1)
        r = found[0]
        self.assertEqual(r["offset"], 66)
        self.assertEqual(r["size"], len(data))
        self.assertEqual(r["name"], "BitonicPreSort64CS")
        self.assertTrue(r["has_xbch"])
        self.assertTrue(r["has_bitcode"])
        self.assertFalse(r["has_fspx"])
        self.assertEqual([c["fourcc"] for c in r["chunks"]], ["SFI0", "ILDN", "DXIL"])

    def test_scan_rejects_a_bare_magic(self):
        data = b"junk" + b"DXBC" + bytes(40)
        found, rejected = psc.inspect(data, "fake.dll")
        self.assertEqual(found, [])
        self.assertEqual(rejected, [4])

    def test_scan_rejects_a_truncated_chunk(self):
        good = container([(b"SFI0", bytes(8)), (b"DXIL", dxil_payload("x"))])
        data = good[:-8]
        found, rejected = psc.inspect(data, "fake.dll")
        self.assertEqual(found, [])
        self.assertEqual(len(rejected), 1)

    def test_fspx_is_reported_and_has_no_xbch(self):
        data = container([(b"SFI0", bytes(8)), (b"ACCC", b"FSPX" + bytes(12))])
        found, _ = psc.inspect(data, "fsr.dll")
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["has_fspx"])
        self.assertFalse(found[0]["has_xbch"])
        self.assertFalse(found[0]["has_bitcode"])
        self.assertIsNone(found[0]["name"])

    def test_xdat_directory(self):
        tags = psc.xdat_tags(dxil_payload("x"))
        self.assertEqual([t["tag"] for t in tags], [0x12, 0x0D, 0x07])
        self.assertEqual(tags[1]["magic"], "XBCH")

    def test_module_name_forms(self):
        self.assertEqual(psc.module_name(ildn("ConstructBVH4")), "ConstructBVH4")
        self.assertEqual(psc.module_name(b"\x00\x00\x0a\x00C:\\a\\b.pdb"), "b")
        self.assertIsNone(psc.module_name(b""))

    def test_extract_writes_container_and_index(self):
        data = container(
            [(b"ILDN", ildn("RefitBVH4")), (b"DXIL", dxil_payload("RefitBVH4"))]
        )
        found, _ = psc.inspect(data, "fake.dll")
        with tempfile.TemporaryDirectory() as tmp:
            index = psc.extract(data, found, tmp)
            with open(os.path.join(tmp, index[0]["file"]), "rb") as f:
                self.assertEqual(f.read(), data)
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8") as f:
                json.dump(index, f)
            with open(os.path.join(tmp, "index.json"), encoding="utf-8") as f:
                self.assertEqual(
                    json.load(f)[0],
                    {"name": "RefitBVH4", "file": "RefitBVH4.dxbc", "stage": "cs"},
                )

    def test_extract_retries_existing_suffixes(self):
        data = container(
            [(b"ILDN", ildn("RefitBVH4")), (b"DXIL", dxil_payload("RefitBVH4"))]
        )
        found, _ = psc.inspect(data, "fake.dll")
        with tempfile.TemporaryDirectory() as tmp:
            # Occupy both initial candidates for the first record (index 0).
            with open(os.path.join(tmp, "RefitBVH4.dxbc"), "wb") as f:
                f.write(b"stale")
            with open(os.path.join(tmp, "RefitBVH4_00.dxbc"), "wb") as f:
                f.write(b"stale")
            index = psc.extract(data, found, tmp)
            self.assertEqual(index[0]["file"], "RefitBVH4_01.dxbc")
            with open(os.path.join(tmp, index[0]["file"]), "rb") as f:
                self.assertEqual(f.read(), data)
            # The occupied candidates were untouched.
            with open(os.path.join(tmp, "RefitBVH4.dxbc"), "rb") as f:
                self.assertEqual(f.read(), b"stale")
            with open(os.path.join(tmp, "RefitBVH4_00.dxbc"), "rb") as f:
                self.assertEqual(f.read(), b"stale")

    def test_safe_stem(self):
        self.assertEqual(psc.safe_stem("a b/c", 1), "a_b_c")
        self.assertEqual(psc.safe_stem(None, 7), "container_07")

    def test_random_buffers_raise_only_container_error(self):
        """The scan must never traceback on bytes that are not a container (dcastell1's review)."""
        rng = random.Random(0xDBC)
        for _ in range(2000):
            size = rng.randrange(0, 400)
            data = bytearray(rng.randbytes(size))
            if size >= 4:
                at = rng.randrange(0, size - 3)
                data[at : at + 4] = b"DXBC"
                try:
                    psc.parse_container(bytes(data), at)
                except psc.ContainerError:
                    pass
                except Exception as e:  # noqa: BLE001 - any other type is the failure being asserted
                    self.fail(f"parse_container raised {type(e).__name__}: {e}")
            try:
                psc.inspect(bytes(data), "fuzz.dll")
            except Exception as e:  # noqa: BLE001 - any exception is the failure being asserted
                self.fail(f"inspect raised {type(e).__name__}: {e}")


class ReportPathTests(unittest.TestCase):
    """--json and --extract must not overwrite an input, each other, or a folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name
        self.image = os.path.join(self.dir, "shader.dll")
        with open(self.image, "wb") as f:
            f.write(container([(b"SFI0", bytes(8))]))

    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = psc.main(argv)
            except SystemExit as e:
                return e.code, out.getvalue(), err.getvalue()
        return code, out.getvalue(), err.getvalue()

    def test_json_must_not_overwrite_an_image(self):
        with open(self.image, "rb") as f:
            before = f.read()
        code, _, err = self.run_main([self.image, "--json", self.image])
        self.assertEqual(code, 2)
        self.assertIn("would overwrite the input image", err)
        with open(self.image, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_json_creates_a_missing_folder(self):
        out = os.path.join(self.dir, "reports", "containers.json")
        code, _, err = self.run_main([self.image, "--json", out])
        self.assertEqual(code, 0, err)
        with open(out, encoding="utf-8") as f:
            self.assertIsInstance(json.load(f), list)

    def test_json_refuses_a_folder(self):
        folder = os.path.join(self.dir, "folder")
        os.makedirs(folder)
        code, _, err = self.run_main([self.image, "--json", folder])
        self.assertEqual(code, 2)
        self.assertIn("is a folder, not a file", err)

    def test_extract_refuses_an_existing_file(self):
        blocker = os.path.join(self.dir, "blocked")
        with open(blocker, "wb") as f:
            f.write(b"keep me")
        code, _, err = self.run_main([self.image, "--extract", blocker])
        self.assertEqual(code, 2)
        self.assertIn("is not a folder", err)
        with open(blocker, "rb") as f:
            self.assertEqual(f.read(), b"keep me")

    def test_extract_index_must_not_overwrite_an_image(self):
        with open(self.image, "rb") as f:
            before = f.read()
        index = os.path.join(self.dir, "index.json")
        with open(index, "wb") as f:
            f.write(before)
        code, stdout, err = self.run_main([self.image, index, "--extract", self.dir])
        self.assertEqual(code, 2)
        self.assertIn("would overwrite the input image", err)
        self.assertEqual(stdout, "")
        self.assertCountEqual(os.listdir(self.dir), ["shader.dll", "index.json"])
        with open(index, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_json_must_not_overwrite_the_extract_index(self):
        extract = os.path.join(self.dir, "out")
        index = os.path.join(extract, "index.json")
        code, _, err = self.run_main([self.image, "--extract", extract, "--json", index])
        self.assertEqual(code, 2)
        self.assertIn("would overwrite the extract index", err)
        self.assertFalse(os.path.exists(extract))

    def test_a_failed_json_write_is_one_line(self):
        blocker = os.path.join(self.dir, "not_a_folder")
        with open(blocker, "wb") as f:
            f.write(b"keep me")
        code, _, err = self.run_main([self.image, "--json", os.path.join(blocker, "containers.json")])
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot write", err)
        self.assertNotIn("Traceback", err)
        with open(blocker, "rb") as f:
            self.assertEqual(f.read(), b"keep me")


def psv0(stage_code: int) -> bytes:
    """Build a minimal PSV0 chunk payload whose ShaderStage byte is `stage_code`."""
    return struct.pack("<I", 48) + bytes(24) + bytes([stage_code]) + bytes(23)


class Psv0StageTests(unittest.TestCase):
    def test_psv0_stage_detection(self):
        for code, expected in ((0, "ps"), (1, "vs"), (2, "gs"), (3, "hs"), (4, "ds"), (5, "cs"), (13, "ms"), (14, "as")):
            with self.subTest(code=code):
                self.assertEqual(psc.psv0_stage(psv0(code)), expected)
        self.assertIsNone(psc.psv0_stage(psv0(99)))
        self.assertIsNone(psc.psv0_stage(b""))
        self.assertIsNone(psc.psv0_stage(bytes(20)))
        self.assertIsNone(psc.psv0_stage(struct.pack("<I", 20) + bytes(28)))

    def test_extract_writes_psv0_stage_to_index(self):
        # Match the unnamed XBCH containers embedded in game_scarlett_master.exe (no ILDN, PSV0 present),
        # plus a container without PSV0 to verify the "cs" fallback.
        vs_blob = container([(b"PSV0", psv0(1)), (b"DXIL", dxil_payload("vs", xbch=True, bitcode=False))])
        ps_blob = container([(b"PSV0", psv0(0)), (b"DXIL", dxil_payload("ps", xbch=True, bitcode=False))])
        no_psv0 = container([(b"DXIL", dxil_payload("cs", xbch=True, bitcode=False))])
        image_bytes = b"MZ" + vs_blob + ps_blob + no_psv0
        records, rejected = psc.inspect(image_bytes, "fake.exe")
        self.assertEqual(rejected, [])
        self.assertEqual([r["stage"] for r in records], ["vs", "ps", None])
        with tempfile.TemporaryDirectory() as tmp:
            index = psc.extract(image_bytes, records, tmp)
            with open(os.path.join(tmp, "index.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(index, f, indent=0)
            with open(os.path.join(tmp, "index.json"), encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(
                loaded,
                [
                    {"name": "container_00", "file": "container_00.dxbc", "stage": "vs"},
                    {"name": "container_01", "file": "container_01.dxbc", "stage": "ps"},
                    {"name": "container_02", "file": "container_02.dxbc", "stage": "cs"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
