"""tools/picture_metrics.py on synthetic frames: every pixel is drawn here, no game data.

    python -I tests/test_picture_metrics.py

The frames are built in memory and written as PNG by the small encoder below (all five filter types, RGB and RGBA,
8 bits, not interlaced); each test checks that the tool gives back the parameters it was drawn with.
"""
import io
import json
import os
import random
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import picture_metrics as pm  # noqa: E402


# ---- a small PNG encoder ---------------------------------------------------------------------------------------

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _filter_row(kind, row, prev, bpp):
    out = bytearray([kind])
    for i, x in enumerate(row):
        a = row[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        pred = (0, a, b, (a + b) // 2, _paeth(a, b, c))[kind]
        out.append((x - pred) & 0xFF)
    return out


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_bytes(width, height, pixels, channels=3, filters=(0,), bit_depth=8, color_type=None, interlace=0):
    """A PNG of row-major 8-bit ``pixels`` (``channels`` bytes each); row y uses filter ``filters[y % len]``."""
    if color_type is None:
        color_type = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    stride = width * channels
    prev = bytearray(stride)
    raw = bytearray()
    for y in range(height):
        row = bytearray(pixels[y * stride:(y + 1) * stride])
        raw += _filter_row(filters[y % len(filters)], row, prev, channels)
        prev = row
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(bytes(raw)))
            + _chunk(b"IEND", b""))


# ---- synthetic frames ------------------------------------------------------------------------------------------

def gray_frame(width, height, value):
    """A grey RGB frame as a list of luma values, which ``rgb()`` turns into pixel bytes."""
    return [value] * (width * height)


def rgb(lumas):
    """Grey pixels: R = G = B, so the luma of each is the value itself."""
    out = bytearray()
    for v in lumas:
        out += bytes((v, v, v))
    return bytes(out)


def lattice_frame(width, height, period, phase, diagonal, corner, tile=8, back=128, dark=0):
    """Small dark triangles in the tiles on one diagonal in ``period``: ``(tx +/- ty) % period == phase``."""
    lumas = gray_frame(width, height, back)
    for ty in range(height // tile):
        for tx in range(width // tile):
            d = (tx + ty) if diagonal == "+" else (tx - ty)
            if d % period != phase:
                continue
            for ly in range(tile):
                for lx in range(tile):
                    u = lx if corner in ("TL", "BL") else tile - 1 - lx
                    v = ly if corner in ("TL", "TR") else tile - 1 - ly
                    if u + v < tile // 2:
                        lumas[(ty * tile + ly) * width + tx * tile + lx] = dark
    return lumas


def smooth_frame(width, height):
    """A slow grey ramp, 64..127, with no step of more than one level and no dark pixel."""
    return [64 + (x + y) // 4 for y in range(height) for x in range(width)]


def blocks_frame(width, height, bw, bh, foreign, other=230):
    """``smooth_frame`` with the ``bw`` x ``bh`` blocks at ``foreign`` (block column, block row) copied from a flat
    frame of ``other``."""
    lumas = smooth_frame(width, height)
    for bx, by in foreign:
        for y in range(by * bh, (by + 1) * bh):
            for x in range(bx * bw, (bx + 1) * bw):
                lumas[y * width + x] = other
    return lumas


def sparse_dark_frame(width, height, count, seed, back=128):
    """A plain grey frame with ``count`` dark pixels at places drawn from ``seed``."""
    rng = random.Random(seed)
    lumas = gray_frame(width, height, back)
    for _ in range(count):
        lumas[rng.randrange(height) * width + rng.randrange(width)] = 0
    return lumas


def frame_of(lumas, width, height):
    return pm.Frame(width, height, rgb(lumas))


class PngReader(unittest.TestCase):
    def test_every_filter_rgb_and_rgba(self):
        w, h = 7, 6
        rgb_px = bytes((x * 37 + y * 11 + c * 90) & 0xFF for y in range(h) for x in range(w) for c in range(3))
        rgba_px = bytes((x * 53 + y * 7 + c * 61) & 0xFF for y in range(h) for x in range(w) for c in range(4))
        for kind in range(5):
            with self.subTest(filter=kind):
                f = pm.decode_png(png_bytes(w, h, rgb_px, 3, (kind,)))
                self.assertEqual((f.width, f.height, f.rgb), (w, h, rgb_px))
                f = pm.decode_png(png_bytes(w, h, rgba_px, 4, (kind,)))
                self.assertEqual(f.rgb, bytes(b for i, b in enumerate(rgba_px) if i % 4 != 3))
        mixed = pm.decode_png(png_bytes(w, h, rgb_px, 3, (0, 1, 2, 3, 4)))
        self.assertEqual(mixed.rgb, rgb_px)

    def test_grey_and_split_idat(self):
        w, h = 5, 3
        grey = bytes(range(w * h))
        self.assertEqual(pm.decode_png(png_bytes(w, h, grey, 1, (4,))).rgb, rgb(grey))
        data = png_bytes(w, h, rgb(grey), 3, (2,))
        start = data.index(b"IDAT") - 4
        length = struct.unpack(">I", data[start:start + 4])[0]
        body = data[start + 8:start + 8 + length]
        split = _chunk(b"IDAT", body[:5]) + _chunk(b"IDAT", body[5:])
        data = data[:start] + split + data[start + 12 + length:]
        self.assertEqual(pm.decode_png(data).rgb, rgb(grey))

    def test_refuses_what_it_does_not_read(self):
        px = bytes(12)
        cases = {
            "interlaced": png_bytes(2, 2, px, 3, interlace=1),
            "16-bit": png_bytes(2, 2, px, 3, bit_depth=16),
            "palette": png_bytes(2, 2, bytes(4), 1, color_type=3),
            "not a PNG": b"GIF89a" + bytes(40),
            "truncated": png_bytes(2, 2, px, 3)[:30],
        }
        bad_crc = bytearray(png_bytes(2, 2, px, 3))
        bad_crc[29] ^= 0xFF                                               # last byte of the IHDR CRC
        cases["bad CRC"] = bytes(bad_crc)
        bad_filter = bytearray(png_bytes(2, 2, px, 3))
        cases["filter 5"] = bytes(bad_filter[:33]) + _chunk(
            b"IDAT", zlib.compress(b"\x05" + bytes(6) + b"\x00" + bytes(6))) + _chunk(b"IEND", b"")
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                pm.decode_png(data)

    def test_ppm(self):
        px = bytes(range(18))
        f = pm.decode_image(b"P6\n# a comment\n3 2\n255\n" + px)
        self.assertEqual((f.width, f.height, f.rgb), (3, 2, px))
        f = pm.decode_image(b"P5 3 2 255\n" + bytes(range(6)))
        self.assertEqual(f.rgb, rgb(range(6)))
        with self.assertRaises(ValueError):
            pm.decode_image(b"P6 3 2 65535\n" + bytes(36))
        with self.assertRaises(ValueError):
            pm.decode_image(b"P6 3 2 255\n" + bytes(5))


class Luma(unittest.TestCase):
    def test_weights(self):
        f = pm.Frame(3, 1, bytes((255, 0, 0, 0, 255, 0, 0, 0, 255)))
        self.assertEqual(list(pm.luma(f)), [76, 150, 29])

    def test_stats_of_a_known_frame(self):
        lumas = [0, 200] * 32                                             # 8x8, half 0 and half 200
        s = pm.luma_stats(pm.luma(frame_of(lumas, 8, 8)))
        self.assertEqual(s, {"mean": 100.0, "min": 0, "max": 200, "sd": 100.0, "dark": 0.5})
        s = pm.luma_stats(pm.luma(frame_of([10, 20, 30, 40], 2, 2)))
        self.assertEqual((s["mean"], s["min"], s["max"], s["sd"], s["dark"]), (25.0, 10, 40, 11.18, 0.5))

    def test_dark_share_per_region(self):
        w = h = 64
        lumas = gray_frame(w, h, 128)
        for y in range(h):
            for x in range(w):
                if (x < 16 and y < 16) or (x >= 56 and y >= 48):        # region (0,0) all, region (3,3) half
                    lumas[y * w + x] = 3
        grid = pm.dark_regions(pm.luma(frame_of(lumas, w, h)), w, h)
        expected = [[0.0] * 4 for _ in range(4)]
        expected[0][0], expected[3][3] = 1.0, 0.5
        self.assertEqual(grid, expected)
        uneven = pm.dark_regions(bytes(10 * 6), 10, 6)                   # regions of 2 or 3 pixels a side
        self.assertEqual(uneven, [[1.0] * 4 for _ in range(4)])


class Lattice(unittest.TestCase):
    def test_recovers_tile_period_phase(self):
        w = h = 256
        for period, phase, diagonal, corner in ((16, 5, "+", "TL"), (16, 11, "-", "BR"), (16, 0, "+", "BL"),
                                                (8, 3, "+", "TR")):
            with self.subTest(period=period, phase=phase, diagonal=diagonal, corner=corner):
                lumas = lattice_frame(w, h, period, phase, diagonal, corner)
                found = pm.find_lattice(pm.luma(frame_of(lumas, w, h)), w, h)
                self.assertIsNotNone(found)
                self.assertEqual((found["tile"], found["period"], found["phase"], found["diagonal"],
                                  found["corner"]), (8, period, phase, diagonal, corner))
                self.assertEqual(found["cover"], 1.0)
                marked = sum(1 for ty in range(32) for tx in range(32)
                             if ((tx + ty) if diagonal == "+" else (tx - ty)) % period == phase)
                self.assertEqual(found["tiles"], marked)

    def test_720p_frame(self):
        w, h = 1280, 720
        lumas = lattice_frame(w, h, 16, 9, "+", "TL")
        found = pm.find_lattice(pm.luma(frame_of(lumas, w, h)), w, h)
        self.assertEqual((found["tile"], found["period"], found["phase"]), (8, 16, 9))

    def test_none_on_plain_frames(self):
        for lumas, w, h in ((gray_frame(64, 64, 128), 64, 64), (smooth_frame(128, 128), 128, 128),
                            (gray_frame(64, 64, 4), 64, 64)):
            self.assertIsNone(pm.find_lattice(pm.luma(frame_of(lumas, w, h)), w, h))

    def test_none_on_sparse_dark_pixels(self):
        # A few dark pixels anywhere must not read as a lattice: a corner diagonal of one or two tiles covers
        # fully at a large period, which is no repetition at all.
        cases = [(256, 256, n, seed) for n in (8, 15, 50) for seed in range(10)]
        cases += [(320, 180, n, seed) for n in (10, 20, 50) for seed in range(10)]
        cases += [(640, 360, n, seed) for n in (20, 50) for seed in range(5)]
        cases += [(1280, 720, n, seed) for n in (4, 50) for seed in range(2)]
        for w, h, n, seed in cases:
            with self.subTest(size=(w, h), pixels=n, seed=seed):
                self.assertIsNone(pm.find_lattice(pm.luma(frame_of(sparse_dark_frame(w, h, n, seed), w, h)), w, h))

    def test_none_on_noise_and_rows(self):
        rng = random.Random(7)
        noise = [min(255, max(0, round(rng.gauss(128, 30)))) for _ in range(256 * 256)]
        self.assertIsNone(pm.find_lattice(pm.luma(frame_of(noise, 256, 256)), 256, 256))
        rows = gray_frame(256, 256, 128)                                 # marks on every tile of rows 5 and 21
        for y in range(256):
            if (y // 8) % 16 == 5 and y % 8 < 4:
                for x in range(256):
                    if x % 8 + y % 8 < 4:
                        rows[y * 256 + x] = 0
        self.assertIsNone(pm.find_lattice(pm.luma(frame_of(rows, 256, 256)), 256, 256))

    def test_recovers_through_noise(self):
        rng = random.Random(3)
        lumas = [min(255, max(0, v + rng.randint(-20, 20))) for v in lattice_frame(256, 256, 16, 7, "-", "TR")]
        found = pm.find_lattice(pm.luma(frame_of(lumas, 256, 256)), 256, 256)
        self.assertEqual((found["tile"], found["period"], found["phase"], found["diagonal"]), (8, 16, 7, "-"))


class Blocks(unittest.TestCase):
    FOREIGN = ((1, 1), (2, 1), (5, 3), (3, 7), (6, 10), (1, 13), (4, 12), (4, 13))

    def test_recovers_block_size_and_share(self):
        w = h = 128
        for bw, bh in ((16, 8), (16, 16), (32, 8)):
            foreign = [(bx, by) for bx, by in self.FOREIGN if bx < w // bw and by < h // bh]
            with self.subTest(block=(bw, bh)):
                lumas = blocks_frame(w, h, bw, bh, foreign)
                found = pm.find_blocks(pm.luma(frame_of(lumas, w, h)), w, h)
                self.assertIsNotNone(found)
                total = (w // bw) * (h // bh)
                self.assertEqual((found["width"], found["height"]), (bw, bh))
                self.assertEqual((found["blocks"], found["of"]), (len(foreign), total))
                self.assertEqual(found["share"], round(len(foreign) / total, 4))

    def test_none_on_plain_frames(self):
        for lumas in (gray_frame(128, 128, 128), smooth_frame(128, 128)):
            self.assertIsNone(pm.find_blocks(pm.luma(frame_of(lumas, 128, 128)), 128, 128))

    def test_none_on_a_checkerboard_or_one_rectangle(self):
        # A checkerboard texture has no picture around its squares; one sharp rectangle (a HUD panel) gives two
        # edges a side, too few to tell a grid from the rectangle's own size.
        board = [40 if (x // 8 + y // 8) % 2 else 200 for y in range(128) for x in range(128)]
        self.assertIsNone(pm.find_blocks(pm.luma(frame_of(board, 128, 128)), 128, 128))
        panel = smooth_frame(256, 256)
        for y in range(100, 132):
            for x in range(90, 150):
                panel[y * 256 + x] = 230
        self.assertIsNone(pm.find_blocks(pm.luma(frame_of(panel, 256, 256)), 256, 256))


class Diff(unittest.TestCase):
    def test_changed_tiles(self):
        w, h = 64, 48
        a = rgb(smooth_frame(w, h))
        b = bytearray(a)
        for x, y in ((3 * 8 + 7, 1 * 8), (5 * 8, 5 * 8 + 7), (5 * 8 + 1, 5 * 8 + 2), (0, 0)):
            b[(y * w + x) * 3 + 1] ^= 0x40                                # one green byte: luma moves too
        d = pm.tile_diff(pm.Frame(w, h, a), pm.Frame(w, h, bytes(b)))
        self.assertEqual((d["changed"], d["of"]), (3, 48))
        self.assertEqual(d["tiles"], [[0, 0], [3, 1], [5, 5]])
        self.assertEqual(d["box"], [0, 0, 5, 5])
        self.assertEqual(d["share"], round(3 / 48, 4))
        same = pm.tile_diff(pm.Frame(w, h, a), pm.Frame(w, h, a))
        self.assertEqual((same["changed"], same["tiles"], same["box"]), (0, [], None))

    def test_tolerance_is_off_by_default(self):
        a = rgb(gray_frame(16, 8, 100))
        b = rgb([101] + [100] * 127)
        self.assertEqual(pm.tile_diff(pm.Frame(16, 8, a), pm.Frame(16, 8, b))["changed"], 1)
        self.assertEqual(pm.tile_diff(pm.Frame(16, 8, a), pm.Frame(16, 8, b), tolerance=1)["changed"], 0)

    def test_sizes_must_match(self):
        with self.assertRaises(ValueError):
            pm.tile_diff(pm.Frame(8, 8, bytes(192)), pm.Frame(16, 8, bytes(384)))


class Classes(unittest.TestCase):
    def measure(self, lumas, w, h, ref=None):
        ref_frame = frame_of(ref, w, h) if ref is not None else None
        return pm.measure(frame_of(lumas, w, h), ref_frame)

    def test_measured_classes(self):
        self.assertEqual(self.measure(gray_frame(64, 64, 128), 64, 64)["class"]["measured"], "OK")
        self.assertEqual(self.measure(smooth_frame(64, 64), 64, 64)["class"]["measured"], "OK")
        self.assertEqual(self.measure(gray_frame(64, 64, 6), 64, 64)["class"]["measured"], "BLACK-NEAR")
        self.assertEqual(self.measure(lattice_frame(256, 256, 16, 5, "+", "TL"), 256, 256)["class"]["measured"],
                         "LATTICE")
        self.assertEqual(self.measure(blocks_frame(128, 128, 16, 8, Blocks.FOREIGN), 128, 128)["class"]["measured"],
                         "BLOCKS")
        flash = self.measure(gray_frame(64, 64, 230), 64, 64, ref=gray_frame(64, 64, 40))
        self.assertEqual(flash["class"]["measured"], "FLASH")
        steady = self.measure(gray_frame(64, 64, 130), 64, 64, ref=gray_frame(64, 64, 128))
        self.assertEqual(steady["class"]["measured"], "OK")

    def test_scene_classes_are_only_given(self):
        self.assertEqual(set(pm.CLASSES), {"LATTICE", "BLACK-NEAR", "BLOCKS", "FLASH", "CHAR-BLACK",
                                           "CHAR-MISSING", "2D-MISSING", "STRAY-GEO", "SMEAR", "OK"})
        self.assertEqual(set(pm.GIVEN_ONLY), {"CHAR-BLACK", "CHAR-MISSING", "2D-MISSING", "STRAY-GEO", "SMEAR"})
        r = pm.measure(frame_of(gray_frame(64, 64, 128), 64, 64), None, given="CHAR-BLACK")
        self.assertEqual(r["class"], {"measured": "OK", "given": "CHAR-BLACK"})


class CommandLine(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = pm.main(argv)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def write(self, tmp, name, data):
        path = os.path.join(tmp, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def test_one_frame_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, "secret_name_frame.png",
                              png_bytes(256, 256, rgb(lattice_frame(256, 256, 16, 5, "+", "TL")), 3, (4,)))
            code, out, err = self.run_main([path, "--region", "40,0,8,8"])
        self.assertEqual(code, 0, err)
        lines = out.splitlines()
        self.assertTrue(lines[0].startswith("PICTURE | "))
        self.assertIn("AI-assisted", lines[0])
        fields = [line.split(":", 1)[0].strip() for line in lines[1:]]
        self.assertEqual(fields, ["FRAME", "LUMA", "DARK 4x4", "LATTICE", "BLOCKS", "DIFF", "REGION", "CLASS",
                                  "EVIDENCE"])
        self.assertIn("tile 8 | period 16 | phase 5 | diagonal + | corner TL", out)
        self.assertIn("CLASS:    LATTICE (measured) | given: none", out)
        self.assertIn("REGION:   8x8 | mean 108.0", out)
        self.assertNotIn("secret_name_frame", out)
        self.assertNotIn(tmp, out)

    def test_two_frames_json_and_given_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.write(tmp, "a.png", png_bytes(64, 64, rgb(gray_frame(64, 64, 230)), 3))
            b = self.write(tmp, "b.ppm", b"P6 64 64 255\n" + rgb(gray_frame(64, 64, 40)))
            code, out, err = self.run_main([a, b, "--json", "--class", "SMEAR"])
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertEqual(data["class"], {"measured": "FLASH", "given": "SMEAR"})
        self.assertEqual((data["diff"]["changed"], data["diff"]["of"]), (64, 64))
        self.assertEqual(data["reference"]["luma"]["mean"], 40.0)
        self.assertNotIn(tmp, out)
        self.assertNotIn("a.png", out)

    def test_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = self.write(tmp, "bad.png", b"not an image")
            code, out, err = self.run_main([bad])
            self.assertEqual((code, out), (2, ""))
            self.assertNotIn(tmp, err)
            small = self.write(tmp, "s.png", png_bytes(8, 8, bytes(192), 3))
            big = self.write(tmp, "b.png", png_bytes(16, 8, bytes(384), 3))
            self.assertEqual(self.run_main([small, big])[0], 2)
            self.assertEqual(self.run_main([small, "--class", "PURPLE"])[0], 2)
            self.assertEqual(self.run_main([small, "--region", "4,4,8,8"])[0], 2)
            self.assertEqual(self.run_main([os.path.join(tmp, "missing.png")])[0], 2)


if __name__ == "__main__":
    unittest.main()
