"""Numbers about a frame on your disk, for the PICTURE block of docs/agent_collaboration.md.

    python -I tools/picture_metrics.py <frame.png|.ppm> [<reference.png|.ppm>] [--json] [--class CLASS]
                                       [--region X,Y,W,H] [--tolerance N]

From one frame the tool prints its size, luma statistics, the dark share of each of a fixed 4 x 4 grid of regions,
a lattice detector (dark marks in square tiles along one diagonal in N: tile, period, phase, diagonal, corner of
the tile), a block-grid detector (rectangular blocks cut off from the picture around them: block size and share)
and a measured defect class. With a second frame, the reference, it adds the per-8x8 diff of the two (count,
share, bounding box and position of the changed tiles) and the reference's luma, and can measure FLASH.

Luma is BT.601 in integers, ``(299 R + 587 G + 114 B + 500) // 1000``; a pixel is dark at luma 24 or less.

Measured classes, first match: ``BLACK-NEAR`` (95% or more of the frame dark), ``FLASH`` (the two luma means
differ by 64 or more), ``LATTICE``, ``BLOCKS``, else ``OK``. ``CHAR-BLACK``, ``CHAR-MISSING``, ``2D-MISSING``,
``STRAY-GEO`` and ``SMEAR`` need to know what the scene should show, so the tool has no detector for them: a human
gives one with ``--class`` and it is printed as ``given``, next to the measured class, never in its place.
``--region`` adds the luma statistics of a rectangle the human picked (a character, a HUD element) so a given
class comes with numbers. ``--tolerance`` lets a diff ignore channel changes up to N; it is 0 (exact) by default.

Reads PNG (8-bit grey, grey + alpha, RGB or RGBA, not interlaced, all five filters; alpha is ignored) and binary
PPM/PGM (P6/P5, maxval 255). Anything else (palette, 16-bit, interlaced PNG, other formats) is refused with one
``error:`` line and exit 2.

What it does not do: it does not look at what the scene contains, it does not write any file, and its output holds
numbers and classes only: no pixel values beyond statistics, no image path (the path stays a local argument).
The detectors are tuned on synthetic frames; on a real frame a scene edge on the grid or a dark texture on a
diagonal can hide or fake them, so a class is a hint for a human to confirm. A lattice needs 8 marked tiles on 3
half-marked diagonals of its class, so a lattice cropped to fewer repeats is not reported. BLOCKS refuses one
sharp rectangle and a checkerboard, but several sharp rectangles (HUD panels, window frames) can still read as
BLOCKS, and blocks off the frame's origin are given at the divisor of their edges (16-pixel blocks shifted by 4
read as 4x4): see ``find_blocks``.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from typing import NamedTuple, Optional

DARK = 24                     # a pixel at this luma or below is dark
BLACK_NEAR_SHARE = 0.95       # dark share of the frame for BLACK-NEAR
FLASH_DELTA = 64              # luma mean difference between the two frames for FLASH
REGIONS = 4                   # the dark-share grid is REGIONS x REGIONS
DIFF_TILE = 8
LATTICE_TILES = (4, 8, 16, 32)
LATTICE_MAX_PERIOD = 32
LATTICE_MIN_SCORE = 0.5       # cover of the diagonal minus the rate of marks off it
LATTICE_MIN_TILES = 8         # marked tiles in the winning class, not in the whole frame
LATTICE_MAX_OUTSIDE = 0.25   # the rate of marks off the diagonal stays under this share of its cover
LATTICE_MIN_REPEATS = 3      # half-marked diagonals in the winning class: the pattern repeats at least this often
NEAR_BEST = 0.9               # the smallest tile and period within this share of the best score win
EDGE_JUMP = 8                 # a step across a block edge: at least this many luma levels ...
EDGE_RATIO = 3                # ... and this many times the steps beside it
MIN_LINE_ROWS = 4             # pixels on one grid line that must step for it to count
MIN_BLOCK = 4
MIN_EDGE_LINES = 3            # edge columns, and edge rows, a block grid needs
MAX_BLOCK_SHARE = 0.5         # cut-off blocks are a minority: past this there is no picture around them
TEXT_TILES = 16               # changed tiles listed in the text block (all of them in --json)

MEASURED = ("BLACK-NEAR", "FLASH", "LATTICE", "BLOCKS", "OK")
GIVEN_ONLY = ("CHAR-BLACK", "CHAR-MISSING", "2D-MISSING", "STRAY-GEO", "SMEAR")
CLASSES = ("LATTICE", "BLACK-NEAR", "BLOCKS", "FLASH") + GIVEN_ONLY + ("OK",)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class Frame(NamedTuple):
    """A decoded frame: row-major RGB, 3 bytes a pixel."""
    width: int
    height: int
    rgb: bytes


# ---- reading ---------------------------------------------------------------------------------------------------

def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, height: int, stride: int, bpp: int) -> bytearray:
    if len(raw) < height * (stride + 1):
        raise ValueError("PNG image data is shorter than the image")
    out = bytearray(height * stride)
    prev = bytearray(stride)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        row = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if kind == 1:
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif kind == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif kind == 3:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif kind == 4:
            for i in range(stride):
                if i >= bpp:
                    row[i] = (row[i] + _paeth(row[i - bpp], prev[i], prev[i - bpp])) & 0xFF
                else:
                    row[i] = (row[i] + prev[i]) & 0xFF
        elif kind != 0:
            raise ValueError(f"PNG filter type {kind} is not one of the five")
        out[y * stride:(y + 1) * stride] = row
        prev = row
    return out


def decode_png(data: bytes) -> Frame:
    """An 8-bit, non-interlaced grey, grey + alpha, RGB or RGBA PNG as a Frame; ValueError for anything else."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    pos, header, idat = len(PNG_SIGNATURE), None, []
    while True:
        if pos + 8 > len(data):
            raise ValueError("PNG is truncated")
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length or pos + 12 + length > len(data):
            raise ValueError("PNG is truncated")
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise ValueError(f"PNG chunk {kind.decode('latin-1')!r} has a bad CRC")
        pos += 12 + length
        if kind == b"IHDR":
            if length != 13:
                raise ValueError("PNG IHDR is not 13 bytes")
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        raise ValueError("PNG has no IHDR")
    width, height, depth, color, _compression, _filter, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if channels is None:
        raise ValueError(f"PNG colour type {color} is not read (grey, grey + alpha, RGB or RGBA only)")
    if depth != 8:
        raise ValueError(f"PNG bit depth {depth} is not read (8 only)")
    if interlace:
        raise ValueError("interlaced PNG is not read")
    if not width or not height:
        raise ValueError("PNG has no pixels")
    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as e:
        raise ValueError(f"PNG image data does not inflate: {e}") from None
    px = _unfilter(raw, height, width * channels, channels)
    return Frame(width, height, _to_rgb(px, channels))


def _to_rgb(px: bytes, channels: int) -> bytes:
    if channels == 3:
        return bytes(px)
    out = bytearray(len(px) // channels * 3)
    if channels in (1, 2):
        grey = px[0::channels]
        out[0::3] = out[1::3] = out[2::3] = grey
    else:
        for c in range(3):
            out[c::3] = px[c::4]
    return bytes(out)


def decode_ppm(data: bytes) -> Frame:
    """A binary PPM (P6) or PGM (P5) with maxval 255 as a Frame; ValueError for anything else."""
    fields, pos = [], 2
    while len(fields) < 3:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while pos < len(data) and data[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        start = pos
        while pos < len(data) and data[pos:pos + 1].isdigit():
            pos += 1
        if start == pos:
            raise ValueError("PPM header is not three numbers")
        fields.append(int(data[start:pos]))
    width, height, maxval = fields
    if maxval != 255:
        raise ValueError(f"PPM maxval {maxval} is not read (255 only)")
    if not width or not height:
        raise ValueError("PPM has no pixels")
    channels = 3 if data[:2] == b"P6" else 1
    px = data[pos + 1:pos + 1 + width * height * channels]
    if len(px) != width * height * channels:
        raise ValueError("PPM is truncated")
    return Frame(width, height, _to_rgb(px, channels))


def decode_image(data: bytes) -> Frame:
    """A PNG or a binary PPM/PGM, told apart by their first bytes."""
    if data.startswith(PNG_SIGNATURE):
        return decode_png(data)
    if data[:2] in (b"P6", b"P5"):
        return decode_ppm(data)
    raise ValueError("neither a PNG nor a binary PPM/PGM")


# ---- one frame -------------------------------------------------------------------------------------------------

def luma(frame: Frame) -> bytes:
    """BT.601 integer luma of every pixel, row-major."""
    rgb = frame.rgb
    return bytes((299 * r + 587 * g + 114 * b + 500) // 1000 for r, g, b in zip(rgb[0::3], rgb[1::3], rgb[2::3]))


def luma_stats(lumas: bytes) -> dict:
    """Mean, minimum, maximum, population standard deviation and dark share of a list of luma values."""
    n = len(lumas)
    total = sum(lumas)
    squares = sum(v * v for v in lumas)
    mean = total / n
    sd = math.sqrt(max(squares / n - mean * mean, 0.0))
    dark = sum(1 for v in lumas if v <= DARK)
    return {"mean": round(mean, 2), "min": min(lumas), "max": max(lumas), "sd": round(sd, 2),
            "dark": round(dark / n, 4)}


def dark_regions(lumas: bytes, width: int, height: int) -> list[list[float]]:
    """Dark share of each region of a REGIONS x REGIONS grid, rows top to bottom; region edges at ``i * size // 4``."""
    xs = [i * width // REGIONS for i in range(REGIONS + 1)]
    ys = [i * height // REGIONS for i in range(REGIONS + 1)]
    grid = []
    for r in range(REGIONS):
        row = []
        for c in range(REGIONS):
            dark = sum(1 for y in range(ys[r], ys[r + 1]) for v in lumas[y * width + xs[c]:y * width + xs[c + 1]]
                       if v <= DARK)
            area = (xs[c + 1] - xs[c]) * (ys[r + 1] - ys[r])
            row.append(round(dark / area, 4) if area else 0.0)
        grid.append(row)
    return grid


def region_stats(lumas: bytes, width: int, rect: tuple[int, int, int, int]) -> dict:
    x, y, w, h = rect
    values = b"".join(lumas[(y + j) * width + x:(y + j) * width + x + w] for j in range(h))
    return {"width": w, "height": h, **luma_stats(values)}


def _diagonal_sizes(nx: int, ny: int) -> list[int]:
    """Number of tiles on each '+' diagonal ``tx + ty = s``, s = 0 .. nx + ny - 2."""
    return [min(s, nx - 1, ny - 1, nx + ny - 2 - s) + 1 for s in range(nx + ny - 1)]


def find_lattice(lumas: bytes, width: int, height: int) -> Optional[dict]:
    """Dark marks in square tiles along one diagonal in N, or None.

    A tile is marked when it holds dark pixels but fewer than half. For each tile size, diagonal direction
    (``+``: ``tx + ty``, ``-``: ``tx - ty``) and period P, the tiles fall into P classes by diagonal index mod P;
    the score of the best class is its cover (marked share) minus the rate of marked tiles outside it. The
    smallest tile, then the smallest period, scoring within NEAR_BEST of the best wins: a lattice of period 16 also
    covers fully at 32, and one in 8-pixel tiles at 16-pixel tiles for some phases.
    """
    nx4, ny4 = width // 4, height // 4
    if nx4 < 2 or ny4 < 2:
        return None
    counts = [0] * (nx4 * ny4)
    for y in range(ny4 * 4):
        base = (y // 4) * nx4
        row = lumas[y * width:y * width + nx4 * 4]
        for x, v in enumerate(row):
            if v <= DARK:
                counts[base + x // 4] += 1
    grids = {4: (nx4, ny4, counts)}
    for tile in LATTICE_TILES[1:]:
        cnx, cny, prev = grids[tile // 2]
        nx, ny = cnx // 2, cny // 2
        grids[tile] = (nx, ny, [prev[(2 * ty) * cnx + 2 * tx] + prev[(2 * ty) * cnx + 2 * tx + 1]
                                + prev[(2 * ty + 1) * cnx + 2 * tx] + prev[(2 * ty + 1) * cnx + 2 * tx + 1]
                                for ty in range(ny) for tx in range(nx)])
    candidates = []
    for tile in LATTICE_TILES:
        nx, ny, grid = grids[tile]
        if nx < 2 or ny < 2:
            continue
        marked = [(tx, ty) for ty in range(ny) for tx in range(nx) if 0 < grid[ty * nx + tx] < tile * tile / 2]
        if len(marked) < LATTICE_MIN_TILES:
            continue
        sizes = _diagonal_sizes(nx, ny)
        total = nx * ny
        for diagonal in "+-":
            offset = 0 if diagonal == "+" else ny - 1     # tx - ty + ny - 1 indexes the same sizes
            hist = [0] * len(sizes)
            for tx, ty in marked:
                hist[(tx + ty) if diagonal == "+" else (tx - ty + offset)] += 1
            for period in range(2, min(LATTICE_MAX_PERIOD, len(sizes)) + 1):
                n_c, m_c, repeats = [0] * period, [0] * period, [0] * period
                for s, size in enumerate(sizes):
                    c = (s - offset) % period
                    n_c[c] += size
                    m_c[c] += hist[s]
                    if hist[s] * 2 >= size:               # this diagonal is a repeat of the pattern
                        repeats[c] += 1
                # A class counts only with LATTICE_MIN_TILES marked tiles on LATTICE_MIN_REPEATS diagonals that
                # are each half marked or more: one or two corner diagonals cover fully at a large period by
                # chance, which is no repetition.
                supported = [c for c in range(period) if n_c[c] and m_c[c] >= LATTICE_MIN_TILES
                             and repeats[c] >= LATTICE_MIN_REPEATS]
                if not supported:
                    continue
                best = max(supported, key=lambda c: (m_c[c] / n_c[c], -c))
                if n_c[best] == total:
                    continue
                cover = m_c[best] / n_c[best]
                outside = (len(marked) - m_c[best]) / (total - n_c[best])
                if outside > LATTICE_MAX_OUTSIDE * cover:     # marks everywhere: dark texture, not a lattice
                    continue
                candidates.append((cover - outside, tile, period, diagonal, best, m_c[best], cover))
    if not candidates:
        return None
    top = max(c[0] for c in candidates)
    if top < LATTICE_MIN_SCORE:
        return None
    score, tile, period, diagonal, phase, tiles, cover = min(
        (c for c in candidates if c[0] >= NEAR_BEST * top), key=lambda c: (c[1], c[2], c[3] != "+"))
    return {"tile": tile, "period": period, "phase": phase, "diagonal": diagonal,
            "corner": _corner(lumas, width, height, tile, period, diagonal, phase),
            "tiles": tiles, "cover": round(cover, 4), "score": round(score, 4)}


def _corner(lumas: bytes, width: int, height: int, tile: int, period: int, diagonal: str, phase: int) -> str:
    """Which corner of its tile the dark pixels of the lattice sit in: TL, TR, BL, BR, or C when centred."""
    sx = sy = n = 0
    for ty in range(height // tile):
        for tx in range(width // tile):
            if ((tx + ty) if diagonal == "+" else (tx - ty)) % period != phase:
                continue
            for ly in range(tile):
                start = (ty * tile + ly) * width + tx * tile
                for lx, v in enumerate(lumas[start:start + tile]):
                    if v <= DARK:
                        sx, sy, n = sx + lx, sy + ly, n + 1
    if not n:
        return "C"
    mid = (tile - 1) / 2
    dx, dy = sx / n - mid, sy / n - mid
    if abs(dx) < 0.5 or abs(dy) < 0.5:
        return "C"
    return ("T" if dy < 0 else "B") + ("L" if dx < 0 else "R")


def _steps(lumas: bytes, width: int, height: int, vertical: bool) -> list[list[bool]]:
    """For each grid line (column x, or row y when ``vertical`` is False) whether each pixel along it steps.

    A pixel steps when it differs from the one before it across the line by EDGE_JUMP or more and by EDGE_RATIO
    times the larger of the steps just before and just after.
    """
    def at(i: int, j: int) -> int:                       # i across the line, j along it
        return lumas[j * width + i] if vertical else lumas[i * width + j]

    across, along = (width, height) if vertical else (height, width)
    lines: list[list[bool]] = [[] for _ in range(across)]
    for i in range(1, across):
        for j in range(along):
            jump = abs(at(i, j) - at(i - 1, j))
            before = abs(at(i - 1, j) - at(i - 2, j)) if i >= 2 else 0
            after = abs(at(i + 1, j) - at(i, j)) if i + 1 < across else 0
            lines[i].append(jump >= EDGE_JUMP and jump >= EDGE_RATIO * max(before, after))
    return lines


def find_blocks(lumas: bytes, width: int, height: int) -> Optional[dict]:
    """Rectangular blocks cut off from the picture around them, or None.

    A grid line is an edge line when MIN_LINE_ROWS or more of its pixels step (see ``_steps``); the block width is
    the greatest common divisor of the edge columns and the height that of the edge rows (both at least
    MIN_BLOCK). On that grid a block edge is cut when half its pixels or more step; the blocks outside the largest
    region joined by uncut edges are the blocks counted.

    Refused: fewer than MIN_EDGE_LINES edge lines on either axis (one sharp rectangle, such as a HUD panel, has two
    a side, and the divisor of two positions is no grid), and a share over MAX_BLOCK_SHARE (a checkerboard texture:
    no picture around the blocks). Still misread: a grid that only ever shows multiples of the true size (every
    cut block on even positions) reads as the larger size; blocks off the frame's origin (a grid shifted by 4)
    read at the divisor of their edges (4x4); several sharp rectangles on a few common divisors (HUD panels,
    window frames) can read as BLOCKS.
    """
    cols = _steps(lumas, width, height, True)
    rows = _steps(lumas, width, height, False)
    edge_cols = [x for x, line in enumerate(cols) if sum(line) >= MIN_LINE_ROWS]
    edge_rows = [y for y, line in enumerate(rows) if sum(line) >= MIN_LINE_ROWS]
    if len(edge_cols) < MIN_EDGE_LINES or len(edge_rows) < MIN_EDGE_LINES:
        return None
    bw, bh = math.gcd(*edge_cols), math.gcd(*edge_rows)
    if bw < MIN_BLOCK or bh < MIN_BLOCK:
        return None
    nx, ny = width // bw, height // bh
    if nx < 2 or ny < 2:
        return None
    parent = list(range(nx * ny))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for by in range(ny):
        for bx in range(nx):
            if bx + 1 < nx and sum(cols[(bx + 1) * bw][by * bh:(by + 1) * bh]) * 2 < bh:
                parent[root(by * nx + bx)] = root(by * nx + bx + 1)
            if by + 1 < ny and sum(rows[(by + 1) * bh][bx * bw:(bx + 1) * bw]) * 2 < bw:
                parent[root(by * nx + bx)] = root((by + 1) * nx + bx)
    sizes: dict[int, int] = {}
    for i in range(nx * ny):
        r = root(i)
        sizes[r] = sizes.get(r, 0) + 1
    cut = nx * ny - max(sizes.values())
    if not cut or cut > MAX_BLOCK_SHARE * nx * ny:
        return None
    return {"width": bw, "height": bh, "blocks": cut, "of": nx * ny, "share": round(cut / (nx * ny), 4)}


# ---- two frames ------------------------------------------------------------------------------------------------

def tile_diff(frame: Frame, reference: Frame, tolerance: int = 0) -> dict:
    """The 8x8 tiles where a channel of ``frame`` differs from ``reference`` by more than ``tolerance``.

    Tiles are listed as ``[tx, ty]`` in row order; partial tiles at the right and bottom edges count.
    """
    if (frame.width, frame.height) != (reference.width, reference.height):
        raise ValueError(f"the frames differ in size: {frame.width}x{frame.height} and "
                         f"{reference.width}x{reference.height}")
    width, height = frame.width, frame.height
    nx, ny = -(-width // DIFF_TILE), -(-height // DIFF_TILE)
    changed = set()
    a, b = frame.rgb, reference.rgb
    for y in range(height):
        start, end = y * width * 3, (y + 1) * width * 3
        if a[start:end] == b[start:end]:
            continue
        ty = y // DIFF_TILE
        for i in range(start, end):
            if abs(a[i] - b[i]) > tolerance:
                changed.add((ty, (i - start) // 3 // DIFF_TILE))
    tiles = [[tx, ty] for ty, tx in sorted(changed)]
    box = None
    if tiles:
        box = [min(t[0] for t in tiles), min(t[1] for t in tiles), max(t[0] for t in tiles),
               max(t[1] for t in tiles)]
    return {"tile": DIFF_TILE, "changed": len(tiles), "of": nx * ny, "share": round(len(tiles) / (nx * ny), 4),
            "box": box, "tiles": tiles}


# ---- the block -------------------------------------------------------------------------------------------------

def measure(frame: Frame, reference: Optional[Frame] = None, given: Optional[str] = None,
            region: Optional[tuple[int, int, int, int]] = None, tolerance: int = 0) -> dict:
    """Every number of the PICTURE block, as a dict (the ``--json`` output)."""
    lumas = luma(frame)
    out: dict = {"picture": 1, "frame": {"width": frame.width, "height": frame.height},
                 "luma": luma_stats(lumas), "dark_regions": dark_regions(lumas, frame.width, frame.height),
                 "lattice": find_lattice(lumas, frame.width, frame.height),
                 "blocks": find_blocks(lumas, frame.width, frame.height),
                 "reference": None, "diff": None, "region": None}
    if reference is not None:
        out["diff"] = tile_diff(frame, reference, tolerance)
        out["reference"] = {"width": reference.width, "height": reference.height,
                            "luma": luma_stats(luma(reference))}
    if region is not None:
        out["region"] = region_stats(lumas, frame.width, region)
    if out["luma"]["dark"] >= BLACK_NEAR_SHARE:
        measured = "BLACK-NEAR"
    elif out["reference"] and abs(out["luma"]["mean"] - out["reference"]["luma"]["mean"]) >= FLASH_DELTA:
        measured = "FLASH"
    elif out["lattice"]:
        measured = "LATTICE"
    elif out["blocks"]:
        measured = "BLOCKS"
    else:
        measured = "OK"
    out["class"] = {"measured": measured, "given": given}
    return out


def _luma_text(s: dict) -> str:
    return f"mean {s['mean']} | min {s['min']} | max {s['max']} | sd {s['sd']} | dark {s['dark']}"


def render(m: dict) -> str:
    """The PICTURE block as text: numbers and classes only."""
    f = m["frame"]
    lines = ["PICTURE | <date UTC> | <GitHub handle> | AI-assisted: <tool>"]
    frame = f"{f['width']}x{f['height']}"
    if m["reference"]:
        frame += f" | reference {m['reference']['width']}x{m['reference']['height']}"
    lines.append(f"FRAME:    {frame}")
    lines.append(f"LUMA:     {_luma_text(m['luma'])}")
    lines.append("DARK 4x4: " + " / ".join(" ".join(f"{v:.2f}" for v in row) for row in m["dark_regions"]))
    lat = m["lattice"]
    lines.append("LATTICE:  " + ("none" if not lat else
                                 f"tile {lat['tile']} | period {lat['period']} | phase {lat['phase']} | "
                                 f"diagonal {lat['diagonal']} | corner {lat['corner']} | tiles {lat['tiles']} | "
                                 f"cover {lat['cover']}"))
    blk = m["blocks"]
    lines.append("BLOCKS:   " + ("none" if not blk else
                                 f"block {blk['width']}x{blk['height']} | blocks {blk['blocks']} of {blk['of']} | "
                                 f"share {blk['share']}"))
    d = m["diff"]
    if d is None:
        lines.append("DIFF:     none (one frame)")
    else:
        text = f"changed {d['changed']} of {d['of']} {d['tile']}x{d['tile']} tiles | share {d['share']}"
        if d["box"]:
            text += " | box {},{}-{},{}".format(*d["box"])
            listed = " ".join(f"{tx},{ty}" for tx, ty in d["tiles"][:TEXT_TILES])
            more = len(d["tiles"]) - TEXT_TILES
            text += f" | tiles {listed}" + (f" (+{more})" if more > 0 else "")
        text += f" | reference luma mean {m['reference']['luma']['mean']}"
        lines.append(f"DIFF:     {text}")
    if m["region"]:
        r = m["region"]
        lines.append(f"REGION:   {r['width']}x{r['height']} | {_luma_text(r)}")
    lines.append(f"CLASS:    {m['class']['measured']} (measured) | given: {m['class']['given'] or 'none'}")
    lines.append("EVIDENCE: MEASURED (tools/picture_metrics.py, 1 frame)" if d is None
                 else "EVIDENCE: MEASURED (tools/picture_metrics.py, 2 frames)")
    return "\n".join(lines) + "\n"


def _read(path: str, what: str) -> Frame:
    try:
        with open(path, "rb") as fh:
            return decode_image(fh.read())
    except OSError as e:
        raise ValueError(f"{what}: cannot read it: {e.strerror or e}") from None
    except ValueError as e:
        raise ValueError(f"{what}: {e}") from None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("frame", help="the frame to measure (PNG or binary PPM/PGM)")
    ap.add_argument("reference", nargs="?", help="a second frame: adds the per-8x8 diff and FLASH")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON instead of the text block")
    ap.add_argument("--class", dest="given", choices=CLASSES, metavar="CLASS",
                    help="a class a human saw, printed as given next to the measured one: " + ", ".join(CLASSES))
    ap.add_argument("--region", help="X,Y,W,H: also print the luma statistics of this rectangle of the frame")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="diff: ignore channel changes up to N (default 0: any change counts)")
    a = ap.parse_args(argv)
    if not 0 <= a.tolerance <= 255:
        ap.error("--tolerance must be within 0..255")
    region = None
    if a.region is not None:
        try:
            region = tuple(int(v) for v in a.region.split(","))
        except ValueError:
            ap.error("--region takes four integers X,Y,W,H")
        if len(region) != 4 or min(region) < 0 or not region[2] or not region[3]:
            ap.error("--region takes four integers X,Y,W,H with W and H above 0")
    try:
        frame = _read(a.frame, "frame")
        reference = _read(a.reference, "reference") if a.reference else None
        if region and (region[0] + region[2] > frame.width or region[1] + region[3] > frame.height):
            raise ValueError(f"--region goes past the {frame.width}x{frame.height} frame")
        m = measure(frame, reference, a.given, region, a.tolerance)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(m, indent=2) + "\n" if a.json else render(m))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
