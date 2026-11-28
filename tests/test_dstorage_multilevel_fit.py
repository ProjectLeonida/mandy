"""Synthetic multi-level raw/zlib sibling mip fitting tests for DirectStorage sidecar generation."""
from array import array
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
from test_bcpack_history_calib import (  # noqa: E402
    _encode_bc6h_mode14_tiled,
    _encode_bc7_mode6_tiled,
    _pack_etpf,
    _write_custom_rds5_pool,
    encode_lanes,
)
import bcpack  # noqa: E402
import build_dstorage_history as history  # noqa: E402


def _solid_bc7_mode6_block(rgba):
    r, g, b, a = rgba
    acc = 1 << 6
    pos = 7
    for ch in (r, g, b, a):
        for _ in range(2):
            acc |= ((ch >> 1) & 0x7F) << pos
            pos += 7
    acc |= (r & 1) << pos
    acc |= (r & 1) << (pos + 1)
    return acc.to_bytes(16, "little")


def _solid_bc6h_mode14_block(w_rgb):
    plan, _ = bcpack._bc6h_plan(0x0F)
    fields = [[bcpack._bc6h_stream_w(14, w_rgb[c]), 0, 0, 0] for c in range(3)]
    acc = 0x0F
    for c, q, bit_idx, pos in plan:
        if c >= 0 and ((fields[c][q] >> bit_idx) & 1):
            acc |= 1 << pos
    return acc.to_bytes(16, "little")


def _refused_bc7_mode6_stream(nblocks):
    solid = bytes([0x40]) + bytes(15)
    lanes, _ = bcpack.decode_lanes(_encode_bc7_mode6_tiled(solid * nblocks), 7, nblocks * 16)
    lanes[11] = [2, 0] * nblocks
    return encode_lanes(lanes, 7)


def _zero_votes(_pools, _trainer, _temp):
    return array("Q", [0]) * (2 * 256 * 35 * 16)


def _build_history(game_input, work, base_history=None):
    sidecar = work / "out" / "history.bcph"
    json_out = work / "out" / "fitted.json"
    with mock.patch.object(history, "collect", side_effect=_zero_votes):
        history.build(
            Path(game_input),
            sidecar,
            Path("stub-trainer"),
            base_history=base_history,
            json_out=json_out,
        )
    raw = sidecar.read_bytes()
    assert raw[:8] == b"BCPHIST1" and len(raw) == 60936
    return json.loads(json_out.read_text(encoding="utf-8"))


class MultilevelMipFitTest(unittest.TestCase):
    def test_bc7_multilevel_parent_downsample_unlocks_provisional_decode(self):
        # 128x128 3-mip 2D array BC7 chain: mip[0] lossless zlib, mip[1] BC7 with an undecodable
        # anchor bit (so neighbor decode masks out mip[1]), and mip[2] BC7 needing entry 176.
        color = (72, 136, 184, 220)
        solid = _solid_bc7_mode6_block(color)
        secret = color * 6
        raw_mip2 = _encode_bc7_mode6_tiled(solid * 256, secret_entry=176, secret_ep=secret, ref_count=255)
        raw_mip1 = _refused_bc7_mode6_stream(256)
        zlib_mip0 = zlib.compress(solid * 1024, 9)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pool = work / "pools" / "parent-multilevel.ttpf"
            pool.parent.mkdir(parents=True, exist_ok=True)
            record = _pack_etpf(128, 128, 3, 2, 98, 4096, mode=4)
            _write_custom_rds5_pool(
                pool,
                [record],
                [(7, 4096, raw_mip2), (7, 4096, raw_mip1), (8, 16384, zlib_mip0)],
            )
            docs = _build_history(pool, work)

        for c, target in enumerate(color):
            for field in (c, 4 + c):
                self.assertGreaterEqual(docs["bc7"]["historyEndpointBits"][176][field], 4)
                self.assertEqual(docs["bc7"]["historyEndpointConfirmedBits"][176][field], 0)
                self.assertLessEqual(abs(docs["bc7"]["historyEndpoints"][176][field] - target), 4)
        decoded = bcpack.decode(raw_mip2, 7, 4096, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 4096)

    def test_bc7_multilevel_child_upsample_unlocks_provisional_decode(self):
        # 128x128 3-mip 2D array BC7 chain: mip[0] BC7 needing entry 209, mip[1] undecodable BC7,
        # and mip[2] lossless zlib two levels smaller.
        color = (160, 88, 44, 248)
        solid = _solid_bc7_mode6_block(color)
        secret = color * 6
        zlib_mip2 = zlib.compress(solid * 256, 9)
        raw_mip1 = _refused_bc7_mode6_stream(256)
        raw_mip0 = _encode_bc7_mode6_tiled(solid * 1024, secret_entry=209, secret_ep=secret, ref_count=48)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pool = work / "pools" / "child-multilevel.ttpf"
            pool.parent.mkdir(parents=True, exist_ok=True)
            record = _pack_etpf(128, 128, 3, 2, 98, 4096, mode=4)
            _write_custom_rds5_pool(
                pool,
                [record],
                [(8, 4096, zlib_mip2), (7, 4096, raw_mip1), (7, 16384, raw_mip0)],
            )
            docs = _build_history(pool, work)

        for c, target in enumerate(color):
            for field in (c, 4 + c):
                self.assertGreaterEqual(docs["bc7"]["historyEndpointBits"][209][field], 4)
                self.assertLessEqual(abs(docs["bc7"]["historyEndpoints"][209][field] - target), 4)
        decoded = bcpack.decode(raw_mip0, 7, 16384, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 16384)

    def test_bc6h_multilevel_fit_and_existing_slot_precedence(self):
        # 128x128 BC6H UF16 pools where mip[0] is lossless zlib, mip[1] is refused BC6H, and mip[2]
        # needs unknown entry 91 (plus a second pool referencing pre-populated entry 12 to verify
        # base_history slots keep precedence over distant-mip estimates).
        w_rgb = (0x3400, 0x2800, 0x1C00)
        solid = _solid_bc6h_mode14_block(w_rgb)
        lanes_mip1 = [[0] * 512 for _ in range(16)]
        lanes_mip1[15] = [13, 0] * 256
        raw_mip1 = encode_lanes(lanes_mip1, 6)
        secret_91 = tuple((w_rgb[c], 0x200, 0x20, 0x20) for c in range(3))
        raw_mip2 = _encode_bc6h_mode14_tiled(
            solid * 256, secret_entry=91, secret_eh=secret_91, ref_count=160
        )
        zlib_mip0 = zlib.compress(solid * 1024, 9)

        other_w = (0x1000, 0x1400, 0x1800)
        other_solid = _solid_bc6h_mode14_block(other_w)
        secret_12 = tuple((other_w[c], 0x200, 0x20, 0x20) for c in range(3))
        other_mip2 = _encode_bc6h_mode14_tiled(
            other_solid * 256, secret_entry=12, secret_eh=secret_12, ref_count=160
        )
        other_zlib = zlib.compress(other_solid * 1024, 9)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pools_dir = work / "pools"
            pools_dir.mkdir(parents=True, exist_ok=True)
            record = _pack_etpf(128, 128, 3, 2, 95, 4096, mode=4)
            _write_custom_rds5_pool(
                pools_dir / "bc6h-multilevel.ttpf",
                [record],
                [(6, 4096, raw_mip2), (6, 4096, raw_mip1), (8, 16384, zlib_mip0)],
            )
            _write_custom_rds5_pool(
                pools_dir / "bc6h-precedence.ttpf",
                [record],
                [(6, 4096, other_mip2), (6, 4096, raw_mip1), (8, 16384, other_zlib)],
            )

            base_dir = work / "base"
            base_dir.mkdir(parents=True, exist_ok=True)
            base_bc6h = history.calib.Bc6hCalibrator().build_document()[0]
            kept_entry = [[0x2AAA, 11, 2, 3], [0x2BBB, 12, 3, 4], [0x2CCC, 13, 4, 5]]
            base_bc6h["endpointClass"][12] = 1
            base_bc6h["endpointHistory"][12] = kept_entry
            base_bc7 = history.calib.Bc7Calibrator().build_document()[0]
            base_bc7["historyEndpointBits"][5][0] = 6
            base_bc7["historyEndpoints"][5][0] = 192
            (base_dir / "bc6h.json").write_text(json.dumps(base_bc6h), encoding="utf-8", newline="\n")
            (base_dir / "bc7.json").write_text(json.dumps(base_bc7), encoding="utf-8", newline="\n")

            docs = _build_history(pools_dir, work, base_history=base_dir)

        self.assertEqual(docs["bc6h"]["endpointClass"][91], 1)
        self.assertEqual(docs["bc6h"]["endpointClass"][12], 1)
        self.assertEqual(docs["bc6h"]["endpointHistory"][12], kept_entry)
        self.assertEqual(docs["bc7"]["historyEndpointBits"][5][0], 6)
        self.assertEqual(docs["bc7"]["historyEndpoints"][5][0], 192)
        decoded = bcpack.decode(raw_mip2, 6, 4096, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 4096)

    def test_refused_bc6h_intermediate_sibling_falls_back_to_multilevel_zlib_mip(self):
        # mip[0] is zlib, mip[1] is a BC6H stream refused by decode (inconsistent anchor index 8), and mip[2] needs
        # unknown endpoint entry 91. Neighbor decoding cannot decode mip[1] as a reference, whereas multi-level
        # fitting pairs mip[2] directly with lossless zlib mip[0] two levels up.
        w_rgb = (0x3400, 0x2800, 0x1C00)
        solid = _solid_bc6h_mode14_block(w_rgb)
        lanes_mip1 = [[0] * 512 for _ in range(16)]
        lanes_mip1[15] = [13, 0] * 256
        raw_mip1 = encode_lanes(lanes_mip1, 6)
        secret_eh = tuple((w_rgb[c], 0x200, 0x20, 0x20) for c in range(3))
        raw_mip2 = _encode_bc6h_mode14_tiled(solid * 256, secret_entry=91, secret_eh=secret_eh, ref_count=160)
        zlib_mip0 = zlib.compress(solid * 1024, 9)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pool = work / "pools" / "bc6h-refused-mid.ttpf"
            pool.parent.mkdir(parents=True, exist_ok=True)
            record = _pack_etpf(128, 128, 3, 2, 95, 4096, mode=4)
            _write_custom_rds5_pool(
                pool,
                [record],
                [(6, 4096, raw_mip2), (6, 4096, raw_mip1), (8, 16384, zlib_mip0)],
            )
            docs = _build_history(pool, work)

        self.assertEqual(docs["bc6h"]["endpointClass"][91], 1)
        decoded = bcpack.decode(raw_mip2, 6, 4096, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 4096)


if __name__ == "__main__":
    unittest.main()
