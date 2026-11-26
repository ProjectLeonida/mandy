"""Synthetic multi-level raw/zlib sibling mip fitting tests for DirectStorage sidecar generation."""
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
from test_bcpack_history_calib import (  # noqa: E402
    _encode_bc6h_mode14_tiled,
    _encode_bc7_mode6_tiled,
    _pack_etpf,
    _write_custom_rds5_pool,
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


def _initial_fit(pools):
    fits = {6: history.calib.Bc6hCalibrator(), 7: history.calib.Bc7Calibrator()}
    history.calib.calibrate_from_pools(pools, fits)
    docs = {history.calib.CODEC_KEYS[c]: f.build_document()[0] for c, f in fits.items()}
    history.extend_missing(docs, history.fit_slices(pools))
    return docs


class MultilevelMipFitTest(unittest.TestCase):
    def test_bc7_multilevel_parent_downsample_unlocks_provisional_decode(self):
        # 128x128 3-mip 2D BC7 chain: mip[0] zlib, mip[1] self-contained BC7, mip[2] BC7 needing entry 176.
        # Because mip[2] is 2 levels away from mip[0] and adjacent mip[1] is BC7, adjacent fitting misses 176.
        color = (72, 136, 184, 220)
        solid = _solid_bc7_mode6_block(color)
        secret = color * 6
        raw_mip2 = _encode_bc7_mode6_tiled(solid * 256, secret_entry=176, secret_ep=secret, ref_count=160)
        raw_mip1 = _encode_bc7_mode6_tiled(solid * 256)
        zlib_mip0 = zlib.compress(solid * 1024, 9)

        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "parent-multilevel.ttpf"
            record = _pack_etpf(128, 128, 3, 2, 98, 4096, mode=4)
            _write_custom_rds5_pool(
                pool,
                [record],
                [(7, 4096, raw_mip2), (7, 4096, raw_mip1), (8, 16384, zlib_mip0)],
            )
            docs = _initial_fit([pool])
            self.assertEqual(docs["bc7"]["historyEndpointBits"][176], [0] * 24)
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(raw_mip2, 7, 4096, allow_provisional=True, history=docs)

            history.extend_missing(docs, history.fit_distant_mips([pool], docs))

        for c, target in enumerate(color):
            for field in (c, 4 + c):
                self.assertGreaterEqual(docs["bc7"]["historyEndpointBits"][176][field], 4)
                self.assertEqual(docs["bc7"]["historyEndpointConfirmedBits"][176][field], 0)
                self.assertLessEqual(abs(docs["bc7"]["historyEndpoints"][176][field] - target), 4)
        decoded = bcpack.decode(raw_mip2, 7, 4096, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 4096)

    def test_bc7_multilevel_child_upsample_unlocks_provisional_decode(self):
        # 128x128 3-mip 2D BC7 chain: mip[0] BC7 needing entry 209, mip[1] self-contained BC7, mip[2] zlib.
        color = (160, 88, 44, 248)
        solid = _solid_bc7_mode6_block(color)
        secret = color * 6
        zlib_mip2 = zlib.compress(solid * 256, 9)
        raw_mip1 = _encode_bc7_mode6_tiled(solid * 256)
        raw_mip0 = _encode_bc7_mode6_tiled(solid * 1024, secret_entry=209, secret_ep=secret, ref_count=48)

        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "child-multilevel.ttpf"
            record = _pack_etpf(128, 128, 3, 2, 98, 4096, mode=4)
            _write_custom_rds5_pool(
                pool,
                [record],
                [(8, 4096, zlib_mip2), (7, 4096, raw_mip1), (7, 16384, raw_mip0)],
            )
            docs = _initial_fit([pool])
            self.assertEqual(docs["bc7"]["historyEndpointBits"][209], [0] * 24)
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(raw_mip0, 7, 16384, allow_provisional=True, history=docs)

            history.extend_missing(docs, history.fit_distant_mips([pool], docs))

        for c, target in enumerate(color):
            for field in (c, 4 + c):
                self.assertGreaterEqual(docs["bc7"]["historyEndpointBits"][209][field], 4)
                self.assertLessEqual(abs(docs["bc7"]["historyEndpoints"][209][field] - target), 4)
        decoded = bcpack.decode(raw_mip0, 7, 16384, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 16384)

    def test_bc6h_multilevel_fit_and_existing_slot_precedence(self):
        # 128x128 3-mip 2D BC6H UF16 chain: mip[0] zlib, mip[1] self-contained BC6H, mip[2] BC6H needing entry 91.
        w_rgb = (0x3400, 0x2800, 0x1C00)
        solid = _solid_bc6h_mode14_block(w_rgb)
        secret_eh = tuple((w_rgb[c], 0x200, 0x20, 0x20) for c in range(3))
        raw_mip2 = _encode_bc6h_mode14_tiled(solid * 256, secret_entry=91, secret_eh=secret_eh, ref_count=160)
        raw_mip1 = _encode_bc6h_mode14_tiled(solid * 256)
        zlib_mip0 = zlib.compress(solid * 1024, 9)

        with tempfile.TemporaryDirectory() as tmp:
            pool = Path(tmp) / "bc6h-multilevel.ttpf"
            record = _pack_etpf(128, 128, 3, 2, 95, 4096, mode=4)
            _write_custom_rds5_pool(
                pool,
                [record],
                [(6, 4096, raw_mip2), (6, 4096, raw_mip1), (8, 16384, zlib_mip0)],
            )
            docs = _initial_fit([pool])
            self.assertEqual(docs["bc6h"]["endpointClass"][91], 0)
            with self.assertRaises(bcpack.BcpackProvisionalError):
                bcpack.decode(raw_mip2, 6, 4096, allow_provisional=True, history=docs)

            # A pre-populated slot in `docs` must keep precedence over distant-mip estimates.
            docs["bc7"]["historyEndpointBits"][5][0] = 6
            docs["bc7"]["historyEndpoints"][5][0] = 192
            extra = history.fit_distant_mips([pool], docs)
            extra["bc7"]["historyEndpointBits"][5][0] = 5
            extra["bc7"]["historyEndpoints"][5][0] = 32
            history.extend_missing(docs, extra)

        self.assertEqual(docs["bc6h"]["endpointClass"][91], 1)
        self.assertEqual(docs["bc7"]["historyEndpointBits"][5][0], 6)
        self.assertEqual(docs["bc7"]["historyEndpoints"][5][0], 192)
        decoded = bcpack.decode(raw_mip2, 6, 4096, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 4096)


if __name__ == "__main__":
    unittest.main()
