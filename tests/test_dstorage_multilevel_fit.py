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
    _deint4,
    _deint_n,
    _encode_bc6h_mode14_tiled,
    _encode_bc7_mode6_tiled,
    _pack_etpf,
    _pack_ix_pair,
    _signed_mod_delta,
    _signed_mod_delta_m,
    _write_custom_rds5_pool,
    _zig,
    encode_lanes,
)
import bcpack  # noqa: E402
import bcpack_tables_bptc as bptc  # noqa: E402
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


def _encode_dual_bc6h_mode14_tiled(tiled_bytes, even_slot, even_eh, odd_slot, odd_eh):
    """Encode Mode-14 BC6H blocks referencing `even_slot` on even blocks and `odd_slot` on odd blocks."""
    assert 192 <= even_slot <= 255 and 192 <= odd_slot <= 255
    nblocks = len(tiled_bytes) // 16
    lanes = [[] for _ in range(16)]
    for i in range(nblocks):
        parsed = history.calib.unpack_bc6h_block(tiled_bytes, i * 16)
        assert parsed is not None and parsed[0] == 14
        _mode, part, f, ixs = parsed
        slot, eref = (even_slot, even_eh) if (i % 2 == 0) else (odd_slot, odd_eh)
        if i < slot:
            de = 256 + i - slot
            assert i < de <= 255
        else:
            de = 0
            eref = bcpack._BC6H_NONE_E
        lanes[0].extend(_deint4(_zig(_signed_mod_delta_m(de, 0, 256))))
        lanes[1].extend(_deint4(0))
        lanes[15].extend([13, part])
        for c in range(3):
            dz0 = _signed_mod_delta(f[c][0], eref[c][0], 16)
            dz1 = _signed_mod_delta(f[c][1], eref[c][1] >> 6, 4)
            lanes[2 + c].extend(_deint_n(_zig(dz0), 8))
            lanes[5 + c].extend(_deint_n(_zig(dz1), 5))
            lanes[8 + c].extend([0, 0])
        for yy in range(4):
            ds = [_zig(_signed_mod_delta(ixs[yy * 4 + xx], 0, 4)) for xx in range(4)]
            lanes[11 + yy].extend([_pack_ix_pair(ds[0], ds[1], yy), _pack_ix_pair(ds[2], ds[3], yy)])
    return encode_lanes(lanes, 6)


def _encode_dual_bc7_mode6_tiled(tiled_bytes, even_slot, even_ep, odd_slot, odd_ep):
    """Encode Mode-6 BC7 blocks referencing `even_slot` on even blocks and `odd_slot` on odd blocks."""
    assert 192 <= even_slot <= 255 and 192 <= odd_slot <= 255
    tab = bptc.load_bc7()
    rank = tab["config"].index((6, 0, 0, 0))
    s15a, s15b = divmod(tab["rank_of"].index(rank), 16)
    nblocks = len(tiled_bytes) // 16
    lanes = [[] for _ in range(16)]
    for i in range(nblocks):
        acc = int.from_bytes(tiled_bytes[i * 16:(i + 1) * 16], "little")
        pos = 7
        ch = [[0, 0] for _ in range(4)]
        for c in range(4):
            for e in range(2):
                ch[c][e] = (acc >> pos) & 0x7F
                pos += 7
        p0 = (acc >> pos) & 1
        p1 = (acc >> (pos + 1)) & 1
        pos += 2
        ch[0][0] = (ch[0][0] << 1) | p0
        ch[0][1] = (ch[0][1] << 1) | p1
        ixs = []
        for p in range(16):
            nb = 3 if p == 0 else 4
            ixs.append((acc >> pos) & ((1 << nb) - 1))
            pos += nb

        slot, rep = (even_slot, even_ep) if (i % 2 == 0) else (odd_slot, odd_ep)
        if i < slot:
            de = 256 + i - slot
            assert i < de <= 255
        else:
            de = 0
            rep = (0,) * 24
        lanes[0].extend(_deint4(_zig(_signed_mod_delta_m(de, 0, 256))))
        lanes[1].extend(_deint4(0))
        lanes[15].extend([s15a, s15b])
        for c, ln in enumerate((2, 3, 4, 5)):
            bits = 8 if c == 0 else 7
            sh = 8 - bits
            lanes[ln].extend([
                _zig(_signed_mod_delta(ch[c][0], rep[c] >> sh, bits)),
                _zig(_signed_mod_delta(ch[c][1], rep[4 + c] >> sh, bits)),
            ])
        for ln in (6, 7, 8, 9, 10):
            lanes[ln].extend([0, 0])
        for y in range(4):
            ds = [_zig(_signed_mod_delta(ixs[y * 4 + x], 0, 4)) for x in range(4)]
            lanes[11 + y].extend([_pack_ix_pair(ds[0], ds[1], y), _pack_ix_pair(ds[2], ds[3], y)])
    return encode_lanes(lanes, 7)


def _zero_votes(_pools, _trainer, _temp):
    return array("Q", [0]) * (2 * 256 * 35 * 16)


class MultilevelMipFitTest(unittest.TestCase):
    def _build_history(self, game_input, work, base_history=None):
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
        self.assertEqual((raw[:8], len(raw)), (b"BCPHIST1", 60936))
        return json.loads(json_out.read_text(encoding="utf-8"))

    def test_bc7_multilevel_parent_downsample_unlocks_provisional_decode(self):
        # 128x128 5-mip (2 non-tail + 3 packed-tail) 2D BC7 chain: mip[0] lossless zlib,
        # mip[1] BC7 with an undecodable anchor bit (so neighbor decode masks out mip[1]),
        # and packed tail mips[2..4] rooting all 256 blocks at entry 255 without wrap-around.
        color = (72, 136, 184, 220)
        solid = _solid_bc7_mode6_block(color)
        secret = color * 6
        raw_tail = _encode_bc7_mode6_tiled(solid * 256, secret_entry=255, secret_ep=secret, ref_count=255)
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
                [(7, 4096, raw_tail), (7, 4096, raw_mip1), (8, 16384, zlib_mip0)],
            )
            docs = self._build_history(pool, work)

        for c, target in enumerate(color):
            for field in (c, 4 + c):
                self.assertGreaterEqual(docs["bc7"]["historyEndpointBits"][255][field], 4)
                self.assertEqual(docs["bc7"]["historyEndpointConfirmedBits"][255][field], 0)
                self.assertLessEqual(abs(docs["bc7"]["historyEndpoints"][255][field] - target), 4)
        decoded = bcpack.decode(raw_tail, 7, 4096, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 4096)
        expected_px = history.calib.decode_bc7_block_rgba(solid, 0)[0]
        for blk in range(256):
            px = history.calib.decode_bc7_block_rgba(decoded, blk * 16)[0]
            for c in range(4):
                self.assertLessEqual(abs(px[c] - expected_px[c]), 4, f"block {blk} ch {c}")

    def test_bc7_multilevel_child_upsample_unlocks_provisional_decode(self):
        # 128x128 5-mip (2 non-tail + 3 packed-tail) 2D BC7 chain: mip[0] BC7 needing entry 209
        # (ref_count=48 <= 209, zero wrap-around), mip[1] undecodable BC7, and packed tail
        # mips[2..4] lossless zlib two levels smaller.
        color = (160, 88, 44, 248)
        solid = _solid_bc7_mode6_block(color)
        secret = color * 6
        zlib_tail = zlib.compress(solid * 256, 9)
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
                [(8, 4096, zlib_tail), (7, 4096, raw_mip1), (7, 16384, raw_mip0)],
            )
            docs = self._build_history(pool, work)

        for c, target in enumerate(color):
            for field in (c, 4 + c):
                self.assertGreaterEqual(docs["bc7"]["historyEndpointBits"][209][field], 4)
                self.assertEqual(docs["bc7"]["historyEndpointConfirmedBits"][209][field], 0)
                self.assertLessEqual(abs(docs["bc7"]["historyEndpoints"][209][field] - target), 4)
        decoded = bcpack.decode(raw_mip0, 7, 16384, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded), 16384)
        expected_px = history.calib.decode_bc7_block_rgba(solid, 0)[0]
        for blk in range(1024):
            px = history.calib.decode_bc7_block_rgba(decoded, blk * 16)[0]
            for c in range(4):
                self.assertLessEqual(abs(px[c] - expected_px[c]), 4, f"block {blk} ch {c}")

    def test_bc6h_and_bc7_multilevel_fit_preserve_existing_base_history_slots(self):
        # Single dual-slot BC6H chunk and single dual-slot BC7 chunk: each chunk references BOTH
        # a missing slot (240, forcing _chunk_needs_missing_slot == True so fit_distant_mips scans
        # the chunk) AND a pre-populated base_history slot (241) whose distant-mip observation
        # conflicts with base_history, proving extend_missing fills 240 without overwriting 241.
        w_rgb = (0x3400, 0x2800, 0x1C00)
        solid_bc6h = _solid_bc6h_mode14_block(w_rgb)
        stream_w = [bcpack._bc6h_stream_w(14, w_rgb[c]) for c in range(3)]
        pool_eh = tuple((stream_w[c], 0, 0, 0) for c in range(3))
        lanes_mip1 = [[0] * 512 for _ in range(16)]
        lanes_mip1[15] = [13, 0] * 256
        refused_bc6h_mip1 = encode_lanes(lanes_mip1, 6)
        dual_bc6h_tail = _encode_dual_bc6h_mode14_tiled(solid_bc6h * 256, 240, pool_eh, 241, pool_eh)
        zlib_bc6h_mip0 = zlib.compress(solid_bc6h * 1024, 9)

        color_bc7 = (72, 136, 184, 220)
        solid_bc7 = _solid_bc7_mode6_block(color_bc7)
        pool_ep_bc7 = color_bc7 * 6
        dual_bc7_tail = _encode_dual_bc7_mode6_tiled(solid_bc7 * 256, 240, pool_ep_bc7, 241, pool_ep_bc7)
        refused_bc7_mip1 = _refused_bc7_mode6_stream(256)
        zlib_bc7_mip0 = zlib.compress(solid_bc7 * 1024, 9)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pools_dir = work / "pools"
            pools_dir.mkdir(parents=True, exist_ok=True)
            rec_bc6h = _pack_etpf(128, 128, 3, 2, 95, 4096, mode=4)
            _write_custom_rds5_pool(
                pools_dir / "bc6h-dual.ttpf",
                [rec_bc6h],
                [(6, 4096, dual_bc6h_tail), (6, 4096, refused_bc6h_mip1), (8, 16384, zlib_bc6h_mip0)],
            )
            rec_bc7 = _pack_etpf(128, 128, 3, 2, 98, 4096, mode=4)
            _write_custom_rds5_pool(
                pools_dir / "bc7-dual.ttpf",
                [rec_bc7],
                [(7, 4096, dual_bc7_tail), (7, 4096, refused_bc7_mip1), (8, 16384, zlib_bc7_mip0)],
            )

            base_dir = work / "base"
            base_dir.mkdir(parents=True, exist_ok=True)
            base_bc6h = history.calib.Bc6hCalibrator().build_document()[0]
            kept_bc6h = [[0x2AAA, 11, 2, 3], [0x2BBB, 12, 3, 4], [0x2CCC, 13, 4, 5]]
            base_bc6h["endpointClass"][241] = 1
            base_bc6h["endpointHistory"][241] = kept_bc6h
            base_bc7 = history.calib.Bc7Calibrator().build_document()[0]
            for f in range(8):
                base_bc7["historyEndpointBits"][241][f] = 6
                base_bc7["historyEndpoints"][241][f] = 192
            (base_dir / "bc6h.json").write_text(json.dumps(base_bc6h), encoding="utf-8", newline="\n")
            (base_dir / "bc7.json").write_text(json.dumps(base_bc7), encoding="utf-8", newline="\n")

            docs = self._build_history(pools_dir, work, base_history=base_dir)

        self.assertEqual(docs["bc6h"]["endpointClass"][240], 1)
        self.assertEqual(docs["bc6h"]["endpointConfirmedBits"][240], [[0, 0, 0, 0]] * 3)
        for c in range(3):
            self.assertLessEqual(abs(docs["bc6h"]["endpointHistory"][240][c][0] - stream_w[c]), 8)
        self.assertEqual(docs["bc6h"]["endpointClass"][241], 1)
        self.assertEqual(docs["bc6h"]["endpointHistory"][241], kept_bc6h)

        for c, target in enumerate(color_bc7):
            for field in (c, 4 + c):
                self.assertGreaterEqual(docs["bc7"]["historyEndpointBits"][240][field], 4)
                self.assertLessEqual(abs(docs["bc7"]["historyEndpoints"][240][field] - target), 4)
        for f in range(8):
            self.assertEqual(docs["bc7"]["historyEndpointBits"][241][f], 6)
            self.assertEqual(docs["bc7"]["historyEndpoints"][241][f], 192)

    def test_refused_bc6h_intermediate_sibling_falls_back_to_multilevel_zlib_mip(self):
        # Full build without base_history across both parent-downsample (slot 200, ref_count=200)
        # and child-upsample (slot 212, ref_count=64) BC6H textures whose intermediate mip[1] is
        # refused by decode (inconsistent anchor index 8), verifying fitted HDR endpoint values
        # and exact decoded block bytes.
        w_parent = (0x3400, 0x2800, 0x1C00)
        solid_parent = _solid_bc6h_mode14_block(w_parent)
        sw_parent = [bcpack._bc6h_stream_w(14, w_parent[c]) for c in range(3)]
        eh_parent = tuple((sw_parent[c], 0, 0, 0) for c in range(3))
        raw_tail_parent = _encode_bc6h_mode14_tiled(
            solid_parent * 256, secret_entry=200, secret_eh=eh_parent, ref_count=200
        )
        zlib_mip0_parent = zlib.compress(solid_parent * 1024, 9)

        w_child = (0x2200, 0x3100, 0x1500)
        solid_child = _solid_bc6h_mode14_block(w_child)
        sw_child = [bcpack._bc6h_stream_w(14, w_child[c]) for c in range(3)]
        eh_child = tuple((sw_child[c], 0, 0, 0) for c in range(3))
        raw_mip0_child = _encode_bc6h_mode14_tiled(
            solid_child * 1024, secret_entry=212, secret_eh=eh_child, ref_count=64
        )
        zlib_tail_child = zlib.compress(solid_child * 256, 9)

        lanes_mip1 = [[0] * 512 for _ in range(16)]
        lanes_mip1[15] = [13, 0] * 256
        raw_mip1 = encode_lanes(lanes_mip1, 6)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pools_dir = work / "pools"
            pools_dir.mkdir(parents=True, exist_ok=True)
            record = _pack_etpf(128, 128, 3, 2, 95, 4096, mode=4)
            _write_custom_rds5_pool(
                pools_dir / "bc6h-parent-refused-mid.ttpf",
                [record],
                [(6, 4096, raw_tail_parent), (6, 4096, raw_mip1), (8, 16384, zlib_mip0_parent)],
            )
            _write_custom_rds5_pool(
                pools_dir / "bc6h-child-refused-mid.ttpf",
                [record],
                [(8, 4096, zlib_tail_child), (6, 4096, raw_mip1), (6, 16384, raw_mip0_child)],
            )
            docs = self._build_history(pools_dir, work)

        for slot, expected_sw in ((200, sw_parent), (212, sw_child)):
            self.assertEqual(docs["bc6h"]["endpointClass"][slot], 1)
            self.assertEqual(docs["bc6h"]["endpointConfirmedBits"][slot], [[0, 0, 0, 0]] * 3)
            for c in range(3):
                self.assertLessEqual(abs(docs["bc6h"]["endpointHistory"][slot][c][0] - expected_sw[c]), 8)

        decoded_parent = bcpack.decode(raw_tail_parent, 6, 4096, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded_parent), 4096)
        exp_parent = history.calib.decode_bc6h_block_unq(solid_parent, 0)[0]
        for blk in range(256):
            px = history.calib.decode_bc6h_block_unq(decoded_parent, blk * 16)[0]
            for c in range(3):
                self.assertLessEqual(abs(px[c] - exp_parent[c]), 64.0, f"parent blk {blk} ch {c}")

        decoded_child = bcpack.decode(raw_mip0_child, 6, 16384, allow_provisional=True, history=docs)
        self.assertEqual(len(decoded_child), 16384)
        exp_child = history.calib.decode_bc6h_block_unq(solid_child, 0)[0]
        for blk in range(1024):
            px = history.calib.decode_bc6h_block_unq(decoded_child, blk * 16)[0]
            for c in range(3):
                self.assertLessEqual(abs(px[c] - exp_child[c]), 64.0, f"child blk {blk} ch {c}")


if __name__ == "__main__":
    unittest.main()
