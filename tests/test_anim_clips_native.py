"""Synthetic comparisons for the optional Rust animation decoder.

    cargo build --release --locked --offline --manifest-path tools/anim_clips_native/Cargo.toml
    python -I tests/test_anim_clips_native.py
"""
import collections
import ctypes
import os
import random
import struct
import sys
import unittest
from unittest.mock import patch

# These tests set the per-module switch themselves; the global LEONIDA_NATIVE=0 opt-out would override it.
os.environ.pop("LEONIDA_NATIVE", None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_anim_clips import anim_clips, encode_sequence  # noqa: E402
import _anim_clips_native as native  # noqa: E402


def animation_case(sequences, ntracks):
    """Pack independently encoded sequences at nonzero offsets in a synthetic system page."""
    data = bytearray(b"\x93" * 17)
    offsets = []
    for _, sequence in sequences:
        data.extend(b"\xa2" * 13)
        offsets.append(len(data))
        data.extend(sequence)
    anim = {"sequences": offsets, "tracks": [(0, 0, i % 2 * 2) for i in range(ntracks)],
            "frames": sum(nf for nf, _ in sequences) - max(0, len(sequences) - 1)}
    return anim_clips.System(bytes(data)), anim


def encoded_case(specs, ntracks):
    return animation_case([(nf, encode_sequence(nf, channels)) for nf, channels in specs], ntracks)


def run_case(case, enabled, stats=None):
    with patch.dict(os.environ, {"LEONIDA_ANIM_NATIVE": str(int(enabled))}):
        try:
            tracks, total = anim_clips.decode_animation(*case, stats)
        except (anim_clips.DecodeError, struct.error, IndexError, OverflowError, KeyError, ValueError) as error:
            result = type(error), str(error)
        else:
            # A packed float compares the sign of zero and all the bits of a NaN.
            result = total, [(track, [struct.pack("<" + "d" * len(frame), *frame) for frame in rows])
                             for track, rows in tracks.items()]
    return result, None if stats is None else list(stats.items())


def random_channels(rng, nf, ntracks, sequence_index):
    def quantized():
        width = rng.choice((0, 1, 5, 9, 12, 28, 31, 32, 53, 56, 63, 64))
        values = [rng.getrandbits(width) for _ in range(nf)]
        q = rng.choice((0.0, -0.0, 2.0 ** -30, -2.0 ** -63, 0.001, 9.0))
        o = rng.choice((0.0, -0.0, 1.0, -0.25, 0.001))
        return width, q, o, values

    channels = []
    for track in range(ntracks):
        mode = (track + rng.randrange(11)) % 11
        if mode == 0 and sequence_index:
            continue
        if mode in (1, 2):
            channels.append((mode - 1, track, 0, (rng.uniform(-1, 1), -0.0, 0.5)))
            # Static whole-track values take precedence over component and marker channels.
            if rng.randrange(2):
                channels.extend(((2, track, 1, 99.0), (7, track, 0, None)))
            continue
        if mode == 3:
            channels.append((3, track, rng.randrange(4), [rng.uniform(-2, 2) for _ in range(nf)]))
            continue
        if mode == 4:
            vb = rng.choice((0, 1, 7, 31, 32, 53, 63, 64))
            table = [rng.getrandbits(vb) for _ in range(5)] if vb else []
            values = [rng.randrange(len(table) or 1) for _ in range(nf)]
            channels.append((5, track, rng.randrange(4), (3 if vb else 0, vb, table, -0.001, -0.0, values)))
            continue
        if mode == 5 and nf:
            width = rng.choice((0, 3, 10, 30, 58, 62))
            values = [rng.getrandbits(width) for _ in range(nf)]
            channels.append((10, track, rng.randrange(4), (-0.015, -0.0, values)))
            continue
        width = 3 if mode in (6, 7) else 4
        for comp in range(width):
            if mode == 6 and comp % 2:
                continue  # Holes in the stored components are zero-filled.
            if mode == 10 and rng.randrange(2):
                channels.append((2, track, comp, rng.choice((-0.0, 0.0, 0.5, -0.61))))
                if rng.randrange(2):  # The last channel for a component wins.
                    channels.append((2, track, comp, 0.1))
            else:
                channels.append((4, track, comp, quantized()))
        if mode == 7:
            channels.append((7, track, rng.randrange(4), None))
        elif mode in (9, 10):
            channels.append((8, track, rng.randrange(4), None))
    rng.shuffle(channels)
    return channels


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.case = encoded_case([(3, [(4, 0, 0, (2, 0.01, -0.0, [0, 1, 2]))])], 1)
        self.expected = run_case(self.case, False, collections.Counter())

    def test_missing_library(self):
        with patch.object(native, "_NATIVE", []), patch.object(native.os.path, "isfile", return_value=False):
            self.assertEqual(run_case(self.case, True, collections.Counter()), self.expected)
            self.assertEqual(native._NATIVE, [None])

    def test_load_error(self):
        with (patch.object(native, "_NATIVE", []), patch.object(native.os.path, "isfile", return_value=True),
              patch.object(ctypes, "CDLL", side_effect=OSError("synthetic load failure"))):
            self.assertEqual(run_case(self.case, True, collections.Counter()), self.expected)
            self.assertEqual(native._NATIVE, [None])

    def test_disabled_native(self):
        with patch.object(native, "_NATIVE", [object()]):
            self.assertEqual(run_case(self.case, False, collections.Counter()), self.expected)

    def test_refused_or_failed_call(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                def decode(*args):
                    if fail:
                        raise ctypes.ArgumentError("synthetic call failure")
                    return None

                def free(*args):
                    self.fail("nothing was allocated")

                with patch.object(native, "_NATIVE", [(ctypes, decode, free)]):
                    self.assertEqual(run_case(self.case, True, collections.Counter()), self.expected)


class NativeTests(unittest.TestCase):
    def setUp(self):
        if native.native() is None:
            self.skipTest("optional library unavailable")

    def assert_parity(self, case, initial=None, *, require_native=False):
        expected = run_case(case, False, initial.copy() if initial is not None else None)
        actual_stats = initial.copy() if initial is not None else None
        if require_native:
            with patch.object(anim_clips, "decode_sequence", side_effect=AssertionError("unexpected Python fallback")):
                self.assertEqual(run_case(case, True, actual_stats), expected)
        else:
            self.assertEqual(run_case(case, True, actual_stats), expected)
        return expected

    def test_randomized_sequences_take_native_path(self):
        rng = random.Random(0xAC45)
        nseq = 0
        for i in range(1800):
            ntracks = rng.randrange(1, 8)
            count = 2 if i % 3 == 0 else 1
            specs = []
            for j in range(count):
                nf = rng.choice((1, 2, 7, 8, 9, 31, 63, 64, 65, 66, 96, 129, 193))
                specs.append((nf, random_channels(rng, nf, ntracks, j)))
            case = encoded_case(specs, ntracks)
            initial = collections.Counter({"preexisting": 7, "quaternion_rebuilt_worst_1e6": 50,
                                           "delta2_boundaries_closed": -4}) if i % 4 else None
            with self.subTest(i=i):
                self.assert_parity(case, initial, require_native=True)
            nseq += count
        self.assertEqual(nseq, 2400)

    def test_empty_zero_width_and_compensated_quaternions(self):
        for nf in (0, 1, 65):
            channels = [(4, 0, 0, (0, -0.0, -0.0, [0] * nf)),
                        (2, 1, 0, 1.0), (2, 1, 1, 2 ** -27), (2, 1, 2, 2 ** -27),
                        (2, 1, 3, 2 ** -27), (8, 1, 0, None),
                        (0, 2, 0, (2, -0.0, 0)), (1, 3, 0, (-0.0, 0.5, 2))]
            for comp in range(4):
                channels.extend(((7, 4 + comp, comp, None), (2, 4 + comp, 0, -0.0),
                                 (2, 4 + comp, 1, 0.0), (2, 4 + comp, 2, 0.0)))
            for tracks in ([], channels):
                with self.subTest(nf=nf, channels=bool(tracks)):
                    case = encoded_case([(nf, tracks)], 8)
                    self.assert_parity(case, collections.Counter(), require_native=True)

    def test_delta2_with_nonclosing_chunk(self):
        nf = 66
        delta = (10, 0, 0, (0.1, -0.0, [100 + i for i in range(nf)]))
        sequence = bytearray(encode_sequence(nf, [delta]))
        # Change the second chunk start without changing the preceding deltas.
        _, packed, _, _ = struct.unpack_from("<IIff", sequence, 32)
        c1, c2 = packed & 255, (packed >> 8) & 255
        bit = 2 * c1 + c2
        sequence[48 + bit // 8] ^= 1 << (bit % 8)
        case = animation_case([(nf, sequence)], 1)
        _, stats = self.assert_parity(case, collections.Counter(), require_native=True)
        self.assertEqual(dict(stats)["delta2_boundaries"], 1)
        self.assertEqual(dict(stats)["delta2_boundaries_closed"], 0)

    def test_delta2_accumulators_wider_than_i64(self):
        nf = 40
        channels = [(10, 0, 0, (0.1, -0.0, [(1 << 63) + 33 * i * i for i in range(nf)])),
                    (10, 0, 1, (0.1, -0.0, [-(1 << 62) * i - 33 * i * i for i in range(nf)]))]
        self.assert_parity(encoded_case([(nf, channels)], 1), collections.Counter(), require_native=True)

    def test_unsupported_floats_wide_fields_and_mutable_buffer(self):
        raw = [float("nan"), float("inf"), -float("inf"), -0.0]
        payloads = [[(3, 0, 2, raw)],
                    [(4, 0, 0, (73, -0.003, -0.0, [0, 1 << 72, 6, 9]))],
                    [(0, 0, 0, (float("nan"), 0.0, -0.0))]]
        for channels in payloads:
            with self.subTest(channel=channels[0][0]):
                case = encoded_case([(4, channels)], 1)
                with patch.object(anim_clips, "decode_sequence", wraps=anim_clips.decode_sequence) as python_seq:
                    self.assert_parity(case, collections.Counter())
                    self.assertEqual(python_seq.call_count, 2)
        case = encoded_case([(3, [(2, 0, 0, -0.0)])], 1)
        case[0].d = bytearray(case[0].d)
        self.assert_parity(case)

    def test_malformed_sequences_preserve_errors_and_partial_counters(self):
        good = encode_sequence(2, [(2, 0, 0, -0.0)])
        delta = (10, 0, 0, (0.1, 0.0, list(range(66))))
        invalid = [
            (2, good[:12]),
            (2, good[:-4]),
            (2, encode_sequence(2, [(2, 1, 0, 0.0)])),
            (2, encode_sequence(2, [(6, 0, 0, None)])),
            (2, encode_sequence(2, [(7, 0, 1, None), (2, 0, 0, 0.0)])),
            (2, encode_sequence(2, [(8, 0, 3, None), (2, 0, 0, 0.0)])),
            (2, encode_sequence(2, [(5, 0, 0, (2, 16, [4], 1.0, 0.0, [0, 3]))])),
            (66, encode_sequence(66, [delta, (7, 0, 0, None)])),
        ]
        for at, fmt, value in ((12, "<I", 5), (32, "<I", 3)):
            damaged = bytearray(encode_sequence(66, [delta]))
            struct.pack_into(fmt, damaged, at, value)
            invalid.append((66, bytes(damaged)))
        for i, (nf, payload) in enumerate(invalid):
            with self.subTest(i=i):
                case = animation_case([(2, good), (nf, payload)], 1)
                result, _ = self.assert_parity(case, collections.Counter({"unrelated": 8}))
                self.assertIsInstance(result[0], type)

    def test_stats_mapping_behavior(self):
        case = encoded_case([(3, [(2, 0, 0, -0.0)])], 1)
        self.assert_parity(case, {})
        self.assert_parity(case, collections.Counter({"channel:02_static_float": 1e20}))


if __name__ == "__main__":
    unittest.main()
