"""tools/image_fuzz.py --oracle hw without a GPU: the T# words per GPU family (exact dwords), the texture layout, the
native kernel text, the model's dwords, the negative control and the exit-77 contract. The last test runs on the
hardware when the oracle tools are present and is skipped otherwise.
"""
import json
import os
import random
import struct
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import image_fuzz as imf  # noqa: E402
from hw_oracle import asm, runner, tsharp  # noqa: E402

GFX11, GFX10, GFX101 = "gfx1150", "gfx1030", "gfx1010"


def test_t_sharp_gfx11_exact_words():
    # 16x8 R8G8B8A8_UNORM, linear: FORMAT 42 at [27:20], WIDTH_LO = 15 & 3 at [31:30], WIDTH_HI = 15 >> 2,
    # HEIGHT = 7 at [27:14], DST_SEL 4 5 6 7, TYPE 9 (2D) at [31:28], pitch 64 texels (256 bytes) - 1 in DEPTH.
    assert tsharp.image_words(GFX11, "R8G8B8A8_UNORM", 16, 8, 4) == [
        0, 0xC2A00000, 0x0001C003, 0x90000FAC, 0x3F, 0x00400000, 0, 0]


def test_t_sharp_gfx10_exact_words():
    # The same texture on GFX10.3: FORMAT 56 at [28:20], RESOURCE_LEVEL (bit 31 of word 2) set, same pitch field.
    assert tsharp.image_words(GFX10, "R8G8B8A8_UNORM", 16, 8, 4) == [
        0, 0xC3800000, 0x8001C003, 0x90000FAC, 0x3F, 0x00400000, 0, 0]


def test_t_sharp_gfx101_has_no_pitch_field():
    assert tsharp.image_words(GFX101, "R8G8B8A8_UNORM", 16, 8, 4)[4] == 0
    assert not tsharp.has_custom_pitch(GFX101) and tsharp.has_custom_pitch(GFX10) and tsharp.has_custom_pitch(GFX11)


def test_t_sharp_wide_texture_and_one_channel_swizzle():
    words = tsharp.image_words(GFX11, "R32_SFLOAT", 64, 32, 4, (4, 0, 0, 1))
    assert words[1] == (22 << 20) | (3 << 30)                  # width 64: WIDTH_LO = 63 & 3
    assert words[2] == 15 | (31 << 14)                         # WIDTH_HI = 63 >> 2
    assert words[3] == 0x90000204                              # DST_SEL x,0,0,1 and the 2D type
    assert words[4] == 63                                      # 64 texels * 4 bytes = 256 bytes: pitch 64
    words = tsharp.image_words(GFX11, "R32G32B32A32_SFLOAT", 8, 8, 16)
    assert words[1] == (63 << 20) | (3 << 30) and words[4] == 15   # 8 * 16 = 128 bytes pads to 256: pitch 16


def test_t_sharp_rejects_what_it_cannot_describe():
    with pytest.raises(ValueError):
        tsharp.image_words("gfx1201", "R8G8B8A8_UNORM", 8, 8, 4)    # GFX12 is another layout
    with pytest.raises(ValueError):
        tsharp.image_words(GFX11, "R8G8B8A8_UNORM", 0, 8, 4)
    with pytest.raises(KeyError):
        tsharp.image_words(GFX11, "D32_SFLOAT", 8, 8, 4)


def test_format_tables_agree_with_the_translator_table_and_each_other():
    with open(os.path.join(imf.TOOLS, "gfx10_formats.json"), encoding="utf-8") as f:
        by_name = {name: int(code) for code, name in json.load(f).items()}
    names = {"R8G8B8A8_UNORM": "8_8_8_8_UNORM", "R16G16B16A16_SFLOAT": "16_16_16_16_FLOAT",
             "R32G32B32A32_SFLOAT": "32_32_32_32_FLOAT", "R32_SFLOAT": "32_FLOAT",
             "R8G8B8A8_UINT": "8_8_8_8_UINT", "R8G8B8A8_SINT": "8_8_8_8_SINT",
             "R8_UINT": "8_UINT", "R8_SINT": "8_SINT", "R16_UINT": "16_UINT", "R16_SINT": "16_SINT",
             "R8G8_UINT": "8_8_UINT", "R8G8_SINT": "8_8_SINT", "R16G16_SINT": "16_16_SINT"}
    assert set(tsharp.IMG_FORMATS["gfx10"]) == set(tsharp.IMG_FORMATS["gfx11"]) == set(imf.HW_FORMATS) == set(names)
    assert set(imf.PCK_FORMATS) == set(names) - set(imf.FORMATS)
    for fmt, short in names.items():
        assert tsharp.IMG_FORMATS["gfx10"][fmt] == by_name["GFX10_FORMAT_" + short] == imf.HW_FORMATS[fmt][1]
    # GFX11 numbers the formats in LLVM's UfmtGFX11 order: no 10_10_10_2-style gaps filled differently from GFX10
    assert len(set(tsharp.IMG_FORMATS["gfx11"].values())) == len(imf.HW_FORMATS)


def test_pitch_and_layout():
    assert [tsharp.pitch_texels(w, 4) for w in (8, 16, 32, 64)] == [64, 64, 64, 64]
    assert [tsharp.pitch_texels(w, 16) for w in (8, 16, 32, 64)] == [16, 16, 32, 64]
    assert tsharp.pitch_texels(64, 8) == 64
    tight = bytes(range(32))                      # 8 x 1 RGBA8 texels
    laid = tsharp.lay_out(tight, 8, 1, 4)
    assert laid == tight + b"\xcd" * 224
    assert len(tsharp.lay_out(bytes(8 * 4 * 3), 8, 3, 4)) == 3 * 256


def test_lanes():
    assert tsharp.lanes(8, 8) == (64, 1)
    assert tsharp.lanes(32, 32) == (1024, 1)
    assert tsharp.lanes(64, 64) == (1024, 4)


def case(fmt="R8G8B8A8_UNORM", width=8, height=8, dmask=0xf, wave=32, index=0, seed=5):
    data = imf.make_texels(fmt, width, height, random.Random(seed))
    return imf.Case(index, fmt, width, height, dmask, data, wave)


def test_kernel_text_per_family():
    c = case(width=64, height=64, dmask=0x5)
    for arch in (GFX11, GFX10):
        k = imf.hw_kernel(arch, c)
        text = "\n".join(k.lines)
        assert k.threads == 1024 and k.words == 64 * 64 * 4 and k.raw and k.wave == 32
        assert "image_load v[8:9], v[2:3], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D unorm" in text
        assert text.count("image_load") == 4                      # 4096 texels over 1024 lanes
        assert "s_getpc_b64 s[4:5]" in text and ".p2align 8" in text and ".Ltex0:" in text
        assert text.index("s_endpgm") < text.index(".Ltex0:")      # the texture sits behind the code
        full = asm.raw_kernel_text("k0", k.lines, arch)
        assert ("global_store_b128 v4, v[8:11], s[2:3]" in full) == (arch == GFX11)
        assert ("global_store_dwordx4 v4, v[8:11], s[2:3]" in full) == (arch == GFX10)
    one = imf.hw_kernel(GFX11, case(dmask=0x2))
    assert "image_load v8, v[2:3]" in "\n".join(one.lines)


def test_kernel_data_is_the_laid_out_texture():
    c = case(width=8, height=2)
    lines = imf.hw_kernel(GFX11, c).lines
    start = lines.index(".Ltex0:") + 1
    data = b"".join(int(x, 16).to_bytes(4, "little") for ln in lines[start:] for x in ln.split(".long")[1].split(","))
    assert data == tsharp.lay_out(c.data[0], 8, 2, 4)


def test_kernel_loads_the_t_sharp_the_builder_made():
    c = case(fmt="R16G16B16A16_SFLOAT", width=16, height=8)
    text = "\n".join(imf.hw_kernel(GFX11, c).lines)
    words = tsharp.image_words(GFX11, c.fmt, 16, 8, 8)
    assert f"s_or_b32 s9, s9, 0x{words[1]:08x}" in text
    for i in range(2, 8):
        assert f"s_mov_b32 s{8 + i}, 0x{words[i]:08x}" in text


def test_expected_words_pack_dmask_and_keep_sentinels():
    c = case(fmt="R32G32B32A32_SFLOAT", width=8, height=8, dmask=0x5)
    want = imf.expected_words(c.fmt, c.data[0], 8, 8, 0x5)
    assert len(want) == 8 * 8 * 4
    texel = struct.unpack_from("<4f", c.data[0], 0)
    first = [struct.unpack("<f", struct.pack("<I", w))[0] for w in want[:4]]
    assert (first[0] == texel[0] or texel[0] != texel[0]) and first[2:] == [3.0, 4.0]


def test_expected_words_integers_are_raw_and_sentinels_are_float_bits():
    texels = bytes([0, 127, 128, 255]) * 64
    got = imf.expected_words("R8G8B8A8_SINT", texels, 8, 8, 0x3)
    assert got[:4] == [0, 127, 0x40400000, 0x40800000]
    got = imf.expected_words("R8G8B8A8_SINT", texels, 8, 8, 0xc)
    assert got[:4] == [0xFFFFFF80, 0xFFFFFFFF, 0x40400000, 0x40800000]


def test_integer_words_do_not_get_nan_leniency():
    assert imf.word_equal(0x7FC00000, 0x7FC00001, "float4")
    assert not imf.word_equal(0x7FC00000, 0x7FC00001, "sint8")
    assert not imf.word_equal(0x7FC00000, 0, "float4")
    assert not imf.word_equal(0x80000000, 0, "float4")           # -0 is not +0


def test_perturbations_change_the_model():
    for name in imf.PERTURBATIONS:
        changed = False
        for op in imf.PERTURB_OPS[name]:
            formats = imf.FORMATS if op == "image_load" else imf.HW_FORMATS
            for fmt in formats:
                for dmask in (0xf, 0x5):
                    c = case(fmt=fmt, width=16, height=8, dmask=dmask)
                    base = imf.expected_words(fmt, c.data[0], 16, 8, dmask, op)
                    imf.PERTURB.add(name)
                    try:
                        changed = changed or imf.expected_words(fmt, c.data[0], 16, 8, dmask, op) != base
                    finally:
                        imf.PERTURB.discard(name)
        assert changed, name
    assert not imf.PERTURB
    assert set(imf.PERTURB_OPS) == set(imf.PERTURBATIONS)


class ModelOracle:
    """Stands in for the GPU: answers every kernel with the words the model expects (or a damaged copy)."""
    arch = GFX11
    device = "model"
    chunk = 0

    def __init__(self, cases, damage=None, op="image_load"):
        self.cases, self.damage, self.op = cases, damage, op

    def run(self, kernels, words=4):
        out = []
        for c, k in zip(self.cases, kernels):
            w = imf.expected_words(c.fmt, c.data[0], c.width, c.height, c.dmask, self.op)
            right = tsharp.image_words(self.arch, c.fmt, c.width, c.height, imf.HW_FORMATS[c.fmt][2])[1]
            if f"0x{right:08x}" not in "\n".join(k.lines):      # a kernel built with another FORMAT reads other bits
                w = [x ^ 0x00400000 for x in w]
            if self.damage:
                w = self.damage(c, w)
            out.append(w)
        return out

    def close(self):
        pass


class Args:
    seed, max_report = 1, 3


def test_run_hw_accepts_the_model_and_reports_a_flipped_word(capsys):
    cases = imf.make_cases(12, 1, list(imf.FORMATS))
    stats, failed, _ = imf.run_hw(ModelOracle(cases), cases, Args())
    assert not failed and sum(s["match"] for s in stats.values()) == 12

    def flip(c, w):
        w = list(w)
        if c.index == 3:
            w[5] ^= 0x1
        return w
    stats, failed, _ = imf.run_hw(ModelOracle(cases, flip), cases, Args())
    assert [c.index for c in failed] == [3]
    out = capsys.readouterr().out
    assert "MISMATCH #3" in out and "texel (1, 0) register 1" in out


def test_run_hw_names_a_misaligned_texture(capsys):
    cases = imf.make_cases(2, 1, ["R8G8B8A8_UNORM"])
    stats, failed, _ = imf.run_hw(ModelOracle(cases, lambda c, w: [tsharp.POISON] * len(w)), cases, Args())
    assert sum(s["rejected"] for s in stats.values()) == 2 and not failed
    assert "not 256-byte aligned" in capsys.readouterr().out


@pytest.mark.parametrize("programs,seed", [(8, 1), (120, 2)])
def test_negative_control_passes_on_a_hardware_that_follows_the_model(capsys, programs, seed):
    cases = imf.make_cases(programs, seed, list(imf.FORMATS))
    assert imf.negative_control(ModelOracle(cases), cases, Args()) == 0
    out = capsys.readouterr().out
    assert f"unperturbed model: 0 of {programs} cases mismatch" in out
    for name in imf.PERTURBATIONS:
        assert f"perturbation {name}" in out
    assert out.count("wrong T# FORMAT") == len(imf.FORMATS) and "NOT CAUGHT" not in out


def test_negative_control_fails_when_the_hardware_disagrees(capsys):
    cases = imf.make_cases(30, 2, list(imf.FORMATS))
    assert imf.negative_control(ModelOracle(cases, lambda c, w: [x ^ 1 for x in w]), cases, Args()) == 1
    assert "unperturbed model: 30 of 30 cases mismatch" in capsys.readouterr().out


def test_cases_are_the_same_for_every_oracle_and_seed():
    a, b = imf.make_cases(30, 7, list(imf.FORMATS)), imf.make_cases(30, 7, list(imf.FORMATS))
    assert [(c.fmt, c.width, c.height, c.dmask, c.wave, c.data) for c in a] == \
           [(c.fmt, c.width, c.height, c.dmask, c.wave, c.data) for c in b]
    assert {c.wave for c in a} == {32, 64}
    assert {c.dmask for c in imf.make_cases(180, 1, list(imf.FORMATS))} == set(range(1, 16))


def test_main_exits_77_when_the_hardware_oracle_is_missing(monkeypatch, capsys):
    def missing(**_kw):
        raise runner.Unavailable("no AMD clang")
    monkeypatch.setattr(runner, "open_oracle", missing)
    monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--oracle", "hw", "--programs", "2"])
    assert imf.main() == 77
    assert "hardware oracle unavailable: no AMD clang" in capsys.readouterr().out


def test_main_exits_77_for_a_family_without_a_t_sharp(monkeypatch, capsys):
    class Gfx12(ModelOracle):
        arch = "gfx1201"
    monkeypatch.setattr(runner, "open_oracle", lambda **_kw: Gfx12([]))
    monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--oracle", "hw", "--programs", "2"])
    assert imf.main() == 77
    assert "no T# layout for gfx12" in capsys.readouterr().out


def test_options_that_need_the_hardware_oracle(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--negative-control"])
    with pytest.raises(SystemExit) as e:
        imf.main()
    assert e.value.code == 2
    monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--oracle", "gpu"])
    with pytest.raises(SystemExit):
        imf.main()


def test_on_the_hardware_when_present(monkeypatch, capsys):
    try:
        oracle = runner.open_oracle()
        tsharp.check_arch(oracle.arch)
    except (runner.Unavailable, ValueError) as e:
        pytest.skip(f"hardware oracle unavailable: {e}")
    oracle.close()
    monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--oracle", "hw", "--programs", "36", "--seed", "1"])
    assert imf.main() == 0
    assert "mismatch 0" in capsys.readouterr().out


def test_pck_oracle_measured_values():
    """The texel-fetch rule measured on gfx1036 (docs/shader_validation.md): the texel's bytes as raw dwords."""
    s = list(tsharp.SENTINEL_BITS)
    # R8G8B8A8_*: one dword, the four bytes little-endian; every dmask bit past the first selects 0.
    texel = bytes([0x81, 0x7f, 0xff, 0x02])
    for fmt in ("R8G8B8A8_UNORM", "R8G8B8A8_UINT", "R8G8B8A8_SINT"):
        assert imf.pck_oracle(fmt, texel, 0xf, False) == [0x02ff7f81, 0, 0, 0]
        assert imf.pck_oracle(fmt, texel, 0xf, True) == [0x02ff7f81, 0, 0, 0]
    assert imf.pck_oracle("R8G8B8A8_UNORM", texel, 0x1, False) == [0x02ff7f81] + s[1:]
    assert imf.pck_oracle("R8G8B8A8_UNORM", texel, 0x2, False) == [0] + s[1:]
    assert imf.pck_oracle("R8G8B8A8_UNORM", texel, 0x3, False) == [0x02ff7f81, 0] + s[2:]
    assert imf.pck_oracle("R8G8B8A8_UNORM", texel, 0x5, False) == [0x02ff7f81, 0] + s[2:]
    assert imf.pck_oracle("R8G8B8A8_UNORM", texel, 0x8, False) == [0] + s[1:]
    assert imf.pck_oracle("R8G8B8A8_UNORM", texel, 0xc, False) == [0, 0] + s[2:]
    assert imf.pck_oracle("R8G8B8A8_UNORM", texel, 0xe, False) == [0, 0, 0] + s[3:]
    # R16G16B16A16_SFLOAT: two dwords.
    texel = bytes([0x01, 0x80, 0xff, 0xfb, 0x34, 0x12, 0x00, 0x80])
    for sgn in (False, True):
        assert imf.pck_oracle("R16G16B16A16_SFLOAT", texel, 0xf, sgn) == [0xfbff8001, 0x80001234, 0, 0]
    assert imf.pck_oracle("R16G16B16A16_SFLOAT", texel, 0x2, False) == [0x80001234] + s[1:]
    assert imf.pck_oracle("R16G16B16A16_SFLOAT", texel, 0x5, False) == [0xfbff8001, 0] + s[2:]
    assert imf.pck_oracle("R16G16B16A16_SFLOAT", texel, 0xe, False) == [0x80001234, 0, 0] + s[3:]
    # R32G32B32A32_SFLOAT: the four texel dwords as-is; R32_SFLOAT's missing components are 0 (not the
    # (r, 0, 0, 1) fill image_load gives).
    texel = bytes(range(0xf0, 0x100))
    assert imf.pck_oracle("R32G32B32A32_SFLOAT", texel, 0xf, False) == [0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc]
    assert imf.pck_oracle("R32_SFLOAT", bytes([0x81, 0x7f, 0xff, 0x02]), 0xf, False) == [0x02ff7f81, 0, 0, 0]
    # Texels under 4 bytes: one dword, zero-extended for pck, sign-extended from the texel's top bit for
    # _sgn (the sign of the whole texel, not per component).
    assert imf.pck_oracle("R8_UINT", bytes([0x81]), 0x1, False) == [0x81] + s[1:]
    assert imf.pck_oracle("R8_UINT", bytes([0x81]), 0x1, True) == [0xFFFFFF81] + s[1:]
    assert imf.pck_oracle("R16_UINT", bytes([0x01, 0x80]), 0x1, False) == [0x8001] + s[1:]
    assert imf.pck_oracle("R16_UINT", bytes([0x01, 0x80]), 0x1, True) == [0xFFFF8001] + s[1:]
    assert imf.pck_oracle("R8G8_UINT", bytes([0x81, 0xf2]), 0x1, False) == [0xF281] + s[1:]
    assert imf.pck_oracle("R8G8_UINT", bytes([0x81, 0xf2]), 0x1, True) == [0xFFFFF281] + s[1:]
    assert imf.pck_oracle("R16G16_SINT", bytes([0x01, 0x80, 0xfe, 0x7f]), 0x1, False) == [0x7FFE8001] + s[1:]
    assert imf.pck_oracle("R16G16_SINT", bytes([0x01, 0x80, 0xfe, 0x7f]), 0x1, True) == [0x7FFE8001] + s[1:]
    # dmask 0 selects nothing: every register keeps its sentinel.
    assert imf.pck_oracle("R8_UINT", bytes([0x81]), 0x0, True) == s


def test_pck_oracle_words_for_every_texel():
    """Four words per texel, in texel order, for the raw formats too."""
    texels, _ = imf.make_texels("R8G8_UINT", 4, 2, random.Random(9))
    words = imf.pck_oracle("R8G8_UINT", texels, 0xf, False)
    assert len(words) == 4 * 2 * 4
    for i in range(8):
        chunk = texels[i * 2:i * 2 + 2]
        assert words[i * 4] == int.from_bytes(chunk, "little")
        assert words[i * 4 + 1:i * 4 + 4] == [0, 0, 0]


def test_kernel_pck_ops_replace_the_opcode_and_drop_unorm():
    c = case(fmt="R8_UINT", width=8, height=8, dmask=0x5)
    for op in ("image_load_pck", "image_load_pck_sgn"):
        for arch in (GFX11, GFX10):
            text = "\n".join(imf.hw_kernel(arch, c, op=op).lines)
            assert f"{op} v[8:9], v[2:3], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D\n" in text + "\n"
            assert " unorm" not in text and "image_load " not in text
    text = "\n".join(imf.hw_kernel(GFX11, c).lines)
    assert "image_load v[8:9], v[2:3], s[8:15] dmask:0x5 dim:SQ_RSRC_IMG_2D unorm" in text


def test_expected_words_pck_ops_and_strict_compare():
    c = case(fmt="R8_UINT", width=8, height=8, dmask=0x1)
    want = imf.expected_words(c.fmt, c.data[0], 8, 8, 0x1, "image_load_pck_sgn")
    got = list(want)
    got[0] ^= 1
    assert imf.compare_hw(c, got, "image_load_pck_sgn")
    # Bit-exact even where the bits look like a NaN: the pck model has no NaN leniency.
    texels = bytes([0x00, 0x00, 0xc0, 0x7f]) * 64
    want = imf.expected_words("R32_SFLOAT", texels, 8, 8, 0xf, "image_load_pck")
    got = [w ^ 0x00000001 if i % 4 == 0 else w for i, w in enumerate(want)]
    c = imf.Case(0, "R32_SFLOAT", 8, 8, 0xf, (texels, None), 32)
    assert imf.compare_hw(c, got, "image_load_pck")


def test_negative_control_catches_a_wrong_pck_model(capsys):
    cases = imf.make_cases(60, 1, list(imf.HW_FORMATS))
    a = Args()
    a.op = "image_load_pck_sgn"
    assert imf.negative_control(ModelOracle(cases, op=a.op), cases, a) == 0
    out = capsys.readouterr().out
    assert "unperturbed model: 0 of 60 cases mismatch" in out
    assert "perturbation pck_sgn_zero_extend" in out and "NOT CAUGHT" not in out
    assert "perturbation unorm_256" in out and "not applicable" in out


def test_main_skips_the_translator_oracle_for_pck_ops(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["image_fuzz.py", "--oracle", "translator", "--op", "image_load_pck"])
    assert imf.main() == 77
    assert "not translated" in capsys.readouterr().out


def test_main_rejects_pck_only_formats_for_image_load():
    result = subprocess.run([sys.executable, "-I", imf.__file__, "--formats", "R8_UINT", "--programs", "1"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2, result.stderr
    assert "only the pck ops model these formats" in result.stderr


@pytest.mark.parametrize("arch,wave", [("gfx1030", 32), ("gfx1030", 64), ("gfx1150", 32), ("gfx1150", 64)])
def test_kernels_assemble_for_both_families(arch, wave, tmp_path):
    """Every format and dmask of the kernel text assembles with AMD clang for the family (no GPU needed)."""
    import subprocess
    try:
        clang = runner.find_clang()
    except runner.Unavailable as e:
        pytest.skip(str(e))
    kernels = []
    for i, fmt in enumerate(imf.FORMATS):
        for dmask in (0x1, 0x5, 0xf):
            c = case(fmt=fmt, width=16, height=8, dmask=dmask, wave=wave, index=len(kernels))
            kernels.append(imf.hw_kernel(arch, c))
    text, names, _ = asm.module_text(kernels, arch, wave=wave)
    src = tmp_path / "k.s"
    src.write_text(text, encoding="utf-8", newline="\n")
    flags = ["-mwavefrontsize64"] if wave == 64 else []
    r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", f"-mcpu={arch}", *flags, "-c", str(src),
                        "-o", str(tmp_path / "k.o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:600]
    assert len(names) == 18
