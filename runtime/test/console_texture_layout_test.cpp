// Portable CPU-only differential tests for leonida::console_texture::Layout::detile and tile
// (runtime/src/console_texture_layout.h). Synthetic textures only; no GPU or game data.
#include "../src/console_texture_layout.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(cond)                                                                                           \
    do {                                                                                                      \
        if (!(cond)) {                                                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                              \
            std::exit(1);                                                                                     \
        }                                                                                                     \
    } while (0)

namespace {

namespace ct = leonida::console_texture;

void fill_pattern(std::vector<uint8_t>& buf, uint32_t seed) {
    uint32_t x = seed ^ 0x9e3779b9u;
    for (size_t i = 0; i < buf.size(); ++i) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        buf[i] = static_cast<uint8_t>(x >> 24);
    }
}

void verify_layout(const ct::Desc& d, size_t extra_row_pad, size_t extra_depth_pad) {
    const auto opt = ct::Layout::make(d);
    CHECK(opt.has_value());
    const ct::Layout& l = *opt;
    CHECK(l.addressable());

    std::vector<uint8_t> tiled(static_cast<size_t>(l.size));
    fill_pattern(tiled, d.width * 31u + d.height * 17u + d.swizzle * 7u + d.element_bytes);

    std::vector<uint8_t> re_tiled(static_cast<size_t>(l.size), 0xCC);
    const uint32_t array_slices = d.dimension == ct::Dimension::tex3d ? 1u : d.depth;

    for (uint32_t slice = 0; slice < array_slices; ++slice) {
        for (uint32_t mip = 0; mip < l.mips.size(); ++mip) {
            const ct::Mip& m = l.mips[mip];
            const size_t row_bytes = size_t{m.width} * d.element_bytes;
            const size_t row_pitch = row_bytes + extra_row_pad;
            const size_t depth_pitch = row_pitch * m.height + extra_depth_pad;
            const size_t linear_len = depth_pitch * m.depth;

            std::vector<uint8_t> linear(linear_len, 0xA5);
            CHECK(l.detile(tiled.data(), tiled.size(), mip, slice, linear.data(), linear.size(), row_pitch,
                           depth_pitch) == ct::Error::ok);

            // Check every element against Layout::offset() AND confirm padding bytes remain untouched.
            for (uint32_t z = 0; z < m.depth; ++z) {
                const uint32_t coord_z = d.dimension == ct::Dimension::tex3d ? z : slice;
                for (uint32_t y = 0; y < m.height; ++y) {
                    const size_t l_row = size_t{z} * depth_pitch + size_t{y} * row_pitch;
                    for (uint32_t x = 0; x < m.width; ++x) {
                        uint64_t t_off = 0;
                        CHECK(l.offset(mip, x, y, coord_z, t_off) == ct::Error::ok);
                        CHECK(std::memcmp(linear.data() + l_row + size_t{x} * d.element_bytes,
                                          tiled.data() + static_cast<size_t>(t_off), d.element_bytes) == 0);
                    }
                    for (size_t p = row_bytes; p < row_pitch; ++p) CHECK(linear[l_row + p] == 0xA5);
                }
                for (size_t p = row_pitch * m.height; p < depth_pitch; ++p)
                    CHECK(linear[size_t{z} * depth_pitch + p] == 0xA5);
            }

            // Round-trip tile() into re_tiled and detile() back to verify bit-exact inverse.
            CHECK(l.tile(re_tiled.data(), re_tiled.size(), mip, slice, linear.data(), linear.size(), row_pitch,
                         depth_pitch) == ct::Error::ok);
            std::vector<uint8_t> back(linear_len, 0xA5);
            CHECK(l.detile(re_tiled.data(), re_tiled.size(), mip, slice, back.data(), back.size(), row_pitch,
                           depth_pitch) == ct::Error::ok);
            CHECK(back == linear);
        }
    }
}

void test_exhaustive_matrix() {
    // 1. All addressable 2D swizzle modes, all element sizes, power-of-two + odd/prime extents + mip tails.
    constexpr uint32_t swizzles[] = {ct::kLinear, ct::kSw256bS, ct::kSw256bD, ct::kSw4kbS,
                                     ct::kSw4kbD, ct::kSw64kbS, ct::kSw64kbD};
    constexpr uint32_t bpes[] = {1, 2, 4, 8, 16};
    struct Size {
        uint32_t w, h, mips, array;
    };
    constexpr Size sizes[] = {
        {1, 1, 1, 1},
        {3, 5, 3, 2},
        {13, 17, 5, 1},
        {37, 29, 6, 2},
        {64, 64, 7, 1},
        {193, 71, 8, 1},
        {517, 33, 3, 1},   // non-power-of-two with tail run + wide enough for many runs
        {1024, 17, 2, 1},  // exceeds stack_runs (128) for narrow run lengths -> exercises heap_runs fallback
    };

    for (uint32_t sw : swizzles) {
        for (uint32_t bpe : bpes) {
            for (const Size& s : sizes) {
                ct::Desc d{};
                d.dimension = ct::Dimension::tex2d;
                d.width = s.w;
                d.height = s.h;
                d.depth = s.array;
                d.mips = s.mips;
                d.element_bytes = bpe;
                d.block = 1;
                d.swizzle = sw;
                // Tight linear packing (exercises contiguous linear fast path) + padded pitches.
                verify_layout(d, 0, 0);
                if (s.w <= 193) verify_layout(d, 7, 13);
            }
        }
    }

    // 2. Block-compressed (4x4 blocks: 8B BC1/BC4 and 16B BC3/BC5/BC6H/BC7), including non-multiple-of-4 extents.
    for (uint32_t sw : swizzles) {
        for (uint32_t bpe : {8u, 16u}) {
            for (const Size& s : {Size{4, 4, 1, 1}, Size{13, 9, 4, 2}, Size{131, 67, 8, 1}, Size{2048, 20, 2, 1}}) {
                ct::Desc d{};
                d.dimension = ct::Dimension::tex2d;
                d.width = s.w;
                d.height = s.h;
                d.depth = s.array;
                d.mips = s.mips;
                d.element_bytes = bpe;
                d.block = 4;
                d.swizzle = sw;
                verify_layout(d, 0, 0);
                if (s.w <= 131) verify_layout(d, 11, 0);
            }
        }
    }

    // 3. 1D textures (including array slices).
    for (uint32_t sw : {ct::kLinear, ct::kSw256bS, ct::kSw4kbS, ct::kSw64kbS}) {
        ct::Desc d{};
        d.dimension = ct::Dimension::tex1d;
        d.width = 515;
        d.height = 1;
        d.depth = 3;
        d.mips = 5;
        d.element_bytes = 4;
        d.swizzle = sw;
        verify_layout(d, 0, 0);
        verify_layout(d, 5, 0);
    }

    // 4. 3D textures (linear + thick S modes: 4KB_S and 64KB_S), including mip tails and odd depth.
    for (uint32_t sw : {ct::kLinear, ct::kSw4kbS, ct::kSw64kbS}) {
        for (uint32_t bpe : {1u, 4u, 16u}) {
            ct::Desc d{};
            d.dimension = ct::Dimension::tex3d;
            d.width = 23;
            d.height = 19;
            d.depth = 13;
            d.mips = 5;
            d.element_bytes = bpe;
            d.swizzle = sw;
            verify_layout(d, 0, 0);
            verify_layout(d, 5, 19);
        }
    }
}

void test_error_paths() {
    ct::Desc d{};
    d.dimension = ct::Dimension::tex2d;
    d.width = 64;
    d.height = 64;
    d.depth = 2;
    d.mips = 3;
    d.element_bytes = 4;
    d.swizzle = ct::kSw64kbS;
    const auto opt = ct::Layout::make(d);
    CHECK(opt.has_value());
    const ct::Layout& l = *opt;
    std::vector<uint8_t> tiled(static_cast<size_t>(l.size));
    std::vector<uint8_t> linear(64 * 64 * 4);

    CHECK(l.detile(tiled.data(), tiled.size(), 3, 0, linear.data(), linear.size(), 256, 0) == ct::Error::invalid);
    CHECK(l.detile(tiled.data(), tiled.size(), 0, 2, linear.data(), linear.size(), 256, 0) == ct::Error::invalid);
    CHECK(l.detile(tiled.data(), tiled.size() - 1, 0, 0, linear.data(), linear.size(), 256, 0) == ct::Error::too_small);
    CHECK(l.detile(tiled.data(), tiled.size(), 0, 0, linear.data(), linear.size() - 1, 256, 0) == ct::Error::too_small);
    CHECK(l.detile(tiled.data(), tiled.size(), 0, 0, linear.data(), linear.size(), 252, 0) == ct::Error::too_small);
    CHECK(l.detile(tiled.data(), tiled.size(), 0, 0, linear.data(), linear.size(), SIZE_MAX, 0) == ct::Error::too_small);
    CHECK(l.tile(tiled.data(), tiled.size(), 0, 0, linear.data(), linear.size(), SIZE_MAX, 0) == ct::Error::too_small);

    // 3D non-zero slice -> invalid; depth_pitch overflow -> too_small.
    ct::Desc d3 = d;
    d3.dimension = ct::Dimension::tex3d;
    d3.width = 8;
    d3.height = 8;
    d3.depth = 4;
    d3.mips = 1;
    const auto l3 = ct::Layout::make(d3);
    CHECK(l3.has_value());
    std::vector<uint8_t> t3(static_cast<size_t>(l3->size));
    std::vector<uint8_t> lin3(8 * 8 * 4 * 4);
    CHECK(l3->detile(t3.data(), t3.size(), 0, 1, lin3.data(), lin3.size(), 32, 256) == ct::Error::invalid);
    CHECK(l3->detile(t3.data(), t3.size(), 0, 0, lin3.data(), lin3.size(), 32, SIZE_MAX) == ct::Error::too_small);

    // Non-addressable XOR mode -> unsupported.
    ct::Desc dx = d;
    dx.swizzle = ct::kSw64kbSX;
    const auto lx = ct::Layout::make(dx);
    CHECK(lx.has_value());
    std::vector<uint8_t> tx(static_cast<size_t>(lx->size));
    CHECK(lx->detile(tx.data(), tx.size(), 0, 0, linear.data(), linear.size(), 256, 0) == ct::Error::unsupported);
    CHECK(lx->tile(tx.data(), tx.size(), 0, 0, linear.data(), linear.size(), 256, 0) == ct::Error::unsupported);
}

void run_bench() {
    struct BenchCase {
        const char* name;
        ct::Desc desc;
    };
    const BenchCase cases[] = {
        {"2048x2048 RGBA8 64KB_S", {ct::Dimension::tex2d, 2048, 2048, 1, 1, 4, 1, 1, ct::kSw64kbS}},
        {"2048x2048 RGBA8 64KB_D", {ct::Dimension::tex2d, 2048, 2048, 1, 1, 4, 1, 1, ct::kSw64kbD}},
        {"2048x2048 BC7 64KB_S", {ct::Dimension::tex2d, 2048, 2048, 1, 1, 16, 4, 1, ct::kSw64kbS}},
        {"1920x1080 RGBA8 64KB_S", {ct::Dimension::tex2d, 1920, 1080, 1, 1, 4, 1, 1, ct::kSw64kbS}},
    };
    for (const auto& c : cases) {
        const auto l = ct::Layout::make(c.desc);
        CHECK(l.has_value());
        std::vector<uint8_t> tiled(static_cast<size_t>(l->size), 0x33);
        const size_t pitch = size_t{l->mips[0].width} * c.desc.element_bytes;
        std::vector<uint8_t> linear(pitch * l->mips[0].height, 0);
        // Warmup
        CHECK(l->detile(tiled.data(), tiled.size(), 0, 0, linear.data(), linear.size(), pitch, 0) == ct::Error::ok);
        constexpr int iters = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            CHECK(l->detile(tiled.data(), tiled.size(), 0, 0, linear.data(), linear.size(), pitch, 0) == ct::Error::ok);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
        const double gbps = (static_cast<double>(linear.size()) / (us * 1e-6)) / 1e9;
        std::printf("bench %-24s: %8.1f us (%5.2f GB/s)\n", c.name, us, gbps);
    }
}

}  // namespace

int main(int argc, char** argv) {
    test_exhaustive_matrix();
    test_error_paths();
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) run_bench();
    std::printf("console_texture_layout_test: PASS\n");
    return 0;
}
