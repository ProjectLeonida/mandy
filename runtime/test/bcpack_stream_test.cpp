// Portable unit and fuzz tests for the BCPack rANS block stream (src/bcpack_stream.h) and decoder (src/bcpack.cpp).
// No game data, no Windows or Vulkan headers.
#include "../src/bcpack.h"
#include "../src/bcpack_stream.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(cond) do {                                             \
    if (!(cond)) {                                                   \
        std::fprintf(stderr, "failed at line %d: %s\n", __LINE__, #cond); \
        return 1;                                                    \
    }                                                                \
} while (false)

namespace {

std::uint32_t scalar_advance(const leonida::bcpack::detail::Model& m, std::uint32_t state, unsigned sym) {
    const int p = std::bit_width(sym);
    std::uint32_t f = m.frequency[p];
    std::uint32_t c = m.base[p] + f * (sym - (p ? 1u << (p - 1) : 0u));
    if (!c) {
        --f;
        ++c;
    }
    return f * (state >> 16) + (state & 65535u) - c;
}

}  // namespace

int main() {
    using namespace leonida::bcpack;
    using namespace leonida::bcpack::detail;

    // 1. Exhaustive bit-exactness across all 128 models: fast symbol() vs scalar symbol(), and
    //    table-driven advance() vs scalar formula for every valid symbol and boundary state.
    for (int w = 1; w <= 8; ++w) {
        for (int q = 0; q < 16; ++q) {
            const Model& m = model_for(w, q);
            for (std::uint32_t slot = 0; slot < 65536u; ++slot) {
                const int fast_sym = m.symbol(slot, true);
                const int slow_sym = m.symbol(slot, false);
                CHECK(fast_sym == slow_sym);
                CHECK(fast_sym >= 0 && fast_sym < (1 << w));
            }
            const unsigned limit = 1u << w;
            for (unsigned sym = 0; sym < limit; ++sym) {
                for (std::uint32_t state : {0x01000001u, 0x01234567u, 0x7fff0000u, 0x7fffffffu, 0xffffffffu}) {
                    CHECK(m.advance(state, sym) == scalar_advance(m, state, sym));
                }
            }
        }
    }

    // 2. Lane::step must reject headers whose width - (byte >> 6) falls outside [1, 8] (preventing negative
    //    indices into models[w - 1] on narrow widths or corrupted/direct-caller inputs).
    for (int lane_width : {0, 1, 2, 3}) {
        for (int shift = 1; shift <= 3; ++shift) {
            if (lane_width - shift >= 1) continue;
            Lane lane{};
            lane.remaining = 1;
            const std::array<std::uint8_t, 4> bad{static_cast<std::uint8_t>(shift << 6), 1, 0, 0};
            std::span<const std::uint8_t> in = bad;
            int budget = 8;
            CHECK(!lane.step(lane_width, in, budget, true));
        }
    }
    // Width > 8 with shift 0 also fails cleanly instead of indexing past models[7].
    {
        Lane lane{};
        lane.remaining = 1;
        const std::array<std::uint8_t, 4> bad{0x00, 1, 0, 0};
        std::span<const std::uint8_t> in = bad;
        int budget = 8;
        CHECK(!lane.step(9, in, budget, true));
    }

    // 3. visit_blocks must reject invalid codec IDs and unaligned output sizes cleanly before indexing widths[].
    {
        const std::array<std::uint8_t, 16> raw{};
        auto noop = [](std::size_t, const Symbols&, const Symbols&) { return true; };
        for (unsigned bad_codec : {0u, 2u, 8u, 9u, 255u, 0xffffffffu}) {
            CHECK(!visit_blocks(bad_codec, raw, 16, noop));
            CHECK(!visit_blocks(bad_codec, raw, 0, noop));
        }
        // Unaligned sizes: single-element codecs (1, 4) require 8-byte alignment; pair codecs (3, 5, 6, 7) require 16.
        CHECK(!visit_blocks(1, raw, 4, noop));
        CHECK(!visit_blocks(4, raw, 12, noop));
        CHECK(!visit_blocks(3, raw, 8, noop));
        CHECK(!visit_blocks(5, raw, 8, noop));
        CHECK(!visit_blocks(6, raw, 8, noop));
        CHECK(!visit_blocks(7, raw, 24, noop));
        // Valid empty decode with zero-padded trailing bytes succeeds.
        for (unsigned good_codec : {1u, 3u, 4u, 5u, 6u, 7u})
            CHECK(visit_blocks(good_codec, raw, 0, noop));
    }

    // 4. Adversarial deterministic fuzzing across all 6 codecs with both fast=true and fast=false:
    //    must never crash, read out of bounds, or disagree between fast and scalar entropy paths.
    {
        std::uint64_t rng = 0x9e3779b97f4a7c15ull;
        const auto next_u8 = [&] {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            return static_cast<std::uint8_t>((rng * 0x2545f4914f6cdd1dull) >> 56);
        };
        const unsigned codecs[] = {1u, 3u, 4u, 5u, 6u, 7u};
        std::vector<std::uint8_t> buf, out_fast(64), out_slow(64);
        for (unsigned iter = 0; iter < 1200; ++iter) {
            const unsigned codec = codecs[iter % 6];
            const std::size_t len = 1 + (iter % 96);
            buf.resize(len);
            for (std::uint8_t& b : buf) b = next_u8();
            const std::size_t out_sz = (codec == 1 || codec == 4) ? 32 : 64;
            const auto q_fast = decode(codec, buf, std::span<std::uint8_t>(out_fast.data(), out_sz),
                                       false, nullptr, nullptr, true);
            const auto q_slow = decode(codec, buf, std::span<std::uint8_t>(out_slow.data(), out_sz),
                                       false, nullptr, nullptr, false);
            CHECK(q_fast == q_slow);
        }
    }

    std::puts("bcpack_stream: exhaustive model/advance parity, width/codec/size guards, and fuzzing OK");
    return 0;
}
