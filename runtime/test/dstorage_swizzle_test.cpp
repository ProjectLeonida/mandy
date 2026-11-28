// Pure-CPU unit tests for runtime/src/dstorage_swizzle.h (DirectStorage macro-unshuffle):
//   - exact byte-for-byte parity with the stripe-major reference across all 4 swizzle modes and multiple group counts;
//   - empty buffer handling, invalid mode rejection, misaligned/mismatched size rejection;
//   - overlapping input/output span rejection (in-place and partial overlaps).
// No Windows headers, Vulkan, or game files required.
#include "../src/dstorage_swizzle.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

bool reference_unshuffle(unsigned mode, std::span<const uint8_t> input, std::span<uint8_t> out) {
    using namespace leonida::dstorage;
    if (!mode || mode > 4 || input.size() != out.size() || input.size() % group_size(mode)) return false;
    const size_t group = group_size(mode), groups = input.size() / group;
    size_t source = 0, offset = 0;
    for (unsigned pages : stripes[mode]) {
        if (!pages) break;
        const size_t bytes = size_t(pages) * 4096;
        for (size_t i = 0; i < groups; ++i) {
            std::memcpy(out.data() + i * group + offset, input.data() + source, bytes);
            source += bytes;
        }
        offset += bytes;
    }
    return true;
}

}  // namespace

int main() {
    using namespace leonida::dstorage;

    // 1. Invalid modes, mismatched sizes, and partial-group sizes are rejected.
    {
        std::vector<uint8_t> a(32768, 0x11), b(32768, 0x22);
        CHECK(!unshuffle(0, a, b));
        CHECK(!unshuffle(5, a, b));
        CHECK(!unshuffle(0xffffffffu, a, b));
        CHECK(!unshuffle(1, std::span(a).first(16384), b));
        CHECK(!unshuffle(1, std::span(a).first(16384 - 4096), std::span(b).first(16384 - 4096)));
        CHECK(!unshuffle(2, std::span(a).first(16384), std::span(b).first(16384)));
        CHECK(!unshuffle(3, std::span(a).first(4096), std::span(b).first(4096)));
        CHECK(!unshuffle(4, std::span(a).first(28672), std::span(b).first(28672)));
    }

    // 2. Empty spans succeed for valid modes and fail for invalid modes.
    {
        std::span<const uint8_t> empty_in;
        std::span<uint8_t> empty_out;
        for (unsigned mode = 1; mode <= 4; ++mode) CHECK(unshuffle(mode, empty_in, empty_out));
        CHECK(!unshuffle(0, empty_in, empty_out));
        CHECK(!unshuffle(5, empty_in, empty_out));
    }

    // 3. Exact parity against stripe-major reference for all 4 modes and 1, 3, and 17 groups.
    for (unsigned mode = 1; mode <= 4; ++mode) {
        const size_t gsz = group_size(mode);
        for (size_t groups : {size_t(1), size_t(3), size_t(17)}) {
            const size_t total = gsz * groups;
            std::vector<uint8_t> input(total);
            for (size_t page = 0; page < total / 4096; ++page) {
                const uint8_t tag = static_cast<uint8_t>((page * 37 + mode * 13 + groups) & 0xffu);
                std::memset(input.data() + page * 4096, tag, 4096);
                input[page * 4096] = static_cast<uint8_t>(page & 0xffu);
                input[page * 4096 + 1] = static_cast<uint8_t>((page >> 8) & 0xffu);
                input[page * 4096 + 4095] = static_cast<uint8_t>(~tag);
            }
            std::vector<uint8_t> got(total, 0xcd), want(total, 0xcd);
            CHECK(reference_unshuffle(mode, input, want));
            CHECK(unshuffle(mode, input, got));
            CHECK(got == want);
        }
    }

    // 4. Overlapping spans (identical, input-before-out, out-before-input) are refused without UB;
    //    adjacent non-overlapping halves of one buffer succeed cleanly.
    {
        std::vector<uint8_t> buf(65536, 0x5a);
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<uint8_t>((i * 17 + 3) & 0xffu);
        const auto snapshot = buf;
        CHECK(!unshuffle(2, std::span(buf).subspan(0, 32768), std::span(buf).subspan(0, 32768)));
        CHECK(!unshuffle(2, std::span(buf).subspan(0, 32768), std::span(buf).subspan(4096, 32768)));
        CHECK(!unshuffle(2, std::span(buf).subspan(4096, 32768), std::span(buf).subspan(0, 32768)));
        CHECK(!unshuffle(1, std::span(buf).subspan(0, 16384), std::span(buf).subspan(16383, 16384)));
        CHECK(buf == snapshot);  // refused calls leave memory untouched

        // Adjacent non-overlapping slices [0, 32768) and [32768, 65536) succeed and match reference.
        std::vector<uint8_t> want(32768);
        CHECK(reference_unshuffle(2, std::span(buf).subspan(0, 32768), want));
        CHECK(unshuffle(2, std::span(buf).subspan(0, 32768), std::span(buf).subspan(32768, 32768)));
        CHECK(std::memcmp(buf.data() + 32768, want.data(), 32768) == 0);
    }

    std::puts("dstorage_swizzle_test: all checks passed");
    return 0;
}
