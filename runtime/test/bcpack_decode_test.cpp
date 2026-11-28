// Portable CPU-only unit tests for runtime/src/bcpack.cpp (RT-BCPACK-DECODE-FAST-BITWRITER-AND-BPTC):
// verifies bit-exact decode across all 6 BCPack codecs (BC1, BC3, BC4, BC5, BC6H, BC7), calibration history
// profiles, fitted-predictor gating, and malformed input rejection against generated synthetic vectors.
// No Windows headers, no Vulkan, no game files.
#include "../src/bcpack.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace {

struct Vector {
    unsigned codec;
    std::span<const uint8_t> raw, plain, wrapped;
};

#include "dstorage_vectors.inc"

leonida::bcpack::Calibration test_profile() {
    leonida::bcpack::Calibration history{};
    auto& six = history.bc6[255];
    for (unsigned c = 0; c < 3; ++c) six.ep[8 * c + 1] = 16;
    six.ep_class = six.ix_class = 3;
    six.ep_confirmed.fill(16);
    six.ix_confirmed.fill(4);
    auto& seven = history.bc7[255];
    seven.ep.fill(64);
    seven.ep_bits.fill(8);
    seven.ep_confirmed.fill(8);
    seven.ix_bits.fill(4);
    seven.ix_confirmed.fill(4);
    return history;
}

}  // namespace

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);   \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main() {
    using leonida::bcpack::Quality;
    using leonida::bcpack::decode;

    // 1. All 6 codecs (BC1, BC3, BC4, BC5, BC6H, BC7) bit-exact with both fast and reference entropy.
    for (const Vector& v : vectors) {
        for (bool fast : {true, false}) {
            std::vector<uint8_t> out(v.plain.size(), 0xA5);
            const char* failure = nullptr;
            const Quality q = decode(v.codec, v.raw, out, false, nullptr, &failure, fast);
            CHECK(q == Quality::confirmed);
            CHECK(std::memcmp(out.data(), v.plain.data(), v.plain.size()) == 0);
        }
    }

    // 2. BPTC (BC6H / BC7) calibration profile history: missing-profile -> confirmed -> unconfirmed -> fitted.
    {
        const auto full_history = test_profile();
        struct ProfileCase {
            unsigned codec;
            std::span<const uint8_t> raw;
            std::span<const uint8_t> first;
        };
        const ProfileCase cases[] = {
            {6, profile_raw6, profile_first6},
            {7, profile_raw7, profile_first7},
        };
        for (const auto& c : cases) {
            std::vector<uint8_t> out(512, 0xCC);
            const char* failure = nullptr;
            CHECK(decode(c.codec, c.raw, out, false, nullptr, &failure) == Quality::failed);
            CHECK(failure != nullptr && std::strcmp(failure, "missing-profile") == 0);

            CHECK(decode(c.codec, c.raw, out, false, &full_history, &failure) == Quality::confirmed);
            for (std::size_t off = 0; off < out.size(); off += 16) {
                CHECK(std::memcmp(out.data() + off, c.first.data(), 16) == 0);
            }

            // Unconfirmed calibration predictor: refused when allow_fitted=false, fitted when allow_fitted=true.
            auto partial = full_history;
            if (c.codec == 6) {
                partial.bc6[255].ep_confirmed.fill(0);
            } else {
                partial.bc7[255].ep_confirmed.fill(0);
            }
            std::fill(out.begin(), out.end(), uint8_t{0xDD});
            CHECK(decode(c.codec, c.raw, out, false, &partial, &failure) == Quality::failed);
            CHECK(failure != nullptr && std::strcmp(failure, "unconfirmed-predictor") == 0);
            CHECK(decode(c.codec, c.raw, out, true, &partial, &failure) == Quality::fitted);
            for (std::size_t off = 0; off < out.size(); off += 16) {
                CHECK(std::memcmp(out.data() + off, c.first.data(), 16) == 0);
            }
        }
    }

    // 3. Classic BC1 fitted-history gating.
    {
        std::vector<uint8_t> out(512, 0xEE);
        CHECK(decode(1, fitted_raw, out, false) == Quality::failed);
        CHECK(decode(1, fitted_raw, out, true) == Quality::fitted);
        CHECK(std::memcmp(out.data(), fitted_first, sizeof(fitted_first)) == 0);
    }

    // 4. Invalid codecs, misaligned output sizes, and truncated streams fail cleanly.
    {
        std::vector<uint8_t> out(64, 0);
        CHECK(decode(0, raw1, out) == Quality::failed);
        CHECK(decode(2, raw1, out) == Quality::failed);
        CHECK(decode(8, raw1, out) == Quality::failed);

        std::vector<uint8_t> bad_single(12, 0);
        CHECK(decode(1, raw1, bad_single) == Quality::failed);
        std::vector<uint8_t> bad_double(24, 0);
        CHECK(decode(7, raw7, bad_double) == Quality::failed);

        std::vector<uint8_t> out7(sizeof(plain7), 0);
        CHECK(decode(7, std::span<const uint8_t>(raw7, sizeof(raw7) / 2), out7) == Quality::failed);
    }

    std::printf("bcpack_decode_test: all checks passed\n");
    return 0;
}
