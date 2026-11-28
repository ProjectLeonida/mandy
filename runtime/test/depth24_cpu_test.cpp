// The CPU side of the Depth24 emulation (src/depth24.h: depth24_quantize, depth24_to_float, depth24_clear_value) must
// equal depth24_convert.comp (roundEven of a correctly rounded float product; integer arithmetic for the upload) on a
// thread that has changed its floating-point rounding mode: a float multiply, std::nearbyint and a double-to-float
// conversion follow that mode, so 0x1p-23f quantised to 1 under FE_DOWNWARD and FE_TOWARDZERO instead of 2.
// Under each mode the results must equal the round-to-nearest ones (a float multiply and std::nearbyint, run under
// the default mode only) and fixed expectations, ties at the integer boundary included. The test sets the mode itself;
// the header never does. No Vulkan loader, GPU, shaders or game data.
#include "../src/depth24.h"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

float as_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// The reference, used under round-to-nearest only: a float multiply, then nearbyint (ties to even).
std::uint32_t reference_quantize(float value) {
    if (!(value > 0.0f)) return 0;
    if (value >= 1.0f) return leonida::depth24_max;
    const volatile float product = value * 16777215.0f;
    return static_cast<std::uint32_t>(std::nearbyint(product));
}

std::vector<float> corpus() {
    std::vector<float> values = {0.0f, -0.0f, -1.0f, 1.0f, 2.0f, 0.5f, 0x1p-23f, 0x1p-24f, 0x1p-25f, 0x1p-26f,
                                 0x1p-149f, 0x1p-126f, std::nextafter(1.0f, 0.0f), std::nextafter(1.0f, 2.0f),
                                 as_float(0x7fc00000u), as_float(0x7f800000u), as_float(0xff800000u)};
    // A stride through the bit patterns of [0, 1] (a prime step), the floats around every 251st code's midpoint
    // (d + 0.5) / 16777215, where the product is a tie or a hair off one, and the powers of two.
    for (std::uint32_t bits = 0; bits <= 0x3f800000u; bits += 4099u) values.push_back(as_float(bits));
    for (std::uint32_t d = 0; d < leonida::depth24_max; d += 251u) {
        const float tie = float((double(d) + 0.5) / 16777215.0);
        values.push_back(tie);
        values.push_back(std::nextafter(tie, 0.0f));
        values.push_back(std::nextafter(tie, 1.0f));
    }
    for (int k = 0; k < 24; ++k) values.push_back(std::ldexp(1.0f, -k));
    return values;
}

}  // namespace

int main() {
    using namespace leonida;
    unsigned checks = 0;

    // Fixed expectations. 0x1p-23f * 16777215 is 2 - 2^-23, representable, and nearest to 2 (rounding down gave 1).
    struct Case { float value; std::uint32_t code; };
    const Case cases[] = {
        {0x1p-23f, 2}, {0.0f, 0}, {-1.0f, 0}, {1.0f, depth24_max}, {2.0f, depth24_max}, {0x1p-25f, 0}, {0x1p-24f, 1},
        {0.5f, 8388608},  // 0.5 * 16777215 = 8388607.5 exactly: the tie goes to the even neighbour
    };
    const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
    const int saved = std::fegetround();
    for (const int mode : modes) {
        std::fesetround(mode);
        std::vector<std::uint32_t> got;
        for (const Case& c : cases) got.push_back(depth24_quantize(c.value));
        std::fesetround(saved);
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (got[i] != cases[i].code) {
                std::fprintf(stderr, "FP mode %d: depth24_quantize(%a) = %u, expected %u\n", mode,
                             double(cases[i].value), got[i], cases[i].code);
                return 1;
            }
            ++checks;
        }
    }

    const std::vector<float> values = corpus();
    std::vector<std::uint32_t> expected(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) expected[i] = reference_quantize(values[i]);
    for (const int mode : modes) {
        std::fesetround(mode);
        std::vector<std::uint32_t> got(values.size());
        std::vector<float> clears(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            got[i] = depth24_quantize(values[i]);
            clears[i] = depth24_clear_value(values[i]);
        }
        // The upload direction: a stride through the codes, and the ends.
        std::vector<float> floats;
        for (std::uint32_t d = 0; d <= depth24_max; d += 4093u) floats.push_back(depth24_to_float(d));
        floats.push_back(depth24_to_float(depth24_max));
        std::fesetround(saved);

        for (std::size_t i = 0; i < values.size(); ++i) {
            if (got[i] != expected[i]) {
                std::fprintf(stderr, "FP mode %d: depth24_quantize(%a) = %u, round-to-nearest gives %u\n", mode,
                             double(values[i]), got[i], expected[i]);
                return 1;
            }
            const float want = float(double(expected[i]) / 16777215.0);  // nearest, under the restored mode
            if (std::memcmp(&clears[i], &want, 4) != 0) {
                std::fprintf(stderr, "FP mode %d: depth24_clear_value(%a) = %a, expected %a\n", mode,
                             double(values[i]), double(clears[i]), double(want));
                return 1;
            }
            ++checks;
        }
        std::size_t k = 0;
        for (std::uint32_t d = 0; d <= depth24_max; d += 4093u, ++k) {
            const float want = float(double(d) / 16777215.0);
            if (std::memcmp(&floats[k], &want, 4) != 0 || depth24_quantize(floats[k]) != d) {
                std::fprintf(stderr, "FP mode %d: depth24_to_float(%u) = %a, expected %a\n", mode, d,
                             double(floats[k]), double(want));
                return 1;
            }
            ++checks;
        }
        const float top = floats.back();
        if (top != 1.0f) {
            std::fprintf(stderr, "FP mode %d: depth24_to_float(0xffffff) = %a, expected 1\n", mode, double(top));
            return 1;
        }
    }
    std::printf("depth24_cpu_test: %u checks passed under FE_TONEAREST, FE_DOWNWARD, FE_UPWARD and FE_TOWARDZERO\n",
                checks);
    return 0;
}
