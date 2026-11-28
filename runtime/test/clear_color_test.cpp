// Value-level replay regression: full and rectangular integer clears must carry integer values, not float bits,
// normalised clears are rounded to nearest even like D3D (0.5 on 8 bits is 128), and the colour channels of an sRGB
// clear are encoded like WARP (0.5 is code 188) and handed over as that code's linear value.
// No Vulkan loader, GPU, shaders or game data. Expected conversions also match the WARP differential GPU tests.
#include "../src/vulkan_replay.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {
VkClearColorValue captured{};
unsigned calls = 0;
VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* color,
    std::uint32_t, const VkImageSubresourceRange*) { captured = *color; ++calls; }
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t n, const VkClearAttachment* a,
    std::uint32_t, const VkClearRect*) { if (n == 1) { captured = a[0].clearValue.color; ++calls; } }

bool replay_color(VkFormat format, const leonida::ClearColor& color, bool rectangle) {
    leonida::CommandList list;
    const leonida::Rect rect{1, 1, 3, 3};
    if (!list.clear_render_target({1, 0, 0, 1}, color,
        rectangle ? std::span<const leonida::Rect>(&rect, 1) : std::span<const leonida::Rect>{}) || !list.close()) return false;
    leonida::ResourceBinding binding{};
    binding.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    binding.format = format;
    binding.width = binding.height = 4;
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.image_view = [](const auto&, const auto&) { return reinterpret_cast<VkImageView>(std::uintptr_t{1}); };
    calls = 0;
    return leonida::replay(list, VK_NULL_HANDLE, [&](auto) { return &binding; }, vk) == leonida::ReplayResult::success && calls == 1;
}

// Integer RTV format with its component range; A2B10G10R10 has a 2-bit alpha.
struct IntegerFormat { VkFormat format; const char* name; double low, high, alpha_high; bool is_signed; };

float from_bits(std::uint32_t word) {
    float value;
    std::memcpy(&value, &word, sizeof value);
    return value;
}

// The ideal sRGB encoding of a linear value in 0..1 (IEC 61966-2-1), on a scale of 1.
double srgb_encode(double linear) {
    return linear < 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

// Reference conversion measured against WARP: NaN -> 0, clamp to the range, truncate fractions toward zero.
std::uint32_t expected_bits(float input, double low, double high, bool is_signed) {
    const double v = double(input);
    const double r = v != v ? 0.0 : v <= low ? low : v >= high ? high : std::trunc(v);
    return is_signed ? static_cast<std::uint32_t>(static_cast<std::int32_t>(r)) : static_cast<std::uint32_t>(r);
}
} // namespace

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const leonida::ClearColor cases[] = {
        {1, 2, 3, 4},
        {1.5f, 2.5f, 3.5f, 4.5f},
        {-1, -0.0f, 254.5f, 255.5f},
        {256, 65536, 16777216, 4294967296.0f},
        {nan, inf, -inf, 0.5f},
        {0.499f, 0.501f, 1.499f, 1.501f},
        {4294967040.0f, std::numeric_limits<float>::max(), 255, 0},
        // Component-range boundaries for the 8-, 16-, 10- and 2-bit formats and the signed 32-bit limits.
        {127.5f, 128, -128.5f, -129},
        {32767.5f, 32768, -32768.5f, -32769},
        {65535.5f, 65536, 1023.5f, 1024},
        {3.5f, 4, 2147483520.0f, -2147483648.0f},
    };
    constexpr double u32max = 4294967295.0, s32min = -2147483648.0, s32max = 2147483647.0;
    const IntegerFormat integer_formats[] = {
        {VK_FORMAT_R8_UINT, "R8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R8G8_UINT, "R8G8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R8G8B8A8_UINT, "R8G8B8A8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R16_UINT, "R16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R16G16_UINT, "R16G16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R16G16B16A16_UINT, "R16G16B16A16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R32_UINT, "R32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_R32G32_UINT, "R32G32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_R32G32B32A32_UINT, "R32G32B32A32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_A2B10G10R10_UINT_PACK32, "A2B10G10R10_UINT_PACK32", 0, 1023, 3, false},
        {VK_FORMAT_R8_SINT, "R8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R8G8_SINT, "R8G8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R8G8B8A8_SINT, "R8G8B8A8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R16_SINT, "R16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R16G16_SINT, "R16G16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R16G16B16A16_SINT, "R16G16B16A16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R32_SINT, "R32_SINT", s32min, s32max, s32max, true},
        {VK_FORMAT_R32G32_SINT, "R32G32_SINT", s32min, s32max, s32max, true},
        {VK_FORMAT_R32G32B32A32_SINT, "R32G32B32A32_SINT", s32min, s32max, s32max, true},
    };
    unsigned checks = 0, failed_formats = 0;
    for (const auto& f : integer_formats) {
        bool format_ok = true;
        unsigned case_index = 0;
        for (const auto& c : cases) {
            for (const bool rect : {false, true}) {
                if (!replay_color(f.format, c, rect)) {
                    std::fprintf(stderr, "%s: replay failed (case %u rect %d)\n", f.name, case_index, int(rect));
                    return 1;
                }
                for (unsigned i = 0; i < 4 && format_ok; ++i) {
                    const std::uint32_t expected = expected_bits(c[i], f.low, i == 3 ? f.alpha_high : f.high, f.is_signed);
                    if (captured.uint32[i] != expected) {
                        std::fprintf(stderr, "integer clear %s case %u rect %d channel %u: got 0x%08x, expected 0x%08x\n",
                            f.name, case_index, int(rect), i, captured.uint32[i], expected);
                        format_ok = false;
                    }
                }
                if (!format_ok) break;
                ++checks;
            }
            if (!format_ok) break;
            ++case_index;
        }
        if (!format_ok) ++failed_formats;
    }
    for (const auto& c : cases) {
        for (const VkFormat format : {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT}) {
            for (const bool rect : {false, true}) {
                if (!replay_color(format, c, rect) || std::memcmp(captured.float32, c.data(), sizeof(captured)) != 0) {
                    std::fprintf(stderr, "float clear: format %d rect %d changed the value\n", int(format), int(rect));
                    return 1;
                }
                ++checks;
            }
        }
    }
    // Normalised formats: D3D rounds to nearest even, the clamp and NaN rules are 0..1 and 0, and an exact tie goes to
    // the even neighbour. NVIDIA rounds 0.5 on 8 bits down to 127 by itself, so the value is rounded before Vulkan sees it.
    struct Norm {
        VkFormat format;
        leonida::ClearColor color;
        std::array<std::int32_t, 4> steps;  // what D3D writes, in units of the channel's largest value
        std::array<double, 4> maximum;
    };
    const Norm norms[] = {
        {VK_FORMAT_R8G8B8A8_UNORM, {0.5f, 0.499f, 0.501f, 1.0f}, {128, 127, 128, 255}, {255, 255, 255, 255}},
        {VK_FORMAT_B8G8R8A8_UNORM, {-1, 2, nan, 0.5f}, {0, 255, 0, 128}, {255, 255, 255, 255}},
        {VK_FORMAT_R8G8_UNORM, {0.5f, 1.5f, 0, 0}, {128, 255, 0, 0}, {255, 255, 255, 255}},
        {VK_FORMAT_A2B10G10R10_UNORM_PACK32, {0.5f, 0.5f, 0.5f, 0.5f}, {512, 512, 512, 2}, {1023, 1023, 1023, 3}},
        {VK_FORMAT_R16G16B16A16_UNORM, {0.5f, 0.25f, 0.75f, 1.0f}, {32768, 16384, 49151, 65535},
         {65535, 65535, 65535, 65535}},
        {VK_FORMAT_R16_UNORM, {0.5f, 0, 0, 0}, {32768, 0, 0, 0}, {65535, 65535, 65535, 65535}},
        // Signed normalised: -1.0 is -127 (-32767), the scale is 127 (32767) and ties go to the even neighbour.
        {VK_FORMAT_R8G8B8A8_SNORM, {-1, -0.5f, 0.5f, 1}, {-127, -64, 64, 127}, {127, 127, 127, 127}},
        {VK_FORMAT_R8G8_SNORM, {nan, inf, -inf, -0.5f}, {0, 127, -127, -64}, {127, 127, 127, 127}},
        {VK_FORMAT_R16G16B16A16_SNORM, {-2, 2, 0.001f, -0.001f}, {-32767, 32767, 33, -33},
         {32767, 32767, 32767, 32767}},
    };
    for (const auto& n : norms) {
        for (const bool rect : {false, true}) {
            if (!replay_color(n.format, n.color, rect)) return 1;
            for (unsigned i = 0; i < 4; ++i) {
                const float expected = n.maximum[i] != 0.0 ? static_cast<float>(double(n.steps[i]) / n.maximum[i]) : n.color[i];
                if (captured.float32[i] != expected) {
                    std::fprintf(stderr, "normalised clear: format %d rect %d channel %u: got %.9g, expected %.9g\n",
                        int(n.format), int(rect), i, double(captured.float32[i]), double(expected));
                    return 1;
                }
            }
            ++checks;
        }
    }
    // sRGB: the colour channels are encoded here the way WARP encodes them and handed over as the linear value of that
    // code, so the byte does not depend on where a GPU puts the steps of its own curve; alpha is linear and rounds like
    // UNORM. The codes are the bytes WARP 10.0.26100.5074 stored for these clears of an R8G8B8A8_UNORM_SRGB target.
    // WARP takes the value to 12 bits first (x * 4095 in single precision, rounded half up), so some are not the
    // nearest code of the ideal curve.
    struct Srgb {
        float linear;
        unsigned code;
    };
    const Srgb srgb[] = {
        {0.5f, 188}, {0.25f, 137}, {0.75f, 225}, {0.125f, 99}, {0.0039f, 13}, {0.5019f, 188}, {0.9961f, 255},
        {0.2f, 124}, {0.7f, 218}, {0.0001f, 0}, {0.001f, 3}, {0.003f, 10}, {0.0031308f, 10}, {0.004f, 13},
        {0.01f, 25}, {0.05f, 63}, {0.1f, 89}, {0.3f, 149}, {0.6f, 203}, {0.9f, 243}, {0.99f, 254}, {0.999f, 255},
        {0.9999f, 255}, {0, 0}, {1, 255}, {-1, 0}, {2, 255}, {nan, 0}, {inf, 255}, {-inf, 0},
        // The ideal curve gives 1.32, 2.64, 5.27 and 6.59 for these.
        {0.0004f, 2}, {0.0008f, 2}, {0.0016f, 6}, {0.002f, 6},
        // Neighbouring floats on the two sides of a step: x * 4095 is just below 0.5 and 0.5 after rounding to
        // single precision (0.50000003 exactly); 2071.49976 and 2071.5 after rounding (2071.4999986 exactly).
        {from_bits(0x39000800), 0}, {from_bits(0x39000801), 1}, {from_bits(0x3f018017), 188},
        {from_bits(0x3f018018), 189},
    };
    for (const VkFormat format : {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB}) {
        for (std::size_t first = 0; first < std::size(srgb); first += 3) {
            leonida::ClearColor color{0, 0, 0, first % 2 ? 0.499f : 0.5f};
            unsigned codes[3] = {0, 0, 0};
            for (std::size_t i = 0; i < 3 && first + i < std::size(srgb); ++i) {
                color[i] = srgb[first + i].linear;
                codes[i] = srgb[first + i].code;
            }
            for (const bool rect : {false, true}) {
                if (!replay_color(format, color, rect)) return 1;
                for (unsigned i = 0; i < 3; ++i) {
                    // The value is the middle of the code's range: its ideal encoding is the code itself.
                    const double encoded = srgb_encode(captured.float32[i]) * 255.0;
                    if (captured.float32[i] != leonida::detail::srgb8_linear(codes[i]) ||
                        std::fabs(encoded - codes[i]) > 0.001) {
                        std::fprintf(stderr, "sRGB clear: format %d rect %d value %.9g: got %.9g, which encodes to "
                                     "%.4f, expected code %u\n", int(format), int(rect), double(color[i]),
                                     double(captured.float32[i]), encoded, codes[i]);
                        return 1;
                    }
                }
                const float alpha = static_cast<float>((first % 2 ? 127.0 : 128.0) / 255.0);
                if (captured.float32[3] != alpha) {
                    std::fprintf(stderr, "sRGB clear: format %d rect %d: alpha %.9g, expected %.9g\n", int(format),
                                 int(rect), double(captured.float32[3]), double(alpha));
                    return 1;
                }
                ++checks;
            }
        }
    }
    // Every code is the code of its own linear value, so a clear to a colour read back from an sRGB target stores
    // that colour again.
    for (unsigned code = 0; code < 256; ++code) {
        if (leonida::detail::srgb8_from_linear(leonida::detail::srgb8_linear(code)) != code) {
            std::fprintf(stderr, "sRGB code %u does not encode to itself\n", code);
            return 1;
        }
    }
    ++checks;
    // The other integer formats clamp to the component range and truncate toward zero (what WARP writes).
    struct Integer {
        VkFormat format;
        leonida::ClearColor color;
        std::array<std::int32_t, 4> expected;
    };
    const Integer integers[] = {
        {VK_FORMAT_R8_SINT, {-1.5f, -2.5f, 2.5f, 3.5f}, {-1, -2, 2, 3}},
        {VK_FORMAT_R8G8B8A8_SINT, {-129, 128, -32769, 32768}, {-128, 127, -128, 127}},
        {VK_FORMAT_R8G8B8A8_SINT, {2147483648.0f, -2147483904.0f, nan, inf}, {127, -128, 0, 127}},
        {VK_FORMAT_R16_SINT, {-129, 128, -32769, 32768}, {-129, 128, -32768, 32767}},
        {VK_FORMAT_R32_SINT, {-129, 128, -32769, 32768}, {-129, 128, -32769, 32768}},
        {VK_FORMAT_R32_SINT, {2147483648.0f, -2147483904.0f, nan, inf}, {2147483647, -2147483647 - 1, 0, 2147483647}},
        {VK_FORMAT_R32G32B32A32_SINT, {-inf, 0.5f, -0.5f, 100000.0f}, {-2147483647 - 1, 0, 0, 100000}},
        {VK_FORMAT_R8_UINT, {256, 1.5f, -1, nan}, {255, 1, 0, 0}},
        {VK_FORMAT_R16_UINT, {65536, 65535.5f, -0.0f, 2.5f}, {65535, 65535, 0, 2}},
        {VK_FORMAT_R8G8_UINT, {256, 1.5f, -1, nan}, {255, 1, 0, 0}},
        {VK_FORMAT_R16G16_SINT, {-32769, 32768, -1.5f, nan}, {-32768, 32767, -1, 0}},
        {VK_FORMAT_R16G16B16A16_UINT, {65536, 65535.5f, -0.0f, 2.5f}, {65535, 65535, 0, 2}},
        // R10G10B10A2_UINT: each channel clamps to its own width, 10, 10, 10 and 2 bits.
        {VK_FORMAT_A2B10G10R10_UINT_PACK32, {1023.5f, 2000, 1024, 4}, {1023, 1023, 1023, 3}},
        {VK_FORMAT_A2B10G10R10_UINT_PACK32, {5.5f, 1.5f, 1022.5f, 2.5f}, {5, 1, 1022, 2}},
    };
    for (const auto& c : integers) {
        for (const bool rect : {false, true}) {
            if (!replay_color(c.format, c.color, rect)) return 1;
            for (unsigned i = 0; i < 4; ++i) {
                if (captured.int32[i] != c.expected[i]) {
                    std::fprintf(stderr, "integer clear: format %d rect %d channel %u: got %d, expected %d\n",
                        int(c.format), int(rect), i, captured.int32[i], c.expected[i]);
                    return 1;
                }
            }
            ++checks;
        }
    }
    // Ties go to even whatever the thread's floating-point rounding mode (std::nearbyint would follow it: 127.5 is
    // 127 when rounding down). Converted under each mode, checked after the default mode is back.
    {
        const int mode = std::fegetround();
        VkClearColorValue unorm[3]{}, snorm[3]{};
        const int modes[3] = {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
        for (int m = 0; m < 3; ++m) {
            std::fesetround(modes[m]);
            unorm[m] = leonida::detail::render_target_clear_color(VK_FORMAT_R8G8B8A8_UNORM, {0.5f, 0.5f, 0.5f, 0.5f});
            snorm[m] = leonida::detail::render_target_clear_color(VK_FORMAT_R8G8B8A8_SNORM, {-0.5f, 0.5f, -0.5f, 0.5f});
        }
        std::fesetround(mode);
        for (int m = 0; m < 3; ++m) {
            const long u = std::lround(double(unorm[m].float32[0]) * 255.0);
            const long s0 = std::lround(double(snorm[m].float32[0]) * 127.0);
            const long s1 = std::lround(double(snorm[m].float32[1]) * 127.0);
            if (u != 128 || s0 != -64 || s1 != 64) {
                std::fprintf(stderr, "tie rounding under FP mode %d: UNORM %ld (128), SNORM %ld %ld (-64 64)\n",
                             modes[m], u, s0, s1);
                return 1;
            }
            ++checks;
        }
        // The sRGB encoding too: every 12-bit value and the floats around each step between two of them give the
        // same code under each mode as under the default one (a float multiply or std::nearbyint would follow it).
        std::vector<float> values;
        for (unsigned j = 0; j < 4096; ++j) {
            values.push_back(static_cast<float>(j / 4095.0));
            const float step = static_cast<float>((j + 0.5) / 4095.0);
            values.insert(values.end(), {std::nextafter(step, 0.0f), step, std::nextafter(step, 2.0f)});
        }
        std::vector<unsigned> codes[4];
        for (int m = 0; m < 4; ++m) {
            if (m < 3) std::fesetround(modes[m]);
            for (const float v : values) codes[m].push_back(leonida::detail::srgb8_from_linear(v));
            std::fesetround(mode);
        }
        for (int m = 0; m < 3; ++m) {
            if (codes[m] != codes[3]) {
                std::fprintf(stderr, "sRGB encoding under FP mode %d differs from the default mode\n", modes[m]);
                return 1;
            }
            ++checks;
        }
    }
    if (failed_formats) {
        std::fprintf(stderr, "clear_color_test: %u integer format(s) failed\n", failed_formats);
        return 1;
    }
    std::printf("clear_color_test: %u full/rectangle integer, normalised, sRGB and float replay checks passed\n",
                checks);
    return 0;
}
