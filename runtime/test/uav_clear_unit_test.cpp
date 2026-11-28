// Portable CPU-only unit tests for runtime/src/uav_clear.h (Vulkan headers only, no GPU, no game files).
// Exercises all 45 typed-UAV formats, uint/float encoding, cross-format decoding, packed sub-byte channels,
// small floats, shared-exponent RGB9E5, out-of-bounds detail::put/get guards, and raw_element().
#include "../src/uav_clear.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);   \
        return 1;                                                               \
    }                                                                           \
} while (0)

namespace {

namespace uc = leonida::uav_clear;

std::array<std::uint32_t, 4> floats(float r, float g, float b, float a) {
    return {std::bit_cast<std::uint32_t>(r), std::bit_cast<std::uint32_t>(g),
            std::bit_cast<std::uint32_t>(b), std::bit_cast<std::uint32_t>(a)};
}

std::vector<std::uint8_t> enc_bytes(VkFormat fmt, bool is_float, const std::array<std::uint32_t, 4>& vals) {
    uc::Element e{};
    if (!uc::encode(fmt, is_float, vals, e)) return {};
    return {e.bytes.begin(), e.bytes.begin() + e.size};
}

std::uint32_t le32(const std::vector<std::uint8_t>& bytes) {
    std::uint32_t w = 0;
    std::memcpy(&w, bytes.data(), bytes.size() < 4 ? bytes.size() : 4);
    return w;
}

// Reference bit-by-bit put/get for exhaustive parity verification.
void ref_put(uc::Element& e, uc::Channel c, std::uint32_t value) {
    value &= uc::detail::mask(c.bits);
    for (unsigned i = 0; i < c.bits; ++i)
        if ((value >> i) & 1u) e.bytes[(c.offset + i) / 8] |= std::uint8_t(1u << ((c.offset + i) % 8));
}

std::uint32_t ref_get(const uc::Element& e, uc::Channel c) {
    std::uint32_t v = 0;
    for (unsigned i = 0; i < c.bits; ++i)
        v |= std::uint32_t((e.bytes[(c.offset + i) / 8] >> ((c.offset + i) % 8)) & 1u) << i;
    return v;
}

}  // namespace

int main() {
    using V = std::vector<std::uint8_t>;

    // 1. Every entry in uc::layouts[] resolves to itself via uc::layout(), and unknown formats return nullptr.
    for (const uc::Layout& l : uc::layouts) {
        CHECK(uc::layout(l.format) == &l);
    }
    CHECK(uc::layout(VK_FORMAT_UNDEFINED) == nullptr);
    CHECK(uc::layout(VK_FORMAT_BC1_RGBA_UNORM_BLOCK) == nullptr);
    CHECK(uc::layout(VK_FORMAT_D24_UNORM_S8_UINT) == nullptr);

    // 2. Bit-exact parity of detail::put() / detail::get() vs bit-by-bit reference across all valid offsets & widths
    //    (including unaligned 32-bit channels crossing byte and 64-bit boundaries).
    for (unsigned width = 1; width <= 32; ++width) {
        const std::uint32_t patterns[] = {0u, 1u, 0x55555555u, 0xaaaaaaaau, 0x12345678u, 0xffffffffu};
        for (unsigned offset = 0; offset + width <= 128; offset += (width >= 16 ? 7 : 5)) {
            const uc::Channel ch{static_cast<std::uint8_t>(offset), static_cast<std::uint8_t>(width)};
            for (const std::uint32_t pat : patterns) {
                uc::Element fast{}, ref{};
                uc::detail::put(fast, ch, pat);
                ref_put(ref, ch, pat);
                CHECK(fast.bytes == ref.bytes);
                CHECK(uc::detail::get(fast, ch) == ref_get(ref, ch));
                CHECK(uc::detail::get(fast, ch) == (pat & uc::detail::mask(width)));
            }
        }
    }

    // 3. Bounds-safe no-op on malformed channels (0 bits, >32 bits, offset past 128 bits).
    {
        uc::Element e{};
        uc::detail::put(e, {0, 0}, 0xffffffffu);
        uc::detail::put(e, {0, 33}, 0xffffffffu);
        uc::detail::put(e, {120, 16}, 0xffffffffu);
        uc::detail::put(e, {128, 1}, 0xffffffffu);
        for (std::uint8_t b : e.bytes) CHECK(b == 0);
        CHECK(uc::detail::get(e, {0, 0}) == 0u);
        CHECK(uc::detail::get(e, {0, 33}) == 0u);
        CHECK(uc::detail::get(e, {120, 16}) == 0u);
        CHECK(uc::detail::get(e, {128, 1}) == 0u);
    }

    // 4. Uint clears copy raw low bits across byte-aligned and packed formats.
    CHECK((enc_bytes(VK_FORMAT_R8G8B8A8_UNORM, false, {0x1ff, 2, 3, 4}) == V{0xff, 2, 3, 4}));
    CHECK(le32(enc_bytes(VK_FORMAT_R32_SFLOAT, false, {0x3f800000, 0, 0, 0})) == 0x3f800000u);
    CHECK((enc_bytes(VK_FORMAT_R16_SINT, false, {0xfffe8000, 0, 0, 0}) == V{0x00, 0x80}));
    CHECK(le32(enc_bytes(VK_FORMAT_A2B10G10R10_UINT_PACK32, false, {1023, 0, 1, 7})) ==
          (1023u | (1u << 20) | (3u << 30)));
    CHECK(le32(enc_bytes(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, false, {0xffffffffu, 1, 2, 3})) == 0u);

    // 5. Float clears: UNORM, SNORM, BGRA, packed 10:10:10:2 / 4:4:4:4 / 5:6:5 / 1:5:5:5, half, 11:11:10, RGB9E5, int.
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    CHECK((enc_bytes(VK_FORMAT_R8G8B8A8_UNORM, true, floats(1.0f, 0.5f, -1.0f, qnan)) == V{0xff, 0x80, 0, 0}));
    CHECK((enc_bytes(VK_FORMAT_R8G8B8A8_SNORM, true, floats(1.0f, -1.0f, 0.5f, 2.0f)) == V{0x7f, 0x81, 0x40, 0x7f}));
    CHECK((enc_bytes(VK_FORMAT_B8G8R8A8_UNORM, true, floats(1.0f, 0.0f, 0.0f, 1.0f)) == V{0, 0, 0xff, 0xff}));
    CHECK(le32(enc_bytes(VK_FORMAT_A2B10G10R10_UNORM_PACK32, true, floats(1.0f, 0.0f, 0.5f, 1.0f))) ==
          (1023u | (512u << 20) | (3u << 30)));
    CHECK((enc_bytes(VK_FORMAT_R4G4B4A4_UNORM_PACK16, true, floats(1.0f, 0, 0, 0)) == V{0x00, 0xf0}));
    CHECK((enc_bytes(VK_FORMAT_R4G4B4A4_UNORM_PACK16, true, floats(0, 0, 0, 1.0f)) == V{0x0f, 0x00}));
    CHECK((enc_bytes(VK_FORMAT_R5G6B5_UNORM_PACK16, true, floats(0.25f, 0.75f, 0.4f, 0.6f)) == V{0xec, 0x45}));
    CHECK((enc_bytes(VK_FORMAT_A1R5G5B5_UNORM_PACK16, true, floats(0.25f, 0.75f, 0.4f, 0.6f)) == V{0xec, 0xa2}));
    CHECK((enc_bytes(VK_FORMAT_A4R4G4B4_UNORM_PACK16, true, floats(0.25f, 0.75f, 0.4f, 0.6f)) == V{0xb6, 0x94}));
    CHECK((enc_bytes(VK_FORMAT_R16_SFLOAT, true, floats(65520.0f, 0, 0, 0)) == V{0xff, 0x7b}));
    CHECK((enc_bytes(VK_FORMAT_R16_SFLOAT, true, floats(inf, 0, 0, 0)) == V{0x00, 0x7c}));
    CHECK((enc_bytes(VK_FORMAT_R16_SFLOAT, true, floats(1e-7f, 0, 0, 0)) == V{0x01, 0x00}));
    CHECK(le32(enc_bytes(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(1e6f, 1e6f, 1e6f, 0))) == 0xf7fdffbfu);
    CHECK(le32(enc_bytes(VK_FORMAT_B10G11R11_UFLOAT_PACK32, true, floats(-1.0f, 0, 0, 0))) == 0u);
    CHECK(le32(enc_bytes(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 4.0f, 0))) == 0x94010040u);
    CHECK(le32(enc_bytes(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(65408.0f, 65472.0f, inf, 0))) == 0xffffffffu);
    CHECK(le32(enc_bytes(VK_FORMAT_R32_SINT, true, floats(-3.7f, 0, 0, 0))) == std::uint32_t(-3));
    CHECK(le32(enc_bytes(VK_FORMAT_R32_SINT, true, floats(-inf, 0, 0, 0))) == 0x80000000u);
    CHECK(le32(enc_bytes(VK_FORMAT_R32_UINT, true, floats(inf, 0, 0, 0))) == 0xffffffffu);
    CHECK(le32(enc_bytes(VK_FORMAT_R32_UINT, true, floats(qnan, 0, 0, 0))) == 0u);

    // 6. All 45 formats round-trip cleanly through encode() -> decode() for both uint and float modes, and size
    //    mismatch is rejected.
    const std::array<std::uint32_t, 4> fvals = floats(0.25f, 0.5f, 0.75f, 1.0f);
    const std::array<std::uint32_t, 4> uvals = {0x13u, 0x27u, 0x3bu, 0x01u};
    for (const uc::Layout& l : uc::layouts) {
        uc::Element ef{}, eu{};
        VkClearColorValue cf{}, cu{};
        CHECK(uc::encode(l.format, true, fvals, ef) && ef.size == l.bytes);
        CHECK(uc::decode(l.format, ef, cf));
        CHECK(uc::encode(l.format, false, uvals, eu) && eu.size == l.bytes);
        CHECK(uc::decode(l.format, eu, cu));
        // Wrong element size must fail cleanly.
        uc::Element wrong = ef;
        wrong.size = l.bytes == 4 ? 8u : 4u;
        CHECK(!uc::decode(l.format, wrong, cf));
    }

    // 7. Cross-format decode & raw_element().
    {
        uc::Element e{};
        VkClearColorValue v{};
        CHECK(uc::encode(VK_FORMAT_R32_UINT, false, {0x40490fdbu, 0, 0, 0}, e) &&
              uc::decode(VK_FORMAT_R32_SFLOAT, e, v) && std::bit_cast<std::uint32_t>(v.float32[0]) == 0x40490fdbu);
        CHECK(uc::encode(VK_FORMAT_R8_UINT, false, {0x80u, 0, 0, 0}, e) &&
              uc::decode(VK_FORMAT_R8_SINT, e, v) && v.int32[0] == -128);
        CHECK(uc::encode(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, true, floats(1.0f, 2.0f, 4.0f, 0), e) &&
              uc::decode(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, e, v) &&
              v.float32[0] == 1.0f && v.float32[1] == 2.0f && v.float32[2] == 4.0f && v.float32[3] == 1.0f);
        const uc::Element raw = uc::raw_element({0xdeadbeefu, 1u, 2u, 3u});
        std::uint32_t rw = 0;
        std::memcpy(&rw, raw.bytes.data(), 4);
        CHECK(raw.size == 4u && rw == 0xdeadbeefu);
    }

    std::printf("uav_clear_unit_test: all checks passed\n");
    return 0;
}
