// The two pure decisions behind D3D-704: how large a session cache may be (LEONIDA_PIPELINE_CACHE_BYTES) and
// whether a blob's VkPipelineCacheHeaderVersionOne names this device. No Vulkan, no GPU, no files.
#include "../src/pipeline_cache_store.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace store = leonida::pipeline_cache_store;

static void put32(std::uint8_t* p, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(value >> (8 * i));
}

static int test_limit() {
    using store::default_save_limit;
    // Absent, empty and the default knob.
    CHECK(store::parse_save_limit(nullptr).bytes == default_save_limit);
    CHECK(store::parse_save_limit(nullptr).valid);
    CHECK(store::parse_save_limit("").bytes == default_save_limit);
    // 0 is the documented "off", not a typo, and leading zeros are still a number.
    CHECK(store::parse_save_limit("0").bytes == 0);
    CHECK(store::parse_save_limit("0").valid);
    CHECK(store::parse_save_limit("007").bytes == 7);
    CHECK(store::parse_save_limit("1048576").bytes == 1048576);
    const std::string largest = std::to_string(std::numeric_limits<std::size_t>::max());
    CHECK(store::parse_save_limit(largest.c_str()).bytes == std::numeric_limits<std::size_t>::max());
    // Anything that is not a plain byte count falls back to the default and says the value was rejected.
    for (const char* text : {"-1", "+1", " 1", "1 ", "12x", "0x10", "1.5", "99999999999999999999999999"}) {
        const auto limit = store::parse_save_limit(text);
        CHECK(!limit.valid);
        CHECK(limit.bytes == default_save_limit);
    }
    return 0;
}

static int test_header() {
    leonida::cached_pso::Identity identity;
    identity.vendor = 0x1002;
    identity.device = 0x73ff;
    identity.driver = 0x02000d0b;

    std::uint8_t header[store::header_bytes] = {};
    put32(header, store::header_bytes);
    put32(header + 4, store::header_version_one);
    put32(header + 8, identity.vendor);
    put32(header + 12, identity.device);
    for (int i = 0; i < 16; ++i) header[16 + i] = identity.cache_uuid[i] = static_cast<std::uint8_t>(i + 1);
    CHECK(store::header_matches(header, sizeof(header), identity));
    // Extra payload past the header does not matter: the driver reads the header and then its own data.
    std::uint8_t blob[store::header_bytes + 8] = {};
    std::memcpy(blob, header, sizeof(header));
    CHECK(store::header_matches(blob, sizeof(blob), identity));

    // Too short and null are "no cache", never a read past the end.
    CHECK(!store::header_matches(header, store::header_bytes - 1, identity));
    CHECK(!store::header_matches(nullptr, store::header_bytes, identity));

    // The four fields a driver checks before trusting a blob from another device, driver or header version.
    auto differs = [&](int offset, std::uint32_t value) {
        std::uint8_t changed[store::header_bytes];
        std::memcpy(changed, header, sizeof(changed));
        put32(changed + offset, value);
        return !store::header_matches(changed, sizeof(changed), identity);
    };
    CHECK(differs(0, store::header_bytes + 4));
    CHECK(differs(4, store::header_version_one + 1));
    CHECK(differs(8, identity.vendor + 1));
    CHECK(differs(12, identity.device + 1));
    std::uint8_t uuid[store::header_bytes];
    std::memcpy(uuid, header, sizeof(uuid));
    uuid[16] ^= 0xff;
    CHECK(!store::header_matches(uuid, sizeof(uuid), identity));
    // vendorID/deviceID are u32s little-endian; the same bytes read the other way name another device.
    leonida::cached_pso::Identity swapped = identity;
    swapped.vendor = 0x02100000;
    CHECK(!store::header_matches(header, sizeof(header), swapped));
    return 0;
}

static int test_growth() {
    using store::GrowthWatch;
    constexpr auto builds = store::stalled_after_builds;
    constexpr auto kept = store::stalled_below_bytes;
    // No size at creation: nothing to compare with yet, and the first save becomes the reference.
    GrowthWatch fresh;
    CHECK(!fresh.stalled(1000, builds * 4));
    CHECK(fresh.stalled(1000, builds * 5));
    // A cache that grows with its builds is fine at any size.
    GrowthWatch growing;
    growing.observe(std::size_t{1} << 30, 10);
    CHECK(!growing.stalled((std::size_t{1} << 30) + kept, 10 + builds));
    CHECK(growing.bytes() == (std::size_t{1} << 30) + kept);
    CHECK(growing.compiled() == 10 + builds);
    // Too few builds compiled to tell: a few cache-sized misses can be refused for other reasons.
    GrowthWatch few;
    few.observe(5000, 0);
    CHECK(!few.stalled(5000, builds - 1));
    // Enough compiled builds and less than stalled_below_bytes of growth: said once, then never again.
    GrowthWatch stuck;
    stuck.observe(1346482377, 100);
    CHECK(stuck.stalled(1346482377 + kept - 1, 100 + builds));
    CHECK(!stuck.stalled(1346482377 + kept - 1, 100 + builds * 3));
    // Builds and growth are counted from the previous save: growth then, none now.
    GrowthWatch interval;
    interval.observe(2000, 0);
    CHECK(!interval.stalled(2000 + kept * 8, builds * 3));
    CHECK(interval.stalled(2000 + kept * 8, builds * 4));
    // Too few builds at one save still moves the reference to that save.
    GrowthWatch moved;
    moved.observe(2000, 0);
    CHECK(!moved.stalled(2000, builds - 1));
    CHECK(moved.stalled(2000, builds * 2 - 1));
    // A count that went backwards (a second device's counter view) is not read as a huge number of builds.
    GrowthWatch backwards;
    backwards.observe(3000, 50);
    CHECK(!backwards.stalled(3000, 10));
    return 0;
}

int main() {
    if (const int failed = test_limit()) return failed;
    if (const int failed = test_header()) return failed;
    return test_growth();
}
