// vk_pipeline_check --merge's check of a saved cache's header (pipeline_cache_header.h), without a Vulkan device.
#include "pipeline_cache_header.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace {

void put(std::vector<uint8_t>& data, size_t at, uint32_t value) {
    for (int i = 0; i < 4; ++i) data[at + i] = uint8_t(value >> (8 * i));
}

std::vector<uint8_t> cache(uint32_t version, uint32_t vendor, uint32_t device, const uint8_t (&uuid)[16]) {
    std::vector<uint8_t> data(48, 0xAB);   // a header and some opaque driver data
    put(data, 0, 32);
    put(data, 4, version);
    put(data, 8, vendor);
    put(data, 12, device);
    for (int i = 0; i < 16; ++i) data[16 + i] = uuid[i];
    return data;
}

}  // namespace

int main() {
    using leonida::vk_test::pipeline_cache_mismatch;
    const uint8_t uuid[16] = {0x64, 0x4b, 0x88, 0x65, 0x79, 0x26, 0x38, 0x47,
                              0xcb, 0x65, 0x75, 0xcc, 0x2a, 0x9e, 0xc8, 0x71};
    uint8_t other[16];
    for (int i = 0; i < 16; ++i) other[i] = uuid[i];
    other[15] ^= 1;
    CHECK(leonida::vk_test::uuid_hex(uuid) == "644b886579263847cb6575cc2a9ec871");

    const auto good = cache(1, 0x10de, 0x2783, uuid);
    CHECK(pipeline_cache_mismatch(good.data(), good.size(), 0x10de, 0x2783, uuid).empty());
    CHECK(pipeline_cache_mismatch(good.data(), 31, 0x10de, 0x2783, uuid) == "shorter than a pipeline cache header");
    CHECK(pipeline_cache_mismatch(nullptr, 0, 0x10de, 0x2783, uuid) == "shorter than a pipeline cache header");

    // Another vendor, another GPU of the same vendor, another driver build (UUID) and another header version.
    CHECK(pipeline_cache_mismatch(good.data(), good.size(), 0x1002, 0x2783, uuid) ==
          "written by vendor 0x10de device 0x2783, this is 0x1002 0x2783");
    CHECK(!pipeline_cache_mismatch(good.data(), good.size(), 0x10de, 0x2704, uuid).empty());
    CHECK(pipeline_cache_mismatch(good.data(), good.size(), 0x10de, 0x2783, other).find("pipelineCacheUUID 644b") == 0);
    const auto v2 = cache(2, 0x10de, 0x2783, uuid);
    CHECK(pipeline_cache_mismatch(v2.data(), v2.size(), 0x10de, 0x2783, uuid) == "header version 2, not 1");

    // A header size field that can't be right: below the version-one size or beyond the data.
    auto small = good;
    put(small, 0, 16);
    CHECK(!pipeline_cache_mismatch(small.data(), small.size(), 0x10de, 0x2783, uuid).empty());
    auto big = good;
    put(big, 0, 4096);
    CHECK(!pipeline_cache_mismatch(big.data(), big.size(), 0x10de, 0x2783, uuid).empty());

    std::puts("pipeline_cache_header: ok");
    return 0;
}
