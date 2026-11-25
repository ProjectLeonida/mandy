// The VkPipelineCacheHeaderVersionOne at the start of a saved pipeline cache, read without a Vulkan device.
//
// vk_pipeline_check --merge refuses an input written by another GPU or driver before it reaches
// vkMergePipelineCaches: a driver may accept such data and quietly drop it, and then the merged cache would miss
// pipelines that make_game.py's done-list (docs/pipeline_check.md, "Shard caches and the done-list") counts as built.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace leonida::vk_test {

struct PipelineCacheHeader {
    uint32_t header_size = 0;
    uint32_t version = 0;
    uint32_t vendor = 0;
    uint32_t device = 0;
    uint8_t uuid[16] = {};
};

// The header fields, little-endian as the Vulkan spec writes them. False when fewer than 32 bytes are there.
inline bool read_pipeline_cache_header(const void* data, size_t size, PipelineCacheHeader& header) {
    if (!data || size < 32) return false;
    const auto* bytes = static_cast<const uint8_t*>(data);
    auto word = [&](size_t at) {
        return uint32_t(bytes[at]) | uint32_t(bytes[at + 1]) << 8 | uint32_t(bytes[at + 2]) << 16 |
               uint32_t(bytes[at + 3]) << 24;
    };
    header.header_size = word(0);
    header.version = word(4);
    header.vendor = word(8);
    header.device = word(12);
    for (size_t i = 0; i < 16; ++i) header.uuid[i] = bytes[16 + i];
    return true;
}

inline std::string uuid_hex(const uint8_t (&uuid)[16]) {
    std::string text;
    char digits[3];
    for (const uint8_t byte : uuid) {
        std::snprintf(digits, sizeof digits, "%02x", byte);
        text += digits;
    }
    return text;
}

// Empty when the data is a version-one cache of this device (vendor, device and pipelineCacheUUID); otherwise why not.
inline std::string pipeline_cache_mismatch(const void* data, size_t size, uint32_t vendor, uint32_t device,
                                           const uint8_t (&uuid)[16]) {
    PipelineCacheHeader header;
    if (!read_pipeline_cache_header(data, size, header)) return "shorter than a pipeline cache header";
    char why[96];
    if (header.version != 1) {
        std::snprintf(why, sizeof why, "header version %u, not 1", header.version);
        return why;
    }
    if (header.header_size < 32 || header.header_size > size) {
        std::snprintf(why, sizeof why, "header size %u in %zu bytes", header.header_size, size);
        return why;
    }
    if (header.vendor != vendor || header.device != device) {
        std::snprintf(why, sizeof why, "written by vendor 0x%04x device 0x%04x, this is 0x%04x 0x%04x", header.vendor,
                      header.device, vendor, device);
        return why;
    }
    for (size_t i = 0; i < 16; ++i)
        if (header.uuid[i] != uuid[i]) return "pipelineCacheUUID " + uuid_hex(header.uuid) + ", this driver's is " +
                                              uuid_hex(uuid);
    return {};
}

}  // namespace leonida::vk_test
