#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <vector>

namespace leonida {

// Checks the binary framing and returns aligned words in host byte order for Vulkan.
// The caller still relies on the driver or spirv-val for instruction and capability validation.
inline std::vector<std::uint32_t> read_spirv_words(std::istream& input, std::streamoff byte_count) {
    constexpr std::streamoff word_bytes = sizeof(std::uint32_t);
    constexpr std::streamoff header_bytes = 5 * word_bytes;
    constexpr std::uint32_t native_magic = 0x07230203u;
    constexpr std::uint32_t reversed_magic = 0x03022307u;
    if (byte_count < header_bytes || byte_count % word_bytes || !input.seekg(0)) return {};

    std::uint32_t magic = 0;
    if (!input.read(reinterpret_cast<char*>(&magic), word_bytes) ||
        (magic != native_magic && magic != reversed_magic)) return {};

    std::vector<std::uint32_t> words(static_cast<std::size_t>(byte_count / word_bytes));
    words.front() = native_magic;
    if (!input.read(reinterpret_cast<char*>(words.data() + 1), byte_count - word_bytes)) return {};
    if (magic == reversed_magic) {
        for (std::size_t i = 1; i < words.size(); ++i) {
            const auto word = words[i];
            words[i] = (word << 24) | ((word & 0x0000ff00u) << 8) |
                       ((word & 0x00ff0000u) >> 8) | (word >> 24);
        }
    }
    return words;
}

}  // namespace leonida
