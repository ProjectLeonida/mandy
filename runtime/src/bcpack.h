// Standalone BC1-BC7 decoder. Publish the caller's staged output only on success.
#pragma once
#include <cstdint>
#include <span>

namespace leonida::bcpack {
enum class Quality { failed, confirmed };
Quality decode(unsigned codec, std::span<const std::uint8_t> source,
               std::span<std::uint8_t> output, const char** failure = nullptr);
} // namespace leonida::bcpack
