// A shader-visible V# keeps 48 bits of (device address - heap_base), read by every translated shader as a value
// sign-extended from bit 47 (docs/design.md, "Heap-relative buffer device addresses"): [0, 2^47) above the heap and
// [2^64 - 2^47, 2^64) below it. [2^47, 2^48) is not representable: a base there is read 2^48 lower.
#pragma once
#include <cstdint>

namespace leonida {

inline constexpr bool vsharp_offset_fits(std::uint64_t offset, std::uint64_t bytes) {
    constexpr std::uint64_t kMask = (1ull << 48) - 1, kLimit = kMask / 2;
    if (!bytes) return false;
    if (offset <= kLimit) return bytes - 1 <= kLimit - offset;
    return offset >= UINT64_MAX - kLimit && bytes - 1 <= kLimit - offset; // unsigned distance across zero
}

} // namespace leonida
