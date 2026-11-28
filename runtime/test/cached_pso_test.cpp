// Portable CPU-only unit tests for runtime/src/cached_pso.h (pure C++20 stdlib, no GPU, no game files).
// Covers blob encode/check round-trips, every Check failure class, short/exact/overlong key zero-padding &
// truncation parity, single-bit tamper detection across all 104 bytes, and KeyBuilder null-pointer guards.
#include "../src/cached_pso.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);   \
        return 1;                                                               \
    }                                                                           \
} while (0)

int main() {
    using namespace leonida::cached_pso;

    Identity id{};
    id.vendor = 0x10deu;
    id.device = 0x2b85u;
    id.driver = 0x01020304u;
    for (std::uint8_t i = 0; i < 16; ++i) id.cache_uuid[i] = std::uint8_t(0xa0u + i);

    const std::string key64(64, 'a');
    const std::string key_short = "short-key";
    const std::string key_long = key64 + "ignored-suffix";

    // 1. Round-trip accepted for exact 64-byte, empty, short, and >64-byte keys.
    {
        for (const std::string& k : {key64, std::string{}, key_short, key_long}) {
            const auto blob = encode(id, k);
            CHECK(blob.size() == blob_bytes);
            CHECK(check(id, k, blob) == Check::accepted);
        }
        // Key > 64 bytes truncates to its first 64 bytes (matching padded.resize(key_bytes)).
        const auto b_long = encode(id, key_long);
        const auto b_64 = encode(id, key64);
        CHECK(b_long == b_64);
        CHECK(check(id, key64, b_long) == Check::accepted);
        CHECK(check(id, key_long, b_64) == Check::accepted);

        // Short key zero-pads trailing bytes; a key with an extra non-zero byte past short length must not match.
        const auto b_short = encode(id, key_short);
        CHECK(check(id, key_short + "x", b_short) == Check::other_description);
        CHECK(check(id, key_short, encode(id, key_short + "x")) == Check::other_description);
        // Embedded NULs inside key are compared byte-for-byte (not truncated as C-strings).
        std::string nul_a = "ab";
        nul_a.push_back('\0');
        nul_a.push_back('c');
        std::string nul_b = "ab";
        nul_b.push_back('\0');
        nul_b.push_back('d');
        CHECK(check(id, nul_a, encode(id, nul_a)) == Check::accepted);
        CHECK(check(id, nul_a, encode(id, nul_b)) == Check::other_description);
    }

    // 2. Single-byte corruption anywhere in the 104-byte blob (and size mismatches) returns Check::invalid.
    {
        const auto blob = encode(id, key64);
        CHECK(check(id, key64, {}) == Check::invalid);
        CHECK(check(id, key64, std::span<const std::uint8_t>(blob.data(), blob.size() - 1)) == Check::invalid);
        std::vector<std::uint8_t> extended = blob;
        extended.push_back(0);
        CHECK(check(id, key64, extended) == Check::invalid);
        for (std::size_t i = 0; i < blob.size(); ++i) {
            auto tampered = blob;
            tampered[i] ^= 0x01u;
            CHECK(check(id, key64, tampered) == Check::invalid);
        }
        // Bad magic / bad version with recomputed valid FNV-1a checksum still returns Check::invalid.
        auto bad_magic = blob;
        bad_magic[0] ^= 0xffu;
        write32(bad_magic.data() + blob_bytes - 4, fnv1a(bad_magic.data(), blob_bytes - 4));
        CHECK(check(id, key64, bad_magic) == Check::invalid);

        auto bad_ver = blob;
        write32(bad_ver.data() + 4, version + 1);
        write32(bad_ver.data() + blob_bytes - 4, fnv1a(bad_ver.data(), blob_bytes - 4));
        CHECK(check(id, key64, bad_ver) == Check::invalid);
    }

    // 3. Adapter, driver, UUID, and description mismatches return the exact D3D12-mapped Check values.
    {
        const auto blob = encode(id, key64);
        Identity other_vendor = id;
        other_vendor.vendor ^= 1u;
        CHECK(check(other_vendor, key64, blob) == Check::other_adapter);

        Identity other_device = id;
        other_device.device ^= 1u;
        CHECK(check(other_device, key64, blob) == Check::other_adapter);

        Identity other_driver = id;
        other_driver.driver += 1u;
        CHECK(check(other_driver, key64, blob) == Check::other_driver);

        Identity other_uuid = id;
        other_uuid.cache_uuid[15] ^= 0x80u;
        CHECK(check(other_uuid, key64, blob) == Check::other_driver);

        std::string other_key = key64;
        other_key[63] = 'b';
        CHECK(check(id, other_key, blob) == Check::other_description);
    }

    // 4. KeyBuilder encoding, reserve(), and null-pointer safety in tag()/bytes()/text().
    {
        KeyBuilder kb;
        kb.reserve(64).tag("pso").u32(0x11223344u).text("main").text(nullptr).bytes(nullptr, 16).tag(nullptr);
        const auto& d = kb.data();
        // tag("pso"): 4 + 3 = 7; u32: 4; text("main"): 4 + 4 = 8; text(nullptr): 4;
        // bytes(nullptr, 16): 4 (clamped to 0); tag(nullptr): 4 -> total 31 bytes.
        CHECK(d.size() == 31u);
        CHECK(get32(d.data() + 0) == 3u && std::memcmp(d.data() + 4, "pso", 3) == 0);
        CHECK(get32(d.data() + 7) == 0x11223344u);
        CHECK(get32(d.data() + 11) == 4u && std::memcmp(d.data() + 15, "main", 4) == 0);
        CHECK(get32(d.data() + 19) == 0xffffffffu);
        CHECK(get32(d.data() + 23) == 0u);
        CHECK(get32(d.data() + 27) == 0u);
    }

    std::printf("cached_pso_test: all checks passed\n");
    return 0;
}
