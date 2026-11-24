// Standalone synthetic unit tests for runtime/src/cached_pso.h (no GPU, no game data).
#include "../src/cached_pso.h"

#include <cstdio>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check_cond(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failed;
    }
}

#define CHECK(expr) check_cond((expr), #expr, __LINE__)

using leonida::cached_pso::Check;
using leonida::cached_pso::Identity;
using leonida::cached_pso::Key;
using leonida::cached_pso::KeyBuilder;

Identity make_identity() {
    Identity id{};
    id.vendor = 0x10deu;
    id.device = 0x2504u;
    id.driver = 0x00123456u;
    for (std::uint8_t i = 0; i < 16; ++i) id.cache_uuid[i] = static_cast<std::uint8_t>(i + 1);
    return id;
}

Key make_hex_key(char fill) {
    return Key(leonida::cached_pso::key_bytes, fill);
}

void test_round_trip_and_mismatches() {
    const Identity id = make_identity();
    const Key key = make_hex_key('a');
    const std::vector<std::uint8_t> blob = leonida::cached_pso::encode(id, key);
    CHECK(blob.size() == leonida::cached_pso::blob_bytes);
    CHECK(leonida::cached_pso::check(id, key, blob) == Check::accepted);

    Identity other_vendor = id;
    other_vendor.vendor ^= 1u;
    CHECK(leonida::cached_pso::check(other_vendor, key, blob) == Check::other_adapter);

    Identity other_device = id;
    other_device.device ^= 1u;
    CHECK(leonida::cached_pso::check(other_device, key, blob) == Check::other_adapter);

    Identity other_driver = id;
    other_driver.driver ^= 1u;
    CHECK(leonida::cached_pso::check(other_driver, key, blob) == Check::other_driver);

    Identity other_uuid = id;
    other_uuid.cache_uuid[15] ^= 0xffu;
    CHECK(leonida::cached_pso::check(other_uuid, key, blob) == Check::other_driver);

    CHECK(leonida::cached_pso::check(id, make_hex_key('b'), blob) == Check::other_description);
}

void test_corruption_and_size_and_null_blob() {
    const Identity id = make_identity();
    const Key key = make_hex_key('c');
    const std::vector<std::uint8_t> blob = leonida::cached_pso::encode(id, key);

    for (std::size_t i = 0; i < blob.size(); ++i) {
        std::vector<std::uint8_t> mutated = blob;
        mutated[i] ^= 0x01u;
        CHECK(leonida::cached_pso::check(id, key, mutated) == Check::invalid);
    }

    CHECK(leonida::cached_pso::check(id, key, std::span<const std::uint8_t>(blob.data(), blob.size() - 1)) ==
          Check::invalid);
    CHECK(leonida::cached_pso::check(id, key, std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(nullptr), leonida::cached_pso::blob_bytes)) ==
          Check::invalid);
}

void test_overlong_key_rejected_in_check() {
    const Identity id = make_identity();
    const Key base_key = make_hex_key('d');
    const std::vector<std::uint8_t> blob = leonida::cached_pso::encode(id, base_key);

    const Key overlong_key = base_key + "extra";
    CHECK(leonida::cached_pso::check(id, overlong_key, blob) == Check::other_description);
}

void test_null_guards_in_fnv1a_and_key_builder() {
    CHECK(leonida::cached_pso::fnv1a(nullptr, 16) == 0x811c9dc5u);

    KeyBuilder kb_null_tag;
    kb_null_tag.tag(nullptr);
    KeyBuilder kb_empty_tag;
    kb_empty_tag.tag("");
    CHECK(kb_null_tag.data() != kb_empty_tag.data());

    KeyBuilder kb_null_bytes;
    kb_null_bytes.bytes(nullptr, 8);
    CHECK(kb_null_bytes.data().size() == 4);
    for (std::uint8_t b : kb_null_bytes.data()) CHECK(b == 0);

    KeyBuilder kb_null_text;
    kb_null_text.text(nullptr);
    KeyBuilder kb_empty_text;
    kb_empty_text.text("");
    CHECK(kb_null_text.data() != kb_empty_text.data());
}

}  // namespace

int main() {
    test_round_trip_and_mismatches();
    test_corruption_and_size_and_null_blob();
    test_overlong_key_rejected_in_check();
    test_null_guards_in_fnv1a_and_key_builder();
    if (g_failed != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("cached_pso_test: OK\n");
    return 0;
}
