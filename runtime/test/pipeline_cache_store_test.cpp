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

static int test_checkpoint_schedule() {
    using store::CheckpointOutcome;
    using store::CheckpointSchedule;

    CHECK(store::default_checkpoint_interval_ms == 30000);
    CHECK(store::default_checkpoint_retry_ms == 5000);
    CHECK(store::checkpoint_priority == 1);

    CheckpointSchedule schedule(30000);
    CHECK(!schedule.in_progress());
    CHECK(!schedule.try_claim(0));
    CHECK(!schedule.try_claim(29999));

    // At the due time, one caller claims the slot; concurrent callers see in_progress and are coalesced.
    CHECK(schedule.try_claim(30000));
    CHECK(schedule.in_progress());
    CHECK(!schedule.try_claim(30000));
    CHECK(!schedule.try_claim(90000));

    // A deferred checkpoint (priority-0 awaited build queued) retries after retry_ms (5 s) rather than burning the
    // full 30 s cooldown.
    schedule.complete(30050, CheckpointOutcome::deferred);
    CHECK(!schedule.in_progress());
    CHECK(schedule.next_due_ms() == 35050);
    CHECK(!schedule.try_claim(35049));
    CHECK(schedule.try_claim(35050));

    // Queue failure (abort_claim) also retries after retry_ms.
    schedule.abort_claim(35060);
    CHECK(!schedule.in_progress());
    CHECK(schedule.next_due_ms() == 40060);
    CHECK(!schedule.try_claim(40059));
    CHECK(schedule.try_claim(40060));

    // A saved snapshot advances the deadline by interval_ms (30 s).
    schedule.complete(41000, CheckpointOutcome::saved);
    CHECK(!schedule.in_progress());
    CHECK(schedule.next_due_ms() == 71000);
    CHECK(!schedule.try_claim(70999));
    CHECK(schedule.try_claim(71000));

    // A skipped snapshot (empty cache, over limit, or I/O error) backs off by interval_ms so disk is not hammered.
    schedule.complete(71010, CheckpointOutcome::skipped);
    CHECK(schedule.next_due_ms() == 101010);
    CHECK(!schedule.try_claim(101009));

    // Near-UINT64_MAX timestamps saturate below the in_flight sentinel rather than wrapping or sticking in_flight.
    const std::uint64_t near_max = (std::numeric_limits<std::uint64_t>::max)() - 10;
    CheckpointSchedule hi(near_max);
    CHECK(hi.try_claim(near_max));
    hi.complete(near_max, CheckpointOutcome::saved);
    CHECK(!hi.in_progress());
    CHECK(hi.next_due_ms() == (std::numeric_limits<std::uint64_t>::max)() - 1);
    return 0;
}

int main() {
    if (const int failed = test_limit()) return failed;
    if (const int failed = test_header()) return failed;
    return test_checkpoint_schedule();
}
