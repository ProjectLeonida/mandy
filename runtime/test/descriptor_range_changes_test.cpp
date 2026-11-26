// Standalone synthetic unit tests for runtime/src/descriptor_range_changes.h (no GPU, no game data).
#include "../src/descriptor_range_changes.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_failed = 0;

void check_cond(bool ok, const char* expr, int line) {
    if (!ok) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expr);
        ++g_failed;
    }
}

#define CHECK(expr) check_cond((expr), #expr, __LINE__)

void test_merge_adjacent_and_uint64_max() {
    leonida::DescriptorRanges dr;
    dr.ranges.push_back({10, 19});
    dr.ranges.push_back({0, 9});
    dr.ranges.push_back({20, 29});
    dr.ranges.push_back({UINT64_MAX - 5, UINT64_MAX});
    dr.ranges.push_back({UINT64_MAX, UINT64_MAX});
    dr.merge();

    CHECK(dr.ranges.size() == 2);
    CHECK(dr.ranges[0].first == 0 && dr.ranges[0].last == 29);
    CHECK(dr.ranges[1].first == UINT64_MAX - 5 && dr.ranges[1].last == UINT64_MAX);
    CHECK(dr.overlaps(0, 1));
    CHECK(dr.overlaps(29, 10));
    CHECK(!dr.overlaps(30, 5));
    CHECK(dr.overlaps(UINT64_MAX, 1));
    CHECK(!dr.overlaps(10, 0));
}

void test_merge_skips_inverted_ranges() {
    leonida::DescriptorRanges dr;
    dr.ranges.push_back({100, 50});   // inverted: must be ignored
    dr.ranges.push_back({20, 30});
    dr.ranges.push_back({10, 19});
    dr.merge();

    CHECK(dr.ranges.size() == 1);
    CHECK(dr.ranges[0].first == 10 && dr.ranges[0].last == 30);
    CHECK(dr.overlaps(15, 1));
    CHECK(!dr.overlaps(60, 10));
}

void test_ring_record_read_and_overflow() {
    leonida::DescriptorRangeChanges log;
    uint64_t cursor = 0;
    leonida::DescriptorRanges out;

    log.record(100, 0);   // empty: ignored
    log.record(100, 16);  // [100, 115]
    log.record(116, 16);  // [116, 131]
    CHECK(log.read(cursor, out));
    out.merge();
    CHECK(out.ranges.size() == 1);
    CHECK(out.ranges[0].first == 100 && out.ranges[0].last == 131);

    // Reading again with up-to-date cursor yields nothing new.
    leonida::DescriptorRanges empty_out;
    CHECK(log.read(cursor, empty_out));
    CHECK(empty_out.ranges.empty());

    // Overflowing the 4096-entry ring forces read() to return false and advance cursor.
    for (uint64_t i = 0; i < 4097; ++i) log.record(i * 16, 8);
    leonida::DescriptorRanges overflow_out;
    CHECK(!log.read(cursor, overflow_out));
    CHECK(log.read(cursor, overflow_out));
    CHECK(overflow_out.ranges.empty());
}

}  // namespace

int main() {
    test_merge_adjacent_and_uint64_max();
    test_merge_skips_inverted_ranges();
    test_ring_record_read_and_overflow();
    if (g_failed != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("descriptor_range_changes_test: OK\n");
    return 0;
}
