// Unit tests for TitleHostRegions (runtime/src/title_host_regions.h).

#include "title_host_regions.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

std::atomic<int> failures{0};

void check(bool ok, const char* what) {
    if (ok) {
        return;
    }
    std::printf("FAIL: %s\n", what);
    failures.fetch_add(1, std::memory_order_relaxed);
}

void test_basic_register_and_lookup() {
    leonida::TitleHostRegions table;
    check(table.empty(), "starts empty");
    check(!table.find(0x1000).found, "empty table find misses");
    check(!table.resolve_bda(0x1000).has_value(), "empty table resolve_bda misses");

    // Zero host_va or zero size is ignored.
    table.register_range(0, 0x1000, 1);
    table.register_range(0x1000, 0, 2);
    check(table.empty(), "zero host_va or size ignored");

    table.register_range(0x10000, 0x1000, 0xAA);
    table.register_range(0x20000, 0x2000, 0xBB);
    check(table.size() == 2, "two disjoint regions registered");

    check(!table.find(0).found, "va=0 never matches");
    check(!table.find(0x0FFFF).found, "one byte before first region misses");
    check(!table.find(0x11000).found, "one byte past first region misses");
    check(!table.find(0x1FFFF).found, "gap before second region misses");
    check(!table.find(0x22000).found, "one byte past second region misses");

    auto l1 = table.find(0x10000);
    check(l1.found && l1.host_va == 0x10000 && l1.size == 0x1000 && l1.gpu_bda == 0 && l1.offset == 0,
          "first byte of first region");

    auto l2 = table.find(0x10FFF);
    check(l2.found && l2.host_va == 0x10000 && l2.size == 0x1000 && l2.offset == 0x0FFF,
          "last byte of first region");

    auto l3 = table.find(0x20500);
    check(l3.found && l3.host_va == 0x20000 && l3.size == 0x2000 && l3.offset == 0x500,
          "interior byte of second region");

    // Before mark_imported, resolve_bda must return nullopt.
    check(!table.resolve_bda(0x10000).has_value(), "unimported region has no BDA");
    check(!table.resolve_bda(0x20500).has_value(), "unimported second region has no BDA");
}

void test_import_resolve_and_retire() {
    leonida::TitleHostRegions table;
    table.register_range(0x100000, 0x4000, 1);
    table.register_range(0x200000, 0x8000, 2);

    auto pending = table.pending_imports();
    check(pending.size() == 2, "two pending imports");
    check(pending[0].host_va == 0x100000 && pending[0].size == 0x4000 && pending[0].tag == 1,
          "first pending import matches");
    check(pending[1].host_va == 0x200000 && pending[1].size == 0x8000 && pending[1].tag == 2,
          "second pending import matches");

    table.mark_imported(0x100000, 0x4000, 0x8000'0000ULL, 0x1111ULL);
    pending = table.pending_imports();
    check(pending.size() == 1 && pending[0].host_va == 0x200000, "only second region remains pending");

    auto bda0 = table.resolve_bda(0x100000);
    auto bda_mid = table.resolve_bda(0x101234);
    auto bda_end = table.resolve_bda(0x103FFF);
    check(bda0.has_value() && *bda0 == 0x8000'0000ULL, "resolve_bda at region start");
    check(bda_mid.has_value() && *bda_mid == 0x8000'1234ULL, "resolve_bda at region interior");
    check(bda_end.has_value() && *bda_end == 0x8000'3FFFULL, "resolve_bda at region last byte");
    check(!table.resolve_bda(0x104000).has_value(), "resolve_bda past region end");

    // Unregistering an imported region detaches its handle into retired.
    table.unregister_range(0x100000, 0x4000);
    check(!table.find(0x100000).found, "unregistered region is gone");
    check(!table.resolve_bda(0x100000).has_value(), "unregistered region BDA is gone");

    auto retired = table.take_retired();
    check(retired.size() == 1, "one retired entry after unregister");
    check(retired[0].host_va == 0x100000 && retired[0].size == 0x4000 &&
              retired[0].gpu_bda == 0x8000'0000ULL && retired[0].handle == 0x1111ULL,
          "retired entry preserves handle and BDA");

    // Second take_retired must be empty.
    check(table.take_retired().empty(), "take_retired drains the queue");

    // Unregistering an unimported region does not emit a retired entry.
    table.unregister_range(0x200000, 0x8000);
    check(table.take_retired().empty(), "unimported region produces no retired entry");
    check(table.empty(), "table is empty after unregistering all");
}

void test_partial_unmap_splitting() {
    leonida::TitleHostRegions table;

    // 1. Punch a hole in the middle of an imported region.
    table.register_range(0x10000, 0x6000, 77); // [0x10000, 0x16000)
    table.mark_imported(0x10000, 0x6000, 0x9000'0000ULL, 0xCAFEULL);

    table.unregister_range(0x12000, 0x2000); // remove [0x12000, 0x14000)
    check(table.size() == 2, "middle unmap splits into two regions");

    auto retired = table.take_retired();
    check(retired.size() == 1 && retired[0].handle == 0xCAFEULL, "split retires original imported handle");

    auto left = table.find(0x11FFF);
    auto hole = table.find(0x12000);
    auto right = table.find(0x14000);
    check(left.found && left.host_va == 0x10000 && left.size == 0x2000 && left.gpu_bda == 0,
          "left split [0x10000, 0x12000) reset to pending");
    check(!hole.found, "hole [0x12000, 0x14000) is unmapped");
    check(right.found && right.host_va == 0x14000 && right.size == 0x2000 && right.gpu_bda == 0,
          "right split [0x14000, 0x16000) reset to pending");

    // Surviving splits should be pending re-import with original tag preserved.
    auto pending = table.pending_imports();
    check(pending.size() == 2 && pending[0].tag == 77 && pending[1].tag == 77,
          "split remainders keep tag and are pending import");

    // 2. Trim head and tail across multiple regions.
    table.unregister_range(0x0F000, 0x2000); // trims [0x10000, 0x11000) from left
    auto trimmed_left = table.find(0x11000);
    check(trimmed_left.found && trimmed_left.host_va == 0x11000 && trimmed_left.size == 0x1000,
          "head trim leaves [0x11000, 0x12000)");
    check(!table.find(0x10000).found, "trimmed head is gone");

    table.unregister_range(0x15000, 0x4000); // trims [0x15000, 0x16000) from right
    auto trimmed_right = table.find(0x14000);
    check(trimmed_right.found && trimmed_right.host_va == 0x14000 && trimmed_right.size == 0x1000,
          "tail trim leaves [0x14000, 0x15000)");
    check(!table.find(0x15000).found, "trimmed tail is gone");

    // 3. Unregister spanning multiple disjoint regions.
    table.register_range(0x20000, 0x1000, 1);
    table.register_range(0x22000, 0x1000, 2);
    table.mark_imported(0x20000, 0x1000, 0xA000'0000ULL, 10);
    table.mark_imported(0x22000, 0x1000, 0xB000'0000ULL, 20);
    table.unregister_range(0x10000, 0x20000); // covers [0x11000..0x23000)
    check(table.empty(), "spanning unregister clears all covered regions");
    retired = table.take_retired();
    check(retired.size() == 2 && retired[0].handle == 10 && retired[1].handle == 20,
          "spanning unregister retires both imported handles");
}

void test_overlapping_reregister_and_stale_import() {
    leonida::TitleHostRegions table;
    table.register_range(0x50000, 0x4000, 1);
    table.mark_imported(0x50000, 0x4000, 0xC000'0000ULL, 0x5555ULL);

    // Registering an overlapping range splits/replaces the stale entry and retires its handle.
    table.register_range(0x51000, 0x2000, 2);
    check(table.size() == 3, "overlapping register splits old region into 3 parts");
    auto retired = table.take_retired();
    check(retired.size() == 1 && retired[0].handle == 0x5555ULL, "overlapping register retires stale handle");

    auto mid = table.find(0x51500);
    check(mid.found && mid.host_va == 0x51000 && mid.size == 0x2000 && mid.gpu_bda == 0,
          "new overlapping region is pending");

    // Stale mark_imported for a region that was already unregistered must retire the handle rather than leak it.
    table.mark_imported(0x999000, 0x1000, 0xD000'0000ULL, 0xDEADULL);
    retired = table.take_retired();
    check(retired.size() == 1 && retired[0].handle == 0xDEADULL,
          "stale mark_imported for missing region retires handle");
}

void test_concurrent_access() {
    leonida::TitleHostRegions table;
    constexpr int kThreads = 4;
    constexpr int kIters = 200;
    std::atomic<bool> stop{false};

    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t]() {
            const std::uint64_t base = 0x1000'0000ULL + static_cast<std::uint64_t>(t) * 0x100'0000ULL;
            for (int i = 0; i < kIters; ++i) {
                const std::uint64_t va = base + static_cast<std::uint64_t>(i) * 0x1000ULL;
                table.register_range(va, 0x1000, static_cast<std::uint64_t>(t));
                table.mark_imported(va, 0x1000, 0x8000'0000ULL + va, va);
                auto bda = table.resolve_bda(va + 0x100);
                check(bda.has_value() && *bda == 0x8000'0000ULL + va + 0x100, "concurrent resolve_bda");
                table.unregister_range(va, 0x800);
            }
        });
    }

    std::thread reader([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            (void)table.pending_imports();
            (void)table.take_retired();
            (void)table.find(0x1000'0100ULL);
        }
    });

    for (auto& w : workers) {
        w.join();
    }
    stop.store(true, std::memory_order_relaxed);
    reader.join();

    (void)table.take_retired();
    check(table.size() == kThreads * kIters, "all half-unmapped regions remain");
}

} // namespace

int main() {
    test_basic_register_and_lookup();
    test_import_resolve_and_retire();
    test_partial_unmap_splitting();
    test_overlapping_reregister_and_stale_import();
    test_concurrent_access();

    if (failures.load() != 0) {
        std::printf("FAILED: %d check(s)\n", failures.load());
        return 1;
    }
    std::printf("PASS: all title_host_regions tests passed\n");
    return 0;
}
