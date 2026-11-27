// src/title_heap.h: the XMemAlloc heap the runtime keeps in the page mirror's range. No Windows headers, no GPU, no
// game data: the "reservation" is host memory here, and commit and decommit are counted.
//   1. blocks are 64-byte aligned, a size of zero is one byte, sizes fall in their class, large blocks are whole pages;
//   2. a randomised run of allocations and frees against a reference (a map of live blocks with their bytes): no two
//      live blocks overlap, every block keeps what was written into it until it is freed, frees give space back;
//   3. a large block is committed when handed out and decommitted when freed; pointers the heap did not hand out are
//      refused.
#include "../src/title_heap.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <vector>

namespace {

int failures = 0;
#define EXPECT(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);    \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using leonida::TitleHeap;

}  // namespace

int main() {
    constexpr std::size_t kBytes = std::size_t(256) << 20;
    std::vector<std::uint8_t> memory(kBytes + TitleHeap::kPage);
    const std::uintptr_t base =
        (reinterpret_cast<std::uintptr_t>(memory.data()) + TitleHeap::kPage - 1) & ~std::uintptr_t(TitleHeap::kPage - 1);
    std::size_t committed = 0, decommitted = 0;
    TitleHeap heap;
    TitleHeap::Hooks hooks;
    hooks.commit = [&](std::uintptr_t, std::size_t bytes) {
        committed += bytes;
        return true;
    };
    hooks.decommit = [&](std::uintptr_t at, std::size_t bytes) {
        decommitted += bytes;
        std::memset(reinterpret_cast<void*>(at), 0, bytes);  // as a decommit reads back
    };
    EXPECT(heap.init(base, kBytes, hooks));

    // 1.
    void* zero = heap.allocate(0);
    EXPECT(zero && reinterpret_cast<std::uintptr_t>(zero) % 64 == 0 && heap.size_of(zero) == 64);
    for (std::size_t size : {1u, 63u, 64u, 65u, 1000u, 1024u, 1025u, 4096u, 30000u, 32768u}) {
        void* p = heap.allocate(size);
        EXPECT(p && reinterpret_cast<std::uintptr_t>(p) % 64 == 0 && heap.size_of(p) >= size && heap.owns(p));
        EXPECT(heap.release(p));
    }
    void* large = heap.allocate(100000);
    EXPECT(large && reinterpret_cast<std::uintptr_t>(large) % TitleHeap::kPage == 0 &&
           heap.size_of(large) == 2 * TitleHeap::kPage);
    // 3.
    const std::size_t before = decommitted;
    std::memset(large, 0x5a, 100000);
    EXPECT(heap.release(large) && decommitted == before + 2 * TitleHeap::kPage);
    void* again = heap.allocate(100000);
    EXPECT(again && static_cast<std::uint8_t*>(again)[0] == 0);  // reads zero handed out again
    EXPECT(heap.release(again));
    int outside = 0;
    EXPECT(!heap.release(&outside) && !heap.owns(&outside));
    EXPECT(!heap.release(static_cast<std::uint8_t*>(zero) + 8));  // not a block's start
    EXPECT(heap.release(zero));

    // 2.
    std::mt19937 random(2026);
    struct Live {
        std::size_t size;
        std::uint8_t tag;
    };
    std::map<std::uintptr_t, Live> live;
    const auto check_block = [&](std::uintptr_t at, const Live& b) {
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(at);
        for (std::size_t i = 0; i < b.size; i += 97)
            if (bytes[i] != std::uint8_t(b.tag + i)) return false;
        return true;
    };
    for (int round = 0; round < 20000; ++round) {
        const bool free_one = !live.empty() && (random() % 100 < 45 || live.size() > 3000);
        if (free_one) {
            auto it = live.begin();
            std::advance(it, random() % live.size());
            EXPECT(check_block(it->first, it->second));
            EXPECT(heap.release(reinterpret_cast<void*>(it->first)));
            live.erase(it);
            continue;
        }
        const unsigned kind = random() % 100;
        const std::size_t size = kind < 70 ? random() % 512 : kind < 95 ? random() % 32768 : 32769 + random() % 300000;
        void* p = heap.allocate(size);
        EXPECT(p != nullptr);
        if (!p) break;
        const auto at = reinterpret_cast<std::uintptr_t>(p);
        EXPECT(at % 64 == 0 && heap.owns(p));
        const std::size_t span = size ? size : 1;
        // No live block overlaps it.
        auto next = live.lower_bound(at);
        if (next != live.end()) EXPECT(at + span <= next->first);
        if (next != live.begin()) {
            const auto previous = std::prev(next);
            EXPECT(previous->first + previous->second.size <= at);
        }
        const Live b{span, std::uint8_t(random())};
        auto* bytes = static_cast<std::uint8_t*>(p);
        for (std::size_t i = 0; i < span; i += 97) bytes[i] = std::uint8_t(b.tag + i);
        live[at] = b;
    }
    for (const auto& [at, b] : live) {
        EXPECT(check_block(at, b));
        EXPECT(heap.release(reinterpret_cast<void*>(at)));
    }
    EXPECT(heap.stats().live == 0 && committed > 0);

    if (failures) {
        std::fprintf(stderr, "title_heap_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("title_heap_test: alignment, classes, large blocks, 20000 random allocations and frees against a "
                "reference (%zu KiB committed, %zu KiB decommitted)\n",
                committed >> 10, decommitted >> 10);
    return 0;
}
