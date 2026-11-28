// ImageMemoryPool and BlockRanges (runtime/src/image_memory.h): textures sharing blocks of device memory. CPU only, with
// counted fake allocations: fixed cases for the block policy, then random requests checked against a list of the live
// ranges (no overlap, aligned, inside the block, and all of it usable again when everything is released).

#include "../src/image_memory.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <thread>
#include <vector>

// Single-threaded failure injection: armed only inside allocation_failures(), before threads().
static int allocations_before_failure = -1;
void* operator new(std::size_t size) {
    if (allocations_before_failure == 0) {
        allocations_before_failure = -1;
        throw std::bad_alloc();
    }
    if (allocations_before_failure > 0) --allocations_before_failure;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

using leonida::BlockRanges;
using Pool = leonida::ImageMemoryPool<std::uintptr_t>;
constexpr std::uint64_t kMiB = std::uint64_t(1) << 20;

int failures = 0;
void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

// A driver: hands out handles 1, 2, 3..., refuses when told, and notes what it freed. The pool frees a block outside
// its own lock, so the driver takes one, as vkAllocateMemory and vkFreeMemory may run on several threads.
struct Driver {
    std::mutex mutex;
    std::uintptr_t next = 1;
    std::set<std::uintptr_t> live;
    int allocations = 0;
    int frees = 0;
    bool full = false;

    Pool pool() {
        return Pool([this](std::uint32_t, std::uint64_t) -> std::uintptr_t {
            std::lock_guard lock(mutex);
            ++allocations;
            if (full) return 0;
            live.insert(next);
            return next++;
        }, [this](std::uintptr_t memory) {
            std::lock_guard lock(mutex);
            ++frees;
            check(live.erase(memory) == 1, "a block freed that was not live");
        });
    }
};

void block_ranges() {
    BlockRanges ranges(1000);
    std::uint64_t a = 0, b = 0, c = 0, d = 0;
    bool reused = true;
    check(ranges.allocate(100, 1, a, reused) && a == 0 && !reused, "the first range starts at 0 and is fresh");
    check(ranges.allocate(1, 256, b, reused) && b == 256, "a range starts at its alignment");
    check(ranges.allocate(700, 1, c, reused) && c == 257, "the best fitting free piece is used");
    check(!ranges.allocate(500, 1, d, reused), "a range larger than any free piece is refused");
    check(!ranges.allocate(0, 1, d, reused) && !ranges.allocate(1001, 1, d, reused),
          "a zero or an oversized range is refused");
    ranges.release(a, 100);
    ranges.release(c, 700);
    ranges.release(b, 1);
    check(ranges.empty() && ranges.largest() == 1000, "released ranges merge back into one piece");
    check(ranges.allocate(1000, 1, d, reused) && d == 0 && reused, "a range over earlier bytes is marked as reused");
}

void shared_blocks() {
    Driver driver;
    {
        Pool pool = driver.pool();
        std::vector<std::unique_ptr<Pool::Range>> ranges;
        for (int i = 0; i < 64; ++i) ranges.push_back(pool.allocate(2, kMiB, 65536, 64 * kMiB));
        check(driver.allocations == 1 && pool.stats().blocks == 1 && pool.stats().ranges == 64 &&
                  pool.stats().block_bytes == 64 * kMiB,
              "64 one-MiB textures fill one block");
        auto extra = pool.allocate(2, kMiB, 65536, 64 * kMiB);
        check(driver.allocations == 2, "the 65th makes a second block");
        check(extra && !extra->reused(), "a range of a new block is fresh");
        const auto other = pool.allocate(3, kMiB, 65536, 64 * kMiB);
        check(driver.allocations == 3 && other->memory() != extra->memory(), "memory types do not share blocks");
        extra.reset();
        check(driver.frees == 0 && pool.stats().blocks == 3, "an emptied block is kept for the next textures");
        ranges.clear();
        check(driver.frees == 1 && pool.stats().blocks == 2, "only one empty block is kept per memory type");
        const auto again = pool.allocate(2, kMiB, 65536, 64 * kMiB);
        check(driver.allocations == 3 && again && again->reused(), "the kept block is used again, and marked as reused");
    }
    check(driver.live.empty(), "the pool frees its blocks");
}

void a_full_heap() {
    Driver driver;
    Pool pool = driver.pool();
    driver.full = true;
    check(!pool.allocate(0, kMiB, 1024, 64 * kMiB), "a driver without room gives no range");
    const int asked = driver.allocations;
    for (int i = 0; i < 64; ++i) check(!pool.allocate(0, kMiB, 1024, 64 * kMiB), "and goes on giving none");
    check(driver.allocations == asked, "a full heap is not asked again at every texture");
    check(!pool.allocate(0, kMiB, 1024, 64 * kMiB) && driver.allocations == asked + 1, "but is asked again later");
    check(!pool.allocate(0, 2 * kMiB, 1024, kMiB) && driver.allocations == asked + 1,
          "a texture larger than a block is the caller's own allocation");

    Driver other;
    Pool second = other.pool();
    auto held = second.allocate(0, kMiB, 1024, 2 * kMiB);
    other.full = true;
    auto filling = second.allocate(0, kMiB, 1024, 2 * kMiB);
    check(filling && other.allocations == 1, "a full heap still serves the block's own free space");
    check(!second.allocate(0, kMiB, 1024, 2 * kMiB) && other.allocations == 2, "and refuses once it is used up");
    filling.reset();
    check(second.allocate(0, kMiB, 1024, 2 * kMiB) != nullptr, "a released range is usable again at once");
    held.reset();
}

void random_requests() {
    std::mt19937 random(1274);
    for (int round = 0; round < 20; ++round) {
        Driver driver;
        Pool pool = driver.pool();
        struct Live {
            std::unique_ptr<Pool::Range> range;
            std::uint64_t size;
            std::uint64_t alignment;
        };
        std::vector<Live> live;
        for (int step = 0; step < 4000; ++step) {
            if (live.empty() || random() % 100 < 55) {
                const std::uint64_t alignment = std::uint64_t(1) << (8 + random() % 9);
                const std::uint64_t size = 1 + random() % (2 * kMiB);
                auto range = pool.allocate(1, size, alignment, 16 * kMiB);
                check(range != nullptr, "a driver with room serves a small range");
                if (!range) return;
                check(range->offset() % alignment == 0 && range->offset() + size <= 16 * kMiB, "aligned, inside the block");
                for (const auto& l : live)
                    check(l.range->memory() != range->memory() || l.range->offset() >= range->offset() + size ||
                              range->offset() >= l.range->offset() + l.size,
                          "live ranges do not overlap");
                live.push_back({std::move(range), size, alignment});
            } else {
                const std::size_t i = random() % live.size();
                live[i] = std::move(live.back());
                live.pop_back();
            }
        }
        check(pool.stats().ranges == live.size(), "the range count follows");
        live.clear();
        check(pool.stats().ranges == 0 && pool.stats().blocks == 1, "everything released leaves one spare block");
        const auto whole = pool.allocate(1, 16 * kMiB, 1, 16 * kMiB);
        check(whole && whole->offset() == 0, "the whole block is free again");
    }
}

void invalid_requests() {
    BlockRanges ranges(1000);
    std::uint64_t a = 0;
    bool reused = false;
    check(ranges.allocate(100, 1, a, reused) && a == 0, "baseline range allocated");
    ranges.release(0, 0);
    ranges.release(100, 0);
    check(ranges.in_use() == 100 && ranges.largest() == 900, "zero-size release leaves ranges untouched");

    Driver driver;
    Pool pool = driver.pool();
    check(!pool.allocate(0, 0, 256, 64 * kMiB), "zero size is refused");
    check(!pool.allocate(0, kMiB, 0, 64 * kMiB), "zero alignment is refused");
    check(driver.allocations == 0 && driver.frees == 0 && pool.stats().blocks == 0 && pool.stats().ranges == 0,
          "invalid requests do not ask the driver for a block");

    driver.full = true;
    check(!pool.allocate(0, kMiB, 1024, 64 * kMiB) && driver.allocations == 1, "first OOM arms retry skip");
    for (int i = 0; i < 10; ++i) {
        check(!pool.allocate(0, 0, 1024, 64 * kMiB), "zero size while full returns null");
        check(!pool.allocate(0, kMiB, 0, 64 * kMiB), "zero alignment while full returns null");
    }
    for (int i = 0; i < 64; ++i) check(!pool.allocate(0, kMiB, 1024, 64 * kMiB), "full skip window intact");
    check(driver.allocations == 1, "invalid requests do not consume the OOM retry skip window");
}

void allocation_failures() {
    // BlockRanges::allocate with both a lead gap and a tail remainder must roll back cleanly on std::bad_alloc.
    for (int fail_after = 0; fail_after < 4; ++fail_after) {
        BlockRanges ranges(4096);
        std::uint64_t a = 0, b = 0;
        bool reused = false;
        check(ranges.allocate(64, 1, a, reused) && a == 0, "seed prefix");
        bool threw = false;
        allocations_before_failure = fail_after;
        try {
            (void)ranges.allocate(128, 256, b, reused);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        allocations_before_failure = -1;
        check(threw, "injected BlockRanges allocation failure reached");
        check(ranges.in_use() == 64 && ranges.largest() == 4032,
              "failed BlockRanges::allocate preserves free space and in_use");
        check(ranges.allocate(4032, 1, b, reused) && b == 64 && ranges.in_use() == 4096 && ranges.largest() == 0,
              "entire remaining space is still usable after failed BlockRanges::allocate");
    }

    // ImageMemoryPool::allocate on both the new-block path and the existing-block path must not leak driver memory,
    // corrupt stats().ranges, lose free space, or invoke free_ while holding mutex_.
    for (bool existing_block : {false, true}) {
        int threw_count = 0;
        for (int fail_after = 0; fail_after < 8; ++fail_after) {
            Driver driver;
            {
                Pool* pool_ptr = nullptr;
                Pool pool(
                    [&](std::uint32_t, std::uint64_t) -> std::uintptr_t {
                        std::lock_guard lock(driver.mutex);
                        ++driver.allocations;
                        const std::uintptr_t h = driver.next++;
                        driver.live.insert(h);
                        return h;
                    },
                    [&](std::uintptr_t memory) {
                        if (pool_ptr) (void)pool_ptr->stats();  // must not deadlock: free_ runs outside pool mutex_
                        std::lock_guard lock(driver.mutex);
                        ++driver.frees;
                        check(driver.live.erase(memory) == 1, "failed allocate frees only live driver memory");
                    });
                pool_ptr = &pool;
                std::unique_ptr<Pool::Range> seed;
                if (existing_block) {
                    seed = pool.allocate(1, 64, 1, 4096);
                    check(seed != nullptr, "seed existing block");
                }
                const auto expected_blocks = existing_block ? 1u : 0u;
                const auto expected_ranges = existing_block ? 1u : 0u;
                bool threw = false;
                std::unique_ptr<Pool::Range> probe;
                allocations_before_failure = fail_after;
                try {
                    probe = pool.allocate(1, 128, 256, 4096);
                } catch (const std::bad_alloc&) {
                    threw = true;
                }
                allocations_before_failure = -1;
                if (threw) {
                    ++threw_count;
                    check(!probe, "failed allocate returned nothing");
                    check(pool.stats().blocks == expected_blocks && pool.stats().ranges == expected_ranges,
                          "failed pool.allocate preserves block and range counts");
                    check(driver.live.size() == expected_blocks, "failed pool.allocate leaks no driver memory");
                    if (existing_block) {
                        auto rest = pool.allocate(1, 4032, 1, 4096);
                        check(rest && rest->offset() == 64 && pool.stats().blocks == 1,
                              "failed existing-block allocate leaks no bytes inside the block");
                    }
                }
                probe.reset();
                seed.reset();
                pool_ptr = nullptr;
            }
            check(driver.live.empty(), "pool teardown frees all driver memory after failure injection");
        }
        check(threw_count > 0, "injected pool.allocate failures were exercised");
    }
}

void threads() {
    Driver driver;
    {
        Pool pool = driver.pool();
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < 4; ++t)
            workers.emplace_back([&pool, t] {
                std::mt19937 random(t);
                std::vector<std::unique_ptr<Pool::Range>> mine;
                for (int i = 0; i < 2000; ++i) {
                    if (mine.size() < 16) mine.push_back(pool.allocate(0, 1 + random() % kMiB, 4096, 16 * kMiB));
                    else mine.erase(mine.begin() + random() % mine.size());
                }
            });
        for (auto& worker : workers) worker.join();
        check(pool.stats().ranges == 0, "concurrent use leaves no range behind");
    }
    check(driver.live.empty(), "nor a block");
}

}  // namespace

int main() {
    block_ranges();
    invalid_requests();
    shared_blocks();
    a_full_heap();
    allocation_failures();
    random_requests();
    threads();
    if (!failures) std::printf("image memory: textures share blocks, and released ranges are used again\n");
    return failures ? 1 : 0;
}
