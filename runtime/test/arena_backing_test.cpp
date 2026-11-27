// leonida::ArenaBacking (src/arena_backing.h): the GPU address arena's backing blocks, with fake hooks. No GPU.
//   - placement: an allocation no larger than a block never crosses one, a larger one takes whole blocks of its own;
//   - binding: a block is bound by the first allocation in it and shared by the next ones;
//   - unbinding: only once the release ticket has completed, and an allocation in a pending block revives it;
//   - the accessor: an offset resolves to its unit's mapping in one table read, an unbacked or unit-crossing range is
//     refused, offset_of() inverts it, with_host() refuses what has no memory;
//   - the arena as one allocation (no sparse binding): every range resolves;
//   - the bind queue choice: a sparse-only family first, else a sparse family with a queue to spare, else none.
// Driven together with the free-block map the device uses (src/arena_blocks.h), as DeviceState::arena_allocate does.
#include "../src/arena_backing.h"
#include "../src/arena_blocks.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr VkDeviceSize kMiB = 1 << 20;
constexpr VkDeviceSize kBlock = 2 * kMiB;

// Memory the fake driver hands out: a VkDeviceMemory is a number, its mapping a vector.
struct FakeDriver {
    std::map<std::uint64_t, std::vector<std::uint8_t>> memory;
    std::uint64_t next = 1;
    std::uint64_t binds = 0, unbinds = 0;
    std::map<VkDeviceSize, std::uint64_t> bound;  // resource offset -> memory (0: unbound)
    bool fail_make = false;

    leonida::ArenaBackingHooks hooks() {
        leonida::ArenaBackingHooks h;
        h.make = [this](VkDeviceSize bytes, VkDeviceMemory& out, std::uint8_t*& host) {
            if (fail_make) return false;
            const std::uint64_t id = next++;
            auto& bytes_of = memory[id];
            bytes_of.assign(std::size_t(bytes), 0);
            out = reinterpret_cast<VkDeviceMemory>(static_cast<std::uintptr_t>(id));
            host = bytes_of.data();
            return true;
        };
        h.destroy = [this](VkDeviceMemory m) { memory.erase(reinterpret_cast<std::uintptr_t>(m)); };
        h.bind = [this](const VkSparseMemoryBind* b, std::uint32_t n) {
            for (std::uint32_t i = 0; i < n; ++i) {
                const auto id = reinterpret_cast<std::uintptr_t>(b[i].memory);
                if (id) {
                    ++binds;
                    bound[b[i].resourceOffset] = id;
                } else {
                    ++unbinds;
                    bound.erase(b[i].resourceOffset);
                }
            }
            return true;
        };
        return h;
    }
};

// DeviceState::arena_allocate's steps: footprint, alignment, the keep-out, the block map, then acquire.
bool allocate(leonida::ArenaBlocks& blocks, leonida::ArenaBacking& backing, VkDeviceSize asked, VkDeviceSize alignment,
              VkDeviceSize& out) {
    const VkDeviceSize bytes = backing.footprint(asked);
    if (!blocks.allocate(bytes, backing.alignment(asked, alignment),
                         [&](std::uint64_t lo, std::uint64_t hi) { return backing.keep_out(lo, hi); }, out))
        return false;
    if (backing.acquire(out, bytes)) return true;
    blocks.release(out, bytes);
    return false;
}
void release(leonida::ArenaBlocks& blocks, leonida::ArenaBacking& backing, VkDeviceSize offset, VkDeviceSize asked,
             std::uint64_t ticket) {
    const VkDeviceSize bytes = backing.footprint(asked);
    if (blocks.release(offset, bytes) == leonida::ArenaBlocks::Fault::none) backing.release(offset, bytes, ticket);
}

int placement_and_binding() {
    FakeDriver driver;
    leonida::ArenaBacking backing;
    EXPECT(backing.init_sparse(64 * kMiB, kBlock, driver.hooks()));
    leonida::ArenaBlocks blocks;
    blocks.reset(0x10000, 64 * kMiB);
    EXPECT(backing.acquire(0, 0x10000, true));  // the head (zero page, quad indices)
    EXPECT(backing.stats().bound == kBlock && driver.binds == 1);

    // Small allocations share the head's block, then move to the next block rather than cross into it.
    VkDeviceSize a = 0, b = 0, c = 0;
    EXPECT(allocate(blocks, backing, 1 * kMiB, 256, a));
    EXPECT(a == 0x10000 && driver.binds == 1);
    EXPECT(allocate(blocks, backing, 1 * kMiB, 256, b));  // [0x110000, +1 MiB) would cross 2 MiB
    EXPECT(b == kBlock && driver.binds == 2);
    EXPECT(allocate(blocks, backing, 512 * 1024, 256, c));
    EXPECT(c == a + kMiB && driver.binds == 2);  // first fit: the rest of the head's block
    EXPECT(backing.stats().bound == 2 * kBlock);

    // The accessor: each allocation is one contiguous run of one mapping; a range that crosses units is refused.
    std::uint8_t* pa = backing.host(a, kMiB);
    std::uint8_t* pb = backing.host(b, kMiB);
    std::uint8_t* pc = backing.host(c, 512 * 1024);
    EXPECT(pa && pb && pc && pc == pa + kMiB);
    std::memset(pb, 0xab, std::size_t(kMiB));  // writable through its whole length
    EXPECT(!backing.host(kBlock - 16, 32));     // crosses from the head's unit into b's
    EXPECT(!backing.host(8 * kMiB, 4));         // nothing there
    EXPECT(!backing.host(64 * kMiB - 4, 8));    // past the arena
    std::uint64_t back = 0;
    EXPECT(backing.offset_of(pc + 100, 16, back) && back == c + 100);
    EXPECT(!backing.offset_of(&back, 8, back));
    bool called = false;
    EXPECT(!backing.with_host(8 * kMiB, 4, [&](std::uint8_t*) { called = true; }) && !called);
    EXPECT(backing.with_host(c, 4, [&](std::uint8_t* p) { called = p == pc; }) && called);
    leonida::ArenaBacking::Location where;
    EXPECT(backing.locate(c, 4096, where) && where.memory_offset == c && where.host == pc);

    // Larger than a block: whole blocks of its own, block-aligned, and nothing else lands in them.
    VkDeviceSize big = 0, after = 0;
    EXPECT(allocate(blocks, backing, 5 * kMiB, 256, big));
    EXPECT(big % kBlock == 0 && backing.stats().bound == 2 * kBlock + 6 * kMiB && driver.binds == 3);
    EXPECT(backing.host(big, 5 * kMiB) != nullptr);
    EXPECT(allocate(blocks, backing, 4096, 256, after));
    EXPECT(after >= big + 6 * kMiB || after < big);
    return 0;
}

int unbinding_waits_for_the_ticket() {
    FakeDriver driver;
    leonida::ArenaBacking backing;
    EXPECT(backing.init_sparse(64 * kMiB, kBlock, driver.hooks()));
    leonida::ArenaBlocks blocks;
    blocks.reset(0x10000, 64 * kMiB);
    EXPECT(backing.acquire(0, 0x10000, true));
    VkDeviceSize a = 0, b = 0;
    EXPECT(allocate(blocks, backing, 3 * kMiB, 256, a));  // a unit of two blocks
    EXPECT(allocate(blocks, backing, 64 * 1024, 256, b));  // in the head's block
    const VkDeviceSize bound = backing.stats().bound;
    EXPECT(bound == kBlock + 4 * kMiB);

    // Released while submission 7 may still use it: pending, still bound, still mapped.
    release(blocks, backing, a, 3 * kMiB, 7);
    release(blocks, backing, b, 64 * 1024, 7);  // the head is permanent: never pending
    EXPECT(backing.has_pending());
    EXPECT(backing.reclaim(6) == 0);
    EXPECT(backing.stats().bound == bound && driver.unbinds == 0 && backing.host(a, 4) != nullptr);
    // A small allocation may not land in a pending unit of several blocks (it would keep 4 MiB bound).
    VkDeviceSize small = 0;
    EXPECT(allocate(blocks, backing, 4096, 4096, small));
    EXPECT(small < a || small >= a + 4 * kMiB);
    // Submission 7 has completed: unbound, then freed; the accessor refuses it.
    EXPECT(backing.reclaim(7) == 4 * kMiB);
    EXPECT(driver.unbinds == 1 && backing.stats().bound == kBlock && !backing.host(a, 4) && !backing.has_pending());
    EXPECT(driver.memory.size() == 1);  // only the head's memory is left

    // A pending single block revived by the next allocation in it: not unbound, not bound again.
    VkDeviceSize c = 0, d = 0;
    EXPECT(allocate(blocks, backing, 64 * 1024, 256, c));
    EXPECT(allocate(blocks, backing, kBlock - 4096, 256, d));
    const std::uint64_t binds = driver.binds;
    release(blocks, backing, d, kBlock - 4096, 9);
    EXPECT(backing.has_pending());
    VkDeviceSize e = 0;
    EXPECT(allocate(blocks, backing, kBlock - 4096, 4096, e));
    EXPECT(e >> 21 == d >> 21 && driver.binds == binds && !backing.has_pending());
    EXPECT(backing.reclaim(100) == 0 && driver.unbinds == 1 && backing.host(e, 4) != nullptr);
    return 0;
}

// Many allocations of mixed sizes and frees: no allocation ever crosses a unit, and the bound bytes stay within the
// blocks the live allocations touch.
int random_workload() {
    FakeDriver driver;
    leonida::ArenaBacking backing;
    constexpr VkDeviceSize size = 512 * kMiB;
    EXPECT(backing.init_sparse(size, kBlock, driver.hooks()));
    leonida::ArenaBlocks blocks;
    blocks.reset(0x10000, size);
    EXPECT(backing.acquire(0, 0x10000, true));
    std::mt19937_64 rng(12345);
    struct Live { VkDeviceSize offset, bytes; };
    std::vector<Live> live;
    std::uint64_t ticket = 1;
    for (int step = 0; step < 20000; ++step) {
        if (live.size() < 400 && (rng() % 3 || live.empty())) {
            // Mostly small (as measured: most arena allocations are below 256 KiB), some larger than a block.
            VkDeviceSize bytes = rng() % 10 ? 256 + rng() % (256 * 1024) : kBlock + rng() % (6 * kMiB);
            VkDeviceSize at = 0;
            if (!allocate(blocks, backing, bytes, 256, at)) continue;
            EXPECT(backing.host(at, bytes) != nullptr);
            live.push_back({at, bytes});
        } else {
            const std::size_t i = std::size_t(rng() % live.size());
            release(blocks, backing, live[i].offset, live[i].bytes, ticket);
            live[i] = live.back();
            live.pop_back();
        }
        if (step % 50 == 0) backing.reclaim(ticket++ - 1);
    }
    backing.reclaim(ticket);
    std::map<std::uint64_t, bool> touched;
    touched[0] = true;  // the head
    for (const auto& l : live) {
        EXPECT(backing.host(l.offset, l.bytes) != nullptr);
        const VkDeviceSize footprint = backing.footprint(l.bytes);
        for (VkDeviceSize at = l.offset >> 21; at <= (l.offset + footprint - 1) >> 21; ++at) touched[at] = true;
    }
    EXPECT(backing.stats().bound == VkDeviceSize(touched.size()) * kBlock);
    EXPECT(backing.stats().bound < size);  // not the whole arena: only the blocks live allocations touch
    std::fprintf(stderr, "random workload: %zu live allocations, %llu MiB bound of %llu, %llu binds, %llu unbind "
                         "batches\n",
                 live.size(), (unsigned long long)(backing.stats().bound >> 20), (unsigned long long)(size >> 20),
                 (unsigned long long)driver.binds, (unsigned long long)backing.stats().unbinds);
    return 0;
}

// A failed bind refuses the allocation and leaves nothing behind.
int bind_failure() {
    FakeDriver driver;
    leonida::ArenaBacking backing;
    EXPECT(backing.init_sparse(16 * kMiB, kBlock, driver.hooks()));
    leonida::ArenaBlocks blocks;
    blocks.reset(0x10000, 16 * kMiB);
    driver.fail_make = true;
    VkDeviceSize at = 0;
    EXPECT(!allocate(blocks, backing, 4096, 256, at));
    EXPECT(blocks.free_bytes() == 16 * kMiB - 0x10000 && backing.stats().bound == 0);
    return 0;
}

// Without sparse binding: one unit over the whole arena, everything resolves, nothing is rounded or bound.
int one_allocation() {
    std::vector<std::uint8_t> memory(std::size_t(8 * kMiB));
    leonida::ArenaBacking backing;
    backing.init_fixed(8 * kMiB, VK_NULL_HANDLE, memory.data());
    EXPECT(!backing.sparse() && backing.footprint(5 * kMiB) == 5 * kMiB && backing.alignment(5 * kMiB, 256) == 256);
    EXPECT(backing.keep_out(kBlock - 16, kBlock + 16) == 0 && backing.placeable(kBlock - 16, 32));
    EXPECT(backing.acquire(kBlock - 16, 32));
    EXPECT(backing.host(kBlock - 16, 32) == memory.data() + kBlock - 16);
    EXPECT(backing.host(0, 8 * kMiB) == memory.data() && !backing.host(0, 8 * kMiB + 1));
    std::uint64_t back = 0;
    EXPECT(backing.offset_of(memory.data() + 77, 4, back) && back == 77);
    EXPECT(backing.stats().bound == 8 * kMiB);
    return 0;
}

int bind_queue_choice() {
    const std::uint32_t none[5] = {};
    // RADV (RX 7900 XTX, iGPU): graphics, compute, two video families and a sparse-only one.
    const VkQueueFamilyProperties radv[5] = {
        {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT, 1, 64, {1, 1, 1}},
        {VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT, 4, 64, {1, 1, 1}},
        {0x20, 1, 64, {1, 1, 1}}, {0x40, 1, 64, {1, 1, 1}},
        {VK_QUEUE_SPARSE_BINDING_BIT, 1, 64, {1, 1, 1}}};
    auto q = leonida::pick_arena_bind_queue(radv, 5, none, 16);
    EXPECT(q.family == 4 && q.index == 0);
    // No sparse-only family: a queue to spare in a sparse family, after the ones the device takes.
    const std::uint32_t used[2] = {1, 2};
    q = leonida::pick_arena_bind_queue(radv, 2, used, 16);
    EXPECT(q.family == 1 && q.index == 2);
    // llvmpipe: one family with one queue, already the device's: none (the arena stays one allocation).
    const VkQueueFamilyProperties lvp[1] = {
        {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT, 1, 64, {1, 1, 1}}};
    const std::uint32_t one[1] = {1};
    EXPECT(leonida::pick_arena_bind_queue(lvp, 1, one, 16).family == UINT32_MAX);
    // No sparse binding anywhere: none.
    const VkQueueFamilyProperties plain[1] = {{VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, 4, 64, {1, 1, 1}}};
    EXPECT(leonida::pick_arena_bind_queue(plain, 1, none, 16).family == UINT32_MAX);
    return 0;
}

}  // namespace

int main() {
    if (int r = placement_and_binding()) return r;
    if (int r = unbinding_waits_for_the_ticket()) return r;
    if (int r = random_workload()) return r;
    if (int r = bind_failure()) return r;
    if (int r = one_allocation()) return r;
    if (int r = bind_queue_choice()) return r;
    std::fprintf(stderr, "arena_backing: ok\n");
    return 0;
}
