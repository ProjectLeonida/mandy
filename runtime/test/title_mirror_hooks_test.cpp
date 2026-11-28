// The page mirror's alias for console descriptor heaps read in place (src/title_mirror.h), without a GPU: fake handles
// stand for the chunks, the units and the arena buffer, and the bind hook records what each batch binds.
//   alias(): the pages of a range are also bound into another buffer at an offset, in the same batches as their chunk
//     binds, following remaps and unmaps, each with a home of its own there (the filtered copy), which hooks.filter
//     fills (fresh when made, not fresh when the title's bytes change again, never when they did not); the page's own
//     home at the title's address keeps the title's bytes exactly and is written back as any page; unalias() unbinds
//     the alias and frees the copies. Without an alias the batches are as before. The title's bytes are only written
//     by write-back.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include "../src/title_mirror.h"

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using leonida::TitleMirror;
constexpr std::uint64_t kPage = TitleMirror::kPage;
constexpr std::uint64_t kBase = std::uint64_t(16) << 40;  // a title reservation in the mirror's range

template <class H>
H fake(std::uintptr_t n) {
    return reinterpret_cast<H>(n);
}

struct Fixture {
    std::vector<std::unique_ptr<std::uint8_t[]>> units;
    std::uintptr_t next_handle = 0x1000;
    // Per buffer, per offset: the memory and offset of the last bind (memory null: unbound).
    std::map<VkBuffer, std::map<VkDeviceSize, std::pair<VkDeviceMemory, VkDeviceSize>>> bound;
    std::vector<std::size_t> batch_sizes;  // binds per flush
    std::uint64_t last_value = 0;
    struct Filtered {
        std::uint64_t va;
        bool fresh;
    };
    std::vector<Filtered> filtered;
    std::map<VkDeviceMemory, std::uint8_t*> hosts;

    leonida::TitleMirrorHooks hooks() {
        leonida::TitleMirrorHooks h;
        h.make_chunk = [this](std::uint64_t, std::uint64_t, VkBuffer& out) {
            out = fake<VkBuffer>(next_handle++);
            return true;
        };
        h.make_unit = [this](VkDeviceMemory& out, std::uint8_t*& host) {
            units.emplace_back(new std::uint8_t[TitleMirror::kUnitBytes]());
            out = fake<VkDeviceMemory>(next_handle++);
            host = units.back().get();
            hosts[out] = host;
            return true;
        };
        h.free_unit = [](VkDeviceMemory) {};
        h.bind = [this](const std::vector<VkSparseBufferMemoryBindInfo>& infos, std::uint64_t value) {
            std::size_t n = 0;
            for (const auto& info : infos)
                for (std::uint32_t i = 0; i < info.bindCount; ++i, ++n)
                    bound[info.buffer][info.pBinds[i].resourceOffset] = {info.pBinds[i].memory,
                                                                         info.pBinds[i].memoryOffset};
            batch_sizes.push_back(n);
            last_value = value;
            return true;
        };
        // The filter of the test: copy, then turn every byte 0xA5 into 0x5A (a stand-in for nulling a V#).
        h.filter = [this](std::uint64_t va, std::uint8_t* to, const std::uint8_t* from, std::size_t bytes, bool fresh) {
            filtered.push_back({va, fresh});
            for (std::size_t i = 0; i < bytes; ++i) to[i] = from[i] == 0xA5 ? 0x5A : from[i];
            return true;
        };
        return h;
    }
    // The bytes the GPU reads at `offset` of `buffer`: the home bound there.
    const std::uint8_t* read(VkBuffer buffer, VkDeviceSize offset) {
        const auto b = bound.find(buffer);
        if (b == bound.end()) return nullptr;
        const auto page = b->second.find(offset & ~(kPage - 1));
        if (page == b->second.end() || !page->second.first) return nullptr;
        return hosts[page->second.first] + page->second.second + (offset & (kPage - 1));
    }
};

int aliases() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 16 * kPage));
    VkBuffer chunk = VK_NULL_HANDLE;
    VkDeviceSize in_chunk = 0;
    EXPECT(mirror.chunk_at(kBase, chunk, in_chunk) && in_chunk == 0);
    std::vector<std::uint8_t> frames(4 * kPage);
    for (std::size_t i = 0; i < frames.size(); ++i) frames[i] = std::uint8_t(i / kPage + 1);
    // Without an alias: two pages, one batch of two binds, as before.
    mirror.map(kBase, &frames[0], nullptr, 1);
    mirror.map(kBase + kPage, &frames[kPage], nullptr, 1);
    EXPECT(mirror.flush(1) && f.batch_sizes.back() == 2 && f.bound.size() == 1);
    // The alias binds the mapped pages' copies into the arena at its offset, in the next batch with the chunk binds.
    const VkBuffer arena = fake<VkBuffer>(0x77);
    constexpr VkDeviceSize kAt = 0xb000000;
    mirror.alias(kBase, kBase + 4 * kPage, arena, kAt);
    mirror.map(kBase + 2 * kPage, &frames[2 * kPage], nullptr, 2);  // a page mapped after the alias
    EXPECT(mirror.flush(2) && f.batch_sizes.back() == 4);           // 1 chunk bind, 3 alias binds
    for (int p = 0; p < 3; ++p) {
        const std::uint8_t* through_arena = f.read(arena, kAt + p * kPage + 17);
        const std::uint8_t* through_chunk = f.read(chunk, p * kPage + 17);
        // Memory of its own at the alias; the same bytes here, as the test's filter changes only 0xA5.
        EXPECT(through_arena && through_chunk && through_arena != through_chunk && *through_arena == p + 1 &&
               *through_chunk == p + 1);
    }
    EXPECT(!f.read(arena, kAt + 3 * kPage));  // not mapped: unbound there too
    // A remap: page 0 shows frame 3 now; the arena follows in the same batch.
    mirror.map(kBase, &frames[3 * kPage], nullptr, 3);
    EXPECT(mirror.flush(3) && f.batch_sizes.back() == 2);
    EXPECT(*f.read(arena, kAt) == 4 && *f.read(chunk, 0) == 4);
    // An unmap unbinds both.
    mirror.map(kBase + kPage, nullptr, nullptr, 4);
    EXPECT(mirror.flush(4) && !f.read(arena, kAt + kPage) && !f.read(chunk, kPage));
    // unalias: the arena pages are unbound; the chunk keeps its pages; later maps bind the chunk only.
    mirror.unalias(kBase, 5);
    EXPECT(mirror.flush(5) && !f.read(arena, kAt) && !f.read(arena, kAt + 2 * kPage) && f.read(chunk, 0));
    EXPECT(!mirror.copy_bytes(kBase));
    mirror.map(kBase + kPage, &frames[kPage], nullptr, 6);
    EXPECT(mirror.flush(6) && f.batch_sizes.back() == 1 && !f.read(arena, kAt + kPage));
    std::printf("alias: pages bound into a second buffer with copies of their own, in the chunk batches, following a "
                "map, a remap and an unmap\n");
    return 0;
}

int filters() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 16 * kPage));
    std::vector<std::uint8_t> frames(3 * kPage, 0xA5);
    const VkBuffer arena = fake<VkBuffer>(0x77);
    constexpr VkDeviceSize kAt = 0xb000000;
    mirror.alias(kBase, kBase + 2 * kPage, arena, kAt);
    mirror.map(kBase, &frames[0], nullptr, 1);                      // aliased: a copy, filtered fresh
    mirror.map(kBase + 2 * kPage, &frames[2 * kPage], nullptr, 1);  // outside: no copy
    EXPECT(f.filtered.size() == 1 && f.filtered[0].va == kBase && f.filtered[0].fresh);
    // The title's address reads the title's bytes; the alias the filter's.
    EXPECT(mirror.home_bytes(kBase)[100] == 0xA5 && mirror.copy_bytes(kBase)[100] == 0x5A);
    EXPECT(mirror.home_bytes(kBase + 2 * kPage)[100] == 0xA5 && !mirror.copy_bytes(kBase + 2 * kPage));
    EXPECT(mirror.flush(1) && *f.read(arena, kAt + 100) == 0x5A);
    EXPECT(mirror.counters().filtered == 1 && mirror.counters().uploads == 2);
    // Write-back: the aliased page's own home is the title's bytes, written back to its physical frame as any page;
    // the title's bytes changed, so its copy is filtered again (not fresh).
    std::size_t written = 0;
    std::uint64_t last_dst = 0;
    const auto write = [&](std::uint64_t dst, const std::uint8_t* bytes, std::uint64_t n) {
        last_dst = dst;
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, std::size_t(n));
        written += std::size_t(n);
        return true;
    };
    mirror.home_bytes(kBase)[300] = 0x77;  // a GPU store through a V# into the heap's page
    EXPECT(mirror.write_back(kBase, kBase + kPage, write) == 256 && written == 256 && frames[300] == 0x77);
    EXPECT(last_dst == reinterpret_cast<std::uintptr_t>(&frames[0]) + 256);
    EXPECT(f.filtered.size() == 2 && !f.filtered[1].fresh && mirror.copy_bytes(kBase)[300] == 0x77);
    // A write and its refill: the home takes the title's bytes, the copy is filtered, not fresh.
    frames[200] = 0x11;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(mirror.home_bytes(kBase)[200] == 0x11 && mirror.copy_bytes(kBase)[200] == 0x11);
    EXPECT(f.filtered.size() == 3 && !f.filtered[2].fresh);
    // The same title bytes again: no filter.
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1 && f.filtered.size() == 3);
    // The same physical page shown at a second, unaliased address: one home, its copy kept for the aliased one.
    mirror.map(kBase + 5 * kPage, &frames[0], nullptr, 2);
    EXPECT(mirror.home_bytes(kBase + 5 * kPage) == mirror.home_bytes(kBase) && mirror.copy_bytes(kBase)[0] == 0x5A);
    // The aliased address unmaps while the other still shows the page: the copy goes, the home stays the title's.
    mirror.map(kBase, nullptr, nullptr, 3);
    EXPECT(!mirror.copy_bytes(kBase + 5 * kPage) && mirror.home_bytes(kBase + 5 * kPage)[0] == 0xA5);
    // An alias made over a page already mapped: a copy, filtered fresh at once.
    mirror.alias(kBase + 2 * kPage, kBase + 3 * kPage, arena, kAt + 2 * kPage);
    EXPECT(mirror.copy_bytes(kBase + 2 * kPage)[0] == 0x5A && f.filtered.back().fresh &&
           f.filtered.back().va == kBase + 2 * kPage);
    EXPECT(mirror.flush(4) && *f.read(arena, kAt + 2 * kPage) == 0x5A && *f.read(arena, kAt + 2 * kPage + 1) == 0x5A);
    mirror.unalias(kBase + 2 * kPage, 5);
    EXPECT(!mirror.copy_bytes(kBase + 2 * kPage) && mirror.home_bytes(kBase + 2 * kPage)[0] == 0xA5);
    // The title's bytes changed only where write-back wrote them.
    std::vector<std::uint8_t> expect(3 * kPage, 0xA5);
    expect[300] = 0x77;
    expect[200] = 0x11;
    EXPECT(frames == expect);
    std::printf("filter: an aliased page's copy filled through the filter (fresh when made, again when the title's "
                "bytes change), its own home the title's bytes, written back as any page\n");
    return 0;
}

int pending_guard_and_frame_dest() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 4 * kPage));
    std::vector<std::uint8_t> f_old(kPage, 0x11);
    std::vector<std::uint8_t> f_new(kPage, 0x22);

    // Map va -> F_old, flush: now flushed, visible to homes_in and write_back.
    mirror.map(kBase, f_old.data(), nullptr, 1);
    EXPECT(mirror.has_pending(kBase));
    EXPECT(mirror.flush(1));
    EXPECT(!mirror.has_pending(kBase));

    // GPU writes to F_old's home: write_back must target F_old's physical frame address, NOT kBase (va).
    mirror.home_bytes(kBase)[512] = 0x99;
    std::uint64_t got_dst = 0;
    const auto write_direct = [&](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        got_dst = dst;
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    EXPECT(mirror.write_back(kBase, kBase + kPage, write_direct) == 256);
    EXPECT(got_dst == reinterpret_cast<std::uintptr_t>(f_old.data()) + 512);
    EXPECT(got_dst != kBase + 512 && f_old[512] == 0x99);

    // Remap va -> F_new WITHOUT flushing: chunk on GPU still holds F_old's bytes.
    // Both homes_in() and write_back() MUST skip va so F_new is never overwritten with stale F_old bytes.
    mirror.map(kBase, f_new.data(), nullptr, 2);
    EXPECT(mirror.has_pending(kBase));
    std::vector<std::uint64_t> pages;
    mirror.homes_in(kBase, kBase + kPage, pages);
    EXPECT(pages.empty());
    EXPECT(mirror.homes_in_ranges({{kBase, kBase + kPage}}).empty());

    std::size_t stale_writes = 0;
    const auto staged_stale = [&](std::uint64_t) -> const std::uint8_t* { return f_old.data(); };
    const auto count_writes = [&](std::uint64_t, const std::uint8_t*, std::size_t n) {
        stale_writes += n;
        return true;
    };
    EXPECT(mirror.write_back(kBase, kBase + kPage, staged_stale, count_writes) == 0);
    EXPECT(stale_writes == 0 && f_new[0] == 0x22 && f_new[512] == 0x22);

    // Once flushed, homes_in() and write_back() include va again and target F_new.
    EXPECT(mirror.flush(2) && !mirror.has_pending(kBase));
    mirror.homes_in(kBase, kBase + kPage, pages);
    EXPECT(pages.size() == 1 && pages[0] == kBase);
    mirror.home_bytes(kBase)[256] = 0x77;
    got_dst = 0;
    EXPECT(mirror.write_back(kBase, kBase + kPage, write_direct) == 256);
    EXPECT(got_dst == reinterpret_cast<std::uintptr_t>(f_new.data()) + 256 && f_new[256] == 0x77);
    std::printf("pending_guard_and_frame_dest: unflushed remaps skipped, write-back targets physical frame\n");
    return 0;
}

int refilter_scaling() {
    // 10,240 distinct homes + 1 aliased home, sharing a single 32 MiB dummy unit for the untouched homes so the test
    // stays light in RAM while exercising O(changed) refiltering over >10,000 entries in homes_.
    constexpr std::size_t kDummyHomes = 10240;
    std::vector<std::uint8_t> shared_unit(TitleMirror::kUnitBytes, 0);
    std::vector<std::uint8_t> aliased_unit(TitleMirror::kUnitBytes, 0);
    std::uintptr_t next_handle = 0x2000;
    std::size_t filter_calls = 0;
    std::size_t units_made = 0;

    leonida::TitleMirrorHooks h;
    h.make_chunk = [&](std::uint64_t, std::uint64_t, VkBuffer& out) {
        out = fake<VkBuffer>(next_handle++);
        return true;
    };
    h.make_unit = [&](VkDeviceMemory& out, std::uint8_t*& host) {
        out = fake<VkDeviceMemory>(next_handle++);
        host = (units_made++ == 0) ? aliased_unit.data() : shared_unit.data();
        return true;
    };
    h.free_unit = [](VkDeviceMemory) {};
    h.bind = [](const std::vector<VkSparseBufferMemoryBindInfo>&, std::uint64_t) { return true; };
    h.filter = [&](std::uint64_t, std::uint8_t* to, const std::uint8_t* from, std::size_t bytes, bool) {
        ++filter_calls;
        std::memcpy(to, from, bytes);
        return true;
    };

    TitleMirror mirror(std::move(h));
    EXPECT(mirror.cover(kBase, kBase + (kDummyHomes + 4) * kPage));

    std::vector<std::uint8_t> aliased_frame(kPage, 0x33);
    const VkBuffer arena = fake<VkBuffer>(0x88);
    mirror.alias(kBase, kBase + kPage, arena, 0);
    mirror.map(kBase, aliased_frame.data(), nullptr, 1);
    EXPECT(filter_calls == 1);

    // Distinct frame pointers shifted by 1 byte each over a single (kPage + kDummyHomes) zero buffer.
    std::vector<std::uint8_t> dummy_buf(kPage + kDummyHomes, 0);
    for (std::size_t i = 0; i < kDummyHomes; ++i)
        mirror.map(kBase + (i + 1) * kPage, dummy_buf.data() + i, nullptr, 1);
    EXPECT(mirror.flush(1));
    EXPECT(mirror.counters().homes == kDummyHomes + 1);

    const auto noop_write = [](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    // 500 unchanged write-backs on 1 page must be O(1), not O(homes_.size()).
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 500; ++i) EXPECT(mirror.write_back(kBase, kBase + kPage, noop_write) == 0);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    EXPECT(filter_calls == 1);

    // One real GPU write to the aliased page triggers exactly one refilter.
    mirror.home_bytes(kBase)[64] = 0x55;
    EXPECT(mirror.write_back(kBase, kBase + kPage, noop_write) == 256);
    EXPECT(aliased_frame[64] == 0x55 && filter_calls == 2 && mirror.copy_bytes(kBase)[64] == 0x55);
    std::printf("refilter_scaling: 500 write_back calls over %zu homes in %lld ms, O(changed) refilter\n",
                std::size_t(mirror.counters().homes), static_cast<long long>(elapsed_ms));
    return 0;
}

int fill_copy_and_fallback() {
    Fixture f;
    auto h = f.hooks();
    constexpr std::size_t kStrip = TitleMirror::kStripBytes;
    constexpr std::size_t kStripsPerPage = kPage / kStrip;
    std::size_t strip_calls = 0, piece_calls = 0;
    std::size_t fail_strip_offset = ~std::size_t(0);
    std::size_t fail_piece_offset = ~std::size_t(0);
    const std::uint8_t* tracked_frame = nullptr;
    h.copy = [&](std::uint8_t* to, const std::uint8_t* from, std::size_t bytes) {
        if (bytes == kStrip) {
            ++strip_calls;
            if (tracked_frame && std::size_t(from - tracked_frame) == fail_strip_offset) return false;
            std::memcpy(to, from, bytes);
            return true;
        }
        ++piece_calls;
        if (tracked_frame && std::size_t(from - tracked_frame) == fail_piece_offset) return false;
        std::memcpy(to, from, bytes);
        return true;
    };

    TitleMirror mirror(std::move(h));
    EXPECT(mirror.cover(kBase, kBase + 4 * kPage));
    std::vector<std::uint8_t> frame(kPage, 0x42);
    tracked_frame = frame.data();

    // Fast path: 4 x 16 KiB L1-hot strip hooks_.copy calls, 0 per-piece calls.
    EXPECT(kStrip == 16384 && TitleMirror::kStripPieces == 64 && kStripsPerPage == 4);
    mirror.map(kBase, frame.data(), nullptr, 1);
    EXPECT(strip_calls == kStripsPerPage && piece_calls == 0 && mirror.counters().upload_failures == 0);
    EXPECT(mirror.home_bytes(kBase)[0] == 0x42 && mirror.home_bytes(kBase)[kPage - 1] == 0x42);

    // Dirty refill also uses 4 x 16 KiB strip hooks_.copy calls.
    frame[1024] = 0x99;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(strip_calls == 2 * kStripsPerPage && piece_calls == 0 && mirror.home_bytes(kBase)[1024] == 0x99);

    // Fallback path: strip 0 fails -> falls back to ONLY 64 per-piece copies in strip 0; unreadable piece 5 is zeroed.
    fail_strip_offset = 0;
    fail_piece_offset = 5 * 256;
    frame[0] = 0x77;
    frame[5 * 256 + 10] = 0xAA;
    frame[6 * 256 + 10] = 0xBB;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(strip_calls == 3 * kStripsPerPage && piece_calls == kStrip / 256 && mirror.counters().upload_failures == 1);
    EXPECT(mirror.home_bytes(kBase)[0] == 0x77);
    EXPECT(mirror.home_bytes(kBase)[5 * 256 + 10] == 0);
    EXPECT(mirror.home_bytes(kBase)[6 * 256 + 10] == 0xBB);
    std::printf("fill_copy_and_fallback: 16 KiB L1-hot strips on fast path, per-strip 64-piece fallback on failure\n");
    return 0;
}

int batched_write_back() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 6 * kPage));
    std::vector<std::uint8_t> f0(kPage, 0x10), f1(kPage, 0x20), f2(kPage, 0x30), f3(kPage, 0x40), f4(kPage, 0x50);
    mirror.map(kBase + 0 * kPage, f0.data(), nullptr, 1);
    mirror.map(kBase + 1 * kPage, f1.data(), nullptr, 1);
    mirror.map(kBase + 2 * kPage, f2.data(), nullptr, 1);
    mirror.map(kBase + 3 * kPage, f3.data(), nullptr, 1);
    EXPECT(mirror.flush(1));

    // Make page 2 pending (unflushed remap): homes_in_ranges must skip it.
    mirror.map(kBase + 2 * kPage, f4.data(), nullptr, 2);
    EXPECT(mirror.has_pending(kBase + 2 * kPage));

    // Unsorted, overlapping, and same-piece sub-ranges across pages 0, 1, 2, 3.
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {
        {kBase + 3 * kPage + 256, kBase + 3 * kPage + 512},
        {kBase + 0 * kPage + 128, kBase + 0 * kPage + 192},  // same piece 0 as below
        {kBase + 0 * kPage + 0, kBase + 0 * kPage + 64},
        {kBase + 0 * kPage + kPage - 64, kBase + 2 * kPage + 512},  // crosses page 0 -> 1 -> pending page 2
    };
    const auto refs = mirror.homes_in_ranges(ranges);
    EXPECT(refs.size() == 3);
    EXPECT(refs[0].page == kBase + 0 * kPage && refs[0].frame == f0.data() && refs[0].chunk != VK_NULL_HANDLE);
    EXPECT(refs[1].page == kBase + 1 * kPage && refs[1].frame == f1.data());
    EXPECT(refs[2].page == kBase + 3 * kPage && refs[2].frame == f3.data());

    // Build contiguous staged buffer for the 3 snapshotted pages.
    std::vector<std::uint8_t> staged(refs.size() * kPage);
    std::memcpy(&staged[0 * kPage], f0.data(), kPage);
    std::memcpy(&staged[1 * kPage], f1.data(), kPage);
    std::memcpy(&staged[2 * kPage], f3.data(), kPage);

    // GPU modified: page 0 piece 0 (covered by two sub-ranges -> must write ONCE), page 1 piece 2 (GPU-only),
    // page 1 piece 3 (BOTH GPU and CPU modified -> conflict, CPU wins), page 3 piece 1 (GPU modified, but page 3 is
    // remapped while mutex was released -> must be skipped!).
    staged[0 * kPage + 10] = 0xA1;
    staged[1 * kPage + 2 * 256 + 5] = 0xB2;
    staged[1 * kPage + 3 * 256 + 5] = 0xB3;
    f1[3 * 256 + 5] = 0xC3;  // CPU also modified piece 3 -> conflict
    staged[2 * kPage + 1 * 256 + 5] = 0xD4;

    // Simulate page 3 remapping & flushing while page_mirror.mutex was released during GPU readback wait.
    std::vector<std::uint8_t> f3_new(kPage, 0x60);
    mirror.map(kBase + 3 * kPage, f3_new.data(), nullptr, 3);
    EXPECT(mirror.flush(3));

    std::size_t write_calls = 0;
    const auto write = [&](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        ++write_calls;
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    const std::uint64_t written = mirror.write_back_batch(refs, ranges, staged.data(), write);
    EXPECT(written == 512 && write_calls == 2);
    EXPECT(f0[10] == 0xA1);
    EXPECT(f1[2 * 256 + 5] == 0xB2);
    EXPECT(f1[3 * 256 + 5] == 0xC3);  // conflict kept CPU bytes
    EXPECT(mirror.counters().conflicts == 1);
    EXPECT(f3[1 * 256 + 5] == 0x40 && f3_new[1 * 256 + 5] == 0x60);  // remapped page 3 skipped
    std::printf("batched_write_back: coalescing, dedup, conflict counting, and concurrent-remap skip ok\n");
    return 0;
}

int filter_pieces_mask_and_scratch() {
    Fixture f;
    auto h = f.hooks();
    std::size_t copy_bytes_total = 0;
    h.copy = [&](std::uint8_t* to, const std::uint8_t* from, std::size_t bytes) {
        copy_bytes_total += bytes;
        std::memcpy(to, from, bytes);
        return true;
    };
    struct PieceCall {
        std::uint64_t va;
        bool fresh;
        std::array<std::uint64_t, 4> mask;
        std::uint8_t sample_byte;
    };
    std::vector<PieceCall> piece_calls;
    h.filter_pieces = [&](std::uint64_t va, std::uint8_t* to, const std::uint8_t* scratch, std::size_t bytes,
                          bool fresh, const std::array<std::uint64_t, 4>& mask) {
        piece_calls.push_back({va, fresh, mask, scratch[0]});
        for (std::size_t k = 0; k < bytes / 256; ++k) {
            if (!fresh && !(mask[k >> 6] & (std::uint64_t(1) << (k & 63)))) continue;
            for (std::size_t j = 0; j < 256; ++j) {
                const std::uint8_t b = scratch[k * 256 + j];
                to[k * 256 + j] = b == 0xA5 ? 0x5A : b;
            }
        }
        return true;
    };

    TitleMirror mirror(std::move(h));
    EXPECT(mirror.cover(kBase, kBase + 4 * kPage));
    std::vector<std::uint8_t> frame(kPage, 0xA5);
    const VkBuffer arena = fake<VkBuffer>(0x99);
    mirror.alias(kBase, kBase + kPage, arena, 0);

    // Initial map on an aliased page: fill() copies once (kPage bytes in 4 KiB strips) into fill_scratch_ and calls
    // filter_pieces fresh with all-1s mask from that exact same snapshot (zero duplicate copy, zero TOCTOU).
    mirror.map(kBase, frame.data(), nullptr, 1);
    EXPECT(copy_bytes_total == kPage && f.filtered.empty() && piece_calls.size() == 1 && piece_calls[0].va == kBase &&
           piece_calls[0].fresh && piece_calls[0].mask[0] == ~0ull && piece_calls[0].mask[1] == ~0ull &&
           piece_calls[0].mask[2] == ~0ull && piece_calls[0].mask[3] == ~0ull);
    EXPECT(mirror.copy_bytes(kBase)[0] == 0x5A);

    // Dirty refill modifying 3 pieces across 64-bit mask word boundaries: pieces 1, 64, and 255.
    // Must copy kPage bytes ONCE (fill's scratch), ZERO extra copies, and call filter_pieces with exact mask!
    frame[1 * 256 + 7] = 0x11;
    frame[64 * 256 + 3] = 0x22;
    frame[255 * 256 + 200] = 0x33;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(copy_bytes_total == 2 * kPage);  // exactly 1 page copied in fill(), 0 duplicate copies in filter_pieces!
    EXPECT(f.filtered.empty() && piece_calls.size() == 2);
    EXPECT(piece_calls[1].va == kBase && !piece_calls[1].fresh);
    EXPECT(piece_calls[1].mask[0] == (std::uint64_t(1) << 1));
    EXPECT(piece_calls[1].mask[1] == (std::uint64_t(1) << 0));
    EXPECT(piece_calls[1].mask[2] == 0);
    EXPECT(piece_calls[1].mask[3] == (std::uint64_t(1) << 63));
    EXPECT(mirror.copy_bytes(kBase)[1 * 256 + 7] == 0x11);
    EXPECT(mirror.copy_bytes(kBase)[64 * 256 + 3] == 0x22);
    EXPECT(mirror.copy_bytes(kBase)[255 * 256 + 200] == 0x33);
    EXPECT(mirror.copy_bytes(kBase)[0] == 0x5A);  // untouched piece preserved

    // Unchanged dirty upload: neither filter_pieces nor filter called.
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(copy_bytes_total == 3 * kPage && piece_calls.size() == 2 && f.filtered.empty());

    // Only filter_pieces installed (no hooks_.filter): make_copy calls filter_pieces with all-1s mask.
    Fixture f2;
    auto h2 = f2.hooks();
    h2.filter = nullptr;
    std::array<std::uint64_t, 4> init_mask{};
    bool init_fresh = false;
    h2.filter_pieces = [&](std::uint64_t, std::uint8_t* to, const std::uint8_t* scratch, std::size_t bytes,
                           bool fresh, const std::array<std::uint64_t, 4>& mask) {
        init_mask = mask;
        init_fresh = fresh;
        std::memcpy(to, scratch, bytes);
        return true;
    };
    TitleMirror mirror2(std::move(h2));
    EXPECT(mirror2.cover(kBase, kBase + 2 * kPage));
    mirror2.alias(kBase, kBase + kPage, arena, 0);
    mirror2.map(kBase, frame.data(), nullptr, 1);
    EXPECT(init_fresh && init_mask[0] == ~0ull && init_mask[1] == ~0ull && init_mask[2] == ~0ull &&
           init_mask[3] == ~0ull);
    EXPECT(mirror2.copy_bytes(kBase)[64 * 256 + 3] == 0x22);

    std::printf("filter_pieces_mask_and_scratch: zero duplicate copy, exact 256-bit changed-piece mask ok\n");
    return 0;
}

int coalesced_fill_and_write_back_soundness() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 2 * kPage));
    std::vector<std::uint8_t> frame(kPage, 0x10);
    mirror.map(kBase, frame.data(), nullptr, 1);
    EXPECT(mirror.flush(1));

    // Modify contiguous runs spanning 64-bit word boundaries: [0..2], [62..65], [127..129], [254..255].
    const std::vector<std::size_t> cpu_pieces = {0, 1, 2, 62, 63, 64, 65, 127, 128, 129, 254, 255};
    for (const std::size_t p : cpu_pieces) std::memset(frame.data() + p * 256, int(0x40 + (p & 0x3f)), 256);
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(!std::memcmp(mirror.home_bytes(kBase), frame.data(), kPage));

    // Now test write-back soundness (home.sums must be exact after coalesced fill!):
    // - Piece 63 (in CPU-updated run): GPU modifies it, CPU does not -> GPU write goes back!
    // - Piece 64 (in CPU-updated run): BOTH GPU and CPU modify it -> conflict, CPU wins!
    // - Piece 10 (untouched since initial whole fill): GPU modifies it -> GPU write goes back!
    mirror.home_bytes(kBase)[63 * 256 + 11] = 0xE1;
    mirror.home_bytes(kBase)[64 * 256 + 12] = 0xE2;
    frame[64 * 256 + 13] = 0xF2;
    mirror.home_bytes(kBase)[10 * 256 + 14] = 0xE3;

    const auto write = [](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    const std::uint64_t written = mirror.write_back(kBase, kBase + kPage, write);
    EXPECT(written == 512);  // pieces 10 and 63 only
    EXPECT(frame[63 * 256 + 11] == 0xE1);
    EXPECT(frame[10 * 256 + 14] == 0xE3);
    EXPECT(frame[64 * 256 + 12] == std::uint8_t(0x40 + (64 & 0x3f)));  // conflict: GPU byte rejected
    EXPECT(frame[64 * 256 + 13] == 0xF2);                               // conflict: CPU byte preserved
    EXPECT(mirror.counters().conflicts == 1);

    std::printf("coalesced_fill_and_write_back_soundness: word-spanning runs and home.sums conflict detection ok\n");
    return 0;
}

// Regression: A -> B -> A writes across home_of() -> make_copy() (alias over mapped page), alias handoff in map(), and
// finish_write_back(). Every filtered copy fill syncs home and home.sums[] from the exact same snapshot so reverting
// B -> A is always detected by fill() and propagated to the filtered copy without clobbering GPU-only pieces.
int make_copy_aba_staleness() {
    Fixture f;
    auto h = f.hooks();
    std::vector<std::array<std::uint64_t, 4>> masks;
    h.filter_pieces = [&](std::uint64_t, std::uint8_t* to, const std::uint8_t* scratch, std::size_t bytes, bool fresh,
                          const std::array<std::uint64_t, 4>& mask) {
        masks.push_back(mask);
        for (std::size_t k = 0; k < bytes / 256; ++k) {
            if (!fresh && !(mask[k >> 6] & (std::uint64_t(1) << (k & 63)))) continue;
            for (std::size_t j = 0; j < 256; ++j) {
                const std::uint8_t b = scratch[k * 256 + j];
                to[k * 256 + j] = b == 0xA5 ? 0x5A : b;
            }
        }
        return true;
    };

    TitleMirror mirror(std::move(h));
    EXPECT(mirror.cover(kBase, kBase + 8 * kPage));
    const VkBuffer arena = fake<VkBuffer>(0xAA);

    // 1. alias() over an already-mapped page (make_copy path): A -> B before alias(), then B -> A on upload().
    std::vector<std::uint8_t> frame0(kPage, 0x11);  // A
    mirror.map(kBase, frame0.data(), nullptr, 1);
    EXPECT(mirror.flush(1));
    // GPU writes piece 5 (must be preserved by make_copy's piece-wise sync!).
    mirror.home_bytes(kBase)[5 * 256 + 9] = 0xEE;
    // Title writes B (0xA5) into piece 2 before alias() calls make_copy(), without upload().
    frame0[2 * 256 + 4] = 0xA5;
    mirror.alias(kBase, kBase + kPage, arena, 0);
    EXPECT(mirror.copy_bytes(kBase)[2 * 256 + 4] == 0x5A);   // filtered B
    EXPECT(mirror.home_bytes(kBase)[2 * 256 + 4] == 0xA5);   // home synced to B
    EXPECT(mirror.home_bytes(kBase)[5 * 256 + 9] == 0xEE);   // GPU-only piece preserved!
    // Title writes A (0x11) back into piece 2: upload() MUST see piece 2 changed vs home.sums[2] and update copy!
    frame0[2 * 256 + 4] = 0x11;
    const std::size_t calls_before = masks.size();
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(masks.size() == calls_before + 1);
    EXPECT(masks.back()[0] == (std::uint64_t(1) << 2));
    EXPECT(mirror.copy_bytes(kBase)[2 * 256 + 4] == 0x11);
    EXPECT(mirror.home_bytes(kBase)[2 * 256 + 4] == 0x11);
    EXPECT(mirror.home_bytes(kBase)[5 * 256 + 9] == 0xEE);

    // 2. Alias handoff in map(): two aliased VAs share one frame; A -> B before va1 unmaps, then B -> A on va2.
    std::vector<std::uint8_t> frame1(kPage, 0x22);  // A
    const std::uint64_t va1 = kBase + 2 * kPage;
    const std::uint64_t va2 = kBase + 3 * kPage;
    mirror.alias(va1, va2 + kPage, arena, 2 * kPage);
    mirror.map(va1, frame1.data(), nullptr, 2);
    mirror.map(va2, frame1.data(), nullptr, 2);
    EXPECT(mirror.flush(2));
    frame1[3 * 256 + 8] = 0xA5;  // B without upload()
    mirror.map(va1, nullptr, nullptr, 3);  // hands off copy to va2
    EXPECT(mirror.copy_bytes(va2)[3 * 256 + 8] == 0x5A);
    EXPECT(mirror.home_bytes(va2)[3 * 256 + 8] == 0xA5);
    frame1[3 * 256 + 8] = 0x22;  // revert B -> A
    EXPECT(mirror.dirty(va2) && mirror.upload() == 1);
    EXPECT(mirror.copy_bytes(va2)[3 * 256 + 8] == 0x22);
    EXPECT(mirror.home_bytes(va2)[3 * 256 + 8] == 0x22);

    // 3. Concurrent title A -> B write during write_back() before finish_write_back() runs, then B -> A on upload().
    std::vector<std::uint8_t> frame2(kPage, 0x33);  // A
    const std::uint64_t va3 = kBase + 5 * kPage;
    mirror.alias(va3, va3 + kPage, arena, 5 * kPage);
    mirror.map(va3, frame2.data(), nullptr, 4);
    EXPECT(mirror.flush(4));
    mirror.home_bytes(va3)[1 * 256 + 10] = 0x77;  // GPU writes piece 1
    const auto write_with_race = [&](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        // Simulate concurrent CPU write A -> B in piece 0 before finish_write_back() refilters.
        frame2[0 * 256 + 12] = 0xA5;
        return true;
    };
    EXPECT(mirror.write_back(va3, va3 + kPage, write_with_race) == 256);
    EXPECT(mirror.copy_bytes(va3)[1 * 256 + 10] == 0x77);
    EXPECT(mirror.copy_bytes(va3)[0 * 256 + 12] == 0x5A);  // filtered B
    EXPECT(mirror.home_bytes(va3)[0 * 256 + 12] == 0xA5);  // home synced to B
    frame2[0 * 256 + 12] = 0x33;  // revert B -> A
    EXPECT(mirror.dirty(va3) && mirror.upload() == 1);
    EXPECT(mirror.copy_bytes(va3)[0 * 256 + 12] == 0x33);
    EXPECT(mirror.home_bytes(va3)[0 * 256 + 12] == 0x33);

    std::printf("make_copy_aba_staleness: make_copy, alias handoff, and finish_write_back sync home.sums[] on A->B->A ok\n");
    return 0;
}

// Regression: re-entrant map() calls inside filter_pieces during alias() (make_copy) and map() (alias handoff) can
// rehash both pages_ and homes_. Neither path may dereference invalidated unordered_map iterators.
int reentrant_rehash_in_alias_and_handoff() {
    Fixture f;
    auto h = f.hooks();
    TitleMirror* mirror_ptr = nullptr;
    std::vector<std::uint8_t> reentrant_frames(128 * kPage, 0x44);
    std::size_t next_reentrant = 0;
    bool trigger_rehash = false;
    h.filter_pieces = [&](std::uint64_t, std::uint8_t* to, const std::uint8_t* scratch, std::size_t bytes, bool,
                          const std::array<std::uint64_t, 4>&) {
        std::memcpy(to, scratch, bytes);
        if (trigger_rehash && mirror_ptr) {
            trigger_rehash = false;
            // Insert 64 distinct pages & homes to force pages_ and homes_ rehashes mid-callback.
            for (std::size_t i = 0; i < 64 && next_reentrant < 128; ++i, ++next_reentrant) {
                mirror_ptr->map(kBase + (100 + next_reentrant) * kPage,
                                &reentrant_frames[next_reentrant * kPage], nullptr, 10);
            }
        }
        return true;
    };

    TitleMirror mirror(std::move(h));
    mirror_ptr = &mirror;
    EXPECT(mirror.cover(kBase, kBase + 300 * kPage));
    const VkBuffer arena = fake<VkBuffer>(0xBB);

    // 1. alias() over mapped pages with re-entrant homes_/pages_ rehash inside make_copy().
    std::vector<std::uint8_t> f0(kPage, 0x12), f1(kPage, 0x34);
    mirror.map(kBase, f0.data(), nullptr, 1);
    mirror.map(kBase + kPage, f1.data(), nullptr, 1);
    trigger_rehash = true;
    mirror.alias(kBase, kBase + 2 * kPage, arena, 0);
    EXPECT(mirror.flush(1));
    EXPECT(mirror.copy_bytes(kBase) && mirror.copy_bytes(kBase)[0] == 0x12);
    EXPECT(mirror.copy_bytes(kBase + kPage) && mirror.copy_bytes(kBase + kPage)[0] == 0x34);
    EXPECT(*f.read(arena, 0) == 0x12 && *f.read(arena, kPage) == 0x34);

    // 2. map() alias handoff with re-entrant pages_/homes_ rehash inside sync_and_fill_copy().
    std::vector<std::uint8_t> shared(kPage, 0x56);
    const std::uint64_t va1 = kBase + 10 * kPage;
    const std::uint64_t va2 = kBase + 11 * kPage;
    mirror.alias(va1, va2 + kPage, arena, 10 * kPage);
    mirror.map(va1, shared.data(), nullptr, 2);
    mirror.map(va2, shared.data(), nullptr, 2);
    EXPECT(mirror.flush(2));
    trigger_rehash = true;
    mirror.map(va1, nullptr, nullptr, 3);  // hands off copy to va2, rehashes pages_/homes_, then erases va1
    EXPECT(mirror.flush(3));
    EXPECT(!mirror.mirrored(va1) && mirror.mirrored(va2));
    EXPECT(mirror.copy_bytes(va2) && mirror.copy_bytes(va2)[0] == 0x56);
    EXPECT(!f.read(arena, 10 * kPage) && *f.read(arena, 11 * kPage) == 0x56);

    std::printf("reentrant_rehash_in_alias_and_handoff: alias() and map() handoff safe across pages_/homes_ rehash ok\n");
    return 0;
}

int lock_free_pending_fast_paths() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 4 * kPage));
    EXPECT(!mirror.has_pending_work() && !mirror.has_pending_binds() && !mirror.has_dirty() &&
           !mirror.has_retired_homes());

    std::vector<std::uint8_t> frame(kPage, 0x21);
    mirror.map(kBase, frame.data(), nullptr, 1);
    EXPECT(mirror.has_pending_binds() && mirror.has_pending_work() && !mirror.has_dirty());
    EXPECT(mirror.flush(1));
    EXPECT(!mirror.has_pending_binds() && !mirror.has_pending_work());

    // Dirty with unchanged bytes: has_dirty() transitions true -> false, and last_upload_bytes() reports 0!
    EXPECT(mirror.dirty(kBase));
    EXPECT(mirror.has_dirty() && mirror.has_pending_work());
    EXPECT(mirror.upload() == 1);
    EXPECT(!mirror.has_dirty() && !mirror.has_pending_work());
    EXPECT(mirror.last_upload_bytes() == 0);

    // Dirty with 1 modified 256-byte piece: last_upload_bytes() reports 256.
    frame[10] = 0x99;
    EXPECT(mirror.dirty(kBase));
    EXPECT(mirror.upload() == 1);
    EXPECT(mirror.last_upload_bytes() == 256);

    // Unmap retires home at ticket 10: has_retired_homes(completed) checks oldest ticket in O(1) without locking.
    mirror.map(kBase, nullptr, nullptr, 10);
    EXPECT(mirror.flush(2));
    EXPECT(mirror.has_retired_homes());
    EXPECT(!mirror.has_retired_homes(9));
    EXPECT(mirror.has_retired_homes(10));
    mirror.retire(9);
    EXPECT(mirror.has_retired_homes());
    mirror.retire(10);
    EXPECT(!mirror.has_retired_homes() && !mirror.has_pending_work());

    std::printf("lock_free_pending_fast_paths: O(1) bind/dirty/retire/upload_bytes queries ok\n");
    return 0;
}

int range_sum_piece_hashing() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 8 * kPage));

    std::vector<std::uint8_t> f0(kPage, 0x11);
    std::vector<std::uint8_t> f1(kPage, 0x22);
    std::vector<std::uint8_t> f2(kPage, 0x33);
    mirror.map(kBase + 0 * kPage, f0.data(), nullptr, 1);
    mirror.map(kBase + 1 * kPage, f1.data(), nullptr, 1);
    EXPECT(mirror.flush(1));

    // Error / boundary cases: zero bytes, overflow, unmirrored page.
    std::uint64_t h = 0;
    EXPECT(!mirror.range_sum(kBase, 0, h));
    EXPECT(!mirror.range_sum(UINT64_MAX - 10, 100, h));
    EXPECT(!mirror.range_sum(kBase + 2 * kPage, 256, h));               // page 2 not mapped
    EXPECT(!mirror.range_sum(kBase + kPage, 2 * kPage, h));             // crosses into unmapped page 2

    // 1. Sub-page range [512, 512 + 1024) inside page 0 (pieces [2, 6)).
    const std::uint64_t tex_a = kBase + 512;
    const std::uint64_t tex_a_bytes = 1024;
    std::uint64_t ha0 = 0;
    EXPECT(mirror.range_sum(tex_a, tex_a_bytes, ha0));

    // Modify neighbor byte in a DIFFERENT 256-byte piece of the SAME 64 KiB page (piece 0 & piece 6):
    // range_sum for tex_a MUST stay identical!
    f0[10] = 0xAA;           // piece 0 (before tex_a)
    f0[6 * 256 + 7] = 0xBB;  // piece 6 (right after tex_a)
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    std::uint64_t ha1 = 0;
    EXPECT(mirror.range_sum(tex_a, tex_a_bytes, ha1));
    EXPECT(ha1 == ha0);

    // Modify a byte INSIDE tex_a (piece 4): range_sum MUST change!
    f0[4 * 256 + 13] = 0xCC;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    std::uint64_t ha2 = 0;
    EXPECT(mirror.range_sum(tex_a, tex_a_bytes, ha2));
    EXPECT(ha2 != ha0);

    // Revert byte back to original value: range_sum MUST return to ha0!
    f0[4 * 256 + 13] = 0x11;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    std::uint64_t ha3 = 0;
    EXPECT(mirror.range_sum(tex_a, tex_a_bytes, ha3));
    EXPECT(ha3 == ha0);

    // 2. Unaligned multi-page range crossing page 0 -> page 1: [kBase + kPage - 100, kBase + kPage + 300).
    // Overlaps piece 255 of page 0 and pieces [0, 2) of page 1.
    const std::uint64_t cross_va = kBase + kPage - 100;
    const std::uint64_t cross_bytes = 400;
    std::uint64_t hc0 = 0;
    EXPECT(mirror.range_sum(cross_va, cross_bytes, hc0));

    // Neighbor write in piece 254 of page 0 and piece 2 of page 1 -> MUST NOT change cross_va's range_sum!
    f0[254 * 256 + 255] = 0x44;
    f1[2 * 256 + 0] = 0x55;
    EXPECT(mirror.dirty(kBase) && mirror.dirty(kBase + kPage) && mirror.upload() == 2);
    std::uint64_t hc1 = 0;
    EXPECT(mirror.range_sum(cross_va, cross_bytes, hc1));
    EXPECT(hc1 == hc0);

    // Write inside piece 255 of page 0 -> MUST change cross_va's range_sum!
    f0[kPage - 1] = 0x66;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    std::uint64_t hc2 = 0;
    EXPECT(mirror.range_sum(cross_va, cross_bytes, hc2));
    EXPECT(hc2 != hc0);

    // Write inside piece 1 of page 1 -> MUST change cross_va's range_sum again!
    f1[256 + 43] = 0x77;
    EXPECT(mirror.dirty(kBase + kPage) && mirror.upload() == 1);
    std::uint64_t hc3 = 0;
    EXPECT(mirror.range_sum(cross_va, cross_bytes, hc3));
    EXPECT(hc3 != hc2);

    // 3. write_back() updates home.sums[], so range_sum() MUST reflect GPU write-backs!
    const auto write = [](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    mirror.home_bytes(kBase + kPage)[0 * 256 + 12] = 0xEE;  // GPU writes piece 0 of page 1 (inside cross_va)
    EXPECT(mirror.write_back(kBase + kPage, kBase + 2 * kPage, write) == 256);
    EXPECT(f1[12] == 0xEE);
    std::uint64_t hc4 = 0;
    EXPECT(mirror.range_sum(cross_va, cross_bytes, hc4));
    EXPECT(hc4 != hc3);

    // 4. Remap page 1 to f2 (different bytes): range_sum() MUST change!
    mirror.map(kBase + kPage, f2.data(), nullptr, 2);
    EXPECT(mirror.flush(2));
    std::uint64_t hc5 = 0;
    EXPECT(mirror.range_sum(cross_va, cross_bytes, hc5));
    EXPECT(hc5 != hc4);

    std::printf("range_sum_piece_hashing: same-page neighbor invariance, multi-page unaligned, write_back, remap ok\n");
    return 0;
}

int unstamped_freed_homes_wait_for_flush() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 4 * kPage));
    const VkBuffer arena = fake<VkBuffer>(0x99);
    constexpr VkDeviceSize kAt = 0xc000000;
    mirror.alias(kBase, kBase + 2 * kPage, arena, kAt);

    std::vector<std::uint8_t> f0(kPage, 0x31);
    std::vector<std::uint8_t> f1(kPage, 0x42);
    mirror.map(kBase, f0.data(), nullptr);
    mirror.map(kBase + kPage, f1.data(), nullptr);
    EXPECT(mirror.flush(1));
    EXPECT(mirror.unstamped_freed_count() == 0 && !mirror.has_retired_homes() && mirror.counters().homes_freed == 0);

    // Unmap page 0 (frees both its home and its aliased filtered copy as unstamped) and unalias page 1 (frees its
    // filtered copy as unstamped): NONE of the 3 freed slots may be retired until flush() + stamp_freed() run.
    mirror.map(kBase, nullptr, nullptr);
    mirror.unalias(kBase);
    EXPECT(mirror.unstamped_freed_count() == 3);
    EXPECT(!mirror.has_retired_homes());
    mirror.retire(UINT64_MAX);
    EXPECT(mirror.counters().homes_freed == 0 && mirror.unstamped_freed_count() == 3);

    // Flush publishes the sparse unbinds; stamp_freed(42) stamps the 3 slots with the post-note_mirror_bind ticket.
    EXPECT(mirror.flush(2));
    mirror.stamp_freed(42);
    EXPECT(mirror.unstamped_freed_count() == 0);
    EXPECT(mirror.has_retired_homes() && !mirror.has_retired_homes(41) && mirror.has_retired_homes(42));
    mirror.retire(41);
    EXPECT(mirror.counters().homes_freed == 0 && mirror.has_retired_homes());
    mirror.retire(42);
    EXPECT(mirror.counters().homes_freed == 3 && !mirror.has_retired_homes());

    std::printf("unstamped_freed_homes_wait_for_flush: unstamped home and alias copies stay held until flush + stamp ok\n");
    return 0;
}

int parallel_upload_matches_serial() {
    constexpr std::size_t kPages = 96;
    constexpr std::size_t kFilteredPages = 16;
    constexpr std::size_t kFailPage = 90;
    constexpr std::size_t kFailPiece = 17;
    const VkBuffer arena = fake<VkBuffer>(0xC1);

    struct FilterLog {
        std::uint64_t va;
        bool fresh;
        std::array<std::uint64_t, 4> mask;
        bool operator==(const FilterLog& o) const = default;
    };

    struct RunResult {
        std::size_t uploaded_count = 0;
        std::uint64_t last_upload_bytes = 0;
        TitleMirror::Counters counters{};
        std::vector<std::uint8_t> all_homes;
        std::vector<std::uint8_t> all_copies;
        std::vector<std::uint64_t> page_hashes;
        std::vector<FilterLog> filter_logs;
        std::uint64_t reentrant_hash = 0;
    };

    const auto run_once = [&](bool parallel, RunResult& out) -> int {
        Fixture f;
        auto h = f.hooks();
        std::vector<std::uint8_t> frames(kPages * kPage);
        for (std::size_t i = 0; i < frames.size(); ++i)
            frames[i] = std::uint8_t((i * 37 + (i / kPage) * 13 + 5) & 0xff);
        const std::uint8_t* const fail_frame = frames.data() + kFailPage * kPage;
        bool enable_copy_fail = false;
        h.copy = [fail_frame, &enable_copy_fail](std::uint8_t* to, const std::uint8_t* from, std::size_t bytes) {
            constexpr std::size_t kFailStripOff = (kFailPiece * 256) & ~(TitleMirror::kStripBytes - 1);
            if (enable_copy_fail && from == fail_frame + kFailStripOff && bytes == TitleMirror::kStripBytes)
                return false;
            if (enable_copy_fail && bytes == 256 && from == fail_frame + kFailPiece * 256) return false;
            std::memcpy(to, from, bytes);
            return true;
        };

        TitleMirror* mirror_ptr = nullptr;
        std::vector<std::uint8_t> reentrant_frames(64 * kPage, 0x6D);
        bool trigger_rehash = false;
        h.filter_pieces = [&](std::uint64_t va, std::uint8_t* to, const std::uint8_t* scratch, std::size_t bytes,
                              bool fresh, const std::array<std::uint64_t, 4>& mask) {
            out.filter_logs.push_back({va, fresh, mask});
            for (std::size_t k = 0; k < bytes / 256; ++k) {
                if (!fresh && !(mask[k >> 6] & (std::uint64_t(1) << (k & 63)))) continue;
                for (std::size_t j = 0; j < 256; ++j) {
                    const std::uint8_t b = scratch[k * 256 + j];
                    to[k * 256 + j] = (b == 0xA5) ? 0x5A : b;
                }
            }
            if (trigger_rehash && va == kBase + 4 * kPage && mirror_ptr) {
                trigger_rehash = false;
                for (std::size_t r = 0; r < 64; ++r) {
                    mirror_ptr->map(kBase + (200 + r) * kPage, reentrant_frames.data() + r * kPage, nullptr, 9);
                }
            }
            return true;
        };

        TitleMirror mirror(std::move(h));
        mirror_ptr = &mirror;
        mirror.set_parallel(parallel);
        EXPECT(mirror.cover(kBase, kBase + 300 * kPage));
        mirror.alias(kBase, kBase + kFilteredPages * kPage, arena, 0);

        for (std::size_t p = 0; p < kPages; ++p) {
            // Pages 20..23 alias physical frames 16..19 (physical-page aliasing).
            const std::size_t frame_idx = (p >= 20 && p < 24) ? (p - 4) : p;
            mirror.map(kBase + p * kPage, frames.data() + frame_idx * kPage, nullptr, 1);
        }
        EXPECT(mirror.flush(1));

        // Inject GPU-only writes into untouched piece 10 of several homes to verify parallel upload preserves them.
        for (std::size_t p = 24; p < 40; ++p) mirror.home_bytes(kBase + p * kPage)[10 * 256 + 3] = 0xEE;

        // Mutate a diverse mix: filtered pages (with 0xA5), aliased frames, word-boundary pieces, full pages,
        // untouched pages (40..55), and the failing-copy page 90.
        for (std::size_t p = 0; p < kPages; ++p) {
            if (p >= 20 && p < 24) continue;  // shared with 16..19
            if (p >= 40 && p < 56) continue;  // intentionally unchanged dirty pages
            std::uint8_t* const fptr = frames.data() + p * kPage;
            if (p < kFilteredPages) {
                fptr[0 * 256 + 1] = 0xA5;
                fptr[63 * 256 + 2] = std::uint8_t(0x20 + p);
                fptr[64 * 256 + 3] = 0xA5;
                fptr[255 * 256 + 4] = std::uint8_t(0x80 + p);
            } else if (p < 40) {
                const std::size_t pcs[] = {0, 1, 62, 63, 64, 65, 127, 128, 254, 255};
                for (std::size_t pc : pcs) fptr[pc * 256 + (p & 31)] ^= 0x5A;
            } else if (p < 72) {
                for (std::size_t pc = 0; pc < 256; ++pc) fptr[pc * 256] ^= std::uint8_t(0xA0 + (pc & 15));
            } else {
                fptr[5 * 256 + 9] ^= 0x33;
                fptr[kFailPiece * 256 + 7] = 0xCC;
                fptr[200 * 256 + 11] ^= 0x77;
            }
        }
        enable_copy_fail = true;
        trigger_rehash = true;
        for (std::size_t p = 0; p < kPages; ++p) EXPECT(mirror.dirty(kBase + p * kPage));

        out.uploaded_count = mirror.upload();
        out.last_upload_bytes = mirror.last_upload_bytes();
        out.counters = mirror.counters();
        out.all_homes.resize(kPages * kPage);
        for (std::size_t p = 0; p < kPages; ++p) {
            std::memcpy(out.all_homes.data() + p * kPage, mirror.home_bytes(kBase + p * kPage), kPage);
        }
        out.all_copies.resize(kFilteredPages * kPage);
        for (std::size_t p = 0; p < kFilteredPages; ++p) {
            std::memcpy(out.all_copies.data() + p * kPage, mirror.copy_bytes(kBase + p * kPage), kPage);
        }
        out.page_hashes.resize(kPages);
        for (std::size_t p = 0; p < kPages; ++p) {
            EXPECT(mirror.range_sum(kBase + p * kPage, kPage, out.page_hashes[p]));
        }
        EXPECT(mirror.range_sum(kBase + 200 * kPage, 64 * kPage, out.reentrant_hash));
        // Verify GPU-only writes survived and failing piece 17 was zeroed.
        for (std::size_t p = 24; p < 40; ++p) EXPECT(mirror.home_bytes(kBase + p * kPage)[10 * 256 + 3] == 0xEE);
        EXPECT(mirror.home_bytes(kBase + kFailPage * kPage)[kFailPiece * 256 + 7] == 0);
        EXPECT(mirror.counters().upload_failures == 1);
        return 0;
    };

    RunResult serial{}, par{};
    if (run_once(false, serial)) return 1;
    if (run_once(true, par)) return 1;

    EXPECT(par.uploaded_count == serial.uploaded_count);
    EXPECT(par.uploaded_count == kPages - 4);  // 4 aliased VAs deduplicated
    EXPECT(par.last_upload_bytes == serial.last_upload_bytes);
    EXPECT(par.counters.uploads == serial.counters.uploads);
    EXPECT(par.counters.uploaded_bytes == serial.counters.uploaded_bytes);
    EXPECT(par.counters.upload_failures == serial.counters.upload_failures);
    EXPECT(par.counters.filtered == serial.counters.filtered);
    EXPECT(par.counters.homes == serial.counters.homes);
    EXPECT(par.counters.pages == serial.counters.pages);
    EXPECT(par.all_homes == serial.all_homes);
    EXPECT(par.all_copies == serial.all_copies);
    EXPECT(par.page_hashes == serial.page_hashes);
    EXPECT(par.filter_logs == serial.filter_logs);
    EXPECT(par.reentrant_hash == serial.reentrant_hash);

    std::printf("parallel_upload_matches_serial: 96 dirty pages (filtered, aliased, partial, fallback, rehash) 100%% "
                "identical\n");
    return 0;
}

int parallel_write_back_batch_matches_serial() {
    constexpr std::size_t kPages = 80;
    constexpr std::size_t kFilteredPages = 8;
    const VkBuffer arena = fake<VkBuffer>(0xC2);

    struct WriteEvent {
        std::size_t frame_offset;
        std::array<std::uint8_t, 256> bytes;
        bool operator==(const WriteEvent& o) const = default;
    };

    struct RunResult {
        std::uint64_t written = 0;
        TitleMirror::Counters counters{};
        std::vector<std::uint8_t> frames;
        std::vector<std::uint8_t> copies;
        std::vector<std::uint64_t> page_hashes;
        std::vector<WriteEvent> writes;
    };

    const auto run_once = [&](bool parallel, RunResult& out) -> int {
        Fixture f;
        TitleMirror mirror(f.hooks());
        mirror.set_parallel(parallel);
        EXPECT(mirror.cover(kBase, kBase + (kPages + 8) * kPage));
        mirror.alias(kBase, kBase + kFilteredPages * kPage, arena, 0);

        out.frames.resize(kPages * kPage);
        for (std::size_t i = 0; i < out.frames.size(); ++i)
            out.frames[i] = std::uint8_t((i * 53 + (i / kPage) * 19 + 11) & 0xff);

        for (std::size_t p = 0; p < kPages; ++p) {
            // Pages 12 and 13 share physical frames 10 and 11 (physical-page aliasing).
            const std::size_t frame_idx = (p == 12 || p == 13) ? (p - 2) : p;
            mirror.map(kBase + p * kPage, out.frames.data() + frame_idx * kPage, nullptr, 1);
        }
        EXPECT(mirror.flush(1));

        // Mix of full-page ranges, partial sub-ranges, overlapping ranges, and out-of-order ranges.
        const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {
            {kBase + 40 * kPage, kBase + kPages * kPage},
            {kBase + 0 * kPage, kBase + 24 * kPage},
            {kBase + 24 * kPage + 256, kBase + 24 * kPage + 1024},       // partial pieces [1..4)
            {kBase + 25 * kPage + 63 * 256, kBase + 25 * kPage + 66 * 256},  // word boundary [63..66)
            {kBase + 26 * kPage, kBase + 41 * kPage},                    // overlaps 40..41
        };
        const auto refs = mirror.homes_in_ranges(ranges);
        EXPECT(refs.size() == kPages);

        std::vector<std::uint8_t> staged(refs.size() * kPage);
        for (std::size_t i = 0; i < refs.size(); ++i) {
            std::memcpy(staged.data() + i * kPage, refs[i].frame, kPage);
        }

        // Apply GPU-only writes, CPU-only writes, and both-sides conflicts across the 80 pages.
        // 1. Filtered aliased pages 2 and 5: GPU-only writes -> must write back AND refilter!
        staged[2 * kPage + 3 * 256 + 7] = 0xA5;  // filtered to 0x5A in copy_bytes
        staged[5 * kPage + 128 * 256 + 15] = 0x91;
        // 2. Shared physical frames 10/12 and 11/13: GPU writes on alias 10, conflict on alias 11.
        staged[10 * kPage + 4 * 256 + 1] = 0xE1;
        staged[12 * kPage + 4 * 256 + 1] = 0xE1;
        staged[11 * kPage + 8 * 256 + 2] = 0xE2;
        out.frames[11 * kPage + 8 * 256 + 2] = 0xC2;  // CPU also modified -> conflict
        // 3. Partial-range page 24: GPU modifies piece 0 (outside [1..4) -> ignored!) and piece 2 (inside -> written!).
        staged[24 * kPage + 0 * 256 + 5] = 0xDD;
        staged[24 * kPage + 2 * 256 + 5] = 0x72;
        // 4. Word-boundary partial page 25: GPU modifies pieces 63, 64, 65 (64 also CPU-modified -> conflict).
        staged[25 * kPage + 63 * 256 + 1] = 0x73;
        staged[25 * kPage + 64 * 256 + 1] = 0x74;
        out.frames[25 * kPage + 64 * 256 + 2] = 0x84;
        staged[25 * kPage + 65 * 256 + 1] = 0x75;
        // 5. High pages 30..75: mix of GPU-only, CPU-only, and conflicts.
        for (std::size_t p = 30; p < 76; ++p) {
            if (p % 3 == 0) {
                // GPU-only across 3 pieces
                staged[p * kPage + 1 * 256 + 3] ^= 0x41;
                staged[p * kPage + 64 * 256 + 4] ^= 0x42;
                staged[p * kPage + 255 * 256 + 5] ^= 0x43;
            } else if (p % 3 == 1) {
                // CPU-only write (GPU unchanged -> no write-back, CPU bytes preserved)
                out.frames[p * kPage + 12 * 256 + 6] ^= 0x55;
            } else {
                // Both sides modify piece 20 -> conflict, CPU wins; GPU-only modifies piece 21 -> written!
                staged[p * kPage + 20 * 256 + 7] ^= 0x61;
                out.frames[p * kPage + 20 * 256 + 8] ^= 0x62;
                staged[p * kPage + 21 * 256 + 9] ^= 0x63;
            }
        }
        // 6. Remap page 77 (flushed) and make page 78 pending (unflushed) after snapshot -> both must be skipped!
        std::vector<std::uint8_t> remap_frame(kPage, 0xFA);
        staged[77 * kPage + 2 * 256 + 1] = 0x19;
        staged[78 * kPage + 2 * 256 + 1] = 0x29;
        mirror.map(kBase + 77 * kPage, remap_frame.data(), nullptr, 2);
        EXPECT(mirror.flush(2));
        mirror.map(kBase + 78 * kPage, remap_frame.data(), nullptr, 3);  // pending, not flushed

        const auto base_ptr = reinterpret_cast<std::uintptr_t>(out.frames.data());
        const auto write_fn = [&](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
            if (n != 256) return false;
            WriteEvent ev{};
            ev.frame_offset = std::size_t(static_cast<std::uintptr_t>(dst) - base_ptr);
            std::memcpy(ev.bytes.data(), bytes, 256);
            out.writes.push_back(ev);
            std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
            return true;
        };

        out.written = mirror.write_back_batch(refs, ranges, staged.data(), write_fn);
        out.counters = mirror.counters();
        out.copies.resize(kFilteredPages * kPage);
        for (std::size_t p = 0; p < kFilteredPages; ++p)
            std::memcpy(out.copies.data() + p * kPage, mirror.copy_bytes(kBase + p * kPage), kPage);
        out.page_hashes.resize(kPages);
        for (std::size_t p = 0; p < kPages; ++p)
            EXPECT(mirror.range_sum(kBase + p * kPage, kPage, out.page_hashes[p]));

        // Spot-check key semantics: filtered refilter ran, partial out-of-range piece 0 skipped, remap/pending skipped.
        EXPECT(out.frames[2 * kPage + 3 * 256 + 7] == 0xA5);
        EXPECT(mirror.copy_bytes(kBase + 2 * kPage)[3 * 256 + 7] == 0x5A);
        EXPECT(out.frames[24 * kPage + 0 * 256 + 5] != 0xDD && out.frames[24 * kPage + 2 * 256 + 5] == 0x72);
        EXPECT(out.frames[77 * kPage + 2 * 256 + 1] != 0x19 && out.frames[78 * kPage + 2 * 256 + 1] != 0x29);
        EXPECT(out.counters.conflicts == 17);  // page 11 + page 25 + 15 pages in [30..75] with p%3==2
        return 0;
    };

    RunResult serial{}, par{};
    if (run_once(false, serial)) return 1;
    if (run_once(true, par)) return 1;

    EXPECT(par.written == serial.written && par.written > 0);
    EXPECT(par.counters.written_back_bytes == serial.counters.written_back_bytes);
    EXPECT(par.counters.conflicts == serial.counters.conflicts);
    EXPECT(par.counters.filtered == serial.counters.filtered);
    EXPECT(par.writes == serial.writes);
    EXPECT(par.frames == serial.frames);
    EXPECT(par.copies == serial.copies);
    EXPECT(par.page_hashes == serial.page_hashes);

    std::printf("parallel_write_back_batch_matches_serial: 80 pages (GPU-only, CPU-only, conflicts, filtered, aliased) "
                "100%% identical (%llu bytes, %llu conflicts)\n",
                static_cast<unsigned long long>(par.written),
                static_cast<unsigned long long>(par.counters.conflicts));
    return 0;
}

int parallel_fill_and_write_back_benchmark() {
    constexpr std::size_t kFillPages = 400;
    constexpr std::size_t kWbPages = 256;
    constexpr int kIters = 8;

    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + kFillPages * kPage));

    std::vector<std::uint8_t> frames(kFillPages * kPage);
    for (std::size_t i = 0; i < frames.size(); ++i) frames[i] = std::uint8_t((i * 131 + 7) & 0xff);
    for (std::size_t p = 0; p < kFillPages; ++p) mirror.map(kBase + p * kPage, frames.data() + p * kPage, nullptr, 1);
    EXPECT(mirror.flush(1));

    const auto bench_upload = [&](bool parallel) -> double {
        mirror.set_parallel(parallel);
        double total_ms = 0.0;
        for (int it = 0; it < kIters; ++it) {
            for (std::size_t p = 0; p < kFillPages; ++p) {
                for (std::size_t k = 0; k < 256; k += 4) frames[p * kPage + k * 256] ^= std::uint8_t(it + 1);
                mirror.dirty(kBase + p * kPage);
            }
            const auto t0 = std::chrono::steady_clock::now();
            const std::size_t n = mirror.upload();
            total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (n != kFillPages) return -1.0;
        }
        return total_ms / kIters;
    };

    const std::vector<std::pair<std::uint64_t, std::uint64_t>> wb_ranges = {{kBase, kBase + kWbPages * kPage}};
    const auto refs = mirror.homes_in_ranges(wb_ranges);
    EXPECT(refs.size() == kWbPages);
    std::vector<std::uint8_t> staged(kWbPages * kPage);
    std::memcpy(staged.data(), frames.data(), staged.size());
    const auto write_fn = [](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };

    const auto bench_wb = [&](bool parallel) -> double {
        mirror.set_parallel(parallel);
        double total_ms = 0.0;
        for (int it = 0; it < kIters; ++it) {
            for (std::size_t q = 0; q < 4; ++q)
                staged[(q * 60 + 7) * kPage + 5 * 256 + 11] ^= std::uint8_t(it + 5);
            const auto t0 = std::chrono::steady_clock::now();
            const std::uint64_t wb = mirror.write_back_batch(refs, wb_ranges, staged.data(), write_fn);
            total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (wb != 4 * 256) return -1.0;
        }
        return total_ms / kIters;
    };

    const double fill_serial_ms = bench_upload(false);
    const double fill_par_ms = bench_upload(true);
    std::memcpy(staged.data(), frames.data(), staged.size());  // sync staged after fill bench
    const double wb_serial_ms = bench_wb(false);
    const double wb_par_ms = bench_wb(true);
    EXPECT(fill_serial_ms > 0.0 && fill_par_ms > 0.0 && wb_serial_ms > 0.0 && wb_par_ms > 0.0);
    std::printf("bench (400-page upload, 25%% dirty): serial %.3f ms -> parallel %.3f ms (%.2fx)\n",
                fill_serial_ms, fill_par_ms, fill_serial_ms / fill_par_ms);
    std::printf("bench (256-page write_back_batch):  serial %.3f ms -> parallel %.3f ms (%.2fx)\n",
                wb_serial_ms, wb_par_ms, wb_serial_ms / wb_par_ms);

    // Measure lock-free Phase 1 prehash + locked Phase 2 write_back_prehashed vs old 1-page-at-a-time fallback.
    auto readback_pool = TitleMirror::make_worker_pool();
    TitleMirror::PrehashedReadback prehashed;
    double prehash_ms = 0.0, phase2_ms = 0.0;
    for (int it = 0; it < kIters; ++it) {
        for (std::size_t q = 0; q < 4; ++q)
            staged[(q * 60 + 7) * kPage + 5 * 256 + 11] ^= std::uint8_t(it + 9);
        const auto t0 = std::chrono::steady_clock::now();
        TitleMirror::prehash_readback_slice(refs.data(), refs.size(), wb_ranges, staged.data(), prehashed,
                                            readback_pool.get());
        const auto t1 = std::chrono::steady_clock::now();
        const std::uint64_t wb =
            mirror.write_back_prehashed(refs.data(), refs.size(), prehashed, staged.data(), nullptr, nullptr, write_fn);
        const auto t2 = std::chrono::steady_clock::now();
        if (wb != 4 * 256) return 1;
        prehash_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        phase2_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }
    std::printf("bench (256-page prehashed wb):      phase1 (unlocked) %.3f ms, phase2 (locked) %.3f ms\n",
                prehash_ms / kIters, phase2_ms / kIters);
    return 0;
}

int coalesced_write_back_runs_and_fallback() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 4 * kPage));
    std::vector<std::uint8_t> f0(kPage, 0x10), f1(kPage, 0x20);
    mirror.map(kBase, f0.data(), nullptr, 1);
    mirror.map(kBase + kPage, f1.data(), nullptr, 1);
    EXPECT(mirror.flush(1));

    // GPU modifies contiguous pieces [0..31] (32 pieces = 8 KiB) and [62..67] (spanning 64-bit word boundary),
    // with CPU also modifying piece 64 -> splits [62..67] into [62..63] and [65..67] with 1 conflict!
    std::uint8_t* h0 = mirror.home_bytes(kBase);
    for (std::size_t k = 0; k < 32; ++k) h0[k * 256 + 3] = std::uint8_t(0xA0 + k);
    for (std::size_t k = 62; k <= 67; ++k) h0[k * 256 + 5] = std::uint8_t(0xB0 + (k - 62));
    f0[64 * 256 + 7] = 0xCC;  // CPU conflict on piece 64

    struct Call {
        std::size_t off;
        std::size_t n;
    };
    std::vector<Call> calls;
    const auto base_ptr = reinterpret_cast<std::uintptr_t>(f0.data());
    const auto write_coalesced = [&](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        calls.push_back({std::size_t(static_cast<std::uintptr_t>(dst) - base_ptr), n});
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    const std::uint64_t written = mirror.write_back(kBase, kBase + kPage, write_coalesced);
    EXPECT(written == (32 + 2 + 3) * 256);
    EXPECT(calls.size() == 3);
    EXPECT(calls[0].off == 0 && calls[0].n == 32 * 256);
    EXPECT(calls[1].off == 62 * 256 && calls[1].n == 2 * 256);
    EXPECT(calls[2].off == 65 * 256 && calls[2].n == 3 * 256);
    EXPECT(f0[0 * 256 + 3] == 0xA0 && f0[31 * 256 + 3] == 0xBF);
    EXPECT(f0[63 * 256 + 5] == 0xB1 && f0[64 * 256 + 7] == 0xCC && f0[65 * 256 + 5] == 0xB3);
    EXPECT(mirror.counters().conflicts == 1);

    // Multi-piece write() failure falls back piece-by-piece so healthy pieces still commit (piece 10 fails).
    std::uint8_t* h1 = mirror.home_bytes(kBase + kPage);
    for (std::size_t k = 8; k <= 12; ++k) h1[k * 256 + 1] = std::uint8_t(0x70 + k);
    const auto base1 = reinterpret_cast<std::uintptr_t>(f1.data());
    std::size_t multi_calls = 0, single_calls = 0;
    const auto write_fallback = [&](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        const std::size_t off = std::size_t(static_cast<std::uintptr_t>(dst) - base1);
        if (n > 256) {
            ++multi_calls;
            return false;  // force piece-by-piece fallback
        }
        ++single_calls;
        if (off == 10 * 256) return false;  // single bad piece fails
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    const std::uint64_t written1 = mirror.write_back(kBase + kPage, kBase + 2 * kPage, write_fallback);
    EXPECT(written1 == 4 * 256 && multi_calls == 1 && single_calls == 5);
    EXPECT(f1[8 * 256 + 1] == 0x78 && f1[9 * 256 + 1] == 0x79);
    EXPECT(f1[10 * 256 + 1] == 0x20);  // failed piece untouched
    EXPECT(f1[11 * 256 + 1] == 0x7B && f1[12 * 256 + 1] == 0x7C);
    std::printf("coalesced_write_back_runs_and_fallback: coalesced runs, conflict split, and fallback ok\n");
    return 0;
}

int prehashed_readback_and_skip_masks() {
    constexpr std::size_t kPages = 48;
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + (kPages + 4) * kPage));

    std::vector<std::uint8_t> frames(kPages * kPage);
    for (std::size_t i = 0; i < frames.size(); ++i)
        frames[i] = std::uint8_t((i * 29 + (i / kPage) * 7 + 3) & 0xff);
    for (std::size_t p = 0; p < kPages; ++p) {
        // Page 11 aliases physical frame 10.
        const std::size_t fidx = (p == 11) ? 10 : p;
        mirror.map(kBase + p * kPage, frames.data() + fidx * kPage, nullptr, 1);
    }
    EXPECT(mirror.flush(1));

    const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {
        {kBase + 0 * kPage, kBase + 20 * kPage},
        {kBase + 20 * kPage + 2 * 256, kBase + 20 * kPage + 6 * 256},  // partial pieces [2..6)
        {kBase + 22 * kPage, kBase + kPages * kPage},
    };
    const auto refs = mirror.homes_in_ranges(ranges);
    EXPECT(refs.size() == kPages - 1);  // page 21 excluded

    // Snapshot GPU staged buffer before concurrent upload() modifies some pieces!
    std::vector<std::uint8_t> staged(refs.size() * kPage);
    for (std::size_t i = 0; i < refs.size(); ++i) std::memcpy(staged.data() + i * kPage, refs[i].frame, kPage);

    // GPU wrote:
    // - Page 0: piece 1 (GPU-only) and piece 2 (will be concurrently uploaded by CPU -> skipped via skip_mask!)
    // - Page 5: piece 10 (will be skipped whole via skip_page!)
    // - Page 10 & alias 11: piece 4 (GPU-only, shared frame -> written ONCE even though both aliases see it)
    // - Page 20 (partial [2..6)): piece 1 (out of range -> ignored), piece 3 (in range -> written)
    // - Pages 25..40: pieces 0..3 (contiguous GPU writes)
    staged[0 * kPage + 1 * 256 + 4] = 0xE1;
    staged[0 * kPage + 2 * 256 + 4] = 0xE2;
    staged[5 * kPage + 10 * 256 + 4] = 0xE5;
    staged[10 * kPage + 4 * 256 + 4] = 0xEA;
    staged[11 * kPage + 4 * 256 + 4] = 0xEA;
    staged[20 * kPage + 1 * 256 + 4] = 0xEE;  // outside [2..6)
    staged[20 * kPage + 3 * 256 + 4] = 0xEC;  // inside [2..6)
    for (std::size_t idx = 24; idx < 39; ++idx) {
        for (std::size_t k = 0; k < 4; ++k) staged[idx * kPage + k * 256 + 9] ^= 0x55;
    }

    // Simulate concurrent refresh_page_mirror() -> upload() modifying piece 2 of page 0 while readback was in flight:
    // on_changed captures exact changed_pieces into skip_masks[0] WITHOUT invalidating piece 1!
    frames[0 * kPage + 2 * 256 + 4] = 0x99;
    std::vector<std::array<std::uint64_t, 4>> skip_masks(refs.size());
    std::vector<std::uint8_t> skip_page(refs.size(), 0);
    skip_page[5] = 1;  // simulate external whole-frame invalidation of page 5
    EXPECT(mirror.dirty(kBase));
    std::size_t changed_callbacks = 0;
    EXPECT(mirror.upload([&](std::uint8_t* frame, const std::array<std::uint64_t, 4>& mask) {
        ++changed_callbacks;
        for (std::size_t i = 0; i < refs.size(); ++i) {
            if (refs[i].frame == frame) {
                for (std::size_t w = 0; w < 4; ++w) skip_masks[i][w] |= mask[w];
            }
        }
    }) == 1);
    EXPECT(changed_callbacks == 1);
    EXPECT(skip_masks[0][0] == (std::uint64_t(1) << 2));

    // Unchanged dirty upload must NOT invoke on_changed!
    EXPECT(mirror.dirty(kBase + 2 * kPage));
    EXPECT(mirror.upload([&](std::uint8_t*, const std::array<std::uint64_t, 4>&) { ++changed_callbacks; }) == 1);
    EXPECT(changed_callbacks == 1);

    // Phase 1 (lock-free parallel prehash) + Phase 2 (locked write_back_prehashed).
    auto pool = TitleMirror::make_worker_pool();
    TitleMirror::PrehashedReadback prehashed;
    TitleMirror::prehash_readback_slice(refs.data(), refs.size(), ranges, staged.data(), prehashed, pool.get());
    const auto write_fn = [](std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) {
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
        return true;
    };
    const std::uint64_t written = mirror.write_back_prehashed(refs.data(), refs.size(), prehashed, staged.data(),
                                                              skip_masks.data(), skip_page.data(), write_fn);
    // Page 0 piece 1 (1) + Page 10 piece 4 (1, alias 11 sees updated home.sums) + Page 20 piece 3 (1) + 15*4 (60) = 63
    EXPECT(written == 63 * 256);
    EXPECT(frames[0 * kPage + 1 * 256 + 4] == 0xE1);  // GPU piece 1 written back!
    EXPECT(frames[0 * kPage + 2 * 256 + 4] == 0x99);  // CPU-uploaded piece 2 NOT overwritten by stale staged 0xE2!
    EXPECT(frames[5 * kPage + 10 * 256 + 4] != 0xE5); // skip_page[5] skipped whole page 5
    EXPECT(frames[10 * kPage + 4 * 256 + 4] == 0xEA); // shared frame written back once
    EXPECT(frames[20 * kPage + 1 * 256 + 4] != 0xEE); // out of partial range
    EXPECT(frames[20 * kPage + 3 * 256 + 4] == 0xEC); // inside partial range

    std::printf("prehashed_readback_and_skip_masks: lock-free Phase 1, piece skip_mask, skip_page, and aliases ok\n");
    return 0;
}

int deferred_map_fill_and_on_changed() {
    constexpr std::size_t kPages = 40;
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + (kPages + 8) * kPage));

    std::vector<std::uint8_t> frames(kPages * kPage);
    for (std::size_t i = 0; i < frames.size(); ++i) frames[i] = std::uint8_t((i * 41 + 17) & 0xff);

    // Map kPages with defer_fill=true: zero immediate uploads, range_sum returns false until upload().
    for (std::size_t p = 0; p < kPages; ++p)
        mirror.map(kBase + p * kPage, frames.data() + p * kPage, nullptr, TitleMirror::kUnstamped, true);
    EXPECT(mirror.counters().uploads == 0);
    std::uint64_t h = 0;
    EXPECT(!mirror.range_sum(kBase, kPage, h));

    // Unmap page 3 before upload(): its deferred home is erased and never wastes a fill!
    mirror.map(kBase + 3 * kPage, nullptr, nullptr, TitleMirror::kUnstamped, true);

    // Alias page 4 before upload(): make_copy() fills home + filtered copy immediately and clears whole_fill/dirty.
    const VkBuffer arena = fake<VkBuffer>(0xD4);
    mirror.alias(kBase + 4 * kPage, kBase + 5 * kPage, arena, 0);
    EXPECT(mirror.counters().uploads == 1);
    EXPECT(mirror.range_sum(kBase + 4 * kPage, kPage, h));

    // upload() fills the remaining kPages - 2 deferred homes in parallel and fires on_changed for each.
    std::size_t notified = 0;
    const std::size_t filled = mirror.upload([&](std::uint8_t*, const std::array<std::uint64_t, 4>& mask) {
        if (mask[0] == ~0ull && mask[1] == ~0ull && mask[2] == ~0ull && mask[3] == ~0ull) ++notified;
    });
    EXPECT(filled == kPages - 2 && notified == kPages - 2);
    EXPECT(mirror.counters().uploads == kPages - 1);
    for (std::size_t p = 0; p < kPages; ++p) {
        if (p == 3) continue;
        EXPECT(!std::memcmp(mirror.home_bytes(kBase + p * kPage), frames.data() + p * kPage, kPage));
        EXPECT(mirror.range_sum(kBase + p * kPage, kPage, h));
    }

    std::printf("deferred_map_fill_and_on_changed: parallel deferred fills, unmap elision, and alias handoff ok\n");
    return 0;
}

// Verifies:
//  1. upload(on_changed, filter_batch) delivers exact AliasedPageJob snapshots with WorkerPool, handles per-page
//     filter failure (zeroing + setting fresh for the next upload), and produces identical output to serial upload().
//  2. Page-aware write-back writer:
//     - scratch frames (frame != ref.page): bypasses pinning (begin_page reports scratch=true);
//     - ordinary pages (frame == ref.page): calls begin_page ONCE per page with exact [first_off, first_off+span)
//       covering multiple scattered coalesced runs, and calls ~PageGuard ONCE after all runs;
//     - conflict-only pages: never calls begin_page;
//     - whole_title_ok threshold lowered from 16 -> 2: 1 differing piece uses 1x256B hooks_.copy, >=2 uses 1x64KiB.
int batched_filter_and_page_aware_write_back() {
    constexpr std::size_t kPages = 24;
    const VkBuffer arena = fake<VkBuffer>(0xE1);

    // 1. Batched upload(on_changed, filter_batch) parity + failure recovery.
    {
        Fixture f;
        TitleMirror mirror(f.hooks());
        EXPECT(mirror.cover(kBase, kBase + (kPages + 4) * kPage));
        mirror.alias(kBase, kBase + kPages * kPage, arena, 0);

        std::vector<std::uint8_t> frames(kPages * kPage, 0x11);
        for (std::size_t p = 0; p < kPages; ++p)
            mirror.map(kBase + p * kPage, frames.data() + p * kPage, nullptr, 1);
        EXPECT(mirror.flush(1));

        // Mutate 3 pieces on each page (including 0xA5 sentinel bytes) and leave unchanged slots with 0xCC in copy.
        for (std::size_t p = 0; p < kPages; ++p) {
            std::uint8_t* copy = mirror.copy_bytes(kBase + p * kPage);
            copy[10 * 256 + 3] = 0xCC;  // untouched slot sentinel: must NOT be clobbered!
            frames[p * kPage + 0 * 256 + 1] = 0xA5;
            frames[p * kPage + 64 * 256 + 2] = std::uint8_t(0x30 + p);
            frames[p * kPage + 255 * 256 + 4] = 0xA5;
            EXPECT(mirror.dirty(kBase + p * kPage));
        }

        std::size_t batch_calls = 0;
        std::size_t batch_jobs = 0;
        bool saw_pool = false;
        const auto run_batch = [&](TitleMirror::AliasedPageJob* jobs, std::size_t count,
                                   TitleMirror::WorkerPool* pool, std::uint8_t* ok_out) {
            ++batch_calls;
            batch_jobs += count;
            if (pool) saw_pool = true;
            const auto work = [&](std::size_t b, std::size_t e, std::size_t) {
                for (std::size_t i = b; i < e; ++i) {
                    if (jobs[i].va == kBase + 5 * kPage && batch_calls == 1) {
                        ok_out[i] = 0;  // simulate filter failure on page 5
                        continue;
                    }
                    // Write only changed 256B pieces from scratch into host buffer then copy into jobs[i].to.
                    std::vector<std::uint8_t> buf(kPage);
                    for (std::size_t k = 0; k < kPage / 256; ++k) {
                        if (!jobs[i].fresh && !(jobs[i].mask[k >> 6] & (std::uint64_t(1) << (k & 63)))) continue;
                        for (std::size_t j = 0; j < 256; ++j) {
                            const std::uint8_t v = jobs[i].scratch[k * 256 + j];
                            buf[k * 256 + j] = (v == 0xA5) ? 0x5A : v;
                        }
                        std::memcpy(jobs[i].to + k * 256, buf.data() + k * 256, 256);
                    }
                    ok_out[i] = 1;
                }
            };
            if (pool) pool->run(count, 2, work);
            else work(0, count, 0);
        };

        std::size_t on_changed_calls = 0;
        const std::size_t uploaded =
            mirror.upload([&](std::uint8_t*, const std::array<std::uint64_t, 4>&) { ++on_changed_calls; }, run_batch);
        EXPECT(uploaded == kPages && on_changed_calls == kPages && batch_calls == 1 && batch_jobs == kPages);
        EXPECT(mirror.counters().upload_failures == 1);
        // Page 5 failed -> zeroed and marked fresh for next upload.
        EXPECT(mirror.copy_bytes(kBase + 5 * kPage)[0] == 0);
        // Other pages filtered changed pieces accurately AND preserved untouched piece 10!
        for (std::size_t p = 0; p < kPages; ++p) {
            if (p == 5) continue;
            const std::uint8_t* copy = mirror.copy_bytes(kBase + p * kPage);
            EXPECT(copy[0 * 256 + 1] == 0x5A);
            EXPECT(copy[64 * 256 + 2] == std::uint8_t(0x30 + p));
            EXPECT(copy[255 * 256 + 4] == 0x5A);
            EXPECT(copy[10 * 256 + 3] == 0xCC);
        }
        // Re-uploading page 5 (even with unchanged bytes) MUST re-run filter with fresh=true!
        bool page5_fresh = false;
        EXPECT(mirror.dirty(kBase + 5 * kPage));
        EXPECT(mirror.upload([](std::uint8_t*, const std::array<std::uint64_t, 4>&) {},
                             [&](TitleMirror::AliasedPageJob* jobs, std::size_t count, TitleMirror::WorkerPool*,
                                 std::uint8_t* ok_out) {
                                 if (count == 1 && jobs[0].va == kBase + 5 * kPage) page5_fresh = jobs[0].fresh;
                                 for (std::size_t i = 0; i < count; ++i) {
                                     std::memcpy(jobs[i].to, jobs[i].scratch, kPage);
                                     ok_out[i] = 1;
                                 }
                             }) == 1);
        EXPECT(page5_fresh);
        (void)saw_pool;
    }

    // 2. Page-aware write-back writer: scratch bypass, ordinary-page single pin batch, conflict skip, whole_title_ok>=2.
    {
        Fixture f;
        auto h = f.hooks();
        std::size_t copy_64k = 0, copy_256 = 0;
        h.copy = [&](std::uint8_t* to, const std::uint8_t* from, std::size_t bytes) {
            if (bytes == kPage) ++copy_64k;
            else if (bytes == 256) ++copy_256;
            std::memcpy(to, from, bytes);
            return true;
        };

        TitleMirror mirror(std::move(h));
        // Allocate 4 aligned 64 KiB pages; map page 0 as "ordinary" (va == frame address!), pages 1..3 as "scratch"
        // (va != frame address).
        std::vector<std::uint8_t> backing(5 * kPage, 0x20);
        const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(backing.data());
        const std::uintptr_t aligned = (raw + kPage - 1) & ~(kPage - 1);
        auto* const ord_frame = reinterpret_cast<std::uint8_t*>(aligned);
        auto* const scr1_frame = ord_frame + kPage;
        auto* const scr2_frame = ord_frame + 2 * kPage;
        auto* const scr3_frame = ord_frame + 3 * kPage;

        const std::uint64_t ord_va = static_cast<std::uint64_t>(aligned);
        // Ensure ord_va and kBase are both covered (ord_va is a real host pointer; kBase is a high title VA).
        EXPECT(mirror.cover(ord_va, ord_va + kPage));
        EXPECT(mirror.cover(kBase, kBase + 4 * kPage));

        mirror.map(ord_va, ord_frame, nullptr, 1);                  // frame == page_va (ordinary page)
        mirror.map(kBase + 0 * kPage, scr1_frame, nullptr, 1);      // frame != page_va (xmem scratch, 1 piece diff)
        mirror.map(kBase + 1 * kPage, scr2_frame, nullptr, 1);      // frame != page_va (xmem scratch, 2 pieces diff)
        mirror.map(kBase + 2 * kPage, scr3_frame, nullptr, 1);      // frame != page_va (conflict-only page)
        EXPECT(mirror.flush(1));

        const std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges = {
            {ord_va, ord_va + kPage},
            {kBase, kBase + 3 * kPage},
        };
        const auto refs = mirror.homes_in_ranges(ranges);
        EXPECT(refs.size() == 4);

        std::vector<std::uint8_t> staged(refs.size() * kPage);
        for (std::size_t i = 0; i < refs.size(); ++i) std::memcpy(staged.data() + i * kPage, refs[i].frame, kPage);

        // Locate each ref index (sorted by page address).
        auto idx_of = [&](std::uint64_t va) -> std::size_t {
            for (std::size_t i = 0; i < refs.size(); ++i)
                if (refs[i].page == va) return i;
            return ~std::size_t(0);
        };
        const std::size_t i_ord = idx_of(ord_va);
        const std::size_t i_scr1 = idx_of(kBase + 0 * kPage);
        const std::size_t i_scr2 = idx_of(kBase + 1 * kPage);
        const std::size_t i_scr3 = idx_of(kBase + 2 * kPage);
        EXPECT(i_ord < 4 && i_scr1 < 4 && i_scr2 < 4 && i_scr3 < 4);

        // Ordinary page: 3 scattered runs (pieces 2, 10..11, 200) -> 1 begin_page([2*256, 201*256)), 3 writes, 1 end!
        staged[i_ord * kPage + 2 * 256 + 1] = 0xA1;
        staged[i_ord * kPage + 10 * 256 + 1] = 0xA2;
        staged[i_ord * kPage + 11 * 256 + 1] = 0xA3;
        staged[i_ord * kPage + 200 * 256 + 1] = 0xA4;
        // Scratch page 1: 1 differing piece (piece 5) -> 1x256B hooks_.copy (NOT 64 KiB!).
        staged[i_scr1 * kPage + 5 * 256 + 1] = 0xB1;
        // Scratch page 2: 2 differing pieces (pieces 3, 70) -> whole_title_ok threshold=2 -> 1x64 KiB hooks_.copy!
        staged[i_scr2 * kPage + 3 * 256 + 1] = 0xC1;
        staged[i_scr2 * kPage + 70 * 256 + 1] = 0xC2;
        // Scratch page 3: 1 differing piece, ALSO modified by CPU -> conflict, write_count==0 -> 0 begin_page calls!
        staged[i_scr3 * kPage + 4 * 256 + 1] = 0xD1;
        scr3_frame[4 * 256 + 2] = 0xD2;

        TitleMirror::PrehashedReadback prehashed;
        TitleMirror::prehash_readback_slice(refs.data(), refs.size(), ranges, staged.data(), prehashed, nullptr);

        copy_64k = copy_256 = 0;
        struct PageEvent {
            std::uint64_t page_va = 0;
            bool scratch = false;
            std::size_t first_off = 0;
            std::size_t span_bytes = 0;
            std::size_t writes = 0;
            bool ended = false;
        };
        std::vector<PageEvent> events;
        struct TestPageWriter {
            std::vector<PageEvent>* events = nullptr;
            struct Guard {
                std::vector<PageEvent>* events = nullptr;
                std::size_t idx = ~std::size_t(0);
                Guard(std::vector<PageEvent>* ev, std::size_t i) : events(ev), idx(i) {}
                Guard(const Guard&) = delete;
                Guard& operator=(const Guard&) = delete;
                Guard(Guard&& o) noexcept : events(o.events), idx(std::exchange(o.idx, ~std::size_t(0))) {}
                ~Guard() {
                    if (events && idx < events->size()) (*events)[idx].ended = true;
                }
            };
            Guard begin_page(std::uint64_t page_va, std::uint8_t* frame, std::size_t first_off,
                             std::size_t span_bytes) const {
                const bool scratch = reinterpret_cast<std::uintptr_t>(frame) != static_cast<std::uintptr_t>(page_va);
                const std::size_t idx = events->size();
                events->push_back({page_va, scratch, first_off, span_bytes, 0, false});
                return Guard(events, idx);
            }
            bool write(const Guard& g, std::uint64_t dst, const std::uint8_t* bytes, std::size_t n) const {
                ++(*events)[g.idx].writes;
                std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(dst)), bytes, n);
                return true;
            }
        } writer{&events};

        const std::uint64_t wb = mirror.write_back_prehashed(refs.data(), refs.size(), prehashed, staged.data(),
                                                             nullptr, nullptr, writer);
        EXPECT(wb == (4 + 1 + 2) * 256);
        EXPECT(mirror.counters().conflicts == 1);
        // 3 pages written (conflict-only scr3 skipped begin_page!).
        EXPECT(events.size() == 3);
        bool saw_ord = false, saw_scr1 = false, saw_scr2 = false;
        for (const PageEvent& ev : events) {
            EXPECT(ev.ended);
            if (ev.page_va == ord_va) {
                saw_ord = true;
                EXPECT(!ev.scratch);
                EXPECT(ev.first_off == 2 * 256 && ev.span_bytes == (201 - 2) * 256);
                EXPECT(ev.writes == 3);  // 3 coalesced runs under 1 begin_page/end_page!
            } else if (ev.page_va == kBase + 0 * kPage) {
                saw_scr1 = true;
                EXPECT(ev.scratch && ev.writes == 1);
            } else if (ev.page_va == kBase + 1 * kPage) {
                saw_scr2 = true;
                EXPECT(ev.scratch && ev.writes == 2);
            }
        }
        EXPECT(saw_ord && saw_scr1 && saw_scr2);
        // whole_title_ok threshold = 2: ord(4 pieces) + scr2(2 pieces) = 2x64KiB copies; scr1(1) + scr3(1) = 2x256B!
        EXPECT(copy_64k == 2 && copy_256 == 2);
        EXPECT(ord_frame[2 * 256 + 1] == 0xA1 && ord_frame[11 * 256 + 1] == 0xA3 && ord_frame[200 * 256 + 1] == 0xA4);
        EXPECT(scr1_frame[5 * 256 + 1] == 0xB1 && scr2_frame[70 * 256 + 1] == 0xC2 && scr3_frame[4 * 256 + 2] == 0xD2);
    }

    std::printf("batched_filter_and_page_aware_write_back: upload filter_batch, scratch bypass, ordinary batched pin, "
                "whole_title_ok>=2 ok\n");
    return 0;
}

int multi_chunk_direct_table_boundaries() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    constexpr std::uint64_t kChunk = TitleMirror::kChunk;
    // Cover chunk 0 at kBase, skip chunk 1 (uncovered gap), cover chunk 2 at kBase + 2 * kChunk.
    EXPECT(mirror.cover(kBase, kBase + kChunk));
    EXPECT(mirror.cover(kBase + 2 * kChunk, kBase + 3 * kChunk));

    std::vector<std::uint8_t> f0(kPage, 0x11), f1(kPage, 0x22), f2(kPage, 0x33), f3(kPage, 0x44);
    const std::uint64_t va_c0_first = kBase;
    const std::uint64_t va_c0_last = kBase + kChunk - kPage;  // page slot 32767
    const std::uint64_t va_c2_first = kBase + 2 * kChunk;
    const std::uint64_t va_c2_last = kBase + 3 * kChunk - kPage;

    mirror.map(va_c0_first, f0.data(), nullptr, 1);
    mirror.map(va_c0_last, f1.data(), nullptr, 1);
    mirror.map(va_c2_first, f2.data(), nullptr, 1);
    mirror.map(va_c2_last, f3.data(), nullptr, 1);
    EXPECT(mirror.counters().pages == 4);
    EXPECT(mirror.has_pending(va_c0_last) && !mirror.has_pending(va_c0_first + kPage));
    EXPECT(mirror.flush(1));
    EXPECT(!mirror.has_pending_binds() && !mirror.has_pending(va_c0_last));

    EXPECT(mirror.frame_at(va_c0_first) == f0.data() && mirror.frame_at(va_c0_last) == f1.data());
    EXPECT(mirror.frame_at(va_c2_first) == f2.data() && mirror.frame_at(va_c2_last) == f3.data());
    EXPECT(!mirror.mirrored(kBase + kChunk) && mirror.frame_at(kBase + kChunk) == nullptr);

    // homes_in_ranges across the uncovered chunk gap must return all 4 boundary pages in sorted order.
    const auto refs = mirror.homes_in_ranges({{va_c0_first, va_c2_last + kPage}});
    EXPECT(refs.size() == 4);
    EXPECT(refs[0].page == va_c0_first && refs[1].page == va_c0_last && refs[2].page == va_c2_first &&
           refs[3].page == va_c2_last);

    // range_sum across chunk boundary (va_c0_last -> va_c0_last + kPage, then uncovered fails).
    std::uint64_t h = 0;
    EXPECT(mirror.range_sum(va_c0_last, kPage, h));
    EXPECT(!mirror.range_sum(va_c0_last, 2 * kPage, h));

    // Unmap and verify pages_count_ returns to 0.
    mirror.map(va_c0_first, nullptr, nullptr, 2);
    mirror.map(va_c0_last, nullptr, nullptr, 2);
    mirror.map(va_c2_first, nullptr, nullptr, 2);
    mirror.map(va_c2_last, nullptr, nullptr, 2);
    EXPECT(mirror.counters().pages == 0 && !mirror.mirrored(va_c0_last) && !mirror.mirrored(va_c2_last));
    std::printf("multi_chunk_direct_table_boundaries: 32768-slot per-chunk direct page table ok\n");
    return 0;
}

int upload_breakdown_and_unit_sums_slab() {
    Fixture f;
    TitleMirror mirror(f.hooks());
    EXPECT(mirror.cover(kBase, kBase + 8 * kPage));
    const VkBuffer arena = fake<VkBuffer>(0xE5);
    mirror.alias(kBase, kBase + 2 * kPage, arena, 0);

    std::vector<std::uint8_t> f0(kPage, 0x11), f1(kPage, 0x22), f2(kPage, 0x33), f3(kPage, 0x44);
    mirror.map(kBase + 0 * kPage, f0.data(), nullptr, 1);                             // aliased (filled immediately)
    mirror.map(kBase + 2 * kPage, f1.data(), nullptr, 1, /*defer_fill=*/true);        // deferred whole fill
    mirror.map(kBase + 3 * kPage, f2.data(), nullptr, 1, /*defer_fill=*/true);        // deferred whole fill
    f0[63 * 256 + 5] = 0xA5;  // cross-strip boundary piece 63 (last piece of strip 0)
    f0[64 * 256 + 5] = 0x77;  // first piece of strip 1
    EXPECT(mirror.dirty(kBase + 0 * kPage));
    EXPECT(mirror.upload() == 3);
    EXPECT(mirror.last_whole_pages() == 2 && mirror.last_inc_pages() == 1 && mirror.last_aliased_pages() == 1);
    EXPECT(mirror.copy_bytes(kBase)[63 * 256 + 5] == 0x5A && mirror.copy_bytes(kBase)[64 * 256 + 5] == 0x77);
    EXPECT(mirror.flush(1));

    // Empty upload resets breakdown counts to 0.
    EXPECT(mirror.upload() == 0);
    EXPECT(mirror.last_whole_pages() == 0 && mirror.last_inc_pages() == 0 && mirror.last_aliased_pages() == 0);

    // Slot recycling over contiguous Unit::sums slab: unmap + retire frees slot, next map reuses it cleanly.
    std::uint64_t h_old = 0;
    EXPECT(mirror.range_sum(kBase + 2 * kPage, kPage, h_old));
    mirror.map(kBase + 2 * kPage, nullptr, nullptr, 5);
    EXPECT(mirror.flush(2));
    mirror.retire(5);
    EXPECT(mirror.counters().homes_freed == 1);
    mirror.map(kBase + 2 * kPage, f3.data(), nullptr, 6);
    EXPECT(mirror.flush(3));
    std::uint64_t h_new = 0;
    EXPECT(mirror.range_sum(kBase + 2 * kPage, kPage, h_new) && h_new != h_old);
    EXPECT(!std::memcmp(mirror.home_bytes(kBase + 2 * kPage), f3.data(), kPage));
    // Verify incremental diff on reused slot detects single-byte changes across all 4 x 16 KiB strips.
    for (std::size_t s = 0; s < 4; ++s) f3[s * 16384 + 256] = std::uint8_t(0xC0 + s);
    EXPECT(mirror.dirty(kBase + 2 * kPage) && mirror.upload() == 1);
    EXPECT(mirror.last_upload_bytes() == 4 * 256);
    EXPECT(!std::memcmp(mirror.home_bytes(kBase + 2 * kPage), f3.data(), kPage));
    std::printf("upload_breakdown_and_unit_sums_slab: breakdown counts, 16 KiB strips, and Unit::sums slab reuse ok\n");
    return 0;
}

}  // namespace

int main() {
    if (aliases()) return 1;
    if (filters()) return 1;
    if (pending_guard_and_frame_dest()) return 1;
    if (refilter_scaling()) return 1;
    if (fill_copy_and_fallback()) return 1;
    if (batched_write_back()) return 1;
    if (filter_pieces_mask_and_scratch()) return 1;
    if (coalesced_fill_and_write_back_soundness()) return 1;
    if (make_copy_aba_staleness()) return 1;
    if (reentrant_rehash_in_alias_and_handoff()) return 1;
    if (lock_free_pending_fast_paths()) return 1;
    if (range_sum_piece_hashing()) return 1;
    if (unstamped_freed_homes_wait_for_flush()) return 1;
    if (parallel_upload_matches_serial()) return 1;
    if (parallel_write_back_batch_matches_serial()) return 1;
    if (parallel_fill_and_write_back_benchmark()) return 1;
    if (coalesced_write_back_runs_and_fallback()) return 1;
    if (prehashed_readback_and_skip_masks()) return 1;
    if (deferred_map_fill_and_on_changed()) return 1;
    if (batched_filter_and_page_aware_write_back()) return 1;
    if (multi_chunk_direct_table_boundaries()) return 1;
    if (upload_breakdown_and_unit_sums_slab()) return 1;
    std::printf("title_mirror_hooks_test: ok\n");
    return 0;
}
