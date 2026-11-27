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
    std::size_t whole_calls = 0, piece_calls = 0;
    bool fail_whole = false;
    std::size_t fail_piece_offset = ~std::size_t(0);
    const std::uint8_t* tracked_frame = nullptr;
    h.copy = [&](std::uint8_t* to, const std::uint8_t* from, std::size_t bytes) {
        if (bytes == kPage) {
            ++whole_calls;
            if (fail_whole) return false;
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

    // Fast path: 1 whole-page hooks_.copy call, 0 per-piece calls.
    mirror.map(kBase, frame.data(), nullptr, 1);
    EXPECT(whole_calls == 1 && piece_calls == 0 && mirror.counters().upload_failures == 0);
    EXPECT(mirror.home_bytes(kBase)[0] == 0x42 && mirror.home_bytes(kBase)[kPage - 1] == 0x42);

    // Dirty refill also uses 1 whole-page hooks_.copy call.
    frame[1024] = 0x99;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(whole_calls == 2 && piece_calls == 0 && mirror.home_bytes(kBase)[1024] == 0x99);

    // Fallback path: whole-page copy fails -> falls back to 256 per-piece copies; unreadable piece 5 is zeroed.
    fail_whole = true;
    fail_piece_offset = 5 * 256;
    frame[0] = 0x77;
    frame[5 * 256 + 10] = 0xAA;
    frame[6 * 256 + 10] = 0xBB;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(whole_calls == 3 && piece_calls == kPage / 256 && mirror.counters().upload_failures == 1);
    EXPECT(mirror.home_bytes(kBase)[0] == 0x77);
    EXPECT(mirror.home_bytes(kBase)[5 * 256 + 10] == 0);
    EXPECT(mirror.home_bytes(kBase)[6 * 256 + 10] == 0xBB);
    std::printf("fill_copy_and_fallback: 1-shot 64 KiB copy on fast path, per-piece fallback on failure\n");
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
    std::size_t copy_calls = 0;
    h.copy = [&](std::uint8_t* to, const std::uint8_t* from, std::size_t bytes) {
        ++copy_calls;
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

    // Initial map on an aliased page: fill() copies once into fill_scratch_ and calls filter_pieces fresh with all-1s
    // mask from that exact same snapshot (zero duplicate copy, zero TOCTOU between home.sums[] and filtered copy).
    mirror.map(kBase, frame.data(), nullptr, 1);
    EXPECT(copy_calls == 1 && f.filtered.empty() && piece_calls.size() == 1 && piece_calls[0].va == kBase &&
           piece_calls[0].fresh && piece_calls[0].mask[0] == ~0ull && piece_calls[0].mask[1] == ~0ull &&
           piece_calls[0].mask[2] == ~0ull && piece_calls[0].mask[3] == ~0ull);
    EXPECT(mirror.copy_bytes(kBase)[0] == 0x5A);

    // Dirty refill modifying 3 pieces across 64-bit mask word boundaries: pieces 1, 64, and 255.
    // Must call hooks_.copy ONCE (fill's scratch), ZERO extra copies, and call filter_pieces with exact mask!
    frame[1 * 256 + 7] = 0x11;
    frame[64 * 256 + 3] = 0x22;
    frame[255 * 256 + 200] = 0x33;
    EXPECT(mirror.dirty(kBase) && mirror.upload() == 1);
    EXPECT(copy_calls == 2);  // exactly 1 copy in fill(), 0 duplicate copies in filter_pieces!
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
    EXPECT(copy_calls == 3 && piece_calls.size() == 2 && f.filtered.empty());

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
    std::printf("title_mirror_hooks_test: ok\n");
    return 0;
}
