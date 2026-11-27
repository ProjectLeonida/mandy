// The page mirror's alias for console descriptor heaps read in place (src/title_mirror.h), without a GPU: fake handles
// stand for the chunks, the units and the arena buffer, and the bind hook records what each batch binds.
//   alias(): the pages of a range are also bound into another buffer at an offset, in the same batches as their chunk
//     binds, following remaps and unmaps, each with a home of its own there (the filtered copy), which hooks.filter
//     fills (fresh when made, not fresh when the title's bytes change again, never when they did not); the page's own
//     home at the title's address keeps the title's bytes exactly and is written back as any page; unalias() unbinds
//     the alias and frees the copies. Without an alias the batches are as before. The title's bytes are only written
//     by write-back.
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
    // Write-back: the aliased page's own home is the title's bytes, written back as any page; the title's bytes
    // changed, so its copy is filtered again (not fresh).
    std::size_t written = 0;
    const auto write = [&](std::uint64_t va, const std::uint8_t* bytes, std::uint64_t n) {
        std::memcpy(&frames[std::size_t(va - kBase)], bytes, std::size_t(n));
        written += std::size_t(n);
        return true;
    };
    mirror.home_bytes(kBase)[300] = 0x77;  // a GPU store through a V# into the heap's page
    EXPECT(mirror.write_back(kBase, kBase + kPage, write) == 256 && written == 256 && frames[300] == 0x77);
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

}  // namespace

int main() {
    if (aliases()) return 1;
    if (filters()) return 1;
    std::printf("title_mirror_hooks_test: ok\n");
    return 0;
}
