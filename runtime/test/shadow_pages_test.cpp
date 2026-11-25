// leonida::shadow_pages::ShadowPages (runtime/src/shadow_pages.h): two trackers over overlapping pages, as the retail
// title gives the D3D12 layer: a console descriptor heap's tracker, and the tracker of a slot-133 buffer placed over the
// last part of the memory the heap is carved from. No game data, no GPU.
//
// A tracker guards the pages it copies (read-only, so the title's next write faults), and a page another tracker made
// read-only is "read-only (not ours)" to it: it can't guard it, so it used to copy it at every submission (27.6 MiB of
// every submission in the retail title). The trackers share what they know instead: a page another tracker guards is
// copied once, and again when the title writes it, a file shim opens it for a kernel write, or the guard goes.
//   1. both trackers' copies match the title's pages after the first refresh, and the one that finds the overlap
//      read-only copied it (and says why);
//   2. with the title writing nothing, neither tracker copies anything at the next refreshes;
//   3. a write the title makes to an overlapped page reaches both copies, from the page's 64 KiB block at most, and the
//      refresh after copies nothing again; so does a write to a page only one tracker covers;
//   4. the same with the trackers refreshed in the other order;
//   5. a kernel write (the file shims open the pages first: open_shadow_pages, no fault) reaches both copies;
//   6. a write made after the tracker that guarded the pages was destroyed, and before another guards them again,
//      reaches the other tracker's copy.
//   7. a tracker that starts inside the block a fault of the other's opened still sees a write to the opened pages.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "../src/shadow_pages.h"

namespace {

namespace sp = leonida::shadow_pages;
using Copy = sp::ShadowPages::Copy;

constexpr std::size_t kPage = 4096;
constexpr std::size_t kBlock = 64 * 1024;
constexpr std::size_t kMiB = 1 << 20;

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

void check_count(std::size_t got, std::size_t want, const char* what) {
    const bool ok = got == want;
    std::printf("%s %s (%zu KiB, want %zu KiB)\n", ok ? "PASS" : "FAIL", what, got >> 10, want >> 10);
    if (!ok) ++g_failures;
}

void check_at_most(std::size_t got, std::size_t most, const char* what) {
    const bool ok = got <= most;
    std::printf("%s %s (%zu KiB, at most %zu KiB)\n", ok ? "PASS" : "FAIL", what, got >> 10, most >> 10);
    if (!ok) ++g_failures;
}

// One tracker with the copy it keeps.
struct Tracked {
    std::vector<std::uint8_t> shadow;
    sp::ShadowPagesPtr tracker;
    const std::uint8_t* pages = nullptr;

    Tracked(const std::uint8_t* from, std::size_t bytes) : shadow(bytes, 0xee), pages(from) {
        tracker = sp::make_shadow_pages(from, shadow.data(), bytes);
    }
    std::size_t refresh() { return tracker->refresh(); }
    // The copy holds the title's pages as they are now.
    bool current() const { return std::memcmp(shadow.data(), pages, shadow.size()) == 0; }
    std::size_t read_only_bytes() const { return tracker->copy_bytes()[unsigned(Copy::read_only)]; }
};

// The title's write: a store from this thread, which faults when a tracker guards the page.
void title_writes(std::uint8_t* at, std::uint8_t value) { *reinterpret_cast<volatile std::uint8_t*>(at) = value; }

}  // namespace

int main() {
    // 2 MiB of the title's memory. The heap's tracker covers the first MiB; the buffer's, the last 512 KiB of that and
    // 512 KiB more: 128 pages that both cover.
    auto* base = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 2 * kMiB, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!base) {
        std::printf("FAIL VirtualAlloc\n");
        return 1;
    }
    for (std::size_t i = 0; i < 2 * kMiB; ++i) base[i] = std::uint8_t(i * 7 + (i >> 12));
    std::uint8_t* const shared = base + kMiB / 2;      // the pages both cover: [512 KiB, 1 MiB)
    std::uint8_t* const only_heap = base + 64 * kPage;  // in the heap's alone
    std::uint8_t* const only_buffer = base + kMiB + kMiB / 4;  // in the buffer's alone

    auto heap = std::make_unique<Tracked>(base, kMiB);
    auto buffer = std::make_unique<Tracked>(shared, kMiB);

    // 1. The first refreshes, in the order of a submission: the console heaps, then the slot-133 shadows.
    const std::size_t heap_first = heap->refresh();
    const std::size_t buffer_first = buffer->refresh();
    check_count(heap_first, kMiB, "1: the heap's tracker copies all its pages at the first refresh");
    check_count(buffer_first, kMiB, "1: the buffer's tracker copies all its pages at the first refresh");
    check_count(buffer->read_only_bytes(), kMiB / 2, "1: the buffer's tracker found the shared pages read-only");
    check(heap->current() && buffer->current(), "1: both copies match the title's pages");

    // 2. Nothing written: nothing to copy, for either.
    for (int round = 1; round <= 3; ++round) {
        char what[96];
        std::snprintf(what, sizeof(what), "2: refresh %d with no write: the heap's tracker copies nothing", round);
        check_count(heap->refresh(), 0, what);
        std::snprintf(what, sizeof(what), "2: refresh %d with no write: the buffer's tracker copies nothing", round);
        check_count(buffer->refresh(), 0, what);
    }

    // 3. The title writes a page both cover.
    title_writes(shared + 40 * kPage + 17, 0x5a);
    const std::size_t heap_written = heap->refresh();
    const std::size_t buffer_written = buffer->refresh();
    check(heap->current() && buffer->current(), "3: a write to a shared page reaches both copies");
    check_at_most(heap_written, kBlock, "3: the heap's tracker copies only the written block");
    check_at_most(buffer_written, kBlock, "3: the buffer's tracker copies only the written block");
    check_count(heap->refresh() + buffer->refresh(), 0, "3: nothing to copy at the refresh after");
    check_count(heap->refresh() + buffer->refresh(), 0, "3: nothing to copy at the one after that");
    // ... and again, the page now guarded the way it was.
    title_writes(shared + 40 * kPage + 18, 0x5b);
    heap->refresh();
    buffer->refresh();
    check(heap->current() && buffer->current(), "3: a second write to the same page reaches both copies");
    check_count(heap->refresh() + buffer->refresh(), 0, "3: nothing to copy after the second write");
    // A page only one covers.
    title_writes(only_buffer + 3, 0x61);
    const std::size_t heap_other = heap->refresh();
    const std::size_t buffer_other = buffer->refresh();
    check(buffer->current(), "3: a write to a page only the buffer's tracker covers reaches its copy");
    check_count(heap_other, 0, "3: the heap's tracker copies nothing for it");
    check_at_most(buffer_other, kBlock, "3: the buffer's tracker copies only the written block");
    title_writes(only_heap + 5, 0x62);
    const std::size_t heap_alone = heap->refresh();
    const std::size_t buffer_alone = buffer->refresh();
    check(heap->current(), "3: a write to a page only the heap's tracker covers reaches its copy");
    check_at_most(heap_alone, kBlock, "3: the heap's tracker copies only the written block");
    check_count(buffer_alone, 0, "3: the buffer's tracker copies nothing for it");

    // 4. The other order: the buffer's tracker before the heap's.
    title_writes(shared + 100 * kPage + 1, 0x63);
    const std::size_t buffer_early = buffer->refresh();
    const std::size_t heap_late = heap->refresh();
    check(heap->current() && buffer->current(), "4: with the buffer's tracker refreshed first, both copies are current");
    check_at_most(buffer_early, kBlock, "4: the buffer's tracker copies only the written block");
    check_at_most(heap_late, kBlock, "4: the heap's tracker copies only the written block");
    check_count(buffer->refresh() + heap->refresh(), 0, "4: nothing to copy at the refresh after");
    check_count(heap->refresh() + buffer->refresh(), 0, "4: nothing to copy in the other order either");

    // 5. A kernel write: the file shims open the pages (open_shadow_pages), the kernel writes, nothing faults.
    std::uint8_t* const kernel_page = shared + 7 * kPage;
    sp::open_shadow_pages(reinterpret_cast<std::uintptr_t>(kernel_page),
                          reinterpret_cast<std::uintptr_t>(kernel_page) + kPage);
    std::memset(kernel_page, 0x77, kPage);  // writable now: no fault
    const std::size_t heap_kernel = heap->refresh();
    const std::size_t buffer_kernel = buffer->refresh();
    check(heap->current() && buffer->current(), "5: a kernel write into shared pages reaches both copies");
    check_at_most(heap_kernel, kBlock, "5: the heap's tracker copies only the opened block");
    check_at_most(buffer_kernel, kBlock, "5: the buffer's tracker copies only the opened block");
    check_count(heap->refresh() + buffer->refresh(), 0, "5: nothing to copy at the refresh after");

    // 6. The tracker that guarded the shared pages goes (its pages are writable again), the title writes, another
    // tracker guards the pages again before the buffer's refreshes.
    heap.reset();
    std::uint8_t* const late_page = shared + 120 * kPage;
    title_writes(late_page + 9, 0x7a);  // no fault: nobody guards it now
    heap = std::make_unique<Tracked>(base, kMiB);
    heap->refresh();  // guards the pages the buffer's tracker had marked
    buffer->refresh();
    check(heap->current() && buffer->current(), "6: a write between one guard's end and the next reaches the buffer's copy");
    check_count(heap->refresh() + buffer->refresh(), 0, "6: nothing to copy at the refresh after");

    // 7. A tracker that starts inside the block the heap's fault reopens (the heap's reopens its whole 64 KiB block; the
    // buffer's doesn't contain the faulting address, and covers part of the block).
    buffer.reset();
    std::uint8_t* const mid_block = shared + 8 * kPage;
    buffer = std::make_unique<Tracked>(mid_block, kMiB);
    heap->refresh();
    buffer->refresh();
    check_count(heap->refresh() + buffer->refresh(), 0, "7: both trackers quiet before the write");
    title_writes(shared + 2 * kPage, 0x31);   // the heap's alone: faults, and its block (pages 0 to 15 of shared) opens
    title_writes(shared + 12 * kPage, 0x32);  // in the buffer's too, in the opened block: no fault
    heap->refresh();
    buffer->refresh();
    check(heap->current() && buffer->current(), "7: a write to a page the heap's fault opened reaches the buffer's copy");
    check_count(heap->refresh() + buffer->refresh(), 0, "7: nothing to copy at the refresh after");

    buffer.reset();
    heap.reset();
    VirtualFree(base, 0, MEM_RELEASE);
    std::printf("%s\n", g_failures ? "FAILED" : "ok");
    return g_failures ? 1 : 0;
}
