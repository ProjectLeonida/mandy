#pragma once
// Title memory whose bytes only xmem's frame table holds (title_frames): the host reads and writes it through the
// physical page's own mapping, never through a view at the title's address.
//
// The console queue's CopyPageMappingsBatchX (queue slot 31) remaps physical pages under the stream pool's GPU-only
// alias. On the console it edits GPU page tables only. xmem does the same: the GPU-only page, and the page of the CPU
// view paired with it (title_gpu_views, which the D3D12 layer and DirectStorage read and write in its place), get the
// physical page in xmem's table alone, with no host view (rt::xmem's map_table_page). A host remap there was two
// wineserver round trips per page under Wine. So a host reader of such a page finds it reserved, and reads it here
// instead: the page's address in its section's scratch view, which xmem maps once per allocation.
//
// What changes such a page's bytes, and how its readers learn of it:
//   - a remap of the page itself: xmem raises title_memory_map::changed and geometry_writes::notify(replaced) at it, as
//     a host remap does;
//   - the title's CPU, through any host view of the same physical page: xmem marks a table page aliased
//     (title_page_aliases) with those views as its other addresses (geometry_writes::aliases), so the slot-133
//     shadows, the geometry copies and the title texture watch guard it there, as #1867's alias cells do for a page
//     mapped twice;
//   - a host writer (DirectStorage, the console's value writes): through kernel_writes (title_page_watch.h), which
//     writes the frame here and pins and opens the page and its other addresses first, as for a kernel write.
//
// No Windows header: the lookup is xmem's (set_lookup), and the layer's copies only split their ranges here.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace leonida::title_frames {

constexpr std::uintptr_t kPage = 0x10000;  // xmem's physical page

// A table page's bytes: the page's first byte in its section's scratch view, and what keeps that view mapped.
struct Frame {
    std::uint8_t* bytes = nullptr;
    std::shared_ptr<void> keep;
};

// xmem's side: the frame the 64 KiB page at `page` holds in its table alone (no host view there), or false. Must not
// wait for the process heap, fault or log: the D3D12 layer calls it with its trackers' locks held.
using Lookup = bool (*)(std::uintptr_t page, Frame& out);
// xmem's side: the host addresses the page's physical page is mapped at (its other addresses), appended to `out`.
using Peers = void (*)(std::uintptr_t page, std::vector<std::uintptr_t>& out);
// xmem's side: single-lock, zero-shared_ptr run_end and batch-backed checks over g_table_frames.
using RunEnd = std::uintptr_t (*)(std::uintptr_t from, std::uintptr_t to, bool& is_backed);
using BackedBatch = void (*)(const std::uintptr_t* pages, std::uint8_t* out_backed, std::size_t count);

struct State {
    std::atomic<Lookup> lookup{nullptr};
    std::atomic<Peers> peers{nullptr};
    std::atomic<RunEnd> run_end{nullptr};
    std::atomic<BackedBatch> backed_batch{nullptr};
    // Every page ever given a table entry lies in [lo, hi): a range outside costs a reader two loads, not a lookup.
    std::atomic<std::uintptr_t> lo{UINTPTR_MAX}, hi{0};
};

inline State& state() {
    static State s;
    return s;
}

inline void set_lookup(Lookup lookup, Peers peers, RunEnd run_end_fn = nullptr, BackedBatch backed_batch_fn = nullptr) {
    State& s = state();
    s.peers.store(peers, std::memory_order_release);
    s.run_end.store(run_end_fn, std::memory_order_release);
    s.backed_batch.store(backed_batch_fn, std::memory_order_release);
    s.lookup.store(lookup, std::memory_order_release);
}

// xmem, before the page at `page` gets a table entry: the bounds grow to hold it (they never shrink).
inline void note_page(std::uintptr_t page) {
    State& s = state();
    const std::uintptr_t end = page <= UINTPTR_MAX - kPage ? page + kPage : UINTPTR_MAX;
    std::uintptr_t lo = s.lo.load(std::memory_order_relaxed);
    while (page < lo && !s.lo.compare_exchange_weak(lo, page, std::memory_order_acq_rel)) {
    }
    std::uintptr_t hi = s.hi.load(std::memory_order_relaxed);
    while (end > hi && !s.hi.compare_exchange_weak(hi, end, std::memory_order_acq_rel)) {
    }
}

// Whether [from, to) may hold a table page at all.
inline bool may_hold(std::uintptr_t from, std::uintptr_t to) {
    const State& s = state();
    return from < to && s.lookup.load(std::memory_order_acquire) && from < s.hi.load(std::memory_order_acquire) &&
           s.lo.load(std::memory_order_acquire) < to;
}

// The byte at `va` of a table page: out.bytes points at it (the rest of its 64 KiB page follows). False when the page
// is not a table page (the host reads it at `va` itself).
inline bool find(std::uintptr_t va, Frame& out) {
    out = {};
    if (!may_hold(va, va + 1)) return false;
    const Lookup lookup = state().lookup.load(std::memory_order_acquire);
    const std::uintptr_t page = va & ~(kPage - 1);
    if (!lookup(page, out) || !out.bytes) {
        out = {};
        return false;
    }
    out.bytes += va - page;
    return true;
}

inline bool backed(std::uintptr_t va) {
    Frame frame;
    return find(va, frame);
}

// The other host addresses of the table page at `va` (64 KiB-aligned, one per view of its physical page).
inline void peers(std::uintptr_t va, std::vector<std::uintptr_t>& out) {
    if (const Peers f = state().peers.load(std::memory_order_acquire)) f(va & ~(kPage - 1), out);
}

// Sets out_backed[k] = 1 iff the 64 KiB page holding pages[k] is a backed table page, else 0.
// Uses xmem's single-lock, zero-shared_ptr batch hook when registered; falls back to backed() otherwise.
inline void backed_batch(const std::uintptr_t* pages, std::uint8_t* out_backed, std::size_t count) {
    if (!count) return;
    const State& s = state();
    if (!s.lookup.load(std::memory_order_acquire) ||
        s.lo.load(std::memory_order_acquire) >= s.hi.load(std::memory_order_acquire)) {
        for (std::size_t k = 0; k < count; ++k) out_backed[k] = 0;
        return;
    }
    if (const BackedBatch fn = s.backed_batch.load(std::memory_order_acquire)) {
        fn(pages, out_backed, count);
        return;
    }
    for (std::size_t k = 0; k < count; ++k) out_backed[k] = backed(pages[k]) ? 1u : 0u;
}

// The end of the run that starts at `from`, at most `to`, whose pages are all table pages or all not; `is_backed`
// says which. At least one page, or up to `to`.
inline std::uintptr_t run_end(std::uintptr_t from, std::uintptr_t to, bool& is_backed) {
    if (!may_hold(from, to)) {
        is_backed = false;
        return to;
    }
    if (const RunEnd fn = state().run_end.load(std::memory_order_acquire)) {
        return fn(from, to, is_backed);
    }
    is_backed = backed(from);
    std::uintptr_t at = (from & ~(kPage - 1)) + kPage;
    while (at < to && at > from && backed(at) == is_backed) at += kPage;
    return at < to && at > from ? at : to;
}

// f(at, n, bytes) for each part of [from, from + size), in order: a table page's part (inside one 64 KiB page) with
// `bytes` its first byte in the frame, or a run of other pages with `bytes` null (the host reads them at `at`). Stops
// at the first f that returns false, and returns false then.
template <class F>
bool for_each_part(std::uintptr_t from, std::size_t size, F&& f) {
    if (!size) return true;
    const std::uintptr_t to = size > UINTPTR_MAX - from ? UINTPTR_MAX : from + size;
    if (!may_hold(from, to)) return f(from, std::size_t(to - from), static_cast<std::uint8_t*>(nullptr));
    std::uintptr_t host = from;  // the start of the host run not handed out yet
    for (std::uintptr_t at = from; at < to;) {
        const std::uintptr_t page_end = (at & ~(kPage - 1)) + kPage;
        const std::uintptr_t end = page_end > at && page_end < to ? page_end : to;
        Frame frame;
        if (find(at, frame)) {
            if (host < at && !f(host, std::size_t(at - host), static_cast<std::uint8_t*>(nullptr))) return false;
            if (!f(at, std::size_t(end - at), frame.bytes)) return false;
            host = end;
        }
        at = end;
    }
    return host >= to || f(host, std::size_t(to - host), static_cast<std::uint8_t*>(nullptr));
}

// The page mirror's change feed (src/title_mirror.h): xmem tells it of every change of what a 64 KiB page of
// src/title_mirror_space.h's range shows, in the order the changes are made: the page shows the physical page whose
// bytes are at `frame` (its section's scratch view, kept alive by `keep`), or nothing (`frame` null). Called with xmem's
// locks held (g_phys_mutex, sometimes the fault lock): the feed only queues, on memory that is not the process heap's.
using Feed = void (*)(void* context, std::uintptr_t page, std::uint8_t* frame, const std::shared_ptr<void>& keep);
// xmem's side: from now on every change goes to `feed` (null: none any more); the pages mapped now are fed first, under
// the same lock as the changes, so none is missed or fed twice.
using Attach = void (*)(void* context, Feed feed);
inline std::atomic<Attach> attach_fn{nullptr};

}  // namespace leonida::title_frames
