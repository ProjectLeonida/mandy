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

struct State {
    std::atomic<Lookup> lookup{nullptr};
    std::atomic<Peers> peers{nullptr};
    // Every page ever given a table entry lies in [lo, hi): a range outside costs a reader two loads, not a lookup.
    std::atomic<std::uintptr_t> lo{UINTPTR_MAX}, hi{0};
};

inline State& state() {
    static State s;
    return s;
}

inline void set_lookup(Lookup lookup, Peers peers) {
    state().peers.store(peers, std::memory_order_release);
    state().lookup.store(lookup, std::memory_order_release);
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

// The end of the run that starts at `from`, at most `to`, whose pages are all table pages or all not; `is_backed`
// says which. At least one page, or up to `to`.
inline std::uintptr_t run_end(std::uintptr_t from, std::uintptr_t to, bool& is_backed) {
    is_backed = backed(from);
    if (!may_hold(from, to)) return to;
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

}  // namespace leonida::title_frames
