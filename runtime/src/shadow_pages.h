#pragma once
// ShadowPages (below), the tracker behind the D3D12 layer's slot-133 shadow buffers, title windows and console
// descriptor heaps, and the write-fault handler that serves every tracker of the process (see title_page_watch.h).
// Moved here from d3d12_vulkan_device.cpp so that a test can drive it without a GPU (shadow_pages_test.cpp).
//
// Header-only, one state per process: the list of trackers and the handler are inline.
#include "guarded_copy.h"
#include "title_geometry_watch.h"
#include "title_page_watch.h"

#include <windows.h>

#include <smmintrin.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

namespace leonida::shadow_pages {

// The streaming copy of copy_guarded, in a frame of its own so that a fault in it reaches copy_guarded's __except
// (src/guarded_copy.h: not in the CRT, which Wine's unwinding does not get out of, and not inline, which clang-cl does
// not cover without /EHa). MOVNTDQA, for write-combined pages: ordinary loads from them are not cached, and memcpy
// read 64 MiB of them in 253 ms on a Core i9-13900K (253 MiB/s), MOVNTDQA in 13 to 17 ms (4 to 5 GiB/s).
#if defined(__clang__) && defined(_M_X64)
__attribute__((target("sse4.1")))
#endif
__declspec(noinline) LEONIDA_GUARDED_NO_BUILTIN inline void copy_streaming(std::uint8_t* to, const std::uint8_t* from,
                                                                    std::size_t bytes) {
    const std::size_t head = std::min<std::size_t>(bytes, (0 - reinterpret_cast<std::uintptr_t>(from)) & 63);
    __movsb(to, from, head);
    to += head, from += head, bytes -= head;
    for (; bytes >= 64; to += 64, from += 64, bytes -= 64) {
        const auto* in = reinterpret_cast<const __m128i*>(from);
        const __m128i a = _mm_stream_load_si128(in), b = _mm_stream_load_si128(in + 1);
        const __m128i c = _mm_stream_load_si128(in + 2), d = _mm_stream_load_si128(in + 3);
        auto* out = reinterpret_cast<__m128i*>(to);
        _mm_storeu_si128(out, a), _mm_storeu_si128(out + 1, b);
        _mm_storeu_si128(out + 2, c), _mm_storeu_si128(out + 3, d);
    }
    __movsb(to, from, bytes);
}

// A copy that a page fault ends instead of killing the process: the title may decommit or remap a page between the
// VirtualQuery that saw it committed and the read. No C++ objects here: it is a __try frame. `streaming`: for
// write-combined pages (copy_streaming). Paid at every submission.
inline bool copy_guarded(std::uint8_t* to, const std::uint8_t* from, std::size_t bytes, bool streaming) {
    __try {
        if (streaming) copy_streaming(to, from, bytes);
        else leonida::guarded::copy(to, from, bytes);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Copies a run of the title's pages that VirtualQuery saw committed and readable. A page decommitted since then reads
// as zeros, and only that page: the run is redone page by page, so the rest of it keeps its bytes.
inline void copy_run(std::uint8_t* to, const std::uint8_t* from, std::size_t bytes, bool streaming) {
    if (copy_guarded(to, from, bytes, streaming)) return;
    for (std::size_t off = 0; off < bytes;) {
        const std::size_t in_page = reinterpret_cast<std::uintptr_t>(from + off) & 4095;
        const std::size_t n = std::min<std::size_t>(bytes - off, 4096 - in_page);
        if (!copy_guarded(to + off, from + off, n, streaming)) std::memset(to + off, 0, n);
        off += n;
    }
}

// Once per distinct error (d3d12_vulkan_device.cpp's log_once, which this header can't reach).
inline void log_protect_refused(int error) {
    static std::mutex mutex;
    static std::set<int> seen;
    {
        std::lock_guard lock(mutex);
        if (!seen.insert(error).second) return;
    }
    std::fprintf(stderr,
                 "[rt] d3d12: slot-133 shadow: VirtualProtect refused to watch the title's pages for writes, so they are "
                 "copied at every submission; error %d\n",
                 error);
}

class ShadowPages;
// Between trackers (defined after the list of them). other_guards: in `out`, one byte per page of [from, to), 1 for the
// pages another tracker than `self` made read-only and still keeps so. forget_foreign_elsewhere: pages of [from, to)
// that `self` made read-only are writable now, so no other tracker takes them for unchanged.
inline void other_guards(const ShadowPages* self, std::uintptr_t from, std::uintptr_t to,
                         leonida::host_heap::vector<std::uint8_t>& out);
inline void forget_foreign_elsewhere(const ShadowPages* self, std::uintptr_t from, std::uintptr_t to);

// The title's pages behind slot-133 shadow buffers and title windows, and which of them the title wrote since the
// copies were last brought up to date. Nothing says which pages a title writes, so the pages say it themselves: once
// copied, a read-write page is made read-only (its cache attribute kept: PAGE_READONLY | PAGE_WRITECOMBINE for a
// write-combined one), and the title's next write to it faults. The fault handler (shadow_write_fault) makes the
// 64 KiB around it writable again and the write goes on; reopening the page clears its watched state. Each refresh()
// walks the range with VirtualQuery, passing over pages still known clean when the title memory map hasn't changed:
//   - read-write pages were written (or newly committed, or newly mapped by xmem): made read-only, then copied. A
//     write landing between the two faults again, so it reaches the next refresh at the latest;
//   - pages this tracker made read-only weren't written: skipped;
//   - pages another tracker made read-only (a console descriptor heap's tracker over the memory a slot-133 buffer is
//     placed on) are copied, and then passed over like the tracker's own (kForeign): the other tracker has the title's
//     write fault, and the fault handler tells every tracker over the address. The mark is cleared by that (reopen),
//     by a file shim's opening of the pages (reopen_range), by the other tracker giving the pages up
//     (forget_foreign_elsewhere) and by a change in the title's memory map (kRequery), like what is known of any page.
//     Copying them at every submission (27.6 MiB of each, in the retail title) took 10 of its 22 ms;
//   - other readable pages (read-only or copy-on-write pages of the title's own) can't be watched: copied every time,
//     as before;
//   - pages not committed, or not readable, are zero in the copies (written once);
//   - pages xmem reports mapped at another address too (title_page_watch.h) are copied every time: a write
//     through the other address doesn't fault here;
//   - pages a file shim pinned for a kernel write (ReadFile into title memory, which fails with ERROR_NOACCESS on a
//     read-only page instead of faulting) are left writable and copied every time until the pin ends.
// If the protection can't be changed (VirtualProtect refuses, or the fault handler couldn't be installed), those
// pages stay read-write and are copied at every submission: the whole-range copy this replaces.
//
// One tracker serves every copy of the pages it covers (its targets). A page has one owner that may make it
// read-only, so copies of the same 4 KiB page must share a tracker: the title places many slot-133 buffers of a few
// KiB back to back, off page boundaries (measured on Windows 11: 3,172 of 3,267 placements shared a page with
// another), and with a tracker each every buffer but one found its shared pages read-only, not its own, and queried
// and copied them again at every submission (3,045 VirtualQuery calls, 57 of 62 ms per refresh). DeviceState::
// add_shadow keeps one tracker per run of pages that overlap or touch, and a written page reaches every target over
// it. A page no target covers any more (its buffer went) is left writable and passed over.
//
// Pages xmem reserved with MEM_WRITE_WATCH (leonida::title_write_watch: the title's ordinary memory, not its
// physical-page views) start guarded like the rest. A write fault on one makes its 64 KiB block hot: never made
// read-only again, it is asked about instead. Each refresh first asks the kernel which hot pages were written since
// the last asking (GetWriteWatch, which clears its bits), through the journal that keeps the answer for every other
// tracker over the same pages; then those, and pages not known yet, are copied, and the rest passed over. A write
// after the asking is in the next one. No fault, no VirtualProtect, and a kernel write needs no opening: hot pages
// stay writable. A block not found written for kCool refreshes is guarded again. Asking costs about 3 ns per page per
// refresh and a fault about three orders more, so the pages written often are asked about and the rest wait for their
// write (measured natively: asking about all 242 MiB of write-watched pages at every submission cost 2.4 ms per
// refresh, against 0.14 ms for the guards with their faults). A file shim's pin still marks a hot page written
// (reopen_range) and keeps it copied until the pin ends: a device's write into locked pages need not set the kernel's
// bit.
//
// The fault handler waits for g_shadow_pages_lock and a tracker's lock_, and the faulting thread may hold the process
// heap's lock (the title's XMemAlloc memory is _aligned_malloc's, whose headers HeapAlloc and HeapFree write on pages a
// tracker may guard). So nothing that holds either lock, or a lock taken under them (title_page_aliases,
// kernel_writes' pins, title_write_watch's record and journal), touches the process heap: their containers allocate
// from leonida::host_heap, a heap of the runtime's own, and logging waits until the lock is released.
class ShadowPages {
    template <class T>
    using Vector = leonida::host_heap::vector<T>;
    using Bits = std::uint16_t;  // a page's state_: the flags below

public:
    static constexpr std::uintptr_t kPage = 0x1000;
    static constexpr std::uintptr_t kBlock = 0x10000;  // what one write fault makes writable again

    ShadowPages(const std::uint8_t* pages, std::uint8_t* shadow, std::size_t size, bool watch) : watch_(watch) {
        begin_ = reinterpret_cast<std::uintptr_t>(pages);
        end_ = begin_ + size;
        first_ = begin_ & ~(kPage - 1);
        last_ = (end_ + kPage - 1) & ~(kPage - 1);
        state_.assign((last_ - first_) / kPage, 0);
        idle_.assign(state_.size(), 0);
        cover_.assign(state_.size(), 1);
        targets_.push_back({begin_, end_, shadow});
        index_targets();
    }
    ~ShadowPages() {
        if (reader_) leonida::title_write_watch::close_reader(reader_);
    }
    // Every page made read-only here is read-write again: the title's pages outlive the buffer.
    void reopen_all() {
        AcquireSRWLockExclusive(&lock_);
        if (std::any_of(state_.begin(), state_.end(), [](Bits s) { return s & kProtected; }))
            reopen_locked(first_, last_);
        ReleaseSRWLockExclusive(&lock_);
    }
    ShadowPages(const ShadowPages&) = delete;
    ShadowPages& operator=(const ShadowPages&) = delete;

    // The bytes of all targets, and how many there are.
    std::size_t size() const { return target_bytes_; }
    std::size_t targets() const { return targets_.size(); }
    // The pages covered: [first(), last()), 4 KiB aligned.
    std::uintptr_t first() const { return first_; }
    std::uintptr_t last() const { return last_; }
    bool contains(std::uintptr_t address) const { return address >= first_ && address < last_; }
    std::uint64_t faults() const { return faults_.load(std::memory_order_relaxed); }
    std::uint64_t refresh_queries() const { return refresh_queries_; }
    std::uint64_t refresh_protects() const { return refresh_protects_; }
    bool refresh_walked_all() const { return refresh_walked_all_; }
    // GetWriteWatch calls of the last refresh(), the write-watched pages it found written, and the bytes of the
    // covered pages that are write-watched, and of those asked about (hot).
    std::uint64_t refresh_asks() const { return refresh_asks_; }
    std::uint64_t refresh_written() const { return refresh_written_; }
    std::size_t watched_bytes() const { return watched_pages_ * kPage; }
    std::size_t hot_bytes() const { return hot_pages_ * kPage; }

    // The ranges the last refresh() copied from the pages, as (offset from the first target's start, bytes). Pages it
    // zeroed are not among them. Only between refreshes, under the lock that serializes them. For a tracker with one
    // target (a console descriptor heap's), whose offsets are the buffer's.
    template <class F>
    void for_each_copied(F&& f) const {
        for (const Job& job : jobs_) f(job.offset, job.bytes);
    }

    // The ranges the last refresh() zeroed in the copies (pages no longer committed or readable), likewise.
    template <class F>
    void for_each_zeroed(F&& f) const {
        for (const Job& job : zeroed_) f(job.offset, job.bytes);
    }

    // Brings every target up to date with the pages. Returns the bytes copied into them. With copy_aliased false, pages
    // xmem maps at another address too are left as they are this time (DeviceState::refresh_shadows throttles them).
    std::size_t refresh(bool copy_aliased = true) {
        copy_aliased_ = copy_aliased;
        jobs_.clear();
        zeroed_.clear();
        copy_bytes_ = {};
        std::uint64_t queries = 0;
        const std::uint64_t generation = leonida::title_page_aliases::generation();
        // LEONIDA_SHADOW_FULL_WALK=1 queries every page at every refresh, as before the generation skip.
        static const bool always_full_walk = [] {
            char value[8] = {};
            std::size_t length = 0;
            return getenv_s(&length, value, sizeof(value), "LEONIDA_SHADOW_FULL_WALK") == 0 && length > 1 &&
                   value[0] != '0';
        }();
        AcquireSRWLockExclusive(&lock_);
        // Pages whose mapping, commitment or protection changed since the last walk are queried again (kRequery); the
        // rest keep what is known of them. A tracker that never walked, or whose changes the journal no longer holds,
        // walks everything.
        std::uint64_t map_generation = leonida::title_memory_map::generation();
        bool full_walk = always_full_walk || map_generation_ == ~std::uint64_t(0);
        if (!full_walk && map_generation != map_generation_ &&
            !leonida::title_memory_map::changes_since(
                map_generation_, map_generation, first_, last_, [&](std::uintptr_t from, std::uintptr_t to) {
                    for (std::size_t i = (from - first_) / kPage, i1 = (to - first_ + kPage - 1) / kPage;
                         i < i1 && i < state_.size(); ++i)
                        state_[i] |= kRequery;
                }))
            full_walk = true;
        refresh_walked_all_ = full_walk;
        if (full_walk)  // nothing is known to have stayed as it was
            for (Bits& s : state_) s &= ~kForeign;
        refresh_protects_ = 0;
        if (generation != alias_generation_) {
            alias_generation_ = generation;
            for (Bits& s : state_) s &= ~kAliased;
            leonida::title_page_aliases::for_each_in(first_, last_, [&](std::uintptr_t va) {
                const std::uintptr_t from = std::max(va, first_), to = std::min(va + kBlock, last_);
                for (std::uintptr_t p = from; p < to; p += kPage) state_[(p - first_) / kPage] |= kAliased;
            });
        }
        if (pinned_) {
            for (Bits& s : state_) s &= ~kPinned;
            pinned_ = false;
        }
        leonida::kernel_writes::for_each_pin(first_, last_, [&](std::uintptr_t from, std::uintptr_t to) {
            for (std::uintptr_t p = from & ~(kPage - 1); p < to; p += kPage) state_[(p - first_) / kPage] |= kPinned;
            pinned_ = true;
        });
        // The write-watched pages written since the last asking are no longer clean: asked before the walk, so a write
        // after this is in the next asking.
        const std::uint64_t watch_generation = leonida::title_write_watch::generation();
        if (watch_generation != watch_generation_) {
            watch_generation_ = watch_generation;
            update_watched();
        }
        if (runs_stale_) rebuild_watch_runs();
        refresh_asks_ = refresh_written_ = 0;
        if (!watch_runs_.empty()) {
            if (!reader_) {
                reader_ = leonida::title_write_watch::open_reader();
                forget_clean();
            }
            refresh_asks_ = watch_runs_.size();
            leonida::title_write_watch::ask(
                reader_, watch_runs_,
                [&](std::uintptr_t from, std::uintptr_t to) {
                    for (std::uintptr_t p = from; p < to; p += kPage) {
                        Bits& s = state_[(p - first_) / kPage];
                        if (!(s & kHot)) continue;  // asked over a gap between hot pages: guarded
                        refresh_written_ += (s & kClean) != 0;
                        s &= ~kClean;
                        warm(p);
                    }
                },
                [&](std::uintptr_t from, std::uintptr_t to) {
                    // Not write-watched after all (released and reserved again): guarded from now on.
                    for (std::uintptr_t p = from; p < to; p += kPage)
                        state_[(p - first_) / kPage] &= ~(kWatched | kClean | kHot);
                    runs_stale_ = true;
                });
            cool();  // a block it guards again leaves the runs at the next refresh
        }
        for (std::uintptr_t at = first_; at < last_;) {
            const std::size_t i0 = (at - first_) / kPage;
            if (!cover_[i0]) {  // no copy reads it: left as the title set it
                std::size_t i = i0 + 1;
                while (i < state_.size() && !cover_[i]) ++i;
                at = first_ + i * kPage;
                continue;
            }
            // Between the refreshes that copy aliased pages (copy_aliased_, LEONIDA_SHADOW_ALIAS_MS), pages that are
            // only aliased, and pages that were not mapped or committed when last looked at (zero in the copies), are
            // passed over like unchanged ones and queried at the next such refresh: a mapping change over them is in
            // the journal (kRequery) anyway. speed-16/17/18: 730-820 VirtualQuery calls a refresh, ~19 us each under
            // Wine (NtQueryVirtualMemory waits for Wine's virtual-memory lock, which the mapping threads hold), a
            // fifth of the submitting thread's time.
            const auto skip = [&](Bits s) {
                return unchanged(s) || (!copy_aliased_ && (s == kAliased || s == kZero || s == (kZero | kAliased)));
            };
            if (!full_walk && skip(state_[i0])) {
                std::size_t i = i0 + 1;
                while (i < state_.size() && skip(state_[i])) ++i;
                at = first_ + i * kPage;
                continue;
            }
            MEMORY_BASIC_INFORMATION region{};
            std::uintptr_t next = at + kPage;
            Kind kind = Kind::zero;
            ++queries;
            if (VirtualQuery(reinterpret_cast<const void*>(at), &region, sizeof(region))) {
                next = std::min(last_, reinterpret_cast<std::uintptr_t>(region.BaseAddress) + region.RegionSize);
                kind = classify(region);
            }
            std::size_t i1 = (next - first_) / kPage;
            for (std::size_t i = i0 + 1; i < i1; ++i)
                if (!cover_[i]) {  // the region goes on past the pages copies read: the rest is passed over above
                    i1 = i;
                    next = first_ + i * kPage;
                    break;
                }
            const bool streaming = (region.Protect & PAGE_WRITECOMBINE) != 0;
            switch (kind) {
            case Kind::zero:
                for (std::size_t i = i0; i < i1; ++i) {
                    if (state_[i] & kZero) continue;
                    const auto [offset, bytes] = span_of(i, i + 1);
                    zero_targets(offset, bytes);
                    if (!zeroed_.empty() && zeroed_.back().offset + zeroed_.back().bytes == offset)
                        zeroed_.back().bytes += bytes;
                    else
                        zeroed_.push_back({offset, bytes, false});
                    state_[i] = Bits(kZero | (state_[i] & (kAliased | kPinned | kWatched)));
                }
                break;
            case Kind::writable:
            case Kind::read_only:
                for (std::size_t i = i0; i < i1;) {
                    const Watch how = watch_of(state_[i]);
                    std::size_t j = i + 1;
                    while (j < i1 && watch_of(state_[j]) == how) ++j;
                    switch (how) {
                    case Watch::clean: break;  // write-watched, not written since it was copied
                    case Watch::written:       // write-watched: written since, or not known yet
                        for (std::size_t k = i; k < j; ++k)
                            state_[k] = Bits((state_[k] & ~(kZero | kProtected | kForeign)) | kClean);
                        add_job(i, j, streaming, Copy::write_watched);
                        break;
                    case Watch::open:  // write-watched, but pinned for a kernel write: copied until the pin ends
                        for (std::size_t k = i; k < j; ++k) state_[k] &= ~(kZero | kProtected | kClean | kForeign);
                        add_job(i, j, streaming, Copy::write_watched);
                        break;
                    case Watch::guard:
                        if (kind == Kind::writable)
                            guard_writable(i, j, region.Protect, streaming);
                        else
                            guard_read_only(i, j, streaming);
                        break;
                    }
                    i = j;
                }
                break;
            case Kind::unwatched:
                for (std::size_t i = i0; i < i1; ++i) state_[i] &= kAliased | kPinned | kWatched;
                add_job(i0, i1, streaming, Copy::copy_on_write);
                break;
            }
            for (std::size_t i = i0; i < i1; ++i) state_[i] &= Bits(~kRequery);  // queried now
            at = next;
        }
        // Remember the generation from before the walk, so a change during the walk triggers another one.
        map_generation_ = map_generation;
        const DWORD protect_error = std::exchange(protect_error_, 0);
        ReleaseSRWLockExclusive(&lock_);
        if (protect_error)
            log_protect_refused(int(protect_error));
        // The copies run without the lock: a write that faults meanwhile is let through and copied next time.
        std::size_t copied = 0;
        for (const Job& job : jobs_)
            for_each_target(begin_ + job.offset, begin_ + job.offset + job.bytes,
                            [&](const Target& t, std::uintptr_t from, std::uintptr_t to) {
                                copy_run(t.shadow + (from - t.begin), reinterpret_cast<const std::uint8_t*>(from),
                                         to - from, job.streaming);
                                copied += to - from;
                            });
        refresh_queries_ = queries;
        return copied;
    }

    // DeviceState::add_shadow, with every other tracker of its device held still (shadow_mutex) and the fault
    // handler kept out (g_shadow_pages_lock exclusive). The pages [first, last) (4 KiB aligned) are covered from
    // now on; the ones new here are unknown, so the next refresh queries them.
    void extend(std::uintptr_t first, std::uintptr_t last) {
        AcquireSRWLockExclusive(&lock_);
        if (first < first_) {
            const std::size_t n = (first_ - first) / kPage;
            state_.insert(state_.begin(), n, Bits(0));
            idle_.insert(idle_.begin(), n, std::uint8_t(0));
            cover_.insert(cover_.begin(), n, 0u);
            first_ = first;
        }
        if (last > last_) {
            state_.resize((last - first_) / kPage, 0);
            idle_.resize(state_.size(), 0);
            cover_.resize(state_.size(), 0);
            last_ = last;
        }
        alias_generation_ = ~std::uint64_t(0);  // the new pages' alias marks
        watch_generation_ = ~std::uint64_t(0);  // and whether they are write-watched
        ReleaseSRWLockExclusive(&lock_);
    }

    // Likewise: takes over `other`'s targets, pages and what is known of them (inside this tracker's range, which
    // never shares a page with it before). `other` is left with nothing to reopen: the caller unregisters and
    // deletes it, and the pages it made read-only are this tracker's now.
    void absorb(ShadowPages& other) {
        AcquireSRWLockExclusive(&lock_);
        AcquireSRWLockExclusive(&other.lock_);
        const std::size_t at = (other.first_ - first_) / kPage;
        for (std::size_t i = 0; i < other.state_.size(); ++i) {
            state_[at + i] |= other.state_[i];
            if (state_[at + i] & kProtected) state_[at + i] &= ~kForeign;  // guarded here now
            cover_[at + i] += other.cover_[i];
            idle_[at + i] = other.idle_[i];
        }
        targets_.insert(targets_.end(), other.targets_.begin(), other.targets_.end());
        if (other.map_generation_ != map_generation_) map_generation_ = ~std::uint64_t(0);
        // Its clean write-watched pages are clean as of its last asking: this tracker asks from the older of the two.
        if (other.reader_) {
            if (!reader_) {
                reader_ = leonida::title_write_watch::open_reader();
                forget_clean();
            }
            leonida::title_write_watch::merge_reader(reader_, other.reader_);
        }
        watch_generation_ = ~std::uint64_t(0);
        pinned_ |= other.pinned_;
        watch_ = watch_ && other.watch_;
        faults_.fetch_add(other.faults(), std::memory_order_relaxed);
        std::fill(other.state_.begin(), other.state_.end(), Bits(0));
        other.targets_.clear();
        ReleaseSRWLockExclusive(&other.lock_);
        index_targets();
        ReleaseSRWLockExclusive(&lock_);
    }

    // Likewise: [begin, end), inside the covered pages, is copied to `shadow` from now on. fill() it next.
    void add_target(std::uintptr_t begin, std::uintptr_t end, std::uint8_t* shadow) {
        AcquireSRWLockExclusive(&lock_);
        targets_.push_back({begin, end, shadow});
        for (std::uintptr_t p = begin & ~(kPage - 1); p < end; p += kPage) ++cover_[(p - first_) / kPage];
        index_targets();
        ReleaseSRWLockExclusive(&lock_);
    }

    // A new target's first copy, of the pages as they are now; their watched state is not touched. A page this
    // tracker keeps read-only is as the other targets have it; one written meanwhile is copied again at the next
    // refresh, like any page that isn't read-only here. Under shadow_mutex, so no refresh runs meanwhile.
    void fill(std::uintptr_t begin, std::uintptr_t end, std::uint8_t* shadow) {
        for (std::uintptr_t at = begin & ~(kPage - 1); at < end;) {
            MEMORY_BASIC_INFORMATION region{};
            std::uintptr_t next = at + kPage;
            Kind kind = Kind::zero;
            if (VirtualQuery(reinterpret_cast<const void*>(at), &region, sizeof(region))) {
                next = reinterpret_cast<std::uintptr_t>(region.BaseAddress) + region.RegionSize;
                kind = classify(region);
            }
            const std::uintptr_t from = std::max(at, begin), to = std::min(next, end);
            if (kind == Kind::zero)
                std::memset(shadow + (from - begin), 0, to - from);
            else
                copy_run(shadow + (from - begin), reinterpret_cast<const std::uint8_t*>(from), to - from,
                         (region.Protect & PAGE_WRITECOMBINE) != 0);
            at = next;
        }
    }

    // Under shadow_mutex: `shadow` is no longer a target. Pages no other target covers are made writable again and
    // passed over from now on. False when it isn't one of this tracker's.
    bool remove_target(const std::uint8_t* shadow) {
        const auto it =
            std::find_if(targets_.begin(), targets_.end(), [shadow](const Target& t) { return t.shadow == shadow; });
        if (it == targets_.end()) return false;
        const std::uintptr_t removed_begin = it->begin, removed_end = it->end;
        AcquireSRWLockExclusive(&lock_);
        const std::size_t i_end = (((it->end + kPage - 1) & ~(kPage - 1)) - first_) / kPage;
        for (std::size_t i = ((it->begin & ~(kPage - 1)) - first_) / kPage; i < i_end;) {
            if (--cover_[i]) {
                ++i;
                continue;
            }
            std::size_t j = i + 1;  // [i, j): pages nothing covers now
            for (; j < i_end && cover_[j] == 1; ++j) cover_[j] = 0;
            reopen_locked(first_ + i * kPage, first_ + j * kPage);
            for (std::size_t k = i; k < j; ++k) state_[k] &= kAliased | kPinned | kWatched;
            i = j;
        }
        targets_.erase(it);
        index_targets();
        watch_generation_ = ~std::uint64_t(0);  // recounts the write-watched pages covered
        const std::uintptr_t from = std::max(removed_begin & ~(kPage - 1), first_);
        const std::uintptr_t to = std::min((removed_end + kPage - 1) & ~(kPage - 1), last_);
        ReleaseSRWLockExclusive(&lock_);
        forget_foreign_elsewhere(this, from, to);  // the pages made writable above: no one else keeps them as copied
        return true;
    }

    // For another tracker's refresh (other_guards, with g_shadow_pages_lock held shared): marks in `out`, one byte per
    // page from `out_first` on, the pages of [from, to) this tracker made read-only and still has so. Doesn't wait:
    // false, nothing marked, when the lock is busy (a refresh, a fault); the pages are copied again then.
    bool guards(std::uintptr_t from, std::uintptr_t to, Vector<std::uint8_t>& out, std::uintptr_t out_first) {
        from = std::max(from, first_);
        to = std::min(to, last_);
        if (from >= to) return true;
        if (!TryAcquireSRWLockExclusive(&lock_)) return false;
        for (std::uintptr_t p = from; p < to; p += kPage)
            if (state_[(p - first_) / kPage] & kProtected) out[(p - out_first) / kPage] = 1;
        ReleaseSRWLockExclusive(&lock_);
        return true;
    }

    // Pages of [from, to) that another tracker made read-only were opened (that tracker's end, or its giving them up,
    // forget_foreign_elsewhere): they are copied at the next refresh.
    void forget_foreign(std::uintptr_t from, std::uintptr_t to) {
        from = std::max(from & ~(kPage - 1), first_);
        to = std::min((to + kPage - 1) & ~(kPage - 1), last_);
        if (from >= to) return;
        AcquireSRWLockExclusive(&lock_);
        forget_foreign_locked(from, to);
        ReleaseSRWLockExclusive(&lock_);
    }

    // From a file shim, before the kernel writes [from, to): the pages there this tracker protected are writable again,
    // and the write-watched ones count as written (a device writing into locked pages need not set the kernel's bit).
    void reopen_range(std::uintptr_t from, std::uintptr_t to) {
        from = std::max(from & ~(kPage - 1), first_);
        to = std::min((to + kPage - 1) & ~(kPage - 1), last_);
        if (from >= to) return;
        AcquireSRWLockExclusive(&lock_);
        bool guarded = false;
        for (std::size_t i = (from - first_) / kPage, i1 = (to - first_) / kPage; i < i1; ++i) {
            state_[i] &= ~(kClean | kForeign);  // the kernel writes it: whoever keeps it read-only, it is not unchanged
            guarded |= (state_[i] & kProtected) != 0;
        }
        if (guarded) reopen_locked(from, to);
        ReleaseSRWLockExclusive(&lock_);
    }

    // From the fault handler: a write to `address`, inside this range, faulted. True when the write can be retried:
    // the page is one this tracker made read-only, and the 64 KiB around it that it protected are read-write again;
    // or the page is writable already, because another thread's fault reopened it after this write faulted. The
    // second is asked here, under the lock, not by the handler afterwards: refresh() protects pages under the same
    // lock, so by a later look the page can be read-only again (and this tracker's again), and the handler refused a
    // write it should have retried, which ended the process (seen natively on Windows 11: threads writing one page
    // across a submission).
    bool reopen(std::uintptr_t address) {
        AcquireSRWLockExclusive(&lock_);
        const std::size_t index = (address - first_) / kPage;
        // A write to this address faulted, whoever's guard it was: the pages of its 64 KiB block that another tracker
        // kept read-only are writable now (that tracker reopens its whole block), so none of them is unchanged.
        const std::uintptr_t block_first = address & ~(kBlock - 1);
        forget_foreign_locked(std::max(block_first, first_), std::min(block_first + kBlock, last_));
        bool retry = (state_[index] & kProtected) != 0;
        if (retry) {
            const std::uintptr_t block = address & ~(kBlock - 1);
            const std::uintptr_t from = std::max(block, first_), to = std::min(block + kBlock, last_);
            // A write-watched block written again: asked about from now on, not guarded (what reopen_locked left
            // read-only stays guarded). No allocation in here: the faulting thread may hold the heap's lock.
            bool watched[kBlock / kPage] = {};
            const std::size_t i0 = (from - first_) / kPage, i1 = (to - first_) / kPage;
            for (std::size_t i = i0; i < i1; ++i)
                watched[i - i0] = (state_[i] & (kWatched | kHot | kProtected)) == (kWatched | kProtected);
            reopen_locked(from, to);
            retry = !(state_[index] & kProtected);
            faults_.fetch_add(1, std::memory_order_relaxed);
            for (std::size_t i = i0; i < i1; ++i) {
                if (!watched[i - i0] || (state_[i] & kProtected)) continue;
                state_[i] = Bits((state_[i] | kHot) & ~kClean);
                idle_[i] = 0;
                runs_stale_ = true;
            }
        } else {
            MEMORY_BASIC_INFORMATION region{};
            retry = VirtualQuery(reinterpret_cast<const void*>(address), &region, sizeof(region)) &&
                    classify(region) == Kind::writable;
        }
        ReleaseSRWLockExclusive(&lock_);
        return retry;
    }

private:
    enum : Bits {
        kProtected = 1,  // made read-only here: unchanged since it was copied
        kZero = 2,       // zero in the copies (not committed when last looked at)
        kAliased = 4,    // xmem maps its physical page at another address too
        kPinned = 8,     // a kernel write into it may be in flight (leonida::kernel_writes)
        kWatched = 16,   // in a write-watched reservation (leonida::title_write_watch)
        kClean = 32,     // hot, and not written since it was copied after an asking
        kHot = 64,       // write-watched and asked about at every refresh, never guarded
        kRequery = 128,  // its mapping, commitment or protection changed (title_memory_map): queried at the next walk
        kForeign = 256,  // read-only, made so by another tracker (guards()), copied: unchanged until it is reopened
    };
    // Refreshes a hot block may go without a write before it is guarded again.
    static constexpr std::uint8_t kCool = 64;
    // Hot runs this close in one reservation are asked about as one (a call costs about as much as 300 pages).
    static constexpr std::uintptr_t kMergeGap = 0x40000;
    enum class Kind { zero, writable, read_only, unwatched };

public:
    // Why the last refresh() copied the bytes it did (copy_bytes): pages guarded now (written since, or new), pages left
    // writable (no guard possible), pages xmem maps at another address too, pages pinned for a kernel write,
    // read-only pages that are not this tracker's, copy-on-write pages, write-watched pages found written.
    enum class Copy { guarded, unguarded, aliased, pinned, read_only, copy_on_write, write_watched, count };
    const std::array<std::uint64_t, unsigned(Copy::count)>& copy_bytes() const { return copy_bytes_; }

private:
    // How a committed, readable page is watched: by the kernel (clean, written or, pinned or aliased, open) or by
    // a guard.
    enum class Watch { clean, written, open, guard };
    static Watch watch_of(Bits s) {
        if ((s & (kWatched | kHot)) != (kWatched | kHot)) return Watch::guard;
        if (s & (kAliased | kPinned)) return Watch::open;
        return s & kClean ? Watch::clean : Watch::written;
    }
    // A page known unchanged since the last refresh, which one that doesn't walk everything passes over.
    static bool unchanged(Bits s) {
        return s == kProtected || s == (kWatched | kProtected) || s == (kWatched | kHot | kClean) || s == kForeign ||
               s == (kWatched | kForeign);
    }

    // With the lock held: no page of [from, to) (inside the covered pages) is known to be an unchanged foreign one.
    void forget_foreign_locked(std::uintptr_t from, std::uintptr_t to) {
        for (std::size_t i = (from - first_) / kPage, i1 = (to - first_) / kPage; i < i1; ++i) state_[i] &= ~kForeign;
    }

    // A hot page `p` was found written: its block is not idle.
    void warm(std::uintptr_t p) {
        const std::uintptr_t block = p & ~(kBlock - 1);
        const std::size_t i0 = (std::max(block, first_) - first_) / kPage;
        const std::size_t i1 = (std::min(block + kBlock, last_) - first_) / kPage;
        for (std::size_t i = i0; i < i1; ++i) idle_[i] = 0;
    }

    // After an asking: the hot pages not found written are one refresh idler, and those idle for kCool refreshes are
    // guarded from now on (not clean: the walk copies them as it makes them read-only).
    void cool() {
        for (const auto& [from, to] : watch_runs_)
            for (std::size_t i = (from - first_) / kPage, i1 = (to - first_) / kPage; i < i1; ++i) {
                if (!(state_[i] & kHot)) continue;
                if (idle_[i] < kCool) {
                    ++idle_[i];
                    continue;
                }
                state_[i] &= ~(kHot | kClean);
                idle_[i] = 0;
                runs_stale_ = true;
            }
    }

    // Guarded pages [i, j) of a read-write region (protection `current`): written since the last refresh, or newly
    // committed or mapped. Watched from now on, then copied.
    void guard_writable(std::size_t i0, std::size_t i1, DWORD current, bool streaming) {
        for (std::size_t i = i0; i < i1;) {
            const bool unwatched = state_[i] & (kAliased | kPinned);
            std::size_t j = i + 1;
            while (j < i1 && bool(state_[j] & (kAliased | kPinned)) == unwatched) ++j;
            for (std::size_t k = i; k < j; ++k) state_[k] &= ~(kZero | kProtected | kForeign);
            if (!unwatched && watch_) protect(i, j, current);
            if (unwatched && !copy_aliased_ && (state_[i] & kAliased)) {
                i = j;
                continue;
            }
            add_job(i, j, streaming,
                    !unwatched                    ? (watch_ && (state_[i] & kProtected) ? Copy::guarded : Copy::unguarded)
                    : (state_[i] & kAliased) ? Copy::aliased
                                             : Copy::pinned);
            i = j;
        }
    }

    // Guarded pages [i, j) of a read-only region: pages this tracker protected, and pages another tracker did that were
    // copied since (kForeign), are as the last refresh left them; any other is copied again.
    void guard_read_only(std::size_t i0, std::size_t i1, bool streaming) {
        for (std::size_t i = i0; i < i1;) {
            const bool clean = quiet(state_[i]);
            std::size_t j = i + 1;
            while (j < i1 && quiet(state_[j]) == clean) ++j;
            if (!clean && !(!copy_aliased_ && (state_[i] & kAliased))) {
                for (std::size_t k = i; k < j; ++k) state_[k] &= ~(kZero | kForeign);
                add_job(i, j, streaming,
                        state_[i] & kAliased  ? Copy::aliased
                        : state_[i] & kPinned ? Copy::pinned
                                              : Copy::read_only);
                note_foreign(i, j);
            }
            i = j;
        }
    }

    // A read-only page left as it is at this refresh: one this tracker made read-only, or another's that was copied
    // since (and has not been reopened, or changed, since).
    static bool quiet(Bits s) {
        if (s & (kAliased | kPinned)) return false;
        return (s & kProtected) || (s & (kForeign | kRequery)) == kForeign;
    }

    // The pages [i, j) were read-only and not this tracker's, and are copied now. The ones another tracker made
    // read-only (and so reopens when the title writes there: the fault handler tells every tracker over the address,
    // reopen()) need no copy at the next refresh unless it did: they are marked, and a write, a kernel write
    // (reopen_range), the end of the other tracker's guard (forget_foreign), or a change in the title's memory map
    // (kRequery) clears the mark. Other read-only pages (the title's own) are copied at every refresh, as before.
    void note_foreign(std::size_t i, std::size_t j) {
        Vector<std::uint8_t> theirs(j - i, 0);
        other_guards(this, first_ + i * kPage, first_ + j * kPage, theirs);
        for (std::size_t k = i; k < j; ++k)
            if (theirs[k - i] && !(state_[k] & (kAliased | kPinned))) state_[k] |= kForeign;
    }

    // With the lock held, when this tracker opens a new write-watch reader: no page is clean until that reader has
    // asked and the walk has copied it. The runs may have closed (their reservation released) and come back over a
    // reservation made at the same address since, which a page's watched mark alone doesn't tell (re-review of #61).
    void forget_clean() {
        for (Bits& page : state_) page &= Bits(~kClean);
    }

    // With the lock held, after the write-watched reservations changed: which pages are write-watched now. A page
    // that changes sides starts guarded: one this tracker keeps read-only stays so, and a hot one is not known any
    // more (the next walk queries, guards and copies it).
    void update_watched() {
        // Pages [i, upto) are write-watched now, or not. Either way a page that changes sides is guarded now (a guard
        // it has stays): only a hot one changes, to not known.
        std::size_t i = 0;
        const auto side = [&](std::size_t upto, bool watched) {
            for (; i < upto; ++i)
                if (bool(state_[i] & kWatched) != watched)
                    state_[i] =
                        Bits((state_[i] & ~(kWatched | kClean | kHot | kZero)) | (watched ? kWatched : 0));
        };
        leonida::title_write_watch::for_each_in(first_, last_, [&](std::uintptr_t from, std::uintptr_t to) {
            side((from - first_) / kPage, false);
            side((to - first_) / kPage, true);
        });
        side(state_.size(), false);
        rebuild_watch_runs();
    }

    // The runs of hot pages to ask about, each inside one reservation, from the pages' marks; runs less than kMergeGap
    // apart are asked about as one.
    void rebuild_watch_runs() {
        runs_stale_ = false;
        Vector<std::pair<std::uintptr_t, std::uintptr_t>>& runs = watch_runs_;
        runs.clear();
        const auto hot = [&](std::uintptr_t p) {
            return (state_[(p - first_) / kPage] & (kWatched | kHot)) == (kWatched | kHot);
        };
        leonida::title_write_watch::for_each_in(first_, last_, [&](std::uintptr_t from, std::uintptr_t to) {
            const std::size_t first_run = runs.size();
            for (std::uintptr_t p = from; p < to;) {
                if (!hot(p)) {
                    p += kPage;
                    continue;
                }
                std::uintptr_t q = p + kPage;
                while (q < to && hot(q)) q += kPage;
                if (runs.size() > first_run && p - runs.back().second < kMergeGap)
                    runs.back().second = q;
                else
                    runs.emplace_back(p, q);
                p = q;
            }
        });
        watched_pages_ = hot_pages_ = 0;
        for (std::size_t i = 0; i < state_.size(); ++i) {
            watched_pages_ += (state_[i] & kWatched) && cover_[i];
            hot_pages_ += (state_[i] & kHot) && cover_[i];
        }
        if (watch_runs_.empty() && reader_) {
            leonida::title_write_watch::close_reader(reader_);
            reader_ = 0;
        }
    }
    struct Job {
        std::size_t offset, bytes;  // from begin_
        bool streaming;
    };
    struct Target {
        std::uintptr_t begin, end;  // the title's bytes
        std::uint8_t* shadow;       // where begin's byte goes
    };

    static Kind classify(const MEMORY_BASIC_INFORMATION& region) {
        if (region.State != MEM_COMMIT || (region.Protect & PAGE_GUARD)) return Kind::zero;
        switch (region.Protect & 0xff) {
        case PAGE_READWRITE:
        case PAGE_EXECUTE_READWRITE: return Kind::writable;
        case PAGE_READONLY:
        case PAGE_EXECUTE_READ: return Kind::read_only;
        case PAGE_WRITECOPY:
        case PAGE_EXECUTE_WRITECOPY: return Kind::unwatched;
        default: return Kind::zero;  // PAGE_NOACCESS, PAGE_EXECUTE: not readable
        }
    }

    // Offset from begin_ and size of pages [i, j): the first and last pages may stick out of the targets.
    std::pair<std::size_t, std::size_t> span_of(std::size_t i, std::size_t j) const {
        const std::uintptr_t from = std::max(first_ + i * kPage, begin_), to = std::min(first_ + j * kPage, end_);
        return {from - begin_, to - from};
    }

    void add_job(std::size_t i, std::size_t j, bool streaming, Copy why) {
        const auto [offset, bytes] = span_of(i, j);
        copy_bytes_[unsigned(why)] += bytes;
        if (!jobs_.empty() && jobs_.back().streaming == streaming &&
            jobs_.back().offset + jobs_.back().bytes == offset) {
            jobs_.back().bytes += bytes;
            return;
        }
        jobs_.push_back({offset, bytes, streaming});
    }

    // The targets sorted by start, the bytes they hold, and reach_[k], the furthest end among targets_[0..k]: the
    // targets over [from, to) are found from the first whose reach passes `from`.
    void index_targets() {
        std::sort(targets_.begin(), targets_.end(), [](const Target& a, const Target& b) { return a.begin < b.begin; });
        reach_.resize(targets_.size());
        target_bytes_ = 0;
        std::uintptr_t reach = 0;
        for (std::size_t k = 0; k < targets_.size(); ++k) {
            reach = std::max(reach, targets_[k].end);
            reach_[k] = reach;
            target_bytes_ += targets_[k].end - targets_[k].begin;
        }
        if (!targets_.empty()) {
            begin_ = targets_.front().begin;
            end_ = reach;
        }
    }

    // f(target, from, to) for each target's part of the title's bytes [from, to).
    template <class F>
    void for_each_target(std::uintptr_t from, std::uintptr_t to, F&& f) const {
        auto k = std::size_t(std::upper_bound(reach_.begin(), reach_.end(), from) - reach_.begin());
        for (; k < targets_.size() && targets_[k].begin < to; ++k) {
            const Target& t = targets_[k];
            const std::uintptr_t a = std::max(from, t.begin), b = std::min(to, t.end);
            if (a < b) f(t, a, b);
        }
    }

    void zero_targets(std::size_t offset, std::size_t bytes) {
        for_each_target(begin_ + offset, begin_ + offset + bytes, [](const Target& t, std::uintptr_t a, std::uintptr_t b) {
            std::memset(t.shadow + (a - t.begin), 0, b - a);
        });
    }

    // Pages [i, j) of one region, read-write with protection `current`: read-only from now on, same cache attribute.
    void protect(std::size_t i, std::size_t j, DWORD current) {
        const DWORD base = current & 0xff;
        const DWORD read_only = (current & ~DWORD(0xff)) | (base == PAGE_READWRITE ? PAGE_READONLY : PAGE_EXECUTE_READ);
        DWORD old = 0;
        if (!VirtualProtect(reinterpret_cast<void*>(first_ + i * kPage), (j - i) * kPage, read_only, &old)) {
            protect_error_ = GetLastError();  // logged once the lock is released
            return;
        }
        for (std::size_t k = i; k < j; ++k) state_[k] |= kProtected;
        ++refresh_protects_;
    }

    // With the lock held: the pages in [from, to) this tracker made read-only are read-write again.
    void reopen_locked(std::uintptr_t from, std::uintptr_t to) {
        for (std::uintptr_t at = from; at < to;) {
            MEMORY_BASIC_INFORMATION region{};
            std::uintptr_t next = at + kPage;
            DWORD base = 0;
            if (VirtualQuery(reinterpret_cast<const void*>(at), &region, sizeof(region))) {
                next = std::min(to, reinterpret_cast<std::uintptr_t>(region.BaseAddress) + region.RegionSize);
                if (region.State == MEM_COMMIT) base = region.Protect & 0xff;
            }
            const std::size_t i0 = (at - first_) / kPage, i1 = (next - first_) / kPage;
            if (base != PAGE_READONLY && base != PAGE_EXECUTE_READ) {
                // Not read-only any more (remapped, decommitted, opened by another thread): nothing to undo.
                for (std::size_t i = i0; i < i1; ++i) state_[i] &= ~kProtected;
            } else {
                const DWORD writable =
                    (region.Protect & ~DWORD(0xff)) | (base == PAGE_READONLY ? PAGE_READWRITE : PAGE_EXECUTE_READWRITE);
                for (std::size_t i = i0; i < i1;) {
                    const bool ours = state_[i] & kProtected;
                    std::size_t j = i + 1;
                    while (j < i1 && bool(state_[j] & kProtected) == ours) ++j;
                    DWORD old = 0;
                    if (ours &&
                        VirtualProtect(reinterpret_cast<void*>(first_ + i * kPage), (j - i) * kPage, writable, &old))
                    {
                        for (std::size_t k = i; k < j; ++k) state_[k] &= ~kProtected;
                        leonida::geometry_writes::notify(first_ + i * kPage, first_ + j * kPage,
                                                        leonida::geometry_writes::Event::write);
                    }
                    i = j;
                }
            }
            at = next;
        }
    }

    bool watch_;
    std::uintptr_t begin_ = 0, end_ = 0;   // the targets' bytes: the first start and the furthest end
    std::uintptr_t first_ = 0, last_ = 0;  // the pages covered
    SRWLOCK lock_ = SRWLOCK_INIT;          // state_, cover_, the targets, and the protection of the pages
    Vector<Bits> state_;                   // per 4 KiB page
    Vector<std::uint32_t> cover_;          // per 4 KiB page: the targets that read it
    Vector<Target> targets_;               // by start
    Vector<std::uintptr_t> reach_;         // see index_targets
    std::size_t target_bytes_ = 0;
    std::uint64_t alias_generation_ = ~std::uint64_t(0);
    std::uint64_t map_generation_ = ~std::uint64_t(0);
    bool pinned_ = false;  // some page has kPinned
    Vector<Job> jobs_;    // refresh() only, which DeviceState::shadow_mutex serializes
    Vector<Job> zeroed_;  // likewise
    DWORD protect_error_ = 0;  // a VirtualProtect refusal of this refresh(), logged after it
    std::array<std::uint64_t, unsigned(Copy::count)> copy_bytes_{};  // refresh() only, likewise
    bool copy_aliased_ = true;                                        // likewise
    std::uint64_t refresh_queries_ = 0;
    std::uint64_t refresh_protects_ = 0;  // VirtualProtect calls of the last refresh()
    bool refresh_walked_all_ = false;     // the last refresh() queried every page (the title memory map changed)
    std::atomic<std::uint64_t> faults_{0};
    // Write watching (leonida::title_write_watch): the reservations' generation the kWatched marks are from, the runs
    // of hot pages, this tracker's reader of the journal (0: none yet) and what the last refresh() asked.
    std::uint64_t watch_generation_ = ~std::uint64_t(0);
    Vector<std::pair<std::uintptr_t, std::uintptr_t>> watch_runs_;
    std::uint64_t reader_ = 0;
    std::size_t watched_pages_ = 0, hot_pages_ = 0;  // covered pages write-watched, and hot
    Vector<std::uint8_t> idle_;  // per 4 KiB page: refreshes a hot page went without a write (kCool)
    bool runs_stale_ = false;         // a page became hot or cold: watch_runs_ is rebuilt at the next refresh
    std::uint64_t refresh_asks_ = 0;
    std::uint64_t refresh_written_ = 0;
};

// The live ShadowPages of every device, for the fault handler. Never freed: a fault can come at any time. In host_heap,
// like everything changed under g_shadow_pages_lock (see ShadowPages).
inline SRWLOCK g_shadow_pages_lock = SRWLOCK_INIT;
inline leonida::host_heap::vector<ShadowPages*>& shadow_pages_list() {
    static auto* list = new leonida::host_heap::vector<ShadowPages*>;
    return *list;
}

// A write to a page a shadow made read-only: the page is made writable again and the write retried. First in the
// handler chain, so a startup trace doesn't log these as exceptions; any other access violation goes on down it.
// Whatever thread wrote: the title's, or a host thread of the runtime's (a DirectStorage queue's copy).
//
// A physical page xmem maps at two addresses (a stream pool's CPU view and its GPU-only twin) has a protection per
// address: a guard on one view never faults a write through the other, which the shadows see by copying aliased
// pages at every submission (title_page_aliases), so the guard to look for is the faulting address's own. When none
// answers for it (each guard reopens only what it protected), xmem makes the page writable again if its window is
// (kernel_writes' unguarder): a read-only page nobody reopens would fault that write, and every later one, for good.
inline LONG CALLBACK shadow_write_fault(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* record = info ? info->ExceptionRecord : nullptr;
    if (!record || record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 ||
        record->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;
    const auto address = static_cast<std::uintptr_t>(record->ExceptionInformation[1]);
    bool tile_reopened = false;
    bool retry = leonida::geometry_writes::notify(address, address + 1, leonida::geometry_writes::Event::fault,
                                                  &tile_reopened);
    // The geometry watch reopened the whole 64 KiB tile around the address: the shadows over it reopen it too, so a
    // page both guarded is not left writable while its shadow takes it for unchanged.
    constexpr std::uintptr_t kTile = leonida::geometry_writes::tile_size;
    const std::uintptr_t tile = address & ~(kTile - 1);
    const std::uintptr_t block = address & ~(ShadowPages::kBlock - 1);
    AcquireSRWLockShared(&g_shadow_pages_lock);
    for (ShadowPages* pages : shadow_pages_list()) {
        if (pages->contains(address))
            retry |= pages->reopen(address);  // reopened, or another thread's fault did
        else if (pages->first() < block + ShadowPages::kBlock && block < pages->last())
            pages->forget_foreign(block, block + ShadowPages::kBlock);  // a tracker over the address reopened this block
        if (tile_reopened && pages->first() < tile + kTile && tile < pages->last()) pages->reopen_range(tile, tile + kTile);
    }
    ReleaseSRWLockShared(&g_shadow_pages_lock);
    if (!retry)
        if (const auto unguard = leonida::kernel_writes::state().unguarder.load(std::memory_order_acquire))
            retry = unguard(address, address + 1);
    return retry ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

// leonida::kernel_writes' opener: a file shim is about to let the kernel write [from, to).
inline void open_shadow_pages(std::uintptr_t from, std::uintptr_t to) {
    bool tiles_reopened = false;
    leonida::geometry_writes::notify(from, to, leonida::geometry_writes::Event::open, &tiles_reopened);
    if (tiles_reopened) {  // whole tiles were reopened: so are the shadows' pages there (see shadow_write_fault)
        constexpr std::uintptr_t kTile = leonida::geometry_writes::tile_size;
        from &= ~(kTile - 1);
        to = (to + kTile - 1) & ~(kTile - 1);
    }
    AcquireSRWLockShared(&g_shadow_pages_lock);
    for (ShadowPages* pages : shadow_pages_list()) pages->reopen_range(from, to);
    ReleaseSRWLockShared(&g_shadow_pages_lock);
}

// Installs the handler once. false: it couldn't be, and shadows copy every page at every submission.
inline bool watch_shadow_writes() {
    static const bool installed = [] {
        if (!AddVectoredExceptionHandler(1, &shadow_write_fault)) return false;
        leonida::kernel_writes::set_opener(&open_shadow_pages);
        return true;
    }();
    return installed;
}

// A ShadowPages the fault handler knows of for its whole life: registered before it protects anything, removed once
// its destructor has made every page writable again.
struct ShadowPagesDeleter {
    void operator()(ShadowPages* pages) const {
        pages->reopen_all();
        AcquireSRWLockExclusive(&g_shadow_pages_lock);
        std::erase(shadow_pages_list(), pages);
        // Its pages are writable now, with no fault to tell the trackers that took them for unchanged read-only ones.
        for (ShadowPages* other : shadow_pages_list()) other->forget_foreign(pages->first(), pages->last());
        ReleaseSRWLockExclusive(&g_shadow_pages_lock);
        delete pages;
    }
};
using ShadowPagesPtr = std::unique_ptr<ShadowPages, ShadowPagesDeleter>;

inline ShadowPagesPtr make_shadow_pages(const std::uint8_t* pages, std::uint8_t* shadow, std::size_t size) {
    const bool watch = watch_shadow_writes();
    auto* tracker = new ShadowPages(pages, shadow, size, watch);
    if (watch) {
        shadow_pages_list();  // made (on the process heap) before the lock the fault handler waits for
        AcquireSRWLockExclusive(&g_shadow_pages_lock);
        shadow_pages_list().push_back(tracker);
        ReleaseSRWLockExclusive(&g_shadow_pages_lock);
    }
    return ShadowPagesPtr(tracker);
}

inline void other_guards(const ShadowPages* self, std::uintptr_t from, std::uintptr_t to,
                         leonida::host_heap::vector<std::uint8_t>& out) {
    // Not waiting for anything: this runs under a tracker's lock, and the fault handler holds this one while it waits
    // for trackers' locks. A tracker that can't be asked is a tracker whose pages are copied again.
    if (!TryAcquireSRWLockShared(&g_shadow_pages_lock)) return;
    for (ShadowPages* other : shadow_pages_list())
        if (other != self && other->first() < to && from < other->last()) other->guards(from, to, out, from);
    ReleaseSRWLockShared(&g_shadow_pages_lock);
}

inline void forget_foreign_elsewhere(const ShadowPages* self, std::uintptr_t from, std::uintptr_t to) {
    AcquireSRWLockShared(&g_shadow_pages_lock);
    for (ShadowPages* other : shadow_pages_list())
        if (other != self && other->first() < to && from < other->last()) other->forget_foreign(from, to);
    ReleaseSRWLockShared(&g_shadow_pages_lock);
}

}  // namespace leonida::shadow_pages
