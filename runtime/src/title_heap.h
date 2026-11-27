#pragma once
// The XMemAlloc heap (title_heap): the title's XMemAlloc / XMemAllocDefault blocks, carved from one reservation that
// xmem places in the page mirror's range (src/title_mirror_space.h), so a V# or pointer the title writes into a block
// reaches the mirror at heap_base + the block's address. Windows' RtlCreateHeap over a given base faults under Wine
// (it writes into the reserved base), so the heap is the runtime's own.
//
// What the title sees, as before: every block 64-byte aligned, a size of zero one byte, blocks that never overlap,
// XMemFree of a block from either entry point. There is no realloc and no size query in the title's ABI; size_of
// answers xmem's own bookkeeping (the bytes a free tells the readers about).
//
// The layout:
//   - the reservation is handed out in 64 KiB pages by a first-fit list of free runs, coalesced on free;
//   - blocks up to kSmallMax come from spans of kSpan bytes, one size class a span (multiples of 64 bytes up to 1 KiB,
//     then powers of two), each class with a stack of free blocks kept outside the title's memory (a freed block's
//     bytes are never written by the heap);
//   - larger blocks are runs of whole pages, committed when handed out and decommitted when freed (they read zero when
//     handed out again, as HeapAlloc's large blocks do).
// Pages are committed through the hooks (xmem: VirtualAlloc, then the page mirror's feed). One lock, taken for every
// call; nothing the title's memory holds is read or written under it. Portable: the hooks do the system calls.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

namespace leonida {

class TitleHeap {
public:
    static constexpr std::size_t kAlign = 64;
    static constexpr std::size_t kPage = 0x10000;
    static constexpr std::size_t kSpan = std::size_t(1) << 20;
    static constexpr std::size_t kSmallMax = 32768;

    struct Hooks {
        std::function<bool(std::uintptr_t, std::size_t)> commit;    // [at, at + bytes) usable from now on
        std::function<void(std::uintptr_t, std::size_t)> decommit;  // [at, at + bytes) given back (reads zero later)
    };
    struct Stats {
        std::size_t committed = 0, spans = 0, large = 0, live = 0;
    };

    bool init(std::uintptr_t base, std::size_t bytes, Hooks hooks) {
        std::lock_guard lock(mutex_);
        if (!base || bytes < kSpan || (base % kPage)) return false;
        base_ = base;
        bytes_ = bytes - bytes % kPage;
        hooks_ = std::move(hooks);
        free_runs_.clear();
        free_runs_[base_] = bytes_;
        large_.clear();
        for (auto& cls : classes_) cls.free.clear();
        span_class_.assign(bytes_ / kSpan, kNoClass);
        span_live_.assign(bytes_ / kSpan, {});
        stats_ = {};
        return true;
    }
    bool owns(const void* p) const {
        const auto at = reinterpret_cast<std::uintptr_t>(p);
        return base_ && at >= base_ && at - base_ < bytes_;
    }

    void* allocate(std::size_t size) {
        if (!size) size = 1;
        std::lock_guard lock(mutex_);
        if (!base_) return nullptr;
        if (size <= kSmallMax) {
            const unsigned c = class_of(size);
            Class& cls = classes_[c];
            if (cls.free.empty() && !grow(c)) return nullptr;
            const std::uintptr_t at = cls.free.back();
            cls.free.pop_back();
            const std::size_t rel = at - base_;
            const std::size_t s = rel / kSpan;
            const std::size_t slot = (rel % kSpan) / class_size(c);
            span_live_[s][slot / 64] |= std::uint64_t(1) << (slot % 64);
            ++stats_.live;
            return reinterpret_cast<void*>(at);
        }
        const std::size_t bytes = (size + kPage - 1) & ~(kPage - 1);
        if (bytes < size) return nullptr;
        const std::uintptr_t at = take_run(bytes);
        if (!at) return nullptr;
        if (hooks_.commit && !hooks_.commit(at, bytes)) {
            give_run(at, bytes);
            return nullptr;
        }
        large_[at] = bytes;
        stats_.committed += bytes;
        ++stats_.large;
        ++stats_.live;
        return reinterpret_cast<void*>(at);
    }

    // The bytes the block at `p` holds (its class's size, or its pages): 0 for a pointer that is not a live block's.
    std::size_t size_of(const void* p) const {
        const auto at = reinterpret_cast<std::uintptr_t>(p);
        std::lock_guard lock(mutex_);
        if (!owns(p)) return 0;
        if (const auto it = large_.find(at); it != large_.end()) return it->second;
        const std::size_t rel = at - base_;
        const std::size_t s = rel / kSpan;
        if (s >= span_class_.size()) return 0;
        const unsigned c = span_class_[s];
        if (c == kNoClass) return 0;
        const std::size_t size = class_size(c);
        const std::size_t in_span = rel % kSpan;
        if (in_span % size != 0) return 0;
        const std::size_t slot = in_span / size;
        if (slot >= kSpan / size) return 0;
        return (span_live_[s][slot / 64] & (std::uint64_t(1) << (slot % 64))) ? size : 0;
    }

    // Frees a block allocate() returned. False (nothing done) for any other pointer or an already-freed block.
    bool release(void* p) {
        const auto at = reinterpret_cast<std::uintptr_t>(p);
        std::lock_guard lock(mutex_);
        if (!owns(p)) return false;
        if (const auto it = large_.find(at); it != large_.end()) {
            const std::size_t bytes = it->second;
            large_.erase(it);
            if (hooks_.decommit) hooks_.decommit(at, bytes);
            stats_.committed -= bytes;
            --stats_.large;
            --stats_.live;
            give_run(at, bytes);
            return true;
        }
        const std::size_t rel = at - base_;
        const std::size_t s = rel / kSpan;
        if (s >= span_class_.size()) return false;
        const unsigned c = span_class_[s];
        if (c == kNoClass) return false;
        const std::size_t size = class_size(c);
        const std::size_t in_span = rel % kSpan;
        if (in_span % size != 0) return false;
        const std::size_t slot = in_span / size;
        if (slot >= kSpan / size) return false;
        std::uint64_t& word = span_live_[s][slot / 64];
        const std::uint64_t bit = std::uint64_t(1) << (slot % 64);
        if ((word & bit) == 0) return false;
        word &= ~bit;
        classes_[c].free.push_back(at);
        --stats_.live;
        return true;
    }

    // f(at, bytes) for each committed range (a span or a large block), for a reader that starts late.
    template <class F>
    void for_each_committed(F&& f) const {
        std::lock_guard lock(mutex_);
        for (std::size_t s = 0; s < span_class_.size(); ++s)
            if (span_class_[s] != kNoClass) f(base_ + s * kSpan, kSpan);
        for (const auto& [at, bytes] : large_) f(at, bytes);
    }

    Stats stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }

    static constexpr unsigned kClasses = 16 + 5;  // 64..1024 by 64, then 2, 4, 8, 16, 32 KiB
    static std::size_t class_size(unsigned c) { return c < 16 ? (c + 1) * 64 : std::size_t(1024) << (c - 15); }
    static unsigned class_of(std::size_t size) {
        if (size <= 1024) return unsigned((size + 63) / 64 - 1);
        unsigned c = 16;
        while (class_size(c) < size) ++c;
        return c;
    }

private:
    static constexpr unsigned kNoClass = 0xff;
    struct Class {
        std::vector<std::uintptr_t> free;
    };

    // A new span for class `c`: committed, cut into blocks pushed on its stack (in address order: the lowest first).
    bool grow(unsigned c) {
        const std::uintptr_t at = take_run(kSpan, kSpan);
        if (!at) return false;
        if (hooks_.commit && !hooks_.commit(at, kSpan)) {
            give_run(at, kSpan);
            return false;
        }
        const std::size_t s = (at - base_) / kSpan;
        span_class_[s] = std::uint8_t(c);
        const std::size_t size = class_size(c);
        const std::size_t slots = kSpan / size;
        span_live_[s].assign((slots + 63) / 64, 0);
        stats_.committed += kSpan;
        ++stats_.spans;
        for (std::size_t off = slots * size; off >= size; off -= size) classes_[c].free.push_back(at + off - size);
        return true;
    }
    // The first free run that holds `bytes` at an offset from the base aligned to `align` (spans are numbered from the
    // base, which is only page-aligned).
    std::uintptr_t take_run(std::size_t bytes, std::size_t align = kPage) {
        for (auto it = free_runs_.begin(); it != free_runs_.end(); ++it) {
            const std::uintptr_t start = base_ + ((it->first - base_ + align - 1) & ~(std::uintptr_t(align) - 1));
            const std::uintptr_t end = it->first + it->second;
            if (start >= end || end - start < bytes) continue;
            const std::uintptr_t run = it->first, run_end = end;
            free_runs_.erase(it);
            if (start > run) free_runs_[run] = start - run;
            if (start + bytes < run_end) free_runs_[start + bytes] = run_end - (start + bytes);
            return start;
        }
        return 0;
    }
    void give_run(std::uintptr_t at, std::size_t bytes) {
        auto next = free_runs_.lower_bound(at);
        if (next != free_runs_.end() && at + bytes == next->first) {
            bytes += next->second;
            next = free_runs_.erase(next);
        }
        if (next != free_runs_.begin()) {
            const auto previous = std::prev(next);
            if (previous->first + previous->second == at) {
                previous->second += bytes;
                return;
            }
        }
        free_runs_[at] = bytes;
    }

    mutable std::mutex mutex_;
    std::uintptr_t base_ = 0;
    std::size_t bytes_ = 0;
    Hooks hooks_;
    std::map<std::uintptr_t, std::size_t> free_runs_;  // start -> bytes
    std::map<std::uintptr_t, std::size_t> large_;      // a large block -> its bytes
    std::vector<std::uint8_t> span_class_;             // per span: its class, or kNoClass
    std::vector<std::vector<std::uint64_t>> span_live_; // per active span: 1 bit per slot (1 = live, 0 = free)
    Class classes_[kClasses];
    Stats stats_;
};

}  // namespace leonida
