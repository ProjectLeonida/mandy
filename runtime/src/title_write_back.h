#pragma once
// GPU writes into title memory reached through a title window: the windows a shader may write through, the host copy
// each one lives in, and the buffer views a shader may write through, so the title's own pages can be brought up to
// date once the GPU has finished. Window copies live for the device's lifetime, so a range never names a freed copy.
// Adapted from DexNom/VIctory's copy-back work (commits cd94d35 and 940d32f, GPL-3.0-or-later), reworked for this
// layer: a view is registered when its UAV is made and marked when a descriptor table that holds it is scanned for a
// submission, instead of having the shaders mark pages.
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>

namespace leonida::title_write_back {

struct Window {
    std::uintptr_t title_va = 0;     // first byte of the title's range
    std::uint8_t* shadow = nullptr;  // the host-visible copy of that range
    std::size_t bytes = 0;
};

struct Range {
    std::uintptr_t title_va = 0;
    std::uint8_t* shadow = nullptr;
    std::size_t bytes = 0;
};

class Registry {
public:
    void add_window(std::uintptr_t title_va, std::uint8_t* shadow, std::size_t bytes) {
        if (!shadow || !bytes) return;
        const std::lock_guard lock(mutex_);
        windows_.push_back({title_va, shadow, bytes});
    }

    // A buffer view a shader may write through, in the title's address space. Adjacent and overlapping views merge.
    void add_view(std::uintptr_t title_va, std::size_t bytes) {
        if (!bytes || title_va > UINTPTR_MAX - bytes) return;
        std::uintptr_t lo = title_va;
        std::uintptr_t hi = title_va + bytes;
        const std::lock_guard lock(mutex_);
        auto it = views_.upper_bound(lo);
        if (it != views_.begin()) {
            const auto prev = std::prev(it);
            if (prev->first + prev->second >= lo) it = prev;  // the view just before touches or overlaps
        }
        while (it != views_.end() && it->first <= hi) {
            lo = lo < it->first ? lo : it->first;
            const std::uintptr_t end = it->first + it->second;
            hi = hi > end ? hi : end;
            it = views_.erase(it);
        }
        views_.emplace(lo, std::size_t(hi - lo));
    }

    // A V# whose base is `va` (a title VA, or an arena base this registry already aliased) holds a view this registry
    // knows: mark the view's window slices as written by the submission whose table is being scanned.
    bool mark_vsharp(std::uintptr_t va, std::size_t bytes) {
        if (!bytes || va > UINTPTR_MAX - bytes) return false;
        const std::lock_guard lock(mutex_);
        View view;
        std::size_t offset = 0;
        if (!find_view_locked(va, view, &offset)) return false;
        const std::uintptr_t start = view.title_va + offset;
        const std::size_t room = view.bytes - offset;
        const std::size_t length = bytes < room ? bytes : room;
        if (!length) return false;
        mark_locked(start, length);
        return true;
    }

    // A V# this registry knows was rebased from `base` to arena byte `to`. Later scans of the rebased words see `to`,
    // so remember which view they stand for.
    void note_rebase(std::uintptr_t base, std::uintptr_t to, std::size_t bytes) {
        if (!bytes || !to) return;
        const std::lock_guard lock(mutex_);
        View view;
        std::size_t offset = 0;
        if (!find_view_locked(base, view, &offset)) return;
        aliases_.insert_or_assign(to, Alias{view, offset});
    }

    // The end of a submission's scans: what they marked is copied at a later boundary, after its GPU work has
    // finished. Marks accumulate while the title windows are not refreshed.
    void boundary() {
        const std::lock_guard lock(mutex_);
        for (const Range& r : marked_) merge(pending_, r.title_va, r.shadow, r.bytes);
        marked_.clear();
    }

    // The ranges to copy back now, and forgets them.
    std::vector<Range> take_pending() {
        std::vector<Range> out;
        const std::lock_guard lock(mutex_);
        out.swap(pending_);
        return out;
    }

    bool has_pending() const {
        const std::lock_guard lock(mutex_);
        return !pending_.empty();
    }

    bool empty() const {
        const std::lock_guard lock(mutex_);
        return views_.empty();
    }

private:
    struct View {
        std::uintptr_t title_va = 0;
        std::size_t bytes = 0;
    };
    struct Alias {
        View view;
        std::size_t offset = 0;  // the V#'s base inside the view
    };

    bool find_view_locked(std::uintptr_t va, View& out, std::size_t* offset) const {
        const auto alias = aliases_.find(va);
        if (alias != aliases_.end()) {
            out = alias->second.view;
            if (offset) *offset = alias->second.offset;
            return true;
        }
        const auto it = views_.upper_bound(va);
        if (it == views_.begin()) return false;
        const auto prev = std::prev(it);  // disjoint views: only the one before can hold `va`
        if (prev->first + prev->second <= va) return false;
        out = {prev->first, prev->second};
        if (offset) *offset = std::size_t(va - prev->first);
        return true;
    }

    void mark_locked(std::uintptr_t title_va, std::size_t bytes) {
        const std::uintptr_t end = title_va + bytes;
        for (const Window& w : windows_) {
            const std::uintptr_t from = title_va > w.title_va ? title_va : w.title_va;
            const std::uintptr_t to = end < w.title_va + w.bytes ? end : w.title_va + w.bytes;
            if (from >= to) continue;
            merge(marked_, from, w.shadow + (from - w.title_va), std::size_t(to - from));
        }
    }

    static void merge(std::vector<Range>& out, std::uintptr_t va, std::uint8_t* shadow, std::size_t bytes) {
        for (Range& r : out) {
            if (r.title_va + r.bytes == va && r.shadow + r.bytes == shadow) {
                r.bytes += bytes;
                return;
            }
            if (va + bytes == r.title_va && shadow + bytes == r.shadow) {
                r.title_va = va;
                r.shadow = shadow;
                r.bytes += bytes;
                return;
            }
        }
        out.push_back({va, shadow, bytes});
    }

    mutable std::mutex mutex_;
    std::vector<Window> windows_;
    std::map<std::uintptr_t, std::size_t> views_;  // disjoint title VA ranges a buffer UAV was made for
    std::map<std::uintptr_t, Alias> aliases_;  // rebased V# base (arena byte) -> the view it stands for
    std::vector<Range> marked_;               // marked by the current submission's table scans
    std::vector<Range> pending_;              // marked, waiting for a refresh boundary to copy them
};

}  // namespace leonida::title_write_back
