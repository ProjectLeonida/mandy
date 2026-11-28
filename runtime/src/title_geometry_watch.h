#pragma once

#include "title_page_watch.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace leonida::geometry_writes {

constexpr std::uintptr_t tile_size = 65536, page_size = 4096;
using Aliases = void (*)(std::uintptr_t, std::vector<std::uintptr_t>&);
inline std::atomic<Aliases> aliases{};

struct Cell {
    static constexpr std::uint64_t kNoMapGen = ~std::uint64_t(0);
    std::atomic<std::uint64_t> version{1}, mapping{1};
    std::array<DWORD, tile_size / page_size> restore{};
    // title_memory_map::generation() observed when `restore` was uniformly armed across all 16 sub-pages (under
    // State::mutex). When title_memory_map::generation() still equals armed_map_gen and `restore` is uniform, reopen()
    // skips VirtualQuery and calls VirtualProtect directly (falling back on unexpected `old`).
    std::uint64_t armed_map_gen = kNoMapGen;
    unsigned refs = 0;
    // Owners that leave the tile writable and copy it at every submission instead (the page mirror's hot pages,
    // src/title_write_policy.h). While it is nonzero the cell guards nothing (arm() does not protect it) and its tile
    // counts as written at every look: a subscriber asks for the hot tiles (hot_tiles()) as it takes its queue. One
    // guard per page, owned here, whatever the number of copies that read it: an owner that stopped guarding a page
    // without saying so left every other copy of it blind (run 64: video planes uploaded until the mirror found their
    // pages hot, then never again).
    unsigned hot = 0;
};
struct ReleaseQueue {
    std::mutex mutex;
    host_heap::vector<std::pair<std::uintptr_t, std::uintptr_t>> ranges;
    std::atomic<bool> pending{false};
};
// A subscriber's list of the tiles notify() changed since it last took them (the title texture refresh's: it
// examines only textures over these tiles). Past kMaxTiles it keeps only `overflow`: every tile counts as changed.
struct WriteQueue {
    static constexpr std::size_t kMaxTiles = 65536;
    std::mutex mutex;  // a leaf lock, taken under State::mutex (in notify)
    host_heap::vector<std::uintptr_t> tiles;
    bool overflow = false;
    std::atomic<bool> pending{false};
};
struct State {
    std::mutex mutex;
    host_heap::map<std::uintptr_t, Cell> cells;
    host_heap::vector<std::uintptr_t> hot;  // the tiles of the cells with hot > 0, under mutex
    std::atomic<std::size_t> hot_count{0};
    std::atomic<std::uint64_t> hot_generation{0};
    host_heap::vector<WriteQueue*> writers;  // under mutex
    std::atomic<std::uint64_t> serial{1};
    std::atomic<std::size_t> count{0};
    // Union of every tile passed to acquire(): [watched_lo, watched_hi). Monotonic; updated under mutex.
    std::atomic<std::uintptr_t> watched_lo{UINTPTR_MAX};
    std::atomic<std::uintptr_t> watched_hi{0};
    std::mutex subscribers_mutex;
    host_heap::vector<ReleaseQueue*> subscribers;
};
inline State& state() { static State s; return s; }

inline void subscribe(ReleaseQueue& queue) {
    auto& s = state();
    std::lock_guard lock(s.subscribers_mutex);
    if (std::find(s.subscribers.begin(), s.subscribers.end(), &queue) == s.subscribers.end())
        s.subscribers.push_back(&queue);
}

inline void unsubscribe(ReleaseQueue& queue) {
    auto& s = state();
    std::lock_guard lock(s.subscribers_mutex);
    std::erase(s.subscribers, &queue);
}

inline void subscribe_writes(WriteQueue& queue) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    if (std::find(s.writers.begin(), s.writers.end(), &queue) == s.writers.end()) s.writers.push_back(&queue);
}

inline void unsubscribe_writes(WriteQueue& queue) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    std::erase(s.writers, &queue);
}

// The tiles changed since the last call (`overflow`: some were not kept, so every tile counts as changed).
template<class Out>
inline void take_written(WriteQueue& queue, Out& out, bool& overflow) {
    out.clear();
    overflow = false;
    if (!queue.pending.load(std::memory_order_acquire)) return;
    std::lock_guard lock(queue.mutex);
    out.assign(queue.tiles.begin(), queue.tiles.end());
    queue.tiles.clear();
    overflow = std::exchange(queue.overflow, false);
    queue.pending.store(false, std::memory_order_release);
}

inline bool has_released(const ReleaseQueue& queue) {
    return queue.pending.load(std::memory_order_acquire);
}

// Records a title address range [begin, end) that was released or unmapped so every subscribed GeometryViews can
// evict its views. Uses host_heap and separate leaf locks so xmem does not contend with the write-fault handler.
// Lock order: subscribers_mutex -> queue.mutex; take_released takes only queue.mutex.
inline void note_released(std::uintptr_t begin, std::uintptr_t end) {
    auto& s = state();
    if (end <= begin || !s.count.load(std::memory_order_acquire)) return;
    const std::uintptr_t lo = s.watched_lo.load(std::memory_order_acquire);
    const std::uintptr_t hi = s.watched_hi.load(std::memory_order_acquire);
    if (end <= lo || begin >= hi) {
        // Views are keyed by the CPU view's address, so a GPU-only range freed by its own address counts through its
        // CPU view (GeometryViews::drain_released maps it there).
        leonida::title_gpu_views::Range pair{};
        if (!leonida::title_gpu_views::view_at(begin, 1, pair) || begin < pair.gpu || begin - pair.gpu >= pair.size)
            return;
        const std::uintptr_t cpu = pair.cpu + (begin - pair.gpu);
        const std::uintptr_t span = (std::min)(end - begin, pair.size - (begin - pair.gpu));
        if (cpu + span <= lo || cpu >= hi) return;
    }
    std::lock_guard subscribers_lock(s.subscribers_mutex);
    for (ReleaseQueue* q : s.subscribers) {
        std::lock_guard lock(q->mutex);
        bool room = q->ranges.size() < 65536;
        if (!room) {
            // Full: first the ranges that overlap or touch become one (unmaps report page runs, often neighbours), and
            // only when that leaves it more than half full does it all become one covering range.
            std::sort(q->ranges.begin(), q->ranges.end());
            std::size_t kept = 0;
            for (std::size_t i = 1; i < q->ranges.size(); ++i) {
                if (q->ranges[i].first <= q->ranges[kept].second)
                    q->ranges[kept].second = (std::max)(q->ranges[kept].second, q->ranges[i].second);
                else
                    q->ranges[++kept] = q->ranges[i];
            }
            q->ranges.resize(kept + 1);
            room = q->ranges.size() <= 32768;
        }
        if (room) {
            q->ranges.emplace_back(begin, end);
        } else {
            std::uintptr_t merged_lo = begin;
            std::uintptr_t merged_hi = end;
            for (const auto& [r_lo, r_hi] : q->ranges) {
                if (r_lo < merged_lo) merged_lo = r_lo;
                if (r_hi > merged_hi) merged_hi = r_hi;
            }
            q->ranges.clear();
            q->ranges.emplace_back(merged_lo, merged_hi);
        }
        q->pending.store(true, std::memory_order_release);
    }
}

template<class Out>
inline void take_released(ReleaseQueue& queue, Out& out) {
    out.clear();
    if (!queue.pending.load(std::memory_order_acquire)) return;
    std::lock_guard lock(queue.mutex);
    out.assign(queue.ranges.begin(), queue.ranges.end());
    queue.ranges.clear();
    queue.pending.store(false, std::memory_order_release);
}

inline Cell* acquire(std::uintptr_t va) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    auto& c = s.cells[va];
    ++c.refs;
    const std::uintptr_t tile_end = va <= UINTPTR_MAX - tile_size ? va + tile_size : UINTPTR_MAX;
    if (va < s.watched_lo.load(std::memory_order_relaxed))
        s.watched_lo.store(va, std::memory_order_release);
    if (tile_end > s.watched_hi.load(std::memory_order_relaxed))
        s.watched_hi.store(tile_end, std::memory_order_release);
    s.count.store(s.cells.size(), std::memory_order_release);
    return &c;
}

// Acquires one reference per tile in `vas[0..count)` under a single State::mutex hold.
inline void acquire_batch(const std::uintptr_t* vas, Cell** out_cells, std::size_t count) {
    if (!count) return;
    auto& s = state();
    std::lock_guard lock(s.mutex);
    std::uintptr_t lo = s.watched_lo.load(std::memory_order_relaxed);
    std::uintptr_t hi = s.watched_hi.load(std::memory_order_relaxed);
    bool lo_changed = false, hi_changed = false;
    for (std::size_t i = 0; i < count; ++i) {
        const std::uintptr_t va = vas[i];
        auto& c = s.cells[va];
        ++c.refs;
        out_cells[i] = &c;
        const std::uintptr_t tile_end = va <= UINTPTR_MAX - tile_size ? va + tile_size : UINTPTR_MAX;
        if (va < lo) {
            lo = va;
            lo_changed = true;
        }
        if (tile_end > hi) {
            hi = tile_end;
            hi_changed = true;
        }
    }
    if (lo_changed) s.watched_lo.store(lo, std::memory_order_release);
    if (hi_changed) s.watched_hi.store(hi, std::memory_order_release);
    s.count.store(s.cells.size(), std::memory_order_release);
}

// Restore only protections this watch installed. A guest remap or explicit new protection supersedes them.
inline DWORD read_only(DWORD protection) {
    switch (protection & 255u) {
    case PAGE_READWRITE: case PAGE_WRITECOPY: return (protection & ~255u) | PAGE_READONLY;
    case PAGE_EXECUTE_READWRITE: case PAGE_EXECUTE_WRITECOPY: return (protection & ~255u) | PAGE_EXECUTE_READ;
    default: return 0;
    }
}
// True when it made a page writable again.
// Pages that are to get the same protection back are reopened together, a VirtualQuery and a VirtualProtect for each
// region the query reports, not for each page: under Wine each call goes to the server, and a fault on a whole guarded
// tile cost 16 us page by page against 2.6 us for one call (runtime/test/guard_fault_bench.cpp).
// Fast path: when `c.restore` is uniform across all 16 sub-pages and `title_memory_map::generation() == c.armed_map_gen`,
// skip VirtualQuery and call VirtualProtect directly (verifying `old == read_only(restore) || old == restore`, else
// restoring `old` and falling back to the slow path).
// When `fault_page_opened` is non-null, sets it to true if VirtualProtect restored a writable, non-guard protection on
// the 4 KiB page `fault_page`, allowing notify(Event::fault) to skip a redundant second VirtualQuery.
inline bool reopen(std::uintptr_t va, Cell& c, std::uintptr_t fault_page = UINTPTR_MAX,
                   bool* fault_page_opened = nullptr) {
    const DWORD r0 = c.restore[0];
    if (r0 && c.armed_map_gen != Cell::kNoMapGen && title_memory_map::generation() == c.armed_map_gen) {
        bool uniform = true;
        for (std::size_t i = 1; i < c.restore.size(); ++i) {
            if (c.restore[i] != r0) {
                uniform = false;
                break;
            }
        }
        if (uniform) {
            const DWORD ro = read_only(r0);
            const bool writable_restore = !(r0 & PAGE_GUARD) && ro != 0;
            if (ro) {
                DWORD old{};
                if (VirtualProtect(reinterpret_cast<void*>(va), tile_size, r0, &old)) {
                    if (old == ro || old == r0) {
                        c.restore.fill(0);
                        c.armed_map_gen = Cell::kNoMapGen;
                        if (fault_page_opened && writable_restore && fault_page >= va && fault_page < va + tile_size)
                            *fault_page_opened = true;
                        return true;
                    }
                    // Unexpected prior protection (caller bypassed title_memory_map::changed): restore `old` and fall
                    // back to the VirtualQuery slow path.
                    DWORD dummy{};
                    VirtualProtect(reinterpret_cast<void*>(va), tile_size, old, &dummy);
                }
            }
        }
    }
    c.armed_map_gen = Cell::kNoMapGen;
    bool opened = false;
    for (std::size_t i = 0; i < c.restore.size();) {
        if (!c.restore[i]) {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < c.restore.size() && c.restore[j] == c.restore[i]) ++j;
        const DWORD restore = c.restore[i];
        const bool writable_restore = !(restore & PAGE_GUARD) && read_only(restore) != 0;
        for (std::uintptr_t at = va + i * page_size, end = va + j * page_size; at < end;) {
            MEMORY_BASIC_INFORMATION m{};
            if (!VirtualQuery(reinterpret_cast<void*>(at), &m, sizeof m)) {
                at += page_size;
                continue;
            }
            const std::uintptr_t next = (std::min)(end, reinterpret_cast<std::uintptr_t>(m.BaseAddress) + m.RegionSize);
            const std::uintptr_t until = next > at ? next : at + page_size;
            bool done = true;
            if (m.State == MEM_COMMIT && m.Protect == read_only(restore)) {
                DWORD old{};
                if (VirtualProtect(reinterpret_cast<void*>(at), until - at, restore, &old)) {
                    opened = true;
                    if (fault_page_opened && writable_restore && fault_page >= at && fault_page < until)
                        *fault_page_opened = true;
                } else {
                    done = false;  // kept: tried again by the next reopen
                }
            }
            if (done)
                for (std::uintptr_t p = at; p < until; p += page_size) c.restore[(p - va) / page_size] = 0;
            at = until;
        }
        i = j;
    }
    return opened;
}
// Releases one owner's reference to `va`. When `was_hot` is true, drops that owner's hot count in the same lock hold
// (avoiding a separate set_hot(false) lock and avoiding a bogus WriteQueue push when the cell's last ref goes).
inline void release(std::uintptr_t va, bool was_hot = false) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    auto it = s.cells.find(va);
    if (it == s.cells.end()) return;
    Cell& c = it->second;
    if (!--c.refs) {
        if (c.hot) {
            std::erase(s.hot, va);
            s.hot_count.store(s.hot.size(), std::memory_order_release);
            s.hot_generation.fetch_add(1, std::memory_order_release);
        }
        // A page reopened here may also be one a slot-133 shadow keeps read-only (both guard the title's geometry):
        // it is writable now with no fault to tell that shadow, which then walks its pages again.
        if (reopen(va, c)) title_memory_map::changed(va, va + tile_size);
        s.cells.erase(it);
        s.count.store(s.cells.size(), std::memory_order_release);
        return;
    }
    if (was_hot && c.hot && !--c.hot) {
        std::erase(s.hot, va);
        // Other owners still hold this cell: writes made while it was hot faulted nowhere, so notify subscribers once
        // as set_hot(false) does.
        const auto version = s.serial.load(std::memory_order_relaxed) + 1;
        c.version.store(version, std::memory_order_release);
        for (WriteQueue* q : s.writers) {
            std::lock_guard queue_lock(q->mutex);
            if (q->overflow) continue;
            if (q->tiles.size() < WriteQueue::kMaxTiles)
                q->tiles.push_back(va);
            else
                q->overflow = true, q->tiles.clear();
            q->pending.store(true, std::memory_order_release);
        }
        s.serial.store(version, std::memory_order_release);
        s.hot_count.store(s.hot.size(), std::memory_order_release);
        s.hot_generation.fetch_add(1, std::memory_order_release);
    }
}

// Releases one reference per {va, was_hot} item under a single State::mutex hold.
inline void release_batch(const std::pair<std::uintptr_t, bool>* items, std::size_t count) {
    if (!count) return;
    auto& s = state();
    std::lock_guard lock(s.mutex);
    bool hot_changed = false;
    bool cells_erased = false;
    host_heap::vector<std::uintptr_t> cooled_shared;
    std::uintptr_t reopened_lo = 0, reopened_hi = 0;
    const auto flush_reopened = [&] {
        if (reopened_lo < reopened_hi) {
            title_memory_map::changed(reopened_lo, reopened_hi);
            reopened_lo = reopened_hi = 0;
        }
    };
    for (std::size_t i = 0; i < count; ++i) {
        const std::uintptr_t va = items[i].first;
        const bool was_hot = items[i].second;
        auto it = s.cells.find(va);
        if (it == s.cells.end()) continue;
        Cell& c = it->second;
        if (!--c.refs) {
            if (c.hot) {
                std::erase(s.hot, va);
                hot_changed = true;
            }
            if (reopen(va, c)) {
                const std::uintptr_t tile_end = va <= UINTPTR_MAX - tile_size ? va + tile_size : UINTPTR_MAX;
                if (reopened_lo < reopened_hi && va == reopened_hi) {
                    reopened_hi = tile_end;
                } else {
                    flush_reopened();
                    reopened_lo = va;
                    reopened_hi = tile_end;
                }
            }
            s.cells.erase(it);
            cells_erased = true;
            continue;
        }
        if (was_hot && c.hot && !--c.hot) {
            std::erase(s.hot, va);
            hot_changed = true;
            cooled_shared.push_back(va);
        }
    }
    flush_reopened();
    if (!cooled_shared.empty()) {
        const auto version = s.serial.load(std::memory_order_relaxed) + 1;
        for (const std::uintptr_t va : cooled_shared) {
            auto it = s.cells.find(va);
            if (it != s.cells.end()) it->second.version.store(version, std::memory_order_release);
        }
        for (WriteQueue* q : s.writers) {
            std::lock_guard queue_lock(q->mutex);
            if (q->overflow) continue;
            if (q->tiles.size() + cooled_shared.size() <= WriteQueue::kMaxTiles) {
                q->tiles.insert(q->tiles.end(), cooled_shared.begin(), cooled_shared.end());
            } else {
                q->overflow = true;
                q->tiles.clear();
            }
            q->pending.store(true, std::memory_order_release);
        }
        s.serial.store(version, std::memory_order_release);
    }
    if (cells_erased) s.count.store(s.cells.size(), std::memory_order_release);
    if (hot_changed) {
        s.hot_count.store(s.hot.size(), std::memory_order_release);
        s.hot_generation.fetch_add(1, std::memory_order_release);
    }
}
enum class Event { write, open, fault, mapping, replaced };
// `restored` (open and fault): set when a tile's pages were made writable again. A tile reopens all of its pages, and
// a page of it a slot-133 shadow also keeps read-only (both can guard one page: each queries, then protects, under its
// own lock) is writable then without that shadow knowing; the caller has the shadows reopen the whole tiles too.
inline bool notify(std::uintptr_t begin, std::uintptr_t end, Event event, bool* restored = nullptr) {
    auto& s = state();
    if (!s.count.load(std::memory_order_acquire) || end <= begin) return false;
    std::lock_guard lock(s.mutex);
    auto it = s.cells.lower_bound(begin & ~(tile_size - 1));
    if (it == s.cells.end() || it->first >= end) return false;
    const auto version = s.serial.load(std::memory_order_relaxed) + 1;
    const std::uintptr_t fault_page = begin & ~(page_size - 1);
    bool opened = false;
    for (; it != s.cells.end() && it->first < end; ++it) {
        auto& cell = it->second;
        if (event == Event::open || event == Event::fault) {
            bool fault_opened = false;
            const bool any = reopen(it->first, cell, event == Event::fault ? fault_page : UINTPTR_MAX,
                                    event == Event::fault ? &fault_opened : nullptr);
            if (any && restored) *restored = true;
            if (event == Event::fault) {
                if (fault_opened) {
                    opened = true;
                } else if (!opened) {
                    MEMORY_BASIC_INFORMATION m{};
                    opened = VirtualQuery(reinterpret_cast<void*>(begin), &m, sizeof m) &&
                             m.State == MEM_COMMIT && !(m.Protect & PAGE_GUARD) && read_only(m.Protect);
                }
            }
        }
        if (event == Event::replaced) {
            cell.armed_map_gen = Cell::kNoMapGen;
            for (unsigned i = 0; i < cell.restore.size(); ++i) {
                auto p = it->first + i * page_size;
                if (p < end && p + page_size > begin) cell.restore[i] = 0;
            }
        }
        if (event == Event::mapping || event == Event::replaced) cell.mapping.store(version, std::memory_order_release);
        cell.version.store(version, std::memory_order_release);
        for (WriteQueue* q : s.writers) {
            std::lock_guard queue_lock(q->mutex);
            if (q->overflow) continue;
            if (q->tiles.size() < WriteQueue::kMaxTiles)
                q->tiles.push_back(it->first);
            else
                q->overflow = true, q->tiles.clear();
            q->pending.store(true, std::memory_order_release);
        }
    }
    s.serial.store(version, std::memory_order_release);
    return opened;
}

// The tile at `va` is hot for one more owner (`hot`) or one fewer: while any owner holds it hot, its guard is off and
// it is not armed again, and hot_tiles() lists it for every subscriber. The cell must be acquired.
inline void set_hot(std::uintptr_t va, bool hot) {
    auto& s = state();
    std::lock_guard lock(s.mutex);
    const auto it = s.cells.find(va);
    if (it == s.cells.end()) return;
    Cell& c = it->second;
    if (hot) {
        if (c.hot++) return;
        reopen(va, c);
        s.hot.push_back(va);
    } else {
        if (!c.hot || --c.hot) return;
        std::erase(s.hot, va);
        // Writes made while it was hot faulted nowhere and the subscribers that looked before this saw it in
        // hot_tiles(): it counts as written once more, for all of them, as a write does.
        const auto version = s.serial.load(std::memory_order_relaxed) + 1;
        c.version.store(version, std::memory_order_release);
        for (WriteQueue* q : s.writers) {
            std::lock_guard queue_lock(q->mutex);
            if (q->overflow) continue;
            if (q->tiles.size() < WriteQueue::kMaxTiles)
                q->tiles.push_back(va);
            else
                q->overflow = true, q->tiles.clear();
            q->pending.store(true, std::memory_order_release);
        }
        s.serial.store(version, std::memory_order_release);
    }
    s.hot_count.store(s.hot.size(), std::memory_order_release);
    s.hot_generation.fetch_add(1, std::memory_order_release);
}

// The tiles some owner holds hot: written, for every subscriber, at every look.
template<class Out>
inline void hot_tiles(Out& out) {
    out.clear();
    auto& s = state();
    if (!s.hot_count.load(std::memory_order_acquire)) return;
    std::lock_guard lock(s.mutex);
    out.assign(s.hot.begin(), s.hot.end());
}

// Arm under the same lock as the write-fault path. A kernel destination may only be guarded again after its pin ends.
// A hot cell is not protected (true: its writes reach every subscriber through hot_tiles()).
inline bool arm(std::uintptr_t va, Cell& c) {
    auto& s = state();
    // Querying an OVERLAPPED can fault into xmem. Never do it with the watch lock held.
    const auto version = c.version.load(std::memory_order_acquire);
    bool pinned = false;
    kernel_writes::for_each_pin(va, va + tile_size, [&](auto, auto) { pinned = true; });
    std::lock_guard lock(s.mutex);
    if (c.hot) return true;
    if (pinned || version != c.version.load(std::memory_order_relaxed)) return false;
    c.armed_map_gen = Cell::kNoMapGen;
    for (std::size_t i = 0; i < c.restore.size();) {
        auto p = va + i * page_size;
        const std::uint64_t query_gen = title_memory_map::generation();
        MEMORY_BASIC_INFORMATION m{};
        if (!VirtualQuery(reinterpret_cast<void*>(p), &m, sizeof m)) return false;
        const auto end = (std::min)(va + tile_size, reinterpret_cast<std::uintptr_t>(m.BaseAddress) + m.RegionSize);
        if (end <= p) return false;
        const auto next = (end - va) / page_size;
        const DWORD guarded = read_only(m.Protect);
        if (m.State == MEM_COMMIT && !(m.Protect & PAGE_GUARD) && guarded) {
            DWORD old{};
            if (!VirtualProtect(reinterpret_cast<void*>(p), end - p, guarded, &old)) return false;
            // What the query saw, not `old`: VirtualProtect returns the first page's protection only, and a slot-133
            // shadow (which guards these pages under its own lock) may have made that page read-only in between. A
            // read-only `old` then marked every page of the run as read-only already, so reopen() never gave them back
            // and they stayed read-only with no guard to answer for them: title writes and kernel writes into them
            // failed for good. Such a page has two guards now; reopening it tells the shadows (notify's `restored`).
            for (auto j = i; j < next; ++j) c.restore[j] = m.Protect;
            if (i == 0 && next == c.restore.size() && title_memory_map::generation() == query_gen)
                c.armed_map_gen = query_gen;
        }
        i = next;
    }
    return true;
}
} // namespace leonida::geometry_writes
