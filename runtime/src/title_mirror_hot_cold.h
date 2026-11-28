#pragma once
// HotColdTracking: the page mirror's tracking (TitleMirrorTracking, src/title_mirror_tracking.h) with the hot/cold
// policy of src/title_write_policy.h over the geometry watch's cells, plus region-batched zero-fault MEM_WRITE_WATCH
// tracking (title_write_watch) for pages inside write-watched private reservations.
//
// - Guarded pages (sections / ordinary pages without MEM_WRITE_WATCH): watched through geometry_writes cells, with
//   contiguous 64 KiB re-arms batched into single VirtualProtect syscalls in CellGuard::arm_batch(). One collect() is
//   one submission interval; only pages written in >= 6 of 8 submissions stay unguarded.
// - Write-watched pages (title_write_watch::contains): queried via cached region-batched runs in
//   title_write_watch::ask() (GetWriteWatch with WRITE_WATCH_FLAG_RESET) in O(regions + dirty) without walking
//   ww_pages_ on steady-state submissions, with zero guard faults, zero VirtualProtect syscalls, and zero interference
//   with geometry_writes::hot_tiles(). Uncommitted holes inside a reservation are never bridged.
#include "submit_phases.h"
#include "title_mirror_tracking.h"
#include "title_write_policy.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace leonida {

class HotColdTracking final : public TitleMirrorTracking {
public:
    struct ArmStats {
        std::uint64_t ns = 0;
        std::uint64_t vp_calls = 0;
        std::uint64_t vq_calls = 0;
        std::uint64_t fast_tiles = 0;
        std::uint64_t slow_tiles = 0;
        std::uint64_t skip_tiles = 0;
        std::uint64_t already_armed_skips = 0;
        std::uint64_t table_skips = 0;
        std::uint64_t bridged_runs = 0;
        std::uint64_t wc_tiles = 0;
    };

    HotColdTracking() : policy_(guard_) {
        geometry_writes::subscribe_writes(queue_);
    }
    ~HotColdTracking() override {
        geometry_writes::unsubscribe_writes(queue_);
        if (ww_reader_) title_write_watch::close_reader(ww_reader_);
        for (const auto& [page, tc] : guard_.cells) {
            geometry_writes::release(std::uintptr_t(page), tc.is_hot);
        }
    }
    HotColdTracking(const HotColdTracking&) = delete;
    HotColdTracking& operator=(const HotColdTracking&) = delete;

    void watch(std::uint64_t page) override {
        const std::uintptr_t addr = std::uintptr_t(page) & ~(Policy::kPage - 1);
        if (guard_.cells.count(addr) || ww_pages_.count(addr)) return;
        if (title_write_watch::enabled() && title_write_watch::contains(addr)) {
            if (!ww_reader_) ww_reader_ = title_write_watch::open_reader();
            ww_pages_.insert(addr);
            ww_initial_.push_back(addr);
            ww_runs_dirty_ = true;
            sync_pending_atomics();
            return;
        }
        watch_guarded(addr);
        sync_pending_atomics();
    }
    void unwatch(std::uint64_t page) override {
        const std::uintptr_t addr = std::uintptr_t(page) & ~(Policy::kPage - 1);
        if (ww_pages_.erase(addr)) {
            ww_runs_dirty_ = true;
            if (ww_pages_.empty()) {
                ww_runs_.clear();
                ww_initial_.clear();
            }
            sync_pending_atomics();
            return;
        }
        const auto it = guard_.cells.find(addr);
        if (it == guard_.cells.end()) return;
        const bool was_hot = it->second.is_hot;
        // Erase from guard_.cells BEFORE policy_.remove() so remove()'s guard_.disarm(addr) is a 0-lock no-op
        // instead of toggling set_hot(true) -> set_hot(false) (2 extra locks + 1 extra reopen + bogus WriteQueue push).
        // geometry_writes::release(addr, was_hot) drops our ref (and hot count if was_hot) in a single lock hold.
        guard_.cells.erase(it);
        guard_.note_wc_page(addr, false);
        policy_.remove(addr, addr + Policy::kPage);
        geometry_writes::release(addr, was_hot);
        sync_pending_atomics();
    }

    // Batched watch: acquires all new guarded cells under one geometry_writes::state().mutex hold + one atomic sync.
    void watch_batch(const std::uint64_t* pages, std::size_t count) {
        if (!count) return;
        if (count == 1) {
            watch(pages[0]);
            return;
        }
        const bool ww_on = title_write_watch::enabled();
        batch_watch_addrs_.clear();
        batch_watch_tcs_.clear();
        for (std::size_t i = 0; i < count; ++i) {
            const std::uintptr_t addr = std::uintptr_t(pages[i]) & ~(Policy::kPage - 1);
            if (guard_.cells.count(addr) || ww_pages_.count(addr)) continue;
            if (ww_on && title_write_watch::contains(addr)) {
                if (!ww_reader_) ww_reader_ = title_write_watch::open_reader();
                ww_pages_.insert(addr);
                ww_initial_.push_back(addr);
                ww_runs_dirty_ = true;
                continue;
            }
            auto [it, fresh] = guard_.cells.try_emplace(addr);
            if (fresh) {
                batch_watch_addrs_.push_back(addr);
                batch_watch_tcs_.push_back(&it->second);
            }
        }
        if (!batch_watch_addrs_.empty()) {
            batch_watch_cells_.resize(batch_watch_addrs_.size());
            geometry_writes::acquire_batch(batch_watch_addrs_.data(), batch_watch_cells_.data(),
                                           batch_watch_addrs_.size());
            for (std::size_t k = 0; k < batch_watch_addrs_.size(); ++k) {
                batch_watch_tcs_[k]->cell = batch_watch_cells_[k];
                policy_.add(batch_watch_addrs_[k], batch_watch_addrs_[k] + Policy::kPage);
            }
        }
        sync_pending_atomics();
    }

    // Batched unwatch: releases all guarded cells under one geometry_writes::state().mutex hold + one atomic sync.
    void unwatch_batch(const std::uint64_t* pages, std::size_t count) {
        if (!count) return;
        if (count == 1) {
            unwatch(pages[0]);
            return;
        }
        batch_unwatch_items_.clear();
        bool erased_ww = false;
        for (std::size_t i = 0; i < count; ++i) {
            const std::uintptr_t addr = std::uintptr_t(pages[i]) & ~(Policy::kPage - 1);
            if (ww_pages_.erase(addr)) {
                ww_runs_dirty_ = true;
                erased_ww = true;
                continue;
            }
            const auto it = guard_.cells.find(addr);
            if (it == guard_.cells.end()) continue;
            const bool was_hot = it->second.is_hot;
            guard_.cells.erase(it);
            guard_.note_wc_page(addr, false);
            policy_.remove(addr, addr + Policy::kPage);
            batch_unwatch_items_.emplace_back(addr, was_hot);
        }
        if (erased_ww && ww_pages_.empty()) {
            ww_runs_.clear();
            ww_initial_.clear();
        }
        if (!batch_unwatch_items_.empty()) {
            geometry_writes::release_batch(batch_unwatch_items_.data(), batch_unwatch_items_.size());
        }
        sync_pending_atomics();
    }

    void collect(std::vector<std::uint64_t>& out) override {
        out.clear();
        guard_.arm_stats = {};
        bool overflow = false;
        geometry_writes::take_written(queue_, tiles_, overflow);
        if (overflow)
            for (const auto& [page, tc] : guard_.cells) policy_.written(std::uintptr_t(page));
        else
            for (const std::uintptr_t tile : tiles_) {
                const auto it = guard_.cells.find(tile);
                if (it == guard_.cells.end()) continue;
                const auto& tc = it->second;
                // A guarded page whose cell has not moved since it was armed was not written since: the entry is
                // older than the arming, or this tracking's own turn of the page from hot to cold.
                if (tc.is_armed && tc.armed_version == tc.cell->version.load(std::memory_order_acquire))
                    continue;
                policy_.written(tile);
            }
        policy_.collect([&](std::uintptr_t page) { out.push_back(page); });
        policy_.frame();  // one submission interval a collect

        if (!ww_pages_.empty()) collect_write_watched(out);
        sync_pending_atomics();
    }

    // Lock-free O(1) clean-submission check: false when no write faults are queued, no guarded pages are hot, dirty,
    // or decaying in history_active_, and no write-watched pages are active.
    bool has_pending_work() const noexcept {
        if (queue_.pending.load(std::memory_order_acquire)) return true;
        if (active_work_.load(std::memory_order_acquire) != 0) return true;
        return ww_count_.load(std::memory_order_acquire) != 0;
    }

    std::uint64_t take_refused() override {
        const std::uint64_t now = policy_.stats().arm_failures;
        return now - std::exchange(refused_seen_, now);
    }
    const auto& stats() const noexcept { return policy_.stats(); }
    const ArmStats& last_arm_stats() const noexcept { return guard_.arm_stats; }
    // Read-only during TitleMirror::upload() worker threads (mutated only under page_mirror.mutex in arm_batch/unwatch).
    bool is_wc_page(std::uintptr_t addr) const noexcept {
        return guard_.is_wc_page(addr);
    }
    bool hot(std::uint64_t page) const { return policy_.hot(std::uintptr_t(page)); }
    // A cold page whose guard no write has reached since it was armed (TitleMirrorTracking::unwritten). A hot or
    // write-watched page returns false (conservative).
    bool unwritten(std::uint64_t page) override {
        const std::uintptr_t addr = std::uintptr_t(page) & ~(Policy::kPage - 1);
        if (ww_pages_.count(addr) || policy_.hot(addr)) return false;
        const auto it = guard_.cells.find(addr);
        return it != guard_.cells.end() && it->second.is_armed &&
               it->second.cell->version.load(std::memory_order_acquire) == it->second.armed_version;
    }

private:
    static constexpr std::uintptr_t kWwMergeGap = 0x100000;      // bridge committed gaps <= 1 MiB inside a reservation
    static constexpr std::uintptr_t kMaxWwRunBytes = 0x4000000;  // cap each GetWriteWatch call at 64 MiB (16k 4K pages)

    void watch_guarded(std::uintptr_t addr) {
        if (guard_.cells.count(addr)) return;
        CellGuard::TrackedCell tc{};
        tc.cell = geometry_writes::acquire(addr);
        guard_.cells.emplace(addr, tc);
        policy_.add(addr, addr + Policy::kPage);  // copied (and armed) at the next collect
    }

    // Advances across contiguous MEM_COMMIT regions starting at `from`, capped at `limit`.
    static std::uintptr_t committed_end_from(std::uintptr_t from, std::uintptr_t limit) {
        std::uintptr_t p = from;
        while (p < limit) {
            MEMORY_BASIC_INFORMATION m{};
            if (!VirtualQuery(reinterpret_cast<const void*>(p), &m, sizeof m) || m.State != MEM_COMMIT) break;
            const std::uintptr_t region_end = reinterpret_cast<std::uintptr_t>(m.BaseAddress) + m.RegionSize;
            if (region_end <= p) break;
            p = (std::min)(limit, region_end);
        }
        return p;
    }

    // Rebuilds cached region-batched runs from ww_pages_. Called only when ww_pages_ or title_write_watch reservations
    // change; steady-state collect() reuses ww_runs_ directly in O(runs + dirty).
    void rebuild_ww_runs() {
        ww_runs_.clear();
        ww_runs_dirty_ = false;
        ww_gen_ = title_write_watch::generation();
        if (ww_pages_.empty()) return;

        ww_sorted_.assign(ww_pages_.begin(), ww_pages_.end());
        std::sort(ww_sorted_.begin(), ww_sorted_.end());

        std::uintptr_t run_from = 0, run_to = 0;
        std::uintptr_t reg_end = 0;         // end of current title_write_watch reservation
        std::uintptr_t run_commit_end = 0;  // verified unbroken MEM_COMMIT end starting from run_from

        for (const std::uintptr_t page : ww_sorted_) {
            const std::uintptr_t page_end = page + Policy::kPage;
            if (page_end <= page) {
                ww_refused_.push_back(page);
                continue;
            }
            if (page >= reg_end) {
                if (run_from != run_to) {
                    ww_runs_.emplace_back(run_from, run_to);
                    run_from = run_to = run_commit_end = 0;
                }
                reg_end = title_write_watch::end_of(page);
            }
            if (!reg_end || page_end > reg_end) {
                ww_refused_.push_back(page);
                continue;
            }
            if (run_from != run_to) {
                const std::uintptr_t gap = page - run_to;
                const std::uintptr_t merged_bytes = page_end - run_from;
                if (merged_bytes <= kMaxWwRunBytes && gap <= kWwMergeGap) {
                    if (run_commit_end < page_end && run_commit_end >= run_to) {
                        const std::uintptr_t cap =
                            reg_end - run_from > kMaxWwRunBytes ? run_from + kMaxWwRunBytes : reg_end;
                        run_commit_end = committed_end_from(run_commit_end, cap);
                    }
                    if (run_commit_end >= page_end) {
                        run_to = page_end;
                        continue;
                    }
                }
                ww_runs_.emplace_back(run_from, run_to);
                run_from = run_to = run_commit_end = 0;
            }
            run_from = page;
            run_to = page_end;
            const std::uintptr_t cap = reg_end - run_from > kMaxWwRunBytes ? run_from + kMaxWwRunBytes : reg_end;
            run_commit_end = committed_end_from(run_from, cap);
        }
        if (run_from != run_to) ww_runs_.emplace_back(run_from, run_to);
    }

    void collect_write_watched(std::vector<std::uint64_t>& out) {
        ww_dirty_.clear();
        ww_refused_.clear();
        ww_retry_runs_.clear();

        if (ww_runs_dirty_ || ww_gen_ != title_write_watch::generation()) rebuild_ww_runs();

        const auto record_dirty = [&](std::uintptr_t from, std::uintptr_t to) {
            for (std::uintptr_t p = from & ~(Policy::kPage - 1); p < to;) {
                if (ww_pages_.count(p)) ww_dirty_.push_back(p);
                if (to - p <= Policy::kPage) break;
                p += Policy::kPage;
            }
        };

        if (!ww_runs_.empty()) {
            title_write_watch::ask(
                ww_reader_, ww_runs_, record_dirty,
                [&](std::uintptr_t from, std::uintptr_t to) {
                    // A multi-page run whose reservation still exists may have acquired an uncommitted hole since
                    // ww_runs_ was cached: split and retry its watched pages before falling back to guards.
                    if (to - from > Policy::kPage && title_write_watch::contains(from)) {
                        ww_retry_runs_.emplace_back(from, to);
                        return;
                    }
                    for (std::uintptr_t p = from & ~(Policy::kPage - 1); p < to;) {
                        if (ww_pages_.count(p)) ww_refused_.push_back(p);
                        if (to - p <= Policy::kPage) break;
                        p += Policy::kPage;
                    }
                });
        }

        if (!ww_retry_runs_.empty()) {
            ww_runs_dirty_ = true;
            ww_split_runs_.clear();
            for (const auto& [from, to] : ww_retry_runs_) {
                // The first ask() already advanced ww_reader_'s journal cursor, so any write journaled by another
                // reader or title_write_watch::mark() before it would be skipped by a second ask(). Mark every
                // watched page in the refused run dirty conservatively, then split-ask to reset kernel bits and find
                // any individual page that genuinely lost write-watching.
                record_dirty(from, to);
                for (std::uintptr_t p = from & ~(Policy::kPage - 1); p < to;) {
                    if (ww_pages_.count(p)) ww_split_runs_.emplace_back(p, p + Policy::kPage);
                    if (to - p <= Policy::kPage) break;
                    p += Policy::kPage;
                }
            }
            if (!ww_split_runs_.empty()) {
                title_write_watch::ask(
                    ww_reader_, ww_split_runs_, [](std::uintptr_t, std::uintptr_t) {},
                    [&](std::uintptr_t from, std::uintptr_t to) {
                        for (std::uintptr_t p = from & ~(Policy::kPage - 1); p < to;) {
                            if (ww_pages_.count(p)) ww_refused_.push_back(p);
                            if (to - p <= Policy::kPage) break;
                            p += Policy::kPage;
                        }
                    });
            }
        }

        for (const std::uintptr_t p : ww_initial_)
            if (ww_pages_.count(p)) ww_dirty_.push_back(p);
        ww_initial_.clear();

        // Any page whose MEM_WRITE_WATCH reservation disappeared falls back to guarded tracking without losing a write.
        if (!ww_refused_.empty()) {
            ww_runs_dirty_ = true;
            for (const std::uintptr_t p : ww_refused_) {
                if (ww_pages_.erase(p)) {
                    watch_guarded(p);
                    ww_dirty_.push_back(p);
                }
            }
            if (ww_pages_.empty()) ww_runs_.clear();
        }
        if (!ww_dirty_.empty()) {
            std::sort(ww_dirty_.begin(), ww_dirty_.end());
            ww_dirty_.erase(std::unique(ww_dirty_.begin(), ww_dirty_.end()), ww_dirty_.end());
            const bool had_guarded = !out.empty();
            for (const std::uintptr_t p : ww_dirty_) out.push_back(p);
            if (had_guarded) {
                std::sort(out.begin(), out.end());
                out.erase(std::unique(out.begin(), out.end()), out.end());
            }
        }
    }

    struct CellGuard {
        static constexpr std::size_t kMaxBatchRunPages = 1024;  // up to 64 MiB per contiguous VirtualProtect run
        static constexpr std::size_t kMaxLockBatchPages = 128;  // pages per pin check + geometry_writes lock hold
        static constexpr std::size_t kMaxBridgeGap = 32;        // max already-armed cold gap tiles to bridge (2 MiB)

        struct TrackedCell {
            geometry_writes::Cell* cell = nullptr;
            std::uint64_t armed_version = 0;
            std::uint64_t known_mapping = 0;
            DWORD known_protect = 0;  // uniform writable protection across all 16 sub-pages (0: unverified)
            DWORD known_guarded = 0;  // geometry_writes::read_only(known_protect)
            bool is_armed = false;
            bool is_hot = false;
        };

        std::unordered_map<std::uint64_t, TrackedCell> cells;
        std::uint64_t known_protect_gen = 0;
        bool has_known_protect = false;
        ArmStats arm_stats{};
        // Guarded tiles observed with PAGE_WRITECOMBINE; mutated only under page_mirror.mutex, read-only during upload().
        std::unordered_set<std::uintptr_t> wc_pages;
        std::atomic<std::size_t> wc_page_count{0};

        void note_wc_page(std::uintptr_t tile, bool is_wc) {
            tile &= ~(geometry_writes::tile_size - 1);
            if (is_wc) {
                if (wc_pages.insert(tile).second)
                    wc_page_count.store(wc_pages.size(), std::memory_order_release);
            } else if (wc_page_count.load(std::memory_order_relaxed) != 0) {
                if (wc_pages.erase(tile))
                    wc_page_count.store(wc_pages.size(), std::memory_order_release);
            }
        }

        bool is_wc_page(std::uintptr_t addr) const noexcept {
            if (wc_page_count.load(std::memory_order_acquire) == 0) return false;
            return wc_pages.count(addr & ~(geometry_writes::tile_size - 1)) != 0;
        }

        template <class StateT>
        static void check_table_backed(const StateT& s, const std::uintptr_t* pages, std::uint8_t* out,
                                       std::size_t count) {
            if constexpr (requires { s.backed_batch.load(std::memory_order_acquire)(pages, out, count); }) {
                if (const auto fn = s.backed_batch.load(std::memory_order_acquire)) {
                    fn(pages, out, count);
                    return;
                }
            }
            for (std::size_t k = 0; k < count; ++k) out[k] = title_frames::backed(pages[k]) ? 1 : 0;
        }

        bool is_tile_pinned(std::uintptr_t tile) const noexcept {
            constexpr std::uintptr_t kTile = geometry_writes::tile_size;
            const std::uintptr_t tile_hi = (tile <= UINTPTR_MAX - kTile) ? (tile + kTile) : UINTPTR_MAX;
            for (const auto& [p_lo, p_hi] : run_pin_ranges_) {
                if (tile < p_hi && p_lo < tile_hi) return true;
            }
            return false;
        }

        // True when no title_memory_map change after `since` overlapped [lo, hi). Sets `gen_out` to the generation
        // checked up to (leaf SRWLOCK shared read when generation moved).
        static bool range_clean_since(std::uint64_t since, std::uintptr_t lo, std::uintptr_t hi,
                                      std::uint64_t& gen_out) {
            const std::uint64_t cur = title_memory_map::generation();
            if (cur == since) {
                gen_out = cur;
                return true;
            }
            bool overlapped = false;
            std::uint64_t now = cur;
            if (!title_memory_map::changes_since(since, now, lo, hi,
                                                 [&](std::uintptr_t c_from, std::uintptr_t c_to) {
                                                     if (c_from < c_to) overlapped = true;
                                                 })) {
                return false;
            }
            gen_out = now;
            return !overlapped;
        }

        void sync_known_protect() {
            const std::uint64_t cur = title_memory_map::generation();
            if (cur == known_protect_gen) return;
            if (!has_known_protect) {
                known_protect_gen = cur;
                return;
            }
            std::uint64_t now = cur;
            bool invalidate_all = false;
            if (!title_memory_map::changes_since(known_protect_gen, now, 0, UINTPTR_MAX,
                                                 [&](std::uintptr_t c_from, std::uintptr_t c_to) {
                                                     if (invalidate_all || c_from >= c_to) return;
                                                     constexpr std::uintptr_t kTile = geometry_writes::tile_size;
                                                     const std::uintptr_t first = c_from & ~(kTile - 1);
                                                     const std::uintptr_t span = c_to - first;
                                                     const std::size_t tiles = std::size_t((span - 1) / kTile) + 1;
                                                     if (tiles > cells.size() / 2 + 64) {
                                                         invalidate_all = true;
                                                         return;
                                                     }
                                                     for (std::size_t t = 0; t < tiles; ++t) {
                                                         const auto it = cells.find(first + std::uintptr_t(t) * kTile);
                                                         if (it != cells.end()) {
                                                             it->second.known_protect = 0;
                                                             it->second.known_guarded = 0;
                                                         }
                                                     }
                                                 })) {
                invalidate_all = true;
            }
            if (invalidate_all) {
                for (auto& entry : cells) {
                    entry.second.known_protect = 0;
                    entry.second.known_guarded = 0;
                }
                has_known_protect = false;
            }
            known_protect_gen = now;
        }

        // Guarded (cold, or a hot page's probe): no longer hot for this owner, then armed.
        bool arm(std::uintptr_t page) {
            std::uint8_t ok = 0;
            arm_batch(&page, &ok, 1);
            return ok != 0;
        }

        // Batched arming of sorted 64 KiB pages: coalesces both contiguous runs (up to 64 MiB in one VirtualProtect)
        // and scattered pages (bridging up to kMaxBridgeGap already-armed unpinned cold tiles) into shared pin checks,
        // shared geometry_writes lock holds, a per-tile known-protection cache synced per lock-batch and range-checked
        // via title_memory_map::changes_since() (never permanently disabled by unrelated VA changes), and a short-lived
        // VirtualQuery region cache within each lock hold.
        void arm_batch(const std::uintptr_t* pages, std::uint8_t* results, std::size_t count) {
            using geometry_writes::Cell;
            constexpr std::uintptr_t kTile = geometry_writes::tile_size;
            constexpr std::uintptr_t kHostPage = geometry_writes::page_size;

            if (!count) return;
            const std::int64_t arm_t0 = submit_phases::now_ns();
            struct ArmTimer {
                std::int64_t t0;
                std::uint64_t& ns;
                ~ArmTimer() {
                    const std::int64_t dt = submit_phases::now_ns() - t0;
                    if (dt > 0) ns += std::uint64_t(dt);
                }
            } arm_timer{arm_t0, arm_stats.ns};

            for (std::size_t i = 0; i < count;) {
                // Re-sync known_protect per lock-batch (outside geometry_writes::state().mutex) so concurrent map
                // changes on unrelated VA ranges never disable the fast path for the rest of arm_batch().
                sync_known_protect();

                // Extend chunk up to kMaxLockBatchPages, or further across one contiguous or bridgeable run up to
                // kMaxBatchRunPages so lock-batch cuts never split a bridgeable cluster mid-bridge.
                std::size_t j = i + 1;
                while (j < count) {
                    const std::uintptr_t prev_va = pages[j - 1];
                    const std::uintptr_t next_va = pages[j];
                    const bool ordered = prev_va <= UINTPTR_MAX - kTile && next_va >= prev_va + kTile;
                    const bool contiguous = ordered && next_va == prev_va + kTile;
                    const bool bridgeable = ordered && (next_va - (prev_va + kTile)) <= kMaxBridgeGap * kTile;
                    const std::size_t len = j - i;
                    if (len < kMaxLockBatchPages) {
                        ++j;
                    } else if ((contiguous || bridgeable) && len < kMaxBatchRunPages) {
                        ++j;
                    } else {
                        break;
                    }
                }
                const std::size_t batch_n = j - i;
                const std::uintptr_t* const batch_pages = pages + i;

                run_entries_.resize(batch_n);
                run_versions_.resize(batch_n);
                run_pinned_.assign(batch_n, 0);
                run_backed_.assign(batch_n, 0);
                run_eligible_.assign(batch_n, 0);
                run_failed_.assign(batch_n, 0);
                run_slow_.assign(batch_n, 0);
                run_wc_.assign(batch_n, 0);
                run_pin_ranges_.clear();
                run_clear_hot_.clear();

                // Clear hot flags (batched under one geometry_writes::state().mutex hold) BEFORE snapshotting versions,
                // querying kernel pins, or querying table-backed frames.
                std::uintptr_t pin_lo = UINTPTR_MAX;
                std::uintptr_t pin_hi = 0;
                for (std::size_t k = 0; k < batch_n; ++k) {
                    const std::uintptr_t page = batch_pages[k];
                    const auto it = cells.find(page);
                    if (it == cells.end()) {
                        run_entries_[k] = nullptr;
                        results[i + k] = 0;
                        continue;
                    }
                    TrackedCell& tc = it->second;
                    if (tc.is_hot) {
                        tc.is_hot = false;
                        run_clear_hot_.push_back(page);
                    }
                    run_entries_[k] = &tc;
                    if (page < pin_lo) pin_lo = page;
                    const std::uintptr_t page_hi = (page <= UINTPTR_MAX - kTile) ? (page + kTile) : UINTPTR_MAX;
                    if (page_hi > pin_hi) pin_hi = page_hi;
                }
                if (!run_clear_hot_.empty()) {
                    geometry_writes::set_hot_batch(run_clear_hot_.data(), run_clear_hot_.size(), false);
                }
                for (std::size_t k = 0; k < batch_n; ++k) {
                    if (run_entries_[k]) {
                        run_versions_[k] = run_entries_[k]->cell->version.load(std::memory_order_acquire);
                    }
                }

                // One pin check for the entire batch (never under geometry_writes::state().mutex). Record every
                // overlapping pin interval in run_pin_ranges_ so gap-bridging NEVER bridges over a pinned tile.
                if (pin_lo < pin_hi) {
                    kernel_writes::for_each_pin(pin_lo, pin_hi, [&](std::uintptr_t p_lo, std::uintptr_t p_hi) {
                        run_pin_ranges_.emplace_back(p_lo, p_hi);
                        const std::uintptr_t first_tile = p_lo & ~(kTile - 1);
                        const std::uintptr_t* it =
                            std::lower_bound(batch_pages, batch_pages + batch_n, first_tile);
                        for (; it < batch_pages + batch_n && *it < p_hi; ++it) {
                            run_pinned_[std::size_t(it - batch_pages)] = 1;
                        }
                    });
                    // Table-backed pages have no host view (MEM_RESERVE): skip outside geometry_writes::state().mutex
                    // so g_frame_lock and s.mutex never nest. Any concurrent remap bumps c->version and is caught below.
                    if (title_frames::may_hold(pin_lo, pin_hi)) {
                        run_backed_addrs_.clear();
                        run_backed_idx_.clear();
                        for (std::size_t k = 0; k < batch_n; ++k) {
                            if (run_entries_[k] && !run_pinned_[k]) {
                                run_backed_addrs_.push_back(batch_pages[k]);
                                run_backed_idx_.push_back(k);
                            }
                        }
                        if (!run_backed_addrs_.empty()) {
                            run_backed_res_.assign(run_backed_addrs_.size(), 0);
                            check_table_backed(title_frames::state(), run_backed_addrs_.data(), run_backed_res_.data(),
                                               run_backed_addrs_.size());
                            for (std::size_t m = 0; m < run_backed_addrs_.size(); ++m) {
                                if (run_backed_res_[m]) run_backed_[run_backed_idx_[m]] = 1;
                            }
                        }
                    }
                }

                auto& s = geometry_writes::state();
                {
                    std::lock_guard lock(s.mutex);
                    for (std::size_t k = 0; k < batch_n; ++k) {
                        TrackedCell* tc = run_entries_[k];
                        if (!tc) continue;
                        Cell* c = tc->cell;
                        if (c->hot) {
                            results[i + k] = 1;
                            tc->is_armed = true;
                            tc->armed_version = run_versions_[k];
                            ++arm_stats.skip_tiles;
                            continue;
                        }
                        if (run_pinned_[k] || run_versions_[k] != c->version.load(std::memory_order_relaxed)) {
                            results[i + k] = 0;
                            tc->is_armed = false;
                            continue;
                        }
                        // Zero-syscall fast path: all 16 sub-pages already armed read-only (e.g. by another owner).
                        bool all_armed = true;
                        for (const DWORD r : c->restore) {
                            if (!r) {
                                all_armed = false;
                                break;
                            }
                        }
                        if (all_armed) {
                            results[i + k] = 1;
                            tc->is_armed = true;
                            tc->armed_version = run_versions_[k];
                            ++arm_stats.already_armed_skips;
                            ++arm_stats.skip_tiles;
                            if (c->restore[0] & PAGE_WRITECOMBINE) {
                                ++arm_stats.wc_tiles;
                                note_wc_page(batch_pages[k], true);
                            }
                            continue;
                        }
                        if (run_backed_[k]) {
                            results[i + k] = 1;
                            tc->is_armed = true;
                            tc->armed_version = run_versions_[k];
                            ++arm_stats.table_skips;
                            ++arm_stats.skip_tiles;
                            continue;
                        }
                        run_eligible_[k] = 1;
                    }

                    // Short-lived VirtualQuery cache within this lock hold: if a uniform MEM_COMMIT (writable or
                    // read-only) or MEM_RESERVE region extends past the current sub-run into subsequent pages in the
                    // same batch and no title map change overlapped it, reuse its state/protection instead of issuing
                    // a redundant VirtualQuery syscall.
                    std::uintptr_t cached_lo = 0;
                    std::uintptr_t cached_hi = 0;
                    DWORD cached_state = 0;
                    DWORD cached_protect = 0;
                    DWORD cached_guarded = 0;
                    std::uint64_t cached_gen = 0;

                    // Protect each contiguous (or gap-bridged) sub-run of eligible cells with batched VirtualProtect
                    // (skipping VirtualQuery when per-tile known_protect is valid).
                    for (std::size_t sub_start = 0; sub_start < batch_n;) {
                        if (!run_eligible_[sub_start]) {
                            ++sub_start;
                            continue;
                        }
                        std::size_t sub_end = sub_start + 1;
                        while (sub_end < batch_n && run_eligible_[sub_end] &&
                               batch_pages[sub_end] == batch_pages[sub_end - 1] + kTile) {
                            ++sub_end;
                        }

                        // Gap-bridging fast path: coalesce scattered eligible tiles separated by <= kMaxBridgeGap
                        // already-armed, unpinned cold tiles sharing identical known_protect/known_guarded/mapping.
                        // Falls back to eligible-only [sub_start, sub_end) on any failure or unexpected old protect.
                        TrackedCell* const tc0 = run_entries_[sub_start];
                        if (tc0->known_protect != 0 &&
                            tc0->known_mapping == tc0->cell->mapping.load(std::memory_order_acquire)) {
                            const DWORD kp = tc0->known_protect;
                            const DWORD kg = tc0->known_guarded;
                            const std::uintptr_t span_lo = batch_pages[sub_start];
                            std::size_t bridge_end = sub_start + 1;
                            while (bridge_end < batch_n && run_eligible_[bridge_end]) {
                                const std::uintptr_t prev_va = batch_pages[bridge_end - 1];
                                const std::uintptr_t next_va = batch_pages[bridge_end];
                                if (prev_va > UINTPTR_MAX - kTile || next_va > UINTPTR_MAX - kTile) break;
                                const std::uintptr_t prev_end = prev_va + kTile;
                                if (next_va < prev_end) break;
                                const std::uintptr_t gap_bytes = next_va - prev_end;
                                if (gap_bytes > kMaxBridgeGap * kTile) break;
                                if ((next_va + kTile - span_lo) > kMaxBatchRunPages * kTile) break;
                                TrackedCell* const tcn = run_entries_[bridge_end];
                                if (tcn->known_protect != kp || tcn->known_guarded != kg ||
                                    tcn->known_mapping != tcn->cell->mapping.load(std::memory_order_acquire))
                                    break;
                                bool gap_ok = true;
                                for (std::uintptr_t gap_va = prev_end; gap_va < next_va; gap_va += kTile) {
                                    if (is_tile_pinned(gap_va)) {
                                        gap_ok = false;
                                        break;
                                    }
                                    const auto git = cells.find(gap_va);
                                    if (git == cells.end()) {
                                        gap_ok = false;
                                        break;
                                    }
                                    const TrackedCell& gtc = git->second;
                                    if (!gtc.is_armed || gtc.is_hot || gtc.cell->hot ||
                                        gtc.cell->version.load(std::memory_order_relaxed) != gtc.armed_version ||
                                        gtc.known_protect != kp || gtc.known_guarded != kg ||
                                        gtc.known_mapping != gtc.cell->mapping.load(std::memory_order_acquire) ||
                                        gtc.cell->restore[0] != kp) {
                                        gap_ok = false;
                                        break;
                                    }
                                }
                                if (!gap_ok) break;
                                ++bridge_end;
                            }
                            if (bridge_end > sub_end) {
                                const std::uintptr_t span_hi = batch_pages[bridge_end - 1] + kTile;
                                std::uint64_t verified_gen = 0;
                                if (range_clean_since(known_protect_gen, span_lo, span_hi, verified_gen)) {
                                    DWORD old{};
                                    ++arm_stats.vp_calls;
                                    if (VirtualProtect(reinterpret_cast<void*>(span_lo), span_hi - span_lo, kg, &old)) {
                                        if (old == kp || old == kg) {
                                            const std::size_t n_bridged = bridge_end - sub_start;
                                            ++arm_stats.bridged_runs;
                                            arm_stats.fast_tiles += n_bridged;
                                            if (kp & PAGE_WRITECOMBINE) arm_stats.wc_tiles += n_bridged;
                                            for (std::size_t k = sub_start; k < bridge_end; ++k) {
                                                TrackedCell* const tc = run_entries_[k];
                                                tc->cell->restore.fill(kp);
                                                tc->cell->armed_map_gen = verified_gen;
                                                results[i + k] = 1;
                                                tc->is_armed = true;
                                                tc->armed_version = run_versions_[k];
                                            }
                                            if (cached_hi && span_lo >= cached_lo && span_hi <= cached_hi)
                                                cached_lo = span_hi;
                                            sub_start = bridge_end;
                                            continue;
                                        }
                                        // Unexpected prior protection: restore `old` and fall back to eligible-only.
                                        DWORD dummy{};
                                        ++arm_stats.vp_calls;
                                        VirtualProtect(reinterpret_cast<void*>(span_lo), span_hi - span_lo, old,
                                                       &dummy);
                                    }
                                }
                            }
                        }

                        const std::uintptr_t sub_lo = batch_pages[sub_start];
                        const std::uintptr_t sub_hi = sub_lo + (sub_end - sub_start) * kTile;

                        for (std::uintptr_t p = sub_lo; p < sub_hi;) {
                            // Fast path: tile-aligned repeat arm with verified uniform protection and no map change
                            // overlapping [p, end) since known_protect_gen -> VirtualProtect directly with 0
                            // VirtualQuery syscalls.
                            if ((p & (kTile - 1)) == 0 && sub_hi - p >= kTile) {
                                const std::size_t tile_k = sub_start + std::size_t((p - sub_lo) / kTile);
                                TrackedCell* const tc_fast = run_entries_[tile_k];
                                if (tc_fast->known_protect != 0 &&
                                    tc_fast->known_mapping == tc_fast->cell->mapping.load(std::memory_order_acquire)) {
                                    const DWORD kp = tc_fast->known_protect;
                                    const DWORD kg = tc_fast->known_guarded;
                                    std::size_t fast_end = tile_k + 1;
                                    while (fast_end < sub_end) {
                                        TrackedCell* const tcn = run_entries_[fast_end];
                                        if (tcn->known_protect != kp || tcn->known_guarded != kg ||
                                            tcn->known_mapping != tcn->cell->mapping.load(std::memory_order_acquire))
                                            break;
                                        ++fast_end;
                                    }
                                    const std::uintptr_t end = p + (fast_end - tile_k) * kTile;
                                    std::uint64_t verified_gen = 0;
                                    if (range_clean_since(known_protect_gen, p, end, verified_gen)) {
                                        DWORD old{};
                                        ++arm_stats.vp_calls;
                                        if (VirtualProtect(reinterpret_cast<void*>(p), end - p, kg, &old)) {
                                            if (old == kp || old == kg) {
                                                const std::size_t n_fast = fast_end - tile_k;
                                                arm_stats.fast_tiles += n_fast;
                                                if (kp & PAGE_WRITECOMBINE) arm_stats.wc_tiles += n_fast;
                                                for (std::size_t t = tile_k; t < fast_end; ++t) {
                                                    run_entries_[t]->cell->restore.fill(kp);
                                                    run_entries_[t]->cell->armed_map_gen = verified_gen;
                                                }
                                                if (cached_hi && p >= cached_lo && end <= cached_hi) cached_lo = end;
                                                p = end;
                                                continue;
                                            }
                                            // Unexpected prior protection (caller bypassed title_memory_map::changed):
                                            // restore `old` before falling through to VirtualQuery + VirtualProtect.
                                            DWORD dummy{};
                                            ++arm_stats.vp_calls;
                                            VirtualProtect(reinterpret_cast<void*>(p), end - p, old, &dummy);
                                        }
                                        // Unexpected protection or VirtualProtect failure: invalidate cached protection
                                        // for these tiles and fall through to VirtualQuery + VirtualProtect.
                                        for (std::size_t t = tile_k; t < fast_end; ++t) {
                                            run_entries_[t]->known_protect = 0;
                                            run_entries_[t]->known_guarded = 0;
                                        }
                                        cached_hi = 0;
                                    }
                                }
                            }

                            std::uintptr_t region_base = 0;
                            std::uintptr_t region_end = 0;
                            DWORD state = 0;
                            DWORD protect = 0;
                            DWORD guarded = 0;
                            std::uint64_t query_gen = 0;
                            bool from_cache = false;

                            std::uint64_t cache_now = 0;
                            if (p >= cached_lo && p < cached_hi &&
                                range_clean_since(cached_gen, p, (std::min)(sub_hi, cached_hi), cache_now)) {
                                region_base = cached_lo;
                                region_end = cached_hi;
                                state = cached_state;
                                protect = cached_protect;
                                guarded = cached_guarded;
                                query_gen = cached_gen;
                                from_cache = true;
                            } else {
                                query_gen = title_memory_map::generation();
                                MEMORY_BASIC_INFORMATION m{};
                                ++arm_stats.vq_calls;
                                if (!VirtualQuery(reinterpret_cast<void*>(p), &m, sizeof m)) {
                                    const std::size_t tile_k = sub_start + std::size_t((p - sub_lo) / kTile);
                                    run_failed_[tile_k] = 1;
                                    if (!run_slow_[tile_k]) {
                                        run_slow_[tile_k] = 1;
                                        ++arm_stats.slow_tiles;
                                    }
                                    p = (p & ~(kTile - 1)) + kTile;
                                    continue;
                                }
                                region_base = reinterpret_cast<std::uintptr_t>(m.BaseAddress);
                                region_end = region_base + m.RegionSize;
                                state = m.State;
                                protect = m.Protect;
                                guarded = geometry_writes::read_only(m.Protect);
                                if (region_end > sub_hi &&
                                    (state == MEM_RESERVE || (state == MEM_COMMIT && !(protect & PAGE_GUARD)))) {
                                    cached_lo = p;
                                    cached_hi = region_end;
                                    cached_state = state;
                                    cached_protect = protect;
                                    cached_guarded = guarded;
                                    cached_gen = query_gen;
                                } else {
                                    cached_hi = 0;
                                }
                            }

                            const std::uintptr_t end = (std::min)(sub_hi, region_end);
                            if (end <= p) {
                                const std::size_t tile_k = sub_start + std::size_t((p - sub_lo) / kTile);
                                run_failed_[tile_k] = 1;
                                if (!run_slow_[tile_k]) {
                                    run_slow_[tile_k] = 1;
                                    ++arm_stats.slow_tiles;
                                }
                                cached_hi = 0;
                                p = (p & ~(kTile - 1)) + kTile;
                                continue;
                            }
                            for (std::uintptr_t t = p & ~(kTile - 1); t < end; t += kTile) {
                                const std::size_t tile_k = sub_start + std::size_t((t - sub_lo) / kTile);
                                if (!run_slow_[tile_k]) {
                                    run_slow_[tile_k] = 1;
                                    ++arm_stats.slow_tiles;
                                }
                                if (state == MEM_COMMIT) {
                                    if (protect & PAGE_WRITECOMBINE) {
                                        if (!run_wc_[tile_k]) {
                                            run_wc_[tile_k] = 1;
                                            ++arm_stats.wc_tiles;
                                        }
                                        note_wc_page(t, true);
                                    } else if (!run_wc_[tile_k]) {
                                        note_wc_page(t, false);
                                    }
                                }
                                if (t > UINTPTR_MAX - kTile) break;
                            }
                            if (state == MEM_COMMIT && !(protect & PAGE_GUARD) && guarded) {
                                DWORD old{};
                                ++arm_stats.vp_calls;
                                if (VirtualProtect(reinterpret_cast<void*>(p), end - p, guarded, &old)) {
                                    for (std::uintptr_t q = p; q < end; q += kHostPage) {
                                        const std::size_t tile_k = sub_start + std::size_t((q - sub_lo) / kTile);
                                        const std::size_t sub_p = std::size_t((q & (kTile - 1)) / kHostPage);
                                        run_entries_[tile_k]->cell->restore[sub_p] = protect;
                                        run_entries_[tile_k]->cell->armed_map_gen = Cell::kNoMapGen;
                                    }
                                    // Cache uniform whole-tile protections (and Cell::armed_map_gen) when no map change
                                    // overlapped [first_full, last_full) between query_gen and VirtualProtect.
                                    const std::uintptr_t first_full = (p + kTile - 1) & ~(kTile - 1);
                                    const std::uintptr_t last_full = end & ~(kTile - 1);
                                    std::uint64_t after_gen = 0;
                                    if (first_full < last_full && region_base <= first_full &&
                                        region_end >= last_full &&
                                        range_clean_since(query_gen, first_full, last_full, after_gen)) {
                                        for (std::uintptr_t t = first_full; t < last_full; t += kTile) {
                                            const std::size_t tile_k =
                                                sub_start + std::size_t((t - sub_lo) / kTile);
                                            TrackedCell* const tc = run_entries_[tile_k];
                                            tc->known_protect = protect;
                                            tc->known_guarded = guarded;
                                            tc->known_mapping = tc->cell->mapping.load(std::memory_order_acquire);
                                            tc->cell->armed_map_gen = after_gen;
                                        }
                                        has_known_protect = true;
                                    }
                                    if (cached_hi && p >= cached_lo && end <= cached_hi) cached_lo = end;
                                } else if (from_cache) {
                                    // Cached region is stale: invalidate and retry this address with a fresh query.
                                    cached_hi = 0;
                                    continue;
                                } else {
                                    // Multi-tile VirtualProtect failed: fall back per-tile so healthy neighbors arm.
                                    cached_hi = 0;
                                    for (std::uintptr_t t = p; t < end;) {
                                        const std::uintptr_t t_end = (std::min)(end, (t & ~(kTile - 1)) + kTile);
                                        const std::size_t tile_k = sub_start + std::size_t((t - sub_lo) / kTile);
                                        ++arm_stats.vp_calls;
                                        if (VirtualProtect(reinterpret_cast<void*>(t), t_end - t, guarded, &old)) {
                                            for (std::uintptr_t q = t; q < t_end; q += kHostPage) {
                                                const std::size_t sub_p = std::size_t((q & (kTile - 1)) / kHostPage);
                                                run_entries_[tile_k]->cell->restore[sub_p] = protect;
                                            }
                                            run_entries_[tile_k]->cell->armed_map_gen = Cell::kNoMapGen;
                                        } else {
                                            run_failed_[tile_k] = 1;
                                        }
                                        t = t_end;
                                    }
                                }
                            } else {
                                if (cached_hi && p >= cached_lo && end <= cached_hi) cached_lo = end;
                            }
                            p = end;
                        }

                        for (std::size_t k = sub_start; k < sub_end; ++k) {
                            TrackedCell* const tc = run_entries_[k];
                            if (!run_failed_[k]) {
                                results[i + k] = 1;
                                tc->is_armed = true;
                                tc->armed_version = run_versions_[k];
                            } else {
                                results[i + k] = 0;
                                tc->is_armed = false;
                            }
                        }
                        sub_start = sub_end;
                    }
                }
                i = j;
            }
        }

        // Hot: the cell is held hot (writable, and written at every look for every other copy of the page), not
        // just reopened behind the other copies' backs (which left them blind: run 64's frozen video).
        void disarm(std::uintptr_t page) {
            const auto it = cells.find(page);
            if (it == cells.end()) return;
            it->second.is_armed = false;
            if (!it->second.is_hot) {
                it->second.is_hot = true;
                geometry_writes::set_hot(page, true);
            }
        }

        // Batched disarm for TitleWritePolicy::frame() (kProbe cycles & hot-promotion batches): sets hot on all newly
        // hot tiles under a single geometry_writes::state().mutex hold.
        void disarm_batch(const std::uintptr_t* pages, std::size_t count) {
            if (!count) return;
            if (count == 1) {
                disarm(pages[0]);
                return;
            }
            run_disarm_hot_.clear();
            for (std::size_t i = 0; i < count; ++i) {
                const auto it = cells.find(pages[i]);
                if (it == cells.end()) continue;
                it->second.is_armed = false;
                if (!it->second.is_hot) {
                    it->second.is_hot = true;
                    run_disarm_hot_.push_back(pages[i]);
                }
            }
            if (!run_disarm_hot_.empty()) {
                geometry_writes::set_hot_batch(run_disarm_hot_.data(), run_disarm_hot_.size(), true);
            }
        }

        std::vector<TrackedCell*> run_entries_;
        std::vector<std::uint64_t> run_versions_;
        std::vector<std::uint8_t> run_pinned_;
        std::vector<std::uint8_t> run_backed_;
        std::vector<std::uint8_t> run_eligible_;
        std::vector<std::uint8_t> run_failed_;
        std::vector<std::uint8_t> run_slow_;
        std::vector<std::uint8_t> run_wc_;
        std::vector<std::uintptr_t> run_clear_hot_;
        std::vector<std::uintptr_t> run_disarm_hot_;
        std::vector<std::uintptr_t> run_backed_addrs_;
        std::vector<std::size_t> run_backed_idx_;
        std::vector<std::uint8_t> run_backed_res_;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> run_pin_ranges_;
    };
    using Policy = TitleWritePolicy<CellGuard>;

    void sync_pending_atomics() noexcept {
        active_work_.store(policy_.hot_count() + policy_.dirty_count() + policy_.history_count() + ww_initial_.size(),
                           std::memory_order_release);
        ww_count_.store(ww_pages_.size(), std::memory_order_release);
    }

    CellGuard guard_;
    Policy policy_;
    geometry_writes::WriteQueue queue_;
    std::vector<std::uintptr_t> tiles_;
    std::uint64_t refused_seen_ = 0;
    std::atomic<std::size_t> active_work_{0};
    std::atomic<std::size_t> ww_count_{0};

    // Write-watched private pages (MEM_WRITE_WATCH via title_write_watch): zero guard faults, zero VirtualProtect.
    std::uint64_t ww_reader_ = 0;
    std::uint64_t ww_gen_ = 0;
    bool ww_runs_dirty_ = false;
    std::unordered_set<std::uintptr_t> ww_pages_;
    std::vector<std::uintptr_t> ww_sorted_;
    std::vector<std::uintptr_t> ww_initial_;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ww_runs_;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ww_retry_runs_;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ww_split_runs_;
    std::vector<std::uintptr_t> ww_dirty_;
    std::vector<std::uintptr_t> ww_refused_;
    std::vector<std::uintptr_t> batch_watch_addrs_;
    std::vector<CellGuard::TrackedCell*> batch_watch_tcs_;
    std::vector<geometry_writes::Cell*> batch_watch_cells_;
    std::vector<std::pair<std::uintptr_t, bool>> batch_unwatch_items_;
};

}  // namespace leonida
