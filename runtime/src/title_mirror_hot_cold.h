#pragma once
// HotColdTracking: the page mirror's tracking (TitleMirrorTracking, src/title_mirror_tracking.h) with the two-tier
// hot/cold policy of src/title_write_policy.h over the geometry watch's cells, plus zero-fault MEM_WRITE_WATCH
// tracking (title_write_watch) for pages inside write-watched private reservations.
//
// - Guarded pages (sections / ordinary pages without MEM_WRITE_WATCH): watched through geometry_writes cells, with
//   contiguous 64 KiB re-arms batched into single VirtualProtect syscalls in CellGuard::arm_batch(). One collect() is
//   one submission interval; on_present() advances the frame-hot tier once per presented frame.
// - Write-watched pages (title_write_watch::contains): queried via title_write_watch::ask() (GetWriteWatch with
//   WRITE_WATCH_FLAG_RESET) before copies, with zero guard faults, zero VirtualProtect syscalls, and zero interference
//   with geometry_writes::hot_tiles().
#include "title_mirror_tracking.h"
#include "title_write_policy.h"

#include <algorithm>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

class HotColdTracking final : public TitleMirrorTracking {
public:
    HotColdTracking() : policy_(guard_) { geometry_writes::subscribe_writes(queue_); }
    ~HotColdTracking() override {
        geometry_writes::unsubscribe_writes(queue_);
        if (ww_reader_) title_write_watch::close_reader(ww_reader_);
        for (const auto& [page, held] : guard_.hot) geometry_writes::set_hot(std::uintptr_t(page), false);
        for (const auto& [page, cell] : guard_.cells) geometry_writes::release(std::uintptr_t(page));
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
            return;
        }
        watch_guarded(addr);
    }
    void unwatch(std::uint64_t page) override {
        const std::uintptr_t addr = std::uintptr_t(page) & ~(Policy::kPage - 1);
        if (ww_pages_.erase(addr)) return;
        const auto it = guard_.cells.find(addr);
        if (it == guard_.cells.end()) return;
        policy_.remove(addr, addr + Policy::kPage);
        guard_.armed.erase(addr);
        if (guard_.hot.erase(addr)) geometry_writes::set_hot(addr, false);
        geometry_writes::release(addr);
        guard_.cells.erase(it);
    }
    // Called once per presented frame (e.g. when submit_phases::totals().present_serial advances).
    void on_present() { policy_.present_frame(); }

    void collect(std::vector<std::uint64_t>& out) override {
        out.clear();
        bool overflow = false;
        geometry_writes::take_written(queue_, tiles_, overflow);
        if (overflow)
            for (const auto& [page, cell] : guard_.cells) policy_.written(std::uintptr_t(page));
        else
            for (const std::uintptr_t tile : tiles_) {
                const auto cell = guard_.cells.find(tile);
                if (cell == guard_.cells.end()) continue;
                // A guarded page whose cell has not moved since it was armed was not written since: the entry is
                // older than the arming, or this tracking's own turn of the page from hot to cold.
                const auto armed = guard_.armed.find(tile);
                if (armed != guard_.armed.end() &&
                    armed->second == cell->second->version.load(std::memory_order_acquire))
                    continue;
                policy_.written(tile);
            }
        policy_.collect([&](std::uintptr_t page) { out.push_back(page); });
        policy_.frame();  // one submission interval a collect

        if (!ww_pages_.empty()) collect_write_watched(out);
    }
    std::uint64_t take_refused() override {
        const std::uint64_t now = policy_.stats().arm_failures;
        return now - std::exchange(refused_seen_, now);
    }
    const auto& stats() const { return policy_.stats(); }
    std::size_t write_watched_pages() const { return ww_pages_.size(); }
    std::uint64_t write_watched_copies() const { return ww_copies_; }
    // A cold page whose guard no write has reached since it was armed (TitleMirrorTracking::unwritten). A hot or
    // write-watched page returns false (conservative).
    bool unwritten(std::uint64_t page) override {
        const std::uintptr_t addr = std::uintptr_t(page) & ~(Policy::kPage - 1);
        if (ww_pages_.count(addr) || policy_.hot(addr)) return false;
        const auto cell = guard_.cells.find(addr);
        const auto armed = guard_.armed.find(addr);
        return cell != guard_.cells.end() && armed != guard_.armed.end() &&
               cell->second->version.load(std::memory_order_acquire) == armed->second;
    }

private:
    void watch_guarded(std::uintptr_t addr) {
        if (guard_.cells.count(addr)) return;
        guard_.cells.emplace(addr, geometry_writes::acquire(addr));
        policy_.add(addr, addr + Policy::kPage);  // copied (and armed) at the next collect
    }

    void collect_write_watched(std::vector<std::uint64_t>& out) {
        ww_runs_.clear();
        ww_dirty_.clear();
        ww_refused_.clear();
        std::uintptr_t run_from = 0, run_to = 0, run_limit = 0;
        for (const std::uintptr_t page : ww_pages_) {
            if (run_from != run_to && page == run_to && page + Policy::kPage <= run_limit) {
                run_to += Policy::kPage;
                continue;
            }
            if (run_from != run_to) ww_runs_.emplace_back(run_from, run_to);
            const std::uintptr_t limit = title_write_watch::end_of(page);
            if (!limit || page + Policy::kPage > limit) {
                ww_refused_.push_back(page);
                run_from = run_to = run_limit = 0;
                continue;
            }
            run_from = page;
            run_to = page + Policy::kPage;
            run_limit = limit;
        }
        if (run_from != run_to) ww_runs_.emplace_back(run_from, run_to);

        if (!ww_runs_.empty()) {
            title_write_watch::ask(
                ww_reader_, ww_runs_,
                [&](std::uintptr_t from, std::uintptr_t to) {
                    for (std::uintptr_t p = from & ~(Policy::kPage - 1); p < to;) {
                        if (ww_pages_.count(p)) ww_dirty_.push_back(p);
                        if (to - p <= Policy::kPage) break;
                        p += Policy::kPage;
                    }
                },
                [&](std::uintptr_t from, std::uintptr_t to) {
                    for (std::uintptr_t p = from & ~(Policy::kPage - 1); p < to;) {
                        if (ww_pages_.count(p)) ww_refused_.push_back(p);
                        if (to - p <= Policy::kPage) break;
                        p += Policy::kPage;
                    }
                });
        }
        for (const std::uintptr_t p : ww_initial_)
            if (ww_pages_.count(p)) ww_dirty_.push_back(p);
        ww_initial_.clear();

        // Any page whose MEM_WRITE_WATCH reservation disappeared falls back to guarded tracking without losing a write.
        for (const std::uintptr_t p : ww_refused_) {
            if (ww_pages_.erase(p)) {
                watch_guarded(p);
                ww_dirty_.push_back(p);
            }
        }
        if (!ww_dirty_.empty()) {
            std::sort(ww_dirty_.begin(), ww_dirty_.end());
            ww_dirty_.erase(std::unique(ww_dirty_.begin(), ww_dirty_.end()), ww_dirty_.end());
            ww_copies_ += ww_dirty_.size();
            const bool had_guarded = !out.empty();
            for (const std::uintptr_t p : ww_dirty_) out.push_back(p);
            if (had_guarded) {
                std::sort(out.begin(), out.end());
                out.erase(std::unique(out.begin(), out.end()), out.end());
            }
        }
    }

    struct CellGuard {
        static constexpr std::size_t kMaxBatchRunPages = 64;  // up to 4 MiB per VirtualProtect run

        std::unordered_map<std::uint64_t, geometry_writes::Cell*> cells;
        std::unordered_map<std::uint64_t, std::uint64_t> armed;  // page -> its cell's version when last armed
        std::unordered_map<std::uint64_t, bool> hot;             // pages this tracking holds hot in their cells

        // Guarded (cold, or a hot page's probe): no longer hot for this owner, then armed.
        bool arm(std::uintptr_t page) {
            const auto it = cells.find(page);
            if (it == cells.end()) return false;
            if (hot.erase(page)) geometry_writes::set_hot(page, false);
            const std::uint64_t version = it->second->version.load(std::memory_order_acquire);
            if (!geometry_writes::arm(page, *it->second)) {
                armed.erase(page);
                return false;
            }
            armed[page] = version;
            return true;
        }

        // Batched arming of sorted 64 KiB pages: merges contiguous runs into one pin check, one geometry_writes lock
        // acquisition, and one VirtualQuery/VirtualProtect per uniform region (cutting Wine server round-trips 4-8x).
        void arm_batch(const std::uintptr_t* pages, std::uint8_t* results, std::size_t count) {
            using geometry_writes::Cell;
            constexpr std::uintptr_t kTile = geometry_writes::tile_size;
            constexpr std::uintptr_t kHostPage = geometry_writes::page_size;

            for (std::size_t i = 0; i < count;) {
                // Single isolated page: reuse direct path.
                if (i + 1 == count || pages[i + 1] != pages[i] + kTile) {
                    results[i] = arm(pages[i]) ? 1 : 0;
                    ++i;
                    continue;
                }
                std::size_t j = i + 1;
                while (j < count && j - i < kMaxBatchRunPages && pages[j] == pages[j - 1] + kTile) ++j;
                const std::size_t run_n = j - i;
                const std::uintptr_t run_lo = pages[i];
                const std::uintptr_t run_hi = run_lo + run_n * kTile;

                run_cells_.resize(run_n);
                run_versions_.resize(run_n);
                run_pinned_.assign(run_n, 0);
                run_eligible_.assign(run_n, 0);
                run_failed_.assign(run_n, 0);

                // Clear hot flags and snapshot versions BEFORE querying kernel pins or taking geometry_writes::mutex.
                for (std::size_t k = 0; k < run_n; ++k) {
                    const std::uintptr_t page = pages[i + k];
                    const auto it = cells.find(page);
                    if (it == cells.end()) {
                        run_cells_[k] = nullptr;
                        results[i + k] = 0;
                        continue;
                    }
                    if (hot.erase(page)) geometry_writes::set_hot(page, false);
                    run_cells_[k] = it->second;
                    run_versions_[k] = it->second->version.load(std::memory_order_acquire);
                }

                // One pin check for the entire contiguous run (never under geometry_writes::state().mutex).
                kernel_writes::for_each_pin(run_lo, run_hi, [&](std::uintptr_t p_lo, std::uintptr_t p_hi) {
                    const std::size_t a = std::size_t((p_lo - run_lo) / kTile);
                    const std::size_t b = std::size_t(((p_hi - run_lo) + kTile - 1) / kTile);
                    for (std::size_t k = a; k < b && k < run_n; ++k) run_pinned_[k] = 1;
                });

                auto& s = geometry_writes::state();
                {
                    std::lock_guard lock(s.mutex);
                    for (std::size_t k = 0; k < run_n; ++k) {
                        Cell* c = run_cells_[k];
                        if (!c) continue;
                        if (c->hot) {
                            results[i + k] = 1;
                            armed[pages[i + k]] = run_versions_[k];
                            continue;
                        }
                        if (run_pinned_[k] || run_versions_[k] != c->version.load(std::memory_order_relaxed)) {
                            results[i + k] = 0;
                            armed.erase(pages[i + k]);
                            continue;
                        }
                        run_eligible_[k] = 1;
                    }

                    // Protect each contiguous sub-run of eligible cells with batched VirtualQuery + VirtualProtect.
                    for (std::size_t sub_start = 0; sub_start < run_n;) {
                        if (!run_eligible_[sub_start]) {
                            ++sub_start;
                            continue;
                        }
                        std::size_t sub_end = sub_start + 1;
                        while (sub_end < run_n && run_eligible_[sub_end]) ++sub_end;
                        const std::uintptr_t sub_lo = run_lo + sub_start * kTile;
                        const std::uintptr_t sub_hi = run_lo + sub_end * kTile;

                        for (std::uintptr_t p = sub_lo; p < sub_hi;) {
                            MEMORY_BASIC_INFORMATION m{};
                            if (!VirtualQuery(reinterpret_cast<void*>(p), &m, sizeof m)) {
                                const std::size_t tile_k = sub_start + std::size_t((p - sub_lo) / kTile);
                                run_failed_[tile_k] = 1;
                                p = (p & ~(kTile - 1)) + kTile;
                                continue;
                            }
                            const std::uintptr_t region_end =
                                reinterpret_cast<std::uintptr_t>(m.BaseAddress) + m.RegionSize;
                            const std::uintptr_t end = (std::min)(sub_hi, region_end);
                            if (end <= p) {
                                const std::size_t tile_k = sub_start + std::size_t((p - sub_lo) / kTile);
                                run_failed_[tile_k] = 1;
                                p = (p & ~(kTile - 1)) + kTile;
                                continue;
                            }
                            const DWORD guarded = geometry_writes::read_only(m.Protect);
                            if (m.State == MEM_COMMIT && !(m.Protect & PAGE_GUARD) && guarded) {
                                DWORD old{};
                                if (VirtualProtect(reinterpret_cast<void*>(p), end - p, guarded, &old)) {
                                    for (std::uintptr_t q = p; q < end; q += kHostPage) {
                                        const std::size_t tile_k = sub_start + std::size_t((q - sub_lo) / kTile);
                                        const std::size_t sub_p = std::size_t((q & (kTile - 1)) / kHostPage);
                                        run_cells_[tile_k]->restore[sub_p] = m.Protect;
                                    }
                                } else {
                                    // Multi-tile VirtualProtect failed: fall back per-tile so healthy neighbors arm.
                                    for (std::uintptr_t t = p; t < end;) {
                                        const std::uintptr_t t_end = (std::min)(end, (t & ~(kTile - 1)) + kTile);
                                        const std::size_t tile_k = sub_start + std::size_t((t - sub_lo) / kTile);
                                        if (VirtualProtect(reinterpret_cast<void*>(t), t_end - t, guarded, &old)) {
                                            for (std::uintptr_t q = t; q < t_end; q += kHostPage) {
                                                const std::size_t sub_p = std::size_t((q & (kTile - 1)) / kHostPage);
                                                run_cells_[tile_k]->restore[sub_p] = m.Protect;
                                            }
                                        } else {
                                            run_failed_[tile_k] = 1;
                                        }
                                        t = t_end;
                                    }
                                }
                            }
                            p = end;
                        }

                        for (std::size_t k = sub_start; k < sub_end; ++k) {
                            if (!run_failed_[k]) {
                                results[i + k] = 1;
                                armed[pages[i + k]] = run_versions_[k];
                            } else {
                                results[i + k] = 0;
                                armed.erase(pages[i + k]);
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
            armed.erase(page);
            if (cells.count(page) && hot.emplace(page, true).second) geometry_writes::set_hot(page, true);
        }

        std::vector<geometry_writes::Cell*> run_cells_;
        std::vector<std::uint64_t> run_versions_;
        std::vector<std::uint8_t> run_pinned_;
        std::vector<std::uint8_t> run_eligible_;
        std::vector<std::uint8_t> run_failed_;
    };
    using Policy = TitleWritePolicy<CellGuard>;

    CellGuard guard_;
    Policy policy_;
    geometry_writes::WriteQueue queue_;
    std::vector<std::uintptr_t> tiles_;
    std::uint64_t refused_seen_ = 0;

    // Write-watched private pages (MEM_WRITE_WATCH via title_write_watch): zero guard faults, zero VirtualProtect.
    std::uint64_t ww_reader_ = 0;
    std::set<std::uintptr_t> ww_pages_;
    std::vector<std::uintptr_t> ww_initial_;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> ww_runs_;
    std::vector<std::uintptr_t> ww_dirty_;
    std::vector<std::uintptr_t> ww_refused_;
    std::uint64_t ww_copies_ = 0;
};

}  // namespace leonida
