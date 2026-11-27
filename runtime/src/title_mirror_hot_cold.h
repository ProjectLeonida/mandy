#pragma once
// HotColdTracking: the page mirror's tracking (TitleMirrorTracking, src/title_mirror_tracking.h) with the hot/cold
// policy of src/title_write_policy.h over the geometry watch's cells. It replaces GuardTracking without a change in
// the mirror: watch() and unwatch() as there; collect() returns the pages written since the last collect (each armed
// again first) and, every time, the pages written so often that they are left writable and copied instead of guarded.
// One collect is one interval of the policy's history (the mirror collects before each submission).
#include "title_mirror_tracking.h"
#include "title_write_policy.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

class HotColdTracking final : public TitleMirrorTracking {
public:
    HotColdTracking() : policy_(guard_) { geometry_writes::subscribe_writes(queue_); }
    ~HotColdTracking() override {
        geometry_writes::unsubscribe_writes(queue_);
        for (const auto& [page, held] : guard_.hot) geometry_writes::set_hot(std::uintptr_t(page), false);
        for (const auto& [page, cell] : guard_.cells) geometry_writes::release(std::uintptr_t(page));
    }
    HotColdTracking(const HotColdTracking&) = delete;
    HotColdTracking& operator=(const HotColdTracking&) = delete;

    void watch(std::uint64_t page) override {
        if (guard_.cells.count(page)) return;
        guard_.cells.emplace(page, geometry_writes::acquire(std::uintptr_t(page)));
        policy_.add(std::uintptr_t(page), std::uintptr_t(page) + Policy::kPage);  // copied (and armed) at the next collect
    }
    void unwatch(std::uint64_t page) override {
        const auto it = guard_.cells.find(page);
        if (it == guard_.cells.end()) return;
        policy_.remove(std::uintptr_t(page), std::uintptr_t(page) + Policy::kPage);
        guard_.armed.erase(page);
        if (guard_.hot.erase(page)) geometry_writes::set_hot(std::uintptr_t(page), false);
        geometry_writes::release(std::uintptr_t(page));
        guard_.cells.erase(it);
    }
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
        policy_.frame();  // one interval a collect
    }
    std::uint64_t take_refused() override {
        const std::uint64_t now = policy_.stats().arm_failures;
        return now - std::exchange(refused_seen_, now);
    }
    const auto& stats() const { return policy_.stats(); }
    // A cold page whose guard no write has reached since it was armed (TitleMirrorTracking::unwritten). A hot page is
    // never: it is left writable.
    bool unwritten(std::uint64_t page) override {
        if (policy_.hot(std::uintptr_t(page))) return false;
        const auto cell = guard_.cells.find(page);
        const auto armed = guard_.armed.find(page);
        return cell != guard_.cells.end() && armed != guard_.armed.end() &&
               cell->second->version.load(std::memory_order_acquire) == armed->second;
    }

private:
    struct CellGuard {
        std::unordered_map<std::uint64_t, geometry_writes::Cell*> cells;
        std::unordered_map<std::uint64_t, std::uint64_t> armed;  // page -> its cell's version when last armed
        std::unordered_map<std::uint64_t, bool> hot;  // pages this tracking holds hot in their cells
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
        // Hot: the cell is held hot (writable, and written at every look for every other copy of the page), not
        // just reopened behind the other copies' backs (which left them blind: run 64's frozen video).
        void disarm(std::uintptr_t page) {
            armed.erase(page);
            if (cells.count(page) && hot.emplace(page, true).second) geometry_writes::set_hot(page, true);
        }
    };
    using Policy = TitleWritePolicy<CellGuard>;

    CellGuard guard_;
    Policy policy_;
    geometry_writes::WriteQueue queue_;
    std::vector<std::uintptr_t> tiles_;
    std::uint64_t refused_seen_ = 0;
};

}  // namespace leonida
