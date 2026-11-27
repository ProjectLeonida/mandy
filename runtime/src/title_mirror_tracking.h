#pragma once
// Which mirrored pages the host wrote (title_mirror_tracking): the page mirror's dirty set (src/title_mirror.h).
//
// The interface is all the mirror asks for, so a tracking policy can replace the one here without touching the mirror:
//   - watch(page) when a page enters the mirror (before its home is filled: a write after the fill must be seen);
//   - unwatch(page) when it leaves;
//   - collect(out): the pages written since the last collect, each made watched again before it is returned (clear
//     and re-arm first, then the caller copies: a write during the copy is in the next collect).
// Pages are 64 KiB-aligned title addresses. Not thread-safe: the mirror's caller serialises.
//
// GuardTracking, the plainest correct policy: every watched page is guarded with the geometry watch's cell at its
// address (src/title_geometry_watch.h). A host view is made read-only and the title's first write faults; a kernel
// write (kernel_writes) or a remap moves the cell's version too. A page whose guard cannot be set (a kernel write into
// it is in progress) is returned at every collect until it can be.
#include "title_geometry_watch.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

class TitleMirrorTracking {
public:
    virtual ~TitleMirrorTracking() = default;
    virtual void watch(std::uint64_t page) = 0;
    virtual void unwatch(std::uint64_t page) = 0;
    virtual void collect(std::vector<std::uint64_t>& out) = 0;
    // Diagnostics since the last call: guards that could not be set.
    virtual std::uint64_t take_refused() { return 0; }
};

class GuardTracking final : public TitleMirrorTracking {
public:
    GuardTracking() { geometry_writes::subscribe_writes(queue_); }
    ~GuardTracking() override {
        geometry_writes::unsubscribe_writes(queue_);
        for (const auto& [page, cell] : cells_) geometry_writes::release(page);
    }
    GuardTracking(const GuardTracking&) = delete;
    GuardTracking& operator=(const GuardTracking&) = delete;

    void watch(std::uint64_t page) override {
        if (cells_.count(page)) return;
        geometry_writes::Cell* cell = geometry_writes::acquire(page);
        cells_.emplace(page, cell);
        arm(page, *cell);
    }
    void unwatch(std::uint64_t page) override {
        const auto it = cells_.find(page);
        if (it == cells_.end()) return;
        geometry_writes::release(page);
        cells_.erase(it);
    }
    void collect(std::vector<std::uint64_t>& out) override {
        out.clear();
        bool overflow = false;
        geometry_writes::take_written(queue_, tiles_, overflow);
        if (overflow) {
            tiles_.clear();
            for (const auto& [page, cell] : cells_) tiles_.push_back(page);
        }
        tiles_.insert(tiles_.end(), unarmed_.begin(), unarmed_.end());
        unarmed_.clear();
        for (const std::uintptr_t tile : tiles_) {
            const auto it = cells_.find(tile);
            if (it == cells_.end()) continue;
            arm(tile, *it->second);  // re-armed before the caller copies
            out.push_back(tile);
        }
    }
    std::uint64_t take_refused() override { return std::exchange(refused_, 0); }

private:
    void arm(std::uint64_t page, geometry_writes::Cell& cell) {
        if (geometry_writes::arm(std::uintptr_t(page), cell)) return;
        ++refused_;
        unarmed_.push_back(std::uintptr_t(page));
    }

    geometry_writes::WriteQueue queue_;
    std::unordered_map<std::uint64_t, geometry_writes::Cell*> cells_;
    std::vector<std::uintptr_t> tiles_, unarmed_;
    std::uint64_t refused_ = 0;
};

}  // namespace leonida
