#pragma once
// Which pages of the title's memory a copy of them (the page mirror, src/title_mirror.h on branch page-mirror; the
// title texture refresh) must take again before a submission, and how each page is watched for that.
//
// Two ways to know a page was written, at two costs (runtime/test/guard_fault_bench.cpp measures them on the host):
//   - cold: the page is guarded (read-only); the title's first write faults, the handler opens it and tells the
//     policy (written()), and the page is copied before the next submission and guarded again. A page written once a
//     minute costs one fault a minute.
//   - hot: the page is left writable and copied before every submission, with no fault. A page the title writes in
//     most frames costs a copy a submission instead of a fault and a re-arm a frame; under Wine a fault and a re-arm
//     cost far more than copying 64 KiB.
// A page moves by its history, one bit an interval (frame() ends one): written in at least kPromote of the last
// kHistory intervals, and it turns hot at the end of the interval; a hot page is guarded again for one interval every
// kProbe intervals, and turns cold when such a probe interval sees no write. The interval is the owner's choice and
// should be the submission: a hot page is copied at every collect(), so it pays off only for a page written between
// most submissions. Measured under Wine on this machine: a 64 KiB copy about 10 us, a fault 2-3 us with a handler
// that reopens in one call (16 us page by page, now fixed in the geometry watch) and a re-arm 4 us, so a page written
// in under half the submissions is cheaper cold; kPromote 6 of 8 keeps only the clearly hot ones uncovered. Nothing is missed in between: a page is armed before it is
// reported dirty (collect() arms, then names it), so a write during the owner's copy faults and is in the next
// collect(); a page that turns hot is named at every collect() from then on, and one that turns cold is armed and
// named once more.
//
// The unit is the guard's: kPage (64 KiB, the geometry watch's tile and the mirror's page). Guard is the host side:
//   bool arm(std::uintptr_t page);     // read-only now; false when it cannot be (then the page is treated as hot)
//   void disarm(std::uintptr_t page);  // writable now (a page turning hot, or leaving the policy)
// The fault handler opens the page itself and calls written(page). Not thread-safe: the owner serialises calls (the
// handler's written() through a lock-free queue of its own, or under the owner's lock).
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace leonida {

template<class Guard>
class TitleWritePolicy {
public:
    static constexpr std::uintptr_t kPage = 0x10000;
    static constexpr unsigned kHistory = 8;  // frames of history a page keeps
    static constexpr unsigned kPromote = 6;  // written in this many of them: hot
    static constexpr unsigned kProbe = 32;   // a hot page is guarded again for one frame every this many frames

    struct Stats {
        std::uint64_t faults = 0;          // written() on a guarded page
        std::uint64_t hot_copies = 0;      // pages named because they are hot
        std::uint64_t cold_copies = 0;     // pages named because they were written while guarded
        std::uint64_t arms = 0, arm_failures = 0, promotions = 0, demotions = 0;
        std::size_t hot = 0, cold = 0;     // pages in each set now
    };

    explicit TitleWritePolicy(Guard& guard) : guard_(guard) {}

    // [begin, end) is to be copied from now on: its pages start cold and written (the owner has no copy yet).
    void add(std::uintptr_t begin, std::uintptr_t end) {
        for (std::uintptr_t p = begin & ~(kPage - 1); p < end; p += kPage) {
            auto [it, fresh] = pages_.try_emplace(p);
            if (!fresh) continue;
            it->second.written = true;
            it->second.armed = false;
            ++stats_.cold;
        }
    }
    // No longer copied: its pages are left writable.
    void remove(std::uintptr_t begin, std::uintptr_t end) {
        for (auto it = pages_.lower_bound(begin & ~(kPage - 1)); it != pages_.end() && it->first < end;) {
            if (it->second.armed) guard_.disarm(it->first);
            --(it->second.hot ? stats_.hot : stats_.cold);
            it = pages_.erase(it);
        }
    }

    // The handler opened a guarded page the title wrote (or a kernel write is about to): copy it before the next
    // submission.
    void written(std::uintptr_t page) {
        const auto it = pages_.find(page & ~(kPage - 1));
        if (it == pages_.end()) return;
        Page& p = it->second;
        if (p.armed) ++stats_.faults;
        p.armed = false;
        p.written = true;
        p.history |= 1;
    }

    // Before a submission: every page the owner must copy, each once, after it was armed again where it is cold, so a
    // write from here on is in the next collect(). `copy(page)` is called in address order.
    template<class Copy>
    void collect(Copy&& copy) {
        for (auto& [address, p] : pages_) {
            if (p.hot && !p.probing) {
                ++stats_.hot_copies;
                copy(address);
                continue;
            }
            if (!p.written && p.armed) continue;  // guarded all along and not written: the copy is current
            // Written, or not guarded since the last copy (a write then faulted nowhere): armed, then copied.
            if (!p.armed) arm(address, p);
            p.written = false;
            ++(p.hot ? stats_.hot_copies : stats_.cold_copies);
            copy(address);
        }
    }

    // A frame ended (the title presented): histories move on, and pages change sets.
    void frame() {
        ++frame_;
        const bool probe = frame_ % kProbe == 0;
        for (auto& [address, p] : pages_) {
            if (p.hot) {
                if (p.probing) {
                    // A probe frame with no write: the page is cold from now on (it stays armed).
                    p.probing = false;
                    if (!(p.history & 1) && p.armed) {
                        p.hot = false;
                        ++stats_.demotions;
                        --stats_.hot;
                        ++stats_.cold;
                    } else if (p.armed) {
                        guard_.disarm(address);
                        p.armed = false;
                    }
                } else if (probe) {
                    p.probing = true;  // guarded for the next frame (collect() arms it)
                    p.written = true;  // and copied once more after the arming
                }
            } else if (unsigned(std::popcount(unsigned(p.history & ((1u << kHistory) - 1)))) >= kPromote) {
                p.hot = true;
                ++stats_.promotions;
                --stats_.cold;
                ++stats_.hot;
                if (p.armed) {
                    guard_.disarm(address);
                    p.armed = false;
                }
            }
            p.history = std::uint16_t(p.history << 1);
        }
    }

    const Stats& stats() const { return stats_; }
    bool hot(std::uintptr_t page) const {
        const auto it = pages_.find(page & ~(kPage - 1));
        return it != pages_.end() && it->second.hot;
    }

private:
    struct Page {
        std::uint16_t history = 0;  // bit 0: this frame
        bool hot = false, probing = false, armed = false, written = false;
    };

    void arm(std::uintptr_t address, Page& p) {
        ++stats_.arms;
        p.armed = guard_.arm(address);
        if (!p.armed) ++stats_.arm_failures;  // copied every time, as a hot page
    }

    Guard& guard_;
    std::map<std::uintptr_t, Page> pages_;
    std::uint64_t frame_ = 0;
    Stats stats_;
};

}  // namespace leonida
