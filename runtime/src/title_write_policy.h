#pragma once
// Which pages of the title's memory a copy of them (the page mirror, src/title_mirror.h on branch page-mirror; the
// title texture refresh) must take again before a submission, and how each page is watched for that.
//
// Two ways to know a page was written, at two costs (runtime/test/guard_fault_bench.cpp measures them on the host):
//   - cold: the page is guarded (read-only); the title's first write faults, the handler opens it and tells the
//     policy (written()), and the page is copied before the next submission and guarded again. A page written once a
//     minute costs one fault a minute.
//   - hot: the page is left writable and copied before every submission, with no fault. A page the title writes in
//     most submissions costs a copy a submission instead of a fault and a re-arm; under Wine a fault and a re-arm
//     cost far more than copying 64 KiB.
// A page moves by its history across two tiers, keeping O(dirty + hot + recent) active lists (never walking quiet cold
// pages):
//   - submission-hot: one bit per submission interval (frame() ends one); written in at least kPromote of the last
//     kHistory submissions, and it turns hot (unguarded, copied every submission). Probed for one submission every
//     kProbe submissions and demoted when quiet.
//   - frame-hot: one bit per presented frame (present_frame() ends one); written in at least kFramePromote of the last
//     kFrameHistory presented frames, and it turns hot (unguarded, copied every submission). Probed for one full
//     presented frame every kFrameProbe frames (armed at the first collect after the probe starts, then observed across
//     the following full frame so writes before submission 0 are never missed) and demoted cleanly when quiet. A page
//     written once per frame across 30-45 submissions/frame promotes within 3-4 frames instead of taking perpetual
//     per-frame write faults and VirtualProtect re-arms, while genuinely cold pages stay guarded.
//
// Nothing is missed in between: a guarded page is armed before it is reported dirty (collect() arms, then names it), so
// a write during the owner's copy faults and is in the next collect(); a page that turns hot is named at every
// collect() from then on, and one that turns cold stays armed (or returns to the dirty list to be armed and copied once
// more).
//
// The unit is the guard's: kPage (64 KiB, the geometry watch's tile and the mirror's page). Guard is the host side:
//   bool arm(std::uintptr_t page);     // read-only now; false when it cannot be (then the page is copied every time)
//   void disarm(std::uintptr_t page);  // writable now (a page turning hot, or leaving the policy)
// Optional batched arming:
//   void arm_batch(const std::uintptr_t* pages, std::uint8_t* results, std::size_t count);  // results[i]: 1 armed, 0 failed
// When present, collect() sorts pages needing arming and passes them in one call so Guard can coalesce contiguous
// 64 KiB pages into a single VirtualProtect syscall.
// The fault handler opens the page itself and calls written(page). Not thread-safe: the owner serialises calls (the
// handler's written() through a lock-free queue of its own, or under the owner's lock).
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida {

template<class Guard>
class TitleWritePolicy {
public:
    static constexpr std::uintptr_t kPage = 0x10000;
    static constexpr unsigned kHistory = 8;       // submission intervals of history a page keeps
    static constexpr unsigned kPromote = 6;       // written in this many submission intervals: submission-hot
    static constexpr unsigned kProbe = 32;        // submission-hot page is guarded for one interval every this many
    static constexpr unsigned kFrameHistory = 4;  // presented frames of history a page keeps
    static constexpr unsigned kFramePromote = 3;  // written in this many presented frames: frame-hot
    static constexpr unsigned kFrameProbe = 16;   // frame-hot page is guarded for one full frame every this many

    struct Stats {
        std::uint64_t faults = 0;          // written() on a guarded page
        std::uint64_t hot_copies = 0;      // pages named because they are hot
        std::uint64_t cold_copies = 0;     // pages named because they were written while guarded
        std::uint64_t arms = 0, arm_failures = 0, promotions = 0, demotions = 0;
        std::uint64_t frame_promotions = 0, frame_demotions = 0;
        std::size_t hot = 0, cold = 0;     // pages in each set now (hot includes sub_hot and frame_hot)
        std::size_t frame_hot = 0;         // pages currently in the frame-hot tier
        std::uint64_t batched_arm_calls = 0;
    };

    explicit TitleWritePolicy(Guard& guard) : guard_(guard) {}

    // [begin, end) is to be copied from now on: its pages start cold and written (the owner has no copy yet).
    void add(std::uintptr_t begin, std::uintptr_t end) {
        if (begin >= end) return;
        for (std::uintptr_t p = begin & ~(kPage - 1); p < end;) {
            auto [it, fresh] = pages_.try_emplace(p);
            if (fresh) {
                it->second.written = true;
                it->second.armed = false;
                ++stats_.cold;
                add_dirty(p, it->second);
            }
            if (end - p <= kPage) break;
            p += kPage;
        }
    }
    // No longer copied: its pages are left writable.
    void remove(std::uintptr_t begin, std::uintptr_t end) {
        if (begin >= end || pages_.empty()) return;
        const std::uintptr_t first = begin & ~(kPage - 1);
        const std::uintptr_t span = end - first;
        const std::size_t count = std::size_t((span - 1) / kPage) + 1;
        if (count <= pages_.size()) {
            for (std::size_t i = 0; i < count; ++i) {
                const std::uintptr_t addr = first + std::uintptr_t(i) * kPage;
                const auto it = pages_.find(addr);
                if (it != pages_.end()) erase_page(it);
            }
        } else {
            for (auto it = pages_.begin(); it != pages_.end();) {
                if (it->first >= first && it->first < end) it = erase_page(it);
                else ++it;
            }
        }
    }

    // The handler opened a guarded page the title wrote (or a kernel write is about to): copy it before the next
    // submission.
    void written(std::uintptr_t page) {
        page &= ~(kPage - 1);
        const auto it = pages_.find(page);
        if (it == pages_.end()) return;
        Page& p = it->second;
        if (p.armed) ++stats_.faults;
        p.armed = false;
        p.written = true;
        p.history |= 1;
        p.frame_history |= 1;
        if (!p.is_hot()) {
            add_dirty(page, p);
            add_history(page, p);
            add_frame_history(page, p);
        } else {
            if (!p.sub_hot) add_history(page, p);
            if (!p.frame_hot) add_frame_history(page, p);
        }
    }

    // Before a submission: every page the owner must copy, each once, after it was armed again where it is cold, so a
    // write from here on is in the next collect(). `copy(page)` is called in address order.
    // Runs in O(hot + dirty) without walking untouched cold pages.
    template<class Copy>
    void collect(Copy&& copy) {
        to_copy_.clear();
        to_arm_.clear();
        for (const std::uintptr_t address : hot_list_) {
            Page& p = pages_.find(address)->second;
            if (p.frame_hot) {
                if (!p.frame_probe) {
                    ++stats_.hot_copies;
                    to_copy_.push_back(address);
                    continue;
                }
                if (p.frame_probe == 1) {
                    if (!p.armed) {
                        to_arm_.push_back(address);
                    } else {
                        p.frame_probe = 2;
                    }
                    p.written = false;
                    ++stats_.hot_copies;
                    to_copy_.push_back(address);
                    continue;
                }
                // frame_probe == 2 (armed during probe-start frame) or 3 (observing full probe frame).
                if (!p.written && p.armed) continue;  // guarded all along and not written: copy is current
                // Written (or unarmed) during probe: still active this frame -> disarm immediately so the rest of the
                // frame takes zero additional write faults.
                p.frame_probe = 0;
                p.probe = 0;
                p.written = false;
                guard_.disarm(address);
                p.armed = false;
                ++stats_.hot_copies;
                to_copy_.push_back(address);
                continue;
            }
            if (!p.probe) {
                ++stats_.hot_copies;
                to_copy_.push_back(address);
                continue;
            }
            if (!p.written && p.armed) continue;  // guarded all along and not written: the copy is current
            if (!p.armed) {
                to_arm_.push_back(address);
            } else if (p.probe == 1) {
                p.probe = 2;
            }
            p.written = false;
            ++stats_.hot_copies;
            to_copy_.push_back(address);
        }
        for (std::size_t i = 0; i < dirty_list_.size();) {
            const std::uintptr_t address = dirty_list_[i];
            Page& p = pages_.find(address)->second;
            if (p.is_hot() || (!p.written && p.armed)) {
                erase_dirty_at(i);
                continue;
            }
            // Written, or not guarded since the last copy (a write then faulted nowhere): armed, then copied.
            p.written = false;
            ++stats_.cold_copies;
            to_copy_.push_back(address);
            if (!p.armed) {
                to_arm_.push_back(address);
                ++i;
            } else {
                erase_dirty_at(i);
            }
        }
        if (!to_arm_.empty()) {
            std::sort(to_arm_.begin(), to_arm_.end());
            arm_results_.assign(to_arm_.size(), 0);
            arm_pages(to_arm_.data(), arm_results_.data(), to_arm_.size());
            for (std::size_t k = 0; k < to_arm_.size(); ++k) {
                const std::uintptr_t address = to_arm_[k];
                Page& p = pages_.find(address)->second;
                p.armed = arm_results_[k] != 0;
                if (!p.armed) {
                    ++stats_.arm_failures;  // copied every time, as a hot page
                } else {
                    if (p.frame_hot && p.frame_probe == 1) p.frame_probe = 2;
                    else if (p.sub_hot && p.probe == 1) p.probe = 2;
                    if (!p.is_hot()) erase_dirty(p);  // clean and guarded: leaves the dirty list
                }
            }
        }
        std::sort(to_copy_.begin(), to_copy_.end());
        for (const std::uintptr_t address : to_copy_) copy(address);
    }

    // A submission interval ended: submission histories move on, and submission-hot pages change sets. Frame-hot pages
    // are never probed or demoted across intra-frame submissions (present_frame() governs their lifecycle).
    // Runs in O(hot + recently-written cold) without walking quiet cold pages.
    void frame() {
        ++frame_;
        const bool probe = frame_ % kProbe == 0;
        demoted_.clear();
        for (std::size_t i = 0; i < hot_list_.size();) {
            const std::uintptr_t address = hot_list_[i];
            Page& p = pages_.find(address)->second;
            if (p.frame_hot) {
                p.history = std::uint16_t((p.history << 1) & kHistoryMask);
                ++i;
                continue;
            }
            bool demoted = false;
            if (p.probe == 2) {
                p.probe = 3;  // armed at the collect just before: the interval from here is observed
            } else if (p.probe == 3) {
                // The interval after the arming saw no write: no longer submission-hot (stays armed).
                p.probe = 0;
                if (!(p.history & 1) && p.armed) {
                    demoted = true;
                    p.sub_hot = false;
                } else {
                    // Written during probe (fault handler may already have cleared p.armed): re-establish hot
                    // ownership in Guard unconditionally so CellGuard::hot stays synchronized.
                    guard_.disarm(address);
                    p.armed = false;
                }
            } else if (probe && !p.probe) {
                p.probe = 1;       // guarded at the next collect()
                p.written = true;  // and copied once more after the arming
            }
            p.history = std::uint16_t((p.history << 1) & kHistoryMask);
            if (demoted) {
                --stats_.hot;
                ++stats_.cold;
                ++stats_.demotions;
                erase_hot_at(i);
                if (p.written || !p.armed) add_dirty(address, p);
                if (p.history != 0) demoted_.push_back(address);
            } else {
                ++i;
            }
        }
        for (std::size_t i = 0; i < history_active_.size();) {
            const std::uintptr_t address = history_active_[i];
            Page& p = pages_.find(address)->second;
            if (p.sub_hot) {
                erase_history_at(i);
                continue;
            }
            if (unsigned(std::popcount(unsigned(p.history & kHistoryMask))) >= kPromote) {
                p.sub_hot = true;
                p.probe = 0;
                if (p.hot_index == kNoIndex) {
                    --stats_.cold;
                    ++stats_.hot;
                    ++stats_.promotions;
                    add_hot(address, p);
                    erase_dirty(p);
                }
                // Re-establish hot ownership in Guard unconditionally (even if written() already cleared p.armed).
                guard_.disarm(address);
                p.armed = false;
                p.history = std::uint16_t((p.history << 1) & kHistoryMask);
                erase_history_at(i);
            } else {
                p.history = std::uint16_t((p.history << 1) & kHistoryMask);
                if (p.history == 0) erase_history_at(i);
                else ++i;
            }
        }
        for (const std::uintptr_t address : demoted_) add_history(address, pages_.find(address)->second);
        demoted_.clear();
    }

    // A presented frame ended: frame histories move on, pages written in >= kFramePromote of the last kFrameHistory
    // presented frames promote to frame-hot (unguarded), and frame-hot pages are probed for one full frame every
    // kFrameProbe presented frames so idle pages demote cleanly back to cold.
    // Runs in O(frame_hot + recently-written) without walking quiet cold pages.
    void present_frame() {
        ++presented_frames_;
        const bool probe = presented_frames_ % kFrameProbe == 0;
        frame_demoted_.clear();
        for (std::size_t i = 0; i < frame_hot_list_.size();) {
            const std::uintptr_t address = frame_hot_list_[i];
            Page& p = pages_.find(address)->second;
            bool demoted = false;
            if (p.frame_probe == 2) {
                p.frame_probe = 3;  // armed before this frame starts: observe the entire upcoming frame
            } else if (p.frame_probe == 3) {
                p.frame_probe = 0;
                if (!(p.frame_history & 1) && p.armed) {
                    // An entire presented frame passed with zero writes: demote cleanly to cold (stays armed).
                    demoted = true;
                    p.frame_hot = false;
                    p.sub_hot = false;
                    p.probe = 0;
                } else {
                    guard_.disarm(address);
                    p.armed = false;
                }
            } else if (probe && !p.frame_probe) {
                p.frame_probe = 1;  // guarded at the next collect()
                p.written = true;   // and copied once more after the arming
            }
            p.frame_history = std::uint8_t((p.frame_history << 1) & kFrameHistoryMask);
            if (demoted) {
                ++stats_.frame_demotions;
                erase_frame_hot_at(i);
                if (p.hot_index != kNoIndex) {
                    --stats_.hot;
                    ++stats_.cold;
                    ++stats_.demotions;
                    erase_hot(p);
                }
                if (p.written || !p.armed) add_dirty(address, p);
                if (p.frame_history != 0) frame_demoted_.push_back(address);
            } else {
                ++i;
            }
        }
        for (std::size_t i = 0; i < frame_history_active_.size();) {
            const std::uintptr_t address = frame_history_active_[i];
            Page& p = pages_.find(address)->second;
            if (p.frame_hot) {
                erase_frame_history_at(i);
                continue;
            }
            if (unsigned(std::popcount(unsigned(p.frame_history & kFrameHistoryMask))) >= kFramePromote) {
                p.frame_hot = true;
                p.frame_probe = 0;
                p.probe = 0;
                add_frame_hot(address, p);
                ++stats_.frame_promotions;
                if (p.hot_index == kNoIndex) {
                    --stats_.cold;
                    ++stats_.hot;
                    ++stats_.promotions;
                    add_hot(address, p);
                    erase_dirty(p);
                }
                guard_.disarm(address);
                p.armed = false;
                p.frame_history = std::uint8_t((p.frame_history << 1) & kFrameHistoryMask);
                erase_frame_history_at(i);
            } else {
                p.frame_history = std::uint8_t((p.frame_history << 1) & kFrameHistoryMask);
                if (p.frame_history == 0) erase_frame_history_at(i);
                else ++i;
            }
        }
        for (const std::uintptr_t address : frame_demoted_) add_frame_history(address, pages_.find(address)->second);
        frame_demoted_.clear();
        stats_.frame_hot = frame_hot_list_.size();
    }

    const Stats& stats() const noexcept { return stats_; }
    bool has_pending_work() const noexcept {
        return !hot_list_.empty() || !dirty_list_.empty() || !history_active_.empty();
    }
    bool has_frame_work() const noexcept { return !frame_hot_list_.empty() || !frame_history_active_.empty(); }
    std::size_t hot_count() const noexcept { return hot_list_.size(); }
    std::size_t dirty_count() const noexcept { return dirty_list_.size(); }
    std::size_t history_count() const noexcept { return history_active_.size(); }
    std::size_t frame_hot_count() const noexcept { return frame_hot_list_.size(); }
    std::size_t frame_history_count() const noexcept { return frame_history_active_.size(); }
    bool hot(std::uintptr_t page) const {
        const auto it = pages_.find(page & ~(kPage - 1));
        return it != pages_.end() && it->second.is_hot();
    }
    bool frame_hot(std::uintptr_t page) const {
        const auto it = pages_.find(page & ~(kPage - 1));
        return it != pages_.end() && it->second.frame_hot;
    }

private:
    static constexpr std::uint16_t kHistoryMask = std::uint16_t((1u << kHistory) - 1);
    static constexpr std::uint8_t kFrameHistoryMask = std::uint8_t((1u << kFrameHistory) - 1);
    static constexpr std::uint32_t kNoIndex = ~std::uint32_t(0);

    struct Page {
        std::uint32_t hot_index = kNoIndex;         // index in hot_list_ when unguarded (sub_hot || frame_hot)
        std::uint32_t dirty_index = kNoIndex;       // index in dirty_list_ when cold and needing arm/copy
        std::uint32_t history_index = kNoIndex;     // index in history_active_ when non-sub_hot with non-zero history
        std::uint32_t frame_hot_index = kNoIndex;   // index in frame_hot_list_ when frame_hot
        std::uint32_t frame_hist_index = kNoIndex;  // index in frame_history_active_ when non-frame_hot with history
        std::uint16_t history = 0;                  // bit 0: this submission interval
        std::uint8_t frame_history = 0;             // bit 0: this presented frame
        std::uint8_t probe = 0;                     // sub_hot probe: 1 asked, 2 armed at last collect, 3 observed
        std::uint8_t frame_probe = 0;               // frame_hot probe: 1 asked, 2 armed, 3 observed full frame
        bool sub_hot = false;
        bool frame_hot = false;
        bool armed = false;
        bool written = false;

        bool is_hot() const noexcept { return sub_hot || frame_hot; }
    };

    void add_hot(std::uintptr_t address, Page& p) {
        if (p.hot_index != kNoIndex) return;
        p.hot_index = std::uint32_t(hot_list_.size());
        hot_list_.push_back(address);
    }
    void erase_hot_at(std::size_t idx) {
        const std::uintptr_t addr = hot_list_[idx];
        const std::uintptr_t last = hot_list_.back();
        hot_list_[idx] = last;
        if (last != addr) pages_.find(last)->second.hot_index = std::uint32_t(idx);
        hot_list_.pop_back();
        pages_.find(addr)->second.hot_index = kNoIndex;
    }
    void erase_hot(Page& p) {
        if (p.hot_index != kNoIndex) erase_hot_at(p.hot_index);
    }

    void add_dirty(std::uintptr_t address, Page& p) {
        if (p.dirty_index != kNoIndex) return;
        p.dirty_index = std::uint32_t(dirty_list_.size());
        dirty_list_.push_back(address);
    }
    void erase_dirty_at(std::size_t idx) {
        const std::uintptr_t addr = dirty_list_[idx];
        const std::uintptr_t last = dirty_list_.back();
        dirty_list_[idx] = last;
        if (last != addr) pages_.find(last)->second.dirty_index = std::uint32_t(idx);
        dirty_list_.pop_back();
        pages_.find(addr)->second.dirty_index = kNoIndex;
    }
    void erase_dirty(Page& p) {
        if (p.dirty_index != kNoIndex) erase_dirty_at(p.dirty_index);
    }

    void add_history(std::uintptr_t address, Page& p) {
        if (p.history_index != kNoIndex) return;
        p.history_index = std::uint32_t(history_active_.size());
        history_active_.push_back(address);
    }
    void erase_history_at(std::size_t idx) {
        const std::uintptr_t addr = history_active_[idx];
        const std::uintptr_t last = history_active_.back();
        history_active_[idx] = last;
        if (last != addr) pages_.find(last)->second.history_index = std::uint32_t(idx);
        history_active_.pop_back();
        pages_.find(addr)->second.history_index = kNoIndex;
    }
    void erase_history(Page& p) {
        if (p.history_index != kNoIndex) erase_history_at(p.history_index);
    }

    void add_frame_hot(std::uintptr_t address, Page& p) {
        if (p.frame_hot_index != kNoIndex) return;
        p.frame_hot_index = std::uint32_t(frame_hot_list_.size());
        frame_hot_list_.push_back(address);
    }
    void erase_frame_hot_at(std::size_t idx) {
        const std::uintptr_t addr = frame_hot_list_[idx];
        const std::uintptr_t last = frame_hot_list_.back();
        frame_hot_list_[idx] = last;
        if (last != addr) pages_.find(last)->second.frame_hot_index = std::uint32_t(idx);
        frame_hot_list_.pop_back();
        pages_.find(addr)->second.frame_hot_index = kNoIndex;
    }
    void erase_frame_hot(Page& p) {
        if (p.frame_hot_index != kNoIndex) erase_frame_hot_at(p.frame_hot_index);
    }

    void add_frame_history(std::uintptr_t address, Page& p) {
        if (p.frame_hist_index != kNoIndex) return;
        p.frame_hist_index = std::uint32_t(frame_history_active_.size());
        frame_history_active_.push_back(address);
    }
    void erase_frame_history_at(std::size_t idx) {
        const std::uintptr_t addr = frame_history_active_[idx];
        const std::uintptr_t last = frame_history_active_.back();
        frame_history_active_[idx] = last;
        if (last != addr) pages_.find(last)->second.frame_hist_index = std::uint32_t(idx);
        frame_history_active_.pop_back();
        pages_.find(addr)->second.frame_hist_index = kNoIndex;
    }
    void erase_frame_history(Page& p) {
        if (p.frame_hist_index != kNoIndex) erase_frame_history_at(p.frame_hist_index);
    }

    typename std::unordered_map<std::uintptr_t, Page>::iterator
    erase_page(typename std::unordered_map<std::uintptr_t, Page>::iterator it) {
        Page& p = it->second;
        if (p.armed) guard_.disarm(it->first);
        const bool was_hot = p.hot_index != kNoIndex;
        erase_hot(p);
        erase_dirty(p);
        erase_history(p);
        erase_frame_hot(p);
        erase_frame_history(p);
        --(was_hot ? stats_.hot : stats_.cold);
        stats_.frame_hot = frame_hot_list_.size();
        return pages_.erase(it);
    }

    void arm_pages(const std::uintptr_t* pages, std::uint8_t* results, std::size_t count) {
        stats_.arms += count;
        if constexpr (requires(Guard& g, const std::uintptr_t* p, std::uint8_t* r, std::size_t n) {
                          g.arm_batch(p, r, n);
                      }) {
            ++stats_.batched_arm_calls;
            guard_.arm_batch(pages, results, count);
        } else {
            for (std::size_t i = 0; i < count; ++i) results[i] = guard_.arm(pages[i]) ? 1 : 0;
        }
    }

    Guard& guard_;
    std::unordered_map<std::uintptr_t, Page> pages_;
    std::vector<std::uintptr_t> hot_list_;
    std::vector<std::uintptr_t> dirty_list_;
    std::vector<std::uintptr_t> history_active_;
    std::vector<std::uintptr_t> frame_hot_list_;
    std::vector<std::uintptr_t> frame_history_active_;
    std::vector<std::uintptr_t> to_copy_;
    std::vector<std::uintptr_t> to_arm_;
    std::vector<std::uint8_t> arm_results_;
    std::vector<std::uintptr_t> demoted_;
    std::vector<std::uintptr_t> frame_demoted_;
    std::uint64_t frame_ = 0;
    std::uint64_t presented_frames_ = 0;
    Stats stats_;
};

}  // namespace leonida
