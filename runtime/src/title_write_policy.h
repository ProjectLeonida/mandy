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
// A page moves by its history, one bit a submission interval (frame() ends one), keeping O(dirty + hot + recent)
// active lists (never walking quiet cold pages): written in at least kPromote of the last kHistory submissions, and it
// turns hot at the end of the interval; a hot page is guarded again for one interval every kProbe intervals, and turns
// cold when such a probe interval sees no write. Measured under Wine on this machine: a 64 KiB copy is ~10 us, a fault
// 2-3 us with a handler that reopens in one call and a re-arm 4 us, so a page written in under half the submissions
// (including 1/frame pages) is cheaper cold; kPromote = 6 of 8 keeps only truly intra-frame hot pages uncovered and
// avoids copying quiet pages on empty submissions.
//
// Nothing is missed in between: a page is armed before it is reported dirty (collect() arms, then names it), so a
// write during the owner's copy faults and is in the next collect(); a page that turns hot is named at every collect()
// from then on, and one that turns cold stays armed (or returns to the dirty list to be armed and copied once more).
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

    struct Stats {
        std::uint64_t faults = 0;          // written() on a guarded page
        std::uint64_t hot_copies = 0;      // pages named because they are hot
        std::uint64_t cold_copies = 0;     // pages named because they were written while guarded
        std::uint64_t arms = 0, arm_failures = 0, promotions = 0, demotions = 0;
        std::size_t hot = 0, cold = 0;     // pages in each set now
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
        if (!p.hot) {
            add_dirty(page, p);
            add_history(page, p);
        }
    }

    // Before a submission: every page the owner must copy, each once, after it was armed again where it is cold, so a
    // write from here on is in the next collect(). `copy(page)` is called in address order.
    // Runs in O(hot + dirty) without walking untouched cold pages.
    template<class Copy>
    void collect(Copy&& copy) {
        to_copy_.clear();
        to_arm_.clear();
        for (const PageRef ref : hot_list_) {
            Page& p = *ref.page;
            if (!p.probe) {
                ++stats_.hot_copies;
                to_copy_.push_back(ref.addr);
                continue;
            }
            if (!p.written && p.armed) continue;  // guarded all along and not written: the copy is current
            if (!p.armed) {
                to_arm_.push_back(ref);
            } else if (p.probe == 1) {
                p.probe = 2;
            }
            p.written = false;
            ++stats_.hot_copies;
            to_copy_.push_back(ref.addr);
        }
        for (std::size_t i = 0; i < dirty_list_.size();) {
            const PageRef ref = dirty_list_[i];
            Page& p = *ref.page;
            if (p.hot || (!p.written && p.armed)) {
                erase_dirty_at(i);
                continue;
            }
            // Written, or not guarded since the last copy (a write then faulted nowhere): armed, then copied.
            p.written = false;
            ++stats_.cold_copies;
            to_copy_.push_back(ref.addr);
            if (!p.armed) {
                to_arm_.push_back(ref);
                ++i;
            } else {
                erase_dirty_at(i);
            }
        }
        if (!to_arm_.empty()) {
            std::sort(to_arm_.begin(), to_arm_.end(),
                      [](const PageRef& a, const PageRef& b) { return a.addr < b.addr; });
            to_arm_addrs_.resize(to_arm_.size());
            for (std::size_t k = 0; k < to_arm_.size(); ++k) to_arm_addrs_[k] = to_arm_[k].addr;
            arm_results_.assign(to_arm_.size(), 0);
            arm_pages(to_arm_addrs_.data(), arm_results_.data(), to_arm_.size());
            for (std::size_t k = 0; k < to_arm_.size(); ++k) {
                Page& p = *to_arm_[k].page;
                p.armed = arm_results_[k] != 0;
                if (!p.armed) {
                    ++stats_.arm_failures;  // copied every time, as a hot page
                } else {
                    if (p.hot && p.probe == 1) p.probe = 2;
                    if (!p.hot) erase_dirty(p);  // clean and guarded: leaves the dirty list
                }
            }
        }
        std::sort(to_copy_.begin(), to_copy_.end());
        for (const std::uintptr_t address : to_copy_) copy(address);
    }

    // A submission interval ended: histories move on, and submission-hot and cold pages change sets.
    // Runs in O(hot + recently-written cold) without walking quiet cold pages.
    void frame() {
        ++frame_;
        const bool probe = frame_ % kProbe == 0;
        demoted_.clear();
        to_disarm_.clear();
        for (std::size_t i = 0; i < hot_list_.size();) {
            const PageRef ref = hot_list_[i];
            Page& p = *ref.page;
            bool demoted = false;
            if (p.probe == 2) {
                p.probe = 3;  // armed at the collect just before: the interval from here is observed
            } else if (p.probe == 3) {
                // The interval after the arming saw no write: no longer submission-hot (stays armed).
                p.probe = 0;
                if (!(p.history & 1) && p.armed) {
                    demoted = true;
                    p.hot = false;
                } else {
                    // Written during probe (fault handler may already have cleared p.armed): re-establish hot
                    // ownership in Guard unconditionally so CellGuard::hot stays synchronized.
                    to_disarm_.push_back(ref.addr);
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
                if (p.written || !p.armed) add_dirty(ref.addr, p);
                if (p.history != 0) demoted_.push_back(ref);
            } else {
                ++i;
            }
        }
        for (std::size_t i = 0; i < history_active_.size();) {
            const PageRef ref = history_active_[i];
            Page& p = *ref.page;
            if (p.hot) {
                erase_history_at(i);
                continue;
            }
            if (unsigned(std::popcount(unsigned(p.history & kHistoryMask))) >= kPromote) {
                p.hot = true;
                p.probe = 0;
                --stats_.cold;
                ++stats_.hot;
                ++stats_.promotions;
                add_hot(ref.addr, p);
                erase_dirty(p);
                // Re-establish hot ownership in Guard unconditionally (even if written() already cleared p.armed).
                to_disarm_.push_back(ref.addr);
                p.armed = false;
                p.history = std::uint16_t((p.history << 1) & kHistoryMask);
                erase_history_at(i);
            } else {
                p.history = std::uint16_t((p.history << 1) & kHistoryMask);
                if (p.history == 0) erase_history_at(i);
                else ++i;
            }
        }
        if (!to_disarm_.empty()) disarm_pages(to_disarm_.data(), to_disarm_.size());
        for (const PageRef ref : demoted_) add_history(ref.addr, *ref.page);
        demoted_.clear();
    }

    const Stats& stats() const noexcept { return stats_; }
    bool has_pending_work() const noexcept {
        return !hot_list_.empty() || !dirty_list_.empty() || !history_active_.empty();
    }
    std::size_t hot_count() const noexcept { return hot_list_.size(); }
    std::size_t dirty_count() const noexcept { return dirty_list_.size(); }
    std::size_t history_count() const noexcept { return history_active_.size(); }
    bool hot(std::uintptr_t page) const {
        const auto it = pages_.find(page & ~(kPage - 1));
        return it != pages_.end() && it->second.hot;
    }

private:
    static constexpr std::uint16_t kHistoryMask = std::uint16_t((1u << kHistory) - 1);
    static constexpr std::uint32_t kNoIndex = ~std::uint32_t(0);

    struct Page {
        std::uint32_t hot_index = kNoIndex;      // index in hot_list_ when hot
        std::uint32_t dirty_index = kNoIndex;    // index in dirty_list_ when cold and needing arm/copy
        std::uint32_t history_index = kNoIndex;  // index in history_active_ when cold with non-zero history
        std::uint16_t history = 0;               // bit 0: this submission interval
        std::uint8_t probe = 0;                  // 1 asked, 2 armed at last collect, 3 observed
        bool hot = false;
        bool armed = false;
        bool written = false;
    };

    struct PageRef {
        std::uintptr_t addr = 0;
        Page* page = nullptr;
    };

    void add_hot(std::uintptr_t address, Page& p) {
        if (p.hot_index != kNoIndex) return;
        p.hot_index = std::uint32_t(hot_list_.size());
        hot_list_.push_back({address, &p});
    }
    void erase_hot_at(std::size_t idx) {
        const PageRef removed = hot_list_[idx];
        const PageRef last = hot_list_.back();
        hot_list_[idx] = last;
        if (last.page != removed.page) last.page->hot_index = std::uint32_t(idx);
        hot_list_.pop_back();
        removed.page->hot_index = kNoIndex;
    }
    void erase_hot(Page& p) {
        if (p.hot_index != kNoIndex) erase_hot_at(p.hot_index);
    }

    void add_dirty(std::uintptr_t address, Page& p) {
        if (p.dirty_index != kNoIndex) return;
        p.dirty_index = std::uint32_t(dirty_list_.size());
        dirty_list_.push_back({address, &p});
    }
    void erase_dirty_at(std::size_t idx) {
        const PageRef removed = dirty_list_[idx];
        const PageRef last = dirty_list_.back();
        dirty_list_[idx] = last;
        if (last.page != removed.page) last.page->dirty_index = std::uint32_t(idx);
        dirty_list_.pop_back();
        removed.page->dirty_index = kNoIndex;
    }
    void erase_dirty(Page& p) {
        if (p.dirty_index != kNoIndex) erase_dirty_at(p.dirty_index);
    }

    void add_history(std::uintptr_t address, Page& p) {
        if (p.history_index != kNoIndex) return;
        p.history_index = std::uint32_t(history_active_.size());
        history_active_.push_back({address, &p});
    }
    void erase_history_at(std::size_t idx) {
        const PageRef removed = history_active_[idx];
        const PageRef last = history_active_.back();
        history_active_[idx] = last;
        if (last.page != removed.page) last.page->history_index = std::uint32_t(idx);
        history_active_.pop_back();
        removed.page->history_index = kNoIndex;
    }
    void erase_history(Page& p) {
        if (p.history_index != kNoIndex) erase_history_at(p.history_index);
    }

    typename std::unordered_map<std::uintptr_t, Page>::iterator
    erase_page(typename std::unordered_map<std::uintptr_t, Page>::iterator it) {
        Page& p = it->second;
        if (p.armed) guard_.disarm(it->first);
        const bool was_hot = p.hot;
        erase_hot(p);
        erase_dirty(p);
        erase_history(p);
        --(was_hot ? stats_.hot : stats_.cold);
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

    void disarm_pages(const std::uintptr_t* pages, std::size_t count) {
        if constexpr (requires(Guard& g, const std::uintptr_t* p, std::size_t n) {
                          g.disarm_batch(p, n);
                      }) {
            guard_.disarm_batch(pages, count);
        } else {
            for (std::size_t i = 0; i < count; ++i) guard_.disarm(pages[i]);
        }
    }

    Guard& guard_;
    std::unordered_map<std::uintptr_t, Page> pages_;
    std::vector<PageRef> hot_list_;
    std::vector<PageRef> dirty_list_;
    std::vector<PageRef> history_active_;
    std::vector<std::uintptr_t> to_copy_;
    std::vector<PageRef> to_arm_;
    std::vector<std::uintptr_t> to_arm_addrs_;
    std::vector<std::uint8_t> arm_results_;
    std::vector<std::uintptr_t> to_disarm_;
    std::vector<PageRef> demoted_;
    std::uint64_t frame_ = 0;
    Stats stats_;
};

}  // namespace leonida
