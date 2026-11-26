#pragma once
// What each buffer V# of a console resource heap's GPU copy resolved to (docs/d3d12_vulkan.md section 3.4), kept in
// host RAM so that a change of the address space re-fixes exactly the slots it touches.
//
// A slot of the copy is the title's descriptor with its V# bases rewritten (DeviceState::fix_console_slot). What a
// base became depends on things that change after the slot was written: a buffer or window released, an alias range
// removed, a geometry copy retired, or (for a base nothing backed yet) a buffer, alias or title memory added. Without
// a record of which slot resolved into which range, a change could only be applied by walking every live slot. Here
// each V# half (the buffer V# and the UAV counter V# of a slot) records:
//   - source: the bytes the title's V# names, [base, base + span) (the base is the V#'s 48 bits);
//   - target: the device addresses its GPU copy names, [target, target + span); for a half that resolved to nothing,
//     those of a copy that was found but refused (not backed yet), or none;
//   - state: resolved, waiting (written with no base and no records until something backs it) or none.
//
// Two indexes, by source and by target, hold the halves by their exact range. The halves that name the same range
// share one group (the title writes the same V# into thousands of slots), which keeps them in two lists, resolved and
// waiting, with each half's place in its list, so a half moves or goes in O(1) and no list holds a stale entry. The
// groups are ordered by start, separately for each power-of-two class of their length: a group of class c (length at
// most 2^c) overlaps [lo, hi) only if it starts in (lo - 2^c, hi), so a query searches each class in use once and walks
// only the groups that start there.
//
// The bound: a change visits exactly the halves whose recorded range overlaps it (a release or removal: resolved and
// waiting ones; an addition: waiting ones only); looked_at() counts those and nothing else. The groups walked to find
// them are the overlapping ones plus, per class, those that start less than 2^c before the change and end before it:
// each costs one comparison, never a slot. Memory: 20 bytes a half (40 a slot, allocated with calloc, so pages of
// slots never written are never touched), and per distinct range a group (two lists of 4 bytes a half, a hash entry
// and an ordered-map entry).
// Portable, standard library only: console_heap_slot_records_test exercises it without a GPU.
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace leonida {

// One change since the last submission, as a range; hi is exclusive and greater than lo.
struct SlotChange {
    enum Kind : std::uint8_t {
        target_gone,   // device addresses: a backing released, a copy's block destroyed, an alias range's copy gone
        source_gone,   // V# bases: an alias range removed or replaced, or a copy that answered for it gone
        target_added,  // device addresses: a backing added (a refused alias target may be backed now)
        source_added,  // V# bases: an alias range or title memory added, or a backing added at those arena offsets
    };
    Kind kind;
    std::uint64_t lo, hi;
};

class ConsoleHeapSlotRecords {
public:
    enum State : std::uint8_t { none = 0, resolved = 1, waiting = 2 };

    void reset(std::size_t slots) {
        slots_ = slots;
        // calloc: the pages of slots never written are never touched (the retail heap uses a tenth of its slots).
        halves_.reset(slots ? static_cast<Half*>(std::calloc(slots * 2, sizeof(Half))) : nullptr);
        if (slots && !halves_) slots_ = 0;
        for (auto& index : indexes_) index = Index{};
        groups_.clear();
        free_groups_.clear();
        recorded_ = waiting_ = 0;
        looked_at_ = groups_looked_at_ = 0;
    }
    std::size_t slots() const { return slots_; }
    std::size_t recorded() const { return recorded_; }      // halves in state resolved or waiting
    std::size_t waiting_count() const { return waiting_; }  // halves in state waiting
    std::size_t ranges() const { return groups_.size() - free_groups_.size(); }  // distinct ranges, both indexes
    std::uint64_t looked_at() const { return looked_at_; }  // halves visited by touched(), all calls
    std::uint64_t ranges_looked_at() const { return groups_looked_at_; }  // groups compared by touched(), all calls

    // Half `half` (0: the buffer V#, 1: the UAV counter V#) of `slot` now names [base, base + span) and resolved as
    // `state` says, its copy at `target` (0: none; see above). span 0 counts as 1.
    void set(std::size_t slot, unsigned half, State state, std::uint64_t base, std::uint64_t span,
             std::uint64_t target) {
        if (slot >= slots_ || half > 1) return;
        const auto key = std::uint32_t(slot * 2 + half);
        Half& h = halves_[key];
        if (state == none) {
            if (h.state != none) {
                for (int which = 0; which < 2; ++which) detach(which, key);
                count(State(h.state), -1);
                h.state = none;
            }
            return;
        }
        if (!span) span = 1;
        const Range ranges[2] = {range(base, span), target ? range(target, span) : Range{}};
        for (int which = 0; which < 2; ++which) {
            const std::uint32_t group = h.group[which];
            const bool keep = group && h.state == state && groups_[group - 1].start == ranges[which].start &&
                              groups_[group - 1].end == ranges[which].end;
            if (keep) continue;
            detach(which, key);
            if (ranges[which].end) attach(which, key, ranges[which], state);
        }
        count(State(h.state), -1);
        h.state = state;
        count(state, +1);
    }

    void clear(std::size_t slot) {
        set(slot, 0, none, 0, 0, 0);
        set(slot, 1, none, 0, 0, 0);
    }
    void clear_range(std::size_t first, std::size_t end) {
        end = (std::min)(end, slots_);
        for (std::size_t slot = first; slot < end; ++slot)
            for (unsigned half = 0; half < 2; ++half)
                if (halves_[slot * 2 + half].state != none) set(slot, half, none, 0, 0, 0);
    }

    State state(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? State(halves_[slot * 2 + half].state) : none;
    }
    std::uint64_t target(std::size_t slot, unsigned half) const {
        if (slot >= slots_ || half > 1) return 0;
        const std::uint32_t group = halves_[slot * 2 + half].group[kTarget];
        return group ? groups_[group - 1].start : 0;
    }

    // The slots `changes` touch, each once, ascending, appended to `out` (cleared first):
    //   target_gone: halves whose [target, target + span) overlaps the range (whatever their state);
    //   source_gone: halves whose [base, base + span) overlaps it;
    //   target_added, source_added: the same, waiting halves only (a resolved one stays valid until something goes).
    void touched(const SlotChange* changes, std::size_t count, std::vector<std::uint32_t>& out) {
        out.clear();
        for (std::size_t i = 0; i < count; ++i) {
            const SlotChange& change = changes[i];
            if (change.hi <= change.lo) continue;
            const bool target = change.kind == SlotChange::target_gone || change.kind == SlotChange::target_added;
            const bool waiting_only = change.kind == SlotChange::target_added || change.kind == SlotChange::source_added;
            find(target ? kTarget : kSource, change.lo, change.hi, waiting_only, out);
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

private:
    static constexpr int kSource = 0, kTarget = 1;
    static constexpr std::uint64_t kMaxSpan = (std::uint64_t(1) << 56) - 1;
    static constexpr unsigned kClasses = 65;

    struct Range {
        std::uint64_t start = 0, end = 0;  // end exclusive, saturated at 2^64 - 1; end 0: no range
        bool operator==(const Range&) const = default;
    };
    struct RangeHash {
        std::size_t operator()(const Range& r) const noexcept {
            return std::hash<std::uint64_t>{}(r.start ^ (r.end * 0x9e3779b97f4a7c15ull));
        }
    };
    // Zero-initialised by calloc: no group (ids are 1-based), state none.
    struct Half {
        std::uint32_t group[2];  // the source and the target group, id + 1, 0: none
        std::uint32_t place[2];  // the half's place in that group's list for its state
        std::uint8_t state;
    };
    struct Free {
        void operator()(Half* p) const { std::free(p); }
    };
    using ByStart = std::multimap<std::uint64_t, std::uint32_t>;  // start -> group id
    struct Group {
        std::uint64_t start = 0, end = 0;
        std::vector<std::uint32_t> keys[2];  // [0] resolved halves, [1] waiting halves
        unsigned cls = 0;
        ByStart::iterator at{};
    };
    struct Index {
        std::unordered_map<Range, std::uint32_t, RangeHash> by_range;  // -> group id
        std::array<ByStart, kClasses> by_start;
        std::uint64_t used[2] = {};  // a bit per class that holds a group (classes 0-63, 64)
    };

    static Range range(std::uint64_t start, std::uint64_t span) {
        if (span > kMaxSpan) span = kMaxSpan;
        return {start, start + span < start ? UINT64_MAX : start + span};
    }
    // The smallest c with length <= 2^c.
    static unsigned length_class(std::uint64_t length) {
        return length <= 1 ? 0u : unsigned(64 - std::countl_zero(length - 1));
    }
    static int list_of(State state) { return state == waiting ? 1 : 0; }

    void count(State state, int delta) {
        if (state != none) recorded_ += std::size_t(std::ptrdiff_t(delta));
        if (state == waiting) waiting_ += std::size_t(std::ptrdiff_t(delta));
    }

    void attach(int which, std::uint32_t key, Range r, State state) {
        Index& index = indexes_[which];
        auto [it, made] = index.by_range.try_emplace(r, 0);
        if (made) {
            std::uint32_t id;
            if (!free_groups_.empty()) {
                id = free_groups_.back();
                free_groups_.pop_back();
            } else {
                id = std::uint32_t(groups_.size());
                groups_.emplace_back();
            }
            Group& g = groups_[id];
            g.start = r.start;
            g.end = r.end;
            g.cls = length_class(r.end - r.start);
            g.at = index.by_start[g.cls].emplace(r.start, id);
            index.used[g.cls >> 6] |= std::uint64_t(1) << (g.cls & 63);
            it->second = id;
        }
        Group& g = groups_[it->second];
        auto& list = g.keys[list_of(state)];
        Half& h = halves_[key];
        h.group[which] = it->second + 1;
        h.place[which] = std::uint32_t(list.size());
        list.push_back(key);
    }

    void detach(int which, std::uint32_t key) {
        Half& h = halves_[key];
        if (!h.group[which]) return;
        const std::uint32_t id = h.group[which] - 1;
        Group& g = groups_[id];
        auto& list = g.keys[list_of(State(h.state))];
        const std::uint32_t place = h.place[which], last = list.back();
        list[place] = last;
        halves_[last].place[which] = place;
        list.pop_back();
        h.group[which] = 0;
        if (!g.keys[0].empty() || !g.keys[1].empty()) return;
        Index& index = indexes_[which];
        index.by_range.erase(Range{g.start, g.end});
        index.by_start[g.cls].erase(g.at);
        if (index.by_start[g.cls].empty()) index.used[g.cls >> 6] &= ~(std::uint64_t(1) << (g.cls & 63));
        for (auto& keys : g.keys) std::vector<std::uint32_t>().swap(keys);
        free_groups_.push_back(id);
    }

    void find(int which, std::uint64_t lo, std::uint64_t hi, bool waiting_only, std::vector<std::uint32_t>& out) {
        const Index& index = indexes_[which];
        for (unsigned word = 0; word < 2; ++word)
            for (std::uint64_t used = index.used[word]; used; used &= used - 1) {
                const unsigned c = word * 64 + unsigned(std::countr_zero(used));
                const ByStart& groups = index.by_start[c];
                // A group of this class (length at most 2^c) overlaps [lo, hi) only if it starts in (lo - 2^c, hi).
                const std::uint64_t reach = c >= 64 ? UINT64_MAX : (std::uint64_t(1) << c);
                const std::uint64_t first = lo >= reach ? lo - reach + 1 : 0;
                for (auto it = groups.lower_bound(first); it != groups.end() && it->first < hi; ++it) {
                    const Group& g = groups_[it->second];
                    ++groups_looked_at_;
                    if (g.end <= lo) continue;
                    for (int list = waiting_only ? 1 : 0; list < 2; ++list)
                        for (const std::uint32_t key : g.keys[list]) {
                            ++looked_at_;
                            out.push_back(key >> 1);
                        }
                }
            }
    }

    std::size_t slots_ = 0;
    std::unique_ptr<Half[], Free> halves_;
    std::array<Index, 2> indexes_;
    std::vector<Group> groups_;
    std::vector<std::uint32_t> free_groups_;
    std::size_t recorded_ = 0, waiting_ = 0;
    std::uint64_t looked_at_ = 0, groups_looked_at_ = 0;
};

}  // namespace leonida
