#pragma once
// What each buffer V# and title-written T# of a console resource heap's GPU copy resolved to (docs/d3d12_vulkan.md
// section 3.4), kept in host RAM so that a change of the address space re-fixes exactly the slots it touches.
//
// A slot of the copy is the title's descriptor with its V# bases rewritten and its T# bound (DeviceState::
// fix_console_slot). What a base became depends on things that change after the slot was written: a buffer or window
// released, an alias range removed, a geometry copy retired, a texture registered, replaced or gone at an address, or
// (for a base nothing backed yet) a buffer, alias or title memory added. Without a record of which slot resolved into
// which range, a change could only be applied by walking every live slot. Here each V# half (the buffer V# and the
// UAV counter V# of a slot) records:
//   - source: the bytes the title's V# names, [base, base + span) (the base is the V#'s 48 bits);
//   - target: the device addresses its GPU copy names, [target, target + span); for a half that resolved to nothing,
//     those of a copy that was found but refused (not backed yet), or none;
//   - state: resolved, waiting (written with no base and no records until something backs it), waiting_room (as
//     waiting, and its geometry copy was refused for lack of room) or none.
// The T# of a slot has a record too (set_texture), in the first half: the address it names (its base), and whether a
// texture bound it or it waits for one (texture_waiting_room: refused because its bindless array was full). Title-
// written T#s are looked up by their exact base address (bind_title_image), so a third index holds them by that
// address alone.
//
// The three indexes (V# source, V# target, T# address) hold the records by their exact range (a T#'s is its address
// and the byte after it). The records that name the same range share one group (the title writes the same descriptor
// into thousands of slots), which keeps them in two lists, settled (resolved, bound) and waiting, with each record's
// place in its list, so a record moves or goes in O(1) and no list holds a stale entry. The groups are ordered by
// start, separately for each power-of-two class of their length: a group of class c (length at most 2^c) overlaps
// [lo, hi) only if it starts in (lo - 2^c, hi), so a query searches each class in use once and walks only the groups
// that start there. Records waiting for room are also kept in one list per kind (V# halves, T#s), with their place,
// which a change saying room was made visits whole.
//
// The bound: a change visits exactly the records it concerns - a release or removal the V# halves whose recorded
// range overlaps it, an addition the waiting ones whose range starts inside it (a V# resolves only through a range
// that holds its base or its refused target, so an addition that does not hold the start cannot settle it; the
// groups with waiting records are kept by start a second time for this), a texture gone the T#s naming its address,
// a texture added the waiting ones, room made the records waiting for that room - and each of them once an index a
// touched() call (a V# half at most twice: by source and by target), however many of the call's changes name it.
// looked_at() counts those and nothing else. The groups walked to
// find them are the ones visited plus, for a removal, per class, those that start less than 2^c before the change and
// end before it: each costs one comparison, never a slot. Memory: 32 bytes a half (64 a slot,
// allocated with calloc, so pages of slots never written are never touched), and per distinct range a group (two lists
// of 4 bytes a record, a hash entry and an ordered-map entry).
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
        target_gone,    // device addresses: a backing released, a copy's block destroyed, an alias range's copy gone
        source_gone,    // V# bases: an alias range removed or replaced, or a copy that answered for it gone
        target_added,   // device addresses: a backing added (a refused alias target may be backed now)
        source_added,   // V# bases: an alias range or title memory added, or a backing added at those arena offsets
        texture_gone,   // a T# address (lo; hi = lo + 1): the texture registered there went or was replaced
        texture_added,  // a T# address: a texture registered or replaced there, or its image published
        geometry_room,  // geometry copies were freed: halves refused a copy for lack of room may get one (no range)
        texture_room,   // bindless image entries were freed: T#s refused for a full array may bind (no range)
    };
    Kind kind;
    std::uint64_t lo, hi;
};

class ConsoleHeapSlotRecords {
public:
    // V# halves: resolved, waiting (for a backing over its base or target), waiting_room (and for geometry room).
    // T#s (the first half): texture_bound, texture_waiting (for a texture at its address), texture_waiting_room (and
    // for room in its bindless array).
    enum State : std::uint8_t {
        none = 0,
        resolved = 1,
        waiting = 2,
        waiting_room = 3,
        texture_bound = 4,
        texture_waiting = 5,
        texture_waiting_room = 6,
    };
    static bool is_vsharp(State s) { return s >= resolved && s <= waiting_room; }
    static bool is_texture(State s) { return s >= texture_bound; }
    static bool is_waiting(State s) { return s == waiting || s == waiting_room || s >= texture_waiting; }

    void reset(std::size_t slots) {
        slots_ = slots;
        // calloc: the pages of slots never written are never touched (the retail heap uses a tenth of its slots).
        halves_.reset(slots ? static_cast<Half*>(std::calloc(slots * 2, sizeof(Half))) : nullptr);
        if (slots && !halves_) slots_ = 0;
        for (auto& index : indexes_) index = Index{};
        for (auto& room : room_) room.clear();
        groups_.clear();
        free_groups_.clear();
        recorded_ = waiting_ = textures_count_ = textures_waiting_ = 0;
        looked_at_ = groups_looked_at_ = 0;
    }
    std::size_t slots() const { return slots_; }
    std::size_t recorded() const { return recorded_; }      // V# halves resolved or waiting
    std::size_t waiting_count() const { return waiting_; }  // V# halves waiting (for a backing or room)
    std::size_t textures() const { return textures_count_; }  // T# records, bound or waiting
    std::size_t textures_waiting() const { return textures_waiting_; }  // T#s waiting for a texture or room
    std::size_t ranges() const { return groups_.size() - free_groups_.size(); }  // distinct ranges, all indexes
    std::uint64_t looked_at() const { return looked_at_; }  // records visited by touched(), all calls
    std::uint64_t ranges_looked_at() const { return groups_looked_at_; }  // groups compared by touched(), all calls

    // Half `half` (0: the buffer V#, 1: the UAV counter V#) of `slot` now names [base, base + span) and resolved as
    // `state` says (none or a V# state), its copy at `target` (0: none; see above). span 0 counts as 1. `geometry`:
    // its resolution asked for a geometry copy (fix_vsharp's copy_geometry), which answer() tells apart.
    void set(std::size_t slot, unsigned half, State state, std::uint64_t base, std::uint64_t span,
             std::uint64_t target, bool geometry = false) {
        if (slot >= slots_ || half > 1 || is_texture(state)) return;
        const std::uint32_t key = std::uint32_t(slot * 2 + half);
        if (state == none) {
            if (halves_[key].state == none) return;
            drop(key);
            return;
        }
        if (!span) span = 1;
        const Range ranges[2] = {range(base, span), target ? range(target, span) : Range{}};
        place(key, state, geometry ? kGeometry : 0, base, ranges);
    }

    // The T# of `slot` names `address` (its base) and is bound or waits as `state` (a T# state) says. The slot's
    // second half is no V# then: its record goes.
    void set_texture(std::size_t slot, State state, std::uint64_t address) {
        if (slot >= slots_ || !is_texture(state)) return;
        const std::uint32_t k1 = std::uint32_t(slot * 2 + 1);
        if (halves_[k1].state != none) drop(k1);
        const Range ranges[2] = {range(address, 1), Range{}};
        place(std::uint32_t(slot * 2), state, 0, address, ranges);
    }

    // What a V# naming [base, base + span) resolves to while another half of the same range has a record made the
    // same way (`geometry`: both asked for a geometry copy, or neither): the same answer, `state` and `target` as that
    // half has them. Sound because a record is fixed again by every change that could alter its answer (touched): a
    // resolved one by every release or removal of its range or its copy, a waiting one also by every addition that
    // could back it. While it is current, the same lookups give the same answer. The caller applies the changes noted
    // since first (they may not be on a record yet). A waiting answer is also taken from a half that asked for a copy
    // when this one does not (it looked further and found nothing).
    bool answer(std::uint64_t base, std::uint64_t span, bool geometry, State& state, std::uint64_t& target) {
        if (!span) span = 1;
        const auto it = indexes_[kSource].by_range.find(range(base, span));
        if (it == indexes_[kSource].by_range.end()) return false;
        const Group& g = groups_[it->second];
        const int geo = geometry ? 1 : 0;
        const std::vector<std::uint32_t>* list =
            !g.keys[kSettled + geo].empty() ? &g.keys[kSettled + geo]
            : !g.keys[kWaiting + 1].empty() ? &g.keys[kWaiting + 1]
            : !geometry && !g.keys[kWaiting].empty() ? &g.keys[kWaiting] : nullptr;
        if (!list) return false;
        const Half& h = halves_[list->front()];
        // One that asks for no copy waits for a backing only, not for geometry room.
        state = !geometry && h.state == waiting_room ? waiting : State(h.state);
        target = h.group[kTarget] ? groups_[h.group[kTarget] - 1].start : 0;
        return true;
    }

    void clear(std::size_t slot) {
        if (slot >= slots_) return;
        const std::uint32_t k0 = std::uint32_t(slot * 2);
        if (halves_[k0].state != none) drop(k0);
        if (halves_[k0 + 1].state != none) drop(k0 + 1);
    }
    void clear_range(std::size_t first, std::size_t end) {
        end = (std::min)(end, slots_);
        for (std::size_t slot = first; slot < end; ++slot)
            for (unsigned half = 0; half < 2; ++half)
                if (halves_[slot * 2 + half].state != none) drop(std::uint32_t(slot * 2 + half));
    }

    std::uint64_t base(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? halves_[slot * 2 + half].base : 0;
    }
    State state(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? State(halves_[slot * 2 + half].state) : none;
    }
    std::uint64_t target(std::size_t slot, unsigned half) const {
        if (slot >= slots_ || half > 1) return 0;
        const Half& h = halves_[slot * 2 + half];
        return is_vsharp(State(h.state)) && h.group[1] ? groups_[h.group[1] - 1].start : 0;
    }

    // The slots `changes` touch, each once, ascending, appended to `out` (cleared first):
    //   target_gone: V# halves whose [target, target + span) overlaps the range (whatever their state);
    //   source_gone: V# halves whose [base, base + span) overlaps it;
    //   target_added, source_added: the same, waiting halves only (a resolved one stays valid until something goes);
    //   texture_gone: T#s naming the address lo, bound or waiting; texture_added: those waiting;
    //   geometry_room: V# halves waiting for room; texture_room: T#s waiting for room (each list once a call).
    //   An addition finds the waiting records whose range starts inside it: a V# resolves only through a backing,
    //   alias range, copy or title memory that holds its base (or a backing that holds its refused target), so one
    //   that does not hold the start cannot settle it. A removal finds every record whose range overlaps it.
    // A record is looked at once an index a call at most, however many of the changes name it.
    void touched(const SlotChange* changes, std::size_t count, std::vector<std::uint32_t>& out) {
        out.clear();
        if (!++epoch_) {  // after 2^32 calls: no group may look visited
            for (auto& g : groups_)
                for (auto& seen : g.seen) seen = 0;
            epoch_ = 1;
        }
        bool room[2] = {};
        for (std::size_t i = 0; i < count; ++i) {
            const SlotChange& change = changes[i];
            switch (change.kind) {
            case SlotChange::geometry_room: room[0] = true; continue;
            case SlotChange::texture_room: room[1] = true; continue;
            case SlotChange::texture_gone:
                if (change.lo != UINT64_MAX) find(kTexture, change.lo, change.lo + 1, out);
                continue;
            case SlotChange::texture_added:
                if (change.lo != UINT64_MAX) find_waiting(kTexture, change.lo, change.lo + 1, out);
                continue;
            default: break;
            }
            if (change.hi <= change.lo) continue;
            switch (change.kind) {
            case SlotChange::target_gone: find(kTarget, change.lo, change.hi, out); break;
            case SlotChange::source_gone: find(kSource, change.lo, change.hi, out); break;
            case SlotChange::target_added: find_waiting(kTarget, change.lo, change.hi, out); break;
            default: find_waiting(kSource, change.lo, change.hi, out); break;
            }
        }
        for (int which = 0; which < 2; ++which)
            if (room[which])
                for (const std::uint32_t key : room_[which]) {
                    ++looked_at_;
                    out.push_back(key >> 1);
                }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

private:
    static constexpr int kSource = 0, kTarget = 1, kTexture = 2;
    static constexpr std::uint8_t kGeometry = 1;  // Half::flags: the resolution asked for a geometry copy
    static constexpr int kSettled = 0, kWaiting = 2;  // Group::keys: + 1 for the records with kGeometry
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
    // Zero-initialised by calloc: no group (ids are 1-based), state none. A V# half is in the source index (group[0])
    // and, with a target, the target index (group[1]); a T# in the texture index (group[0]).
    struct Half {
        std::uint64_t base;      // the V#'s base or the T#'s address, as given
        std::uint32_t group[2];  // id + 1, 0: none
        std::uint32_t place[2];  // the record's place in that group's list for its state
        std::uint32_t room;      // its place in room_ while it waits for room
        std::uint8_t state;
        std::uint8_t flags;      // kGeometry
    };
    struct Free {
        void operator()(Half* p) const { std::free(p); }
    };
    using ByStart = std::multimap<std::uint64_t, std::uint32_t>;  // start -> group id
    struct Group {
        std::uint64_t start = 0, end = 0;
        // [0] settled records (resolved, bound), [1] settled V# halves that asked for a geometry copy, [2] waiting
        // records, [3] waiting V# halves that asked for a geometry copy
        std::vector<std::uint32_t> keys[4];
        int index = 0;
        unsigned cls = 0;
        ByStart::iterator at{};          // in by_start[cls]
        ByStart::iterator waiting_at{};  // in waiting_by_start, while keys[2] or keys[3] is not empty
        std::uint32_t seen[4] = {};      // the touched() call that last visited each list (epoch_)
        bool waits() const { return !keys[kWaiting].empty() || !keys[kWaiting + 1].empty(); }
        bool empty() const { return keys[kSettled].empty() && keys[kSettled + 1].empty() && !waits(); }
    };
    struct Index {
        std::unordered_map<Range, std::uint32_t, RangeHash> by_range;  // -> group id
        std::array<ByStart, kClasses> by_start;
        std::uint64_t used[2] = {};  // a bit per class that holds a group (classes 0-63, 64)
        ByStart waiting_by_start;    // the groups with waiting records, by start
    };

    static Range range(std::uint64_t start, std::uint64_t span) {
        if (span > kMaxSpan) span = kMaxSpan;
        return {start, start + span < start ? UINT64_MAX : start + span};
    }
    // The smallest c with length <= 2^c.
    static unsigned length_class(std::uint64_t length) {
        return length <= 1 ? 0u : unsigned(64 - std::countl_zero(length - 1));
    }
    static int list_of(State state, std::uint8_t flags) {
        return (is_waiting(state) ? kWaiting : kSettled) + ((flags & kGeometry) ? 1 : 0);
    }
    // The index group[n] of a record in `state` is in.
    static int index_of(State state, int n) { return is_texture(state) ? kTexture : n; }
    // The room list a record in `state` is in, or -1.
    static int room_of(State state) { return state == waiting_room ? 0 : state == texture_waiting_room ? 1 : -1; }

    void count(State state, int delta) {
        const std::size_t d = std::size_t(std::ptrdiff_t(delta));
        if (is_vsharp(state)) recorded_ += d;
        if (state == waiting || state == waiting_room) waiting_ += d;
        if (is_texture(state)) textures_count_ += d;
        if (state == texture_waiting || state == texture_waiting_room) textures_waiting_ += d;
    }

    // Records `key` in `state` under ranges[0] (and ranges[1], a V#'s target); what is unchanged stays where it is.
    void place(std::uint32_t key, State state, std::uint8_t flags, std::uint64_t base, const Range (&ranges)[2]) {
        Half& h = halves_[key];
        const State old = State(h.state);
        if (old == state && h.flags == flags && h.base == base) {
            bool all_same = true;
            for (int n = 0; n < 2; ++n) {
                const std::uint32_t group = h.group[n];
                if (ranges[n].end == 0) {
                    if (group != 0) {
                        all_same = false;
                        break;
                    }
                } else if (!group || groups_[group - 1].start != ranges[n].start ||
                           groups_[group - 1].end != ranges[n].end) {
                    all_same = false;
                    break;
                }
            }
            if (all_same) return;
        }
        for (int n = 0; n < 2; ++n) {
            const std::uint32_t group = h.group[n];
            const bool keep = group && list_of(old, h.flags) == list_of(state, flags) &&
                              index_of(old, n) == index_of(state, n) && groups_[group - 1].start == ranges[n].start &&
                              groups_[group - 1].end == ranges[n].end;
            if (keep) continue;
            detach(n, key);  // in the list of the old state, which h.state and h.flags still hold
            if (ranges[n].end) attach(n, index_of(state, n), key, ranges[n], list_of(state, flags));
        }
        h.flags = flags;
        if (room_of(old) != room_of(state)) {
            leave_room(key);
            if (room_of(state) >= 0) {
                auto& room = room_[room_of(state)];
                h.room = std::uint32_t(room.size());
                room.push_back(key);
            }
        }
        count(old, -1);
        h.state = state;
        h.base = base;
        count(state, +1);
    }

    void drop(std::uint32_t key) {
        Half& h = halves_[key];
        if (h.state == none) return;
        for (int n = 0; n < 2; ++n) detach(n, key);
        leave_room(key);
        count(State(h.state), -1);
        h.state = none;
        h.flags = 0;
        h.base = 0;
    }

    void leave_room(std::uint32_t key) {
        Half& h = halves_[key];
        const int which = room_of(State(h.state));
        if (which < 0) return;
        auto& room = room_[which];
        const std::uint32_t last = room.back();
        room[h.room] = last;
        halves_[last].room = h.room;
        room.pop_back();
    }

    void attach(int n, int which, std::uint32_t key, Range r, int list_index) {
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
            g.index = which;
            g.cls = length_class(r.end - r.start);
            g.at = index.by_start[g.cls].emplace(r.start, id);
            index.used[g.cls >> 6] |= std::uint64_t(1) << (g.cls & 63);
            it->second = id;
        }
        Group& g = groups_[it->second];
        const bool waited = g.waits();
        auto& list = g.keys[list_index];
        Half& h = halves_[key];
        h.group[n] = it->second + 1;
        h.place[n] = std::uint32_t(list.size());
        list.push_back(key);
        if (list_index >= kWaiting && !waited) g.waiting_at = index.waiting_by_start.emplace(g.start, it->second);
    }

    // Takes `key` out of its group[n], from the list of its current state.
    void detach(int n, std::uint32_t key) {
        Half& h = halves_[key];
        if (!h.group[n]) return;
        const std::uint32_t id = h.group[n] - 1;
        Group& g = groups_[id];
        Index& index = indexes_[g.index];
        const int which_list = list_of(State(h.state), h.flags);
        auto& list = g.keys[which_list];
        const std::uint32_t at = h.place[n], last = list.back();
        list[at] = last;
        halves_[last].place[n] = at;
        list.pop_back();
        h.group[n] = 0;
        if (which_list >= kWaiting && !g.waits()) index.waiting_by_start.erase(g.waiting_at);
        if (!g.empty()) return;
        index.by_range.erase(Range{g.start, g.end});
        index.by_start[g.cls].erase(g.at);
        if (index.by_start[g.cls].empty()) index.used[g.cls >> 6] &= ~(std::uint64_t(1) << (g.cls & 63));
        for (auto& keys : g.keys) std::vector<std::uint32_t>().swap(keys);
        free_groups_.push_back(id);
    }

    // A list of `g` into `out`, unless this call has visited it already.
    void visit(Group& g, int list, std::vector<std::uint32_t>& out) {
        if (g.seen[list] == epoch_) return;
        g.seen[list] = epoch_;
        for (const std::uint32_t key : g.keys[list]) {
            ++looked_at_;
            out.push_back(key >> 1);
        }
    }

    // The waiting records of index `which` whose range starts in [lo, hi): only groups that hold some are walked.
    void find_waiting(int which, std::uint64_t lo, std::uint64_t hi, std::vector<std::uint32_t>& out) {
        const Index& index = indexes_[which];
        for (auto it = index.waiting_by_start.lower_bound(lo); it != index.waiting_by_start.end() && it->first < hi;
             ++it) {
            ++groups_looked_at_;
            visit(groups_[it->second], kWaiting, out);
            visit(groups_[it->second], kWaiting + 1, out);
        }
    }

    // Every record of index `which` whose range overlaps [lo, hi).
    void find(int which, std::uint64_t lo, std::uint64_t hi, std::vector<std::uint32_t>& out) {
        const Index& index = indexes_[which];
        for (unsigned word = 0; word < 2; ++word)
            for (std::uint64_t used = index.used[word]; used; used &= used - 1) {
                const unsigned c = word * 64 + unsigned(std::countr_zero(used));
                const ByStart& groups = index.by_start[c];
                // A group of this class (length at most 2^c) overlaps [lo, hi) only if it starts in (lo - 2^c, hi).
                const std::uint64_t reach = c >= 64 ? UINT64_MAX : (std::uint64_t(1) << c);
                const std::uint64_t first = lo >= reach ? lo - reach + 1 : 0;
                for (auto it = groups.lower_bound(first); it != groups.end() && it->first < hi; ++it) {
                    Group& g = groups_[it->second];
                    ++groups_looked_at_;
                    if (g.end <= lo) continue;
                    for (int list = 0; list < 4; ++list) visit(g, list, out);
                }
            }
    }

    std::size_t slots_ = 0;
    std::unique_ptr<Half[], Free> halves_;
    std::array<Index, 3> indexes_;
    std::array<std::vector<std::uint32_t>, 2> room_;  // 0: V# halves in waiting_room, 1: T#s in texture_waiting_room
    std::vector<Group> groups_;
    std::vector<std::uint32_t> free_groups_;
    std::size_t recorded_ = 0, waiting_ = 0, textures_count_ = 0, textures_waiting_ = 0;
    std::uint64_t looked_at_ = 0, groups_looked_at_ = 0;
    std::uint32_t epoch_ = 0;  // touched() calls, for Group::seen
};

}  // namespace leonida
