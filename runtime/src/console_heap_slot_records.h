#pragma once
// What each buffer V# of a console resource heap's GPU copy resolved to (docs/d3d12_vulkan.md section 3.4), kept in
// host RAM so that a change of the address space re-fixes exactly the slots it touches.
//
// A slot of the copy is the title's descriptor with its V# bases rewritten (DeviceState::fix_console_slot). What a
// base became depends on things that change after the slot was written: a buffer or window released, an alias range
// removed, a geometry copy retired, or (for a base nothing backed yet) a buffer, alias or title memory added. Without
// a record of which slot resolved into which range, a change could only be applied by walking every live slot. Here
// each V# half (the buffer V# and the UAV counter V# of a slot) records:
//   - base and span: the bytes the title's V# names, [base, base + span) (the base is the V#'s 48 bits);
//   - target: the device address its GPU copy names, [target, target + span); for a half that resolved to nothing,
//     the device address of a copy that was found but refused (not backed yet), or 0;
//   - state: resolved, waiting (written with no base and no records until something backs it) or none.
// Two indexes find the halves a change touches: by base ("source") and by target. Each holds a half once, under the
// power-of-two class of its span and the granule of its start (a granule is at least 64 KiB and at least the class),
// so a range [lo, hi) is answered by visiting, per class in use, the granules from lo - 2^class to hi: the cost is the
// halves near the range, never the heap. An entry is checked against the record when visited (a half that moved
// leaves a stale entry behind; a bucket drops its stale entries whenever it has doubled since it was last cleaned).
//
// The T# of a slot has a record too (set_texture), in the first half: the address it names (its base), and whether a
// texture bound it or it waits for one. Title-written T#s are looked up by their exact base address
// (bind_title_image), so a third index holds them by that address alone; a texture registered, replaced or gone at an
// address touches exactly the T#s naming it.
// Waiting for room: a V# half whose geometry copy was refused for lack of room, and a T# refused because its bindless
// array was full, wait for room, not for an address; each kind is kept in a list that a change saying room was made
// (SlotChange::geometry_room, SlotChange::texture_room) visits once.
// Portable, standard library only: console_heap_slot_records_test exercises it without a GPU.
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
        textures_.clear();
        for (auto& room : room_) room = Room{};
        recorded_ = waiting_ = textures_count_ = textures_waiting_ = 0;
        looked_at_ = 0;
    }
    std::size_t slots() const { return slots_; }
    std::size_t recorded() const { return recorded_; }  // halves in state resolved or waiting
    std::size_t waiting_count() const { return waiting_; }  // V# halves waiting (for a backing or room)
    std::size_t textures() const { return textures_count_; }  // T# records, bound or waiting
    std::size_t textures_waiting() const { return textures_waiting_; }  // T#s waiting for a texture or room
    std::uint64_t looked_at() const { return looked_at_; }  // index entries visited by touched(), all calls

    // Half `half` (0: the buffer V#, 1: the UAV counter V#) of `slot` now names [base, base + span) and resolved as
    // `state` says (none or a V# state), its copy at `target` (see above). span 0 counts as 1.
    void set(std::size_t slot, unsigned half, State state, std::uint64_t base, std::uint64_t span,
             std::uint64_t target) {
        if (slot >= slots_ || half > 1 || is_texture(state)) return;
        if (state == none) base = span = target = 0;
        if (!span) span = 1;
        if (span > kMaxSpan) span = kMaxSpan;
        const std::size_t key = slot * 2 + half;
        Half& h = halves_[key];
        const Half old = h;
        count(old.state(), -1);
        h.base = base;
        h.target = target;
        h.span_state = span | (std::uint64_t(state) << 56);
        count(state, +1);
        if (state == none) return;  // entries left behind are stale now
        if (state == waiting_room && old.state() != waiting_room) push_room(0, std::uint32_t(key));
        const auto add = [&](int which, std::uint64_t start, std::uint64_t old_start) {
            const unsigned c = span_class(span);
            const std::uint64_t granule = start >> granule_shift(c);
            // Unchanged key: the entry already there stands for it.
            if (is_vsharp(old.state()) &&(which == kSource || old.target) && span_class(old.span()) == c &&
                (old_start >> granule_shift(c)) == granule)
                return;
            push(which, c, granule, std::uint32_t(key));
        };
        add(kSource, base, old.base);
        if (target) add(kTarget, target, old.target);
    }

    // The T# of `slot` names `address` (its base) and is bound or waits as `state` (a T# state) says. The slot's
    // second half is no V# then: its record goes.
    void set_texture(std::size_t slot, State state, std::uint64_t address) {
        if (slot >= slots_ || !is_texture(state)) return;
        set(slot, 1, none, 0, 0, 0);
        const std::size_t key = slot * 2;
        Half& h = halves_[key];
        const Half old = h;
        count(old.state(), -1);
        h.base = address;
        h.target = 0;
        h.span_state = 1 | (std::uint64_t(state) << 56);
        count(state, +1);
        if (state == texture_waiting_room && old.state() != texture_waiting_room) push_room(1, std::uint32_t(key));
        if (is_texture(old.state()) && old.base == address) return;  // the entry already there stands for it
        Bucket& bucket = textures_[address];
        bucket.keys.push_back(std::uint32_t(key));
        if (bucket.keys.size() < 32 || bucket.keys.size() < 2 * bucket.clean) return;
        clean(bucket.keys, [&](std::uint32_t k) { return current_texture(k, address); });
        bucket.clean = bucket.keys.size();
    }

    void clear(std::size_t slot) {
        set(slot, 0, none, 0, 0, 0);
        set(slot, 1, none, 0, 0, 0);
    }
    void clear_range(std::size_t first, std::size_t end) {
        end = (std::min)(end, slots_);
        for (std::size_t slot = first; slot < end; ++slot)
            for (unsigned half = 0; half < 2; ++half)
                if (halves_[slot * 2 + half].state() != none) set(slot, half, none, 0, 0, 0);
    }

    std::uint64_t base(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? halves_[slot * 2 + half].base : 0;
    }
    State state(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? halves_[slot * 2 + half].state() : none;
    }
    std::uint64_t target(std::size_t slot, unsigned half) const {
        return slot < slots_ && half < 2 ? halves_[slot * 2 + half].target : 0;
    }

    // The slots `changes` touch, each once, ascending, appended to `out` (cleared first):
    //   target_gone: halves whose [target, target + span) overlaps the range (whatever their state);
    //   source_gone: halves whose [base, base + span) overlaps it;
    //   target_added, source_added: the same, waiting halves only (a resolved one stays valid until something goes);
    //   texture_gone: T#s naming the address lo, bound or waiting; texture_added: those waiting;
    //   geometry_room: V# halves waiting for room; texture_room: T#s waiting for room (each list once a call).
    void touched(const SlotChange* changes, std::size_t count, std::vector<std::uint32_t>& out) {
        out.clear();
        bool room[2] = {};
        for (std::size_t i = 0; i < count; ++i) {
            const SlotChange& change = changes[i];
            if (change.kind == SlotChange::geometry_room || change.kind == SlotChange::texture_room) {
                room[change.kind == SlotChange::texture_room] = true;
                continue;
            }
            if (change.kind == SlotChange::texture_gone || change.kind == SlotChange::texture_added) {
                find_texture(change.lo, change.kind == SlotChange::texture_added, out);
                continue;
            }
            if (change.hi <= change.lo) continue;
            const bool target = change.kind == SlotChange::target_gone || change.kind == SlotChange::target_added;
            const bool waiting_only = change.kind == SlotChange::target_added || change.kind == SlotChange::source_added;
            find(target ? kTarget : kSource, change.lo, change.hi, waiting_only, out);
        }
        for (int which = 0; which < 2; ++which)
            if (room[which]) find_room(which, out);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
    }

private:
    static constexpr int kSource = 0, kTarget = 1;
    static constexpr std::uint64_t kMaxSpan = (std::uint64_t(1) << 56) - 1;
    static constexpr unsigned kMinGranule = 16;  // 64 KiB
    static constexpr unsigned kClasses = 57;

    struct Half {
        std::uint64_t base, target, span_state;
        std::uint64_t span() const { return span_state & kMaxSpan; }
        State state() const { return State(span_state >> 56); }
    };
    struct Free {
        void operator()(Half* p) const { std::free(p); }
    };
    struct Bucket {
        std::vector<std::uint32_t> keys;
        std::size_t clean = 0;  // its size when it was last cleaned
    };
    struct Index {
        std::array<std::unordered_map<std::uint64_t, Bucket>, kClasses> classes;
        std::uint64_t used = 0;  // a bit per class that ever held an entry
    };
    struct Room {
        std::vector<std::uint32_t> keys;
        std::size_t clean = 0;
    };

    // The smallest c with span <= 2^c.
    static unsigned span_class(std::uint64_t span) {
        return span <= 1 ? 0u : unsigned(64 - std::countl_zero(span - 1));
    }
    static unsigned granule_shift(unsigned c) { return c < kMinGranule ? kMinGranule : c; }

    void count(State state, int delta) {
        const std::size_t d = std::size_t(std::ptrdiff_t(delta));
        if (is_vsharp(state)) recorded_ += d;
        if (state == waiting || state == waiting_room) waiting_ += d;
        if (is_texture(state)) textures_count_ += d;
        if (state == texture_waiting || state == texture_waiting_room) textures_waiting_ += d;
    }

    template<class Keep>
    static void clean(std::vector<std::uint32_t>& keys, Keep keep) {
        keys.erase(std::remove_if(keys.begin(), keys.end(), [&](std::uint32_t k) { return !keep(k); }), keys.end());
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    }

    bool current_texture(std::uint32_t key, std::uint64_t address) const {
        const Half& h = halves_[key];
        return is_texture(h.state()) && h.base == address;
    }
    State room_state(int which) const { return which ? texture_waiting_room : waiting_room; }

    void push_room(int which, std::uint32_t key) {
        Room& room = room_[which];
        room.keys.push_back(key);
        if (room.keys.size() < 32 || room.keys.size() < 2 * room.clean) return;
        clean(room.keys, [&](std::uint32_t k) { return halves_[k].state() == room_state(which); });
        room.clean = room.keys.size();
    }

    void find_texture(std::uint64_t address, bool waiting_only, std::vector<std::uint32_t>& out) {
        const auto it = textures_.find(address);
        if (it == textures_.end()) return;
        for (const std::uint32_t key : it->second.keys) {
            ++looked_at_;
            if (!current_texture(key, address)) continue;
            if (waiting_only && !is_waiting(halves_[key].state())) continue;
            out.push_back(key >> 1);
        }
    }

    // Every entry of the list is visited, and the stale ones (no longer waiting for room) leave it.
    void find_room(int which, std::vector<std::uint32_t>& out) {
        Room& room = room_[which];
        std::size_t kept = 0;
        for (const std::uint32_t key : room.keys) {
            ++looked_at_;
            if (halves_[key].state() != room_state(which)) continue;
            room.keys[kept++] = key;
            out.push_back(key >> 1);
        }
        room.keys.resize(kept);
        clean(room.keys, [](std::uint32_t) { return true; });
        room.clean = room.keys.size();
    }

    // Whether index `which` still holds `key` under (c, granule).
    bool current(int which, std::uint32_t key, unsigned c, std::uint64_t granule) const {
        const Half& h = halves_[key];
        if (!is_vsharp(h.state()) || span_class(h.span()) != c) return false;
        if (which == kTarget && !h.target) return false;
        return ((which == kSource ? h.base : h.target) >> granule_shift(c)) == granule;
    }

    void push(int which, unsigned c, std::uint64_t granule, std::uint32_t key) {
        Index& index = indexes_[which];
        index.used |= std::uint64_t(1) << c;
        Bucket& bucket = index.classes[c][granule];
        bucket.keys.push_back(key);
        if (bucket.keys.size() < 32 || bucket.keys.size() < 2 * bucket.clean) return;
        clean(bucket.keys, [&](std::uint32_t k) { return current(which, k, c, granule); });
        bucket.clean = bucket.keys.size();
    }

    void find(int which, std::uint64_t lo, std::uint64_t hi, bool waiting_only, std::vector<std::uint32_t>& out) {
        Index& index = indexes_[which];
        for (std::uint64_t used = index.used; used; used &= used - 1) {
            const unsigned c = unsigned(std::countr_zero(used));
            auto& buckets = index.classes[c];
            if (buckets.empty()) continue;
            const unsigned shift = granule_shift(c);
            // A half of this class overlaps [lo, hi) only if it starts in (lo - 2^c, hi).
            const std::uint64_t reach = std::uint64_t(1) << c;
            const std::uint64_t first = (lo >= reach ? lo - reach + 1 : 0) >> shift;
            const std::uint64_t last = (hi - 1) >> shift;
            const auto visit = [&](std::uint64_t granule, const Bucket& bucket) {
                for (const std::uint32_t key : bucket.keys) {
                    ++looked_at_;
                    if (!current(which, key, c, granule)) continue;
                    const Half& h = halves_[key];
                    if (waiting_only && !is_waiting(h.state())) continue;
                    const std::uint64_t start = which == kSource ? h.base : h.target;
                    const std::uint64_t span = h.span();
                    const std::uint64_t end = start + span < start ? UINT64_MAX : start + span;
                    if (start < hi && end > lo) out.push_back(key >> 1);
                }
            };
            if (last - first >= buckets.size()) {
                for (const auto& [granule, bucket] : buckets)
                    if (granule >= first && granule <= last) visit(granule, bucket);
            } else {
                for (std::uint64_t granule = first;; ++granule) {
                    const auto it = buckets.find(granule);
                    if (it != buckets.end()) visit(granule, it->second);
                    if (granule == last) break;
                }
            }
        }
    }

    std::size_t slots_ = 0;
    std::unique_ptr<Half[], Free> halves_;
    std::array<Index, 2> indexes_;
    std::unordered_map<std::uint64_t, Bucket> textures_;  // T# records by the exact address they name
    std::array<Room, 2> room_;  // 0: V# halves in waiting_room, 1: T#s in texture_waiting_room
    std::size_t recorded_ = 0, waiting_ = 0, textures_count_ = 0, textures_waiting_ = 0;
    std::uint64_t looked_at_ = 0;
};

}  // namespace leonida
