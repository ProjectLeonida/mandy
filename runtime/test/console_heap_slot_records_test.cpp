// leonida::ConsoleHeapSlotRecords (src/console_heap_slot_records.h): a change of the address space finds exactly the
// console-heap slots whose V#s it touches, without walking the heap. Portable, CPU only, no game data.
//
// A small model stands in for the device: backings (device ranges), alias ranges (title VA -> arena offset) and a
// resolver that follows DeviceState::vsharp_address's order (an alias first, refused when its copy is not backed; then
// the base as an arena offset). fix() is fix_console_slot: it resolves both halves of a slot, writes the "copy" and the
// records. apply() is apply_slot_changes: the slots touched() names are fixed again. After every step the copy is
// compared with a full re-resolution of every slot: a slot left naming a range that went (a GPU page fault in the
// runtime) or a waiting slot left null after its buffer came fails the comparison.
#include "../src/console_heap_slot_records.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <random>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using leonida::ConsoleHeapSlotRecords;
using leonida::SlotChange;

constexpr std::uint64_t kArena = 0x7f0000000000ull;  // the arena's device address
constexpr std::uint64_t kMask = (1ull << 48) - 1;

struct HalfWords {
    std::uint64_t base = 0, span = 0;  // span 0: no V# in this half
};

struct Model {
    using Words = HalfWords;
    struct Alias {
        std::uint64_t end, shader;
    };
    std::vector<std::array<Words, 2>> slots;
    std::vector<std::array<std::uint64_t, 2>> copy;  // the device address each half names, 0: null
    std::map<std::uint64_t, std::uint64_t> backings;  // device start -> end
    std::map<std::uint64_t, Alias> aliases;           // title VA -> range
    ConsoleHeapSlotRecords records;
    std::vector<SlotChange> changes;
    std::uint64_t fixes = 0;

    explicit Model(std::size_t n) : slots(n), copy(n, {0, 0}) { records.reset(n); }

    bool backed(std::uint64_t address, std::uint64_t span) const {
        auto it = backings.upper_bound(address);
        if (it == backings.begin()) return false;
        --it;
        return address >= it->first && address + span <= it->second;
    }
    // As vsharp_address: 0 when nothing backs it, with `refused` an alias copy that is not backed.
    std::uint64_t resolve(std::uint64_t base, std::uint64_t span, std::uint64_t& refused) const {
        refused = 0;
        auto it = aliases.upper_bound(base);
        if (it != aliases.begin() && base < std::prev(it)->second.end) {
            --it;
            const std::uint64_t to = kArena + it->second.shader + (base - it->first);
            if (base + span > it->second.end || !backed(to, span)) {
                refused = to;
                return 0;
            }
            return to;
        }
        if (base < (1ull << 40) && backed(kArena + base, span)) return kArena + base;
        return 0;
    }
    void fix(std::size_t slot) {
        ++fixes;
        for (unsigned half = 0; half < 2; ++half) {
            const Words& w = slots[slot][half];
            if (!w.span) {
                records.set(slot, half, ConsoleHeapSlotRecords::none, 0, 0, 0);
                copy[slot][half] = 0;
                continue;
            }
            std::uint64_t refused = 0;
            const std::uint64_t to = resolve(w.base, w.span, refused);
            records.set(slot, half, to ? ConsoleHeapSlotRecords::resolved : ConsoleHeapSlotRecords::waiting, w.base,
                        w.span, to ? to : refused);
            copy[slot][half] = to;
        }
    }
    void write(std::size_t slot, Words a, Words b = Words{}) {
        slots[slot] = {a, b};
        fix(slot);
    }
    void zero(std::size_t slot) {
        slots[slot] = {};
        copy[slot] = {0, 0};
        records.clear(slot);
    }
    // The changes the device notes (DeviceState::backing, alias_changed).
    void add_backing(std::uint64_t offset, std::uint64_t bytes) {
        backings[kArena + offset] = kArena + offset + bytes;
        changes.push_back({SlotChange::target_added, kArena + offset, kArena + offset + bytes});
        changes.push_back({SlotChange::source_added, offset & kMask, (offset & kMask) + bytes});
    }
    void release_backing(std::uint64_t offset) {
        const auto it = backings.find(kArena + offset);
        changes.push_back({SlotChange::target_gone, it->first, it->second});
        backings.erase(it);
    }
    void add_alias(std::uint64_t title, std::uint64_t bytes, std::uint64_t shader) {
        for (auto it = aliases.begin(); it != aliases.end();) {
            if (it->first < title + bytes && it->second.end > title) {
                remove_change(it->first, it->second);
                it = aliases.erase(it);
            } else {
                ++it;
            }
        }
        aliases[title] = {title + bytes, shader};
        changes.push_back({SlotChange::source_added, title, title + bytes});
    }
    void remove_alias(std::uint64_t title) {
        const auto it = aliases.find(title);
        remove_change(it->first, it->second);
        aliases.erase(it);
    }
    void remove_change(std::uint64_t title, const Alias& a) {
        changes.push_back({SlotChange::source_gone, title, a.end});
        changes.push_back({SlotChange::target_gone, kArena + a.shader, kArena + a.shader + (a.end - title)});
    }
    // apply_slot_changes: the slots looked at (index entries) and fixed again.
    std::uint64_t looked = 0;
    std::vector<std::uint32_t> touched;
    void apply() {
        const std::uint64_t before = records.looked_at();
        records.touched(changes.data(), changes.size(), touched);
        looked = records.looked_at() - before;
        changes.clear();
        for (const std::uint32_t slot : touched) fix(slot);
    }
    // Every slot as a full re-resolution would leave it: the copy the walk this replaced would have produced.
    bool consistent() const {
        for (std::size_t slot = 0; slot < slots.size(); ++slot)
            for (unsigned half = 0; half < 2; ++half) {
                const Words& w = slots[slot][half];
                std::uint64_t refused = 0;
                const std::uint64_t want = w.span ? resolve(w.base, w.span, refused) : 0;
                if (copy[slot][half] != want) {
                    std::fprintf(stderr, "slot %zu half %u names 0x%llx, a full re-resolution gives 0x%llx\n", slot, half,
                                 static_cast<unsigned long long>(copy[slot][half]),
                                 static_cast<unsigned long long>(want));
                    return false;
                }
            }
        return true;
    }
};

// The design's case: 60,010 live slots, 10 of them name buffer B. Releasing B looks at those 10 and fixes only them;
// the walk this replaced looked at all 60,010.
int release_looks_at_its_slots_only() {
    constexpr std::size_t kSlots = 65536;
    Model m(kSlots);
    constexpr std::uint64_t a = 0x100000, b = 0x900000;
    m.add_backing(a, 0x10000);
    m.add_backing(b, 0x10000);
    m.changes.clear();
    for (std::size_t slot = 0; slot < 60000; ++slot) m.write(slot, {a + (slot % 64) * 0x100, 0x100});
    for (std::size_t slot = 60000; slot < 60010; ++slot) m.write(slot, {b + (slot - 60000) * 0x100, 0x100});
    EXPECT(m.records.recorded() == 60010 && m.records.waiting_count() == 0 && m.consistent());
    m.fixes = 0;
    m.release_backing(b);
    m.apply();
    const std::uint64_t looked = m.looked;
    EXPECT(looked <= 10);
    EXPECT(m.touched.size() == 10 && m.touched.front() == 60000 && m.touched.back() == 60009);
    EXPECT(m.fixes == 10 && m.records.waiting_count() == 10 && m.consistent());
    // Nothing changed: nothing looked at.
    m.apply();
    EXPECT(m.looked == 0 && m.touched.empty());
    std::printf("release: 60,010 live slots, 10 name the buffer: %llu looked at, 10 fixed again\n",
                static_cast<unsigned long long>(looked));
    return 0;
}

// An alias range removed, replaced and added again; a slot overwritten or zeroed after it was recorded; a release that
// only partly overlaps a slot's range; two slots naming one backing; a buffer that comes back.
int cases() {
    Model m(4096);
    constexpr std::uint64_t title = 0x200000000000ull, copy_at = 0x400000, other_copy = 0x800000, c = 0x1000000;
    m.add_backing(copy_at, 0x20000);
    m.add_backing(other_copy, 0x20000);
    m.add_backing(c, 0x10000);
    m.add_alias(title, 0x20000, copy_at);
    m.apply();
    for (std::size_t slot = 0; slot < 8; ++slot) m.write(slot, {title + slot * 0x1000, 0x800});
    EXPECT(m.copy[3][0] == kArena + copy_at + 0x3000 && m.consistent());

    // Alias change: the range moves to another copy; every slot through it follows, nothing else is fixed.
    m.write(100, {c, 0x100});
    m.add_alias(title, 0x20000, other_copy);
    m.apply();
    EXPECT(m.touched.size() == 8 && m.copy[3][0] == kArena + other_copy + 0x3000 && m.consistent());
    // Removed: they wait (null in the copy); added again: they are fixed when it comes.
    m.remove_alias(title);
    m.apply();
    EXPECT(m.touched.size() == 8 && m.copy[3][0] == 0 && m.records.waiting_count() == 8 && m.consistent());
    m.add_alias(title, 0x20000, copy_at);
    m.apply();
    EXPECT(m.touched.size() == 8 && m.records.waiting_count() == 0 && m.consistent());

    // A slot overwritten after it was recorded (it names C now), and one zeroed: a release of their old copy does not
    // touch them; the zeroed one keeps no record.
    m.write(5, {c + 0x200, 0x100});
    m.zero(6);
    m.release_backing(copy_at);
    m.apply();
    EXPECT(m.touched.size() == 6);  // slots 0-4 and 7
    for (const std::uint32_t slot : m.touched) EXPECT(slot != 5 && slot != 6 && slot != 100);
    EXPECT(m.copy[5][0] == kArena + c + 0x200 && m.records.state(6, 0) == ConsoleHeapSlotRecords::none);
    EXPECT(m.consistent());

    // Partial overlap: a release that covers only the tail of a slot's range touches it; one that ends right before
    // its range does not. The two slots naming the same bytes are both fixed.
    m.add_backing(0x2000000, 0x1000);
    m.add_backing(0x2001000, 0x3000);
    m.apply();
    m.write(200, {0x2001800, 0x400});
    m.write(201, {0x2001800, 0x400});
    m.write(202, {0x2000000, 0x1000});  // ends where the second backing starts
    EXPECT(m.copy[200][0] == kArena + 0x2001800 && m.copy[202][0] == kArena + 0x2000000);
    m.release_backing(0x2001000);
    m.apply();
    EXPECT(m.touched.size() == 2 && m.touched[0] == 200 && m.touched[1] == 201);
    EXPECT(m.copy[200][0] == 0 && m.copy[201][0] == 0 && m.copy[202][0] != 0 && m.consistent());
    // The buffer comes back: the waiting slots are fixed again by the addition, not by a probe.
    m.add_backing(0x2001000, 0x3000);
    m.apply();
    EXPECT(m.touched.size() == 2 && m.copy[200][0] == kArena + 0x2001800 && m.copy[201][0] == kArena + 0x2001800);
    EXPECT(m.consistent());

    // The counter half on its own, and a refused alias copy: waiting on the copy's address, fixed when it is backed.
    constexpr std::uint64_t title2 = 0x300000000000ull, unbacked = 0x3000000;
    m.add_alias(title2, 0x10000, unbacked);
    m.apply();
    m.write(300, {c, 0x40}, {title2 + 0x40, 4});
    EXPECT(m.copy[300][0] == kArena + c && m.copy[300][1] == 0 && m.records.state(300, 1) == ConsoleHeapSlotRecords::waiting);
    m.add_backing(unbacked, 0x10000);
    m.apply();
    EXPECT(m.copy[300][1] == kArena + unbacked + 0x40 && m.consistent());

    // A large range in a granule before the release: found through its span class.
    m.add_backing(0x10000000, 0x400000);
    m.apply();
    m.write(400, {0x10000000, 0x400000});
    m.release_backing(0x10000000);
    m.apply();
    EXPECT(m.touched.size() == 1 && m.touched[0] == 400 && m.copy[400][0] == 0 && m.consistent());

    // The heap destroyed (its records with it): a new heap of the same size starts with none.
    m.records.reset(4096);
    m.release_backing(c);
    m.apply();
    EXPECT(m.touched.empty() && m.records.recorded() == 0);
    std::printf("cases: alias moved, removed and added again; overwritten and zeroed slots; partial overlap; two slots "
                "on one backing; a buffer back; a refused alias copy backed later; a large range; a destroyed heap\n");
    return 0;
}

// Random writes, zeroes, releases, additions and alias changes; after each round of changes the copy must equal a full
// re-resolution, and the entries looked at must stay near what the changes touch.
int differential() {
    std::mt19937_64 rng(0x5107);
    constexpr std::size_t kSlots = 3000;
    Model m(kSlots);
    constexpr std::uint64_t kGranule = 0x10000;
    std::vector<std::uint64_t> titles;
    for (int i = 0; i < 6; ++i) titles.push_back(0x100000000000ull + std::uint64_t(i) * 0x100000);
    const auto pick = [&](std::uint64_t n) { return std::uint64_t(rng() % n); };
    std::uint64_t looked = 0, touched = 0;
    for (int round = 0; round < 400; ++round) {
        for (int op = 0; op < 20; ++op) {
            const auto slot = std::size_t(pick(kSlots));
            switch (pick(9)) {
            case 0: case 1: case 2: {  // a V# into an arena offset, one through an alias, or a counter half too
                const auto span = 1 + pick(pick(4) ? 0x2000 : 0x80000);
                Model::Words a{pick(2) ? pick(64) * kGranule / 2 + pick(0x800) : titles[pick(6)] + pick(0x80000), span};
                Model::Words b{};
                if (!pick(4)) b = {pick(64) * kGranule / 2, 4};
                m.write(slot, a, b);
                break;
            }
            case 3: m.zero(slot); break;
            case 4: {
                const auto at = pick(64) * kGranule / 2;
                const auto it = m.backings.find(kArena + at);
                if (it == m.backings.end()) {
                    // Not over another backing.
                    const auto bytes = (1 + pick(8)) * kGranule / 4;
                    bool free = true;
                    for (const auto& [lo, hi] : m.backings) free &= !(lo < kArena + at + bytes && hi > kArena + at);
                    if (free) m.add_backing(at, bytes);
                } else {
                    m.release_backing(at);
                }
                break;
            }
            case 5: case 6: {
                const auto title = titles[pick(6)] + pick(4) * 0x20000;
                m.add_alias(title, (1 + pick(4)) * 0x8000, pick(64) * kGranule / 2);
                break;
            }
            case 7:
                if (!m.aliases.empty()) {
                    auto it = m.aliases.begin();
                    std::advance(it, std::ptrdiff_t(pick(m.aliases.size())));
                    m.remove_alias(it->first);
                }
                break;
            default: m.fix(slot); break;
            }
        }
        m.apply();
        looked += m.looked;
        touched += m.touched.size();
        if (!m.consistent()) {
            std::fprintf(stderr, "round %d\n", round);
            return 1;
        }
    }
    EXPECT(m.records.recorded() > 0);
    // The model's address space is a few MiB, so most changes touch most of it: this case checks results, not cost.
    std::printf("differential: 400 rounds, copy equal to a full re-resolution after each; %llu entries looked at, "
                "%llu slots fixed again, %zu halves recorded at the end\n",
                static_cast<unsigned long long>(looked), static_cast<unsigned long long>(touched),
                m.records.recorded());
    return 0;
}

// T# records (set_texture): a T# waits on the exact address it names; a texture registered there touches the slots
// waiting on it and only those; a texture gone touches the bound ones; a slot rewritten while waiting leaves no
// stale entry; many slots on one texture; V# and T# changes never touch each other's records.
int textures() {
    using R = ConsoleHeapSlotRecords;
    R records;
    records.reset(70000);
    constexpr std::uint64_t kTex = 0x100000100ull, kOther = 0x100000200ull, kNear = 0x100000300ull,
                            kRebound = 0x100000400ull;
    std::vector<std::uint32_t> out;
    const auto touched = [&](SlotChange::Kind kind, std::uint64_t address) {
        const SlotChange change{kind, address, address + 1};
        records.touched(&change, 1, out);
        return out;
    };
    // 60,000 T#s wait on kTex, 10 on kOther, 5 are bound to kNear (each address another texture).
    for (std::uint32_t slot = 0; slot < 60000; ++slot) records.set_texture(slot, R::texture_waiting, kTex);
    for (std::uint32_t slot = 60000; slot < 60010; ++slot) records.set_texture(slot, R::texture_waiting, kOther);
    for (std::uint32_t slot = 60010; slot < 60015; ++slot) records.set_texture(slot, R::texture_bound, kNear);
    EXPECT(records.textures() == 60015 && records.textures_waiting() == 60010 && records.recorded() == 0);
    // A texture at kOther: its 10 slots, looking at 10 entries.
    std::uint64_t looked = records.looked_at();
    EXPECT(touched(SlotChange::texture_added, kOther).size() == 10 && out.front() == 60000 && out.back() == 60009);
    EXPECT(records.looked_at() - looked == 10);
    // A texture added where no T# waits, or where only bound ones are: nothing.
    EXPECT(touched(SlotChange::texture_added, kOther + 0x1000).empty());
    EXPECT(touched(SlotChange::texture_added, kNear).empty());
    // V# changes over those addresses never touch T# records.
    for (const auto kind : {SlotChange::target_gone, SlotChange::source_gone, SlotChange::target_added,
                            SlotChange::source_added}) {
        const SlotChange change{kind, kTex - 0x10000, kTex + 0x10000};
        records.touched(&change, 1, out);
        EXPECT(out.empty());
    }
    // The texture at kNear goes: its bound slots, which the fix makes waiting.
    EXPECT(touched(SlotChange::texture_gone, kNear).size() == 5);
    for (std::uint32_t slot = 60010; slot < 60015; ++slot) records.set_texture(slot, R::texture_waiting, kNear);
    EXPECT(records.textures_waiting() == 60015);
    // Slot 7 is rewritten while waiting: now a T# of kOther, then a V#, then zeroed. Each time kTex no longer names it.
    records.set_texture(7, R::texture_waiting, kOther);
    EXPECT(touched(SlotChange::texture_added, kTex).size() == 59999);
    EXPECT(touched(SlotChange::texture_added, kOther).size() == 11);
    records.set(7, 0, R::resolved, 0x5000, 256, kArena + 0x5000);
    EXPECT(touched(SlotChange::texture_added, kOther).size() == 10);
    EXPECT(records.textures() == 60014 && records.recorded() == 1);
    records.clear(7);
    EXPECT(touched(SlotChange::texture_gone, kOther).size() == 10 && records.recorded() == 0);
    // The texture comes: the fix binds all 59,999; a later texture added there touches none (they wait no more), a
    // texture gone touches them all again.
    for (std::uint32_t slot = 0; slot < 60000; ++slot)
        if (slot != 7) records.set_texture(slot, R::texture_bound, kTex);
    EXPECT(touched(SlotChange::texture_added, kTex).empty());
    EXPECT(touched(SlotChange::texture_gone, kTex).size() == 59999);
    // Rebinding the same address many times leaves one entry per slot (the bucket is cleaned as it grows).
    for (int round = 0; round < 8; ++round)
        for (std::uint32_t slot = 0; slot < 100; ++slot)
            records.set_texture(slot, round & 1 ? R::texture_bound : R::texture_waiting, kRebound);
    looked = records.looked_at();
    EXPECT(touched(SlotChange::texture_gone, kRebound).size() == 100);
    EXPECT(records.looked_at() - looked <= 200);
    std::printf("textures: a texture registered touches only the T#s waiting on its address (10 of 60,015, 10 looked "
                "at); one gone touches the bound ones; rewritten slots leave no stale entry; V# changes touch none\n");
    return 0;
}

// Waiting for room: a V# half refused a geometry copy for lack of room is fixed again when copies are freed
// (geometry_room) whatever its address; a T# refused for a full bindless array when entries are freed (texture_room).
// The A1 gap: such a half has no target, so no target change ever found it.
int room() {
    using R = ConsoleHeapSlotRecords;
    R records;
    records.reset(64);
    records.set(1, 0, R::waiting_room, 0x7000000000ull, 4096, 0);  // geometry refused for room
    records.set(2, 0, R::waiting, 0x7000100000ull, 4096, 0);       // nothing at its base
    records.set(3, 1, R::waiting_room, 0x7000200000ull, 64, 0);    // the counter half
    records.set_texture(4, R::texture_waiting_room, 0x100000400ull);
    records.set_texture(5, R::texture_waiting, 0x100000500ull);
    std::vector<std::uint32_t> out;
    const SlotChange geometry{SlotChange::geometry_room, 0, 1}, textures{SlotChange::texture_room, 0, 1};
    records.touched(&geometry, 1, out);
    EXPECT(out == std::vector<std::uint32_t>({1, 3}));
    records.touched(&textures, 1, out);
    EXPECT(out == std::vector<std::uint32_t>({4}));
    // Both at once, each list once; duplicates of one change count once.
    const SlotChange both[] = {geometry, textures, geometry};
    records.touched(both, 3, out);
    EXPECT(out == std::vector<std::uint32_t>({1, 3, 4}));
    // Resolved since (a copy was made): out of the list; waiting for an address alone: still found by the address.
    records.set(1, 0, R::resolved, 0x7000000000ull, 4096, kArena + 0x10000);
    records.set_texture(4, R::texture_waiting, 0x100000400ull);
    records.touched(both, 3, out);
    EXPECT(out == std::vector<std::uint32_t>({3}));
    const SlotChange added{SlotChange::texture_added, 0x100000400ull, 0x100000401ull};
    records.touched(&added, 1, out);
    EXPECT(out == std::vector<std::uint32_t>({4}));
    // A room waiter is also found by its base, as any waiting half.
    const SlotChange source{SlotChange::source_added, 0x7000200000ull, 0x7000200040ull};
    records.touched(&source, 1, out);
    EXPECT(out == std::vector<std::uint32_t>({3}));
    EXPECT(records.waiting_count() == 2 && records.textures_waiting() == 2);
    std::printf("room: geometry room touches only the halves refused a copy (2 of 3 waiting), texture room only the "
                "T#s refused for a full array; resolved ones leave the lists\n");
    return 0;
}

}  // namespace

// Many slots in a few 64 KiB blocks, as the world's heaps have them (82,000-472,000 entries looked at for 2-15
// changes in one run when the index grouped spans by 64 KiB block): a change looks at the halves whose range overlaps
// it, however many others share its block. 300,000 slots name 16 bytes each, in 74 blocks; a release of 48 bytes
// overlaps three of them. An addition over a range of resolved halves looks at the waiting ones only.
int dense_blocks_look_at_overlaps_only() {
    constexpr std::size_t kSlots = 300000;
    constexpr std::uint64_t kBase = 0x10000000;
    ConsoleHeapSlotRecords records;
    records.reset(kSlots);
    for (std::size_t slot = 0; slot < kSlots; ++slot) {
        const std::uint64_t base = kBase + slot * 16;
        records.set(slot, 0, ConsoleHeapSlotRecords::resolved, base, 16, kArena + base);
    }
    // A slot whose 1 MiB range starts two blocks before the change, and its counter half waiting on a refused copy.
    records.set(5, 1, ConsoleHeapSlotRecords::resolved, kBase + 0x80000 - 0x20000, 0x100000, kArena + 0x7000000);
    std::vector<std::uint32_t> touched;
    const SlotChange release{SlotChange::target_gone, kArena + kBase + 0x80000 + 8, kArena + kBase + 0x80000 + 48};
    std::uint64_t before = records.looked_at();
    records.touched(&release, 1, touched);
    const std::uint64_t looked = records.looked_at() - before;
    // [base + 0x80008, base + 0x80030) overlaps slots 0x8000 (from byte 8), 0x8001 and 0x8002 (to byte 0x30).
    EXPECT(looked == 3);
    EXPECT(touched.size() == 3 && touched[0] == 0x8000 && touched[2] == 0x8002);
    // The same bytes by source: the 1 MiB half of slot 5 overlaps them too.
    const SlotChange removal{SlotChange::source_gone, kBase + 0x80000 + 8, kBase + 0x80000 + 48};
    before = records.looked_at();
    records.touched(&removal, 1, touched);
    EXPECT(records.looked_at() - before == 4 && touched.size() == 4 && touched[0] == 5);
    // A half that moves leaves nothing behind: slot 0x8001 now names other bytes, so the release looks at two.
    records.set(0x8001, 0, ConsoleHeapSlotRecords::resolved, kBase - 0x1000, 16, kArena + kBase - 0x1000);
    before = records.looked_at();
    records.touched(&release, 1, touched);
    EXPECT(records.looked_at() - before == 2 && touched.size() == 2 && touched[1] == 0x8002);

    // Additions: 1,000 slots wait among the 300,000 resolved ones; a backing added over the whole range looks at the
    // 1,000 only.
    for (std::size_t slot = 100000; slot < 101000; ++slot) {
        const std::uint64_t base = kBase + slot * 16;
        records.set(slot, 0, ConsoleHeapSlotRecords::waiting, base, 16, 0);
    }
    EXPECT(records.waiting_count() == 1000);
    const SlotChange added{SlotChange::source_added, kBase, kBase + kSlots * 16};
    before = records.looked_at();
    records.touched(&added, 1, touched);
    EXPECT(records.looked_at() - before == 1000 && touched.size() == 1000 && touched[0] == 100000);
    // Zeroed or reset, they are gone from every index.
    records.clear_range(0, kSlots);
    EXPECT(records.recorded() == 0 && records.ranges() == 0);
    before = records.looked_at();
    records.touched(&removal, 1, touched);
    EXPECT(records.looked_at() == before && touched.empty());
    std::printf("dense blocks: 300,000 slots in 74 blocks of 64 KiB; a 40-byte release looks at the 3 it overlaps, "
                "an addition over all of them at the 1,000 waiting\n");
    return 0;
}

int main() {
    if (release_looks_at_its_slots_only()) return 1;
    if (cases()) return 1;
    if (dense_blocks_look_at_overlaps_only()) return 1;
    if (differential()) return 1;
    if (textures()) return 1;
    if (room()) return 1;
    std::printf("console_heap_slot_records_test: ok\n");
    return 0;
}
