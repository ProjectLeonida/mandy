// TitleTextureWatch (src/title_texture_watch.h): a title texture is looked at only when something says it may have
// changed. A fake geometry watch stands in for the cells: a "write" bumps a tile's version and queues the tile, as
// the write-fault handler does through geometry_writes::notify.
#include "../src/title_texture_watch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace {

struct FakeWatch {
    struct Cell {
        std::atomic<std::uint64_t> version{1}, mapping{1};
        int refs = 0;
        bool armed = false;
    };
    std::map<std::uintptr_t, Cell> cells;
    std::vector<std::uintptr_t> queue;
    std::map<std::uintptr_t, std::vector<std::uintptr_t>> alias;  // tile -> its other addresses
    std::set<std::uintptr_t> unarmable, pinned;
    bool on = true, overflow = false;
    std::uint64_t serial = 1;

    Cell* acquire(std::uintptr_t tile) {
        Cell& c = cells[tile];
        ++c.refs;
        return &c;
    }
    void release(std::uintptr_t tile) {
        auto it = cells.find(tile);
        if (it != cells.end() && !--it->second.refs) cells.erase(it);
    }
    int arm(std::uintptr_t tile, Cell& c) {
        if (pinned.count(tile)) return 0;
        if (unarmable.count(tile)) return -1;
        c.armed = true;
        return 1;
    }
    void aliases(std::uintptr_t tile, std::vector<std::uintptr_t>& out) {
        if (const auto it = alias.find(tile); it != alias.end()) out = it->second;
    }
    std::vector<std::uintptr_t> hot;
    void hot_tiles(std::vector<std::uintptr_t>& out) { out = hot; }
    void take_written(std::vector<std::uintptr_t>& out, bool& lost) {
        out = std::move(queue);
        queue.clear();
        lost = std::exchange(overflow, false);
    }
    bool enabled() const { return on; }

    // A write (fault, kernel write, mapping change) at `address`: only an armed tile's write faults.
    void write(std::uintptr_t address, bool mapping = false) {
        const std::uintptr_t tile = address & ~std::uintptr_t(0xffff);
        const auto it = cells.find(tile);
        if (it == cells.end()) return;
        if (!it->second.armed && !mapping) return;  // writable: no fault, nobody told (the watch's unarmed case)
        it->second.armed = false;
        it->second.version.store(++serial);
        if (mapping) it->second.mapping.store(serial);
        queue.push_back(tile);
    }
};

using Watch = leonida::TitleTextureWatch<FakeWatch>;

std::vector<std::uint64_t> collect(Watch& w) {
    std::vector<std::uint64_t> out;
    w.collect([&](std::uint64_t k) {
        out.push_back(k);
        w.examine(k);
    });
    return out;
}

int failures = 0;
#define EXPECT(c)                                                                \
    do {                                                                         \
        if (!(c)) {                                                              \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);             \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

using V = std::vector<std::uint64_t>;

}  // namespace

int main() {
    constexpr std::uintptr_t kTile = 0x10000, base = 0x7f0000000000;
    {  // a new texture is examined once; untouched afterwards it is never examined again
        FakeWatch f;
        Watch w(f);
        w.track(1, base, 3 * kTile);
        w.track(2, base + 8 * kTile, kTile / 2);
        EXPECT((collect(w) == V{1, 2}));
        for (int pass = 0; pass < 5; ++pass) EXPECT(collect(w).empty());
        EXPECT(f.cells.at(base).armed && f.cells.at(base + 2 * kTile).armed);
        w.track(1, base, 3 * kTile);  // tracked again over the same tiles (a Map, a view): not named, still armed
        EXPECT(collect(w).empty());
        EXPECT(f.cells.at(base).armed);
        w.track(1, base, 4 * kTile);  // over more tiles: named
        EXPECT((collect(w) == V{1}));
    }
    {  // a write to one page names exactly the textures over that page's tile
        FakeWatch f;
        Watch w(f);
        w.track(1, base, 2 * kTile);                     // tiles 0, 1
        w.track(2, base + kTile + 0x8000, 0x1000);       // tile 1 (shares it with 1)
        w.track(3, base + 4 * kTile, kTile);             // tile 4
        collect(w);
        f.write(base + 0x100);                           // tile 0: texture 1 only
        EXPECT((collect(w) == V{1}));
        f.write(base + kTile + 0x9000);                  // tile 1: textures 1 and 2
        EXPECT((collect(w) == V{1, 2}));
        EXPECT(collect(w).empty());
        f.write(base + 6 * kTile);                       // no texture there
        EXPECT(collect(w).empty());
    }
    {  // a texture mapped at two addresses: a write through the other address names it too
        FakeWatch f;
        const std::uintptr_t other = 0x5000000000;
        f.alias[base] = {other};
        Watch w(f);
        w.track(7, base, kTile);
        EXPECT((collect(w) == V{7}));
        EXPECT(f.cells.count(other) && f.cells.at(other).armed);
        f.write(other + 0x40);
        EXPECT((collect(w) == V{7}));
        EXPECT(collect(w).empty());
    }
    {  // a remap: the tile's mapping changes, the texture is named, and its new aliases are looked up
        FakeWatch f;
        Watch w(f);
        w.track(9, base, kTile);
        collect(w);
        const std::uintptr_t other = 0x6000000000;
        f.alias[base] = {other};
        f.write(base, true);
        EXPECT((collect(w) == V{9}));
        EXPECT(f.cells.count(other) && f.cells.at(other).armed);
        f.alias[base] = {};
        f.write(base, true);
        EXPECT((collect(w) == V{9}));
        EXPECT(!f.cells.count(other));  // the old alias is released
    }
    {  // a texture written every frame is named every frame, and only then
        FakeWatch f;
        Watch w(f);
        w.track(4, base, 2 * kTile);
        w.track(5, base + 4 * kTile, kTile);
        collect(w);
        for (int frame = 0; frame < 10; ++frame) {
            f.write(base + kTile + 16 * frame);
            EXPECT((collect(w) == V{4}));
            EXPECT(collect(w).empty());  // the frame's other submissions
        }
    }
    {  // memory that cannot be armed: the texture is named at every collect until it can be (the one fallback)
        FakeWatch f;
        f.unarmable.insert(base);
        Watch w(f);
        w.track(3, base, kTile);
        w.track(8, base + 2 * kTile, kTile);
        EXPECT((collect(w) == V{3, 8}));
        EXPECT((collect(w) == V{3}));
        EXPECT((collect(w) == V{3}));
        f.unarmable.clear();
        EXPECT((collect(w) == V{3}));  // armed now
        EXPECT(collect(w).empty());
        f.on = false;                  // no fault handler: nothing is ever armed, so a texture is always named
        w.track(6, base + 5 * kTile, kTile);
        EXPECT((collect(w) == V{6}));
        EXPECT((collect(w) == V{6}));
    }
    {  // a kernel write in progress: named, but busy (not to be read) until the pin ends; then read once
        FakeWatch f;
        Watch w(f);
        w.track(2, base, 2 * kTile);
        collect(w);
        f.write(base + kTile);
        f.pinned.insert(base + kTile);
        std::vector<Watch::Look> looks;
        const auto pass = [&] {
            looks.clear();
            w.collect([&](std::uint64_t k) { looks.push_back(w.examine(k)); });
        };
        pass();
        EXPECT(looks.size() == 1 && looks[0] == Watch::Look::busy);
        pass();
        EXPECT(looks.size() == 1 && looks[0] == Watch::Look::busy);
        f.pinned.clear();
        pass();
        EXPECT(looks.size() == 1 && looks[0] == Watch::Look::armed);
        pass();
        EXPECT(looks.empty());
        f.unarmable.insert(base);  // a guard that cannot be set: read every time
        f.write(base);
        pass();
        EXPECT(looks.size() == 1 && looks[0] == Watch::Look::unwatched);
    }
    {  // two textures in one tile, one written: both are named (the fault is per tile), exactly one is read whole
        FakeWatch f;
        Watch w(f);
        static std::uint8_t memory[3 * kTile];
        const auto at = [&](std::uintptr_t a) { return memory + (a - base); };
        const std::uintptr_t a = base + 0x1000, b = base + 0x9000;  // 16 KiB each, in tile 0
        const std::uintptr_t c = base + kTile - 0x2000;              // 72 KiB, tiles 0 and 1
        const std::uint64_t sa = 0x4000, sb = 0x4000, sc = 0x12000;
        for (std::size_t i = 0; i < sizeof memory; ++i) memory[i] = std::uint8_t(i * 7 + 1);
        const auto hash = [](const std::uint8_t* p, std::uint64_t n) {
            std::uint64_t h = 1469598103934665603ull;
            for (std::uint64_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
            return h;
        };
        std::map<std::uint64_t, leonida::TilePieces> pieces;
        const std::map<std::uint64_t, std::pair<std::uintptr_t, std::uint64_t>> range{
            {1, {a, sa}}, {2, {b, sb}}, {3, {c, sc}}};
        for (const auto& [k, r] : range) w.track(k, r.first, r.second);
        unsigned whole = 0;
        std::uint64_t compared = 0;
        std::vector<std::uintptr_t> written;
        const auto pass = [&] {
            std::vector<std::uint64_t> read;
            w.collect([&](std::uint64_t k) {
                w.examine(k, &written);
                const auto [cpu, span] = range.at(k);
                auto& p = pieces[k];
                if (p.valid_for(cpu, span) &&
                    p.unchanged(written, [&](std::uintptr_t from, std::uint64_t n) -> std::optional<std::uint64_t> {
                        compared += n;
                        return hash(at(from), n);
                    }))
                    return;
                ++whole;
                read.push_back(k);
                p.compute(cpu, span, at(cpu), hash);
            });
            return read;
        };
        EXPECT((pass() == V{1, 2, 3}));  // new: read whole
        whole = 0;
        compared = 0;
        memory[b - base + 5] ^= 0xff;  // texture 2 written; the fault names tile 0 (textures 1, 2 and 3)
        f.write(b + 5);
        EXPECT((pass() == V{2}));
        EXPECT(whole == 1);
        EXPECT(compared == sa + sb + 0x2000);  // 1 and 3 hashed only their parts of tile 0
        whole = 0;
        memory[c - base + 0x3000] ^= 0xff;  // texture 3 written in tile 1: only 3 is named, and read
        f.write(c + 0x3000);
        EXPECT((pass() == V{3}));
        EXPECT(whole == 1);
    }
    {  // a lost queue names every texture once; untrack releases the tiles
        FakeWatch f;
        Watch w(f);
        w.track(1, base, kTile);
        w.track(2, base + kTile, kTile);
        collect(w);
        f.overflow = true;
        f.write(base);
        EXPECT((collect(w) == V{1}));  // the tile written; the other's version did not move
        w.untrack(1);
        w.untrack(2);
        EXPECT(f.cells.empty());
        EXPECT(w.tiles() == 0 && w.textures() == 0);
    }
    {  // O(1) lock-free clean fast-path (may_have_work) and buffer-hot decoupling (hot_count / hot_tiles)
        struct FastFakeWatch : FakeWatch {
            std::uint64_t hot_tile_calls = 0;
            bool has_written() const { return !queue.empty() || overflow; }
            std::size_t hot_count() const { return hot.size(); }
            void hot_tiles(std::vector<std::uintptr_t>& out) {
                ++hot_tile_calls;
                out = hot;
            }
        };
        using FastWatch = leonida::TitleTextureWatch<FastFakeWatch>;
        FastFakeWatch f;
        FastWatch w(f);
        for (std::uint64_t i = 0; i < 1000; ++i) w.track(i + 1, base + i * kTile, kTile);
        EXPECT(w.may_have_work());
        std::vector<std::uint64_t> seen;
        w.collect([&](std::uint64_t k) {
            seen.push_back(k);
            w.examine(k);
        });
        EXPECT(seen.size() == 1000);
        // Clean steady state: O(1) lock-free fast-path reports no work.
        EXPECT(!w.may_have_work());

        // 500 buffer-only hot pages in TitleMirror (disjoint from watched textures):
        const std::uintptr_t buf_base = base + 0x10000000;
        for (std::size_t i = 0; i < 500; ++i) f.hot.push_back(buf_base + i * kTile);
        // hot_count changed -> one sync pass detects zero texture overlap and decouples them.
        EXPECT(w.may_have_work());
        seen.clear();
        w.collect([&](std::uint64_t k) {
            seen.push_back(k);
            w.examine(k);
        });
        EXPECT(seen.empty());
        EXPECT(w.hot_overlap() == 0);
        EXPECT(!w.may_have_work());  // buffer-only hot pages no longer wake refresh!
        const std::uint64_t calls_before = f.hot_tile_calls;
        for (int pass = 0; pass < 5; ++pass) {
            w.collect([&](std::uint64_t k) { seen.push_back(k); });
        }
        EXPECT(seen.empty());
        EXPECT(f.hot_tile_calls == calls_before);  // hot_tiles() was completely skipped on quiet passes!
        EXPECT(w.stats().hot_skips >= 5);

        // One hot page that DOES overlap texture #42: only texture #42 is named, and examine() still delegates to
        // watch_.arm() so kernel-write pin checks (Look::busy) and concurrent set_hot(false) transitions stay safe.
        f.hot.push_back(base + 41 * kTile);
        EXPECT(w.may_have_work());
        const std::uint64_t arms_before = w.stats().arms;
        seen.clear();
        w.collect([&](std::uint64_t k) {
            seen.push_back(k);
            w.examine(k);
        });
        EXPECT((seen == V{42}));
        EXPECT(w.hot_overlap() == 1);
        EXPECT(w.may_have_work());  // overlapping hot tile stays active
        EXPECT(w.stats().arms == arms_before + 1);  // delegates to watch_.arm() for pin & set_hot(false) safety

        // Tile cools down (TitleMirror::set_hot(false) removes from hot and queues write):
        f.hot.pop_back();
        f.write(base + 41 * kTile, true);  // simulate set_hot(false) bumping version + queuing tile
        seen.clear();
        w.collect([&](std::uint64_t k) {
            seen.push_back(k);
            w.examine(k);
        });
        EXPECT((seen == V{42}));
        EXPECT(w.hot_overlap() == 0);
        EXPECT(!w.may_have_work());

        // Equal-swap edge case with hot_generation(): one buffer page cools while one texture page heats in the
        // same pass (hot.size() stays numerically equal, but hot_generation() increments).
        struct GenFakeWatch : FastFakeWatch {
            std::uint64_t hot_gen = 0;
            std::uint64_t hot_generation() const { return hot_gen; }
        };
        using GenWatch = leonida::TitleTextureWatch<GenFakeWatch>;
        GenFakeWatch gf;
        GenWatch gw(gf);
        gw.track(1, base, kTile);
        gw.track(2, base + kTile, kTile);
        gw.collect([&](std::uint64_t k) { gw.examine(k); });
        gf.hot = {buf_base};
        ++gf.hot_gen;
        gw.collect([&](std::uint64_t k) { gw.examine(k); });
        EXPECT(gw.hot_overlap() == 0 && !gw.may_have_work());
        // Swap buffer page out and texture-2 page in: hot.size() == 1 before and after, hot_gen bumped!
        gf.hot = {base + kTile};
        ++gf.hot_gen;
        EXPECT(gw.may_have_work());
        seen.clear();
        gw.collect([&](std::uint64_t k) {
            seen.push_back(k);
            gw.examine(k);
        });
        EXPECT((seen == V{2}));
        EXPECT(gw.hot_overlap() == 1);

        // Single-tile write on a multi-tile texture names ONLY that texture and examine() returns ONLY that 1 tile.
        const std::uintptr_t big_base = base + 0x20000000;
        w.track(9999, big_base, 16 * kTile);
        w.collect([&](std::uint64_t k) { w.examine(k); });
        EXPECT(!w.may_have_work());
        f.write(big_base + 7 * kTile + 123);
        EXPECT(w.may_have_work());
        std::vector<std::uintptr_t> written_tiles;
        seen.clear();
        w.collect([&](std::uint64_t k) {
            seen.push_back(k);
            w.examine(k, &written_tiles);
        });
        EXPECT((seen == V{9999}));
        EXPECT(written_tiles.size() == 1 && written_tiles[0] == big_base + 7 * kTile);
        EXPECT(!w.may_have_work());
    }
    {  // 1-tile hash-before-reserve skips staging reserve & whole read when bytes are unchanged
        FakeWatch f;
        Watch w(f);
        static std::uint8_t mem[kTile];
        for (std::size_t i = 0; i < sizeof mem; ++i) mem[i] = std::uint8_t(i * 13 + 5);
        const auto at = [&](std::uintptr_t a) { return mem + (a - base); };
        const auto hash = [](const std::uint8_t* p, std::uint64_t n) {
            std::uint64_t h = 1469598103934665603ull;
            for (std::uint64_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
            return h;
        };
        // Two 1-tile textures sharing tile 0.
        const std::map<std::uint64_t, std::pair<std::uintptr_t, std::uint64_t>> range{
            {1, {base + 0x0000, 0x4000}}, {2, {base + 0x8000, 0x4000}}};
        std::map<std::uint64_t, leonida::TilePieces> pieces;
        for (const auto& [k, r] : range) w.track(k, r.first, r.second);
        unsigned reserves = 0;
        std::vector<std::uintptr_t> written;
        const auto pass = [&] {
            std::vector<std::uint64_t> uploaded;
            w.collect([&](std::uint64_t k) {
                w.examine(k, &written);
                const auto [cpu, span] = range.at(k);
                const std::size_t tiles = leonida::TilePieces::tiles(cpu, span);
                auto& p = pieces[k];
                // Hash-before-reserve: `written.size() <= tiles` (including tiles == 1) hashes before reserving.
                if (p.valid_for(cpu, span) && !written.empty() && written.size() <= tiles &&
                    p.unchanged(written, [&](std::uintptr_t from, std::uint64_t n) -> std::optional<std::uint64_t> {
                        return hash(at(from), n);
                    }))
                    return;
                ++reserves;
                uploaded.push_back(k);
                p.compute(cpu, span, at(cpu), hash);
            });
            return uploaded;
        };
        EXPECT((pass() == V{1, 2}));
        EXPECT(reserves == 2);
        reserves = 0;
        // Mutate only texture 2 in shared tile 0: texture 1 (1-tile!) must NOT reserve or upload.
        mem[0x8010] ^= 0xff;
        f.write(base + 0x8010);
        EXPECT((pass() == V{2}));
        EXPECT(reserves == 1);
    }
    {  // note_changed() / note_changed_and_for_each() on already-armed unchanged tiles skips redundant
       // watch_.arm(), while remapped, faulted, hot, pinned, and rearm=true tiles still call watch_.arm().
        FakeWatch f;
        Watch w(f);
        w.track(1, base, 2 * kTile);
        w.track(2, base + 4 * kTile, kTile);
        EXPECT((collect(w) == V{1, 2}));
        const std::uint64_t arms_initial = w.stats().arms;
        EXPECT(arms_initial == 3);

        // 1. note_changed_and_for_each on already-armed unchanged tiles: names textures & populates `written`,
        //    ZERO redundant watch_.arm() calls!
        std::vector<std::uint64_t> over;
        w.note_changed_and_for_each(base, base + 5 * kTile, [&](std::uint64_t k) { over.push_back(k); });
        std::sort(over.begin(), over.end());
        EXPECT((over == V{1, 2}));
        std::vector<std::uint64_t> seen;
        std::vector<std::uintptr_t> written_1, written_2;
        w.collect([&](std::uint64_t k) {
            seen.push_back(k);
            auto look = w.examine(k, k == 1 ? &written_1 : &written_2);
            EXPECT(look == Watch::Look::armed);
        });
        EXPECT((seen == V{1, 2}));
        EXPECT(written_1.size() == 2 && written_2.size() == 1);
        EXPECT(w.stats().arms == arms_initial);  // ZERO redundant watch_.arm() calls!

        // 2. Remap (mapping moved) + note_changed: MUST call watch_.arm() and pick up new aliases.
        const std::uintptr_t alias_addr = 0x5500000000;
        f.alias[base] = {alias_addr};
        f.write(base, true);
        w.note_changed(base, base + kTile);
        EXPECT((collect(w) == V{1}));
        EXPECT(w.stats().arms == arms_initial + 1);
        EXPECT(f.cells.count(alias_addr) && f.cells.at(alias_addr).armed);

        // 3. Faulted (version moved) + pinned: MUST call watch_.arm() and return Look::busy.
        const std::uint64_t arms_before_pin = w.stats().arms;
        f.write(base + kTile);
        f.pinned.insert(base + kTile);
        w.note_changed(base + kTile, base + 2 * kTile);
        Watch::Look pin_look = Watch::Look::armed;
        w.collect([&](std::uint64_t k) { pin_look = w.examine(k); });
        EXPECT(pin_look == Watch::Look::busy);
        EXPECT(w.stats().arms == arms_before_pin + 1);
        f.pinned.clear();
        EXPECT((collect(w) == V{1}));  // armed once unpinned
        EXPECT(w.stats().arms == arms_before_pin + 2);

        // 4. Hot tile: MUST still call watch_.arm() on hot pass.
        const std::uint64_t arms_before_hot = w.stats().arms;
        f.hot = {base + 4 * kTile};
        EXPECT((collect(w) == V{2}));
        EXPECT(w.stats().arms == arms_before_hot + 1);
        f.hot.clear();
        f.write(base + 4 * kTile, true);
        EXPECT((collect(w) == V{2}));
        EXPECT(w.stats().arms == arms_before_hot + 2);

        // 5. Explicit rearm=true (mapped texture whose pages may be committed without xmem): MUST call watch_.arm().
        const std::uint64_t arms_before_rearm = w.stats().arms;
        w.note_changed(base + 4 * kTile, base + 5 * kTile, true);
        EXPECT((collect(w) == V{2}));
        EXPECT(w.stats().arms == arms_before_rearm + 1);

        // 6. Shared-tile re-examine safety (for synced_frames reuse gate `look == armed && arms == arms_before`):
        //    - Two textures sharing a non-hot tile: first examine() arms (+1), second examine() with NO concurrent
        //      write keeps arms unchanged (+0), whereas a concurrent write between them bumps version and arms (+1)!
        //    - Two textures sharing a hot tile: BOTH examine() calls MUST delegate to watch_.arm() (+1 each) so a
        //      hot tile never appears unchanged/armed to the second texture in the same pass!
        w.track(10, base + 8 * kTile, 0x4000);
        w.track(11, base + 8 * kTile + 0x8000, 0x4000);
        EXPECT((collect(w) == V{10, 11}));
        // 6a. Single pre-pass fault on shared tile, no concurrent write between examine(10) and examine(11):
        f.write(base + 8 * kTile);
        std::vector<std::uint64_t> shared_keys;
        w.collect([&](std::uint64_t k) { shared_keys.push_back(k); });
        EXPECT((shared_keys == V{10, 11}));
        const std::uint64_t a0 = w.stats().arms;
        EXPECT(w.examine(10) == Watch::Look::armed && w.stats().arms == a0 + 1);
        const std::uint64_t a1 = w.stats().arms;
        EXPECT(w.examine(11) == Watch::Look::armed && w.stats().arms == a1);  // unchanged since examine(10)

        // 6b. Concurrent write T2 lands on shared tile BETWEEN examine(10) and examine(11):
        f.write(base + 8 * kTile);
        shared_keys.clear();
        w.collect([&](std::uint64_t k) { shared_keys.push_back(k); });
        EXPECT((shared_keys == V{10, 11}));
        const std::uint64_t b0 = w.stats().arms;
        EXPECT(w.examine(10) == Watch::Look::armed && w.stats().arms == b0 + 1);
        f.write(base + 8 * kTile + 0x8010);  // concurrent title write to texture 11 before examine(11)!
        const std::uint64_t b1 = w.stats().arms;
        EXPECT(w.examine(11) == Watch::Look::armed && w.stats().arms == b1 + 1);  // MUST arm (+1) to signal resync!

        // 6c. Shared HOT tile: BOTH examine(10) and examine(11) MUST call watch_.arm() (+1 each)!
        f.hot = {base + 8 * kTile};
        shared_keys.clear();
        w.collect([&](std::uint64_t k) { shared_keys.push_back(k); });
        EXPECT((shared_keys == V{10, 11}));
        const std::uint64_t c0 = w.stats().arms;
        EXPECT(w.examine(10) == Watch::Look::armed && w.stats().arms == c0 + 1);
        const std::uint64_t c1 = w.stats().arms;
        EXPECT(w.examine(11) == Watch::Look::armed && w.stats().arms == c1 + 1);  // hot tile never skips arm()!
        f.hot.clear();
        f.write(base + 8 * kTile, true);
        EXPECT((collect(w) == V{10, 11}));
    }
    {  // 86k-tile scale test verifying O(overlapping) chunked region index (no O(86k) std::sort).
        struct FastScaleWatch {
            struct Cell {
                std::atomic<std::uint64_t> version{1}, mapping{1};
            };
            Cell shared_cell;
            Cell* acquire(std::uintptr_t) { return &shared_cell; }
            void release(std::uintptr_t) {}
            int arm(std::uintptr_t, Cell&) { return 1; }
            void aliases(std::uintptr_t, std::vector<std::uintptr_t>&) {}
            void hot_tiles(std::vector<std::uintptr_t>& out) { out.clear(); }
            void take_written(std::vector<std::uintptr_t>& out, bool& lost) {
                out.clear();
                lost = false;
            }
            bool enabled() const { return true; }
            bool has_written() const { return false; }
            std::uint64_t hot_generation() const { return 0; }
        };
        FastScaleWatch sf;
        leonida::TitleTextureWatch<FastScaleWatch> sw(sf);
        constexpr std::size_t kTextures = 43000;  // 2 tiles each = 86,000 tiles across ~336 x 16 MiB regions
        for (std::size_t i = 0; i < kTextures; ++i) {
            sw.track(i + 1, base + i * 2 * kTile, 2 * kTile);
        }
        EXPECT(sw.tiles() == kTextures * 2);
        sw.collect([&](std::uint64_t k) { sw.examine(k); });

        // Interleave track/untrack mutations with > 4 MiB range queries (20 iterations).
        // Previously each track/untrack dirtied sorted_tiles_ and forced O(86k) std::sort (~5-12 ms each!).
        constexpr std::size_t kIters = 20;
        const std::uintptr_t probe_from = base + 50000 * kTile;  // spans region boundary
        const std::uintptr_t probe_to = probe_from + 128 * kTile;  // 8 MiB (> 4 MiB threshold)
        std::size_t total_over = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t iter = 0; iter < kIters; ++iter) {
            const std::uint64_t temp_key = 1000000 + iter;
            sw.track(temp_key, base + (kTextures * 2 + iter * 2) * kTile, 2 * kTile);
            std::size_t over_count = 0;
            sw.note_changed_and_for_each(probe_from, probe_to, [&](std::uint64_t) { ++over_count; });
            EXPECT(over_count == 64);  // 128 tiles / 2 tiles per texture = 64 textures
            total_over += over_count;
            sw.untrack(temp_key);
        }
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        const double per_iter_ms = double(elapsed_us) / 1000.0 / double(kIters);
        EXPECT(total_over == kIters * 64);
        EXPECT(per_iter_ms < 0.5);
        // Untrack one texture inside [probe_from, probe_to) and verify region index swap-and-pop stays exact.
        sw.untrack(25001);  // covers tiles 50000..50001
        std::size_t after_untrack = 0;
        sw.for_each_over(probe_from, probe_to, [&](std::uint64_t) { ++after_untrack; });
        EXPECT(after_untrack == 63);
        std::printf("86k-tile region index: %.4f ms/iter across %zu tiles (%zu iters, total %lld us)\n",
                    per_iter_ms, sw.tiles(), kIters, static_cast<long long>(elapsed_us));
    }
    if (failures) return 1;
    std::puts("title texture watch: new, untouched, one page, two addresses, remap, every frame, unarmable, kernel "
              "write in progress, lost queue, O(1) fast-path, buffer-hot decoupling, 1-tile hash-before-reserve, "
              "note_changed arm-skip, 86k-tile region index");
    return 0;
}
