// TitleTextureWatch (src/title_texture_watch.h): a title texture is looked at only when something says it may have
// changed. A fake geometry watch stands in for the cells: a "write" bumps a tile's version and queues the tile, as
// the write-fault handler does through geometry_writes::notify.
#include "../src/title_texture_refresh.h"
#include "../src/title_texture_watch.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
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
    {  // hot_generation() skips watch_.hot_tiles() even while overlapping hot tiles stay active, and non_hot_arms
       // distinguishes steady-state shared hot-tile pin-checks from real re-arms / concurrent set_hot(false).
        struct GenHotWatch : FakeWatch {
            std::uint64_t hot_gen = 0;
            std::uint64_t hot_tile_calls = 0;
            bool has_written() const { return !queue.empty() || overflow; }
            std::uint64_t hot_generation() const { return hot_gen; }
            void hot_tiles(std::vector<std::uintptr_t>& out) {
                ++hot_tile_calls;
                out = hot;
            }
        };
        using GenWatch = leonida::TitleTextureWatch<GenHotWatch>;
        GenHotWatch f;
        GenWatch w(f);
        w.track(1, base, 0x4000);
        w.track(2, base + 0x8000, 0x4000);            // shares tile 0 with texture 1
        w.track(3, base + 2 * kTile, kTile);          // tile 2 (separate)
        w.collect([&](std::uint64_t k) { w.examine(k); });

        // Mark tile 0 + 150 unrelated buffer tiles hot (1 hot_generation bump).
        f.hot.push_back(base);
        for (std::size_t i = 0; i < 150; ++i) f.hot.push_back(base + (1000 + i) * kTile);
        ++f.hot_gen;

        std::vector<std::uint64_t> seen;
        w.collect([&](std::uint64_t k) { seen.push_back(k); });
        EXPECT((seen == V{1, 2}));
        EXPECT(w.hot_overlap() == 1);
        const std::uint64_t calls_after_sync = f.hot_tile_calls;

        // First examine on shared hot tile increments both arms and non_hot_arms; second examine on the same hot tile
        // in the same pass still delegates to watch_.arm() (+1 arms) without bumping non_hot_arms (+0 non_hot_arms)!
        const std::uint64_t arms0 = w.stats().arms, nha0 = w.stats().non_hot_arms;
        EXPECT(w.examine(1) == GenWatch::Look::armed);
        EXPECT(w.stats().arms == arms0 + 1 && w.stats().non_hot_arms == nha0 + 1);
        EXPECT(w.examine(2) == GenWatch::Look::armed);
        EXPECT(w.stats().arms == arms0 + 2 && w.stats().non_hot_arms == nha0 + 1);

        // Subsequent passes with unchanged hot_generation() MUST NOT call watch_.hot_tiles() even though
        // active_hot_tiles_ is non-empty AND even when an unrelated tile faults!
        const std::uint64_t skips_before = w.stats().hot_skips;
        for (int pass = 0; pass < 5; ++pass) {
            f.write(base + 2 * kTile);
            seen.clear();
            w.collect([&](std::uint64_t k) {
                seen.push_back(k);
                w.examine(k);
            });
            EXPECT((seen == V{1, 2, 3}));
        }
        EXPECT(f.hot_tile_calls == calls_after_sync);
        EXPECT(w.stats().hot_skips == skips_before + 5);

        // Concurrent set_hot(false) (version bump) between examine(1) and examine(2) MUST bump non_hot_arms!
        seen.clear();
        w.collect([&](std::uint64_t k) { seen.push_back(k); });
        EXPECT((seen == V{1, 2}));
        const std::uint64_t nha1 = w.stats().non_hot_arms;
        EXPECT(w.examine(1) == GenWatch::Look::armed && w.stats().non_hot_arms == nha1 + 1);
        f.cells.at(base).version.fetch_add(1, std::memory_order_release);  // simulate concurrent set_hot(false)
        EXPECT(w.examine(2) == GenWatch::Look::armed && w.stats().non_hot_arms == nha1 + 2);

        // Cooling tile 0 bumps hot_generation() -> triggers re-poll and drops hot_overlap to 0.
        f.hot.erase(f.hot.begin());
        ++f.hot_gen;
        f.write(base, true);
        seen.clear();
        w.collect([&](std::uint64_t k) {
            seen.push_back(k);
            w.examine(k);
        });
        EXPECT((seen == V{1, 2}));
        EXPECT(f.hot_tile_calls == calls_after_sync + 1);
        EXPECT(w.hot_overlap() == 0);
        EXPECT(!w.may_have_work());
    }
    {  // mirror_range_sum_bytes() and mirror_range_sum_from_pieces() parity + single-byte mutation sensitivity.
        std::vector<std::uint8_t> buf(2 * kTile);
        for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = static_cast<std::uint8_t>((i * 37u + 11u) & 0xffu);
        EXPECT(leonida::mirror_range_sum_bytes(buf.data(), 0) == 0);
        for (const std::uint64_t sz : {std::uint64_t(256), std::uint64_t(4096), std::uint64_t(kTile),
                                       std::uint64_t(2 * kTile)}) {
            const std::uint64_t direct = leonida::mirror_range_sum_bytes(buf.data(), sz);
            std::vector<std::uint64_t> sums;
            for (std::uint64_t off = 0; off < sz; off += leonida::shadow_keep::kPiece)
                sums.push_back(leonida::shadow_keep::sum_of(buf.data() + off, leonida::shadow_keep::kPiece));
            const std::uint64_t combined = leonida::mirror_range_sum_from_pieces(sums.data(), sums.size(), sz);
            EXPECT(direct != 0 && direct == combined);
            // Flipping one byte changes the digest.
            buf[sz - 1] ^= 0x80u;
            EXPECT(leonida::mirror_range_sum_bytes(buf.data(), sz) != direct);
            buf[sz - 1] ^= 0x80u;
            EXPECT(leonida::mirror_range_sum_bytes(buf.data(), sz) == direct);
        }
    }
    {  // examine_batch(): batched arming with arm_batch()-capable watch, deduplication of shared tiles & mutual alias
       // peers, contiguous address ordering, busy > unwatched > armed precedence, hot tiles, and fallback without
       // arm_batch().
        struct BatchFakeWatch : FakeWatch {
            std::uint64_t batch_calls = 0;
            std::uint64_t batch_tiles_total = 0;
            std::vector<std::uintptr_t> last_batch_vas;
            void arm_batch(const std::uintptr_t* tiles, Cell* const* cell_ptrs, std::int8_t* results,
                           std::size_t count) {
                ++batch_calls;
                batch_tiles_total += count;
                last_batch_vas.assign(tiles, tiles + count);
                for (std::size_t i = 0; i < count; ++i) {
                    const int r = arm(tiles[i], *cell_ptrs[i]);
                    results[i] = static_cast<std::int8_t>(r > 0 ? 1 : (r == 0 ? 0 : -1));
                }
            }
        };
        using BatchWatch = leonida::TitleTextureWatch<BatchFakeWatch>;
        BatchFakeWatch bf;
        BatchWatch bw(bf);

        // Textures 1 & 2 share tile 0; texture 3 covers contiguous tiles 1..4; texture 4 & 5 are mutual aliases (tiles
        // 10 and 20); texture 6 covers tiles 30 (unarmable) & 31 (pinned) to verify busy > unwatched precedence.
        bw.track(1, base + 0 * kTile, 0x4000);
        bw.track(2, base + 0 * kTile + 0x8000, 0x4000);
        bw.track(3, base + 1 * kTile, 4 * kTile);
        bf.alias[base + 10 * kTile] = {base + 20 * kTile};
        bf.alias[base + 20 * kTile] = {base + 10 * kTile};
        bw.track(4, base + 10 * kTile, kTile);
        bw.track(5, base + 20 * kTile, kTile);
        bf.unarmable.insert(base + 30 * kTile);
        bf.pinned.insert(base + 31 * kTile);
        bw.track(6, base + 30 * kTile, 2 * kTile);

        std::vector<std::uint64_t> keys;
        bw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT((keys == V{1, 2, 3, 4, 5, 6}));

        std::vector<BatchWatch::BatchExamine> results;
        bw.examine_batch(keys, results);
        bw.update_idle_hint();
        EXPECT(results.size() == 6);
        EXPECT(bf.batch_calls == 1);  // ALL 6 textures armed in ONE single arm_batch() call!
        // Unique tiles armed: 0, 1, 2, 3, 4, 10, 20, 30, 31 = 9 unique tiles (shared tile 0 and mutual aliases 10/20
        // deduplicated!), sorted in ascending address order for contiguous VirtualProtect coalescing.
        EXPECT(bf.last_batch_vas.size() == 9);
        EXPECT(std::is_sorted(bf.last_batch_vas.begin(), bf.last_batch_vas.end()));
        EXPECT(results[0].look == BatchWatch::Look::armed && results[0].written.size() == 1);
        EXPECT(results[1].look == BatchWatch::Look::armed && results[1].written.size() == 1);
        EXPECT(results[2].look == BatchWatch::Look::armed && results[2].written.size() == 4);
        EXPECT(results[3].look == BatchWatch::Look::armed && results[3].written.size() == 1);
        EXPECT(results[4].look == BatchWatch::Look::armed && results[4].written.size() == 1);
        // Texture 6 has tile 30 unarmable (-1) AND tile 31 pinned (0): busy MUST override unwatched!
        EXPECT(results[5].look == BatchWatch::Look::busy && results[5].written.size() == 2);

        // Release pin on tile 31 while tile 30 stays unarmable: texture 6 now reports Look::unwatched.
        bf.pinned.clear();
        keys.clear();
        bw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT((keys == V{6}));
        bw.examine_batch(keys, results);
        EXPECT(results.size() == 1 && results[0].look == BatchWatch::Look::unwatched);

        // Clear unarmable too: texture 6 arms cleanly and steady state goes quiet.
        bf.unarmable.clear();
        keys.clear();
        bw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT((keys == V{6}));
        bw.examine_batch(keys, results);
        EXPECT(results.size() == 1 && results[0].look == BatchWatch::Look::armed);
        keys.clear();
        bw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT(keys.empty());

        // Duplicate tiles in write queue: tile 0 pushed 10 times still names textures 1 & 2 once each with 1 tile in
        // written and 1 unique arm in arm_batch().
        bf.write(base + 0 * kTile);
        for (int dup = 0; dup < 9; ++dup) bf.queue.push_back(base + 0 * kTile);
        const std::uint64_t calls_before = bf.batch_calls;
        keys.clear();
        bw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT((keys == V{1, 2}));
        bw.examine_batch(keys, results);
        EXPECT(bf.batch_calls == calls_before + 1);
        EXPECT(bf.last_batch_vas.size() == 1 && bf.last_batch_vas[0] == base + 0 * kTile);
        EXPECT(results[0].look == BatchWatch::Look::armed && results[0].written.size() == 1);
        EXPECT(results[1].look == BatchWatch::Look::armed && results[1].written.size() == 1);

        // Peer busy > primary unwatched precedence: primary tile 10 unarmable (-1), peer tile 20 pinned (0) -> busy!
        bf.write(base + 10 * kTile);
        bf.unarmable.insert(base + 10 * kTile);
        bf.pinned.insert(base + 20 * kTile);
        keys.clear();
        bw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT((keys == V{4, 5}));
        bw.examine_batch(keys, results);
        EXPECT(results[0].look == BatchWatch::Look::busy);
        EXPECT(results[1].look == BatchWatch::Look::busy);
        bf.unarmable.clear();
        bf.pinned.clear();
        keys.clear();
        bw.collect([&](std::uint64_t k) { keys.push_back(k); });
        bw.examine_batch(keys, results);

        // Fallback examine_batch() on Watch WITHOUT arm_batch() (FakeWatch): identical semantics.
        FakeWatch ff;
        Watch fw(ff);
        fw.track(1, base, 2 * kTile);
        fw.track(2, base + kTile, kTile);
        keys.clear();
        fw.collect([&](std::uint64_t k) { keys.push_back(k); });
        std::vector<Watch::BatchExamine> fres;
        fw.examine_batch(keys, fres);
        EXPECT(fres.size() == 2 && fres[0].look == Watch::Look::armed && fres[1].look == Watch::Look::armed);
        EXPECT(fres[0].written.size() == 2 && fres[1].written.size() == 1);
    }
    {  // 41k-texture / 81k-tile / 350-written-tile microbenchmark: verifies collect() + examine_batch() throughput
       // and arm_batch() syscall coalescing vs per-tile examine() at realistic open-world scale.
        struct ScaleBatchWatch {
            struct Cell {
                std::atomic<std::uint64_t> version{1}, mapping{1};
            };
            std::deque<Cell> pool;
            std::unordered_map<std::uintptr_t, std::size_t> index;
            std::vector<std::uintptr_t> queue;
            std::uint64_t arm_calls = 0;
            std::uint64_t batch_calls = 0;
            std::uint64_t contiguous_runs = 0;

            Cell* acquire(std::uintptr_t tile) {
                auto [it, fresh] = index.try_emplace(tile, pool.size());
                if (fresh) pool.emplace_back();
                return &pool[it->second];
            }
            void release(std::uintptr_t) {}
            int arm(std::uintptr_t, Cell&) {
                ++arm_calls;
                return 1;
            }
            void arm_batch(const std::uintptr_t* tiles, Cell* const*, std::int8_t* results, std::size_t count) {
                ++batch_calls;
                for (std::size_t i = 0; i < count; ++i) {
                    if (i == 0 || tiles[i] != tiles[i - 1] + kTile) ++contiguous_runs;
                    results[i] = 1;
                }
            }
            void aliases(std::uintptr_t, std::vector<std::uintptr_t>&) {}
            void hot_tiles(std::vector<std::uintptr_t>& out) { out.clear(); }
            void take_written(std::vector<std::uintptr_t>& out, bool& lost) {
                out.swap(queue);
                queue.clear();
                lost = false;
            }
            bool enabled() const { return true; }
            bool has_written() const { return !queue.empty(); }
            std::uint64_t hot_generation() const { return 0; }
        };
        ScaleBatchWatch sw_watch;
        leonida::TitleTextureWatch<ScaleBatchWatch> sw(sw_watch);
        constexpr std::size_t kTextures = 41000;  // 41k textures over ~81k tiles
        std::uintptr_t cursor = base;
        std::vector<std::uintptr_t> texture_tiles;
        texture_tiles.reserve(kTextures);
        for (std::size_t i = 0; i < kTextures; ++i) {
            const std::size_t ntiles = (i % 10 == 0) ? 1 : ((i % 10 == 1) ? 3 : 2);
            sw.track(i + 1, cursor, ntiles * kTile);
            texture_tiles.push_back(cursor);
            cursor += ntiles * kTile;
        }
        EXPECT(sw.textures() == kTextures && sw.tiles() >= 81000);
        // Initial arm pass via examine_batch.
        std::vector<std::uint64_t> named;
        sw.collect([&](std::uint64_t k) { named.push_back(k); });
        std::vector<leonida::TitleTextureWatch<ScaleBatchWatch>::BatchExamine> batch_out;
        sw.examine_batch(named, batch_out);
        EXPECT(named.size() == kTextures);
        EXPECT(sw_watch.batch_calls == 1);

        // Simulate 200 passes: 350 written tiles/pass (300 unrelated buffer tiles + 50 watched texture tiles in
        // contiguous runs across ~35 textures).
        constexpr std::size_t kPasses = 200;
        const std::uintptr_t buf_base = base + 0x400000000ull;
        sw_watch.batch_calls = 0;
        sw_watch.contiguous_runs = 0;
        std::uint64_t serial = 2;
        std::size_t total_named = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t p = 0; p < kPasses; ++p) {
            sw_watch.queue.clear();
            for (std::size_t b = 0; b < 300; ++b) sw_watch.queue.push_back(buf_base + ((p * 37 + b) % 4096) * kTile);
            const std::size_t start_tex = (p * 173) % (kTextures - 64);
            for (std::size_t k = 0; k < 25; ++k) {
                const std::uintptr_t a0 = texture_tiles[start_tex + k];
                const std::uintptr_t a1 = a0 + kTile;
                sw_watch.pool[sw_watch.index.at(a0)].version.store(++serial, std::memory_order_relaxed);
                sw_watch.queue.push_back(a0);
                if (const auto it = sw_watch.index.find(a1); it != sw_watch.index.end()) {
                    sw_watch.pool[it->second].version.store(++serial, std::memory_order_relaxed);
                    sw_watch.queue.push_back(a1);
                }
            }
            named.clear();
            sw.collect([&](std::uint64_t k) { named.push_back(k); });
            sw.examine_batch(named, batch_out);
            sw.update_idle_hint();
            total_named += named.size();
        }
        const auto elapsed_us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        const double per_pass_ms = double(elapsed_us) / 1000.0 / double(kPasses);
        EXPECT(sw_watch.batch_calls == kPasses);
        EXPECT(sw_watch.contiguous_runs < kPasses * 10);  // ~50 tiles coalesced into <= 3 contiguous runs/pass!
        EXPECT(total_named >= kPasses * 25);
        EXPECT(per_pass_ms < 0.25);
        std::printf("41k-tex/81k-tile/350-written bench: %.4f ms/pass (%zu passes, %.1f contiguous runs/pass, %zu "
                    "named/pass)\n",
                    per_pass_ms, kPasses, double(sw_watch.contiguous_runs) / double(kPasses), total_named / kPasses);
    }
    {  // Slot-70 deferral contract: touch() on a clean tracked texture wakes may_have_work() and collect() so
       // refresh_title_textures() never fast-skips it, and examine_batch() with 1 tile still routes through arm_batch().
        struct SingleBatchWatch : FakeWatch {
            std::uint64_t batch_calls = 0;
            std::uint64_t single_calls = 0;
            int arm(std::uintptr_t tile, Cell& c) {
                ++single_calls;
                return FakeWatch::arm(tile, c);
            }
            void arm_batch(const std::uintptr_t* tiles, Cell* const* cell_ptrs, std::int8_t* results,
                           std::size_t count) {
                ++batch_calls;
                for (std::size_t i = 0; i < count; ++i) {
                    const int r = FakeWatch::arm(tiles[i], *cell_ptrs[i]);
                    results[i] = static_cast<std::int8_t>(r > 0 ? 1 : (r == 0 ? 0 : -1));
                }
            }
            bool has_written() const { return !queue.empty() || overflow; }
            std::uint64_t hot_generation() const { return 0; }
        };
        SingleBatchWatch sfw;
        leonida::TitleTextureWatch<SingleBatchWatch> sw(sfw);
        sw.track(101, base, kTile);
        EXPECT(sw.may_have_work());
        std::vector<std::uint64_t> keys;
        sw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT((keys == V{101}));
        std::vector<leonida::TitleTextureWatch<SingleBatchWatch>::BatchExamine> res;
        sw.examine_batch(keys, res);
        sw.update_idle_hint();
        EXPECT(sfw.batch_calls == 1 && sfw.single_calls == 0);  // even count==1 uses arm_batch()!
        EXPECT(!sw.may_have_work());

        // Existing texture deferred from slot-70 refresh_title_texture(): touch() MUST wake may_have_work() & collect().
        sw.touch(101);
        EXPECT(sw.may_have_work());
        keys.clear();
        sw.collect([&](std::uint64_t k) { keys.push_back(k); });
        EXPECT((keys == V{101}));
        sw.examine_batch(keys, res);
        sw.update_idle_hint();
        EXPECT(!sw.may_have_work());
    }
    if (failures) return 1;
    std::puts("title texture watch: new, untouched, one page, two addresses, remap, every frame, unarmable, kernel "
              "write in progress, lost queue, O(1) fast-path, buffer-hot decoupling, 1-tile hash-before-reserve, "
              "note_changed arm-skip, 86k-tile region index, hot_generation skip, non_hot_arms, mirror piece hash, "
              "examine_batch, 41k-tex scale bench");
    return 0;
}
