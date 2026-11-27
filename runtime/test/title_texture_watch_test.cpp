// TitleTextureWatch (src/title_texture_watch.h): a title texture is looked at only when something says it may have
// changed. A fake geometry watch stands in for the cells: a "write" bumps a tile's version and queues the tile, as
// the write-fault handler does through geometry_writes::notify.
#include "../src/title_texture_watch.h"

#include <atomic>
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
    if (failures) return 1;
    std::puts("title texture watch: new, untouched, one page, two addresses, remap, every frame, unarmable, kernel "
              "write in progress, lost queue");
    return 0;
}
