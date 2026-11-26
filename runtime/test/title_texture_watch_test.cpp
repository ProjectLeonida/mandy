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
    std::set<std::uintptr_t> unarmable;
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
    bool arm(std::uintptr_t tile, Cell& c) {
        if (unarmable.count(tile)) return false;
        c.armed = true;
        return true;
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
    std::puts("title texture watch: new, untouched, one page, two addresses, remap, every frame, unarmable, lost queue");
    return 0;
}
