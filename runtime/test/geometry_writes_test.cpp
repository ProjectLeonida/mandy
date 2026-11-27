// A geometry cell guards a title tile read-only so the watch sees the title's writes to it. A page of that tile can
// also be one a slot-133 shadow keeps read-only (both guard the title's geometry). When the last reference to the cell
// goes (release), the pages are made writable again with no write fault to tell the shadow: the shadow takes them for
// unchanged and writes its older copy back over the title's write at the next submission
// (RT-NULL-VTABLE-P4E683D6: a live object's vtable word came back zero). release() therefore calls the reopener the
// layer installs, with the tile's range and this watch's mutex released.
//
// No GPU and no game data: pages of a private allocation.

#include "../src/title_geometry_watch.h"

#include <windows.h>

#include <cstdio>

namespace gw = leonida::geometry_writes;

namespace {

int failures = 0;

#define CHECK(what)                                                       \
    do {                                                                  \
        if (!(what)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #what);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

int calls = 0;
std::uintptr_t seen_from = 0, seen_to = 0;
bool watch_mutex_free = false;

// The layer's reopener shape: called for a tile whose pages it just made writable. It asks the watch again, which
// takes the watch's mutex, so a release() that called this with the mutex held deadlocks here (ctest: a timeout).
void recorder(std::uintptr_t from, std::uintptr_t to) {
    ++calls;
    seen_from = from;
    seen_to = to;
    gw::notify(from, from + 1, gw::Event::write);
    watch_mutex_free = true;
}

DWORD protection(std::uintptr_t at) {
    MEMORY_BASIC_INFORMATION m{};
    return VirtualQuery(reinterpret_cast<const void*>(at), &m, sizeof m) ? (m.Protect & 0xff) : 0;
}

struct Tile {
    void* base = VirtualAlloc(nullptr, gw::tile_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::uintptr_t at() const { return reinterpret_cast<std::uintptr_t>(base); }
    ~Tile() {
        if (base) VirtualFree(base, 0, MEM_RELEASE);
    }
};

void reset() {
    calls = 0;
    seen_from = seen_to = 0;
    watch_mutex_free = false;
}

}  // namespace

int main() {
    gw::set_reopener(&recorder);

    // 1. An armed cell is read-only; dropping it opens the page and tells the reopener about the whole tile, with the
    //    watch's mutex released.
    {
        Tile tile;
        CHECK(tile.base != nullptr);
        gw::Cell* cell = gw::acquire(tile.at());
        CHECK(gw::arm(tile.at(), *cell));
        CHECK(protection(tile.at()) == PAGE_READONLY);
        const auto generation = leonida::title_memory_map::generation();
        reset();
        gw::release(tile.at());
        CHECK(calls == 1);
        CHECK(seen_from == tile.at());
        CHECK(seen_to == tile.at() + gw::tile_size);
        CHECK(watch_mutex_free);
        CHECK(protection(tile.at()) == PAGE_READWRITE);
        CHECK(gw::state().count.load() == 0);
        CHECK(leonida::title_memory_map::generation() > generation);  // the shadows' next refresh sees it too
    }

    // 2. A cell that never guarded anything (no arm, or its pages were already writable) opens nothing, so there is
    //    nothing to tell: no call, and no journal entry either.
    {
        Tile tile;
        gw::Cell* cell = gw::acquire(tile.at());
        (void)cell;
        const auto generation = leonida::title_memory_map::generation();
        reset();
        gw::release(tile.at());
        CHECK(calls == 0);
        CHECK(leonida::title_memory_map::generation() == generation);
        CHECK(protection(tile.at()) == PAGE_READWRITE);
    }

    // 3. Two holders: the page is opened, and the reopener told, only when the last one goes.
    {
        Tile tile;
        gw::Cell* cell = gw::acquire(tile.at());
        gw::acquire(tile.at());
        CHECK(gw::arm(tile.at(), *cell));
        reset();
        gw::release(tile.at());
        CHECK(calls == 0);
        CHECK(protection(tile.at()) == PAGE_READONLY);
        gw::release(tile.at());
        CHECK(calls == 1);
        CHECK(protection(tile.at()) == PAGE_READWRITE);
    }

    gw::set_reopener(nullptr);
    if (failures) {
        std::printf("geometry_writes_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("geometry_writes_test: OK\n");
    return 0;
}
