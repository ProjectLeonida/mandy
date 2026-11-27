// HotColdTracking (src/title_mirror_hot_cold.h) on real guards: the geometry watch's cells over VirtualAlloc'd pages,
// and a vectored handler that does what the D3D12 layer's does first (geometry_writes::notify on the fault). A page
// written at every interval stops faulting, a rarely written one faults once an interval, and the copy the collects
// name equals the memory after every collect.
#include <windows.h>

#include "../src/title_mirror_hot_cold.h"
#include "../src/title_texture_watch.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kPage = 0x10000, kPages = 16;
std::atomic<std::uintptr_t> g_lo{0}, g_hi{0};
std::atomic<std::uint64_t> g_faults[kPages];

LONG CALLBACK on_fault(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* r = info ? info->ExceptionRecord : nullptr;
    if (!r || r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2 || r->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;
    const auto address = static_cast<std::uintptr_t>(r->ExceptionInformation[1]);
    if (address < g_lo.load() || address >= g_hi.load()) return EXCEPTION_CONTINUE_SEARCH;
    g_faults[(address - g_lo.load()) / kPage].fetch_add(1);
    bool restored = false;
    leonida::geometry_writes::notify(address, address + 1, leonida::geometry_writes::Event::fault, &restored);
    return restored ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

int failures = 0;
#define EXPECT(c)                                                         \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

// The title texture refresh's view of the same cells (as the D3D12 layer's TitleTextureCells): a second copy of a page
// the mirror tracks, which must hear of every write the mirror's tracking lets through unguarded.
struct TextureCells {
    using Cell = leonida::geometry_writes::Cell;
    leonida::geometry_writes::WriteQueue queue;
    TextureCells() { leonida::geometry_writes::subscribe_writes(queue); }
    ~TextureCells() { leonida::geometry_writes::unsubscribe_writes(queue); }
    Cell* acquire(std::uintptr_t tile) { return leonida::geometry_writes::acquire(tile); }
    void release(std::uintptr_t tile) { leonida::geometry_writes::release(tile); }
    int arm(std::uintptr_t tile, Cell& cell) { return leonida::geometry_writes::arm(tile, cell) ? 1 : -1; }
    void aliases(std::uintptr_t, std::vector<std::uintptr_t>&) {}
    void take_written(std::vector<std::uintptr_t>& out, bool& overflow) {
        leonida::geometry_writes::take_written(queue, out, overflow);
    }
    void hot_tiles(std::vector<std::uintptr_t>& out) { leonida::geometry_writes::hot_tiles(out); }
    bool enabled() const { return true; }
};

}  // namespace

int main() {
    auto* memory = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, kPages * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    EXPECT(memory);
    if (!memory) return 1;
    std::vector<std::uint8_t> copy(kPages * kPage);
    const auto base = reinterpret_cast<std::uintptr_t>(memory);
    g_lo.store(base);
    g_hi.store(base + kPages * kPage);
    EXPECT(AddVectoredExceptionHandler(1, &on_fault) != nullptr);
    std::mt19937 random(7);
    {
        leonida::HotColdTracking tracking;
        for (std::size_t p = 0; p < kPages; ++p) tracking.watch(base + p * kPage);
        // A texture over page 0 (a video plane the title writes every interval) and one over page 2, watched by the
        // texture refresh's watch over the same cells.
        TextureCells cells;
        leonida::TitleTextureWatch<TextureCells> textures(cells);
        textures.track(1, base, kPage);
        textures.track(2, base + 2 * kPage, kPage);
        unsigned plane_missed = 0;
        std::vector<std::uint64_t> out;
        std::uint64_t hot_faults_late = 0;
        for (unsigned interval = 0; interval < 200; ++interval) {
            const std::uint64_t before = g_faults[0].load();
            memory[interval % kPage] = std::uint8_t(interval);  // page 0: every interval
            if (interval > 20 && (interval % 32) != 1) hot_faults_late += g_faults[0].load() - before;
            if (interval % 10 == 0) memory[kPage + interval] = std::uint8_t(interval);  // page 1: every 10th
            if (random() % 4 == 0) memory[(2 + random() % (kPages - 2)) * kPage + random() % kPage] = std::uint8_t(random());
            tracking.collect(out);
            for (const std::uint64_t page : out)
                std::memcpy(copy.data() + (page - base), reinterpret_cast<const void*>(page), kPage);
            EXPECT(!std::memcmp(copy.data(), memory, copy.size()));
            // The texture refresh's pass after the mirror's: the plane, written this interval, must be named.
            bool plane = false;
            textures.collect([&](std::uint64_t key) {
                plane |= key == 1;
                textures.examine(key);
            });
            if (interval > 0 && !plane) {
                ++plane_missed;
                if (plane_missed <= 8)
                    std::printf("plane not named at interval %u (hot %d)\n", interval, int(tracking.stats().hot));
            }
        }
        EXPECT(tracking.stats().hot >= 1);  // the plane's page turned hot in the mirror's tracking
        EXPECT(plane_missed == 0);          // and the texture watch still heard of every write to it
        EXPECT(hot_faults_late == 0);
        EXPECT(g_faults[1].load() <= 21);
        const auto& s = tracking.stats();
        std::printf("hot/cold on cells: %zu hot, %zu cold; %llu faults on the hot page, %llu on the 10th-interval "
                    "page; %llu hot and %llu cold copies, %llu promotions, %llu demotions\n",
                    s.hot, s.cold, static_cast<unsigned long long>(g_faults[0].load()),
                    static_cast<unsigned long long>(g_faults[1].load()), static_cast<unsigned long long>(s.hot_copies),
                    static_cast<unsigned long long>(s.cold_copies), static_cast<unsigned long long>(s.promotions),
                    static_cast<unsigned long long>(s.demotions));
        for (std::size_t p = 0; p < kPages; ++p) tracking.unwatch(base + p * kPage);
    }
    memory[5] = 1;  // every guard is gone: no fault left
    VirtualFree(memory, 0, MEM_RELEASE);
    return failures ? 1 : 0;
}
