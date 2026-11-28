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

    // Verify HotColdTracking::unwatch() single-lock behavior:
    //  1. Cold armed page with sole owner (refs==1): unwatch() reopens once without toggling set_hot(true/false)
    //     (hot_generation unchanged) and without pushing a bogus tile entry to WriteQueue subscribers.
    //  2. Hot page with sole owner (refs==1): unwatch() clears hot state without pushing a bogus WriteQueue entry.
    //  3. Hot page with a second owner (refs==2): unwatch() drops our hot count and notifies WriteQueue so the
    //     remaining owner observes writes made while hot.
    {
        leonida::geometry_writes::WriteQueue observer;
        leonida::geometry_writes::subscribe_writes(observer);
        std::vector<std::uintptr_t> written_tiles;
        bool overflow = false;
        std::vector<std::uint64_t> out;

        leonida::HotColdTracking tracking;
        const std::uintptr_t p_cold = base + 0 * kPage;
        const std::uintptr_t p_hot_sole = base + 1 * kPage;
        const std::uintptr_t p_hot_shared = base + 2 * kPage;
        tracking.watch(p_cold);
        tracking.watch(p_hot_sole);
        tracking.watch(p_hot_shared);

        // Acquire a second owner on p_hot_shared.
        TextureCells extra_owner;
        extra_owner.acquire(p_hot_shared);

        // Arm p_cold (never written after initial collect) and promote p_hot_sole + p_hot_shared to hot.
        for (unsigned i = 0; i < 12; ++i) {
            memory[1 * kPage + i] = std::uint8_t(i + 1);
            memory[2 * kPage + i] = std::uint8_t(i + 2);
            tracking.collect(out);
        }
        EXPECT(!tracking.hot(p_cold) && tracking.unwritten(p_cold));
        EXPECT(tracking.hot(p_hot_sole) && tracking.hot(p_hot_shared));

        // Drain observer queue before testing unwatch().
        leonida::geometry_writes::take_written(observer, written_tiles, overflow);
        const std::uint64_t hot_gen_before =
            leonida::geometry_writes::state().hot_generation.load(std::memory_order_acquire);

        // 1. Cold armed unwatch (sole owner): zero hot_generation bump, zero bogus WriteQueue push, page writable!
        tracking.unwatch(p_cold);
        const std::uint64_t hot_gen_after_cold =
            leonida::geometry_writes::state().hot_generation.load(std::memory_order_acquire);
        leonida::geometry_writes::take_written(observer, written_tiles, overflow);
        EXPECT(hot_gen_after_cold == hot_gen_before);
        EXPECT(!overflow && written_tiles.empty());
        const std::uint64_t faults_before_write = g_faults[0].load();
        memory[0 * kPage + 7] = 0x5A;  // must NOT fault: guard was reopened on release
        EXPECT(g_faults[0].load() == faults_before_write);

        // 2. Hot unwatch (sole owner): hot_generation bumps once (erased from s.hot), zero bogus WriteQueue push!
        tracking.unwatch(p_hot_sole);
        leonida::geometry_writes::take_written(observer, written_tiles, overflow);
        EXPECT(!overflow && written_tiles.empty());
        EXPECT(!tracking.hot(p_hot_sole));

        // 3. Hot unwatch (shared owner): remaining owner still holds ref, so WriteQueue DOES receive notification!
        tracking.unwatch(p_hot_shared);
        leonida::geometry_writes::take_written(observer, written_tiles, overflow);
        EXPECT(!overflow && written_tiles.size() == 1 && written_tiles[0] == p_hot_shared);
        extra_owner.release(p_hot_shared);

        // 4. Verify notify(Event::fault) returns opened=true both when reopen() restores begin's page (fast path,
        //    skipping 2nd VirtualQuery) and when the page is already writable (concurrent fault fallback VirtualQuery).
        tracking.watch(p_cold);
        tracking.collect(out);  // arms p_cold
        EXPECT(tracking.unwritten(p_cold));
        bool restored = false;
        EXPECT(leonida::geometry_writes::notify(p_cold + 123, p_cold + 124,
                                                leonida::geometry_writes::Event::fault, &restored));
        EXPECT(restored);
        restored = false;
        EXPECT(leonida::geometry_writes::notify(p_cold + 123, p_cold + 124,
                                                leonida::geometry_writes::Event::fault, &restored));
        EXPECT(!restored);
        tracking.unwatch(p_cold);

        leonida::geometry_writes::unsubscribe_writes(observer);
        std::printf("unwatch single-lock & fault reopen fast path: ok\n");
    }

    // Verify watch_batch / unwatch_batch, gap-bridging with pinned-tile guardrail, mid-batch unrelated map change,
    // and reopen() fast path + unexpected-protection fallback.
    {
        std::vector<std::uint64_t> out;
        leonida::HotColdTracking tracking;
        std::vector<std::uint64_t> pages(kPages);
        for (std::size_t p = 0; p < kPages; ++p) pages[p] = base + p * kPage;
        tracking.watch_batch(pages.data(), pages.size());
        tracking.collect(out);  // arms all 16 tiles and populates known_protect + armed_map_gen
        EXPECT(out.size() == kPages);
        for (std::size_t p = 0; p < kPages; ++p) EXPECT(tracking.unwritten(pages[p]));

        // Fault tiles 0, 2, 5 (gaps at 1 and 3..4 are already-armed cold tiles -> eligible for gap-bridging!).
        memory[0 * kPage + 1] = 0x11;
        memory[2 * kPage + 1] = 0x22;
        memory[5 * kPage + 1] = 0x55;
        // Bump title_memory_map on an UNRELATED VA range to exercise range-scoped changes_since().
        leonida::title_memory_map::changed(base + 0x10000000, base + 0x10010000);
        tracking.collect(out);
        EXPECT(out.size() == 3 && out[0] == pages[0] && out[1] == pages[2] && out[2] == pages[5]);
        for (std::size_t p = 0; p < 6; ++p) EXPECT(tracking.unwritten(pages[p]));

        // Pinned-tile guardrail: pin tile 1 via kernel_writes::hold() (opener opens tile 1 writable), fault tiles 0
        // and 2, then collect() MUST NOT bridge over pinned tile 1 and must leave tile 1 writable!
        leonida::kernel_writes::set_opener([](std::uintptr_t b, std::uintptr_t e) {
            leonida::geometry_writes::notify(b, e, leonida::geometry_writes::Event::open);
        });
        const std::uint64_t pin = leonida::kernel_writes::hold(memory + 1 * kPage, kPage);
        EXPECT(pin != 0);
        memory[0 * kPage + 2] = 0x33;
        memory[2 * kPage + 2] = 0x44;
        tracking.collect(out);
        // Tile 0 and tile 2 re-armed; pinned tile 1 stayed writable (writing it must NOT fault!).
        EXPECT(tracking.unwritten(pages[0]) && tracking.unwritten(pages[2]));
        const std::uint64_t f1_before = g_faults[1].load();
        memory[1 * kPage + 2] = 0x77;
        EXPECT(g_faults[1].load() == f1_before);
        leonida::kernel_writes::release(pin);
        leonida::kernel_writes::set_opener(nullptr);

        // Unwatch all pages in one batch and verify zero faults on subsequent writes.
        tracking.unwatch_batch(pages.data(), pages.size());
        const std::uint64_t f0_before = g_faults[0].load();
        memory[0 * kPage + 3] = 0x99;
        EXPECT(g_faults[0].load() == f0_before);
        std::printf("watch_batch/unwatch_batch, gap-bridging pin guardrail, reopen fast path: ok\n");
    }

    // Verify WS1 Round 9 fast paths:
    //  1. Zero-syscall all_armed skip when another owner already armed all 16 sub-pages.
    //  2. VirtualQuery region cache for MEM_RESERVE and PAGE_READONLY (explicit cached_state).
    //  3. Range-scoped reopen() fast path across unrelated map changes + unexpected-protection rollback.
    //  4. kMaxBridgeGap = 32 bridging (>8 tile gap bridged in 1 VirtualProtect) + set_hot_batch/disarm_batch.
    //  5. WC tile detection (wc_tiles + is_wc_page) when host preserves PAGE_WRITECOMBINE.
    {
        std::vector<std::uint64_t> out;

        // 1. Zero-syscall all_armed skip.
        {
            TextureCells other;
            leonida::geometry_writes::Cell* cell = other.acquire(base);
            EXPECT(other.arm(base, *cell) == 1);  // arms all 16 sub-pages
            leonida::HotColdTracking tracking;
            tracking.watch(base);
            tracking.collect(out);
            const auto arm_s = tracking.last_arm_stats();
            EXPECT(arm_s.already_armed_skips == 1 && arm_s.vp_calls == 0 && arm_s.vq_calls == 0);
            EXPECT(tracking.unwritten(base));
            tracking.unwatch(base);
            other.release(base);
        }

        // 2. MEM_RESERVE and PAGE_READONLY VQ region caching (explicit cached_state).
        {
            auto* reserved = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE, PAGE_NOACCESS));
            auto* ro_mem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY));
            EXPECT(reserved && ro_mem);
            if (reserved && ro_mem) {
                const auto res_base = reinterpret_cast<std::uintptr_t>(reserved);
                const auto ro_base = reinterpret_cast<std::uintptr_t>(ro_mem);
                leonida::HotColdTracking tracking;
                std::vector<std::uint64_t> mix = {res_base, res_base + 2 * kPage, res_base + 3 * kPage,
                                                  ro_base,  ro_base + 2 * kPage,  ro_base + 3 * kPage};
                tracking.watch_batch(mix.data(), mix.size());
                tracking.collect(out);
                const auto arm_s = tracking.last_arm_stats();
                EXPECT(arm_s.vp_calls == 0);
                EXPECT(arm_s.vq_calls <= 2);  // cached across non-contiguous sub-runs in each region!
                EXPECT(tracking.take_refused() == 0);
                tracking.unwatch_batch(mix.data(), mix.size());
            }
            if (reserved) VirtualFree(reserved, 0, MEM_RELEASE);
            if (ro_mem) VirtualFree(ro_mem, 0, MEM_RELEASE);
        }

        // 3. Range-scoped reopen() fast path + unexpected-protection rollback.
        {
            leonida::HotColdTracking tracking;
            tracking.watch(base);
            tracking.collect(out);
            EXPECT(tracking.unwritten(base));
            // Unrelated map change must NOT disable range-scoped reopen() fast path.
            leonida::title_memory_map::changed(base + 0x20000000, base + 0x20010000);
            bool restored = false;
            EXPECT(leonida::geometry_writes::notify(base + 64, base + 65, leonida::geometry_writes::Event::fault,
                                                    &restored));
            EXPECT(restored);
            // Re-arm, then bypass title_memory_map::changed and change protection to PAGE_NOACCESS: reopen() MUST
            // detect old != ro && old != r0, restore PAGE_NOACCESS, and NOT make it PAGE_READWRITE!
            tracking.collect(out);
            EXPECT(tracking.unwritten(base));
            DWORD prev_prot = 0;
            EXPECT(VirtualProtect(memory, kPage, PAGE_NOACCESS, &prev_prot) != 0);
            restored = false;
            leonida::geometry_writes::notify(base, base + 1, leonida::geometry_writes::Event::open, &restored);
            EXPECT(!restored);
            MEMORY_BASIC_INFORMATION mbi{};
            EXPECT(VirtualQuery(memory, &mbi, sizeof mbi) && mbi.Protect == PAGE_NOACCESS);
            EXPECT(VirtualProtect(memory, kPage, PAGE_READWRITE, &prev_prot) != 0);
            leonida::title_memory_map::changed(base, base + kPage);
            tracking.unwatch(base);
        }

        // 4. kMaxBridgeGap = 32 bridging (gap of 16 cold armed tiles bridged in 1 VirtualProtect) + disarm_batch.
        {
            constexpr std::size_t kBridgePages = 24;
            auto* bmem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, kBridgePages * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            EXPECT(bmem != nullptr);
            if (bmem) {
                const auto bbase = reinterpret_cast<std::uintptr_t>(bmem);
                leonida::HotColdTracking tracking;
                std::vector<std::uint64_t> bpages(kBridgePages);
                for (std::size_t p = 0; p < kBridgePages; ++p) bpages[p] = bbase + p * kPage;
                tracking.watch_batch(bpages.data(), bpages.size());
                tracking.collect(out);  // initial arm populates known_protect
                // Open tiles 0 and 18 via notify(open) -> gap of 17 already-armed tiles (>8, <=32).
                leonida::geometry_writes::notify(bbase, bbase + 1, leonida::geometry_writes::Event::open);
                leonida::geometry_writes::notify(bbase + 18 * kPage, bbase + 18 * kPage + 1,
                                                 leonida::geometry_writes::Event::open);
                tracking.collect(out);
                const auto arm_s = tracking.last_arm_stats();
                EXPECT(arm_s.bridged_runs == 1 && arm_s.vp_calls == 1 && arm_s.vq_calls == 0);
                // Promote 4 tiles simultaneously and run through a kProbe=32 cycle to exercise disarm_batch/set_hot_batch.
                for (unsigned step = 0; step < 40; ++step) {
                    for (std::size_t p = 0; p < 4; ++p)
                        leonida::geometry_writes::notify(bbase + p * kPage, bbase + p * kPage + 1,
                                                         leonida::geometry_writes::Event::open);
                    tracking.collect(out);
                }
                for (std::size_t p = 0; p < 4; ++p) EXPECT(tracking.hot(bpages[p]));
                tracking.unwatch_batch(bpages.data(), bpages.size());
                VirtualFree(bmem, 0, MEM_RELEASE);
            }
        }

        // 5. PAGE_WRITECOMBINE detection (when OS/Wine preserves PAGE_WRITECOMBINE on VirtualAlloc).
        {
            auto* wc_mem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, 2 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_WRITECOMBINE));
            if (wc_mem) {
                MEMORY_BASIC_INFORMATION mbi{};
                const bool host_has_wc =
                    VirtualQuery(wc_mem, &mbi, sizeof mbi) && (mbi.Protect & PAGE_WRITECOMBINE) != 0;
                const auto wc_base = reinterpret_cast<std::uintptr_t>(wc_mem);
                leonida::HotColdTracking tracking;
                std::vector<std::uint64_t> wpages = {wc_base, wc_base + kPage};
                tracking.watch_batch(wpages.data(), wpages.size());
                tracking.collect(out);
                if (host_has_wc) {
                    EXPECT(tracking.last_arm_stats().wc_tiles == 2);
                    EXPECT(tracking.is_wc_page(wc_base) && tracking.is_wc_page(wc_base + kPage + 123));
                }
                tracking.unwatch_batch(wpages.data(), wpages.size());
                EXPECT(!tracking.is_wc_page(wc_base));
                VirtualFree(wc_mem, 0, MEM_RELEASE);
            }
        }
        std::printf("WS1 fast paths (all_armed, reserve/ro cache, range-scoped reopen, 32-gap bridge, disarm_batch, WC): ok\n");
    }

    VirtualFree(memory, 0, MEM_RELEASE);
    return failures ? 1 : 0;
}
