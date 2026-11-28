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

        // 2. MEM_RESERVE and PAGE_NOACCESS VQ region caching (explicit cached_state), while PAGE_READONLY is NEVER
        //    cached across sub-runs in the intra-lock VQ cache (ShadowPages::reopen() bypasses map_gen).
        {
            auto* reserved = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE, PAGE_NOACCESS));
            auto* noacc_mem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS));
            auto* ro_mem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY));
            EXPECT(reserved && noacc_mem && ro_mem);
            if (reserved && noacc_mem && ro_mem) {
                const auto res_base = reinterpret_cast<std::uintptr_t>(reserved);
                const auto noacc_base = reinterpret_cast<std::uintptr_t>(noacc_mem);
                const auto ro_base = reinterpret_cast<std::uintptr_t>(ro_mem);
                {
                    leonida::HotColdTracking tracking;
                    std::vector<std::uint64_t> mix = {res_base,   res_base + 2 * kPage,   res_base + 3 * kPage,
                                                      noacc_base, noacc_base + 2 * kPage, noacc_base + 3 * kPage};
                    tracking.watch_batch(mix.data(), mix.size());
                    tracking.collect(out);
                    const auto arm_s = tracking.last_arm_stats();
                    EXPECT(arm_s.vp_calls == 0);
                    EXPECT(arm_s.vq_calls <= 2);  // cached across non-contiguous sub-runs in each region!
                    EXPECT(tracking.take_refused() == 0);
                    tracking.unwatch_batch(mix.data(), mix.size());
                }
                {
                    leonida::HotColdTracking tracking;
                    std::vector<std::uint64_t> ro_mix = {ro_base, ro_base + 2 * kPage, ro_base + 3 * kPage};
                    tracking.watch_batch(ro_mix.data(), ro_mix.size());
                    tracking.collect(out);
                    const auto arm_s = tracking.last_arm_stats();
                    EXPECT(arm_s.vp_calls == 0);
                    EXPECT(arm_s.vq_calls == 2);  // PAGE_READONLY NOT cached across sub-runs!
                    EXPECT(tracking.take_refused() == 0);
                    tracking.unwatch_batch(ro_mix.data(), ro_mix.size());
                }
            }
            if (reserved) VirtualFree(reserved, 0, MEM_RELEASE);
            if (noacc_mem) VirtualFree(noacc_mem, 0, MEM_RELEASE);
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

    // Verify Round 10 WS2 improvements & guardrails:
    //  1. Shared Cell::known_* populated on all_armed skip -> subsequent re-arm takes 0 VQ, 1 VP fast path.
    //  2. Cross-pass PAGE_NOACCESS and MEM_RESERVE caching in CellGuard (0 VQ, 0 VP on repeat arm; invalidated by
    //     title_memory_map::changed()), while PAGE_READONLY is NEVER cached cross-pass (reopened RO->RW without
    //     map change MUST be re-queried and re-guarded!).
    //  3. BB1-BB4: bridging across already_armed batch tiles (no double-counting), bridging across a tile watched
    //     only by another owner (s.cells.find fallback), and 48-tile gap (<= kMaxBridgeGap = 64) in 1 VirtualProtect.
    //  4. C1: title_write_watch::ask() single-reader fast path reports+erases mark() entries inside [from, to),
    //     preserves mark() entries outside [from, to), and does not insert GetWriteWatch pages into s.written.
    {
        std::vector<std::uint64_t> out;

        // 1. Shared Cell::known_* seeded from all_armed skip.
        {
            TextureCells other;
            leonida::geometry_writes::Cell* cell = other.acquire(base);
            EXPECT(other.arm(base, *cell) == 1);
            leonida::HotColdTracking tracking;
            tracking.watch(base);
            tracking.collect(out);  // all_armed skip seeds Cell::known_*
            EXPECT(tracking.last_arm_stats().already_armed_skips == 1 && tracking.last_arm_stats().vq_calls == 0);
            leonida::geometry_writes::notify(base, base + 1, leonida::geometry_writes::Event::open);
            tracking.collect(out);  // repeat arm hits shared Cell::known_* -> 0 VQ, 1 VP!
            const auto arm_s = tracking.last_arm_stats();
            EXPECT(arm_s.vq_calls == 0 && arm_s.vp_calls == 1 && arm_s.fast_tiles == 1);
            tracking.unwatch(base);
            other.release(base);
        }

        // 2. Cross-pass PAGE_NOACCESS / MEM_RESERVE caching + PAGE_READONLY NEVER cached cross-pass.
        {
            auto* res = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_RESERVE, PAGE_NOACCESS));
            auto* noacc =
                static_cast<std::uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS));
            auto* ro =
                static_cast<std::uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY));
            EXPECT(res && noacc && ro);
            if (res && noacc && ro) {
                const auto res_va = reinterpret_cast<std::uintptr_t>(res);
                const auto noacc_va = reinterpret_cast<std::uintptr_t>(noacc);
                const auto ro_va = reinterpret_cast<std::uintptr_t>(ro);
                leonida::HotColdTracking tracking;
                std::vector<std::uint64_t> mix = {res_va, noacc_va, ro_va};
                tracking.watch_batch(mix.data(), mix.size());
                tracking.collect(out);  // initial arm queries all 3
                // Re-arm MEM_RESERVE and PAGE_NOACCESS -> 0 VQ, 0 VP!
                leonida::geometry_writes::notify(res_va, res_va + 1, leonida::geometry_writes::Event::write);
                leonida::geometry_writes::notify(noacc_va, noacc_va + 1, leonida::geometry_writes::Event::write);
                tracking.collect(out);
                const auto s_noguard = tracking.last_arm_stats();
                EXPECT(s_noguard.vq_calls == 0 && s_noguard.vp_calls == 0);

                // Reopen PAGE_READONLY -> PAGE_READWRITE WITHOUT title_memory_map::changed() (ShadowPages::reopen):
                // re-arm MUST re-query and protect it!
                DWORD old_prot = 0;
                EXPECT(VirtualProtect(ro, kPage, PAGE_READWRITE, &old_prot) != 0);
                leonida::geometry_writes::notify(ro_va, ro_va + 1, leonida::geometry_writes::Event::write);
                tracking.collect(out);
                const auto s_ro = tracking.last_arm_stats();
                EXPECT(s_ro.vq_calls >= 1 && s_ro.vp_calls == 1);
                MEMORY_BASIC_INFORMATION mbi{};
                EXPECT(VirtualQuery(ro, &mbi, sizeof mbi) && mbi.Protect == PAGE_READONLY);

                tracking.unwatch_batch(mix.data(), mix.size());
            }
            if (res) VirtualFree(res, 0, MEM_RELEASE);
            if (noacc) VirtualFree(noacc, 0, MEM_RELEASE);
            if (ro) VirtualFree(ro, 0, MEM_RELEASE);
        }

        // 3. BB1-BB4: bridge across already_armed batch tile, other-owner-only gap tile, and 48-tile gap (<=64).
        {
            constexpr std::size_t kBigBridge = 56;
            auto* bmem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, kBigBridge * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            EXPECT(bmem != nullptr);
            if (bmem) {
                const auto bbase = reinterpret_cast<std::uintptr_t>(bmem);
                leonida::HotColdTracking tracking;
                TextureCells other;
                std::vector<std::uint64_t> bpages;
                for (std::size_t p = 0; p < kBigBridge; ++p) {
                    if (p == 2) continue;  // watched ONLY by `other` (s.cells.find fallback!)
                    bpages.push_back(bbase + p * kPage);
                }
                leonida::geometry_writes::Cell* c2 = other.acquire(bbase + 2 * kPage);
                EXPECT(other.arm(bbase + 2 * kPage, *c2) == 1);
                // Populate shared known_* on tile 2 via a brief watch+collect or direct arm (already armed via other).
                {
                    leonida::HotColdTracking seed;
                    seed.watch(bbase + 2 * kPage);
                    seed.collect(out);  // all_armed skip populates Cell::known_* on tile 2
                    seed.unwatch(bbase + 2 * kPage);
                }
                tracking.watch_batch(bpages.data(), bpages.size());
                tracking.collect(out);  // arms all remaining tiles and populates Cell::known_*

                // Open tiles 0 and 50 (gap of 49 tiles: includes tile 2 owned only by `other`, plus tile 1 armed in
                // batch via write-notify on already-armed tile 1? Wait: make tile 1 already-armed in this batch by
                // opening tile 1 via `other` re-arming it before collect!).
                leonida::geometry_writes::notify(bbase + 0 * kPage, bbase + 0 * kPage + 1,
                                                 leonida::geometry_writes::Event::open);
                leonida::geometry_writes::notify(bbase + 1 * kPage, bbase + 1 * kPage + 1,
                                                 leonida::geometry_writes::Event::open);
                leonida::geometry_writes::notify(bbase + 50 * kPage, bbase + 50 * kPage + 1,
                                                 leonida::geometry_writes::Event::open);
                // Re-arm tile 1 via `other` so tile 1 is in this batch's dirty list AND already_armed!
                leonida::geometry_writes::Cell* c1 = other.acquire(bbase + 1 * kPage);
                EXPECT(other.arm(bbase + 1 * kPage, *c1) == 1);
                tracking.collect(out);
                const auto arm_s = tracking.last_arm_stats();
                EXPECT(arm_s.bridged_runs == 1 && arm_s.vp_calls == 1 && arm_s.vq_calls == 0);
                EXPECT(arm_s.already_armed_skips == 1 && arm_s.fast_tiles == 2);  // tile 1 NOT double-counted!

                tracking.unwatch_batch(bpages.data(), bpages.size());
                other.release(bbase + 1 * kPage);
                other.release(bbase + 2 * kPage);
                VirtualFree(bmem, 0, MEM_RELEASE);
            }
        }

        // 4. C1: title_write_watch::ask() single-reader fast path with mark() inside and outside [from, to).
        {
            auto* ww = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE));
            if (ww) {
                EXPECT(VirtualAlloc(ww, 4 * kPage, MEM_COMMIT, PAGE_READWRITE) != nullptr);
                const auto ww_base = reinterpret_cast<std::uintptr_t>(ww);
                leonida::title_write_watch::add(ww_base, 4 * kPage);
                const std::uint64_t r = leonida::title_write_watch::open_reader();
                const std::vector<std::pair<std::uintptr_t, std::uintptr_t>> run0 = {{ww_base, ww_base + 2 * kPage}};
                const std::vector<std::pair<std::uintptr_t, std::uintptr_t>> run1 = {
                    {ww_base + 2 * kPage, ww_base + 4 * kPage}};
                // Prime reader past reservation's `born`.
                leonida::title_write_watch::ask(r, run0, [](std::uintptr_t, std::uintptr_t) {},
                                                [](std::uintptr_t, std::uintptr_t) {});
                // Write page 0 via CPU and mark page 1 (inside run0) + page 3 (outside run0).
                ww[10] = 0x42;
                leonida::title_write_watch::mark(ww_base + kPage, ww_base + kPage + 4096);
                leonida::title_write_watch::mark(ww_base + 3 * kPage, ww_base + 3 * kPage + 4096);
                std::vector<std::pair<std::uintptr_t, std::uintptr_t>> got;
                leonida::title_write_watch::ask(
                    r, run0, [&](std::uintptr_t a, std::uintptr_t b) { got.emplace_back(a, b); },
                    [](std::uintptr_t, std::uintptr_t) {});
                bool saw_p0 = false, saw_p1 = false;
                for (const auto& [a, b] : got) {
                    if (a <= ww_base && b >= ww_base + 4096) saw_p0 = true;
                    if (a <= ww_base + kPage && b >= ww_base + kPage + 4096) saw_p1 = true;
                }
                EXPECT(saw_p0 && saw_p1);
                // Entry outside run0 (page 3) was preserved in s.written; entries inside run0 were erased and
                // GetWriteWatch page 0 was never inserted into s.written.
                {
                    auto& st = leonida::title_write_watch::state();
                    std::lock_guard lock(st.journal);
                    EXPECT(st.written.find(ww_base) == st.written.end());
                    EXPECT(st.written.find(ww_base + kPage) == st.written.end());
                    EXPECT(st.written.find(ww_base + 3 * kPage) != st.written.end());
                }
                // Drain run1 so s.written stays clean.
                leonida::title_write_watch::ask(r, run1, [](std::uintptr_t, std::uintptr_t) {},
                                                [](std::uintptr_t, std::uintptr_t) {});
                leonida::title_write_watch::close_reader(r);
                leonida::title_write_watch::remove(ww_base);
                VirtualFree(ww, 0, MEM_RELEASE);
            }
        }
        std::printf("WS2 fast paths (shared Cell::known_*, no-guard cache + RO guardrail, 64-gap bridge, C1): ok\n");
    }

    // Verify Round 10b WS-C fixes:
    //  1. C2 bridged multi-tile rollback: when tile 0 has an unexpected prior protection (e.g., PAGE_NOACCESS changed
    //     without title_memory_map::changed()), rollback restores ONLY eligible tiles individually (tile 0 -> old,
    //     other eligible tiles -> verified kp), preserves already-armed gap tiles (both batch run_already_armed_ and
    //     other-owner s.cells tiles) at kg (PAGE_READONLY), and clears Cell::known_* across the bridged span.
    //  2. C2 non-bridged multi-tile fast-path rollback: tile 0 restored to old, remaining tiles restored to fkp and
    //     re-armed cleanly via slow path (never clobbered to old).
    {
        std::vector<std::uint64_t> out;

        // 1. Bridged rollback preserves already-armed gap tiles and clears Cell::known_*.
        {
            constexpr std::size_t kSpan = 6;
            auto* smem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, kSpan * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            EXPECT(smem != nullptr);
            if (smem) {
                const auto sbase = reinterpret_cast<std::uintptr_t>(smem);
                leonida::HotColdTracking tracking;
                TextureCells other;
                // Tile 2 is watched ONLY by `other` (s.cells gap tile); tiles 0,1,3,4,5 watched by `tracking`.
                leonida::geometry_writes::Cell* c2 = other.acquire(sbase + 2 * kPage);
                EXPECT(other.arm(sbase + 2 * kPage, *c2) == 1);
                {
                    leonida::HotColdTracking seed;
                    seed.watch(sbase + 2 * kPage);
                    seed.collect(out);  // seed Cell::known_* on tile 2
                    seed.unwatch(sbase + 2 * kPage);
                }
                std::vector<std::uint64_t> spages = {sbase + 0 * kPage, sbase + 1 * kPage, sbase + 3 * kPage,
                                                     sbase + 4 * kPage, sbase + 5 * kPage};
                tracking.watch_batch(spages.data(), spages.size());
                tracking.collect(out);  // arms all tiles and populates Cell::known_*

                // Open tiles 0, 1, 4; re-arm tile 1 via `other` so tile 1 is run_already_armed_ in this batch,
                // tile 2 is an other-owner gap tile, tile 3 is a tracking-owned armed gap tile, and tiles 0 & 4
                // are eligible -> bridged span covers [tile 0 .. tile 4].
                leonida::geometry_writes::notify(sbase + 0 * kPage, sbase + 0 * kPage + 1,
                                                 leonida::geometry_writes::Event::open);
                leonida::geometry_writes::notify(sbase + 1 * kPage, sbase + 1 * kPage + 1,
                                                 leonida::geometry_writes::Event::open);
                leonida::geometry_writes::notify(sbase + 4 * kPage, sbase + 4 * kPage + 1,
                                                 leonida::geometry_writes::Event::open);
                leonida::geometry_writes::Cell* c1 = other.acquire(sbase + 1 * kPage);
                EXPECT(other.arm(sbase + 1 * kPage, *c1) == 1);

                // Bypass title_memory_map::changed() on tile 0 -> PAGE_NOACCESS triggers unexpected `old` in
                // bridged VirtualProtect!
                DWORD prev_prot = 0;
                EXPECT(VirtualProtect(smem + 0 * kPage, kPage, PAGE_NOACCESS, &prev_prot) != 0);

                tracking.collect(out);

                MEMORY_BASIC_INFORMATION mbi{};
                // Tile 0 restored to PAGE_NOACCESS; gap tiles 1, 2, 3 preserved at PAGE_READONLY; tile 4 re-armed!
                EXPECT(VirtualQuery(smem + 0 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_NOACCESS);
                EXPECT(VirtualQuery(smem + 1 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_READONLY);
                EXPECT(VirtualQuery(smem + 2 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_READONLY);
                EXPECT(VirtualQuery(smem + 3 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_READONLY);
                EXPECT(VirtualQuery(smem + 4 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_READONLY);
                EXPECT(tracking.unwritten(sbase + 4 * kPage));

                // Cell::known_* cleared across the bridged span's gap tiles (tile 4 was re-populated by slow path).
                {
                    auto& st = leonida::geometry_writes::state();
                    std::lock_guard lock(st.mutex);
                    for (std::size_t g : {std::size_t(1), std::size_t(2), std::size_t(3)}) {
                        const auto it = st.cells.find(sbase + g * kPage);
                        EXPECT(it != st.cells.end() && it->second.known_state == 0);
                    }
                }

                // Restore tile 0 before cleanup.
                EXPECT(VirtualProtect(smem + 0 * kPage, kPage, PAGE_READWRITE, &prev_prot) != 0);
                leonida::title_memory_map::changed(sbase, sbase + kPage);
                tracking.unwatch_batch(spages.data(), spages.size());
                other.release(sbase + 1 * kPage);
                other.release(sbase + 2 * kPage);
                VirtualFree(smem, 0, MEM_RELEASE);
            }
        }

        // 2. Non-bridged multi-tile fast-path rollback: tile 0 -> old (PAGE_NOACCESS), tiles 1 & 2 -> fkp then armed!
        {
            constexpr std::size_t kRun = 3;
            auto* rmem = static_cast<std::uint8_t*>(
                VirtualAlloc(nullptr, kRun * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            EXPECT(rmem != nullptr);
            if (rmem) {
                const auto rbase = reinterpret_cast<std::uintptr_t>(rmem);
                leonida::HotColdTracking tracking;
                std::vector<std::uint64_t> rpages = {rbase + 0 * kPage, rbase + 1 * kPage, rbase + 2 * kPage};
                tracking.watch_batch(rpages.data(), rpages.size());
                tracking.collect(out);  // arms all 3 and populates Cell::known_*
                for (std::size_t p = 0; p < kRun; ++p)
                    leonida::geometry_writes::notify(rbase + p * kPage, rbase + p * kPage + 1,
                                                     leonida::geometry_writes::Event::open);
                // Bypass title_memory_map::changed() on tile 0 -> PAGE_NOACCESS.
                DWORD prev_prot = 0;
                EXPECT(VirtualProtect(rmem + 0 * kPage, kPage, PAGE_NOACCESS, &prev_prot) != 0);

                tracking.collect(out);

                MEMORY_BASIC_INFORMATION mbi{};
                EXPECT(VirtualQuery(rmem + 0 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_NOACCESS);
                EXPECT(VirtualQuery(rmem + 1 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_READONLY);
                EXPECT(VirtualQuery(rmem + 2 * kPage, &mbi, sizeof mbi) && mbi.Protect == PAGE_READONLY);
                EXPECT(tracking.unwritten(rbase + 1 * kPage) && tracking.unwritten(rbase + 2 * kPage));

                EXPECT(VirtualProtect(rmem + 0 * kPage, kPage, PAGE_READWRITE, &prev_prot) != 0);
                leonida::title_memory_map::changed(rbase, rbase + kPage);
                tracking.unwatch_batch(rpages.data(), rpages.size());
                VirtualFree(rmem, 0, MEM_RELEASE);
            }
        }
        std::printf("Round 10b WS-C (bridged/non-bridged rollback preserving gap tiles, RO VQ exclusion): ok\n");
    }

    VirtualFree(memory, 0, MEM_RELEASE);
    return failures ? 1 : 0;
}
