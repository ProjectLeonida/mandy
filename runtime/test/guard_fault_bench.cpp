// What tracking a page by its guard costs on this host (under Wine, the numbers the tracking policy is set from,
// src/title_write_policy.h): VirtualProtect per call and per page for ranges of 4 KiB to 4 MiB, one call over a range
// against one call per page; one write fault from the fault to the resumed write, through a minimal vectored handler and
// through the geometry watch's cells (title_geometry_watch.h: notify(fault) reopens a whole 64 KiB tile, page by page),
// on one thread and on eight at once; re-arming a tile; whether GetWriteWatch works for memory VirtualAlloc made and for
// a file-mapped section; and whether Linux's soft-dirty bits (/proc/self/pagemap) can be read from here.
//
// A benchmark: it prints its table and fails only when a measurement could not be made at all.
#include <windows.h>

#include "../src/title_geometry_watch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
double us_since(clock_type::time_point start) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - start).count();
}

constexpr std::size_t kPage = 4096, kTile = 65536;

// The region the handlers below answer for, and how: 0 the faulting page, 1 its 64 KiB tile, 2 the geometry cells.
std::atomic<std::uintptr_t> g_lo{0}, g_hi{0};
std::atomic<int> g_mode{0};
std::atomic<std::uint64_t> g_faults{0};

LONG CALLBACK on_fault(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD* r = info ? info->ExceptionRecord : nullptr;
    if (!r || r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2 || r->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;
    const auto address = static_cast<std::uintptr_t>(r->ExceptionInformation[1]);
    if (address < g_lo.load(std::memory_order_relaxed) || address >= g_hi.load(std::memory_order_relaxed))
        return EXCEPTION_CONTINUE_SEARCH;
    g_faults.fetch_add(1, std::memory_order_relaxed);
    DWORD old = 0;
    switch (g_mode.load(std::memory_order_relaxed)) {
    case 0:
        return VirtualProtect(reinterpret_cast<void*>(address & ~(kPage - 1)), kPage, PAGE_READWRITE, &old)
                   ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
    case 1:
        return VirtualProtect(reinterpret_cast<void*>(address & ~(kTile - 1)), kTile, PAGE_READWRITE, &old)
                   ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
    default: {
        bool restored = false;
        leonida::geometry_writes::notify(address, address + 1, leonida::geometry_writes::Event::fault, &restored);
        return restored ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
    }
    }
}

int failures = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);     \
            ++failures;                                                  \
        }                                                                \
    } while (0)

void protect_costs(std::uint8_t* base) {
    std::printf("VirtualProtect, read-only and back, a call / a page:\n");
    for (const std::size_t bytes : {kPage, kTile, std::size_t(256) << 10, std::size_t(1) << 20, std::size_t(4) << 20}) {
        constexpr int kRounds = 64;
        DWORD old = 0;
        auto start = clock_type::now();
        for (int i = 0; i < kRounds; ++i) {
            CHECK(VirtualProtect(base, bytes, PAGE_READONLY, &old));
            CHECK(VirtualProtect(base, bytes, PAGE_READWRITE, &old));
        }
        const double call = us_since(start) / (2.0 * kRounds);
        // The same range one page at a time.
        const std::size_t pages = bytes / kPage;
        const int rounds = std::max<int>(1, int(256 / pages));
        start = clock_type::now();
        for (int i = 0; i < rounds; ++i)
            for (std::size_t p = 0; p < pages; ++p) {
                CHECK(VirtualProtect(base + p * kPage, kPage, PAGE_READONLY, &old));
                CHECK(VirtualProtect(base + p * kPage, kPage, PAGE_READWRITE, &old));
            }
        const double per_page_calls = us_since(start) / (2.0 * rounds * double(pages));
        std::printf("  %7zu KiB: one call %9.2f us (%7.3f us a page); one call a page %7.2f us a page\n", bytes >> 10,
                    call, call / double(pages), per_page_calls);
    }
}

// Faults: `pages` pages armed in one call, each written once; mode as g_mode.
double fault_cost(std::uint8_t* base, std::size_t pages, int mode, std::size_t stride) {
    DWORD old = 0;
    g_mode.store(mode);
    if (mode == 2) {
        for (std::size_t p = 0; p < pages; ++p) {
            const auto tile = reinterpret_cast<std::uintptr_t>(base + p * stride) & ~(kTile - 1);
            auto* cell = leonida::geometry_writes::acquire(tile);
            CHECK(leonida::geometry_writes::arm(tile, *cell));
        }
    } else {
        CHECK(VirtualProtect(base, pages * stride, PAGE_READONLY, &old));
    }
    const std::uint64_t before = g_faults.load();
    const auto start = clock_type::now();
    for (std::size_t p = 0; p < pages; ++p) base[p * stride + 8] = std::uint8_t(p);
    const double us = us_since(start);
    const std::uint64_t taken = g_faults.load() - before;
    if (mode == 2)
        for (std::size_t p = 0; p < pages; ++p)
            leonida::geometry_writes::release(reinterpret_cast<std::uintptr_t>(base + p * stride) & ~(kTile - 1));
    VirtualProtect(base, pages * stride, PAGE_READWRITE, &old);
    return taken ? us / double(taken) : -1;
}

void fault_costs(std::uint8_t* base, std::size_t span) {
    std::printf("A write fault, from the fault to the resumed write (one thread):\n");
    const std::size_t pages = 256;
    std::printf("  handler reopens the page:              %7.2f us\n", fault_cost(base, pages, 0, kPage));
    std::printf("  handler reopens the 64 KiB tile:       %7.2f us\n", fault_cost(base, pages, 1, kTile));
    std::printf("  geometry cells (VirtualQuery + VirtualProtect a page, 16 a tile): %7.2f us\n",
                fault_cost(base, pages, 2, kTile));
    // Eight threads at once, each over its own tiles.
    constexpr unsigned kThreads = 8;
    const std::size_t each = std::min<std::size_t>(64, span / kTile / kThreads);
    for (const int mode : {0, 2}) {
        g_mode.store(mode);
        DWORD old = 0;
        if (mode == 2) {
            for (std::size_t p = 0; p < kThreads * each; ++p) {
                const auto tile = reinterpret_cast<std::uintptr_t>(base) + p * kTile;
                CHECK(leonida::geometry_writes::arm(tile, *leonida::geometry_writes::acquire(tile)));
            }
        } else {
            CHECK(VirtualProtect(base, kThreads * each * kTile, PAGE_READONLY, &old));
        }
        std::atomic<unsigned> ready{0};
        std::atomic<bool> go{false};
        std::vector<double> per(kThreads);
        std::vector<std::thread> threads;
        const std::uint64_t before = g_faults.load();
        for (unsigned t = 0; t < kThreads; ++t)
            threads.emplace_back([&, t] {
                ++ready;
                while (!go.load()) std::this_thread::yield();
                const auto start = clock_type::now();
                for (std::size_t p = 0; p < each; ++p) base[(t * each + p) * kTile + 16] = std::uint8_t(p);
                per[t] = us_since(start) / double(each);
            });
        while (ready.load() < kThreads) std::this_thread::yield();
        const auto start = clock_type::now();
        go.store(true);
        for (auto& thread : threads) thread.join();
        const double wall = us_since(start);
        const std::uint64_t taken = g_faults.load() - before;
        std::printf("  8 threads at once, %s: %7.2f us a fault on each thread, %7.2f us of wall time a fault "
                    "(%llu faults)\n",
                    mode ? "geometry cells" : "reopening the page", *std::max_element(per.begin(), per.end()),
                    wall / double(std::max<std::uint64_t>(1, taken)), static_cast<unsigned long long>(taken));
        if (mode == 2)
            for (std::size_t p = 0; p < kThreads * each; ++p)
                leonida::geometry_writes::release(reinterpret_cast<std::uintptr_t>(base) + p * kTile);
        VirtualProtect(base, kThreads * each * kTile, PAGE_READWRITE, &old);
    }
    // Re-arming a tile through its cell (VirtualQuery a region, VirtualProtect it).
    const auto tile = reinterpret_cast<std::uintptr_t>(base);
    auto* cell = leonida::geometry_writes::acquire(tile);
    constexpr int kArms = 256;
    double arm_us = 0;
    for (int i = 0; i < kArms; ++i) {
        const auto start = clock_type::now();
        CHECK(leonida::geometry_writes::arm(tile, *cell));
        arm_us += us_since(start);
        base[8] = 1;  // fault, reopen
    }
    leonida::geometry_writes::release(tile);
    std::printf("Re-arming a 64 KiB tile through its cell: %7.2f us\n", arm_us / kArms);
}

void write_watch() {
    std::printf("GetWriteWatch:\n");
    constexpr std::size_t bytes = std::size_t(16) << 20;
    auto* watched = static_cast<std::uint8_t*>(
        VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
    std::printf("  VirtualAlloc with MEM_WRITE_WATCH: %s (error %lu)\n", watched ? "made" : "refused",
                watched ? 0ul : GetLastError());
    if (watched) {
        std::vector<void*> found(bytes / kPage);
        ULONG_PTR count = found.size();
        ULONG granularity = 0;
        GetWriteWatch(WRITE_WATCH_FLAG_RESET, watched, bytes, found.data(), &count, &granularity);  // clear
        for (std::size_t p = 0; p < 37; ++p) watched[p * 3 * kPage] = 1;
        count = found.size();
        const auto start = clock_type::now();
        const UINT r = GetWriteWatch(WRITE_WATCH_FLAG_RESET, watched, bytes, found.data(), &count, &granularity);
        const double us = us_since(start);
        std::printf("  private memory: GetWriteWatch %s, %llu of 37 written pages reported, %.1f us for 16 MiB "
                    "(%.3f us a page)\n",
                    r == 0 ? "works" : "fails", static_cast<unsigned long long>(r == 0 ? count : 0), us,
                    us / double(bytes / kPage));
        VirtualFree(watched, 0, MEM_RELEASE);
    }
    // A section view, as xmem maps the title's physical pages: write watching is not a flag of a view.
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, DWORD(bytes), nullptr);
    auto* view = section ? static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes)) : nullptr;
    if (view) {
        std::vector<void*> found(bytes / kPage);
        ULONG_PTR count = found.size();
        ULONG granularity = 0;
        view[0] = 1;
        const UINT r = GetWriteWatch(0, view, bytes, found.data(), &count, &granularity);
        std::printf("  section view: GetWriteWatch %s (error %lu): a view cannot be write-watched (on Windows "
                    "neither: only VirtualAlloc'd reservations made with MEM_WRITE_WATCH)\n",
                    r == 0 ? "answers" : "fails", r == 0 ? 0ul : GetLastError());
        UnmapViewOfFile(view);
    }
    if (section) CloseHandle(section);
}

// Linux's soft-dirty bit (bit 55 of /proc/self/pagemap, cleared by writing 4 to /proc/self/clear_refs), through Wine's
// file layer (the Z: drive is the host's root): no Wine patch, no syscall of our own.
void soft_dirty(std::uint8_t* base) {
    std::printf("Linux soft-dirty bits through Z:\\proc\\self:\n");
    HANDLE clear = CreateFileA("Z:\\proc\\self\\clear_refs", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                               OPEN_EXISTING, 0, nullptr);
    HANDLE map = CreateFileA("Z:\\proc\\self\\pagemap", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             OPEN_EXISTING, 0, nullptr);
    if (clear == INVALID_HANDLE_VALUE || map == INVALID_HANDLE_VALUE) {
        std::printf("  not reachable (clear_refs %s, pagemap %s; error %lu)\n",
                    clear == INVALID_HANDLE_VALUE ? "no" : "yes", map == INVALID_HANDLE_VALUE ? "no" : "yes",
                    GetLastError());
        if (clear != INVALID_HANDLE_VALUE) CloseHandle(clear);
        if (map != INVALID_HANDLE_VALUE) CloseHandle(map);
        return;
    }
    const auto read_bits = [&](std::uintptr_t va, std::size_t pages, std::vector<std::uint64_t>& out) {
        out.assign(pages, 0);
        LARGE_INTEGER at{};
        at.QuadPart = LONGLONG(va / kPage * 8);
        DWORD got = 0;
        return SetFilePointerEx(map, at, nullptr, FILE_BEGIN) &&
               ReadFile(map, out.data(), DWORD(pages * 8), &got, nullptr) && got == pages * 8;
    };
    constexpr std::size_t pages = 256;
    for (std::size_t p = 0; p < pages; ++p) base[p * kPage] = 1;  // present
    DWORD wrote = 0;
    auto start = clock_type::now();
    const bool cleared = WriteFile(clear, "4", 1, &wrote, nullptr) && wrote == 1;
    const double clear_us = us_since(start);
    for (std::size_t p = 0; p < pages; p += 4) base[p * kPage] = 2;
    std::vector<std::uint64_t> bits;
    start = clock_type::now();
    const bool read = read_bits(reinterpret_cast<std::uintptr_t>(base), pages, bits);
    const double read_us = us_since(start);
    std::size_t dirty = 0, wrong = 0;
    for (std::size_t p = 0; p < pages && read; ++p) {
        const bool d = (bits[p] >> 55) & 1;
        dirty += d;
        wrong += d != (p % 4 == 0);
    }
    std::printf("  clear_refs %s (%.1f us, for the whole process), pagemap %s (%.1f us for %zu pages): %zu of %zu "
                "written pages soft-dirty, %zu wrong\n",
                cleared ? "written" : "refused", clear_us, read ? "read" : "refused", read_us, pages, dirty, pages / 4,
                wrong);
    CloseHandle(clear);
    CloseHandle(map);
}

}  // namespace

int main() {
    constexpr std::size_t span = std::size_t(64) << 20;
    auto* base = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, span, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(base != nullptr);
    if (!base) return 1;
    std::memset(base, 0, span);
    g_lo.store(reinterpret_cast<std::uintptr_t>(base));
    g_hi.store(reinterpret_cast<std::uintptr_t>(base) + span);
    CHECK(AddVectoredExceptionHandler(1, &on_fault) != nullptr);
    protect_costs(base);
    fault_costs(base, span);
    write_watch();
    soft_dirty(base);
    VirtualFree(base, 0, MEM_RELEASE);
    return failures ? 1 : 0;
}
