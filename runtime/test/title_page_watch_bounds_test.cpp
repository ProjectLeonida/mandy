// runtime/src/title_page_watch.h at the edges of the address space and of a write-watched reservation. Only address
// arithmetic: nothing is mapped at the addresses used, nothing reads or writes them. No game data, no GPU.
//   1. title_page_aliases::for_each_in over a range that ends at UINTPTR_MAX ends, and finds the aliased pages in it;
//   2. title_write_watch::mark of a range that starts before a reservation marks the reservation's pages in it;
//   3. title_write_watch::mark of a range that ends near UINTPTR_MAX is not dropped;
//   4. title_write_watch::add of a reservation at the top of the address space does not wrap;
//   5. kernel_writes::begin_write and hold pin a range at the top of the address space without wrapping;
//   6. title_texel_writes::note keeps such a range in order.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "../src/title_page_watch.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

void opener(std::uintptr_t, std::uintptr_t) {}

}  // namespace

int main() {
    // 1. Aliased pages near the top; a walk to UINTPTR_MAX must end (it wrapped to 0 and went on before). Run on a
    //    thread so a hang is reported as a failure instead of stopping the test.
    {
        namespace aliases = leonida::title_page_aliases;
        const std::uintptr_t top = UINTPTR_MAX & ~(aliases::kPageSize - 1);
        aliases::set(top, true);
        aliases::set(top - 4 * aliases::kPageSize, true);
        aliases::set(0x10000000, true);  // outside the range below
        std::atomic<bool> done{false};
        std::vector<std::uintptr_t> found;
        std::thread walk([&] {
            aliases::for_each_in(top - 8 * aliases::kPageSize, UINTPTR_MAX, [&](std::uintptr_t va) {
                if (found.size() < 64) found.push_back(va);
            });
            done = true;
        });
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!done && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        check(done, "for_each_in over a range ending at UINTPTR_MAX ends");
        if (!done) {
            std::printf("title_page_watch_bounds_test: FAILED (the walk did not end)\n");
            std::fflush(stdout);
            TerminateProcess(GetCurrentProcess(), 1);
        }
        walk.join();
        const bool has_top = std::find(found.begin(), found.end(), top) != found.end();
        const bool has_lower = std::find(found.begin(), found.end(), top - 4 * aliases::kPageSize) != found.end();
        check(found.size() == 2 && has_top && has_lower, "it finds the two aliased pages in the range, and not the third");
        found.clear();
        aliases::for_each_in(0x10000000 + 5, 0x10000000 + 6, [&](std::uintptr_t va) { found.push_back(va); });
        check(found.size() == 1 && found[0] == 0x10000000, "a range inside one aliased page finds it");
        aliases::set(top, false);
        aliases::set(top - 4 * aliases::kPageSize, false);
        aliases::set(0x10000000, false);
    }

    namespace ww = leonida::title_write_watch;
    const std::uint64_t reader = ww::open_reader();

    // 2. A reservation [base, base + 16 pages); a change from one page before it to one page into it.
    {
        const std::uintptr_t base = 0x7a0000000000;
        ww::add(base, 16 * ww::kPage);
        ww::mark(base - ww::kPage, base + ww::kPage);
        auto& s = ww::state();
        bool marked = false, outside = false;
        {
            std::lock_guard lock(s.journal);
            marked = s.written.count(base) != 0;
            outside = s.written.count(base - ww::kPage) != 0;
        }
        check(marked, "mark of a range starting before a reservation marks its first page");
        check(!outside, "and not the page before it, which is not write-watched");
        ww::remove(base);
    }

    // 3 and 4. A reservation that ends at the top of the address space.
    {
        const std::uintptr_t base = UINTPTR_MAX - 0x20000 + 1;  // 0x...fffe0000
        ww::add(base, 0x40000);                                // past the top: ends at UINTPTR_MAX
        check(ww::contains(base) && ww::contains(UINTPTR_MAX - 1),
              "add of a reservation past the top of the address space keeps it (its end does not wrap)");
        ww::mark(base + 3 * ww::kPage, UINTPTR_MAX);
        auto& s = ww::state();
        bool marked = false;
        {
            std::lock_guard lock(s.journal);
            marked = s.written.count(base + 3 * ww::kPage) != 0 && s.written.count(UINTPTR_MAX & ~(ww::kPage - 1)) != 0;
        }
        check(marked, "mark of a range ending at UINTPTR_MAX marks its pages (the rounding saturates)");
        ww::remove(base);
    }
    ww::close_reader(reader);

    // 5. Pins at the top of the address space.
    {
        namespace kw = leonida::kernel_writes;
        kw::set_opener(&opener);
        auto* const near_top = reinterpret_cast<void*>(UINTPTR_MAX - 0xfff);
        const std::uint64_t pin = kw::begin_write(near_top, 0x2000, nullptr);
        int seen = 0;
        kw::for_each_pin(UINTPTR_MAX - 0x1000, UINTPTR_MAX, [&](std::uintptr_t, std::uintptr_t) { ++seen; });
        check(pin != 0 && seen == 1, "begin_write past the top pins a range for_each_pin finds");
        kw::end_write(pin, false);
        const std::uint64_t held = kw::hold(near_top, 0x2000);
        seen = 0;
        kw::for_each_pin(UINTPTR_MAX - 0x1000, UINTPTR_MAX, [&](std::uintptr_t, std::uintptr_t) { ++seen; });
        check(held != 0 && seen == 1, "hold past the top pins a range for_each_pin finds");
        kw::release(held);
        kw::set_opener(nullptr);
    }

    // 6. A texel write noted past the top.
    {
        namespace tw = leonida::title_texel_writes;
        bool overflow = false;
        (void)tw::take(overflow);
        tw::note(reinterpret_cast<const void*>(UINTPTR_MAX - 0xfff), 0x2000);
        const auto ranges = tw::take(overflow);
        check(ranges.size() == 1 && ranges[0].first < ranges[0].second && ranges[0].second == UINTPTR_MAX,
              "a texel write past the top is kept as [begin, UINTPTR_MAX)");
    }

    std::printf("title_page_watch_bounds_test: %s (%d checks, %d failures)\n", g_failures ? "FAILED" : "OK", g_checks,
                g_failures);
    return g_failures ? 1 : 0;
}
