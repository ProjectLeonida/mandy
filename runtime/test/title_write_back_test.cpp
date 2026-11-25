// The registry of title-window views a shader may write through: matching a scanned V# to a registered view (directly
// or through a rebased arena base), clipping and merging the marked ranges, and holding them for the next refresh
// boundary (src/title_write_back.h). Synthetic memory only: no GPU and no game data.
#include "../src/title_write_back.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace wb = leonida::title_write_back;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_failures;
}

}  // namespace

int main() {
    std::vector<std::uint8_t> first(64, 0x11), second(64, 0x22);
    wb::Registry registry;
    registry.add_window(0x10000, first.data(), first.size());
    registry.add_window(0x20000, second.data(), second.size());

    check(registry.empty(), "a fresh registry has no views");
    check(!registry.has_pending(), "a fresh registry has nothing to copy back");

    // A V# outside every view marks nothing, even when it is inside a window.
    check(!registry.mark_vsharp(0x10010, 0x10), "a V# no view registered is not marked");
    registry.boundary();
    check(!registry.has_pending(), "an unmatched V# leaves nothing to copy back");

    // A registered view's V# marks the view, clipped to the window that holds it and pinned to that copy.
    registry.add_view(0x10010, 0x20);
    check(registry.mark_vsharp(0x10010, 0x20), "a V# that names a registered view is marked");
    registry.boundary();
    auto ranges = registry.take_pending();
    check(ranges.size() == 1 && ranges[0].title_va == 0x10010 && ranges[0].shadow == first.data() + 0x10 &&
              ranges[0].bytes == 0x20,
          "the mark is pinned to the first window's copy");
    check(!registry.has_pending(), "taking the pending ranges clears them");

    // A V# that covers only part of a view marks only that part.
    check(registry.mark_vsharp(0x10018, 4), "a V# inside a view is marked");
    registry.boundary();
    ranges = registry.take_pending();
    check(ranges.size() == 1 && ranges[0].title_va == 0x10018 && ranges[0].bytes == 4,
          "only the part of the view the V# covers is marked");

    // Adjacent views merge, and adjacent marks merge in the boundary.
    registry.add_view(0x10030, 0x20);
    registry.add_view(0x10010, 0x20);
    registry.add_view(0x10000, 0x10);
    check(registry.mark_vsharp(0x10000, 0x10) && registry.mark_vsharp(0x10010, 0x20) &&
              registry.mark_vsharp(0x10030, 0x20),
          "each V# of the merged views is marked");
    registry.boundary();
    ranges = registry.take_pending();
    check(ranges.size() == 1 && ranges[0].title_va == 0x10000 && ranges[0].shadow == first.data() &&
              ranges[0].bytes == 0x40,
          "adjacent marks in one window become one run");

    // A view in the second window keeps that window's copy.
    registry.add_view(0x20008, 0x8);
    check(registry.mark_vsharp(0x20008, 0x8), "a V# in the second window is marked");
    registry.boundary();
    ranges = registry.take_pending();
    check(ranges.size() == 1 && ranges[0].title_va == 0x20008 && ranges[0].shadow == second.data() + 0x8 &&
              ranges[0].bytes == 0x8,
          "the second window's view keeps its own copy");

    // A rebased V# (its words now name the arena) still matches through the alias.
    registry.add_view(0x30010, 0x20);
    registry.add_window(0x30000, first.data(), first.size());
    registry.note_rebase(0x30010, 0x9F000010, 0x20);
    check(registry.mark_vsharp(0x9F000010, 0x20), "a rebased V# matches its view through the alias");
    registry.boundary();
    ranges = registry.take_pending();
    check(ranges.size() == 1 && ranges[0].title_va == 0x30010 && ranges[0].shadow == first.data() + 0x10 &&
              ranges[0].bytes == 0x20,
          "the alias carries the view's title range and copy");

    // Marks accumulate over boundaries while the windows are not refreshed.
    registry.add_view(0x40000, 0x10);
    registry.add_window(0x40000, first.data(), 0x20);
    registry.mark_vsharp(0x40000, 0x8);
    registry.boundary();
    registry.mark_vsharp(0x40008, 0x8);
    registry.boundary();
    ranges = registry.take_pending();
    check(ranges.size() == 1 && ranges[0].title_va == 0x40000 && ranges[0].bytes == 0x10,
          "marks from two boundaries merge until they are taken");

    // A window made after its view (fixing a console heap can add one) still holds it.
    {
        wb::Registry late;
        std::vector<std::uint8_t> third(64, 0x33);
        late.add_view(0x50010, 0x10);
        late.mark_vsharp(0x50010, 0x10);
        late.boundary();
        check(!late.has_pending(), "before its window exists a view marks nothing");
        late.add_window(0x50000, third.data(), third.size());
        check(late.mark_vsharp(0x50010, 0x10), "a window added later still holds its view");
        late.boundary();
        ranges = late.take_pending();
        check(ranges.size() == 1 && ranges[0].title_va == 0x50010 && ranges[0].shadow == third.data() + 0x10 &&
                  ranges[0].bytes == 0x10,
              "the late window's view is pinned to its copy");
    }

    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("title_write_back: OK\n");
    return 0;
}
