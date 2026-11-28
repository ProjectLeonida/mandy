// add_reached_range (src/title_write_ranges.h): the title-memory ranges a submission reached, kept for the copy back
// (DeviceState::note_title_write). kevincommits' finding in #1608: a range that began before an entry lost its prefix,
// a range that spanned entries left them overlapping, and the byte count missed what a merge added. No device, no
// game data: plain numbers.
#include "../src/title_write_ranges.h"

#include <cstdio>
#include <map>
#include <utility>
#include <vector>

namespace {

using Ranges = std::map<std::uint64_t, std::uint64_t>;

bool check(bool condition, const char* description) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", description);
    return condition;
}

struct Recorder {
    Ranges ranges;
    std::uint64_t bytes = 0;
    bool add(std::uint64_t start, std::uint64_t end, std::size_t max_ranges = 8192, std::uint64_t max_bytes = 1ull << 40) {
        return leonida::add_reached_range(ranges, bytes, start, end, max_ranges, max_bytes);
    }
    bool is(std::vector<std::pair<std::uint64_t, std::uint64_t>> want) const {
        return Ranges(want.begin(), want.end()) == ranges;
    }
};

}  // namespace

int main() {
    bool ok = true;

    {  // kevincommits' first repro: the prefix before an existing entry
        Recorder r;
        r.add(200, 300);
        r.add(100, 250);
        ok &= check(r.is({{100, 300}}), "[200,300) then [100,250) keeps [100,300), not only [200,300)");
        ok &= check(r.bytes == 200, "and counts 200 bytes");
    }
    {  // kevincommits' second repro: a range spanning two entries
        Recorder r;
        r.add(100, 200);
        r.add(300, 400);
        r.add(150, 350);
        ok &= check(r.is({{100, 400}}), "[100,200), [300,400) then [150,350) leaves one disjoint [100,400)");
        ok &= check(r.bytes == 300, "and counts the union's 300 bytes, not 200");
    }
    {  // a range covering several entries and reaching past both ends
        Recorder r;
        r.add(10, 20);
        r.add(30, 40);
        r.add(50, 60);
        r.add(5, 70);
        ok &= check(r.is({{5, 70}}) && r.bytes == 65, "a range over three entries absorbs them all");
    }
    {  // extending an entry forward across the next one
        Recorder r;
        r.add(100, 200);
        r.add(250, 300);
        r.add(180, 260);
        ok &= check(r.is({{100, 300}}) && r.bytes == 200, "extending the previous entry absorbs the next");
    }
    {  // touching ranges join; a gap keeps them apart; a covered range changes nothing
        Recorder r;
        r.add(0, 100);
        r.add(100, 200);
        r.add(300, 400);
        r.add(120, 180);
        ok &= check(r.is({{0, 200}, {300, 400}}) && r.bytes == 300, "touching joins, a gap stays, covered adds nothing");
    }
    {  // the byte cap holds for merges too
        Recorder r;
        ok &= check(r.add(0, 100, 8192, 100), "the first 100 bytes fit a 100-byte cap");
        ok &= check(!r.add(50, 200, 8192, 100), "a merge that adds bytes past the cap is refused");
        ok &= check(r.is({{0, 100}}) && r.bytes == 100, "and leaves the ranges as they were");
        ok &= check(r.add(10, 90, 8192, 100), "a range already covered still fits");
    }
    {  // the byte cap is soft: the range that crosses it goes in whole, the next byte past it does not (akimfomin, #2517)
        Recorder r;
        ok &= check(r.add(0, 63, 8192, 64) && r.bytes == 63, "[0,63) fits a 64-byte cap");
        ok &= check(r.add(128, 132, 8192, 64), "a disjoint [128,132) still goes in: 63 bytes is under the cap");
        ok &= check(r.is({{0, 63}, {128, 132}}) && r.bytes == 67, "and is kept whole: 63 -> 67 bytes");
        ok &= check(!r.add(200, 201, 8192, 64), "the next range that adds a byte is refused");
        ok &= check(!r.add(60, 70, 8192, 64), "so is a merge that adds bytes");
        ok &= check(r.is({{0, 63}, {128, 132}}) && r.bytes == 67, "and refusals leave the map and count unchanged");
        ok &= check(r.add(129, 131, 8192, 64) && r.bytes == 67, "a covered range still fits");
    }
    {  // the range cap refuses only new entries
        Recorder r;
        r.add(0, 10, 2);
        r.add(20, 30, 2);
        ok &= check(!r.add(40, 50, 2), "a third entry past a two-range cap is refused");
        ok &= check(r.add(5, 25, 2) && r.is({{0, 30}}), "a range joining entries fits the range cap");
        ok &= check(r.add(40, 50, 2) && r.is({{0, 30}, {40, 50}}), "and frees room for a new one");
    }

    if (ok) std::printf("title_write_ranges_test: PASS\n");
    return ok ? 0 : 1;
}
