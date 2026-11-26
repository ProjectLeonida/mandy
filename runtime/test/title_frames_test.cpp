// src/title_frames.h: the host's reads and writes of title pages that only xmem's frame table holds (queue slot 31's
// copies onto a stream pool's GPU-only range and its CPU twin, which get no host view). No Windows headers, no GPU, no
// game data: the "frames" are arrays here, the table a map, and the addresses made up.
#include "../src/title_frames.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace {

int failures = 0;
#define EXPECT(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                      \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);    \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

namespace frames = leonida::title_frames;
constexpr std::uintptr_t kPage = frames::kPage;

// The fake xmem: a table of page address -> frame (one physical page each), and each frame's host views.
struct Table {
    std::map<std::uintptr_t, std::shared_ptr<std::vector<std::uint8_t>>> pages;
    std::map<std::uintptr_t, std::vector<std::uintptr_t>> views;
    unsigned lookups = 0;
};
Table g_table;

bool lookup(std::uintptr_t page, frames::Frame& out) {
    ++g_table.lookups;
    const auto it = g_table.pages.find(page);
    if (it == g_table.pages.end()) return false;
    out.bytes = it->second->data();
    out.keep = it->second;
    return true;
}
void peers(std::uintptr_t page, std::vector<std::uintptr_t>& out) {
    const auto it = g_table.views.find(page);
    if (it != g_table.views.end()) out.insert(out.end(), it->second.begin(), it->second.end());
}

// A copy of [from, from + n) as the D3D12 layer's copy_guarded makes it: a table page from its frame, the rest from
// `host`, a stand-in for the title's own memory at those addresses.
bool copy(std::uint8_t* to, std::uintptr_t from, std::size_t n, const std::uint8_t* host, std::uintptr_t host_base) {
    return frames::for_each_part(from, n, [&](std::uintptr_t at, std::size_t bytes, std::uint8_t* frame) {
        std::memcpy(to + (at - from), frame ? frame : host + (at - host_base), bytes);
        return true;
    });
}

std::shared_ptr<std::vector<std::uint8_t>> frame_of(std::uint8_t fill) {
    return std::make_shared<std::vector<std::uint8_t>>(kPage, fill);
}

}  // namespace

int main() {
    constexpr std::uintptr_t base = 0x7f0000000000;  // a GPU-only range's first page (made up)
    std::vector<std::uint8_t> host(8 * kPage, 0xee);  // what the host shows where no table page is

    // Before xmem registers its table nothing is a table page, and no range costs a lookup.
    EXPECT(!frames::may_hold(base, base + kPage) && !frames::backed(base));
    frames::set_lookup(&lookup, &peers);
    EXPECT(!frames::may_hold(base, base + kPage));  // still no page in the bounds
    g_table.lookups = 0;
    std::vector<std::uint8_t> out(3 * kPage);
    EXPECT(copy(out.data(), base, out.size(), host.data(), base) && out[0] == 0xee && g_table.lookups == 0);

    // Pages 1 and 2 get frames (a slot-31 copy): the copy reads them there, the host around them.
    auto a = frame_of(0xa1), b = frame_of(0xb2);
    g_table.pages[base + kPage] = a;
    g_table.pages[base + 2 * kPage] = b;
    frames::note_page(base + kPage);
    frames::note_page(base + 2 * kPage);
    EXPECT(frames::may_hold(base, base + kPage + 1) && !frames::may_hold(base + 3 * kPage, base + 4 * kPage));
    EXPECT(!frames::backed(base) && frames::backed(base + kPage + 0x1234) && frames::backed(base + 3 * kPage - 1));
    frames::Frame f;
    EXPECT(frames::find(base + kPage + 0x10, f) && f.bytes == a->data() + 0x10 && f.keep);
    // An unaligned run over all four pages: host bytes, a's, b's, host bytes again.
    std::vector<std::uint8_t> run(3 * kPage);
    const std::uintptr_t from = base + 0x8000;
    EXPECT(copy(run.data(), from, run.size(), host.data(), base));
    EXPECT(run[0] == 0xee && run[0x7fff] == 0xee && run[0x8000] == 0xa1 && run[0x17fff] == 0xa1 &&
           run[0x18000] == 0xb2 && run[0x27fff] == 0xb2 && run[0x28000] == 0xee && run.back() == 0xee);
    // The parts come in order, a table page's inside its own 64 KiB, neighbouring host pages as one.
    std::vector<std::pair<std::uintptr_t, bool>> parts;
    frames::for_each_part(base, 4 * kPage, [&](std::uintptr_t at, std::size_t n, std::uint8_t* frame) {
        parts.emplace_back(at, frame != nullptr);
        EXPECT(!frame || n == kPage);
        return true;
    });
    EXPECT(parts.size() == 4 && !parts[0].second && parts[1].second && parts[2].second && !parts[3].second &&
           parts[3].first == base + 3 * kPage);
    bool backed = false;
    EXPECT(frames::run_end(base, base + 4 * kPage, backed) == base + kPage && !backed);
    EXPECT(frames::run_end(base + kPage + 4, base + 4 * kPage, backed) == base + 3 * kPage && backed);
    EXPECT(frames::run_end(base + 3 * kPage, base + 4 * kPage, backed) == base + 4 * kPage && !backed);

    // A remap is a table update: the next read sees the new frame, with nothing else to tell.
    auto c = frame_of(0xc3);
    g_table.pages[base + kPage] = c;
    EXPECT(copy(run.data(), from, run.size(), host.data(), base) && run[0x8000] == 0xc3 && run[0x18000] == 0xb2);
    // A frame a reader holds stays alive after the table lets it go (the section's scratch view stays mapped).
    EXPECT(frames::find(base + 2 * kPage, f));
    g_table.pages.erase(base + 2 * kPage);
    b.reset();
    EXPECT(f.bytes[0] == 0xb2 && !frames::backed(base + 2 * kPage));
    f = {};
    // The title's CPU writes the frame through its own mapping (here: the frame itself): the reader sees it, and asks
    // xmem for that mapping to guard it there.
    (*c)[0x40] = 0x55;
    EXPECT(copy(run.data(), from, run.size(), host.data(), base) && run[0x8040] == 0x55);
    g_table.views[base + kPage] = {0x10000000};
    std::vector<std::uintptr_t> others;
    frames::peers(base + kPage + 0x40, others);
    EXPECT(others.size() == 1 && others[0] == 0x10000000);

    // A part that fails stops the walk and says so.
    unsigned calls = 0;
    EXPECT(!frames::for_each_part(base, 4 * kPage, [&](std::uintptr_t, std::size_t, std::uint8_t*) { return ++calls < 2; }));
    EXPECT(calls == 2);
    // Empty and top-of-memory ranges.
    EXPECT(frames::for_each_part(base, 0, [](std::uintptr_t, std::size_t, std::uint8_t*) { return false; }));
    EXPECT(frames::for_each_part(UINTPTR_MAX - 16, 64, [](std::uintptr_t at, std::size_t n, std::uint8_t* frame) {
        return !frame && at == UINTPTR_MAX - 16 && n == 16;
    }));

    if (failures) {
        std::fprintf(stderr, "title_frames_test: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("title_frames_test: table pages read through their frames, host pages around them, remaps and frees seen");
    return 0;
}
