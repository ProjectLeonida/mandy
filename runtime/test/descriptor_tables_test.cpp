// BorderColorTable and ReplacedImages (runtime/src/descriptor_tables.h): the border colour table behind
// DeviceState::register_border_color and the replaced views forget_image releases. CPU only.
//
// Both are checked against copies of the code they replace (the linear scan of register_border_color and the
// std::erase_if of forget_image) on fixed cases and random sequences: every index, every released entry and the
// order of the releases must match. Then the old and new code are timed on the same sequences. `--bench` times
// larger tables.

#include "../src/descriptor_tables.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using leonida::BorderColorTable;
using leonida::ReplacedImages;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

// ---- the code before: register_border_color's scan, minus the lock and the log ----
struct OldBorderColors {
    struct CustomBorderColor {
        bool is_int = false;
        unsigned char value[16]{};
        bool operator==(const CustomBorderColor& o) const {
            return is_int == o.is_int && std::memcmp(&value, &o.value, sizeof(value)) == 0;
        }
    };
    std::vector<CustomBorderColor> table;
    bool last_full = false;

    std::uint32_t add(bool is_int, const void* components) {
        last_full = false;
        CustomBorderColor entry{};
        entry.is_int = is_int;
        std::memcpy(&entry.value, components, sizeof(entry.value));
        for (std::size_t i = 0; i < table.size(); ++i) {
            if (table[i] == entry) return std::uint32_t(i);
        }
        if (table.size() >= 4000) {
            last_full = true;
            return 3999;
        }
        const std::uint32_t idx = std::uint32_t(table.size());
        table.push_back(entry);
        return idx;
    }
};

struct Color {
    bool is_int;
    std::uint32_t bits[4];
};

bool same_tables(const OldBorderColors& old_table, const BorderColorTable& table) {
    if (old_table.table.size() != table.size()) return false;
    for (std::size_t i = 0; i < table.size(); ++i) {
        if (old_table.table[i].is_int != table[i].is_int) return false;
        if (std::memcmp(old_table.table[i].value, table[i].bits.data(), 16) != 0) return false;
    }
    return true;
}

// Runs `colors` through both and checks every answer.
void compare_colors(const std::vector<Color>& colors, const char* what) {
    OldBorderColors old_table;
    BorderColorTable table;
    bool ok = true;
    for (const Color& c : colors) {
        const std::uint32_t want = old_table.add(c.is_int, c.bits);
        const auto got = table.find_or_add(c.is_int, c.bits);
        ok &= want == got.index && old_table.last_full == got.full;
    }
    check(ok, what);
    check(same_tables(old_table, table), what);
}

std::uint32_t float_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

void border_fixed_cases() {
    BorderColorTable table;
    const float red[4] = {1, 0, 0, 1};
    const float half[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    const std::uint32_t red_int[4] = {float_bits(1), 0, 0, float_bits(1)};  // the same bytes as `red`, integer
    check(table.find_or_add(false, red).index == 0, "first colour gets index 0");
    check(table.find_or_add(false, half).index == 1, "second colour gets index 1");
    check(table.find_or_add(false, red).index == 0, "a repeated colour gets its first index");
    check(table.find_or_add(true, red_int).index == 2, "the integer flag is part of the key");
    check(table.size() == 3, "repeats add nothing");
    const float zero[4] = {0, 0, 0, 0.25f};
    const float negative_zero[4] = {-0.0f, 0, 0, 0.25f};
    check(table.find_or_add(false, zero).index == 3, "zero");
    check(table.find_or_add(false, negative_zero).index == 4, "-0.0 is not 0.0: the bytes differ, as with memcmp");
    const std::uint32_t nan[4] = {0x7fc00000u, 0, 0, 0};
    check(table.find_or_add(false, nan).index == 5 && table.find_or_add(false, nan).index == 5,
          "a NaN finds itself (bytes, not ==)");
    check(table[1].bits[0] == float_bits(0.5f) && !table[1].is_int && table[2].is_int, "entries read back");

    // The scan answers up to kScanMax entries, the hash index after; both find the old entries.
    check(!table.hashed(), "a small table scans");
    for (std::uint32_t i = 0; table.size() <= BorderColorTable::kScanMax; ++i) {
        const std::uint32_t c[4] = {i, 0xfeed, 0, 0};
        table.find_or_add(false, c);
    }
    check(table.hashed() && table.size() == BorderColorTable::kScanMax + 1, "the table hashes past kScanMax");
    check(table.find_or_add(false, red).index == 0 && table.find_or_add(false, negative_zero).index == 4 &&
              table.find_or_add(false, nan).index == 5 && table.size() == BorderColorTable::kScanMax + 1,
          "entries added while scanning are found through the hash");

    // Full: new colours clamp to 3999 and are not stored; old ones still find their index.
    std::vector<Color> fill;
    for (std::uint32_t i = 0; i < 4100; ++i) fill.push_back({(i & 1) != 0, {i, i * 7u, 0x12345678u, ~i}});
    for (std::uint32_t i = 0; i < 4100; i += 3) fill.push_back(fill[i]);
    compare_colors(fill, "a table filled past 4000 answers as the scan did");
    BorderColorTable full;
    for (std::uint32_t i = 0; i < 4000; ++i) {
        const std::uint32_t c[4] = {i, 1, 2, 3};
        full.find_or_add(false, c);
    }
    const std::uint32_t extra[4] = {9999, 1, 2, 3};
    const auto r = full.find_or_add(false, extra);
    check(r.index == 3999 && r.full && full.size() == 4000, "a new colour in a full table clamps to 3999");
    const std::uint32_t c17[4] = {17, 1, 2, 3};
    const auto again = full.find_or_add(false, c17);
    check(again.index == 17 && !again.full, "an old colour in a full table finds its index");
}

std::vector<Color> random_colors(std::mt19937_64& rng, std::size_t count, std::uint32_t distinct) {
    // `distinct` colours drawn from, so repeats are frequent; some differ in one byte or only in is_int.
    std::vector<Color> pool;
    for (std::uint32_t i = 0; i < distinct; ++i) {
        Color c{(rng() & 3) == 0, {std::uint32_t(rng()), std::uint32_t(rng()), std::uint32_t(rng()), std::uint32_t(rng())}};
        if (i && (rng() & 7) == 0) {
            c = pool[rng() % pool.size()];
            if (rng() & 1) c.is_int = !c.is_int;
            else c.bits[rng() % 4] ^= 1u << (rng() % 32);
        }
        pool.push_back(c);
    }
    std::vector<Color> out;
    for (std::size_t i = 0; i < count; ++i) out.push_back(pool[rng() % pool.size()]);
    return out;
}

void border_random_cases() {
    std::mt19937_64 rng(0x5eed1234);
    for (int round = 0; round < 200; ++round) {
        const std::uint32_t distinct = 1 + std::uint32_t(rng() % (round < 150 ? 300 : 5000));
        compare_colors(random_colors(rng, 1 + rng() % 6000, distinct), "random colour sequence");
    }
}

// ---- the code before: forget_image's std::erase_if over one vector ----
struct Entry {
    std::uint64_t address;
    std::uint32_t id;
};

struct OldReplaced {
    std::vector<Entry> list;
    void add(const Entry& e) { list.push_back(e); }
    template <class F>
    void take(std::uint64_t address, F&& release) {
        std::erase_if(list, [&](const Entry& e) {
            if (e.address != address) return false;
            release(e);
            return true;
        });
    }
};

void replaced_case(std::mt19937_64& rng, int ops, int addresses, bool& ok) {
    OldReplaced old_list;
    ReplacedImages<Entry> list;
    std::uint32_t next = 0;
    for (int i = 0; i < ops; ++i) {
        const std::uint64_t address = (std::uint64_t(1) << 32) + 256 * (rng() % addresses);
        if (rng() % 3) {
            const Entry e{address, next++};  // past kScanMax adds the list becomes keyed
            old_list.add(e);
            list.add(address, e);
        } else {
            std::vector<std::uint32_t> want, got;
            old_list.take(address, [&](const Entry& e) { want.push_back(e.id); });
            list.take(address, [&](const Entry& e) {
                got.push_back(e.id);
                ok &= e.address == address;
            });
            ok &= want == got;  // the same entries, in the same order
        }
        ok &= old_list.list.size() == list.size();
    }
}

void replaced_cases() {
    ReplacedImages<Entry> list;
    const std::uint64_t a = 0x100000000ull, b = 0x100000100ull;
    list.add(a, {a, 1});
    list.add(b, {b, 2});
    list.add(a, {a, 3});
    std::vector<std::uint32_t> got;
    list.take(a, [&](const Entry& e) { got.push_back(e.id); });
    check((got == std::vector<std::uint32_t>{1, 3}) && list.size() == 1, "an address releases its own, oldest first");
    got.clear();
    list.take(a, [&](const Entry& e) { got.push_back(e.id); });
    check(got.empty(), "a second take finds nothing");
    list.take(0x42, [&](const Entry& e) { got.push_back(e.id); });
    check(got.empty() && list.size() == 1, "an unknown address releases nothing");
    check(!list.keyed(), "a short list scans");
    for (std::uint32_t i = 0; i < ReplacedImages<Entry>::kScanMax; ++i) list.add(i % 5 ? a + 0x100000 + 256 * i : a, {a, 100 + i});
    check(list.keyed() && list.size() == ReplacedImages<Entry>::kScanMax + 1, "the list is keyed past kScanMax");
    got.clear();
    list.take(b, [&](const Entry& e) { got.push_back(e.id); });
    check((got == std::vector<std::uint32_t>{2}), "an entry from before the move is still released");
    got.clear();
    list.take(a, [&](const Entry& e) { got.push_back(e.id); });
    bool ordered = got.size() == (ReplacedImages<Entry>::kScanMax + 4) / 5;  // i % 5 == 0
    for (std::size_t i = 0; ordered && i < got.size(); ++i) ordered = got[i] == 100 + 5 * i;
    check(ordered, "keyed entries leave oldest first");

    std::mt19937_64 rng(0xabcdef);
    bool ok = true;
    for (int round = 0; round < 200; ++round) replaced_case(rng, 1 + int(rng() % 4000), 1 + int(rng() % 500), ok);
    check(ok, "random add/take sequences release what erase_if released, in its order");
}

// ---- timing ----
template <class F>
double seconds(F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

volatile std::uint64_t sink;

void bench_border(std::uint32_t distinct, std::size_t lookups) {
    std::mt19937_64 rng(distinct);
    std::vector<Color> pool;
    for (std::uint32_t i = 0; i < distinct; ++i)
        pool.push_back({false, {std::uint32_t(rng()), std::uint32_t(rng()), std::uint32_t(rng()), std::uint32_t(rng())}});
    std::vector<Color> seq;
    for (std::size_t i = 0; i < lookups; ++i) seq.push_back(pool[rng() % pool.size()]);
    OldBorderColors old_table;
    BorderColorTable table;
    for (const Color& c : pool) {  // both tables hold every colour before the timed lookups
        old_table.add(c.is_int, c.bits);
        table.find_or_add(c.is_int, c.bits);
    }
    std::uint64_t a = 0, b = 0;
    const double t_old = seconds([&] {
        for (const Color& c : seq) a += old_table.add(c.is_int, c.bits);
    });
    const double t_new = seconds([&] {
        for (const Color& c : seq) b += table.find_or_add(c.is_int, c.bits).index;
    });
    sink = a + b;
    check(a == b, "bench: same indices");
    std::printf("border colours %5u in table: scan %8.1f ns/lookup, hashed %6.1f ns/lookup, %7.1fx\n", distinct,
                t_old * 1e9 / double(lookups), t_new * 1e9 / double(lookups), t_old / t_new);
}

// `live` textures alive, each with `per` replaced views; every texture is destroyed once (forget_image).
void bench_replaced(int live, int per) {
    std::vector<std::uint64_t> addresses;
    for (int i = 0; i < live; ++i) addresses.push_back((std::uint64_t(1) << 32) + 256ull * std::uint64_t(i));
    std::mt19937_64 rng(std::uint64_t(live) * 31 + std::uint64_t(per));
    std::vector<std::uint64_t> order = addresses;
    std::shuffle(order.begin(), order.end(), rng);
    OldReplaced old_list;
    ReplacedImages<Entry> list;
    std::uint32_t id = 0;
    for (int k = 0; k < per; ++k)
        for (std::uint64_t a : addresses) {
            old_list.add({a, id});
            list.add(a, {a, id});
            ++id;
        }
    std::uint64_t x = 0, y = 0;
    const double t_old = seconds([&] {
        for (std::uint64_t a : order) old_list.take(a, [&](const Entry& e) { x += e.id; });
    });
    const double t_new = seconds([&] {
        for (std::uint64_t a : order) list.take(a, [&](const Entry& e) { y += e.id; });
    });
    sink = x + y;
    check(x == y && old_list.list.empty() && list.size() == 0, "bench: same releases");
    std::printf("replaced views %6d (%5d textures x %d): erase_if %9.1f ns/destroy, by address %6.1f ns/destroy, %7.1fx\n",
                live * per, live, per, t_old * 1e9 / live, t_new * 1e9 / live, t_old / t_new);
}

}  // namespace

int main(int argc, char** argv) {
    const bool big = argc > 1 && std::string(argv[1]) == "--bench";
    border_fixed_cases();
    border_random_cases();
    replaced_cases();
    for (std::uint32_t distinct : {1u, 4u, 16u, 64u, 256u, 1000u, 4000u})
        bench_border(distinct, big ? 2000000 : 200000);
    for (int live : {16, 64, 128, 256, 512, 1024, 4096}) bench_replaced(big ? live * 4 : live, 1);
    bench_replaced(big ? 4096 : 1024, 4);
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("descriptor_tables: all checks passed\n");
    return 0;
}
