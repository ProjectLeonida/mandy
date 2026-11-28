// The pipeline journal's file naming, reading and writing (pipeline_cache_journal.h). No Vulkan, no GPU.
#include "../src/pipeline_cache_journal.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace journal = leonida::pipeline_journal;
namespace store = leonida::pipeline_cache_store;

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

static void put32(std::uint8_t* p, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(value >> (8 * i));
}

static leonida::cached_pso::Identity identity() {
    leonida::cached_pso::Identity id;
    id.vendor = 0x10de;
    id.device = 0x2782;
    for (int i = 0; i < 16; ++i) id.cache_uuid[i] = static_cast<std::uint8_t>(0xa0 + i);
    return id;
}

// A VkPipelineCacheHeaderVersionOne for `id` followed by `payload` bytes.
static std::vector<char> blob(const leonida::cached_pso::Identity& id, std::size_t payload) {
    std::vector<char> out(store::header_bytes + payload, '\x5a');
    auto* p = reinterpret_cast<std::uint8_t*>(out.data());
    put32(p, store::header_bytes);
    put32(p + 4, store::header_version_one);
    put32(p + 8, id.vendor);
    put32(p + 12, id.device);
    std::memcpy(p + 16, id.cache_uuid, sizeof(id.cache_uuid));
    return out;
}

static int test_hash() {
    const std::uint32_t words[] = {0x07230203, 0x00010600, 1, 2, 3};
    journal::Hasher a, b, c;
    a.text("cs");
    a.bytes(words, sizeof(words));
    a.value(std::uint32_t(32));
    b.text("cs");
    b.bytes(words, sizeof(words));
    b.value(std::uint32_t(32));
    CHECK(a.digest() == b.digest());  // deterministic: the next boot finds the same file
    c.text("cs");
    c.bytes(words, sizeof(words));
    c.value(std::uint32_t(64));  // another subgroup size is another pipeline
    CHECK(a.digest() != c.digest());
    journal::Hasher d, e;
    d.text("ab");
    d.text("c");
    e.text("a");
    e.text("bc");
    CHECK(d.digest() != e.digest());  // text is length-prefixed
    return 0;
}

static int test_path() {
    const auto path = journal::entry_path("dir", "cs", 0x0123456789abcdefull);
    CHECK(path.filename().string() == "cs-0123456789abcdef.bin");
    CHECK(journal::entry_path("dir", "cs", 1).filename().string() == "cs-0000000000000001.bin");
    return 0;
}

static int test_files(const fs::path& dir) {
    const auto id = identity();
    const auto path = journal::entry_path(dir / "pipeline_cache.d", "cs", 42);
    CHECK(journal::read_entry(path, id).empty());  // missing

    // write_entry creates the directory and publishes the whole blob.
    const auto data = blob(id, 1000);
    CHECK(!journal::write_entry(path, data.data(), data.size()));
    CHECK(journal::read_entry(path, id) == data);

    // A second write replaces the first.
    const auto next = blob(id, 10);
    CHECK(!journal::write_entry(path, next.data(), next.size()));
    CHECK(journal::read_entry(path, id) == next);

    // Another device or driver (pipeline-cache UUID): not read, so the build starts over and rewrites it.
    auto other = id;
    other.cache_uuid[3] ^= 1;
    CHECK(journal::read_entry(path, other).empty());
    other = id;
    other.device = 0x1234;
    CHECK(journal::read_entry(path, other).empty());

    // Truncated to less than a header, or over the read limit: not read.
    {
        std::ofstream(path, std::ios::binary | std::ios::trunc).write(next.data(), 16);
    }
    CHECK(journal::read_entry(path, id).empty());
    CHECK(!journal::write_entry(path, data.data(), data.size()));
    CHECK(journal::read_entry(path, id, data.size() - 1).empty());
    CHECK(journal::read_entry(path, id, data.size()) == data);

    CHECK(journal::directory_bytes(path.parent_path()) == data.size());
    CHECK(journal::directory_bytes(dir / "missing") == 0);

    // No staging files are left behind.
    std::size_t files = 0;
    for (const auto& item : fs::directory_iterator(path.parent_path())) {
        (void)item;
        ++files;
    }
    CHECK(files == 1);
    return 0;
}

// When a build writes its file, and when the shared cache's answer was a compile.
static int test_policy() {
    // No file (or one the driver declined): always written, whatever the feedback or time.
    CHECK(journal::should_write(false, 1, 0));
    CHECK(journal::should_write(false, -1, 0));
    CHECK(journal::should_write(false, 0, 0));
    // From the file: the feedback decides; without feedback, the length of the driver call.
    CHECK(!journal::should_write(true, 1, 5000));
    CHECK(journal::should_write(true, 0, 1));
    CHECK(!journal::should_write(true, -1, journal::unreported_compile_ms - 1));
    CHECK(journal::should_write(true, -1, journal::unreported_compile_ms));
    // FAIL_ON_PIPELINE_COMPILE_REQUIRED: a slow pipeline without the hit bit was compiled; a fast one came out of
    // the driver's own disk cache, and a hit or no feedback is never taken for a compile.
    CHECK(journal::probe_compiled(0, journal::probe_compile_ms));
    CHECK(!journal::probe_compiled(0, journal::probe_compile_ms - 1));
    CHECK(!journal::probe_compiled(1, 60000));
    CHECK(!journal::probe_compiled(-1, 60000));
    return 0;
}

static int test_accounting() {
    CHECK(journal::replace_bytes(100, 0, 30) == 130);    // a new file
    CHECK(journal::replace_bytes(100, 40, 30) == 90);    // a rewrite replaces the old size
    CHECK(journal::replace_bytes(10, 40, 30) == 30);     // never below zero
    const auto t0 = fs::file_time_type::clock::now();
    using std::chrono::hours;
    const std::vector<journal::EntryInfo> entries = {
        {"a", 40, t0 - hours(3)}, {"b", 40, t0 - hours(1)}, {"c", 40, t0 - hours(2)}, {"d", 40, t0}};
    CHECK(journal::plan_eviction(entries, 160, 120).empty());  // at the budget: nothing goes
    // Over it: least recently used first (a, then c), until the target holds.
    const auto out = journal::plan_eviction(entries, 150, 100);
    CHECK(out.size() == 2 && out[0] == 0 && out[1] == 2);
    CHECK(journal::plan_eviction(entries, 0, 0).size() == 4);
    return 0;
}

static int test_names() {
    CHECK(journal::entry_name("cs-0123456789abcdef.bin"));
    CHECK(!journal::entry_name("cs-0123456789abcde.bin"));    // 15 digits
    CHECK(!journal::entry_name("cs-0123456789ABCDEF.bin"));   // entry_path writes lowercase
    CHECK(!journal::entry_name("-0123456789abcdef.bin"));
    CHECK(!journal::entry_name("pipeline_cache.session.bin"));
    CHECK(journal::staging_name("cs-0123456789abcdef.bin.tmp.123456.0"));
    CHECK(!journal::staging_name("cs-0123456789abcdef.bin.tmp.123456"));
    CHECK(!journal::staging_name("cs-0123456789abcdef.bin.tmp.12x.0"));
    CHECK(!journal::staging_name("pipeline_cache.session.bin.tmp.1.0"));  // the session file's, swept elsewhere
    return 0;
}

static void put_file(const fs::path& path, std::size_t bytes, fs::file_time_type when) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << std::string(bytes, 'x');
    fs::last_write_time(path, when);
}

static int test_tidy(const fs::path& root) {
    const auto dir = root / "tidy";
    fs::create_directories(dir);
    const auto now = fs::file_time_type::clock::now();
    using std::chrono::hours;
    using std::chrono::minutes;
    const auto a = journal::entry_path(dir, "cs", 0xa), b = journal::entry_path(dir, "cs", 0xb),
               c = journal::entry_path(dir, "cs", 0xc);
    put_file(a, 400, now - hours(5));
    put_file(b, 400, now - hours(1));
    put_file(c, 400, now - hours(3));
    const fs::path old_stage = a.string() + ".tmp.111.0", new_stage = b.string() + ".tmp.222.1";
    put_file(old_stage, 50, now - minutes(30));
    put_file(new_stage, 50, now - minutes(2));
    put_file(dir / "notes.txt", 999, now - hours(9));

    // Under the budget: only the old staging file goes; the young one may still be being written.
    auto result = journal::tidy(dir, 2000, minutes(10), now);
    CHECK(result.swept == 1 && result.evicted == 0 && result.entries == 3 && result.bytes == 1200);
    CHECK(!fs::exists(old_stage) && fs::exists(new_stage) && fs::exists(dir / "notes.txt"));

    // Over it: the least recently used go until three quarters is left (1200 > 1000; 750 holds b alone).
    result = journal::tidy(dir, 1000, minutes(10), now);
    CHECK(result.evicted == 2 && result.entries == 1 && result.bytes == 400);
    CHECK(!fs::exists(a) && !fs::exists(c) && fs::exists(b));

    // A hit refreshes a file's last use, which keeps it over older ones.
    put_file(a, 400, now - hours(5));
    journal::touch(a, now);
    put_file(c, 400, now - hours(3));
    result = journal::tidy(dir, 1000, minutes(10), now);  // 1200 again: c (3 h) is now the oldest, then b (1 h)
    CHECK(fs::exists(a) && !fs::exists(c) && !fs::exists(b) && result.entries == 1);
    CHECK(journal::tidy(root / "missing", 1000).entries == 0);
    return 0;
}

static int test_entry_file(const fs::path& dir) {
    const auto id = identity();
    const auto path = journal::entry_path(dir, "cs", 77);
    auto missing = journal::read_entry_file(path, id);
    CHECK(missing.data.empty() && missing.bytes == 0);
    // A foreign file still reports its size, which its rewrite replaces in the budget.
    auto other = id;
    other.device ^= 1;
    const auto data = blob(other, 100);
    CHECK(!journal::write_entry(path, data.data(), data.size()));
    const auto foreign = journal::read_entry_file(path, id);
    CHECK(foreign.data.empty() && foreign.bytes == data.size());
    const auto own = journal::read_entry_file(path, other);
    CHECK(own.data == data && own.bytes == data.size());
    return 0;
}

// The writer stamp before the blob: read and stripped, files without one read as before, and only a file this build
// wrote under this device-create config is current (what gta6rt --precompile-pipelines leaves alone).
static int test_stamp(const fs::path& dir) {
    const auto id = identity();
    constexpr std::uint64_t ours = 0x0123456789abcdefull, theirs = 0xfedcba9876543210ull;
    constexpr std::uint64_t plain = 0x1111222233334444ull, diagnostics = 0x5555666677778888ull;
    const journal::Writer we{ours, plain}, other_build{theirs, plain}, other_config{ours, diagnostics};

    // Round trip; the stamp is its own size and does not look like a blob header.
    std::uint8_t raw[journal::stamp_bytes];
    static_assert(journal::stamp_bytes == 24);
    journal::put_stamp(raw, we);
    auto stamp = journal::read_stamp(raw, sizeof(raw));
    CHECK(stamp.present && stamp.build == ours && stamp.config == plain);
    CHECK(std::memcmp(raw, "LPJ1", 4) == 0);
    CHECK(!journal::read_stamp(raw, sizeof(raw) - 1).present);
    CHECK(!store::header_matches(raw, sizeof(raw), id));
    const auto bare = blob(id, 64);
    CHECK(!journal::read_stamp(bare.data(), bare.size()).present);  // a blob alone: no stamp
    // The 16-byte stamp of an earlier draft (size field 16) is not this stamp.
    std::uint8_t old[journal::stamp_bytes];
    journal::put_stamp(old, we);
    old[4] = 16;
    CHECK(!journal::read_stamp(old, sizeof(old)).present);

    // A stamped file: read_entry gives the blob alone, with the stamp beside it.
    const auto path = journal::entry_path(dir / "stamp", "cs", 5);
    const auto file = journal::stamped(we, bare.data(), bare.size());
    CHECK(file.size() == journal::stamp_bytes + bare.size());
    CHECK(!journal::write_entry(path, file.data(), file.size()));
    auto entry = journal::read_entry_file(path, id);
    CHECK(entry.data == bare && entry.bytes == file.size() && entry.stamp.present && entry.stamp.build == ours &&
          entry.stamp.config == plain);
    CHECK(journal::read_entry(path, id) == bare);
    CHECK(!entry.other_build(ours) && entry.other_build(theirs) && entry.other_build(0));
    CHECK(!entry.other_config(plain) && entry.other_config(diagnostics));
    CHECK(!entry.other_writer(we) && entry.other_writer(other_build) && entry.other_writer(other_config));
    CHECK(journal::entry_usable(path, id));
    CHECK(journal::entry_current(path, id, we));
    CHECK(!journal::entry_current(path, id, other_build));   // another build wrote it: built against again
    CHECK(!journal::entry_current(path, id, other_config));  // another device config (diagnostics on): the same
    CHECK(!journal::entry_current(path, id, {0, plain}));    // a build that does not know itself never skips
    auto other = id;
    other.cache_uuid[0] ^= 1;
    CHECK(!journal::entry_usable(path, other) && !journal::entry_current(path, other, we));
    entry = journal::read_entry_file(path, other);
    CHECK(entry.data.empty() && !entry.stamp.present && !entry.other_build(theirs) && !entry.other_config(diagnostics));
    // The read limit counts the stamp: it is what is on disk.
    CHECK(journal::read_entry(path, id, file.size() - 1).empty() && !journal::entry_usable(path, id, file.size() - 1));
    CHECK(journal::entry_current(path, id, we, file.size()));

    // A stamp for build 0 (a writer that could not read itself) is never current.
    const auto zero = journal::stamped({0, plain}, bare.data(), bare.size());
    CHECK(!journal::write_entry(path, zero.data(), zero.size()));
    CHECK(journal::entry_usable(path, id) && !journal::entry_current(path, id, {0, plain}));
    CHECK(journal::read_entry_file(path, id).other_build(0));

    // Written under another device config by this build: other_config, not other_build.
    const auto diag = journal::stamped(other_config, bare.data(), bare.size());
    CHECK(!journal::write_entry(path, diag.data(), diag.size()));
    entry = journal::read_entry_file(path, id);
    CHECK(!entry.other_build(ours) && entry.other_config(plain) && entry.other_writer(we));
    CHECK(journal::entry_current(path, id, other_config) && !journal::entry_current(path, id, we));

    // A file an older runtime wrote (no stamp): still usable, read as from another build (its config unknown, so not
    // also counted as another config), never current.
    CHECK(!journal::write_entry(path, bare.data(), bare.size()));
    entry = journal::read_entry_file(path, id);
    CHECK(entry.data == bare && !entry.stamp.present && entry.other_build(ours) && !entry.other_config(plain));
    CHECK(entry.other_writer(we));
    CHECK(journal::entry_usable(path, id) && !journal::entry_current(path, id, we));

    // A stamp then less than a header, or a stamp then a foreign blob: not usable.
    CHECK(!journal::write_entry(path, file.data(), journal::stamp_bytes + 8));
    CHECK(journal::read_entry(path, id).empty() && !journal::entry_usable(path, id));
    const auto foreign = blob(other, 64);
    const auto foreign_file = journal::stamped(we, foreign.data(), foreign.size());
    CHECK(!journal::write_entry(path, foreign_file.data(), foreign_file.size()));
    CHECK(journal::read_entry(path, id).empty() && !journal::entry_usable(path, id));
    CHECK(journal::read_entry(path, other) == foreign && journal::entry_current(path, other, we));

    // A seed file's stamp comes with its data.
    const auto local = dir / "stamp-local" / "pipeline_cache.d", seed = dir / "stamp-seed";
    const auto seed_file = journal::stamped({theirs, diagnostics}, bare.data(), bare.size());
    CHECK(!journal::write_entry(journal::entry_path(seed, "cs", 6), seed_file.data(), seed_file.size()));
    const auto roots = journal::resolve_roots(local, seed);
    entry = journal::read_with_seed(journal::entry_path(local, "cs", 6), roots.seed, id, journal::ReadRole::runtime);
    CHECK(entry.from_seed && entry.data == bare && entry.stamp.present && entry.stamp.build == theirs &&
          entry.stamp.config == diagnostics);
    CHECK(entry.other_build(ours) && !entry.other_build(theirs) && entry.other_config(plain));

    // A binary's build: its contents, not its name or folder.
    const auto a = dir / "bin" / "gta6rt.exe", copy = dir / "bin2" / "renamed.exe", rebuilt = dir / "bin" / "b.exe";
    std::vector<char> exe(3u << 20);  // more than one read chunk
    for (std::size_t i = 0; i < exe.size(); ++i) exe[i] = static_cast<char>(i * 2654435761u >> 13);
    CHECK(!journal::write_entry(a, exe.data(), exe.size()));
    CHECK(!journal::write_entry(copy, exe.data(), exe.size()));
    exe[exe.size() / 2] ^= 1;  // a one-byte change
    CHECK(!journal::write_entry(rebuilt, exe.data(), exe.size()));
    const auto build_a = journal::file_build(a);
    CHECK(build_a != 0 && build_a == journal::file_build(a) && build_a == journal::file_build(copy));
    CHECK(journal::file_build(rebuilt) != 0 && journal::file_build(rebuilt) != build_a);
    CHECK(!journal::write_entry(rebuilt, exe.data(), exe.size() - 1));  // the same bytes, one fewer
    CHECK(journal::file_build(rebuilt) != build_a);
    CHECK(journal::file_build(dir / "bin" / "missing.exe") == 0);
    CHECK(!journal::write_entry(rebuilt, exe.data(), 0));
    CHECK(journal::file_build(rebuilt) == 0);  // empty: unknown
    return 0;
}

static int test_overlay(const fs::path& dir) {
    using std::chrono::hours;
    const auto local = dir / "overlay" / "pipeline_cache.d", seed = dir / "seed";
    const auto dst = journal::entry_path(local, "cs", 1), src = journal::entry_path(seed, "cs", 1);
    const auto id = identity();
    auto wrong = id;
    wrong.cache_uuid[0] ^= 1;
    const auto donor = blob(id, 32), rebuilt = blob(id, 53), stale = blob(wrong, 45);
    CHECK(!journal::write_entry(src, donor.data(), donor.size()));
    const auto now = fs::file_time_type::clock::now();
    fs::last_write_time(src, now - hours(8));
    const auto donor_time = fs::last_write_time(src);
    const auto roots = journal::resolve_roots(local, seed);
    CHECK(!roots.writable.empty() && fs::equivalent(roots.seed, seed));

    // No local entry: donor bytes may seed the driver, but do not count against the writable budget.
    constexpr auto runtime = journal::ReadRole::runtime;
    constexpr auto builder = journal::ReadRole::materialize_local;
    auto entry = journal::read_with_seed(dst, roots.seed, id, runtime);
    CHECK(entry.from_seed && entry.data == donor && entry.bytes == 0 && entry.used == fs::file_time_type{});
    // Same roots, but the host building a requested local file must still see a miss.
    CHECK(!journal::entry_usable(dst, id) && journal::entry_usable(src, id));
    const auto missing_build = journal::read_with_seed(dst, roots.seed, id, builder);
    CHECK(missing_build.data.empty() && !missing_build.from_seed && missing_build.bytes == 0);
    journal::touch_local_hit(dst, entry, now);
    CHECK(!fs::exists(local) && fs::last_write_time(src) == donor_time);
    CHECK(journal::replace_bytes(1000, entry.bytes, rebuilt.size()) == 1000 + rebuilt.size());
    CHECK(journal::read_with_seed(dst, {}, id, runtime).data.empty());
    CHECK(journal::read_with_seed(dst, seed, wrong, runtime).data.empty());
    CHECK(journal::read_with_seed(dst, seed, id, runtime, donor.size() - 1).data.empty());
    CHECK(journal::read_with_seed(dst, seed, id, runtime, donor.size()).data == donor);
    CHECK(journal::read_with_seed(journal::entry_path(local, "cs", 2), seed, id, runtime).data.empty());

    // Even with a stale local object the recorded byte count and LRU timestamp belong to that object.
    CHECK(!journal::write_entry(dst, stale.data(), stale.size()));
    fs::last_write_time(dst, now - hours(3));
    const auto stale_time = fs::last_write_time(dst);
    entry = journal::read_with_seed(dst, seed, id, runtime);
    CHECK(entry.from_seed && entry.data == donor && entry.bytes == stale.size() && entry.used == stale_time);
    const auto stale_build = journal::read_with_seed(dst, roots.seed, id, builder);
    CHECK(stale_build.data.empty() && !stale_build.from_seed && stale_build.bytes == stale.size() &&
          stale_build.used == stale_time && !journal::entry_usable(dst, id));
    journal::touch_local_hit(dst, entry, now);
    CHECK(fs::last_write_time(dst) == stale_time && fs::last_write_time(src) == donor_time);
    CHECK(journal::read_entry(dst, wrong) == stale);  // neither promoted nor rewritten by the hit
    auto used = journal::tidy(roots.writable, stale.size());
    CHECK(used.entries == 1 && used.bytes == stale.size() && used.evicted == 0);
    CHECK(journal::replace_bytes(used.bytes, entry.bytes, rebuilt.size()) == rebuilt.size());

    CHECK(!journal::write_entry(dst, rebuilt.data(), 3));  // truncated local is not a seed hit on the local file
    entry = journal::read_with_seed(dst, seed, id, runtime);
    CHECK(entry.from_seed && entry.data == donor && entry.bytes == 3 && fs::file_size(dst) == 3);
    CHECK(!journal::write_entry(dst, rebuilt.data(), rebuilt.size()));
    fs::last_write_time(dst, now - hours(3));
    entry = journal::read_with_seed(dst, seed, id, runtime);
    CHECK(!entry.from_seed && entry.data == rebuilt && entry.bytes == rebuilt.size());
    CHECK(journal::entry_usable(dst, id));
    CHECK(!journal::entry_usable(dst, id, rebuilt.size() - 1) && journal::entry_usable(dst, id, rebuilt.size()));
    const auto ready_build = journal::read_with_seed(dst, roots.seed, id, builder);
    CHECK(!ready_build.from_seed && ready_build.data == rebuilt && ready_build.bytes == rebuilt.size());
    journal::touch_local_hit(dst, entry, now);
    const auto refreshed = fs::last_write_time(dst);
    CHECK(refreshed > entry.used);
    entry = journal::read_with_seed(dst, seed, id, runtime);
    journal::touch_local_hit(dst, entry, now + std::chrono::minutes(30));
    CHECK(fs::last_write_time(dst) == refreshed);
    used = journal::tidy(roots.writable, 0);
    CHECK(used.evicted == 1 && used.bytes == 0 && !fs::exists(dst));
    CHECK(journal::read_entry(src, id) == donor && fs::last_write_time(src) == donor_time);

    const auto unset = journal::resolve_roots(local, {});
    CHECK(unset.writable == local && unset.seed.empty());
    const auto disabled = [](const fs::path& sink, const fs::path& source) {
        const auto r = journal::resolve_roots(sink, source);
        return r.writable.empty() && r.seed.empty();
    };
    CHECK(disabled({}, seed) && disabled(local, "relative") && disabled(local, dir / "no-seed"));
    CHECK(disabled(local, src) && disabled(src, seed));
    CHECK(disabled(seed / ".", seed) && disabled(seed / "missing" / "journal", seed) && disabled(dir, seed));
#ifdef _WIN32
    if (fs::exists(dir / "SEED")) CHECK(disabled(dir / "SEED", seed));
#endif
    return 0;
}

int main() {
    const auto dir = fs::temp_directory_path() /
        ("leonida_pipeline_journal_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::fprintf(stderr, "cannot create %s\n", dir.string().c_str());
        return 1;
    }
    int failed = test_hash();
    if (!failed) failed = test_path();
    if (!failed) failed = test_files(dir);
    if (!failed) failed = test_policy();
    if (!failed) failed = test_accounting();
    if (!failed) failed = test_names();
    if (!failed) failed = test_tidy(dir);
    if (!failed) failed = test_entry_file(dir);
    if (!failed) failed = test_stamp(dir);
    if (!failed) failed = test_overlay(dir);
    fs::remove_all(dir, ec);
    if (failed) return failed;
    std::printf("pipeline_cache_journal_test: ok\n");
    return 0;
}
