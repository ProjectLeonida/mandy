// precompile_inflight.h: an entry a dead run left inside the driver is a suspect, built again; only a second death
// on the same module and driver holds it as crashed (a kill from outside lands on whatever was building, a driver crash
// recurs). A stop the run is told about (abandon) leaves nothing; a new key, a returning build or --all clears it.
// Standard library only, no GPU.
#include "../src/precompile_inflight.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace fs = std::filesystem;
using leonida::precompile_inflight::Record;

namespace {

std::set<std::string> lines(const fs::path& path) {
    std::set<std::string> out;
    std::ifstream in(path, std::ios::binary);
    for (std::string line; std::getline(in, line);)
        if (!line.empty()) out.insert(line);
    return out;
}

// A run that dies with `name` (as `key`) inside the driver: its record is left as it was.
void die_building(const fs::path& dir, const std::string& name, const std::string& key) {
    Record r(dir, false);
    r.begin(name, key);
}

int run(const fs::path& dir) {
    const fs::path inflight = dir / "inflight.txt", suspect = dir / "suspect.txt", crashed = dir / "crashed.txt";

    {   // A clean run: the file lists what is being built and goes when nothing is.
        Record r(dir, false);
        EXPECT(r.crashed().empty() && r.suspects().empty() && r.left_in_flight().empty());
        r.begin("lib_cs_a", "k1");
        r.begin("lib_cs_b", "k2");
        EXPECT((lines(inflight) == std::set<std::string>{"lib_cs_a k1", "lib_cs_b k2"}));
        r.end("lib_cs_a");
        r.end("lib_cs_b");
        EXPECT(!fs::exists(inflight) && !fs::exists(suspect) && !fs::exists(crashed));
    }
    // One death: a suspect, not skipped (a kill from outside looks the same).
    die_building(dir, "lib_cs_big", "k7");
    {
        Record r(dir, false);
        EXPECT(r.left_in_flight().size() == 1 && r.left_in_flight().at("lib_cs_big") == "k7");
        EXPECT(!r.is_crashed("lib_cs_big", "k7"));
        EXPECT((lines(suspect) == std::set<std::string>{"lib_cs_big k7"}));
        EXPECT(!fs::exists(crashed) && !fs::exists(inflight));
        // The retry dies on it again: that run is what makes it a crash.
        r.begin("lib_cs_big", "k7");
    }
    {   // Second death on the same module and driver: crashed, skipped from now on.
        Record r(dir, false);
        EXPECT(r.is_crashed("lib_cs_big", "k7"));
        EXPECT(r.suspects().empty() && !fs::exists(suspect));
        EXPECT((lines(crashed) == std::set<std::string>{"lib_cs_big k7"}));
        EXPECT(!r.is_crashed("lib_cs_big", "k9"));  // a new translation or driver: tried again
    }
    {   // A build of a new key that returns clears it.
        Record r(dir, false);
        r.begin("lib_cs_big", "k9");
        r.end("lib_cs_big");
        EXPECT(r.crashed().empty() && !fs::exists(crashed));
    }
    // Kills that land on different entries never make a crash.
    die_building(dir, "lib_cs_one", "k1");
    die_building(dir, "lib_cs_two", "k2");
    {
        Record r(dir, false);
        EXPECT(r.crashed().empty());
        EXPECT((lines(suspect) == std::set<std::string>{"lib_cs_one k1", "lib_cs_two k2"}));
        r.begin("lib_cs_one", "k1");
        r.end("lib_cs_one");  // built this time: no longer a suspect
        EXPECT((lines(suspect) == std::set<std::string>{"lib_cs_two k2"}));
    }
    // A suspect that dies again with a different key (it was translated again in between) is a suspect again.
    die_building(dir, "lib_cs_two", "k3");
    {
        Record r(dir, false);
        EXPECT(!r.is_crashed("lib_cs_two", "k3") && r.suspects().at("lib_cs_two") == "k3");
    }
    {   // abandon (Ctrl+C, the parent gone): nothing in flight is held against anything, and nothing more is written.
        Record r(dir, true);  // --all: start with no record
        EXPECT(r.crashed().empty() && r.suspects().empty() && !fs::exists(suspect));
        r.begin("lib_cs_long", "k5");
        r.abandon();
        EXPECT(!fs::exists(inflight));
        r.begin("lib_cs_after", "k6");  // a worker still finishing: ignored
        EXPECT(!fs::exists(inflight));
    }
    {
        Record r(dir, false);
        EXPECT(r.left_in_flight().empty() && r.suspects().empty() && r.crashed().empty());
    }
    {   // Many threads at once: the file always holds whole lines, and nothing is left when all are done.
        Record r(dir, false);
        std::vector<std::thread> pool;
        for (int t = 0; t < 8; ++t)
            pool.emplace_back([&r, t] {
                for (int i = 0; i < 50; ++i) {
                    const std::string name = "cs_" + std::to_string(t) + "_" + std::to_string(i);
                    r.begin(name, "k");
                    r.end(name);
                }
            });
        for (std::thread& t : pool) t.join();
        EXPECT(!fs::exists(inflight));
    }
    {   // A Windows line ending in a hand-edited list is not part of the key; a line with no key matches no build.
        { std::ofstream(crashed, std::ios::binary) << "lib_cs_crlf k3\r\nlib_cs_nokey\n"; }
        Record r(dir, false);
        EXPECT(r.is_crashed("lib_cs_crlf", "k3"));
        EXPECT(!r.is_crashed("lib_cs_nokey", "k4"));
    }
    return 0;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("leonida_precompile_inflight_" + std::to_string(
                                                          std::hash<std::thread::id>{}(std::this_thread::get_id())));
    std::error_code ec;
    fs::remove_all(dir, ec);
    const int result = run(dir);
    fs::remove_all(dir, ec);
    if (!result) std::printf("precompile_inflight_test: ok\n");
    return result;
}
