// precompile_inflight.h: an entry left in flight by a run that died is known to the next run and skipped there while
// it is the same module on the same driver; a new key (a new translation, a new driver) or --all tries it again.
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

int run(const fs::path& dir) {
    const fs::path inflight = dir / "inflight.txt", crashed = dir / "crashed.txt";

    {   // A clean run: nothing crashed, the file lists what is being built and goes when nothing is.
        Record r(dir, false);
        EXPECT(r.crashed().empty() && r.left_in_flight().empty());
        r.begin("lib_cs_a", "k1");
        r.begin("lib_cs_b", "k2");
        EXPECT((lines(inflight) == std::set<std::string>{"lib_cs_a k1", "lib_cs_b k2"}));
        r.end("lib_cs_a");
        EXPECT((lines(inflight) == std::set<std::string>{"lib_cs_b k2"}));
        r.end("lib_cs_b");
        EXPECT(!fs::exists(inflight));
        EXPECT(!fs::exists(crashed));
    }
    {   // A run that dies inside the driver: its record is left as it was (no end()).
        Record r(dir, false);
        r.begin("lib_cs_big", "k7");
        r.begin("lib_cs_other", "k8");
        r.end("lib_cs_other");
    }
    {   // The next run learns which entry and key it was, moves it to crashed.txt and starts with no in-flight list.
        Record r(dir, false);
        EXPECT(r.left_in_flight().size() == 1 && r.left_in_flight().at("lib_cs_big") == "k7");
        EXPECT(r.is_crashed("lib_cs_big", "k7") && !r.is_crashed("lib_cs_other", "k8"));
        EXPECT(!fs::exists(inflight));
        EXPECT((lines(crashed) == std::set<std::string>{"lib_cs_big k7"}));
    }
    {   // A later run still skips it while it is the same module on the same driver...
        Record r(dir, false);
        EXPECT(r.left_in_flight().empty());
        EXPECT(r.is_crashed("lib_cs_big", "k7"));
        // ...and tries it again once the key changes (a new translation of it, or a new driver).
        EXPECT(!r.is_crashed("lib_cs_big", "k9"));
    }
    {   // A new build of it that dies again records the new key.
        { Record r(dir, false); r.begin("lib_cs_big", "k9"); }
        Record r(dir, false);
        EXPECT(r.is_crashed("lib_cs_big", "k9") && !r.is_crashed("lib_cs_big", "k7"));
    }
    {   // A build of a new key that returns clears the entry from crashed.txt.
        Record r(dir, false);
        r.begin("lib_cs_big", "k10");
        r.end("lib_cs_big");
        EXPECT(r.crashed().empty() && !fs::exists(crashed));
        { Record dead(dir, false); dead.begin("lib_cs_big", "k11"); }  // and dies once more for the --all case
    }
    {   // --all: every entry is tried again, the list is cleared.
        Record r(dir, true);
        EXPECT(r.crashed().empty());
        EXPECT(!fs::exists(crashed));
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
        EXPECT(r.crashed().empty());
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
