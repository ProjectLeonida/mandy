// The loader's import cache (Loader::ImportedDll and Loader::ImportCache, runtime/src/pe_loader.cpp): binding an
// import table through it gives every import the binding, address and record that resolving each import from scratch
// gives, and touches the disk and the host loader once per imported DLL instead of once per import.
//
// Two parts, both without game files:
//   1. A model of Loader::resolve before and after the cache, over a fake world (shim table, files beside the game,
//      bundled modules with exports and forwarders, host DLLs with names and ordinals). The "before" body is main's
//      resolve; the "after" body is the patched one, through the real rt::Lazy, with one cache per bind_imports call
//      keyed by the DLL name as the descriptors spell it. Fixed cases (an empty descriptor,
//      one import, all-shim, a CRT DLL, a missing DLL, a bundled module with forwarders) and 2000 random import
//      tables must give the same results and the same visible side effects, with exists/load/LoadLibraryExA counted.
//   2. A microbenchmark of what the cache removes, with the real calls: std::filesystem::exists on <game>/<dll>,
//      LoadLibraryExA(LOAD_LIBRARY_SEARCH_SYSTEM32) and GetProcAddress over the exports of System32 DLLs, before
//      (all three per import) and after (the first two once per DLL). The addresses must match.
//
//   clang++ -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime/src runtime/test/pe_loader_import_cache_test.cpp -o t.exe
//   t.exe            (add --bench-only to skip part 1, --reps N for more benchmark repetitions)

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <new>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "lazy.h"

// ---- allocation counter (the benchmark reports heap allocations per bind) ----
static std::atomic<uint64_t> g_allocs{0};
void* operator new(size_t n) {
    ++g_allocs;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

// ---- 0. rt::Lazy ----

void test_lazy() {
    rt::Lazy<int> v;
    int calls = 0;
    CHECK(!v.has_value());
    CHECK(v.get([&] { return ++calls, 7; }) == 7);
    CHECK(v.get([&] { return ++calls, 9; }) == 7);  // kept, not recomputed
    CHECK(calls == 1 && v.has_value());

    rt::Lazy<void*> p;  // a null result is a result for get(), but get_if_truthy() leaves it unset so probes can retry
    CHECK(p.get([&] { return ++calls, static_cast<void*>(nullptr); }) == nullptr);
    CHECK(p.get([&] { return ++calls, static_cast<void*>(&calls); }) == nullptr);
    CHECK(calls == 2);

    rt::Lazy<void*> retryable;
    CHECK(retryable.get_if_truthy([&] { return ++calls, static_cast<void*>(nullptr); }) == nullptr);
    CHECK(!retryable.has_value());
    CHECK(retryable.get_if_truthy([&] { return ++calls, static_cast<void*>(&calls); }) == static_cast<void*>(&calls));
    CHECK(retryable.has_value());
    CHECK(retryable.get_if_truthy([&] { return ++calls, static_cast<void*>(nullptr); }) == static_cast<void*>(&calls));
    CHECK(calls == 4);

    rt::Lazy<int> t;  // a throw leaves it unset, so the next get runs make again
    bool threw = false;
    try {
        t.get([]() -> int { throw 1; });
    } catch (int) {
        threw = true;
    }
    CHECK(threw && !t.has_value());
    CHECK(t.get([] { return 3; }) == 3);
}

// ---- 1. model of Loader::resolve ----

enum class Binding { Shim, Host, Module, Unimplemented };

struct Record {
    std::string dll, sym;
    Binding binding;
    bool operator==(const Record&) const = default;
};

struct FakeModule {
    std::map<std::string, uintptr_t> exports;
    std::map<std::string, std::string> forwarders;
};

struct FakeHostDll {
    std::map<std::string, uintptr_t> names;
    std::map<uint16_t, uintptr_t> ordinals;
};

const std::set<std::string> kHostCrt = {"vcruntime140.dll", "msvcp140.dll", "concrt140.dll"};

// The world resolve sees, with every call it makes counted. Names on disk and host DLL names compare without case,
// as Windows does.
struct World {
    std::set<std::string> shims;                  // "key!sym"
    std::set<std::string> files;                  // lowered names beside the game
    std::map<std::string, FakeModule> bundled;    // lowered name -> mapped module
    std::map<std::string, FakeHostDll> host;      // lowered name -> host DLL
    std::map<std::string, uintptr_t> forwarded;   // forwarder spec -> address (absent: not found)

    // What a run did, the visible part first.
    std::vector<Record> records;
    std::vector<uintptr_t> redirected_at;    // redirect_pc_to_file_header_at arguments, in order
    std::set<std::string> redirected;        // modules redirect_pc_to_file_header patched (it dedupes itself)
    std::vector<std::string> loaded_order;   // first load() of each bundled module, in order
    uint64_t exists_calls = 0, load_calls = 0, loadlib_calls = 0, getproc_calls = 0, redirect_calls = 0;

    void reset_run() {
        records.clear();
        redirected_at.clear();
        redirected.clear();
        loaded_order.clear();
        exists_calls = load_calls = loadlib_calls = getproc_calls = redirect_calls = 0;
    }

    void* shim_find(const std::string& key, const std::string& sym) const {
        auto it = shims.find(key + "!" + sym);
        return it == shims.end() ? nullptr : reinterpret_cast<void*>(std::hash<std::string>{}(*it) | 1);
    }
    bool exists(const std::string& dll) {
        ++exists_calls;
        return files.count(lower(dll)) != 0;
    }
    FakeModule& load(const std::string& dll) {
        ++load_calls;
        const std::string k = lower(dll);
        if (std::find(loaded_order.begin(), loaded_order.end(), k) == loaded_order.end()) loaded_order.push_back(k);
        return bundled[k];  // the same module every time, as by_name_ gives
    }
    void* resolve_forwarder(const std::string& spec, Binding& binding) {
        auto it = forwarded.find(spec);
        if (it == forwarded.end()) return nullptr;
        binding = (it->second & 2) ? Binding::Host : Binding::Module;
        return reinterpret_cast<void*>(it->second);
    }
    FakeHostDll* load_library(const std::string& dll) {
        ++loadlib_calls;
        auto it = host.find(lower(dll));
        return it == host.end() ? nullptr : &it->second;
    }
    void* get_proc(FakeHostDll* h, const std::string& name, uint16_t ordinal, bool by_ordinal) {
        ++getproc_calls;
        if (by_ordinal) {
            auto it = h->ordinals.find(ordinal);
            return it == h->ordinals.end() ? nullptr : reinterpret_cast<void*>(it->second);
        }
        auto it = h->names.find(name);
        return it == h->names.end() ? nullptr : reinterpret_cast<void*>(it->second);
    }
    void redirect(FakeHostDll* h) {
        ++redirect_calls;
        for (auto& [name, dll] : host)
            if (&dll == h) redirected.insert(name);
    }
    void* unimplemented(const std::string& dll, const std::string& sym) {
        return reinterpret_cast<void*>((std::hash<std::string>{}(lower(dll) + "!" + sym) << 4) | 8);
    }
};

// main's Loader::resolve, call for call.
void* resolve_before(World& w, const std::string& dll, const std::string& name, uint16_t ordinal, bool by_ordinal) {
    std::string key = lower(dll);
    std::string sym = by_ordinal ? "#" + std::to_string(ordinal) : name;
    if (void* p = w.shim_find(key, sym)) {
        w.records.push_back({dll, sym, Binding::Shim});
        return p;
    }
    if (!kHostCrt.count(key) && w.exists(dll)) {
        FakeModule& dep = w.load(dll);
        if (auto it = dep.exports.find(sym); it != dep.exports.end()) {
            w.records.push_back({dll, sym, Binding::Module});
            return reinterpret_cast<void*>(it->second);
        }
        if (auto it = dep.forwarders.find(sym); it != dep.forwarders.end()) {
            Binding binding = Binding::Unimplemented;
            if (void* p = w.resolve_forwarder(it->second, binding)) {
                w.records.push_back({dll, sym, binding});
                return p;
            }
        }
    }
    if (FakeHostDll* h = w.load_library(dll)) {
        if (kHostCrt.count(key)) w.redirect(h);
        void* p = w.get_proc(h, name, ordinal, by_ordinal);
        if (p) {
            if (kHostCrt.count(key)) w.redirected_at.push_back(reinterpret_cast<uintptr_t>(p));
            w.records.push_back({dll, sym, Binding::Host});
            return p;
        }
    }
    w.records.push_back({dll, sym, Binding::Unimplemented});
    return w.unimplemented(dll, sym);
}

// The patched Loader::resolve with Loader::ImportedDll and Loader::ImportCache, call for call.
struct ImportedDll {
    explicit ImportedDll(const std::string& dll) : key(lower(dll)), host_crt(kHostCrt.count(key) != 0) {}
    const std::string key;
    const bool host_crt;
    rt::Lazy<FakeModule*> bundled;
    rt::Lazy<FakeHostDll*> host;
};
using ImportCache = std::map<std::string, ImportedDll>;  // one per bind_imports call

void* resolve_after(World& w, ImportCache& cache, const std::string& dll, const std::string& name, uint16_t ordinal,
                    bool by_ordinal) {
    ImportedDll& imported = cache.try_emplace(dll, dll).first->second;
    std::string sym = by_ordinal ? "#" + std::to_string(ordinal) : name;
    if (void* p = w.shim_find(imported.key, sym)) {
        w.records.push_back({dll, sym, Binding::Shim});
        return p;
    }
    FakeModule* bundled = imported.bundled.get_if_truthy([&]() -> FakeModule* {
        if (imported.host.has_value()) return nullptr;
        return !imported.host_crt && w.exists(dll) ? &w.load(dll) : nullptr;
    });
    if (bundled) {
        FakeModule& dep = *bundled;
        if (auto it = dep.exports.find(sym); it != dep.exports.end()) {
            w.records.push_back({dll, sym, Binding::Module});
            return reinterpret_cast<void*>(it->second);
        }
        if (auto it = dep.forwarders.find(sym); it != dep.forwarders.end()) {
            Binding binding = Binding::Unimplemented;
            if (void* p = w.resolve_forwarder(it->second, binding)) {
                w.records.push_back({dll, sym, binding});
                return p;
            }
        }
    }
    if (FakeHostDll* h = imported.host.get_if_truthy([&] {
            FakeHostDll* loaded = w.load_library(dll);
            if (loaded && imported.host_crt) w.redirect(loaded);
            return loaded;
        })) {
        void* p = w.get_proc(h, name, ordinal, by_ordinal);
        if (p) {
            if (imported.host_crt) w.redirected_at.push_back(reinterpret_cast<uintptr_t>(p));
            w.records.push_back({dll, sym, Binding::Host});
            return p;
        }
    }
    w.records.push_back({dll, sym, Binding::Unimplemented});
    return w.unimplemented(dll, sym);
}

struct Import {
    std::string name;
    uint16_t ordinal = 0;
    bool by_ordinal = false;
};
struct Descriptor {
    std::string dll;
    std::vector<Import> imports;
};

struct Run {
    std::vector<uintptr_t> iat;
    std::vector<Record> records;
    std::vector<uintptr_t> redirected_at;
    std::set<std::string> redirected;
    std::vector<std::string> loaded_order;
    uint64_t exists_calls, load_calls, loadlib_calls, getproc_calls, redirect_calls;
};

Run bind(World& w, const std::vector<Descriptor>& table, bool cached) {
    w.reset_run();
    Run r{};
    ImportCache cache;  // one for the whole table, as bind_imports installs it
    for (const Descriptor& d : table) {
        for (const Import& i : d.imports)
            r.iat.push_back(reinterpret_cast<uintptr_t>(
                cached ? resolve_after(w, cache, d.dll, i.name, i.ordinal, i.by_ordinal)
                       : resolve_before(w, d.dll, i.name, i.ordinal, i.by_ordinal)));
    }
    r.records = w.records;
    r.redirected_at = w.redirected_at;
    r.redirected = w.redirected;
    r.loaded_order = w.loaded_order;
    r.exists_calls = w.exists_calls;
    r.load_calls = w.load_calls;
    r.loadlib_calls = w.loadlib_calls;
    r.getproc_calls = w.getproc_calls;
    r.redirect_calls = w.redirect_calls;
    return r;
}

// Same results and visible side effects; the cached run never makes more calls than the uncached run, and for any
// resolved DLL it calls load() and redirect at most once per descriptor; GetProcAddress runs exactly as often.
bool same_and_bounded(const Run& a, const Run& b, size_t descriptors) {
    bool ok = a.iat == b.iat && a.records == b.records && a.redirected_at == b.redirected_at &&
              a.redirected == b.redirected && a.loaded_order == b.loaded_order && a.getproc_calls == b.getproc_calls;
    ok = ok && b.exists_calls <= a.exists_calls;
    ok = ok && b.load_calls <= std::min<uint64_t>(a.load_calls, descriptors);
    ok = ok && b.loadlib_calls <= a.loadlib_calls;
    ok = ok && b.redirect_calls <= std::min<uint64_t>(a.redirect_calls, descriptors);
    // Each cached call happens iff the uncached run made that call at least once for the descriptor's DLL.
    ok = ok && (a.exists_calls == 0) == (b.exists_calls == 0) && (a.loadlib_calls == 0) == (b.loadlib_calls == 0);
    return ok;
}

World fixed_world() {
    World w;
    w.shims = {"xmem.dll!XMemAlloc", "xmem.dll!XMemFree", "kernel32.dll!LoadLibraryW", "d3d12_xs.dll!#3"};
    w.files = {"xcurl.dll", "bink2scarlett.dll", "vcruntime140.dll"};  // the CRT ships, but the host copy is used
    w.bundled["xcurl.dll"] = {{{"curl_easy_init", 0x1000}, {"curl_easy_cleanup", 0x1010}},
                              {{"curl_fwd", "kernel32.GetTickCount"}, {"curl_lost", "nowhere.Nothing"}}};
    w.bundled["bink2scarlett.dll"] = {{{"BinkOpen", 0x2000}}, {}};
    w.host["kernel32.dll"] = {{{"GetTickCount", 0x7000}, {"Sleep", 0x7010}}, {{1, 0x7100}}};
    w.host["vcruntime140.dll"] = {{{"_CxxThrowException", 0x8000}, {"memcpy", 0x8010}}, {}};
    w.host["xcurl.dll"] = {{{"curl_lost", 0x9000}}, {}};  // a host namesake the bundled module's miss falls to
    w.forwarded["kernel32.GetTickCount"] = 0x7002;
    return w;
}

void test_fixed() {
    World w = fixed_world();
    const std::vector<std::vector<Descriptor>> cases = {
        {},                                                                     // no descriptor
        {{"KERNEL32.dll", {}}},                                                 // an empty descriptor
        {{"kernel32.dll", {{"Sleep"}}}},                                        // one import (small table)
        {{"xmem.dll", {{"XMemAlloc"}, {"XMemFree"}}}},                          // all shim: nothing probed
        {{"Kernel32.DLL", {{"LoadLibraryW"}, {"Sleep"}, {"GetTickCount"}, {"", 1, true}, {"", 2, true}, {"Nope"}}}},
        {{"VCRUNTIME140.dll", {{"_CxxThrowException"}, {"memcpy"}, {"missing"}}}},  // CRT: host copy, redirected
        {{"XCurl.dll", {{"curl_easy_init"}, {"curl_fwd"}, {"curl_lost"}, {"curl_easy_cleanup"}, {"none"}}}},
        {{"missing.dll", {{"a"}, {"b"}, {"", 7, true}}}},                       // not anywhere: all unimplemented
        {{"d3d12_xs.dll", {{"", 3, true}, {"", 4, true}}}},
        {{"xcurl.dll", {{"curl_easy_init"}}}, {"bink2scarlett.dll", {{"BinkOpen"}}}, {"XCURL.DLL", {{"curl_fwd"}}}},
        {{"kernel32.dll", {{"Sleep"}}}, {"xmem.dll", {{"XMemAlloc"}}}, {"kernel32.dll", {{"GetTickCount"}, {"x"}}}},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        const Run a = bind(w, cases[c], false), b = bind(w, cases[c], true);
        if (!same_and_bounded(a, b, cases[c].size())) {
            std::fprintf(stderr, "fixed case %zu differs\n", c);
            ++g_failures;
        }
    }
    // The counts on one table: 6 imports of kernel32 with one shim hit, 5 reaching the disk probe and the host.
    const Run a = bind(w, cases[4], false), b = bind(w, cases[4], true);
    CHECK(a.exists_calls == 5 && a.loadlib_calls == 5 && b.exists_calls == 1 && b.loadlib_calls == 1);
    CHECK(a.getproc_calls == 5 && b.getproc_calls == 5);
    // The bundled module is loaded once, its miss falls to the host namesake exactly as before.
    const Run c = bind(w, cases[6], false), d = bind(w, cases[6], true);
    CHECK(c.load_calls == 5 && d.load_calls == 1 && d.loadlib_calls == 1 && c.loadlib_calls == 2);
    CHECK(d.records[2].binding == Binding::Host && d.records[1].binding == Binding::Host);
    // A CRT DLL is never probed beside the game, and its host module is patched once.
    const Run e = bind(w, cases[5], false), f = bind(w, cases[5], true);
    CHECK(e.exists_calls == 0 && f.exists_calls == 0 && e.redirect_calls == 3 && f.redirect_calls == 1);
    CHECK(f.redirected == std::set<std::string>{"vcruntime140.dll"} && f.redirected_at.size() == 2);
    // Two descriptors that spell the same DLL share one entry: one probe and one load for the module's three imports.
    const Run g = bind(w, cases[10], false), h = bind(w, cases[10], true);
    CHECK(g.exists_calls == 3 && g.loadlib_calls == 3 && h.exists_calls == 1 && h.loadlib_calls == 1);

    // Failed probes are not frozen: if A.dll initially fails to load, then B.dll makes A.dll available (e.g. via
    // DllMain or a later load), a subsequent import of A.dll in the same bind_imports still resolves as Host.
    {
        auto run_dynamic = [&](bool cached) {
            World dw;
            dw.host["b.dll"] = {{{"InitB", 0xB000}}, {}};
            ImportCache cache;
            auto call = [&](const std::string& dll, const std::string& name) {
                return cached ? resolve_after(dw, cache, dll, name, 0, false)
                              : resolve_before(dw, dll, name, 0, false);
            };
            call("a.dll", "FnA");  // fails -> Unimplemented
            call("b.dll", "InitB");
            dw.host["a.dll"] = {{{"FnA", 0xA000}}, {}};  // B's DllMain made A.dll available
            call("a.dll", "FnA");                        // must retry and bind as Host
            return dw.records;
        };
        const auto rec_before = run_dynamic(false);
        const auto rec_after = run_dynamic(true);
        CHECK(rec_before == rec_after);
        CHECK(rec_after.size() == 3 && rec_after[0].binding == Binding::Unimplemented &&
              rec_after[1].binding == Binding::Host && rec_after[2].binding == Binding::Host);
    }
}

void test_random() {
    std::mt19937_64 rng(0x1e0a1da);
    auto pick = [&](size_t n) { return static_cast<size_t>(rng() % n); };
    const char* dll_names[] = {"kernel32.dll", "user32.dll", "xmem.dll", "xcurl.dll", "bink2scarlett.dll",
                               "vcruntime140.dll", "msvcp140.dll", "missing.dll", "d3d12_xs.dll", "concrt140.dll"};
    uint64_t before_probe = 0, after_probe = 0, before_load = 0, after_load = 0, total_imports = 0;
    for (int iter = 0; iter < 2000; ++iter) {
        World w;
        std::vector<std::string> syms;
        for (int i = 0; i < 24; ++i) syms.push_back("fn" + std::to_string(i));
        for (const char* n : dll_names) {
            const std::string dn = n;
            if (pick(3) == 0) w.files.insert(dn);
            if (pick(3) != 0) {
                FakeHostDll& h = w.host[dn];
                for (const auto& s : syms)
                    if (pick(2)) h.names[s] = 0x10000 + rng() % 0x10000 * 16;
                for (uint16_t o = 1; o < 12; ++o)
                    if (pick(2)) h.ordinals[o] = 0x40000 + rng() % 0x10000 * 16;
            }
            if (w.files.count(dn)) {
                FakeModule& m = w.bundled[dn];
                for (const auto& s : syms) {
                    const size_t r = pick(4);
                    if (r == 0) m.exports[s] = 0x80000 + rng() % 0x10000 * 16;
                    else if (r == 1) m.forwarders[s] = "fwd" + std::to_string(pick(8)) + "." + s;
                }
                for (uint16_t o = 1; o < 12; ++o)
                    if (pick(3) == 0) m.exports["#" + std::to_string(o)] = 0xc0000 + o * 16;
            }
            for (const auto& s : syms)
                if (pick(5) == 0) w.shims.insert(dn + "!" + s);
        }
        for (int f = 0; f < 8; ++f)
            for (const auto& s : syms)
                if (pick(2)) w.forwarded["fwd" + std::to_string(f) + "." + s] = 0x100000 + rng() % 0x10000 * 16;

        std::vector<Descriptor> table(pick(12));
        for (Descriptor& d : table) {
            d.dll = dll_names[pick(std::size(dll_names))];
            if (pick(2)) d.dll[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(d.dll[0])));
            if (pick(4) == 0) std::transform(d.dll.begin(), d.dll.end(), d.dll.begin(), ::toupper);
            d.imports.resize(pick(4) == 0 ? pick(3) : pick(60));  // small tables (0-2 imports) a quarter of the time
            for (Import& i : d.imports) {
                i.by_ordinal = pick(5) == 0;
                if (i.by_ordinal) i.ordinal = static_cast<uint16_t>(1 + pick(12));
                else i.name = pick(10) ? syms[pick(syms.size())] : "absent" + std::to_string(pick(3));
            }
            total_imports += d.imports.size();
        }
        const Run a = bind(w, table, false), b = bind(w, table, true);
        if (!same_and_bounded(a, b, table.size())) {
            std::fprintf(stderr, "random table %d differs\n", iter);
            ++g_failures;
        }
        before_probe += a.exists_calls, after_probe += b.exists_calls;
        before_load += a.loadlib_calls, after_load += b.loadlib_calls;
    }
    std::printf("model: 2000 random tables, %llu imports: exists %llu -> %llu, LoadLibraryExA %llu -> %llu\n",
                (unsigned long long)total_imports, (unsigned long long)before_probe, (unsigned long long)after_probe,
                (unsigned long long)before_load, (unsigned long long)after_load);
}

// ---- 2. benchmark with the real calls ----

std::vector<std::string> export_names(HMODULE h, size_t limit) {
    std::vector<std::string> out;
    auto* base = reinterpret_cast<const uint8_t*>(h);
    auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    if (!dir.Size) return out;
    auto* ex = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + dir.VirtualAddress);
    auto* names = reinterpret_cast<const uint32_t*>(base + ex->AddressOfNames);
    for (uint32_t i = 0; i < ex->NumberOfNames && out.size() < limit; ++i)
        out.emplace_back(reinterpret_cast<const char*>(base + names[i]));
    return out;
}

struct BenchDescriptor {
    std::string dll;
    std::vector<std::string> names;
};

struct BenchResult {
    double us;
    uint64_t allocs;
    std::vector<uintptr_t> iat;
};

BenchResult bind_real(const std::filesystem::path& game_dir, const std::vector<BenchDescriptor>& table, bool cached) {
    BenchResult r{};
    const uint64_t a0 = g_allocs.load();
    const auto t0 = std::chrono::steady_clock::now();
    for (const BenchDescriptor& d : table) {
        rt::Lazy<bool> bundled;
        rt::Lazy<HMODULE> host;
        [[maybe_unused]] const std::string key_once = lower(d.dll);  // ImportedDll lowers the name once
        for (const std::string& name : d.names) {
            bool is_bundled;
            HMODULE h;
            if (cached) {
                is_bundled = bundled.get([&] { return std::filesystem::exists(game_dir / d.dll); });
                h = host.get([&] { return LoadLibraryExA(d.dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32); });
            } else {
                const std::string key = lower(d.dll);  // resolve() lowered the DLL name per import
                is_bundled = !key.empty() && std::filesystem::exists(game_dir / d.dll);
                h = LoadLibraryExA(d.dll.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            }
            FARPROC p = (!is_bundled && h) ? GetProcAddress(h, name.c_str()) : nullptr;
            r.iat.push_back(reinterpret_cast<uintptr_t>(p));
        }
    }
    r.us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    r.allocs = g_allocs.load() - a0;
    return r;
}

void bench(int reps) {
    // An import table the size of the retail executable's host part plus its four DLLs' (docs/runtime_imports.md:
    // 432 + 52 + 16 + 170 + 55 host imports): System32 DLLs, up to N exports each.
    const struct {
        const char* dll;
        size_t count;
    } spec[] = {{"kernel32.dll", 300}, {"user32.dll", 60}, {"advapi32.dll", 40}, {"ntdll.dll", 60},
                {"ole32.dll", 30},     {"ws2_32.dll", 40}, {"bcrypt.dll", 20},   {"shell32.dll", 20},
                {"api-ms-win-core-synch-l1-2-0.dll", 40},  {"api-ms-win-core-file-l1-1-0.dll", 60},
                {"api-ms-win-core-libraryloader-l1-2-0.dll", 20}, {"crypt32.dll", 35}};
    std::vector<BenchDescriptor> table;
    size_t imports = 0;
    for (const auto& s : spec) {
        HMODULE h = LoadLibraryExA(s.dll, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!h) continue;
        BenchDescriptor d{s.dll, export_names(h, s.count)};
        if (d.names.empty()) {  // an API set has no names of its own here; bind kernel32's through it
            HMODULE k = GetModuleHandleA("kernel32.dll");
            d.names = export_names(k, s.count);
        }
        imports += d.names.size();
        table.push_back(std::move(d));
    }
    // A game folder that holds none of them, as on a real install: every probe misses.
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    const std::filesystem::path game_dir = std::filesystem::path(tmp) / L"leonida-import-cache-bench";
    std::filesystem::create_directories(game_dir);

    std::vector<double> before, after;
    uint64_t allocs_before = 0, allocs_after = 0;
    bind_real(game_dir, table, false);  // warm the page cache and the loader
    bind_real(game_dir, table, true);
    for (int i = 0; i < reps; ++i) {
        // Alternate the order, so neither variant always runs on a warmer cache.
        BenchResult a, b;
        if (i & 1) {
            b = bind_real(game_dir, table, true);
            a = bind_real(game_dir, table, false);
        } else {
            a = bind_real(game_dir, table, false);
            b = bind_real(game_dir, table, true);
        }
        if (a.iat != b.iat) {
            std::fprintf(stderr, "benchmark: bound addresses differ\n");
            ++g_failures;
        }
        before.push_back(a.us), after.push_back(b.us);
        allocs_before = a.allocs, allocs_after = b.allocs;
    }
    std::sort(before.begin(), before.end());
    std::sort(after.begin(), after.end());
    const double mb = before[before.size() / 2], ma = after[after.size() / 2];
    std::printf("bench: %zu descriptors, %zu imports, %d reps (median)\n", table.size(), imports, reps);
    std::printf("  per import (before): %9.1f us  exists=%zu LoadLibraryExA=%zu allocs=%llu\n", mb, imports, imports,
                (unsigned long long)allocs_before);
    std::printf("  per descriptor (after): %6.1f us  exists=%zu LoadLibraryExA=%zu allocs=%llu\n", ma, table.size(),
                table.size(), (unsigned long long)allocs_after);
    std::printf("  speedup %.2fx (%.1f%% less time), %.2f us saved per import\n", mb / ma, 100.0 * (1 - ma / mb),
                (mb - ma) / static_cast<double>(imports));
    std::error_code ec;
    std::filesystem::remove(game_dir, ec);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench_only = false;
    int reps = 21;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bench-only")) bench_only = true;
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    }
    if (!bench_only) {
        test_lazy();
        test_fixed();
        test_random();
    }
    bench(reps);
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("pe_loader_import_cache_test: ok\n");
    return 0;
}
