// UnimplementedImport (runtime/shims/unimplemented_import.h): the per-import state rt_unimplemented reads on every call
// of an import that has no binding. CPU only, no Windows headers.
//
// The bookkeeping rt_unimplemented did before is kept here as the reference: one process-wide mutex over a std::map
// keyed by the thunk's `what` pointer, the return class worked out from the "dll!name" string on the first call, and the
// set of called imports. The test drives both with the same call sequences and checks that every call gets the same
// return class and the same "first call" answer and that the called sets match: fixed sequences over small tables (1,
// 2 and 4 imports), random sequences over tables of 1 to 600 imports, and many threads calling the same imports at
// once (each import must report exactly one first call). It then counts heap allocations per call and times a call of
// each. `--bench` runs the timings longer.

#include "../shims/unimplemented_import.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

// ---- Heap allocation counter ------------------------------------------------------------------------------------

namespace {
std::atomic<uint64_t> g_allocations{0};
}

void* operator new(std::size_t n) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

using rt::shims::UnimplementedImport;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
}

// A stand-in for registry.cpp's classify (the same for both paths, so only the bookkeeping differs): 1 (HRESULT) for a
// D3D12 factory-like name, 0 otherwise.
std::string lower_ascii(std::string s) {
    for (auto& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + 32);
    return s;
}
uint8_t classify(const std::string& dll, const std::string& name) {
    return (dll == "d3d12_xs.dll" && name.rfind("D3D12Create", 0) == 0) || name.rfind("XGame", 0) == 0 ? 1 : 0;
}
uint8_t classify_key(const std::string& full) {
    const auto bang = full.find('!');
    return classify(lower_ascii(bang == std::string::npos ? "" : full.substr(0, bang)),
                    bang == std::string::npos ? full : full.substr(bang + 1));
}

struct Result {
    uint8_t cls;
    bool first;
    bool operator==(const Result& o) const { return cls == o.cls && first == o.first; }
};

// The old rt_unimplemented bookkeeping, as it was in registry.cpp.
struct OldState {
    std::mutex mutex;
    std::set<std::string> called;
    std::map<const char*, uint8_t> seen;

    Result call(const char* what) {
        uint8_t cls;
        bool first = false;
        {
            std::lock_guard lock(mutex);
            auto it = seen.find(what);
            if (it == seen.end()) {
                const std::string full = what;
                cls = classify_key(full);
                seen.emplace(what, cls);
                called.insert(full);
                first = true;
            } else {
                cls = it->second;
            }
        }
        return {cls, first};
    }
};

// The new one: the record the thunk passes, and the mutex taken only by a first call.
struct NewState {
    std::mutex mutex;
    std::set<std::string> called;

    Result call(UnimplementedImport* import) {
        const uint8_t cls = import->return_class;
        const bool first = import->first_call();
        if (first) {
            std::lock_guard lock(mutex);
            called.insert(import->what);
        }
        return {cls, first};
    }
};

// A table of imports as make_unimplemented_thunk builds it: one process-lifetime key string and one record each.
struct Imports {
    std::vector<std::string> keys;
    std::vector<const char*> what;
    std::vector<UnimplementedImport*> records;

    explicit Imports(size_t n, uint32_t seed = 1) {
        static const char* const dlls[] = {"d3d12_xs.dll", "xgameruntime.dll", "kernelx.dll", "leonida_probe.dll"};
        static const char* const names[] = {"D3D12CreateThing", "XGameUiShow", "XSystemGetX", "Probe", "Other"};
        std::mt19937 rng(seed);
        keys.reserve(n);
        for (size_t i = 0; i < n; ++i)
            keys.push_back(std::string(dlls[rng() % 4]) + "!" + names[rng() % 5] + std::to_string(i));
        for (const auto& k : keys) {
            what.push_back(k.c_str());
            records.push_back(new UnimplementedImport(k.c_str(), classify_key(k)));
        }
    }
    ~Imports() {
        for (auto* r : records) delete r;
    }
    Imports(const Imports&) = delete;
    Imports& operator=(const Imports&) = delete;
};

// Runs the same call sequence (indices into the table) through both and compares every answer and the called sets.
void compare(size_t n, const std::vector<size_t>& sequence, const std::string& tag, uint32_t seed = 1) {
    Imports imports(n, seed);
    OldState old_state;
    NewState new_state;
    size_t mismatches = 0, firsts = 0;
    for (size_t i : sequence) {
        const Result a = old_state.call(imports.what[i]);
        const Result b = new_state.call(imports.records[i]);
        if (!(a == b)) ++mismatches;
        firsts += b.first;
    }
    check(mismatches == 0, tag + ": " + std::to_string(mismatches) + " calls answered differently");
    check(old_state.called == new_state.called, tag + ": the called sets differ");
    std::set<size_t> distinct(sequence.begin(), sequence.end());
    check(firsts == distinct.size(), tag + ": " + std::to_string(firsts) + " first calls for " +
                                         std::to_string(distinct.size()) + " distinct imports");
}

void fixed_cases() {
    compare(1, {0}, "one import, one call");
    compare(1, {0, 0, 0, 0}, "one import, repeated");
    compare(2, {1, 0, 1, 0, 0, 1}, "two imports, interleaved");
    compare(2, {1, 1, 1}, "two imports, only the second called");
    compare(4, {3, 2, 1, 0, 0, 1, 2, 3}, "four imports, descending then ascending");
    compare(4, {0, 0, 2, 2, 0, 2}, "four imports, two never called");
    compare(4, {}, "four imports, no call");

    // Return classes come out as the classifier gives them, first call and repeat alike.
    UnimplementedImport hr("d3d12_xs.dll!D3D12CreateProbe", classify_key("d3d12_xs.dll!D3D12CreateProbe"));
    UnimplementedImport other("leonida_probe.dll!Probe", classify_key("leonida_probe.dll!Probe"));
    NewState s;
    check(s.call(&hr).cls == 1 && s.call(&hr).cls == 1, "a D3D12 factory name is HRESULT on every call");
    check(s.call(&other).cls == 0, "an unknown DLL's import is other");
    check(!s.call(&other).first, "a second call is not first");
}

void random_cases() {
    std::mt19937 rng(0x5eed);
    for (int round = 0; round < 300; ++round) {
        const size_t n = 1 + rng() % 600;
        const size_t calls = rng() % 5000;
        std::vector<size_t> sequence(calls);
        // Skewed: a few imports take most calls, as a game's per-frame unimplemented calls do.
        const size_t hot = 1 + rng() % std::min<size_t>(n, 8);
        for (auto& i : sequence) i = (rng() % 4) ? rng() % hot : rng() % n;
        compare(n, sequence, "random round " + std::to_string(round) + " (" + std::to_string(n) + " imports, " +
                                 std::to_string(calls) + " calls)",
                static_cast<uint32_t>(round + 1));
    }
}

void threaded_cases() {
    for (size_t n : {size_t{1}, size_t{4}, size_t{64}}) {
        Imports imports(n);
        NewState s;
        std::vector<std::atomic<int>> firsts(n);
        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t)
            threads.emplace_back([&, t] {
                std::mt19937 rng(static_cast<uint32_t>(t + 1));
                while (!go.load()) std::this_thread::yield();
                for (int i = 0; i < 20000; ++i) {
                    const size_t k = rng() % n;
                    if (s.call(imports.records[k]).first) firsts[k].fetch_add(1);
                }
            });
        go.store(true);
        for (auto& t : threads) t.join();
        bool once = true;
        for (auto& f : firsts) once &= f.load() == 1;
        check(once, std::to_string(n) + " imports over 8 threads: each reports exactly one first call");
        check(s.called.size() == n, std::to_string(n) + " imports over 8 threads: all are in the called set");
    }
}

uint64_t allocations_of(const auto& f) {
    const uint64_t before = g_allocations.load();
    f();
    return g_allocations.load() - before;
}

void allocation_counts() {
    Imports imports(64);
    OldState old_state;
    NewState new_state;
    const uint64_t old_first = allocations_of([&] { old_state.call(imports.what[7]); });
    const uint64_t new_first = allocations_of([&] { new_state.call(imports.records[7]); });
    const uint64_t old_repeat = allocations_of([&] {
        for (int i = 0; i < 1000; ++i) old_state.call(imports.what[7]);
    });
    const uint64_t new_repeat = allocations_of([&] {
        for (int i = 0; i < 1000; ++i) new_state.call(imports.records[7]);
    });
    std::printf("heap allocations: first call old %llu, new %llu; 1000 repeat calls old %llu, new %llu\n",
                static_cast<unsigned long long>(old_first), static_cast<unsigned long long>(new_first),
                static_cast<unsigned long long>(old_repeat), static_cast<unsigned long long>(new_repeat));
    check(new_repeat == 0 && old_repeat == 0, "a repeat call allocates nothing");
    check(new_first <= old_first, "a first call allocates no more than before");
}

template <class F>
double ns_per_call(size_t calls, F&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(calls);
}

volatile uint32_t g_sink = 0;

void bench(bool longer) {
    const size_t calls = longer ? 20000000 : 2000000;
    std::printf("\nsingle thread, ns per call (every import called once first, then a random hot import per call)\n");
    std::printf("%8s %10s %10s %9s\n", "imports", "old", "new", "speedup");
    for (size_t n : {size_t{1}, size_t{4}, size_t{32}, size_t{256}, size_t{600}}) {
        Imports imports(n);
        OldState old_state;
        NewState new_state;
        for (size_t i = 0; i < n; ++i) {
            old_state.call(imports.what[i]);
            new_state.call(imports.records[i]);
        }
        std::vector<uint32_t> order(4096);
        std::mt19937 rng(7);
        for (auto& o : order) o = static_cast<uint32_t>(rng() % n);
        const double old_ns = ns_per_call(calls, [&] {
            uint32_t acc = 0;
            for (size_t i = 0; i < calls; ++i) acc += old_state.call(imports.what[order[i & 4095]]).cls;
            g_sink = g_sink + acc;
        });
        const double new_ns = ns_per_call(calls, [&] {
            uint32_t acc = 0;
            for (size_t i = 0; i < calls; ++i) acc += new_state.call(imports.records[order[i & 4095]]).cls;
            g_sink = g_sink + acc;
        });
        std::printf("%8zu %10.2f %10.2f %8.1fx\n", n, old_ns, new_ns, old_ns / new_ns);
    }

    const unsigned hw = std::max(2u, std::min(8u, std::thread::hardware_concurrency()));
    std::printf("\n%u threads, 64 imports, wall ns per call (all threads together)\n", hw);
    Imports imports(64);
    OldState old_state;
    NewState new_state;
    for (size_t i = 0; i < 64; ++i) {
        old_state.call(imports.what[i]);
        new_state.call(imports.records[i]);
    }
    const size_t per_thread = calls / hw;
    auto run = [&](auto&& body) {
        return ns_per_call(per_thread * hw, [&] {
            std::vector<std::thread> threads;
            for (unsigned t = 0; t < hw; ++t)
                threads.emplace_back([&, t] {
                    uint32_t acc = 0;
                    for (size_t i = 0; i < per_thread; ++i) acc += body((i * 7 + t) & 63);
                    g_sink = g_sink + acc;
                });
            for (auto& th : threads) th.join();
        });
    };
    const double old_ns = run([&](size_t k) { return old_state.call(imports.what[k]).cls; });
    const double new_ns = run([&](size_t k) { return new_state.call(imports.records[k]).cls; });
    std::printf("%10s %10s %9s\n%10.2f %10.2f %8.1fx\n", "old", "new", "speedup", old_ns, new_ns, old_ns / new_ns);
}

}  // namespace

int main(int argc, char** argv) {
    const bool longer = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    fixed_cases();
    random_cases();
    threaded_cases();
    allocation_counts();
    bench(longer);
    if (g_failures) {
        std::printf("\n%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
