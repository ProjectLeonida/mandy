// TitleWritePolicy (src/title_write_policy.h) with a scripted and a randomised write pattern over fake memory: a fake
// guard (a page is read-only while armed; the first write to it "faults" into written()), the title's writes, and the
// owner's copy taken in collect(). Pages written in most frames stop faulting, the others fault at most once between
// two collects, and no write is missed: the copy equals the memory byte for byte after every collect, except pages
// written during that collect's own copy, which the next one brings over.
#include "../src/title_write_policy.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace {

constexpr std::uintptr_t kPage = 0x10000, kBase = 0x100000000;
constexpr std::size_t kPages = 64;

struct Fake {
    std::vector<std::uint8_t> memory = std::vector<std::uint8_t>(kPages * kPage), copy = memory;
    std::set<std::uintptr_t> armed;
    bool refuse = false;
    std::uint64_t faults = 0;
    leonida::TitleWritePolicy<Fake>* policy = nullptr;
    std::map<std::uintptr_t, std::uint64_t> faults_by_page;

    bool arm(std::uintptr_t page) {
        if (refuse) return false;
        armed.insert(page);
        return true;
    }
    void disarm(std::uintptr_t page) { armed.erase(page); }
    void write(std::uintptr_t va, std::uint8_t value) {
        const std::uintptr_t page = va & ~(kPage - 1);
        if (armed.erase(page)) {  // the fault: the handler opens the page and tells the policy
            ++faults;
            ++faults_by_page[page];
            policy->written(page);
        }
        memory[va - kBase] = value;
    }
    void take(std::uintptr_t page) {
        std::memcpy(copy.data() + (page - kBase), memory.data() + (page - kBase), kPage);
    }
    bool same(std::uintptr_t page) const {
        return !std::memcmp(copy.data() + (page - kBase), memory.data() + (page - kBase), kPage);
    }
};

int failures = 0;
#define EXPECT(c)                                                          \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);       \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

using Policy = leonida::TitleWritePolicy<Fake>;

}  // namespace

int main() {
    {  // scripted: page 0 written every frame, page 1 every 10th frame, page 2 never after the start
        Fake f;
        Policy policy(f);
        f.policy = &policy;
        policy.add(kBase, kBase + 3 * kPage);
        policy.collect([&](std::uintptr_t p) { f.take(p); });
        std::uint64_t hot_faults_late = 0, rare_faults = 0;
        for (unsigned frame = 1; frame <= 200; ++frame) {
            for (unsigned submission = 0; submission < 4; ++submission) {
                const std::uint64_t before = f.faults_by_page[kBase];
                f.write(kBase + 100 + submission, std::uint8_t(frame + submission));
                // (a probe: armed at the collect after the probe's frame() and observed for the next interval)
                const unsigned phase = frame % Policy::kProbe;
                if (frame > 20 && phase != 1 && phase != 2) hot_faults_late += f.faults_by_page[kBase] - before;
                if (frame % 10 == 0 && submission == 0) {
                    const std::uint64_t b = f.faults_by_page[kBase + kPage];
                    f.write(kBase + kPage + frame, std::uint8_t(frame));
                    f.write(kBase + kPage + frame + 1, std::uint8_t(frame));  // same page, same interval: no fault
                    rare_faults += f.faults_by_page[kBase + kPage] - b;
                }
                policy.collect([&](std::uintptr_t p) { f.take(p); });
                for (std::uintptr_t p = kBase; p < kBase + 3 * kPage; p += kPage) EXPECT(f.same(p));
            }
            policy.frame();
        }
        EXPECT(policy.hot(kBase));            // written every frame: hot
        EXPECT(!policy.hot(kBase + kPage));   // every 10th: cold
        EXPECT(hot_faults_late == 0);         // a hot page stops faulting (but in its probe frames)
        EXPECT(rare_faults == 20);            // a cold page: one fault per written interval
        EXPECT(f.faults_by_page[kBase + 2 * kPage] == 0);
        // The hot page stops being written: a probe frame finds it quiet and it turns cold.
        for (unsigned frame = 0; frame < 2 * Policy::kProbe; ++frame) {
            policy.collect([&](std::uintptr_t p) { f.take(p); });
            policy.frame();
        }
        EXPECT(!policy.hot(kBase));
        EXPECT(policy.stats().demotions == 1);
    }
    {  // randomised: writes anywhere, also during the owner's copy; compared with the memory after every collect
        std::mt19937_64 random(12345);
        Fake f;
        Policy policy(f);
        f.policy = &policy;
        policy.add(kBase, kBase + kPages * kPage);
        std::vector<double> rate(kPages);
        for (auto& r : rate) r = std::uniform_real_distribution<double>(0, 1)(random) < 0.2 ? 0.95 : 0.03;
        std::uint64_t mismatches = 0;
        for (unsigned frame = 0; frame < 400; ++frame) {
            if (frame == 200) f.refuse = true;  // a guard that cannot be set: copied every time
            if (frame == 260) f.refuse = false;
            for (unsigned submission = 0; submission < 3; ++submission) {
                for (std::size_t p = 0; p < kPages; ++p)
                    if (std::uniform_real_distribution<double>(0, 1)(random) < rate[p] / 3)
                        f.write(kBase + p * kPage + random() % kPage, std::uint8_t(random()));
                std::set<std::uintptr_t> during;
                policy.collect([&](std::uintptr_t page) {
                    f.take(page);
                    if (random() % 16 == 0) {  // the title writes while the owner copies: armed first, so it faults
                        const std::uintptr_t other = kBase + (random() % kPages) * kPage;
                        f.write(other + random() % kPage, std::uint8_t(random()));
                        during.insert(other);
                    }
                });
                for (std::size_t p = 0; p < kPages; ++p) {
                    const std::uintptr_t page = kBase + p * kPage;
                    if (!during.count(page) && !f.same(page)) ++mismatches;
                }
            }
            policy.frame();
        }
        policy.collect([&](std::uintptr_t p) { f.take(p); });
        EXPECT(mismatches == 0);
        EXPECT(f.copy == f.memory);  // byte-exact once the last writes are collected
        const auto& s = policy.stats();
        EXPECT(s.hot > 0 && s.cold > 0);
        std::printf("randomised: %zu hot and %zu cold pages; %llu faults, %llu hot and %llu cold copies, %llu "
                    "promotions, %llu demotions, %llu refused arms\n",
                    s.hot, s.cold, static_cast<unsigned long long>(s.faults),
                    static_cast<unsigned long long>(s.hot_copies), static_cast<unsigned long long>(s.cold_copies),
                    static_cast<unsigned long long>(s.promotions), static_cast<unsigned long long>(s.demotions),
                    static_cast<unsigned long long>(s.arm_failures));
    }
    {  // scale + correctness: 200,000 cold pages + 100 hot pages stays O(dirty + hot), sub-millisecond per collect+frame
        struct LightGuard {
            std::uint64_t arms = 0, disarms = 0;
            bool arm(std::uintptr_t) {
                ++arms;
                return true;
            }
            void disarm(std::uintptr_t) { ++disarms; }
        } lg;
        constexpr std::size_t kCold = 200000, kHot = 100, kTotal = kCold + kHot;
        leonida::TitleWritePolicy<LightGuard> policy(lg);
        policy.add(kBase, kBase + kTotal * kPage);
        // Initial collect arms all 200,100 pages once.
        std::size_t initial = 0;
        policy.collect([&](std::uintptr_t) { ++initial; });
        EXPECT(initial == kTotal);
        EXPECT(policy.stats().cold == kTotal && policy.stats().hot == 0);
        // Promote the first kHot pages by writing them every frame for kPromote frames.
        for (unsigned f = 0; f < Policy::kPromote; ++f) {
            for (std::size_t h = 0; h < kHot; ++h) policy.written(kBase + h * kPage);
            std::size_t copied = 0;
            std::uintptr_t prev = 0;
            policy.collect([&](std::uintptr_t p) {
                EXPECT(p > prev);  // delivered in address order
                prev = p;
                ++copied;
            });
            EXPECT(copied == kHot);
            policy.frame();
        }
        EXPECT(policy.stats().hot == kHot && policy.stats().cold == kCold);
        for (std::size_t h = 0; h < kHot; ++h) EXPECT(policy.hot(kBase + h * kPage));
        EXPECT(!policy.hot(kBase + kHot * kPage));
        // Steady-state benchmark: 50 submissions (collect + frame) with 100 hot + 10 cold writes each.
        constexpr unsigned kIters = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned it = 0; it < kIters; ++it) {
            for (unsigned c = 0; c < 10; ++c)
                policy.written(kBase + (kHot + ((it * 13 + c * 97) % kCold)) * kPage);
            std::size_t copied = 0;
            std::uintptr_t prev = 0;
            policy.collect([&](std::uintptr_t p) {
                EXPECT(p > prev);
                prev = p;
                ++copied;
            });
            EXPECT(copied >= 10);  // at least the 10 cold pages (hot pages may skip during probe wait)
            policy.frame();
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double per_iter_ms = ms / double(kIters);
        std::printf("scale (%zu cold + %zu hot): %.3f ms per collect+frame (%.2f ms for %u iters)\n",
                    kCold, kHot, per_iter_ms, ms, kIters);
        EXPECT(per_iter_ms < 1.0);
        // Stop writing the hot pages: they demote after a probe cycle, and removing pages updates stats cleanly.
        for (unsigned f = 0; f < 2 * Policy::kProbe; ++f) {
            policy.collect([&](std::uintptr_t) {});
            policy.frame();
        }
        EXPECT(policy.stats().hot == 0 && policy.stats().cold == kTotal);
        EXPECT(policy.stats().demotions >= kHot);
        policy.remove(kBase, kBase + 50 * kPage);
        EXPECT(policy.stats().cold == kTotal - 50);
    }
    if (failures) return 1;
    std::puts("title write policy: hot pages stop faulting, cold pages fault once an interval, nothing missed");
    return 0;
}
