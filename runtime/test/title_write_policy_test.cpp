// TitleWritePolicy (src/title_write_policy.h) with a scripted and a randomised write pattern over fake memory: a fake
// guard (a page is read-only while armed; the first write to it "faults" into written()), the title's writes, and the
// owner's copy taken in collect(). Pages written in most submissions (kPromote = 6 of 8) stop faulting, 1/frame and
// rare pages stay cold without over-copying on empty submissions, and no write is missed: the copy equals the memory
// byte for byte after every collect, except pages written during that collect's own copy, which the next one brings
// over.
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
        // Promote the first kHot pages by writing them every submission for kPromote intervals.
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
    {  // submission-hot threshold: kPromote == 6 of 8 promotes, 5 of 8 stays cold
        Fake f;
        Policy policy(f);
        f.policy = &policy;
        policy.add(kBase, kBase + 2 * kPage);
        policy.collect([&](std::uintptr_t p) { f.take(p); });
        // Page 0 written on 6 of 8 submissions; page 1 written on 5 of 8 submissions.
        for (unsigned sub = 0; sub < 8; ++sub) {
            if (sub != 2 && sub != 5) f.write(kBase + sub, std::uint8_t(sub + 1));
            if (sub != 1 && sub != 4 && sub != 7) f.write(kBase + kPage + sub, std::uint8_t(sub + 1));
            policy.collect([&](std::uintptr_t p) { f.take(p); });
            EXPECT(f.same(kBase) && f.same(kBase + kPage));
            policy.frame();
        }
        EXPECT(Policy::kPromote == 6 && Policy::kHistory == 8);
        EXPECT(policy.hot(kBase));
        EXPECT(!policy.hot(kBase + kPage));
    }
    {  // 1-write/frame pages stay cold across low-sub and high-sub frames: zero per-submission over-copying,
       // zero copies on empty submissions, while truly intra-frame hot pages (6+/8 subs) promote cleanly.
        Fake f;
        Policy policy(f);
        f.policy = &policy;
        policy.add(kBase, kBase + 2 * kPage);
        policy.collect([&](std::uintptr_t p) { f.take(p); });
        // 20 frames of 4 submissions/frame: page 0 written once per frame (sub 0), page 1 written every sub.
        std::uint64_t p0_copies = 0, p0_empty_sub_copies = 0;
        for (unsigned frame = 1; frame <= 20; ++frame) {
            for (unsigned sub = 0; sub < 4; ++sub) {
                const bool wrote_p0 = (sub == 0);
                if (wrote_p0) f.write(kBase + frame, std::uint8_t(frame));
                f.write(kBase + kPage + sub, std::uint8_t(frame + sub));
                policy.collect([&](std::uintptr_t p) {
                    if (p == kBase) {
                        ++p0_copies;
                        if (!wrote_p0) ++p0_empty_sub_copies;
                    }
                    f.take(p);
                });
                EXPECT(f.same(kBase) && f.same(kBase + kPage));
                policy.frame();
            }
        }
        EXPECT(!policy.hot(kBase));         // 1/frame (2 of 8 subs) never promotes to hot
        EXPECT(policy.hot(kBase + kPage));  // written every sub: hot
        EXPECT(p0_copies == 20);            // copied strictly once per written submission
        EXPECT(p0_empty_sub_copies == 0);   // zero over-copying on empty submissions
    }
    {  // batched arming: Guard with arm_batch() receives sorted pages in one call per collect(), handles partial refusal
        struct BatchGuard {
            std::set<std::uintptr_t> armed;
            std::uintptr_t refuse_page = 0;
            std::uint64_t single_arms = 0, batch_calls = 0, batch_pages = 0;
            bool sorted_ok = true;

            bool arm(std::uintptr_t page) {
                ++single_arms;
                if (page == refuse_page) return false;
                armed.insert(page);
                return true;
            }
            void arm_batch(const std::uintptr_t* pages, std::uint8_t* results, std::size_t count) {
                ++batch_calls;
                batch_pages += count;
                for (std::size_t i = 0; i < count; ++i) {
                    if (i > 0 && pages[i] <= pages[i - 1]) sorted_ok = false;
                    if (pages[i] == refuse_page) {
                        results[i] = 0;
                    } else {
                        armed.insert(pages[i]);
                        results[i] = 1;
                    }
                }
            }
            void disarm(std::uintptr_t page) { armed.erase(page); }
        } bg;

        leonida::TitleWritePolicy<BatchGuard> policy(bg);
        policy.add(kBase, kBase + 16 * kPage);
        bg.refuse_page = kBase + 5 * kPage;
        std::vector<std::uintptr_t> copied;
        policy.collect([&](std::uintptr_t p) { copied.push_back(p); });
        EXPECT(copied.size() == 16);
        EXPECT(bg.single_arms == 0 && bg.batch_calls == 1 && bg.batch_pages == 16 && bg.sorted_ok);
        EXPECT(policy.stats().batched_arm_calls == 1 && policy.stats().arm_failures == 1);
        EXPECT(bg.armed.size() == 15 && !bg.armed.count(kBase + 5 * kPage));

        // Next collect retries only the refused page (5) plus newly written pages (10, 2) in sorted order.
        bg.refuse_page = 0;
        policy.written(kBase + 10 * kPage);
        policy.written(kBase + 2 * kPage);
        copied.clear();
        policy.collect([&](std::uintptr_t p) { copied.push_back(p); });
        EXPECT(copied.size() == 3 && copied[0] == kBase + 2 * kPage && copied[1] == kBase + 5 * kPage &&
               copied[2] == kBase + 10 * kPage);
        EXPECT(bg.batch_calls == 2 && bg.batch_pages == 19 && bg.sorted_ok);
        EXPECT(bg.armed.size() == 16);
    }
    {  // two-tier per-frame promotion (advance_frame): pages written once every 30 submissions promote within
       // kFramePromote frames without taking faults every frame, and has_pending_work() is O(1) false when clean.
        Fake f;
        Policy policy(f);
        f.policy = &policy;
        policy.add(kBase, kBase + kPages * kPage);
        EXPECT(policy.has_pending_work());
        policy.collect([&](std::uintptr_t p) { f.take(p); });
        EXPECT(!policy.has_pending_work());  // all 64 pages armed & clean: O(1) clean fast-path returns false!

        constexpr unsigned kSubsPerFrame = 30;
        std::uint64_t steady_frame_hot_faults = 0;
        for (unsigned frame = 1; frame <= 40; ++frame) {
            const unsigned write_sub = (frame - 1) % kSubsPerFrame;  // vary which of the 30 subs writes page 0
            for (unsigned sub = 0; sub < kSubsPerFrame; ++sub) {
                if (sub == write_sub) {
                    const std::uint64_t before = f.faults_by_page[kBase];
                    f.write(kBase + (frame * 17 + sub) % kPage, std::uint8_t(frame + sub));
                    // After kFramePromote frames, page 0 stays unguarded except during kFrameProbe's probe window.
                    const unsigned probe_phase = frame % Policy::kFrameProbe;
                    if (frame > Policy::kFramePromote + 1 && probe_phase != 1 && probe_phase != 2)
                        steady_frame_hot_faults += f.faults_by_page[kBase] - before;
                }
                if (frame % 10 == 0 && sub == 5) {
                    f.write(kBase + kPage + frame, std::uint8_t(frame));
                }
                policy.collect([&](std::uintptr_t p) { f.take(p); });
                EXPECT(f.same(kBase) && f.same(kBase + kPage));
                policy.frame();
            }
            policy.advance_frame();
        }
        EXPECT(policy.frame_hot(kBase));
        EXPECT(policy.hot(kBase));
        EXPECT(!policy.sub_hot(kBase));       // 1/30 subs is frame-hot, not submission-hot
        EXPECT(!policy.hot(kBase + kPage));   // 1/10 frames stays cold
        EXPECT(steady_frame_hot_faults == 0);

        // Stop writing page 0: after a probe cycle across quiet 30-sub frames, it demotes and has_pending_work() is false.
        for (unsigned frame = 0; frame < Policy::kFrameProbe + 4; ++frame) {
            for (unsigned sub = 0; sub < kSubsPerFrame; ++sub) {
                policy.collect([&](std::uintptr_t p) { f.take(p); });
                policy.frame();
            }
            policy.advance_frame();
        }
        EXPECT(!policy.hot(kBase) && !policy.frame_hot(kBase));
        EXPECT(!policy.has_pending_work());
        std::puts("two-tier advance_frame: 1/30-sub per-frame page promoted with zero steady faults, clean fast-path ok");
    }
    if (failures) return 1;
    std::puts("title write policy: hot pages stop faulting, cold pages fault once an interval, nothing missed");
    return 0;
}
