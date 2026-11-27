// The "submit frame" and "submit floor" lines (runtime/src/submit_phases.h): their sums against the per-submission
// phases the "submit phases" line adds up.
//   1. A scripted run on the CPU, with known phase times: two frames, a submission on each of two queue kinds, a phase
//      on one of our threads (a WorkerScope), a title call (TitleCall) and a contended lock. The frame sums, the
//      split between title and own threads, the worst frame and the floor must come out as scripted.
//   2. Real submissions through VulkanQueue (empty lists; 77-free: without a Vulkan device this part is skipped and
//      the test still passes): the floor's phase sums must equal what the phase totals grew by, submission for
//      submission, and every one of them is a floor submission.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
namespace sp = leonida::submit_phases;
constexpr std::uint64_t ms = 1000000;  // ns

std::uint64_t load(const std::atomic<std::uint64_t>& a) { return a.load(std::memory_order_relaxed); }

std::uint64_t phase_total() {
    std::uint64_t n = 0;
    for (const auto& p : sp::totals().ns) n += load(p);
    return n;
}

int scripted() {
    sp::FrameTotals& f = sp::frame_totals();
    // Frame 1: a direct submission with heaps 1 ms and replay 2 ms, nothing changed (a floor submission).
    {
        const sp::Submission s(0);
        sp::add(sp::heaps, 1 * ms);
        sp::add(sp::replay, 2 * ms);
    }
    // One of our threads: its 5 ms of textures are the phase line's, not the title threads'.
    std::thread([] {
        const sp::WorkerScope busy(sp::worker_texture_reads);
        sp::add(sp::textures, 5 * ms);
    }).join();
    // A title call that spends about 3 ms, 1 ms of it in a phase: about 2 ms of fences.
    const auto call_start = std::chrono::steady_clock::now();
    {
        const sp::TitleCall call(sp::part_fences);
        sp::add(sp::replay, 1 * ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    const double call_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - call_start).count());
    const std::uint64_t fences = load(f.part_ns[sp::part_fences]);
    EXPECT(fences > 1 * ms && double(fences) <= call_ns - 1e6);
    // A lock another thread holds for 4 ms: counted as a lock wait, inside the other parts (not in the frame's sum).
    std::mutex m;
    std::atomic<bool> taken{false};
    std::thread holder([&] {  // takes the lock and lets it go on its own thread, as a mutex must be
        const std::lock_guard held(m);
        taken.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(4));
    });
    while (!taken.load()) std::this_thread::yield();
    {
        std::unique_lock lock(m, std::defer_lock);
        sp::lock_timed(lock);
    }
    holder.join();
    EXPECT(load(f.part_ns[sp::part_lock_wait]) >= 2 * ms);
    const std::uint64_t frame1 = 3 * ms + 1 * ms + fences;  // heaps, replay, replay in the call, fences
    EXPECT(load(f.current_ns) == frame1);
    sp::frame_presented();
    // Frame 2: a compute submission with geometry 4 ms that uploaded geometry: not a floor submission.
    {
        const sp::Submission s(1);
        sp::add(sp::geometry, 4 * ms);
        sp::note_change(sp::change_geometry);
    }
    sp::frame_presented();

    EXPECT(load(f.frames) == 2);
    EXPECT(load(f.submissions[0]) == 1 && load(f.submissions[1]) == 1 && load(f.submissions[2]) == 0);
    EXPECT(load(f.phase_ns[sp::heaps]) == 1 * ms && load(f.phase_ns[sp::replay]) == 3 * ms);
    EXPECT(load(f.phase_ns[sp::geometry]) == 4 * ms && load(f.phase_ns[sp::textures]) == 0);
    EXPECT(load(f.worker_ns[sp::worker_texture_reads]) > 0);
    // The phase line counts every thread: the title threads' phases and the worker's add up to its total.
    std::uint64_t title = 0;
    for (const auto& p : f.phase_ns) title += load(p);
    EXPECT(phase_total() == title + 5 * ms);
    EXPECT(load(f.worst_ns) == (frame1 > 4 * ms ? frame1 : 4 * ms));
    EXPECT(load(f.all_count) == 2 && load(f.all_ns) == 7 * ms);
    EXPECT(load(f.floor_count) == 1 && load(f.floor_min_ns) == 3 * ms);
    EXPECT(load(f.floor_ns[sp::heaps]) == 1 * ms && load(f.floor_ns[sp::replay]) == 2 * ms &&
           load(f.floor_ns[sp::geometry]) == 0);
    sp::frame_lines(1.0);  // the two lines' format, on stderr; a new window
    EXPECT(load(f.frames) == 0 && load(f.all_count) == 0 && load(f.floor_count) == 0 && load(f.worst_ns) == 0);
    for (const auto& p : f.phase_ns) EXPECT(load(p) == 0);
    // Mirror refresh/write-back inside a list submission: time inside Scope(shadows) is billed to shadows, not lists,
    // and contended lock_timed inside it is billed to lock waits.
    {
        const sp::TitleCall call(sp::part_lists);
        const sp::Scope shadows_scope(sp::shadows);
        std::mutex mirror_m;
        std::atomic<bool> held_flag{false};
        std::thread holder([&] {
            const std::lock_guard held(mirror_m);
            held_flag.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        });
        while (!held_flag.load()) std::this_thread::yield();
        {
            std::unique_lock lock(mirror_m, std::defer_lock);
            sp::lock_timed(lock);
        }
        holder.join();
    }
    EXPECT(load(f.phase_ns[sp::shadows]) >= 2 * ms);
    EXPECT(load(f.part_ns[sp::part_lock_wait]) >= 2 * ms);
    EXPECT(load(f.part_ns[sp::part_lists]) < load(f.phase_ns[sp::shadows]));
    sp::frame_lines(1.0);
    std::printf("scripted: frame sums, title and own threads, worst frame and floor as scripted\n");
    return 0;
}

int submissions() {
    vktest::Device d;
    if (!d.create()) {
        std::printf("no usable Vulkan device: the submissions part is skipped\n");
        return 0;
    }
    VulkanContext context(d.device_proc, d.device, true);
    EXPECT(context.loaded());
    VulkanQueue direct(context, d.queue, d.family, VulkanQueue::QueueKind::direct);
    EXPECT(direct.ok());
    CommandList list;
    EXPECT(list.close());
    const CommandList* lists[] = {&list};
    sp::FrameTotals& f = sp::frame_totals();
    const std::uint64_t phases_before = phase_total(), submitted_before = load(sp::totals().submissions);
    const std::uint64_t all_before = load(f.all_count), all_ns_before = load(f.all_ns);
    const std::uint64_t floor_before = load(f.floor_count), direct_before = load(f.submissions[0]);
    std::uint64_t floor_ns_before = 0;
    for (const auto& p : f.floor_ns) floor_ns_before += load(p);
    constexpr std::uint64_t kSubmissions = 20;
    for (std::uint64_t i = 0; i < kSubmissions; ++i) EXPECT(direct.execute(lists) == ReplayResult::success);
    EXPECT(direct.idle());
    std::uint64_t floor_ns = 0;
    for (const auto& p : f.floor_ns) floor_ns += load(p);
    const std::uint64_t phases = phase_total() - phases_before;
    std::printf("submissions: %llu, phases %.3f ms in all, floor %llu of them, %.3f ms\n",
                static_cast<unsigned long long>(load(f.all_count) - all_before), double(phases) * 1e-6,
                static_cast<unsigned long long>(load(f.floor_count) - floor_before),
                double(floor_ns - floor_ns_before) * 1e-6);
    EXPECT(load(sp::totals().submissions) - submitted_before == kSubmissions);
    EXPECT(load(f.submissions[0]) - direct_before == kSubmissions);
    EXPECT(load(f.all_count) - all_before == kSubmissions);
    EXPECT(load(f.floor_count) - floor_before == kSubmissions);  // empty lists: nothing to bring up to date
    // Every phase of these submissions ran on this thread inside them: the sums agree to the nanosecond.
    EXPECT(load(f.all_ns) - all_ns_before == phases);
    EXPECT(floor_ns - floor_ns_before == phases);
    EXPECT(phases > 0);
    return 0;
}

}  // namespace

int main() {
    if (scripted() || submissions()) return 1;
    std::printf("submit_frame_test: OK\n");
    return 0;
}
