// PipelineCompiler (shims/d3d12_pipeline_compile.h): one job per owner runs at a time across the general and
// graphics-only workers, so after_running() never deletes a PSO while another build of it runs on the second worker.
// Windows only, no GPU, no Vulkan device, no game data.
#include <windows.h>
#include "../shims/d3d12_pipeline_compile.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using rt::d3d12::PipelineCompiler;

constexpr auto kTimeout = std::chrono::seconds(5);

struct Event {
    std::mutex mutex;
    std::condition_variable cv;
    bool signaled = false;

    void set() {
        {
            std::lock_guard lock(mutex);
            signaled = true;
        }
        cv.notify_all();
    }

    bool wait_for(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex);
        return cv.wait_for(lock, timeout, [this] { return signaled; });
    }
};

// Two graphics jobs for owner A: while the first is running, the second stays queued (the second worker skips it)
// and after_running drops the queued second job and runs its callback once after the first finishes, with no job of
// A running concurrently.
int same_owner_graphics_after_running(PipelineCompiler& compiler) {
    struct State {
        Event job1_started;
        Event job1_release;
        Event job1_done;
        Event job2_started_ev;
        Event last_done;
        std::atomic<int> active_a{0};
        std::atomic<bool> job2_started{false};
        std::atomic<bool> job2_running{false};
        std::atomic<int> last_runs{0};
        std::atomic<bool> last_saw_job2_started{false};
        std::atomic<bool> last_saw_job2_running{false};
        std::atomic<bool> last_saw_active_a{false};
    };

    auto s = std::make_shared<State>();
    struct Guard {
        std::shared_ptr<State> state;
        ~Guard() { state->job1_release.set(); }
    } guard{s};

    int owner_a = 0;
    int device = 0;

    EXPECT(compiler.submit(
        &owner_a, &device, 0,
        [s] {
            s->active_a.fetch_add(1, std::memory_order_acq_rel);
            s->job1_started.set();
            s->job1_release.wait_for(kTimeout);
            s->active_a.fetch_sub(1, std::memory_order_acq_rel);
            s->job1_done.set();
        },
        true));

    EXPECT(s->job1_started.wait_for(kTimeout));
    const std::uint64_t dropped_before = compiler.stats().dropped;

    EXPECT(compiler.submit(
        &owner_a, &device, 1,
        [s] {
            s->active_a.fetch_add(1, std::memory_order_acq_rel);
            s->job2_running.store(true, std::memory_order_release);
            s->job2_started.store(true, std::memory_order_release);
            s->job2_started_ev.set();
            s->job1_release.wait_for(std::chrono::milliseconds(50));
            s->job2_running.store(false, std::memory_order_release);
            s->active_a.fetch_sub(1, std::memory_order_acq_rel);
        },
        true));

    // Hold job 1 for 200 ms: job 2 must not start on the other worker.
    const bool job2_started_early = s->job2_started_ev.wait_for(std::chrono::milliseconds(200));
    EXPECT(!job2_started_early);
    EXPECT(!s->job2_started.load(std::memory_order_acquire));

    const bool deferred = compiler.after_running(&owner_a, [s] {
        s->last_runs.fetch_add(1, std::memory_order_acq_rel);
        if (s->job2_started.load(std::memory_order_acquire))
            s->last_saw_job2_started.store(true, std::memory_order_release);
        if (s->job2_running.load(std::memory_order_acquire))
            s->last_saw_job2_running.store(true, std::memory_order_release);
        if (s->active_a.load(std::memory_order_acquire) != 0)
            s->last_saw_active_a.store(true, std::memory_order_release);
        s->last_done.set();
    });
    EXPECT(deferred);

    s->job1_release.set();
    EXPECT(s->job1_done.wait_for(kTimeout));
    EXPECT(s->last_done.wait_for(kTimeout));
    compiler.wait_idle();

    EXPECT(s->last_runs.load(std::memory_order_acquire) == 1);
    EXPECT(!s->last_saw_active_a.load(std::memory_order_acquire));
    EXPECT(!s->last_saw_job2_started.load(std::memory_order_acquire));
    EXPECT(!s->last_saw_job2_running.load(std::memory_order_acquire));
    EXPECT(!s->job2_started.load(std::memory_order_acquire));
    EXPECT(compiler.stats().dropped == dropped_before + 1);
    std::printf("same_owner_graphics_after_running: passed\n");
    return 0;
}

// Graphics jobs of different owners still run concurrently across both workers.
int different_owners_graphics_concurrent(PipelineCompiler& compiler) {
    struct State {
        Event a_started;
        Event a_release;
        Event a_done;
        Event b_started;
        Event b_done;
        std::atomic<bool> a_released{false};
        std::atomic<bool> a_held_when_b_started{false};
    };

    auto s = std::make_shared<State>();
    struct Guard {
        std::shared_ptr<State> state;
        ~Guard() { state->a_release.set(); }
    } guard{s};

    int owner_a = 0;
    int owner_b = 0;
    int device = 0;

    EXPECT(compiler.submit(
        &owner_a, &device, 0,
        [s] {
            s->a_started.set();
            s->a_release.wait_for(kTimeout);
            s->a_done.set();
        },
        true));

    EXPECT(s->a_started.wait_for(kTimeout));

    EXPECT(compiler.submit(
        &owner_b, &device, 0,
        [s] {
            s->a_held_when_b_started.store(!s->a_released.load(std::memory_order_acquire), std::memory_order_release);
            s->b_started.set();
            s->b_done.set();
        },
        true));

    EXPECT(s->b_started.wait_for(kTimeout));
    EXPECT(s->a_held_when_b_started.load(std::memory_order_acquire));

    s->a_released.store(true, std::memory_order_release);
    s->a_release.set();
    EXPECT(s->a_done.wait_for(kTimeout));
    EXPECT(s->b_done.wait_for(kTimeout));
    compiler.wait_idle();
    std::printf("different_owners_graphics_concurrent: passed\n");
    return 0;
}

// Two compute (non-graphics) jobs of the same owner run in priority order without overlap.
int same_owner_compute_serial(PipelineCompiler& compiler) {
    struct State {
        Event job1_started;
        Event job1_release;
        Event job1_done;
        Event job2_started;
        Event job2_done;
        std::mutex order_mutex;
        std::vector<int> order;
        std::atomic<int> active{0};
        std::atomic<bool> overlapped{false};
    };

    auto s = std::make_shared<State>();
    struct Guard {
        std::shared_ptr<State> state;
        ~Guard() { state->job1_release.set(); }
    } guard{s};

    int owner_c = 0;
    int device = 0;

    EXPECT(compiler.submit(
        &owner_c, &device, 0,
        [s] {
            if (s->active.fetch_add(1, std::memory_order_acq_rel) != 0)
                s->overlapped.store(true, std::memory_order_release);
            {
                std::lock_guard lock(s->order_mutex);
                s->order.push_back(1);
            }
            s->job1_started.set();
            s->job1_release.wait_for(kTimeout);
            s->active.fetch_sub(1, std::memory_order_acq_rel);
            s->job1_done.set();
        },
        false));

    EXPECT(s->job1_started.wait_for(kTimeout));

    EXPECT(compiler.submit(
        &owner_c, &device, 1,
        [s] {
            if (s->active.fetch_add(1, std::memory_order_acq_rel) != 0)
                s->overlapped.store(true, std::memory_order_release);
            {
                std::lock_guard lock(s->order_mutex);
                s->order.push_back(2);
            }
            s->job2_started.set();
            s->active.fetch_sub(1, std::memory_order_acq_rel);
            s->job2_done.set();
        },
        false));

    EXPECT(!s->job2_started.wait_for(std::chrono::milliseconds(50)));

    s->job1_release.set();
    EXPECT(s->job1_done.wait_for(kTimeout));
    EXPECT(s->job2_done.wait_for(kTimeout));
    compiler.wait_idle();

    EXPECT(!s->overlapped.load(std::memory_order_acquire));
    EXPECT(s->order.size() == 2 && s->order[0] == 1 && s->order[1] == 2);
    std::printf("same_owner_compute_serial: passed\n");
    return 0;
}

// A non-graphics job held back by a same-owner graphics job on the graphics worker wakes and runs once that finishes.
int held_back_job_wakes_on_finish(PipelineCompiler& compiler) {
    struct State {
        Event x_started;
        Event x_release;
        Event x_done;
        Event d_gfx_started;
        Event d_gfx_release;
        Event d_gfx_done;
        Event d_cs_started;
        Event d_cs_done;
        std::atomic<int> active_d{0};
        std::atomic<bool> overlapped{false};
    };

    auto s = std::make_shared<State>();
    struct Guard {
        std::shared_ptr<State> state;
        ~Guard() {
            state->x_release.set();
            state->d_gfx_release.set();
        }
    } guard{s};

    int owner_x = 0;
    int owner_d = 0;
    int device = 0;

    // Occupy the general worker so D's graphics job runs on the graphics-only worker.
    EXPECT(compiler.submit(
        &owner_x, &device, 0,
        [s] {
            s->x_started.set();
            s->x_release.wait_for(kTimeout);
            s->x_done.set();
        },
        false));
    EXPECT(s->x_started.wait_for(kTimeout));

    EXPECT(compiler.submit(
        &owner_d, &device, 0,
        [s] {
            if (s->active_d.fetch_add(1, std::memory_order_acq_rel) != 0)
                s->overlapped.store(true, std::memory_order_release);
            s->d_gfx_started.set();
            s->d_gfx_release.wait_for(kTimeout);
            s->active_d.fetch_sub(1, std::memory_order_acq_rel);
            s->d_gfx_done.set();
        },
        true));
    EXPECT(s->d_gfx_started.wait_for(kTimeout));

    // Free the general worker, then queue a compute job for D that only the general worker can take.
    s->x_release.set();
    EXPECT(s->x_done.wait_for(kTimeout));

    EXPECT(compiler.submit(
        &owner_d, &device, 1,
        [s] {
            if (s->active_d.fetch_add(1, std::memory_order_acq_rel) != 0)
                s->overlapped.store(true, std::memory_order_release);
            s->d_cs_started.set();
            s->active_d.fetch_sub(1, std::memory_order_acq_rel);
            s->d_cs_done.set();
        },
        false));
    EXPECT(!s->d_cs_started.wait_for(std::chrono::milliseconds(50)));

    s->d_gfx_release.set();
    EXPECT(s->d_gfx_done.wait_for(kTimeout));
    EXPECT(s->d_cs_done.wait_for(kTimeout));
    compiler.wait_idle();

    EXPECT(!s->overlapped.load(std::memory_order_acquire));
    std::printf("held_back_job_wakes_on_finish: passed\n");
    return 0;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    auto& compiler = PipelineCompiler::instance();
    compiler.retain();
    if (same_owner_graphics_after_running(compiler) || different_owners_graphics_concurrent(compiler) ||
        same_owner_compute_serial(compiler) || held_back_job_wakes_on_finish(compiler)) {
        compiler.stop_for_exit(kTimeout);
        return 1;
    }
    compiler.release();
    std::printf("d3d12_compile_owner_test: all passed\n");
    return 0;
}
