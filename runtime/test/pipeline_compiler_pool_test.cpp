// The rules of rt::d3d12::PipelineCompiler's workers (shims/d3d12_pipeline_compile.h), with made-up jobs:
//   threads      LEONIDA_COMPILE_THREADS as text: the default, 1, 0, not a number.
//   lane         with N workers at most N-1 heavy jobs run (large demanded builds, and every speculative one), and a
//                small demanded build runs while they are held; a large demanded build waits for a place and gets it
//                before the speculative ones.
//   very large   of the builds submitted as huge, as many at a time as LEONIDA_COMPILE_HEAVY_MAX says (1), demanded or
//                speculative, with idle workers; a large build and a small one are not held by them.
//   two freed    a job whose end lets two others start: both do.
//   owner        one job of an owner at a time, the others' jobs meanwhile; after_running() with one job running and
//                two queued drops the two and runs `last` once, after the job.
//   checkpoint   starts only when no worker is in a build; has_compile_waiter() counts a build that is running; a
//                build starts while a checkpoint runs.
//   one worker   the order there was before the pool: demanded, speculative by size, checkpoint.
//   cancel       waits for every running job of a device; release() from inside a job; stop_for_exit() with jobs
//                running and with none.
//   mixed        thousands of short jobs: no owner twice at a time, never more than N-1 heavy or one very large,
//                every job run.
// The jobs wait on events the test holds, so "running" is a state and not a race; a check that waits gives up after
// 10 s, and the whole test after 120 s, so a missing wake-up fails it and does not hang it. No Vulkan device, no GPU.
#include "../shims/d3d12_pipeline_compile.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using rt::d3d12::PipelineCompiler;
using Size = PipelineCompiler::Size;

// Ends the process: the workers a failed check leaves in their jobs would keep the rest from ending.
#define CHECK(condition) do {                                                           \
    if (!(condition)) {                                                                 \
        std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition);               \
        std::fflush(stderr);                                                            \
        std::_Exit(1);                                                                  \
    }                                                                                   \
} while (0)

bool eventually(const std::function<bool()>& what) {
    const auto end = std::chrono::steady_clock::now() + 10s;
    while (!what()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

// Long enough for a worker that may start a job to have started it: what "does not start" is measured against.
void settle() { std::this_thread::sleep_for(150ms); }

class Flag {
public:
    void set() {
        {
            std::lock_guard lock(mutex_);
            on_ = true;
        }
        changed_.notify_all();
    }
    bool wait() {
        std::unique_lock lock(mutex_);
        return changed_.wait_for(lock, 10s, [this] { return on_; });
    }
    void wait_forever() {
        std::unique_lock lock(mutex_);
        changed_.wait(lock, [this] { return on_; });
    }
    bool is() {
        std::lock_guard lock(mutex_);
        return on_;
    }

private:
    std::mutex mutex_;
    std::condition_variable changed_;
    bool on_ = false;
};

// A job that says when it started and stays in its work until go is set.
struct Held {
    Flag started, go, done;
    std::function<void()> job() {
        return [this] {
            started.set();
            go.wait_forever();
            done.set();
        };
    }
};

constexpr std::size_t kCheckpoint = SIZE_MAX;
int owners[64];  // addresses to file jobs under
int devices[4];

std::size_t count_started(std::vector<std::unique_ptr<Held>>& jobs) {
    std::size_t n = 0;
    for (auto& job : jobs) n += job->started.is();
    return n;
}

void test_threads() {
    using rt::d3d12::compile_threads;
    CHECK(compile_threads("", 16) == 8 && compile_threads(nullptr, 32) == 8 && compile_threads("", 12) == 6);
    CHECK(compile_threads("", 4) == 2 && compile_threads("", 2) == 2 && compile_threads("", 0) == 2);  // at least 2
    CHECK(compile_threads("1", 16) == 1);  // the one thread there was
    CHECK(compile_threads("0", 16) == 1);
    CHECK(compile_threads("3", 16) == 3 && compile_threads("12", 4) == 12);
    CHECK(compile_threads("many", 16) == 8 && compile_threads("-1", 16) == 8);
    using rt::d3d12::compile_heavy_max;
    CHECK(compile_heavy_max("") == 1 && compile_heavy_max(nullptr) == 1 && compile_heavy_max("no") == 1);
    CHECK(compile_heavy_max("1") == 1 && compile_heavy_max("3") == 3 && compile_heavy_max("0") == UINT_MAX);
    std::printf("threads: half the processors, 2 to 8; 1 keeps one thread; one very large build at a time\n");
}

void test_lane(unsigned workers) {
    const std::size_t lane = workers - 1;
    // Large demanded builds: more than there are workers.
    {
        std::vector<std::unique_ptr<Held>> heavy;
        Held small, late;
        PipelineCompiler pool(workers);
        CHECK(pool.threads() == workers);
        for (unsigned i = 0; i < workers + 2; ++i) {
            heavy.push_back(std::make_unique<Held>());
            CHECK(pool.submit(&owners[i], &devices[0], 0, heavy.back()->job(), Size::large));
        }
        CHECK(eventually([&] { return count_started(heavy) == lane; }));
        settle();
        CHECK(count_started(heavy) == lane);  // one worker is kept out of them
        // The small demanded build was submitted after all of them, and runs while they are held.
        const auto start = std::chrono::steady_clock::now();
        small.go.set();
        CHECK(pool.submit(&owners[40], &devices[0], 0, small.job()));
        CHECK(small.done.wait());
        const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        CHECK(count_started(heavy) == lane);
        std::printf("lane (%u workers): %zu of %u large builds running, a small one ran past them in %lld ms\n",
                    workers, lane, workers + 2, static_cast<long long>(took));
        // A place that frees goes to the next large build, in the order they came.
        heavy[0]->go.set();
        CHECK(heavy[0]->done.wait());
        CHECK(heavy[lane]->started.wait());
        settle();
        CHECK(count_started(heavy) == lane + 1);
        for (auto& job : heavy) job->go.set();
        for (auto& job : heavy) CHECK(job->done.wait());
        pool.wait_idle();
        CHECK(pool.stats().completed == workers + 3);
    }
    // The warm-up is heavy too, and a large demanded build goes before what is left of it.
    {
        std::vector<std::unique_ptr<Held>> warm;
        Held small, large;
        PipelineCompiler pool(workers);
        for (unsigned i = 0; i < workers + 2; ++i) {
            warm.push_back(std::make_unique<Held>());
            CHECK(pool.submit(&owners[i], &devices[0], 1000 + i, warm.back()->job()));
        }
        CHECK(eventually([&] { return count_started(warm) == lane; }));
        settle();
        CHECK(count_started(warm) == lane);
        CHECK(pool.submit(&owners[41], &devices[0], 0, large.job(), Size::large));
        small.go.set();
        CHECK(pool.submit(&owners[40], &devices[0], 0, small.job()));
        CHECK(small.done.wait());
        settle();
        CHECK(!large.started.is());  // no place: the free worker is not for it
        CHECK(count_started(warm) == lane);
        // Smallest first, so the first `lane` of them are the ones running.
        warm[0]->go.set();
        CHECK(large.started.wait());
        settle();
        CHECK(count_started(warm) == lane);  // the place went to the demanded build
        large.go.set();
        for (auto& job : warm) job->go.set();
        for (auto& job : warm) CHECK(job->done.wait());
        pool.wait_idle();
        std::printf("lane (%u workers): %zu speculative builds running, a small demanded one ran past them, a large "
                    "demanded one took the next place\n", workers, lane);
    }
}

// Very large builds: `limit` of them at a time, demanded or speculative, whatever the workers that are free. The other
// heavy builds and the small ones are not held by that.
void test_huge(unsigned workers, unsigned limit) {
    std::vector<std::unique_ptr<Held>> huge;
    Held large, small, warm;
    PipelineCompiler pool(workers, limit);
    for (unsigned i = 0; i < limit + 2; ++i) {
        huge.push_back(std::make_unique<Held>());
        CHECK(pool.submit(&owners[i], &devices[0], 0, huge.back()->job(), Size::huge));
    }
    CHECK(pool.submit(&owners[30], &devices[0], 2000000, warm.job(), Size::huge));  // a warm-up of that size
    CHECK(eventually([&] { return count_started(huge) == limit; }));
    settle();
    CHECK(count_started(huge) == limit && !warm.started.is());
    // A large build that is not very large takes a place in the lane beside them, and a small one runs.
    CHECK(pool.submit(&owners[31], &devices[0], 0, large.job(), Size::large));
    small.go.set();
    CHECK(pool.submit(&owners[32], &devices[0], 0, small.job()));
    CHECK(small.done.wait());
    CHECK(large.started.wait());
    settle();
    CHECK(count_started(huge) == limit && !warm.started.is());
    // The place of one that ends goes to the next, the demanded ones first.
    huge[0]->go.set();
    CHECK(huge[limit]->started.wait());
    settle();
    CHECK(count_started(huge) == limit + 1 && !warm.started.is());
    large.go.set();
    for (auto& job : huge) job->go.set();
    CHECK(warm.started.wait());
    warm.go.set();
    pool.wait_idle();
    std::printf("very large (%u workers, %u at a time): %u of %u demanded and one speculative held back, a large and "
                "a small build meanwhile\n", workers, limit, 2u, limit + 2);
}

// The end of one job lets two start: a small build of its owner and, in the place it leaves, a heavy one. The worker
// that ended it takes the first; another one has to be told of the second.
void test_two_freed(unsigned workers) {
    const std::size_t lane = workers - 1;
    std::vector<std::unique_ptr<Held>> heavy;
    Held same_owner, next_heavy;
    PipelineCompiler pool(workers);
    for (unsigned i = 0; i < lane; ++i) {
        heavy.push_back(std::make_unique<Held>());
        CHECK(pool.submit(&owners[i], &devices[0], 0, heavy.back()->job(), Size::large));
        CHECK(heavy.back()->started.wait());
    }
    CHECK(pool.submit(&owners[0], &devices[0], 0, same_owner.job()));  // waits for its owner's build
    CHECK(pool.submit(&owners[50], &devices[0], 900, next_heavy.job()));  // waits for a place
    settle();
    CHECK(!same_owner.started.is() && !next_heavy.started.is());
    heavy[0]->go.set();
    CHECK(same_owner.started.wait());
    CHECK(next_heavy.started.wait());
    same_owner.go.set();
    next_heavy.go.set();
    for (auto& job : heavy) job->go.set();
    pool.wait_idle();
    std::printf("two freed (%u workers): the end of a large build started a small one of its owner and the next "
                "heavy one\n", workers);
}

void test_owner(unsigned workers) {
    if (workers > 1) {
        std::vector<std::unique_ptr<Held>> same;
        Held other;
        PipelineCompiler pool(workers);
        for (unsigned i = 0; i < 3; ++i) {
            same.push_back(std::make_unique<Held>());
            CHECK(pool.submit(&owners[0], &devices[0], 0, same.back()->job()));
        }
        other.go.set();
        CHECK(pool.submit(&owners[1], &devices[0], 0, other.job()));
        CHECK(other.done.wait());  // behind them in the queue, and not held by them
        settle();
        CHECK(count_started(same) == 1 && same[0]->started.is());
        same[0]->go.set();
        CHECK(same[1]->started.wait());
        settle();
        CHECK(count_started(same) == 2);
        same[1]->go.set();
        same[2]->go.set();
        CHECK(same[2]->done.wait());
        pool.wait_idle();
        std::printf("owner (%u workers): three jobs of one owner ran one at a time, another owner's meanwhile\n",
                    workers);
    }
    // after_running: one job running and two queued. `last` is the PSO's deletion in the runtime.
    {
        Held running;
        std::atomic<int> queued_ran{0}, last_ran{0};
        std::atomic<bool> last_after_job{false};
        PipelineCompiler pool(workers);
        CHECK(pool.submit(&owners[0], &devices[0], 0, running.job()));
        CHECK(running.started.wait());
        CHECK(pool.submit(&owners[0], &devices[0], 0, [&] { ++queued_ran; }));
        CHECK(pool.submit(&owners[0], &devices[0], 500, [&] { ++queued_ran; }));
        settle();
        const auto before = pool.stats();
        CHECK(pool.after_running(&owners[0], [&] {
            last_after_job = running.done.is();
            ++last_ran;
        }));
        const auto after = pool.stats();
        CHECK(after.dropped == before.dropped + 2 && after.deferred == before.deferred + 1);
        settle();
        CHECK(last_ran == 0);
        running.go.set();
        CHECK(pool.cancel(&devices[0]) == 0);  // a device's teardown: back after the job and after `last`
        CHECK(last_ran == 1 && last_after_job && queued_ran == 0);
        pool.wait_idle();
        CHECK(!pool.after_running(&owners[0], [&] { ++last_ran; }) && last_ran == 1);  // nothing running: not kept
        std::printf("owner (%u workers): after_running dropped the two queued jobs and ran `last` once, after the "
                    "running one\n", workers);
    }
}

void test_checkpoint(unsigned workers) {
    Held build, heavy, checkpoint, second, during;
    std::atomic<bool> waiter_at_start{true};
    PipelineCompiler pool(workers);
    CHECK(!pool.has_compile_waiter());
    CHECK(pool.submit(&owners[0], &devices[0], 0, build.job()));
    CHECK(build.started.wait());
    CHECK(pool.has_compile_waiter());  // nothing queued, one build running on a worker
    CHECK(pool.submit(&devices[0], &devices[0], kCheckpoint, [&] {
        waiter_at_start = pool.has_compile_waiter();
        checkpoint.job()();
    }));
    settle();
    CHECK(!checkpoint.started.is());  // idle workers, and it waits for the build
    CHECK(pool.submit(&owners[1], &devices[0], 700, heavy.job()));
    if (workers > 1) CHECK(heavy.started.wait());
    build.go.set();
    CHECK(build.done.wait());
    CHECK(heavy.started.wait());
    settle();
    CHECK(!checkpoint.started.is());  // the speculative build is one too
    heavy.go.set();
    CHECK(checkpoint.started.wait());
    CHECK(!waiter_at_start);  // nothing queued and nothing running but the checkpoint itself
    CHECK(!pool.has_compile_waiter());
    if (workers > 1) {
        // A build that comes while the checkpoint runs is not held, and the checkpoint is told.
        CHECK(pool.submit(&owners[2], &devices[0], 0, during.job()));
        CHECK(during.started.wait());
        CHECK(pool.has_compile_waiter());
        // A checkpoint that comes now waits for that build.
        CHECK(pool.submit(&devices[1], &devices[1], kCheckpoint, second.job()));
        settle();
        CHECK(!second.started.is());
        during.go.set();
        CHECK(second.started.wait());
        second.go.set();
    }
    checkpoint.go.set();
    CHECK(checkpoint.done.wait());
    pool.wait_idle();
    std::printf("checkpoint (%u workers): waited for a demanded and a speculative build, then ran\n", workers);
}

void test_one_worker() {
    Held gate;
    std::mutex mutex;
    std::string order;
    const auto note = [&](char c) {
        return [&, c] {
            std::lock_guard lock(mutex);
            order += c;
        };
    };
    PipelineCompiler pool(1);
    CHECK(pool.submit(&owners[0], &devices[0], 0, gate.job()));
    CHECK(gate.started.wait());
    CHECK(pool.submit(&devices[0], &devices[0], kCheckpoint, note('c')));
    CHECK(pool.submit(&owners[1], &devices[0], 300, note('w')));
    CHECK(pool.submit(&owners[2], &devices[0], 100, note('v')));
    CHECK(pool.submit(&owners[3], &devices[0], 0, note('L'), Size::large));
    CHECK(pool.submit(&owners[4], &devices[0], 0, note('d')));
    CHECK(pool.submit(&owners[2], &devices[0], 0, note('e')));
    settle();
    CHECK(order.empty());
    gate.go.set();
    pool.wait_idle();
    std::printf("one worker: order %s\n", order.c_str());
    CHECK(order == "Ldevwc");
}

void test_cancel(unsigned workers) {
    // A device's teardown with its jobs running on several workers, and more queued.
    {
        std::vector<std::unique_ptr<Held>> running;
        std::atomic<int> queued_ran{0};
        std::atomic<bool> returned{false};
        Held other;
        PipelineCompiler pool(workers);
        for (unsigned i = 0; i < (workers > 1 ? workers - 1 : 1); ++i) {
            running.push_back(std::make_unique<Held>());
            CHECK(pool.submit(&owners[i], &devices[0], 0, running.back()->job()));
            CHECK(running.back()->started.wait());
        }
        for (unsigned i = 0; i < 5; ++i) CHECK(pool.submit(&owners[20 + i], &devices[0], 50 + i, [&] { ++queued_ran; }));
        if (workers > 1) {
            CHECK(pool.submit(&owners[30], &devices[1], 0, other.job()));
            CHECK(other.started.wait());
        }
        const auto before = pool.stats();
        std::thread teardown([&] {
            pool.cancel(&devices[0]);
            returned = true;
        });
        for (std::size_t i = 0; i < running.size(); ++i) {
            settle();
            CHECK(!returned);  // one of its jobs is still running
            running[i]->go.set();
        }
        teardown.join();
        CHECK(pool.stats().waited == before.waited + 1 && queued_ran == 0);
        CHECK(!other.done.is());  // the other device's job is not waited for
        other.go.set();
        pool.wait_idle();
        std::printf("cancel (%u workers): waited for the device's %zu running jobs, dropped its queued ones\n", workers,
                    running.size());
    }
    // The last release from inside a job: that worker lets itself go, the others are joined. Not destroyed, as the
    // process's own is not: the worker that let itself go still returns through it.
    {
        auto* pool = new PipelineCompiler(workers);
        Flag released;
        Held neighbour;
        pool->retain();
        neighbour.go.set();
        CHECK(pool->submit(&owners[1], &devices[0], 0, neighbour.job()));
        CHECK(neighbour.done.wait());
        CHECK(pool->submit(&owners[0], &devices[0], 0, [&] {
            pool->release();
            released.set();
        }));
        CHECK(released.wait());
        CHECK(!pool->worker_running());
        pool->wait_idle();
        // A device made afterwards gets new workers.
        Flag again;
        pool->retain();
        CHECK(pool->submit(&owners[0], &devices[0], 0, [&] { again.set(); }));
        CHECK(again.wait() && pool->worker_running());
        pool->wait_idle();
        pool->release();
        CHECK(!pool->worker_running());
        std::printf("release (%u workers): from inside a job, then new workers for the next device\n", workers);
    }
    // Process exit with jobs in the driver: not waited for past the time given, and nothing more is taken.
    {
        auto* pool = new PipelineCompiler(workers);
        std::vector<Held*> held;  // one on every worker; not freed, as the workers that hold them are let go
        std::atomic<int> queued_ran{0};
        for (unsigned i = 0; i < workers; ++i) {
            held.push_back(new Held);
            CHECK(pool->submit(&owners[i], &devices[0], 0, held.back()->job()));
            CHECK(held.back()->started.wait());
        }
        for (unsigned i = 0; i < 3; ++i) CHECK(pool->submit(&owners[20 + i], &devices[0], 10, [&] { ++queued_ran; }));
        CHECK(!pool->stop_for_exit(50ms));
        CHECK(pool->closed() && !pool->worker_running());
        CHECK(!pool->submit(&owners[9], &devices[0], 0, [] {}));
        CHECK(pool->stats().refused == 1 && pool->stats().dropped == 3);
        for (Held* job : held) job->go.set();
        pool->wait_idle();
        CHECK(queued_ran == 0);

        PipelineCompiler idle(workers);
        Flag ran;
        CHECK(idle.submit(&owners[0], &devices[0], 0, [&] { ran.set(); }));
        CHECK(ran.wait());
        idle.wait_idle();
        CHECK(idle.stop_for_exit(10000ms) && !idle.worker_running());
        std::printf("exit (%u workers): a stop with jobs running gave up after its wait; one with none joined\n",
                    workers);
    }
}

// Short jobs of every kind from several threads. Each job counts itself in and out.
void test_mixed(unsigned workers) {
    constexpr unsigned kOwners = 12;
    constexpr unsigned kJobs = 6000;
    const int lane = workers > 1 ? int(workers) - 1 : 1;
    std::atomic<int> in_owner[kOwners]{};
    std::atomic<int> heavy{0}, huge{0}, all{0}, peak_heavy{0}, peak_all{0}, ran{0};
    std::atomic<bool> owner_twice{false}, lane_broken{false}, two_huge{false};
    PipelineCompiler pool(workers, 1);
    const auto lift = [](std::atomic<int>& peak, int value) {
        int seen = peak.load();
        while (seen < value && !peak.compare_exchange_weak(seen, value)) {}
    };
    std::vector<std::thread> submitters;
    for (unsigned t = 0; t < 3; ++t) {
        submitters.emplace_back([&, t] {
            std::uint32_t seed = 12345 + t;
            for (unsigned i = t; i < kJobs; i += 3) {
                seed = seed * 1664525u + 1013904223u;
                const unsigned owner = (seed >> 8) % kOwners;
                const unsigned kind = (seed >> 16) % 8;  // 0: checkpoint, 1-2: large, 3-4: speculative, 5-7: small
                const bool is_huge = kind >= 1 && kind <= 4 && (seed >> 20) % 3 == 0;  // a third of the heavy ones
                const bool is_heavy = kind >= 1 && kind <= 4;
                const std::size_t priority = kind == 0 ? kCheckpoint : kind >= 3 && kind <= 4 ? 100 + (seed >> 24) : 0;
                const unsigned spin = (seed >> 4) % 4;
                CHECK(pool.submit(&owners[owner], &devices[0], priority, [&, owner, is_heavy, is_huge, spin] {
                    if (in_owner[owner].fetch_add(1) != 0) owner_twice = true;
                    if (is_huge && huge.fetch_add(1) != 0) two_huge = true;
                    lift(peak_all, all.fetch_add(1) + 1);
                    if (is_heavy) {
                        const int now = heavy.fetch_add(1) + 1;
                        if (now > lane) lane_broken = true;
                        lift(peak_heavy, now);
                    }
                    // Long enough to overlap with the others, without the scheduler's 15 ms for a sleep.
                    if (spin == 0) std::this_thread::yield();
                    else if (spin == 1) for (volatile int i = 0; i < 20000; i = i + 1) {}
                    if (is_heavy) --heavy;
                    if (is_huge) --huge;
                    --all;
                    --in_owner[owner];
                    ++ran;
                }, is_huge ? Size::huge : kind >= 1 && kind <= 2 ? Size::large : Size::regular));
                if ((seed & 15) == 0) std::this_thread::yield();
            }
        });
    }
    for (auto& thread : submitters) thread.join();
    pool.wait_idle();
    const auto stats = pool.stats();
    std::printf("mixed (%u workers): %d jobs, at most %d at a time and %d heavy (the lane holds %d)\n", workers,
                ran.load(), peak_all.load(), peak_heavy.load(), lane);
    CHECK(ran == int(kJobs) && stats.completed == kJobs && stats.dropped == 0);
    CHECK(!owner_twice);
    CHECK(!lane_broken);
    CHECK(!two_huge);
    CHECK(peak_all <= int(workers));
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // a case that fails or hangs leaves the lines of those before it
    std::thread([] {
        std::this_thread::sleep_for(120s);
        std::fprintf(stderr, "FAIL: not finished after 120 s (a job that never started, or a worker that never came "
                             "back)\n");
        std::fflush(stderr);
        std::_Exit(1);
    }).detach();
    test_threads();
    for (const unsigned workers : {2u, 4u, 8u}) test_lane(workers);
    for (const unsigned workers : {2u, 4u}) test_two_freed(workers);
    test_huge(4, 1);
    test_huge(8, 1);
    test_huge(8, 2);
    for (const unsigned workers : {1u, 4u}) test_owner(workers);
    for (const unsigned workers : {1u, 2u, 4u}) test_checkpoint(workers);
    test_one_worker();
    for (const unsigned workers : {1u, 4u}) test_cancel(workers);
    for (const unsigned workers : {1u, 2u, 4u, 8u}) test_mixed(workers);
    std::printf("ok\n");
    return 0;
}
