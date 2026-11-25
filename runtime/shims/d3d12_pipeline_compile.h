// Timing and scheduling for Vulkan pipeline creation on the D3D12 device.
#pragma once

#include "../src/pipeline_workers.h"
#include "../src/vulkan_replay.h"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <string>
#include <utility>
#include <vector>

namespace rt::d3d12 {
inline bool trace_pipeline_compile() {
    static const bool on = [] {
        wchar_t v[2]{};
        return GetEnvironmentVariableW(L"LEONIDA_TRACE_PIPELINE_COMPILE", v, 2) == 1 && v[0] == L'1';
    }();
    return on;
}

inline bool async_replay_compile() {
    static const bool on = [] {
        wchar_t v[2]{};
        return GetEnvironmentVariableW(L"LEONIDA_ASYNC_PIPELINES", v, 2) == 1 && v[0] == L'1';
    }();
    return on && !leonida::strict_replay();
}

// Absolute monotonic timestamps let trace readers distinguish time in the ICD from time queued for the worker.
using CompileClock = std::chrono::steady_clock;
inline long long compile_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(CompileClock::now().time_since_epoch()).count();
}

inline long long compile_wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

template<class Build> VkPipeline replay_pipeline(const char* trigger, Build&& get) {
    if (!trace_pipeline_compile()) return get();
    const auto start = compile_us();
    auto handle = get();
    auto elapsed = compile_us() - start;
    if (elapsed >= 1000)
        std::fprintf(stderr, "[rt] pipeline-use trigger=%s tid=%lu us=%lld ready=%d end_ms=%lld\n",
                     trigger, GetCurrentThreadId(), elapsed, int(handle != VK_NULL_HANDLE), compile_wall_ms());
    return handle;
}

struct CompileTrace {
    VkPipelineCreationFeedback feedback{};
    VkPipelineCreationFeedbackCreateInfo chain{VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                               nullptr, &feedback, 0, nullptr};
    long long started = 0;
    const char* key = "-";

    template<class Info> void begin(Info& info, bool supported, const char* trigger, const char* shaders,
                                    std::uint32_t variant, std::size_t bytes, const std::string& description_key) {
        if (!trace_pipeline_compile()) return;
        if (!description_key.empty()) key = description_key.c_str();
        if (supported) {
            chain.pNext = info.pNext;
            info.pNext = &chain;
        }
        std::fprintf(stderr, "[rt] pipeline-compile-begin trigger=%s tid=%lu bytes=%zu variant=%u begin_ms=%lld key=%s shaders=%s\n",
                     trigger, GetCurrentThreadId(), bytes, variant, compile_wall_ms(), key, shaders);
        started = compile_us();
    }

    void end(const char* trigger, const char* shaders, std::uint32_t variant, std::size_t bytes, VkResult result) const {
        if (!started) return;
        const auto ended = compile_us();
        const int hit = !(feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) ? -1 :
                        int((feedback.flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT) != 0);
        std::fprintf(stderr, "[rt] pipeline-compile trigger=%s tid=%lu start_us=%lld us=%lld hit=%d bytes=%zu "
                             "variant=%u result=%d end_ms=%lld key=%s shaders=%s\n", trigger, GetCurrentThreadId(), started,
                             ended - started, hit, bytes, variant, int(result), compile_wall_ms(), key, shaders);
    }
};


// LEONIDA_COMPILE_THREADS: how many threads the PipelineCompiler below runs its jobs on. Unset, empty or not a number:
// half the logical processors, at least 2 and at most 8 (leonida::pipeline_threads). 1 is the one thread there was
// before the pool; 0 counts as 1, and more than 64 as 64.
inline constexpr unsigned kMaxCompileThreads = leonida::PipelineWorkers::kMaxThreads;
inline unsigned compile_threads(const char* text, unsigned logical_processors) {
    const auto parsed = leonida::pipeline_threads(text, logical_processors);
    if (!text || !*text || !parsed.valid) return (std::max)(2u, parsed.count);
    return (std::max)(1u, parsed.count);
}

// LEONIDA_COMPILE_HEAVY_MAX: how many very large builds (PipelineCompiler::Size::huge) its workers are in at a time.
// Unset, empty or not a number: 1. 0: as many as the other rules allow.
inline unsigned compile_heavy_max(const char* text) {
    if (!text || !*text) return 1;
    const auto parsed = leonida::pipeline_threads(text, 0);
    if (!parsed.valid) return 1;
    return parsed.count ? parsed.count : UINT_MAX;
}

// Demanded work has priority zero, speculative work is ordered by module size, and
// an idle checkpoint runs last. No Vulkan call runs with the queue lock held.
//
// Workers. N threads take the jobs (compile_threads above). A worker takes the first queued job, in that order, that
// these rules let start; with one worker that is always the first job, so the order is the one above.
//   - Lane. A job is heavy when it is speculative (a priority other than zero) or was submitted as large or huge. At most
//     N-1 heavy jobs run at a time (one, with one worker), so a worker is always left for a small demanded build: a
//     submission's pipeline is not held behind N modules the driver takes minutes over. A heavy job that waits for
//     memory inside its work (the gate of large_module_gate.h) still counts.
//   - Very large. Of the jobs submitted as huge, one runs at a time (LEONIDA_COMPILE_HEAVY_MAX), demanded or
//     speculative: a driver that compiles several modules it takes half an hour over side by side may fail all of
//     them. They are heavy too. The memory gate does not do this: it lets in what fits the memory that is free.
//   - Owner. One job of an owner at a time. Two builds of one PSO would wait for each other on the PSO's lock, and
//     after_running() hands the PSO's deletion to the one running job it finds: a second one would use a deleted PSO.
//   - Checkpoint (priority SIZE_MAX). Starts only when no worker is in a build. A build may start while it runs:
//     has_compile_waiter() tells it so.
//
// Lifetime. A job names its owner (the PSO it builds, or the device for a cache checkpoint) and the device whose
// Vulkan objects it uses, and holds no reference to either: a reference kept the device alive past the title's last
// Release until the worker reached the job, so a released device's builds ran on (a test that then destroyed its
// VkDevice crashed).
//   - A PSO's last Release does not wait for its running build, which can be a dispatch-worker build of a module the
//     driver takes minutes over: after_running() drops its queued jobs and hands the PSO's deletion to the worker,
//     to run when that build returns (PipelineState::last_release). The PSO lets its device reference go at once.
//   - A device's teardown waits: cancel() drops its queued jobs (by owner or device) and waits for its running one
//     and what runs after it, so an in-flight vkCreate*Pipelines finishes before the device's objects go.
//   - The workers run while a device is alive (retain/release): the last device's release stops and joins them.
//   - stop_for_exit() is for a process that ends with its device alive (main.cpp: the title's exit, a console Ctrl+C
//     or close). After it nothing more is queued: submit() refuses, and the callers build in place or not at all.
class PipelineCompiler {
public:
    static PipelineCompiler& instance() {
        static auto* pool = new PipelineCompiler;  // never destroyed: static destruction must not race the workers
        return *pool;
    }

    // What the build is, by the size of its module: `large` and `huge` builds are heavy (the lane rule), and `huge`
    // ones have a limit of their own.
    enum class Size : unsigned char { regular, large, huge };

    // `threads` workers and `huge_max` very large builds at a time; 0 reads LEONIDA_COMPILE_THREADS and
    // LEONIDA_COMPILE_HEAVY_MAX, once for the process.
    explicit PipelineCompiler(unsigned threads = 0, unsigned huge_max = 0) : threads_(threads), huge_max_(huge_max) {}
    PipelineCompiler(const PipelineCompiler&) = delete;
    PipelineCompiler& operator=(const PipelineCompiler&) = delete;
    // Tests make their own: drops the queue and joins the workers, after the jobs they are in.
    ~PipelineCompiler() {
        std::vector<Job> dropped;
        std::vector<std::thread> workers;
        {
            std::lock_guard lock(mutex_);
            workers = stop_locked(dropped);
        }
        finish(std::move(workers), true);
    }

    struct Stats {
        std::uint64_t submitted = 0;
        std::uint64_t completed = 0;  // jobs the worker ran
        std::uint64_t dropped = 0;    // queued jobs thrown away by cancel, after_running, release or stop_for_exit
        std::uint64_t refused = 0;    // submissions after stop_for_exit
        std::uint64_t waited = 0;     // cancel calls that waited for a running job
        std::uint64_t deferred = 0;   // after_running calls that left something to run after a running job
    };

    // False, with nothing queued, after stop_for_exit. `size`: a demanded build (priority zero) of a module the
    // driver takes long over counts as heavy, as every speculative build does, and a huge one has its own limit too.
    bool submit(const void* owner, const void* device, std::size_t priority, std::function<void()> work,
                Size size = Size::regular) {
        std::lock_guard lock(mutex_);
        if (closed_) {
            ++stats_.refused;
            return false;
        }
        if (workers_.empty()) {
            const std::uint64_t generation = ++generation_;
            const unsigned count = threads_locked();
            workers_.reserve(count);
            // A thread that cannot start leaves fewer workers, and the rules count the ones there are.
            try {
                for (unsigned i = 0; i < count; ++i) workers_.emplace_back([this, generation] { run(generation); });
            } catch (...) {
                if (workers_.empty()) throw;
            }
        }
        const bool checkpoint = priority == SIZE_MAX;
        const Kind kind = checkpoint ? Kind::checkpoint : size != Size::regular || priority ? Kind::heavy : Kind::demand;
        jobs_.emplace(priority, Job{owner, device, std::move(work), kind, !checkpoint && size == Size::huge});
        ++stats_.submitted;
        wake_.notify_one();
        return true;
    }

    bool closed() {
        std::lock_guard lock(mutex_);
        return closed_;
    }

    // True when a build is queued, or running on another worker: what a checkpoint gives way to.
    bool has_compile_waiter() {
        std::lock_guard lock(mutex_);
        if (!jobs_.empty() && jobs_.begin()->first != SIZE_MAX) return true;
        const auto me = std::this_thread::get_id();
        return std::any_of(running_.begin(), running_.end(), [&](const Running& r) {
            return r.kind != Kind::checkpoint && r.thread != me;
        });
    }
    // Returns when no job is queued or running. A job owns its pipeline state, and through it the device, until it
    // finishes; a caller that counts references (rt::d3d12_xs::device_proxy_ref_count) waits here first.
    void wait_idle() {
        std::unique_lock wait(mutex_);
        done_.wait(wait, [this] { return jobs_.empty() && running_.empty(); });
    }

    // Drops the queued jobs whose owner or device is `key` and waits for the running ones, with what runs after them
    // (after_running), unless one is on this thread. Jobs a running one queues for `key` meanwhile are dropped too.
    // Returns how many were dropped.
    std::size_t cancel(const void* key) {
        std::vector<Job> dropped;
        {
            std::unique_lock lock(mutex_);
            const auto me = std::this_thread::get_id();
            bool counted = false;
            for (;;) {
                drop_locked(dropped, [key](const Job& job) { return job.owner == key || job.device == key; });
                const bool busy = std::any_of(running_.begin(), running_.end(), [&](const Running& r) {
                    return (r.owner == key || r.device == key) && r.thread != me;
                });
                if (!busy) break;
                if (!counted) ++stats_.waited;
                counted = true;
                done_.wait(lock);
            }
        }
        return dropped.size();  // the dropped jobs' captures are destroyed here, outside the lock
    }

    // Drops the queued jobs filed under `owner`. When one of its jobs is running (there is at most one: the owner
    // rule), `last` is run on that worker after it returns, before the job counts as done (a device's cancel waits for
    // both), and the answer is true; otherwise false, and `last` is not kept. On the worker inside that job itself,
    // `last` runs after the job returns too.
    bool after_running(const void* owner, std::function<void()> last) {
        std::vector<Job> dropped;
        std::lock_guard lock(mutex_);
        drop_locked(dropped, [owner](const Job& job) { return job.owner == owner; });
        for (Running& r : running_) {
            if (r.owner != owner || r.finished) continue;
            r.after = r.after ? [first = std::move(r.after), then = std::move(last)] { first(); then(); }
                              : std::move(last);
            ++stats_.deferred;
            return true;
        }
        return false;
    }

    // True on the worker while it runs a job whose owner or device is `key` (not what runs after it).
    bool running_here(const void* key) {
        std::lock_guard lock(mutex_);
        const auto me = std::this_thread::get_id();
        return std::any_of(running_.begin(), running_.end(), [&](const Running& r) {
            return r.thread == me && !r.finished && (r.owner == key || r.device == key);
        });
    }

    // A device holds the workers while it lives. The last release drops whatever is still queued, stops the workers
    // and joins them; a release on a worker lets that one return after the job it is in.
    void retain() {
        std::lock_guard lock(mutex_);
        ++users_;
    }
    void release() {
        std::vector<Job> dropped;
        std::vector<std::thread> workers;
        {
            std::lock_guard lock(mutex_);
            if (users_ && --users_) return;
            workers = stop_locked(dropped);
        }
        finish(std::move(workers), true);
    }

    // Process exit with a device still alive: drops the queue, refuses new jobs and waits up to `wait` for the running
    // ones, so ExitProcess does not end a worker inside the driver. False when a job was still running.
    bool stop_for_exit(std::chrono::milliseconds wait) {
        std::vector<Job> dropped;
        std::vector<std::thread> workers;
        bool idle = true;
        {
            std::unique_lock lock(mutex_);
            closed_ = true;
            workers = stop_locked(dropped);
            const auto me = std::this_thread::get_id();
            idle = done_.wait_for(lock, wait, [&] {
                return std::none_of(running_.begin(), running_.end(), [&](const Running& r) { return r.thread != me; });
            });
        }
        finish(std::move(workers), idle);
        return idle;
    }

    Stats stats() {
        std::lock_guard lock(mutex_);
        return stats_;
    }
    bool worker_running() {
        std::lock_guard lock(mutex_);
        return !workers_.empty();
    }
    // How many workers a start makes (LEONIDA_COMPILE_THREADS, or the constructor's number).
    unsigned threads() {
        std::lock_guard lock(mutex_);
        return threads_locked();
    }
    // Tests: called on the worker with each job's owner, once the job counts as running and before its work.
    void set_job_probe(std::function<void(const void* owner)> probe) {
        std::lock_guard lock(mutex_);
        probe_ = std::move(probe);
    }

private:
    enum class Kind : unsigned char { demand, heavy, checkpoint };
    struct Job {
        const void* owner = nullptr;
        const void* device = nullptr;
        std::function<void()> work;
        Kind kind = Kind::demand;
        bool huge = false;
    };
    struct Running {
        const void* owner;
        const void* device;
        std::thread::id thread;
        Kind kind;
        bool huge;
        std::function<void()> after;  // after_running
        bool finished = false;        // the job returned; `after` may be running
    };

    template<class Match> void drop_locked(std::vector<Job>& dropped, const Match& match) {
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            if (match(it->second)) {
                dropped.push_back(std::move(it->second));
                it = jobs_.erase(it);
                ++stats_.dropped;
            } else {
                ++it;
            }
        }
    }

    unsigned threads_locked() {
        if (!threads_) {
            static const unsigned from_environment = [] {
                char text[32]{};
                const DWORD n = GetEnvironmentVariableA("LEONIDA_COMPILE_THREADS", text, sizeof(text));
                const bool present = n > 0 && n < sizeof(text);
                const unsigned count = compile_threads(present ? text : "", std::thread::hardware_concurrency());
                if (present && !leonida::pipeline_threads(text, 0).valid)
                    std::fprintf(stderr, "[rt] d3d12: LEONIDA_COMPILE_THREADS=%s is not a number; using %u\n", text,
                                 count);
                return count;
            }();
            threads_ = from_environment;
        }
        return threads_;
    }

    unsigned huge_max_locked() {
        if (!huge_max_) {
            static const unsigned from_environment = [] {
                char text[32]{};
                const DWORD n = GetEnvironmentVariableA("LEONIDA_COMPILE_HEAVY_MAX", text, sizeof(text));
                const bool present = n > 0 && n < sizeof(text);
                if (present && !leonida::pipeline_threads(text, 0).valid)
                    std::fprintf(stderr, "[rt] d3d12: LEONIDA_COMPILE_HEAVY_MAX=%s is not a number; using 1\n", text);
                return compile_heavy_max(present ? text : "");
            }();
            huge_max_ = from_environment;
        }
        return huge_max_;
    }

    std::vector<std::thread> stop_locked(std::vector<Job>& dropped) {
        drop_locked(dropped, [](const Job&) { return true; });
        ++generation_;  // a worker returns when it next looks: after the job it is running, if any
        wake_.notify_all();
        return std::exchange(workers_, {});
    }

    static void finish(std::vector<std::thread> workers, bool join) {
        for (std::thread& worker : workers) {
            if (!worker.joinable()) continue;
            if (join && worker.get_id() != std::this_thread::get_id()) worker.join();
            else worker.detach();
        }
    }

    // The first queued job the rules let a worker start now (the class comment), or the end.
    std::multimap<std::size_t, Job>::iterator next_locked() {
        std::size_t heavy = 0, huge = 0;
        bool compiling = false;
        for (const Running& r : running_) {  // a job has its place until it counts as done (after_running included)
            heavy += r.kind == Kind::heavy;
            huge += r.huge;
            compiling |= r.kind != Kind::checkpoint;
        }
        // A stopped worker may still be in a job: its job counts, and it is not among workers_.
        const std::size_t lane = workers_.size() > 1 ? workers_.size() - 1 : 1;
        const bool lane_full = heavy >= lane;
        const bool huge_full = huge >= huge_max_locked();
        for (auto it = jobs_.begin(); it != jobs_.end();) {
            const Job& job = it->second;
            const bool blocked = job.kind == Kind::heavy ? lane_full || (job.huge && huge_full)
                                                         : job.kind == Kind::checkpoint && compiling;
            const bool owner_busy = std::any_of(running_.begin(), running_.end(), [&](const Running& r) {
                return r.owner == job.owner;
            });
            if (!blocked && !owner_busy) return it;
            // Every job with a priority between zero and the checkpoint's is heavy: none of them can start either.
            if (job.kind == Kind::heavy && lane_full && it->first) it = jobs_.lower_bound(SIZE_MAX);
            else ++it;
        }
        return jobs_.end();
    }

    Running& mine(std::thread::id me) {
        return *std::find_if(running_.begin(), running_.end(), [me](const Running& r) { return r.thread == me; });
    }

    void run(std::uint64_t generation) {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
        const auto me = std::this_thread::get_id();
        std::unique_lock lock(mutex_);
        for (;;) {
            auto first = jobs_.end();
            wake_.wait(lock, [&] { return generation_ != generation || (first = next_locked()) != jobs_.end(); });
            if (generation_ != generation) return;
            Job job = std::move(first->second);
            jobs_.erase(first);
            running_.push_back({job.owner, job.device, me, job.kind, job.huge});
            const auto probe = probe_;
            lock.unlock();
            try {
                if (probe) probe(job.owner);
                job.work();
            } catch (...) {
                std::fprintf(stderr, "[rt] pipeline compile worker: a job threw\n");
            }
            job.work = nullptr;  // its captures go before the job counts as done
            lock.lock();
            ++stats_.completed;
            Running& self = mine(me);
            self.finished = true;
            if (std::function<void()> after = std::move(self.after)) {
                lock.unlock();
                try {
                    after();
                } catch (...) {
                    std::fprintf(stderr, "[rt] pipeline compile worker: work after a job threw\n");
                }
                after = nullptr;
                lock.lock();
            }
            running_.erase(std::find_if(running_.begin(), running_.end(), [me](const Running& r) {
                return r.thread == me;
            }));
            done_.notify_all();
            wake_.notify_all();  // a job of this owner, a heavy job or a checkpoint may start now
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::multimap<std::size_t, Job> jobs_;
    std::vector<Running> running_;  // one per worker; a stopped worker may still be finishing one
    std::vector<std::thread> workers_;
    unsigned threads_ = 0;          // how many workers a start makes; 0 until LEONIDA_COMPILE_THREADS is read
    unsigned huge_max_ = 0;         // how many huge jobs run at a time; 0 until LEONIDA_COMPILE_HEAVY_MAX is read
    std::uint64_t generation_ = 0;  // a worker runs while this is the value it started with
    unsigned users_ = 0;            // live devices (retain/release)
    bool closed_ = false;           // stop_for_exit: nothing more is queued
    Stats stats_;
    std::function<void(const void*)> probe_;
};
}  // namespace rt::d3d12
