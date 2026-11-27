#pragma once
// Where a submission's CPU time goes, always on and cheap: each phase of VulkanQueue::execute_now adds its time here
// (two steady_clock reads and two relaxed atomic adds), and every LEONIDA_SUBMIT_PHASES_SECONDS (5 by default, 0 turns
// the line off) one line says, since the last line, how many submissions there were, how many per second, the
// milliseconds per submission of each phase with the largest single one, and the share of the wall time the
// submitting threads spent in them. The time is the process's age (GetProcessTimes), so it lines up with the boot.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace leonida::submit_phases {

enum Phase : unsigned {
    wait_room,  // VulkanQueue::wait_for_room: the queue at its cap waits for the GPU
    heaps,      // DeviceState::refresh_console_heaps
    shadows,    // slot-133 shadow buffers and title windows (ShadowPages::refresh)
    geometry,   // GeometryViews::refresh
    textures,   // Device::refresh_title_textures
    pipelines,  // no longer timed (the pre-pass that waited for pipelines is gone); kept for the column order
    replay,     // the rest of execute_now: preparing, recording and submitting the lists
    kCount
};

inline const char* name(unsigned p) {
    static const char* const names[kCount] = {"wait", "heaps", "shadows", "geometry", "textures", "pipes",
                                                    "replay"};
    return p < kCount ? names[p] : "?";
}

struct Totals {
    std::atomic<std::uint64_t> ns[kCount]{};
    std::atomic<std::uint64_t> max_ns[kCount]{};
    std::atomic<std::uint64_t> submissions{0}, all_submissions{0};
    std::atomic<std::int64_t> last_log{0};
    std::atomic<std::uint64_t> presents{0};  // frames the presenter showed (VulkanPresenter), since the last line
    std::atomic<std::uint64_t> present_serial{0};  // the same, never reset (LEONIDA_REFRESH_PER_FRAME)
    // Threads whose innermost Scope is this phase now, and when the first of them entered it (steady_clock). Counted
    // per phase, since queues submit from several threads: one value would go stale when their scopes overlap.
    std::atomic<unsigned> active[kCount]{};
    std::atomic<std::int64_t> active_since[kCount]{};
    std::atomic<std::uint32_t> thread{0};  // the thread of the last submission (the thread sampler reports it apart)
};
inline Totals& totals() {
    static Totals t;
    return t;
}

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Seconds since the process started.
inline double process_age() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{}, now{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    GetSystemTimeAsFileTime(&now);
    const auto ticks = [](const FILETIME& f) { return (std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    return double(ticks(now) - ticks(created)) * 1e-7;
#else
    return 0;
#endif
}

inline std::int64_t period_ns() {
    static const std::int64_t value = [] {
        const char* text = std::getenv("LEONIDA_SUBMIT_PHASES_SECONDS");
        const unsigned long long s = text && *text ? std::strtoull(text, nullptr, 10) : 5;
        return std::int64_t(s) * 1000000000;
    }();
    return value;
}

// ---- The per-frame line ("submit frame") and the floor ("submit floor"), every LEONIDA_SUBMIT_FRAME_SECONDS --------
//
// The phases above, summed per presented frame instead of per submission, and split by who spent them: the title's
// threads (whoever calls into the layer) or our own (a WorkerScope marks a stretch of one of our threads). Added to
// them, what the phase line leaves out, from title calls timed where they enter the layer (TitleCall, the outermost
// one on a thread): the time of an ExecuteCommandLists outside the phases (list unwrapping, the console's list-value
// waits and writes, the queue lock), of the queue's Signal and Wait, and of the present. Inside those, two parts are
// also counted on their own: the vkQueueSubmit calls and the waits for a contended lock (lock_wait), which lie inside
// the phases or the calls above and are not added again. A frame ends at each present the presenter takes.
//
// The floor: the submissions in which nothing changed - no console heap slot or page copied, no shadow page copied,
// no geometry uploaded, no texture filled and no pipeline built (note_change, counted process-wide, so a change made
// by another thread during a submission takes it out of the floor too) - with their phases, average and least.
enum Part : unsigned {
    part_lists,        // ExecuteCommandLists outside the phases
    part_fences,       // queue Signal and Wait, fence Signal from the CPU
    part_present,      // the present (with its wait for a swapchain image)
    part_submit_call,  // inside the above: vkQueueSubmit
    part_lock_wait,    // inside the above: waiting for a contended lock of the submission path
    kParts
};
inline constexpr unsigned kOuterParts = part_present + 1;  // the parts that add to the title threads' total
enum Worker : unsigned {
    worker_submit,        // LEONIDA_SUBMIT_ASYNC: the queue's worker (the replay leaves the title's thread)
    worker_texture_reads, // the title texture refresh's read threads
    worker_texture_fill,  // the slot-70 texture filler
    worker_heap_resolver, // the console heap resolver (LEONIDA_CONSOLE_HEAP_GPU)
    worker_geometry,      // LEONIDA_GEOMETRY_ASYNC: the geometry refresher
    worker_value_pump,    // the console list values' pump and releaser
    worker_fence_thread,  // the fence dispatcher's callbacks
    kWorkers
};
enum Change : unsigned { change_slots, change_pages, change_geometry, change_textures, change_pipelines, kChanges };

inline const char* part_name(unsigned p) {
    static const char* const names[kParts] = {"lists", "fences", "present", "vkQueueSubmit", "lock waits"};
    return p < kParts ? names[p] : "?";
}
inline const char* worker_name(unsigned w) {
    static const char* const names[kWorkers] = {"submit",   "texture-reads", "texture-fill", "heap-resolver",
                                                "geometry", "value-pump",    "fence-thread"};
    return w < kWorkers ? names[w] : "?";
}

struct FrameTotals {
    std::atomic<std::uint64_t> phase_ns[kCount]{};  // title threads
    std::atomic<std::uint64_t> part_ns[kParts]{};   // title threads
    std::atomic<std::uint64_t> worker_ns[kWorkers]{};
    std::atomic<std::uint64_t> submissions[3]{};    // by queue kind: direct, compute, copy
    std::atomic<std::uint64_t> frames{0};
    std::atomic<std::uint64_t> current_ns{0};       // the title threads' time since the last present
    std::atomic<std::uint64_t> worst_ns{0};         // the most of a frame in this window
    std::atomic<std::int64_t> last_log{0};
    std::atomic<std::uint64_t> changes[kChanges]{};  // never reset: a submission compares them before and after
    // The floor: submissions with no change, their phases and their least total; and every submission's total.
    std::atomic<std::uint64_t> floor_count{0}, floor_ns[kCount]{}, floor_min_ns{UINT64_MAX};
    std::atomic<std::uint64_t> all_count{0}, all_ns{0};
};
inline FrameTotals& frame_totals() {
    static FrameTotals t;
    return t;
}

// What one thread is doing, for the split above. Trivially destructible (no TLS callback in gta6rt.exe).
struct ThreadState {
    int worker = -1;                  // a WorkerScope's worker, -1 on a title thread
    unsigned calls = 0;               // TitleCalls open on this thread
    unsigned submissions = 0;         // execute_now calls open on this thread (begin_submission)
    std::uint64_t phase_ns = 0;       // phases this thread added since it started, for TitleCall
    std::uint64_t submission_ns[kCount] = {};  // the outermost open submission's phases
    std::uint64_t changes_at = 0;     // the change count when it began
};
inline ThreadState& thread_state() {
    thread_local ThreadState state;
    return state;
}

inline std::uint64_t change_count() {
    std::uint64_t n = 0;
    for (const auto& c : frame_totals().changes) n += c.load(std::memory_order_relaxed);
    return n;
}
// Something a submission had to bring up to date: the submissions running now leave the floor.
inline void note_change(Change c) { frame_totals().changes[c].fetch_add(1, std::memory_order_relaxed); }

// Time a title thread spent in the layer outside the phases (an outer part), or inside them (an inner part).
inline void add_part(Part p, std::uint64_t ns) {
    ThreadState& s = thread_state();
    if (s.worker >= 0) return;  // ours: a worker's time is its WorkerScope's
    FrameTotals& f = frame_totals();
    f.part_ns[p].fetch_add(ns, std::memory_order_relaxed);
    if (p < kOuterParts) f.current_ns.fetch_add(ns, std::memory_order_relaxed);
}

// The title's call into the layer this thread is in, timed when it is the outermost: its time less the phases inside
// it is `part`'s.
struct TitleCall {
    Part part;
    std::int64_t start = 0;
    std::uint64_t phases_at = 0;
    explicit TitleCall(Part p) : part(p) {
        ThreadState& s = thread_state();
        if (s.calls++ || s.worker >= 0) return;
        start = now_ns();
        phases_at = s.phase_ns;
    }
    ~TitleCall() {
        ThreadState& s = thread_state();
        --s.calls;
        if (!start) return;
        const std::uint64_t all = std::uint64_t(now_ns() - start), inside = s.phase_ns - phases_at;
        add_part(part, all > inside ? all - inside : 0);
    }
    TitleCall(const TitleCall&) = delete;
    TitleCall& operator=(const TitleCall&) = delete;
};

// A stretch of one of our own threads: its time is `worker`'s, and the phases inside it are not the title's.
struct WorkerScope {
    Worker worker;
    int outer;
    std::int64_t start = now_ns();
    explicit WorkerScope(Worker w) : worker(w), outer(thread_state().worker) { thread_state().worker = int(w); }
    ~WorkerScope() {
        thread_state().worker = outer;
        if (outer < 0) frame_totals().worker_ns[worker].fetch_add(std::uint64_t(now_ns() - start), std::memory_order_relaxed);
    }
    WorkerScope(const WorkerScope&) = delete;
    WorkerScope& operator=(const WorkerScope&) = delete;
};

// Locks `lock`, and counts the wait as part_lock_wait when it was held by someone else.
template<class Lock>
inline void lock_timed(Lock& lock) {
    if (lock.try_lock()) return;
    const std::int64_t start = now_ns();
    lock.lock();
    add_part(part_lock_wait, std::uint64_t(now_ns() - start));
}

inline void frame_add_phase(Phase p, std::uint64_t ns) {
    ThreadState& s = thread_state();
    s.phase_ns += ns;
    if (s.submissions) s.submission_ns[p] += ns;
    if (s.worker >= 0) return;
    FrameTotals& f = frame_totals();
    f.phase_ns[p].fetch_add(ns, std::memory_order_relaxed);
    f.current_ns.fetch_add(ns, std::memory_order_relaxed);
}

// At each present the presenter takes: the frame's title-thread time so far, and the window's worst.
inline void frame_presented() {
    FrameTotals& f = frame_totals();
    f.frames.fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t ns = f.current_ns.exchange(0, std::memory_order_relaxed);
    std::uint64_t worst = f.worst_ns.load(std::memory_order_relaxed);
    while (ns > worst && !f.worst_ns.compare_exchange_weak(worst, ns, std::memory_order_relaxed)) {
    }
}

// At the start of VulkanQueue::execute_now, before its first phase: the floor's per-submission sums begin.
inline void begin_submission() {
    ThreadState& s = thread_state();
    if (s.submissions++) return;
    for (auto& ns : s.submission_ns) ns = 0;
    s.changes_at = change_count();
}

inline void add(Phase p, std::uint64_t ns) {
    Totals& t = totals();
    t.ns[p].fetch_add(ns, std::memory_order_relaxed);
    std::uint64_t m = t.max_ns[p].load(std::memory_order_relaxed);
    while (ns > m && !t.max_ns[p].compare_exchange_weak(m, ns, std::memory_order_relaxed)) {
    }
    frame_add_phase(p, ns);
}

// Times its own lifetime into a phase.
struct Scope {
    Phase phase;
    std::int64_t start = now_ns();
    int outer;  // the phase of this thread's Scope around this one, -1 for none
    explicit Scope(Phase p) : phase(p), outer(innermost()) {
        if (outer >= 0) leave(outer);
        enter(int(p), start);
        innermost() = int(p);
    }
    ~Scope() {
        const std::int64_t end = now_ns();
        add(phase, std::uint64_t(end - start));
        leave(int(phase));
        if (outer >= 0) enter(outer, end);
        innermost() = outer;
    }
    static int& innermost() {
        thread_local int phase_of_this_thread = -1;
        return phase_of_this_thread;
    }
    static void enter(int p, std::int64_t at) {
        Totals& t = totals();
        if (t.active[p].fetch_add(1, std::memory_order_relaxed) == 0) t.active_since[p].store(at, std::memory_order_relaxed);
    }
    static void leave(int p) { totals().active[p].fetch_sub(1, std::memory_order_relaxed); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// The phase that some thread has been in longest, -1 when none is in one.
struct Running {
    int phase = -1;
    std::int64_t since = 0;
};
inline Running running() {
    Totals& t = totals();
    Running r;
    for (unsigned p = 0; p < kCount; ++p) {
        if (!t.active[p].load(std::memory_order_relaxed)) continue;
        const std::int64_t since = t.active_since[p].load(std::memory_order_relaxed);
        if (r.phase < 0 || since < r.since) r = {int(p), since};
    }
    return r;
}

inline std::int64_t frame_period_ns() {
    static const std::int64_t value = [] {
        const char* text = std::getenv("LEONIDA_SUBMIT_FRAME_SECONDS");
        const unsigned long long s = text && *text ? std::strtoull(text, nullptr, 10) : 10;
        return std::int64_t(s) * 1000000000;
    }();
    return value;
}

// The two lines over the window since the last ones, and a new window. `seconds`: the window's length.
inline void frame_lines(double seconds) {
    FrameTotals& f = frame_totals();
    const std::uint64_t frames = f.frames.exchange(0, std::memory_order_relaxed);
    const double per = frames ? double(frames) : 1.0;  // per frame, or the window's totals when none was presented
    const char* unit = frames ? "per frame" : "in the window (no frame presented)";
    std::uint64_t subs[3];
    for (unsigned k = 0; k < 3; ++k) subs[k] = f.submissions[k].exchange(0, std::memory_order_relaxed);
    double phase[kCount], part[kParts], worker[kWorkers], title = 0, own = 0;
    for (unsigned p = 0; p < kCount; ++p) title += phase[p] = double(f.phase_ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6 / per;
    for (unsigned p = 0; p < kParts; ++p) {
        part[p] = double(f.part_ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6 / per;
        if (p < kOuterParts) title += part[p];
    }
    for (unsigned w = 0; w < kWorkers; ++w) own += worker[w] = double(f.worker_ns[w].exchange(0, std::memory_order_relaxed)) * 1e-6 / per;
    const double worst = double(f.worst_ns.exchange(0, std::memory_order_relaxed)) * 1e-6;
    std::fprintf(stderr,
                 "[rt] submit frame t=%.0fs: %llu frames in %.1f s; %s: %.1f submissions (direct %.1f, compute %.1f, "
                 "copy %.1f); title threads %.2f ms: %s %.2f %s %.2f %s %.2f %s %.2f %s %.2f %s %.2f %s %.2f, %s %.2f, "
                 "%s %.2f, %s %.2f (inside these: %s %.2f, %s %.2f); worst frame %.1f ms; own threads %.2f ms: %s %.2f "
                 "%s %.2f %s %.2f %s %.2f %s %.2f %s %.2f %s %.2f\n",
                 process_age(), static_cast<unsigned long long>(frames), seconds, unit,
                 double(subs[0] + subs[1] + subs[2]) / per, double(subs[0]) / per, double(subs[1]) / per,
                 double(subs[2]) / per, title, name(0), phase[0], name(1), phase[1], name(2), phase[2], name(3),
                 phase[3], name(4), phase[4], name(5), phase[5], name(6), phase[6], part_name(0), part[0],
                 part_name(1), part[1], part_name(2), part[2], part_name(3), part[3], part_name(4), part[4], worst, own,
                 worker_name(0), worker[0], worker_name(1), worker[1], worker_name(2), worker[2], worker_name(3),
                 worker[3], worker_name(4), worker[4], worker_name(5), worker[5], worker_name(6), worker[6]);
    const std::uint64_t floor = f.floor_count.exchange(0, std::memory_order_relaxed);
    const std::uint64_t all = f.all_count.exchange(0, std::memory_order_relaxed);
    const double all_ms = double(f.all_ns.exchange(0, std::memory_order_relaxed)) * 1e-6;
    const std::uint64_t least = f.floor_min_ns.exchange(UINT64_MAX, std::memory_order_relaxed);
    double floor_phase[kCount], floor_ms = 0;
    for (unsigned p = 0; p < kCount; ++p)
        floor_ms += floor_phase[p] =
            floor ? double(f.floor_ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6 / double(floor) : 0;
    std::fprintf(stderr,
                 "[rt] submit floor t=%.0fs: %llu of %llu submissions changed nothing (no heap slot, shadow page, "
                 "geometry, texture or pipeline), %.3f ms each on average (all submissions %.3f), %.3f the least: "
                 "%s %.3f %s %.3f %s %.3f %s %.3f %s %.3f %s %.3f %s %.3f\n",
                 process_age(), static_cast<unsigned long long>(floor), static_cast<unsigned long long>(all), floor_ms,
                 all ? all_ms / double(all) : 0.0, floor ? double(least) * 1e-6 : 0.0, name(0), floor_phase[0],
                 name(1), floor_phase[1], name(2), floor_phase[2], name(3), floor_phase[3], name(4), floor_phase[4],
                 name(5), floor_phase[5], name(6), floor_phase[6]);
}

// The end of an execute_now: the submission is counted for the frame line (by queue kind: 0 direct, 1 compute,
// 2 copy) and, when nothing changed while it ran, for the floor; the lines are printed when the period has passed.
inline void frame_submitted(int kind) {
    FrameTotals& f = frame_totals();
    f.submissions[kind < 0 || kind > 2 ? 0 : kind].fetch_add(1, std::memory_order_relaxed);
    ThreadState& s = thread_state();
    if (s.submissions && !--s.submissions) {
        std::uint64_t ns = 0;
        for (const std::uint64_t n : s.submission_ns) ns += n;
        f.all_count.fetch_add(1, std::memory_order_relaxed);
        f.all_ns.fetch_add(ns, std::memory_order_relaxed);
        if (change_count() == s.changes_at) {
            f.floor_count.fetch_add(1, std::memory_order_relaxed);
            for (unsigned p = 0; p < kCount; ++p) f.floor_ns[p].fetch_add(s.submission_ns[p], std::memory_order_relaxed);
            std::uint64_t least = f.floor_min_ns.load(std::memory_order_relaxed);
            while (ns < least && !f.floor_min_ns.compare_exchange_weak(least, ns, std::memory_order_relaxed)) {
            }
        }
    }
    const std::int64_t period = frame_period_ns();
    if (!period) return;
    const std::int64_t now = now_ns();
    std::int64_t last = f.last_log.load(std::memory_order_relaxed);
    if (!last) {
        f.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < period || !f.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    frame_lines(double(now - last) * 1e-9);
}

// The whole of one VulkanQueue::execute_now, from before its first phase to after its last.
struct Submission {
    int kind;
    explicit Submission(int k) : kind(k) { begin_submission(); }
    ~Submission() { frame_submitted(kind); }
    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;
};

// After each submission: counts it, and prints the line when the period has passed. One thread wins the line.
inline void submitted() {
    Totals& t = totals();
    t.submissions.fetch_add(1, std::memory_order_relaxed);
    t.all_submissions.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    t.thread.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    const std::int64_t period = period_ns();
    if (!period) return;
    const std::int64_t now = now_ns();
    std::int64_t last = t.last_log.load(std::memory_order_relaxed);
    if (!last) {
        t.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed);
        return;
    }
    if (now - last < period || !t.last_log.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
    const double seconds = double(now - last) * 1e-9;
    const std::uint64_t n = t.submissions.exchange(0, std::memory_order_relaxed);
    const std::uint64_t frames = t.presents.exchange(0, std::memory_order_relaxed);
    double ms[kCount], max_ms[kCount], all = 0;
    for (unsigned p = 0; p < kCount; ++p) {
        const double total = double(t.ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6;
        max_ms[p] = double(t.max_ns[p].exchange(0, std::memory_order_relaxed)) * 1e-6;
        ms[p] = n ? total / double(n) : 0;
        all += total;
    }
    std::fprintf(stderr,
                 "[rt] submit phases t=%.0fs: %llu submissions in %.1f s (%.2f/s, %llu in all), %.2f frames/s; ms per "
                 "submission (max): %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f (%.0f) %s %.2f "
                 "(%.0f) %s %.2f (%.0f); "
                 "%.0f%% of the wall time\n",
                 process_age(), static_cast<unsigned long long>(n), seconds, double(n) / seconds,
                 static_cast<unsigned long long>(t.all_submissions.load(std::memory_order_relaxed)),
                 double(frames) / seconds, name(0), ms[0],
                 max_ms[0], name(1), ms[1], max_ms[1], name(2), ms[2], max_ms[2], name(3), ms[3], max_ms[3], name(4),
                 ms[4], max_ms[4], name(5), ms[5], max_ms[5], name(6), ms[6], max_ms[6], 100.0 * all * 1e-3 / seconds);
}

}  // namespace leonida::submit_phases
