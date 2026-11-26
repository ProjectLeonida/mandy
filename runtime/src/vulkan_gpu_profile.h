#pragma once
// LEONIDA_GPU_PROFILE=1 (diagnostic, off by default): GPU time per pipeline, from timestamp queries.
//
// Each submission of a queue whose family writes timestamps gets a query pool, reset once at the start of its command
// buffer. A bottom-of-pipe timestamp goes after the submission's opening barrier and bindless upload, one before each
// run of draws or dispatches of one pipeline (consecutive commands of one pipeline share it), one after the run when
// any other command follows (a barrier, copy, clear, query or one of the layer's own passes), and one at the end. A
// bottom-of-pipe timestamp is written once everything recorded before it has finished, so on a GPU that runs a queue's
// work in order (a desktop GPU without tiling) the difference between two consecutive timestamps is the GPU time by
// which what lies between them extended the queue's work: a run's own time, or "other" when no draw or dispatch lies
// between. Work that overlaps the previous run (no barrier between) counts only for its tail; the layer's helper passes
// for a draw (index expansion, indirect clamp) count for the draw. The results are read when the submission is known
// done (the fence callback), without waiting, and summed by pipeline: by the ShaderLibrary names of its stages, as the
// pass trace and the device-fault report name them, else by the PSO's id.
//
// Every 10 s, on the thread that reads the results, `[rt] gpu profile` lines give the window's totals per frame (by
// presents) for compute, graphics and mesh pipelines and for other work, per queue kind the busy time measured as
// LEONIDA_GPU_LATENCY does (from the later of submit and the previous completion to completion, on the CPU) against
// the time between the first and the last timestamp of its submissions, and the 20 pipelines with the most GPU time.
// Queues run at the same time, so their sums can exceed the frame time. Cost when off: one relaxed load and branch
// per submission and one null test per command; when on: a query reset per submission, one timestamp per run
// boundary, a name lookup per run, and the result read on the fence thread.
#include "submit_phases.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace leonida::gpu_profile {

inline std::atomic<bool>& switch_state() {
    static std::atomic<bool> on{[] {
        const char* v = std::getenv("LEONIDA_GPU_PROFILE");
        return v && std::strcmp(v, "1") == 0;
    }()};
    return on;
}
inline bool enabled() { return switch_state().load(std::memory_order_relaxed); }
// Tests only: before the context is made (it loads the query functions only when the profile is on).
inline void set_enabled_for_test(bool on) { switch_state().store(on, std::memory_order_relaxed); }

enum class Kind : std::uint8_t { compute, graphics, mesh };
inline const char* kind_name(Kind kind) {
    return kind == Kind::compute ? "compute" : kind == Kind::graphics ? "graphics" : "mesh";
}

// The ShaderLibrary names of a pipeline's stages, as CheckpointShaders: [0] the compute, vertex or mesh shader, [1] the
// geometry shader, [2] the pixel shader. Immutable strings, valid until the device goes.
using Names = std::array<const char*, 3>;
using NameMap = std::unordered_map<std::uintptr_t, Names>;

// What the profile sums by: the pipeline's kind and stage names, or its id when it has no names.
struct Key {
    Kind kind = Kind::compute;
    Names names{};
    std::uintptr_t id = 0;
    bool operator==(const Key&) const = default;
};
struct KeyHash {
    std::size_t operator()(const Key& k) const noexcept {
        std::size_t h = std::hash<std::uintptr_t>{}(k.id) ^ (std::size_t(k.kind) * 0x9e3779b97f4a7c15ull);
        for (const char* n : k.names) h = (h ^ std::hash<const void*>{}(n)) * 0x100000001b3ull;
        return h;
    }
};

// The name a pipeline goes by in the lines: its stages' entry names without their folder, as the pass trace prints
// them, or "pso 0x..." for one registered without names.
inline std::string label(const Key& key) {
    std::string text;
    static const char* const graphics_stages[3] = {"vs", "gs", "ps"};
    for (std::size_t i = 0; i < key.names.size(); ++i) {
        const char* name = key.names[i];
        if (!name) continue;
        if (const char* slash = std::strrchr(name, '/')) name = slash + 1;
        if (!text.empty()) text += ' ';
        text += key.kind == Kind::compute ? "cs" : i == 0 && key.kind == Kind::mesh ? "ms" : graphics_stages[i];
        text += ' ';
        text += name;
    }
    if (text.empty()) {
        char id[48];
        std::snprintf(id, sizeof id, "pso 0x%llx", static_cast<unsigned long long>(key.id));
        text = id;
    }
    return text;
}

class Profiler;

// One submission's timestamps, recorded by the replay (ReplayFunctions::gpu_profile) on the submitting thread.
class Recording {
public:
    Recording(const Recording&) = delete;
    Recording& operator=(const Recording&) = delete;
    inline ~Recording();

    // A draw or dispatch of pipeline `id` comes next. A run of the same pipeline goes on; otherwise a timestamp closes
    // the interval since the last one (the open run's, or other work's) and a run of this pipeline starts.
    void work(VkCommandBuffer cb, Kind kind, std::uintptr_t id) {
        if (open_ && id == open_id_ && kind == open_kind_) {
            ++runs_.back().calls;
            return;
        }
        if (!stamp(cb, open_ ? std::int32_t(runs_.size() - 1) : -1)) {
            open_ = false;
            ++dropped_;
            return;
        }
        open_ = true;
        open_id_ = id;
        open_kind_ = kind;
        runs_.push_back({key(kind, id), 1, 0});
    }
    // Any other command comes next: the open run ends here.
    void other(VkCommandBuffer cb) {
        if (!open_) return;
        stamp(cb, std::int32_t(runs_.size() - 1));
        open_ = false;
    }
    // After the last list: the last interval.
    void finish(VkCommandBuffer cb) {
        stamp(cb, open_ ? std::int32_t(runs_.size() - 1) : -1);
        open_ = false;
    }
    void mark_submitted() { submitted_ = std::chrono::steady_clock::now(); }

private:
    friend class Profiler;
    struct Run {
        Key key;
        std::uint32_t calls;
        std::uint64_t ticks;
    };
    Recording() = default;
    bool stamp(VkCommandBuffer cb, std::int32_t owner) {
        if (used_ >= capacity_) return false;
        write_(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool_, used_);
        if (used_) owners_.push_back(owner);
        ++used_;
        return true;
    }
    Key key(Kind kind, std::uintptr_t id) const {
        Key k;
        k.kind = kind;
        if (names_) {
            const auto it = names_->find(id);
            if (it != names_->end()) k.names = it->second;
        }
        if (!k.names[0] && !k.names[1] && !k.names[2]) k.id = id;
        return k;
    }

    Profiler* owner_ = nullptr;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    std::uint32_t capacity_ = 0, used_ = 0;
    PFN_vkCmdWriteTimestamp write_ = nullptr;
    const NameMap* names_ = nullptr;  // read under the context's pipeline lock, which the replay holds
    std::vector<std::int32_t> owners_;  // per interval between timestamps k and k+1: a run, or -1 for other work
    std::vector<Run> runs_;
    bool open_ = false;
    Kind open_kind_ = Kind::compute;
    std::uintptr_t open_id_ = 0;
    int queue_ = 0;  // VulkanQueue::QueueKind: 0 direct, 1 compute, 2 copy
    std::uint64_t mask_ = ~0ull;
    double period_ = 0;  // ns per tick
    std::uint64_t dropped_ = 0;
    std::chrono::steady_clock::time_point submitted_{};
};

class Profiler {
public:
    static constexpr unsigned kTop = 20;
    static constexpr auto kWindow = std::chrono::seconds(10);

    Profiler() = default;
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;
    ~Profiler() { stop(); }

    // Loads the query functions. False (and the profile stays off for this device) when one is missing.
    bool configure(VkDevice device, PFN_vkGetDeviceProcAddr get) {
        if (!enabled() || !device || !get) return false;
        std::lock_guard lock(mutex_);
#define LEONIDA_PROFILE_FN(name) reinterpret_cast<PFN_##name>(get(device, #name))
        create_ = LEONIDA_PROFILE_FN(vkCreateQueryPool);
        destroy_ = LEONIDA_PROFILE_FN(vkDestroyQueryPool);
        reset_ = LEONIDA_PROFILE_FN(vkCmdResetQueryPool);
        write_ = LEONIDA_PROFILE_FN(vkCmdWriteTimestamp);
        results_ = LEONIDA_PROFILE_FN(vkGetQueryPoolResults);
#undef LEONIDA_PROFILE_FN
        if (!create_ || !destroy_ || !reset_ || !write_ || !results_) return false;
        device_ = device;
        stopped_ = false;
        since_ = std::chrono::steady_clock::now();
        frames_at_ = submit_phases::totals().present_serial.load(std::memory_order_relaxed);
        std::fprintf(stderr, "[rt] gpu profile: on (LEONIDA_GPU_PROFILE=1): the GPU time of every draw and dispatch, "
                             "summed by pipeline from timestamps, every 10 s\n");
        return true;
    }

    // A recording for the command buffer just begun, with its query pool reset and its first timestamp written, or
    // null: the profile is off, the queue's family writes no timestamps (`bits` 0) or no pool could be made.
    // `commands`: the commands of its lists, which bounds the timestamps (at most one per command, and two more).
    std::unique_ptr<Recording> begin(VkCommandBuffer cb, int queue, double period_ns, std::uint32_t bits,
                                     std::size_t commands, const NameMap* names) {
        if (!enabled() || !cb) return nullptr;
        if (!bits || period_ns <= 0) {
            if (!no_timestamps_.exchange(true))
                std::fprintf(stderr, "[rt] gpu profile: a %s queue's family has no timestamps the profile can reset "
                                     "and read; the submissions of such queues are not profiled\n", queue_name(queue));
            return nullptr;
        }
        const std::uint64_t wanted = std::uint64_t(commands) + 2;
        if (wanted > (1u << 24)) return nullptr;
        std::uint32_t capacity = 256;
        while (capacity < wanted) capacity *= 2;
        VkQueryPool pool = take(capacity);
        if (!pool) return nullptr;
        std::unique_ptr<Recording> r(new Recording());
        r->owner_ = this;
        r->pool_ = pool;
        r->capacity_ = capacity;
        r->write_ = write_;
        r->names_ = names;
        r->queue_ = queue < 0 || queue > 2 ? 0 : queue;
        r->mask_ = bits >= 64 ? ~0ull : (1ull << bits) - 1;
        r->period_ = period_ns;
        r->owners_.reserve(std::size_t(wanted));
        reset_(cb, pool, 0, capacity);
        r->stamp(cb, -1);
        return r;
    }

    // On the fence callback, once the submission is done: reads its timestamps, adds them up, and every 10 s prints.
    void resolve(Recording& r) {
        const auto now = std::chrono::steady_clock::now();
        std::vector<std::uint64_t> ticks(r.used_);
        const bool read = r.used_ >= 2 && device_ &&
                          results_(device_, r.pool_, 0, r.used_, ticks.size() * sizeof(std::uint64_t), ticks.data(),
                                   sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
        double other_ns = 0, span_ns = 0;
        if (read) {
            for (std::uint32_t k = 1; k < r.used_; ++k) {
                const std::uint64_t d = (ticks[k] - ticks[k - 1]) & r.mask_;
                const std::int32_t owner = r.owners_[k - 1];
                if (owner < 0) other_ns += double(d) * r.period_;
                else r.runs_[std::size_t(owner)].ticks += d;
            }
            span_ns = double((ticks[r.used_ - 1] - ticks[0]) & r.mask_) * r.period_;
        }
        std::lock_guard lock(mutex_);
        QueueTotals& q = queues_[r.queue_];
        // As LEONIDA_GPU_LATENCY: an in-order queue is busy from the later of the submit and the previous
        // completion it saw to this one.
        if (r.submitted_ != std::chrono::steady_clock::time_point{}) {
            const auto from = (std::max)(r.submitted_, q.last_done);
            if (now > from) q.busy_ns += double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - from).count());
            q.last_done = now;
        }
        ++q.submissions;
        q.dropped += r.dropped_;
        if (!read) {
            ++q.unread;
        } else {
            q.span_ns += span_ns;
            q.other_ns += other_ns;
            q.stamps += r.used_;
            for (const auto& run : r.runs_) {
                const double ns = double(run.ticks) * r.period_;
                auto [it, inserted] = pipelines_.try_emplace(run.key);
                if (inserted) it->second.label = label(run.key);
                it->second.ns += ns;
                it->second.calls += run.calls;
                ++it->second.runs;
                q.pipelines_ns += ns;
                kind_ns_[std::size_t(run.key.kind)] += ns;
            }
        }
        if (now - since_ >= kWindow) report_locked(now);
    }

    // Tests: the sums so far, by pipeline.
    struct Entry {
        Key key;
        std::string label;
        double ms = 0;
        std::uint64_t calls = 0, runs = 0;
    };
    std::vector<Entry> snapshot() {
        std::lock_guard lock(mutex_);
        std::vector<Entry> out;
        for (const auto& [key, t] : pipelines_) out.push_back({key, t.label, t.ns / 1e6, t.calls, t.runs});
        return out;
    }
    struct QueueSnapshot {
        std::uint64_t submissions = 0, unread = 0, stamps = 0, dropped = 0;
        double busy_ms = 0, span_ms = 0, pipelines_ms = 0, other_ms = 0;
    };
    QueueSnapshot queue(int kind) {
        std::lock_guard lock(mutex_);
        const QueueTotals& q = queues_[kind < 0 || kind > 2 ? 0 : kind];
        return {q.submissions, q.unread, q.stamps, q.dropped, q.busy_ns / 1e6, q.span_ns / 1e6, q.pipelines_ns / 1e6,
                q.other_ns / 1e6};
    }
    // Tests: prints the lines now and starts a new window.
    void report_now() {
        std::lock_guard lock(mutex_);
        report_locked(std::chrono::steady_clock::now());
    }

    // Before the device goes: destroys the idle pools; a recording that ends later destroys its own.
    void stop() {
        std::lock_guard lock(mutex_);
        for (auto& [capacity, pools] : idle_)
            for (VkQueryPool pool : pools) destroy_(device_, pool, nullptr);
        idle_.clear();
        stopped_ = true;
    }

private:
    friend class Recording;
    struct Totals {
        std::string label;
        double ns = 0;
        std::uint64_t calls = 0, runs = 0;
    };
    struct QueueTotals {
        std::uint64_t submissions = 0, unread = 0, stamps = 0, dropped = 0;
        double busy_ns = 0, span_ns = 0, pipelines_ns = 0, other_ns = 0;
        std::chrono::steady_clock::time_point last_done{};
    };

    static const char* queue_name(int queue) { return queue == 0 ? "direct" : queue == 1 ? "compute" : "copy"; }

    VkQueryPool take(std::uint32_t capacity) {
        {
            std::lock_guard lock(mutex_);
            if (!device_ || stopped_) return VK_NULL_HANDLE;
            auto it = idle_.find(capacity);
            if (it != idle_.end() && !it->second.empty()) {
                VkQueryPool pool = it->second.back();
                it->second.pop_back();
                return pool;
            }
        }
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = capacity;
        VkQueryPool pool = VK_NULL_HANDLE;
        return create_(device_, &info, nullptr, &pool) == VK_SUCCESS ? pool : VK_NULL_HANDLE;
    }
    void recycle(VkQueryPool pool, std::uint32_t capacity) {
        if (!pool) return;
        std::lock_guard lock(mutex_);
        if (stopped_) destroy_(device_, pool, nullptr);
        else idle_[capacity].push_back(pool);
    }

    // mutex_ held.
    void report_locked(std::chrono::steady_clock::time_point now) {
        const double seconds = std::chrono::duration<double>(now - since_).count();
        const std::uint64_t serial = submit_phases::totals().present_serial.load(std::memory_order_relaxed);
        const std::uint64_t frames = serial - frames_at_;
        // Per frame when the window presented any, else the window's totals.
        const double per = frames ? double(frames) : 1.0;
        const char* unit = frames ? "/frame" : "";
        double pipelines_ns = 0, other_ns = 0;
        std::uint64_t calls = 0, runs = 0, stamps = 0;
        for (const auto& [key, t] : pipelines_) {
            pipelines_ns += t.ns;
            calls += t.calls;
            runs += t.runs;
        }
        for (const auto& q : queues_) {
            other_ns += q.other_ns;
            stamps += q.stamps;
        }
        std::fprintf(stderr,
                     "[rt] gpu profile: %.1f s, %llu frames%s: compute %.2f ms%s, graphics %.2f ms%s, mesh %.2f ms%s "
                     "(%.1f calls%s in %.1f runs%s), other %.2f ms%s (barriers, copies, clears, queries, the layer's "
                     "own passes); %.0f timestamps%s\n",
                     seconds, static_cast<unsigned long long>(frames), frames ? "" : " (none presented: totals)",
                     kind_ns_[0] / 1e6 / per, unit, kind_ns_[1] / 1e6 / per, unit, kind_ns_[2] / 1e6 / per, unit,
                     double(calls) / per, unit, double(runs) / per, unit, other_ns / 1e6 / per, unit,
                     double(stamps) / per, unit);
        for (int k = 0; k < 3; ++k) {
            const QueueTotals& q = queues_[k];
            if (!q.submissions) continue;
            std::fprintf(stderr,
                         "[rt] gpu profile (%s): %llu submissions, busy %.2f ms%s, %.2f ms%s from first to last "
                         "timestamp (%.1f%% of busy), %.2f ms%s in draws and dispatches (%.1f%% of busy)",
                         queue_name(k), static_cast<unsigned long long>(q.submissions), q.busy_ns / 1e6 / per, unit,
                         q.span_ns / 1e6 / per, unit, q.busy_ns > 0 ? 100.0 * q.span_ns / q.busy_ns : 0.0,
                         q.pipelines_ns / 1e6 / per, unit, q.busy_ns > 0 ? 100.0 * q.pipelines_ns / q.busy_ns : 0.0);
            if (q.unread || q.dropped)
                std::fprintf(stderr, "; %llu submissions unread, %llu runs untimed",
                             static_cast<unsigned long long>(q.unread), static_cast<unsigned long long>(q.dropped));
            std::fputc('\n', stderr);
        }
        std::vector<const std::pair<const Key, Totals>*> ranked;
        ranked.reserve(pipelines_.size());
        for (const auto& entry : pipelines_) ranked.push_back(&entry);
        const std::size_t top = (std::min)(ranked.size(), std::size_t(kTop));
        std::partial_sort(ranked.begin(), ranked.begin() + std::ptrdiff_t(top), ranked.end(),
                          [](const auto* a, const auto* b) { return a->second.ns > b->second.ns; });
        for (std::size_t i = 0; i < top; ++i) {
            const auto& [key, t] = *ranked[i];
            std::fprintf(stderr,
                         "[rt] gpu profile #%zu: %.3f ms%s (%.1f%%), %s, %.1f calls%s, %.4f ms per call: %s\n", i + 1,
                         t.ns / 1e6 / per, unit, pipelines_ns > 0 ? 100.0 * t.ns / pipelines_ns : 0.0,
                         kind_name(key.kind), double(t.calls) / per, unit, t.calls ? t.ns / 1e6 / double(t.calls) : 0.0,
                         t.label.c_str());
        }
        pipelines_.clear();
        for (auto& q : queues_) {
            const auto last = q.last_done;
            q = {};
            q.last_done = last;
        }
        kind_ns_[0] = kind_ns_[1] = kind_ns_[2] = 0;
        since_ = now;
        frames_at_ = serial;
    }

    std::mutex mutex_;
    VkDevice device_ = VK_NULL_HANDLE;
    PFN_vkCreateQueryPool create_ = nullptr;
    PFN_vkDestroyQueryPool destroy_ = nullptr;
    PFN_vkCmdResetQueryPool reset_ = nullptr;
    PFN_vkCmdWriteTimestamp write_ = nullptr;
    PFN_vkGetQueryPoolResults results_ = nullptr;
    bool stopped_ = false;
    std::atomic<bool> no_timestamps_{false};
    std::map<std::uint32_t, std::vector<VkQueryPool>> idle_;  // by capacity
    std::unordered_map<Key, Totals, KeyHash> pipelines_;
    QueueTotals queues_[3];
    double kind_ns_[3] = {};
    std::chrono::steady_clock::time_point since_ = std::chrono::steady_clock::now();
    std::uint64_t frames_at_ = 0;
};

inline Recording::~Recording() {
    if (owner_) owner_->recycle(pool_, capacity_);
}

}  // namespace leonida::gpu_profile
