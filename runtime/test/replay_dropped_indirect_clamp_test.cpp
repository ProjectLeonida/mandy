// An ExecuteIndirect(DISPATCH) whose compute pipeline is dropped (skipped on request, or still building with
// async_build) prepares and emits no clamp pass (vulkan_indirect_clamp.h): before, its clamp was prepared and emitted
// although no dispatch followed, and as the list's first clamp it owned the list's shared work budget, which every
// later clamp of the list then read. No GPU: pipelines, buffers and the clamp pass are fakes.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {
using namespace leonida;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

constexpr ResourceId arguments_id = 3;
constexpr std::uintptr_t good = 1, skipped = 2, building = 3;

struct Counts {
    int clamps_prepared = 0;
    int clamps_emitted = 0;
    int dispatches = 0;
    std::vector<VkDeviceAddress> emitted_budgets;  // list_budget of each emitted clamp
    std::vector<VkDeviceAddress> emitted_stats;    // stats of each emitted clamp
};
Counts counts;

std::vector<ComputePipeline> pipelines() {
    std::vector<ComputePipeline> out(4);
    out[good].pipeline = handle<VkPipeline>(0x1000);
    out[good].layout = handle<VkPipelineLayout>(7);
    out[skipped].layout = handle<VkPipelineLayout>(7);
    out[skipped].skipped = true;
    out[building].layout = handle<VkPipelineLayout>(7);
    out[building].async_build = true;
    return out;
}
const std::vector<ComputePipeline> registry = pipelines();

ReplayFunctions functions() {
    ReplayFunctions vk{};
    vk.bind_pipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    vk.push_constants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                            const void*) {};
    vk.dispatch = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {};
    vk.dispatch_indirect = +[](VkCommandBuffer, VkBuffer, VkDeviceSize) { ++counts.dispatches; };
    vk.compute_pipeline = [](std::uintptr_t id) -> const ComputePipeline* {
        return id < registry.size() && (registry[id].pipeline || registry[id].skipped || registry[id].async_build)
                   ? &registry[id]
                   : nullptr;
    };
    // Each clamp gets scratch of its own; its stats block holds the list budget word it owns (stats + 16).
    vk.prepare_indirect_clamp = [](VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, std::uint32_t, std::uint32_t stride,
                                   std::uint32_t max_count, ClampedIndirect& out) {
        const int n = ++counts.clamps_prepared;
        out.buffer = handle<VkBuffer>(0x100 + std::uintptr_t(n));
        out.offset = 0;
        out.stride = stride;
        out.records = max_count;
        out.stats = VkDeviceAddress(0x10000) * VkDeviceAddress(n);
        out.list_budget = out.stats + 16;
        return true;
    };
    vk.emit_indirect_clamp = [](VkCommandBuffer, const ClampedIndirect& clamp) {
        ++counts.clamps_emitted;
        counts.emitted_budgets.push_back(clamp.list_budget);
        counts.emitted_stats.push_back(clamp.stats);
    };
    return vk;
}

const ResourceBinding* resolve(ResourceId id) {
    static const ResourceBinding arguments = [] {
        ResourceBinding b{};
        b.buffer = handle<VkBuffer>(12);
        b.size = 4096;
        return b;
    }();
    return id == arguments_id ? &arguments : nullptr;
}

ExecuteIndirect dispatch_with(std::uintptr_t pipeline) {
    ExecuteIndirect e{};
    e.kind = IndirectKind::dispatch;
    e.stride = 12;
    e.max_count = 1;
    e.argument_buffer = arguments_id;
    e.pipeline = pipeline;
    return e;
}

// Prepares and emits `pipelines`, one indirect dispatch each, as one list.
void run(const char* what, const std::vector<std::uintptr_t>& order, int want_clamps, int want_dispatches) {
    counts = {};
    CommandList list;
    for (const auto p : order) list.execute_indirect(dispatch_with(p));
    list.close();
    const ReplayFunctions vk = functions();
    PreparedList prepared;
    const ReplayResult r = prepare(list, resolve, vk, prepared);
    check(r == ReplayResult::success, std::string(what) + ": prepare succeeds");
    if (r != ReplayResult::success) return;
    emit(prepared, handle<VkCommandBuffer>(5), vk);
    const std::string where = std::string(what) + ": ";
    check(counts.clamps_prepared == want_clamps,
          where + std::to_string(counts.clamps_prepared) + " clamps prepared, want " + std::to_string(want_clamps));
    check(counts.clamps_emitted == want_clamps,
          where + std::to_string(counts.clamps_emitted) + " clamps emitted, want " + std::to_string(want_clamps));
    check(counts.dispatches == want_dispatches,
          where + std::to_string(counts.dispatches) + " dispatches, want " + std::to_string(want_dispatches));
    // The list's budget word belongs to the first clamp emitted (its pass zeroes it), and every clamp shares it.
    for (std::size_t i = 0; i < counts.emitted_budgets.size(); ++i)
        check(counts.emitted_budgets[i] == counts.emitted_stats.front() + 16,
              where + "clamp " + std::to_string(i) + " reads a budget word no emitted clamp owns");
}
}  // namespace

int main() {
    // kevincommits' repro: the dropped dispatch alone.
    run("a skipped pipeline", {skipped}, 0, 0);
    run("a pipeline still building (async_build)", {building}, 0, 0);
    // The dropped dispatch first: the one that runs owns the budget it reads.
    run("skipped, then a built one", {skipped, good}, 1, 1);
    run("building, then a built one", {building, good}, 1, 1);
    run("built, skipped, built", {good, skipped, good}, 2, 2);
    // The control: two built dispatches share the first one's budget.
    run("two built", {good, good}, 2, 2);
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("replay_dropped_indirect_clamp: ok\n");
    return 0;
}
