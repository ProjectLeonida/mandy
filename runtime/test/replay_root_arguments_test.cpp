// Immutable argument snapshots, execution-time console VAs, and preparation failures. No GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdio>
#include <cstring>

using namespace leonida;
namespace {
int failures = 0;
std::vector<ShaderUserData> pushes;
std::vector<std::array<std::uint32_t, 3>> groups;
void expect(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
VKAPI_ATTR void VKAPI_CALL push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                               std::uint32_t, std::uint32_t size, const void* data) {
    expect(size == sizeof(ShaderUserData), "push size unchanged");
    ShaderUserData copy{};
    std::memcpy(&copy, data, sizeof(copy));
    pushes.push_back(copy);
}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    groups.push_back({x, y, z});
}
VKAPI_ATTR void VKAPI_CALL bind(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL indirect(VkCommandBuffer, VkBuffer, VkDeviceSize) {}
}  // namespace

int main() {
    ReplayFunctions vk{};
    vk.bind_pipeline = bind;
    vk.push_constants = push;
    vk.dispatch = dispatch;
    vk.dispatch_indirect = indirect;
    const ComputePipeline pipeline{reinterpret_cast<VkPipeline>(1), reinterpret_cast<VkPipelineLayout>(2)};
    vk.compute_pipeline = [&](std::uintptr_t) { return &pipeline; };
    vk.heap_base = 0x123400000000ull;
    ResourceBinding args{};
    args.buffer = reinterpret_cast<VkBuffer>(3);
    args.size = 12;
    const auto resolve = [&](ResourceId) { return &args; };
    std::vector<std::vector<std::uint32_t>> uploads;
    std::uint64_t va = 0x123400;  // deliberately unlike heap_base; high bits also tested below
    bool allocate = true;
    vk.prepare_root_arguments = [&](std::span<const std::uint32_t> words, std::uint64_t& out) {
        if (!allocate) return false;
        uploads.emplace_back(words.begin(), words.end());
        out = va;
        va += 0x100;
        return true;
    };
    CommandList list;
    Dispatch first{1, {}, 2, 3, 4, {11, 22, 33}};
    first.user_data[4] = 77;
    expect(list.dispatch(first), "record first");
    first.root_arguments[1] = 44;
    expect(list.dispatch(first), "record second");
    ExecuteIndirect ind{};
    ind.kind = IndirectKind::dispatch;
    ind.argument_buffer = 1;
    ind.max_count = 1;
    ind.stride = 12;
    ind.pipeline = 1;
    ind.root_arguments = {55, 66};
    expect(list.execute_indirect(ind) && list.close(), "record indirect and close");
    PreparedList prepared;
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare snapshots");
    expect(uploads == std::vector<std::vector<std::uint32_t>>({{11, 22, 33}, {11, 44, 33}, {55, 66}}),
           "all direct and indirect snapshots copied independently");
    emit(prepared, VK_NULL_HANDLE, vk);
    expect(groups == std::vector<std::array<std::uint32_t, 3>>({{2, 3, 4}, {2, 3, 4}}),
           "dispatch dimensions not shifted into block metadata");
    expect(pushes.size() == 3 && pushes[0].s[0] == 0x123400 && pushes[1].s[0] == 0x123500 &&
               pushes[2].s[0] == 0x123600 && pushes[0].s[1] == 0 && pushes[0].s[4] == 77 &&
               pushes[0].heap_base == vk.heap_base,
           "s0:s1 receive console offsets, inline data and heap base preserved");
    va = 0x1200000000ull;
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "repeat execution allocates fresh blocks");
    pushes.clear();
    emit(prepared, VK_NULL_HANDLE, vk);
    expect(pushes.size() == 3 && pushes[0].s[0] == 0 && pushes[0].s[1] == 0x12,
           "40-bit console pointer high word preserved");
    {
        // Main's deferred compile ticket and this execution's argument VA are independent metadata.
        int builds = 0;
        ComputePipeline pending{};
        pending.layout = pipeline.layout;
        pending.build_context = &builds;
        pending.build = [](void* context) {
            ++*static_cast<int*>(context);
            return reinterpret_cast<VkPipeline>(7);
        };
        pending.built = [](void* context) {
            return *static_cast<int*>(context) ? reinterpret_cast<VkPipeline>(7) : VK_NULL_HANDLE;
        };
        PipelineCompiles compiles;
        vk.compiles = &compiles;
        vk.compute_pipeline = [&](std::uintptr_t) { return &pending; };
        va = 0x123400;
        pushes.clear();
        groups.clear();
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success,
               "deferred pipelines prepare immutable root blocks");
        expect(compiles.size() == 1 && builds == 0, "direct and indirect dispatches share one deferred build");
        expect(build_pipelines(compiles, std::span<PreparedList>(&prepared, 1), ReplayResult::success,
                               [](std::size_t count, const std::function<void(std::size_t)>& job) {
                                   for (std::size_t i = 0; i < count; ++i) job(i);
                               }) == ReplayResult::success && builds == 1,
               "finish the deferred pipeline before replay");
        emit(prepared, VK_NULL_HANDLE, vk);
        expect(pushes.size() == 3 && pushes[0].s[0] == 0x123400 && pushes[1].s[0] == 0x123500 &&
                   pushes[2].s[0] == 0x123600 && groups.size() == 2,
               "deferred compile resolution preserves every direct and indirect argument VA");
        vk.compiles = nullptr;
        vk.compute_pipeline = [&](std::uintptr_t) { return &pipeline; };
    }
    va = vk.heap_base;
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::unsupported, "raw BDA is not a 40-bit console VA");
    va = 0x123401;
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::unsupported, "misaligned argument block refused");
    allocate = false;
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::unsupported, "allocation failure refuses in prepare");
    vk.prepare_root_arguments = {};
    expect(prepare(list, resolve, vk, prepared) == ReplayResult::unsupported, "missing uploader refuses in prepare");
    CommandList bad;
    first.root_arguments.resize(65);
    expect(bad.dispatch(first) && bad.close(), "record oversized block");
    expect(prepare(bad, resolve, vk, prepared) == ReplayResult::unsupported, "oversized snapshot refused");
    CommandList legacy;
    first.root_arguments.clear();
    expect(legacy.dispatch(first) && legacy.close(), "record legacy inline dispatch");
    expect(prepare(legacy, resolve, vk, prepared) == ReplayResult::success, "legacy callers need no uploader");
    std::printf("root argument replay: %d failures\n", failures);
    return failures ? 1 : 0;
}
