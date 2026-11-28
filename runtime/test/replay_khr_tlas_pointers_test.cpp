// A KHR TLAS build whose instances are an array of pointers (D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS) stages them with
// prepare_khr_instances. prepare() used to return that call's result straight away, so on success the commands after
// the build were never prepared (a later dispatch had no pipeline, a later build no structure) and the list's
// end-of-list checks were skipped. Only a failure ends the list there now, as on the XDXR path. No GPU: the
// acceleration structures, buffers and pipelines are fakes.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <string>

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

VKAPI_ATTR void VKAPI_CALL fake_build(VkCommandBuffer, std::uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR*,
                                      const VkAccelerationStructureBuildRangeInfoKHR* const*) {}

constexpr std::uint64_t tlas_va = 0x10000, blas_va = 0x20000, scratch_va = 0x30000, instances_va = 0x40000;
constexpr std::uintptr_t compute_id = 1;
const ComputePipeline compute = [] {
    ComputePipeline p{};
    p.pipeline = handle<VkPipeline>(0x1000);
    p.layout = handle<VkPipelineLayout>(7);
    return p;
}();

int staged = 0;
ReplayResult stage_result = ReplayResult::success;

ReplayFunctions functions() {
    ReplayFunctions vk{};
    vk.build_acceleration_structures = fake_build;
    vk.resolve_acceleration_structure = [](std::uint64_t va) {
        return va ? handle<VkAccelerationStructureKHR>(std::uintptr_t(va)) : VkAccelerationStructureKHR(VK_NULL_HANDLE);
    };
    vk.host_at = [](std::uint64_t, std::uint64_t, const std::uint8_t*&) { return HostReadStatus::absent; };
    vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer& buffer, VkDeviceSize& offset) {
        buffer = handle<VkBuffer>(9);
        offset = 0;
        return true;
    };
    vk.prepare_khr_instances = [](const BuildAccelerationStructure&, VkDeviceAddress& out) {
        ++staged;
        out = 0x80000;  // the staged instance array (kevincommits' repro value)
        return stage_result;
    };
    vk.bind_pipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    vk.push_constants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                            const void*) {};
    vk.dispatch = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {};
    vk.compute_pipeline = [](std::uintptr_t id) -> const ComputePipeline* { return id == compute_id ? &compute : nullptr; };
    return vk;
}

const ResourceBinding* resolve(ResourceId) { return nullptr; }

BuildAccelerationStructure tlas_of_pointers() {
    BuildAccelerationStructure b{};
    b.is_tlas = true;
    b.dst_va = tlas_va;
    b.scratch_va = scratch_va;
    b.instances_va = instances_va;
    b.instance_count = 2;
    b.instance_pointers = true;
    return b;
}

BuildAccelerationStructure blas() {
    BuildAccelerationStructure b{};
    b.dst_va = blas_va;
    b.scratch_va = scratch_va;
    return b;
}
}  // namespace

int main() {
    const ReplayFunctions vk = functions();

    // kevincommits' repro: the TLAS build, then a dispatch (and here a second build) after it.
    {
        CommandList list;
        check(list.build_acceleration_structure(tlas_of_pointers()), "record the TLAS build");
        check(list.dispatch({compute_id, {}, 1, 1, 1}), "record the dispatch");
        check(list.build_acceleration_structure(blas()), "record the BLAS build");
        list.close();
        staged = 0;
        stage_result = ReplayResult::success;
        PreparedList prepared;
        const ReplayResult r = prepare(list, resolve, vk, prepared);
        check(r == ReplayResult::success, "prepare succeeds");
        check(staged == 1, "the pointer array is staged once");
        check(prepared.commands.size() == 3, "three prepared commands");
        if (prepared.commands.size() == 3) {
            check(prepared.commands[0].khr_instances == 0x80000, "the TLAS build has its staged instances");
            check(prepared.commands[1].pipeline.pipeline == compute.pipeline,
                  "the dispatch after the TLAS build has its pipeline (was VK_NULL_HANDLE: dropped)");
            check(prepared.commands[2].accel_dst == handle<VkAccelerationStructureKHR>(std::uintptr_t(blas_va)),
                  "the build after the TLAS build has its structure");
        }
    }

    // A failed staging still ends the list with its result.
    {
        CommandList list;
        list.build_acceleration_structure(tlas_of_pointers());
        list.dispatch({compute_id, {}, 1, 1, 1});
        list.close();
        staged = 0;
        stage_result = ReplayResult::unresolved_resource;
        PreparedList prepared;
        check(prepare(list, resolve, vk, prepared) == ReplayResult::unresolved_resource,
              "a failed staging fails the list with its own result");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("replay_khr_tlas_pointers: ok\n");
    return 0;
}
