// Real Vulkan GPU regression test for prepare() early return on KHR TLAS ARRAY_OF_POINTERS
// (runtime/src/vulkan_replay.h):
//   When BuildAccelerationStructure (KHR TLAS with instance_pointers=true, instance_count > 0) is followed by
//   subsequent commands in the same CommandList (here a real GPU CopyBufferRegion into a host-readable buffer,
//   and in a second list an invalid CopyBufferRegion), prepare() previously did:
//       return vk.prepare_khr_instances(*build, p.khr_instances);
//   which returned ReplayResult::success immediately on command 0, skipping preparation/validation of every
//   subsequent command in the list (dropping the GPU copy so host readback stayed poisoned, and skipping
//   validation errors on later commands).
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
std::atomic<std::uint32_t> build_as_calls{0};

// Stub vkCmdBuildAccelerationStructuresKHR so the test runs on every Vulkan 1.3 GPU (with or without RT)
// while executing the trailing CopyBufferRegion for real on the GPU and reading it back on the host.
VKAPI_ATTR void VKAPI_CALL stub_build_as(VkCommandBuffer, std::uint32_t,
                                         const VkAccelerationStructureBuildGeometryInfoKHR*,
                                         const VkAccelerationStructureBuildRangeInfoKHR* const*) {
    build_as_calls.fetch_add(1, std::memory_order_relaxed);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkCmdBuildAccelerationStructuresKHR")
        return reinterpret_cast<PFN_vkVoidFunction>(stub_build_as);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc) {
    real_get_device = device_proc;
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_khr_tlas_prepare_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc);

    void* ptr_buf_host = nullptr;
    void* inst_buf_host = nullptr;
    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    const VkBuffer ptr_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &ptr_buf_host);
    const VkBuffer inst_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &inst_buf_host);
    const VkBuffer src_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    const VkBuffer dst_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_ptr);
    EXPECT(ptr_buf && inst_buf && src_buf && dst_buf && ptr_buf_host && inst_buf_host && src_ptr && dst_ptr);

    constexpr std::uint64_t kPtrVa = 0x10000;
    constexpr std::uint64_t kInstVa = 0x20000;
    constexpr std::uint64_t kTlasVa = 0x30000;
    constexpr std::uint64_t kScratchVa = 0x40000;
    std::memcpy(ptr_buf_host, &kInstVa, sizeof(kInstVa));
    std::memset(inst_buf_host, 0, 64);

    std::uint32_t expected[16];
    for (int i = 0; i < 16; ++i) expected[i] = 0xCAFE0000u + std::uint32_t(i);
    std::memcpy(src_ptr, expected, sizeof(expected));
    std::memset(dst_ptr, 0x5A, sizeof(expected));

    {
        VulkanContext context(driver::get_device, d.device, true, false, /*ray_tracing=*/true, /*xdxr=*/false);
        EXPECT(context.loaded());
        // Supply physical memory properties so stage_khr_instances() can allocate its host-visible staging buffer.
        VkPhysicalDeviceMemoryProperties mem_props{};
        auto get_mem = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            d.instance_proc(d.instance, "vkGetPhysicalDeviceMemoryProperties"));
        EXPECT(get_mem);
        get_mem(d.physical, &mem_props);
        context.set_occlusion_resolve_memory(mem_props);
        context.set_heap_base(0x100000000ull);

        ResourceBinding ptr_res{};
        ptr_res.buffer = ptr_buf;
        ptr_res.size = 64;
        ptr_res.va = kPtrVa;
        ptr_res.host = static_cast<const std::uint8_t*>(ptr_buf_host);
        context.add_resource(1, ptr_res);

        ResourceBinding inst_res{};
        inst_res.buffer = inst_buf;
        inst_res.size = 64;
        inst_res.va = kInstVa;
        inst_res.host = static_cast<const std::uint8_t*>(inst_buf_host);
        context.add_resource(2, inst_res);

        ResourceBinding src_res{};
        src_res.buffer = src_buf;
        src_res.size = 64;
        context.add_resource(3, src_res);

        ResourceBinding dst_res{};
        dst_res.buffer = dst_buf;
        dst_res.size = 64;
        dst_res.host_readable = true;
        context.add_resource(4, dst_res);

        context.register_acceleration_structure(
            kTlasVa, reinterpret_cast<VkAccelerationStructureKHR>(std::uintptr_t{0x1234}));

        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());

        BuildAccelerationStructure tlas{};
        tlas.is_tlas = true;
        tlas.instance_pointers = true;
        tlas.instance_count = 1;
        tlas.instances_va = kPtrVa;
        tlas.dst_va = kTlasVa;
        tlas.scratch_va = kScratchVa;

        // 1. TLAS build (ARRAY_OF_POINTERS) followed by a real GPU CopyBufferRegion + host readback.
        CommandList list;
        EXPECT(list.build_acceleration_structure(tlas));
        EXPECT(list.copy_buffer({4, 0, 3, 0, 64}));
        EXPECT(list.close());
        const CommandList* lists[] = {&list};
        driver::build_as_calls.store(0);
        EXPECT(queue.execute(lists) == ReplayResult::success);
        EXPECT(context.vk.vkQueueWaitIdle(d.queue) == VK_SUCCESS);
        EXPECT(driver::build_as_calls.load() == 1);
        EXPECT(std::memcmp(dst_ptr, expected, sizeof(expected)) == 0);

        // 2. TLAS build followed by an invalid (OOB) CopyBufferRegion must still be rejected by prepare().
        CommandList bad_list;
        EXPECT(bad_list.build_acceleration_structure(tlas));
        EXPECT(bad_list.copy_buffer({4, 32, 3, 0, 64}));  // 32 + 64 > 64 -> invalid_copy
        EXPECT(bad_list.close());
        const CommandList* bad_lists[] = {&bad_list};
        EXPECT(queue.execute(bad_lists) == ReplayResult::invalid_copy);
    }

    std::printf("vk_khr_tlas_prepare_test: PASS (trailing GPU copy readback + validation verified)\n");
    return 0;
}
