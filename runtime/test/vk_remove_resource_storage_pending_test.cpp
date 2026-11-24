// Real Vulkan GPU regression test for VulkanContext::remove_resource() clearing storage_pending_
// (runtime/src/vulkan_backend.h):
//   1. Image A is registered at ResourceId 1 and marked via note_shader_image(1).
//   2. Image A is removed via remove_resource(1) before any shader submission runs.
//   3. A new Image B reuses ResourceId 1 (as COM heap allocations reuse addresses) without note_shader_image(1).
//   4. A real GPU compute dispatch runs on VulkanQueue.
//   Without storage_pending_.erase(id) in remove_resource(), execute_now() drains the stale id=1 entry,
//   resolves it to Image B, transitions Image B to GENERAL, and zeroes it with vkCmdClearColorImage.
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

// Minimal valid SPIR-V 1.0 compute shader: local_size(1,1,1), void main() {}
constexpr std::uint32_t kNoopComputeSpirv[] = {
    0x07230203u, 0x00010000u, 0x00000000u, 0x00000006u, 0x00000000u,
    0x00020011u, 0x00000001u,                                                 // OpCapability Shader
    0x0003000eu, 0x00000000u, 0x00000001u,                                    // OpMemoryModel Logical GLSL450
    0x0005000fu, 0x00000005u, 0x00000004u, 0x6e69616du, 0x00000000u,          // OpEntryPoint GLCompute %4 "main"
    0x00060010u, 0x00000004u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u, // OpExecutionMode %4 LocalSize 1 1 1
    0x00020013u, 0x00000002u,                                                 // %2 = OpTypeVoid
    0x00030021u, 0x00000003u, 0x00000002u,                                    // %3 = OpTypeFunction %2
    0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u,          // %4 = OpFunction %2 None %3
    0x000200f8u, 0x00000005u,                                                 // %5 = OpLabel
    0x000100fdu,                                                              // OpReturn
    0x00010038u,                                                              // OpFunctionEnd
};

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCmdClearColorImage real_clear_color = nullptr;
std::atomic<std::uint32_t> clear_color_calls{0};

VKAPI_ATTR void VKAPI_CALL cmd_clear_color(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
                                           const VkClearColorValue* color, std::uint32_t range_count,
                                           const VkImageSubresourceRange* ranges) {
    clear_color_calls.fetch_add(1, std::memory_order_relaxed);
    real_clear_color(cb, image, layout, color, range_count, ranges);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkCmdClearColorImage")
        return reinterpret_cast<PFN_vkVoidFunction>(cmd_clear_color);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_clear_color = reinterpret_cast<PFN_vkCmdClearColorImage>(device_proc(device, "vkCmdClearColorImage"));
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_remove_resource_storage_pending_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);

    const VkImage img_a = d.image(VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, VK_IMAGE_USAGE_STORAGE_BIT);
    const VkImage img_b = d.image(VK_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, VK_IMAGE_USAGE_SAMPLED_BIT);
    EXPECT(img_a && img_b);

    // Build a real no-op compute pipeline for Dispatch.
    const auto create_shader = d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    const auto destroy_shader = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    const auto create_pl = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    const auto destroy_pl = d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    const auto create_cp = d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    const auto destroy_cp = d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    EXPECT(create_shader && destroy_shader && create_pl && destroy_pl && create_cp && destroy_cp);

    VkShaderModuleCreateInfo sm_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm_info.codeSize = sizeof(kNoopComputeSpirv);
    sm_info.pCode = kNoopComputeSpirv;
    VkShaderModule sm = VK_NULL_HANDLE;
    EXPECT(create_shader(d.device, &sm_info, nullptr, &sm) == VK_SUCCESS);

    VkPipelineLayoutCreateInfo pl_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkPipelineLayout pl = VK_NULL_HANDLE;
    EXPECT(create_pl(d.device, &pl_info, nullptr, &pl) == VK_SUCCESS);

    VkComputePipelineCreateInfo cp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
    cp_info.layout = pl;
    VkPipeline cp = VK_NULL_HANDLE;
    EXPECT(create_cp(d.device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &cp) == VK_SUCCESS);
    destroy_shader(d.device, sm, nullptr);

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());

        ComputePipeline pso{};
        pso.pipeline = cp;
        pso.layout = pl;
        context.add_pipeline(100, pso);

        ResourceBinding bind_a{};
        bind_a.image = img_a;
        bind_a.format = VK_FORMAT_R8G8B8A8_UNORM;
        bind_a.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        bind_a.width = 4;
        bind_a.height = 4;

        ResourceBinding bind_b = bind_a;
        bind_b.image = img_b;

        // 1. Register image A at id=1 and mark it as pending shader-image init.
        context.add_resource(1, bind_a, VK_IMAGE_LAYOUT_UNDEFINED);
        context.note_shader_image(1);

        // 2. Remove image A before any shader submission runs.
        context.remove_resource(1);

        // 3. Reuse id=1 for image B (NOT a storage image, NOT passed to note_shader_image).
        context.add_resource(1, bind_b, VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED);

        // 4. Execute a real GPU compute dispatch. Image B must NOT be initialized/cleared by the stale id=1 entry.
        CommandList list;
        Dispatch disp{};
        disp.pipeline = 100;
        disp.x = 1;
        disp.y = 1;
        disp.z = 1;
        EXPECT(list.dispatch(disp));
        EXPECT(list.close());
        const CommandList* lists[] = {&list};
        driver::clear_color_calls.store(0);
        EXPECT(queue.execute(lists) == ReplayResult::success);
        EXPECT(context.vk.vkQueueWaitIdle(d.queue) == VK_SUCCESS);

        EXPECT(driver::clear_color_calls.load() == 0);
        EXPECT(context.layout(1, 0) == VK_IMAGE_LAYOUT_UNDEFINED);
    }

    destroy_cp(d.device, cp, nullptr);
    destroy_pl(d.device, pl, nullptr);

    std::printf("vk_remove_resource_storage_pending_test: PASS (0 spurious clears, reused ResourceId untouched)\n");
    return 0;
}
