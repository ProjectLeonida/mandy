// Native GPU check of the immutable upload/replay contract under synchronization validation. The Windows COM
// companion tests the setters; this test needs neither Wine nor a title package.
#include "vk_test_device.h"
#include "shader_fixture_path.h"
#include "../src/vulkan_replay.h"
#include <cstring>
#include <fstream>

#define EXPECT(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
using namespace leonida;
int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    if (!d.buffer_device_address) return 77;
    std::ifstream input(vktest::shader_fixture_path(LEONIDA_ROOT_NATIVE_SPV), std::ios::binary | std::ios::ate);
    EXPECT(input);
    const auto bytes = input.tellg();
    EXPECT(bytes > 0 && bytes % 4 == 0);
    std::vector<std::uint32_t> code(std::size_t(bytes) / 4);
    input.seekg(0);
    EXPECT(input.read(reinterpret_cast<char*>(code.data()), bytes));
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = std::size_t(bytes);
    module_info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    EXPECT(d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &module_info, nullptr, &module) == VK_SUCCESS);
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 144};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    ComputePipeline pipeline;
    EXPECT(d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &layout_info, nullptr,
                                                                     &pipeline.layout) == VK_SUCCESS);
    VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    create.layout = pipeline.layout;
    create.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    create.stage.module = module;
    create.stage.pName = "main";
    EXPECT(d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(d.device, VK_NULL_HANDLE, 1, &create,
                                                                       nullptr, &pipeline.pipeline) == VK_SUCCESS);
    void* mapped = nullptr;
    const VkBuffer arena = d.buffer(16384, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    EXPECT(arena && mapped);
    auto* words = static_cast<std::uint32_t*>(mapped);
    ReplayFunctions vk{};
    vk.bind_pipeline = d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    vk.push_constants = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    vk.dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
    vk.compute_pipeline = [&](std::uintptr_t) { return &pipeline; };
    vk.heap_base = d.device_address(arena);
    std::uint64_t next = 8192;
    vk.prepare_root_arguments = [&](std::span<const std::uint32_t> data, std::uint64_t& va) {
        if (data.size() > 64 || next + data.size_bytes() + 64 > 16384) return false;
        va = next;
        std::memcpy(static_cast<std::uint8_t*>(mapped) + next, data.data(), data.size_bytes());
        std::memset(static_cast<std::uint8_t*>(mapped) + next + data.size_bytes(), 0, 64);
        next += 512;
        return true;
    };
    CommandList list;
    for (std::uint32_t n = 0; n < 4; ++n) {
        Dispatch cmd{1, {}, 2, 3, 4, std::vector<std::uint32_t>(32)};
        for (std::uint32_t i = 0; i < 30; ++i) cmd.root_arguments[i] = 1000 * n + i;
        cmd.root_arguments[30] = n * 256; // output console VA; all outputs lie below root storage
        for (std::uint32_t i = 0; i < 27; ++i) cmd.user_data[4 + i] = cmd.root_arguments[i];
        EXPECT(list.dispatch(cmd));
    }
    EXPECT(list.close());
    const auto resolve = [](ResourceId) -> const ResourceBinding* { return nullptr; };
    for (unsigned run = 0; run < 2; ++run) {
        PreparedList prepared;
        EXPECT(prepare(list, resolve, vk, prepared) == ReplayResult::success);
        EXPECT(d.run([&](VkCommandBuffer cb) {
            emit(prepared, cb, vk);
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
            return true;
        }));
        for (unsigned n = 0; n < 4; ++n) {
            for (unsigned i = 0; i < 30; ++i) EXPECT(words[n * 64 + i] == 1000 * n + i);
            EXPECT(words[n * 64 + 30] == n * 256 && words[n * 64 + 31] == 0);
            for (unsigned i = 0; i < 27; ++i) EXPECT(words[n * 64 + 32 + i] == 1000 * n + i);
            EXPECT(words[n * 64 + 59] == 0 && words[n * 64 + 60] == 2 && words[n * 64 + 61] == 3 &&
                   words[n * 64 + 62] == 4 && words[n * 64 + 63] == 0);
        }
    }
    d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, pipeline.pipeline, nullptr);
    d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, pipeline.layout, nullptr);
    d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, module, nullptr);
    EXPECT(d.validation_errors() == 0);
    std::puts("native root argument GPU readback: 8 snapshots, no validation errors");
    return 0;
}
