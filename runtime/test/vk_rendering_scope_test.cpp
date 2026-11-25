// Consecutive draws retain their colour and depth attachments in one dynamic rendering scope.
// Hand-written shaders below use only a full-screen triangle and colour/depth push constants. Each draw scissors
// one pixel; the CPU applies the same LEQUAL depth test and alpha blend in order, then checks every colour/depth
// texel. --bench prints CPU emit, record/submit/wait and GPU timestamp medians (3 warmups, 7 samples) for compatible
// runs of 1, 4, 16, 64 and all draws. No timing threshold is tested.
// No game files or external shader compiler are needed. LEONIDA_VK_VALIDATION=1 enables the usual validation layer.
#include "vk_test_device.h"
#include "../src/vulkan_replay.h"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <vector>

namespace {
using namespace leonida;
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t side = 32, pixels = side * side;
constexpr VkFormat color_format = VK_FORMAT_R32G32B32A32_SFLOAT;
constexpr VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

// Generated solely from these hand-written synthetic GLSL sources with glslangValidator -V:
// scope.vert:
// #version 450
// layout(push_constant) uniform Inputs { vec4 color; float depth; } input_data;
// void main() {
//     vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
//     gl_Position = vec4(positions[gl_VertexIndex], input_data.depth, 1.0);
// }
// scope.frag:
// #version 450
// layout(push_constant) uniform Inputs { vec4 color; float depth; } input_data;
// layout(location = 0) out vec4 output_color;
// void main() { output_color = input_data.color; }
constexpr std::uint32_t vertex_words[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x0000002eu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0007000fu, 0x00000000u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000018u, 0x0000001cu, 0x00030003u,
    0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00050005u, 0x0000000cu,
    0x69736f70u, 0x6e6f6974u, 0x00000073u, 0x00060005u, 0x00000016u, 0x505f6c67u, 0x65567265u, 0x78657472u,
    0x00000000u, 0x00060006u, 0x00000016u, 0x00000000u, 0x505f6c67u, 0x7469736fu, 0x006e6f69u, 0x00070006u,
    0x00000016u, 0x00000001u, 0x505f6c67u, 0x746e696fu, 0x657a6953u, 0x00000000u, 0x00070006u, 0x00000016u,
    0x00000002u, 0x435f6c67u, 0x4470696cu, 0x61747369u, 0x0065636eu, 0x00070006u, 0x00000016u, 0x00000003u,
    0x435f6c67u, 0x446c6c75u, 0x61747369u, 0x0065636eu, 0x00030005u, 0x00000018u, 0x00000000u, 0x00060005u,
    0x0000001cu, 0x565f6c67u, 0x65747265u, 0x646e4978u, 0x00007865u, 0x00040005u, 0x00000021u, 0x75706e49u,
    0x00007374u, 0x00050006u, 0x00000021u, 0x00000000u, 0x6f6c6f63u, 0x00000072u, 0x00050006u, 0x00000021u,
    0x00000001u, 0x74706564u, 0x00000068u, 0x00050005u, 0x00000023u, 0x75706e69u, 0x61645f74u, 0x00006174u,
    0x00030047u, 0x00000016u, 0x00000002u, 0x00050048u, 0x00000016u, 0x00000000u, 0x0000000bu, 0x00000000u,
    0x00050048u, 0x00000016u, 0x00000001u, 0x0000000bu, 0x00000001u, 0x00050048u, 0x00000016u, 0x00000002u,
    0x0000000bu, 0x00000003u, 0x00050048u, 0x00000016u, 0x00000003u, 0x0000000bu, 0x00000004u, 0x00040047u,
    0x0000001cu, 0x0000000bu, 0x0000002au, 0x00030047u, 0x00000021u, 0x00000002u, 0x00050048u, 0x00000021u,
    0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u, 0x00000021u, 0x00000001u, 0x00000023u, 0x00000010u,
    0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u, 0x00000006u, 0x00000020u,
    0x00040017u, 0x00000007u, 0x00000006u, 0x00000002u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000000u,
    0x0004002bu, 0x00000008u, 0x00000009u, 0x00000003u, 0x0004001cu, 0x0000000au, 0x00000007u, 0x00000009u,
    0x00040020u, 0x0000000bu, 0x00000007u, 0x0000000au, 0x0004002bu, 0x00000006u, 0x0000000du, 0xbf800000u,
    0x0005002cu, 0x00000007u, 0x0000000eu, 0x0000000du, 0x0000000du, 0x0004002bu, 0x00000006u, 0x0000000fu,
    0x40400000u, 0x0005002cu, 0x00000007u, 0x00000010u, 0x0000000fu, 0x0000000du, 0x0005002cu, 0x00000007u,
    0x00000011u, 0x0000000du, 0x0000000fu, 0x0006002cu, 0x0000000au, 0x00000012u, 0x0000000eu, 0x00000010u,
    0x00000011u, 0x00040017u, 0x00000013u, 0x00000006u, 0x00000004u, 0x0004002bu, 0x00000008u, 0x00000014u,
    0x00000001u, 0x0004001cu, 0x00000015u, 0x00000006u, 0x00000014u, 0x0006001eu, 0x00000016u, 0x00000013u,
    0x00000006u, 0x00000015u, 0x00000015u, 0x00040020u, 0x00000017u, 0x00000003u, 0x00000016u, 0x0004003bu,
    0x00000017u, 0x00000018u, 0x00000003u, 0x00040015u, 0x00000019u, 0x00000020u, 0x00000001u, 0x0004002bu,
    0x00000019u, 0x0000001au, 0x00000000u, 0x00040020u, 0x0000001bu, 0x00000001u, 0x00000019u, 0x0004003bu,
    0x0000001bu, 0x0000001cu, 0x00000001u, 0x00040020u, 0x0000001eu, 0x00000007u, 0x00000007u, 0x0004001eu,
    0x00000021u, 0x00000013u, 0x00000006u, 0x00040020u, 0x00000022u, 0x00000009u, 0x00000021u, 0x0004003bu,
    0x00000022u, 0x00000023u, 0x00000009u, 0x0004002bu, 0x00000019u, 0x00000024u, 0x00000001u, 0x00040020u,
    0x00000025u, 0x00000009u, 0x00000006u, 0x0004002bu, 0x00000006u, 0x00000028u, 0x3f800000u, 0x00040020u,
    0x0000002cu, 0x00000003u, 0x00000013u, 0x00050036u, 0x00000002u, 0x00000004u, 0x00000000u, 0x00000003u,
    0x000200f8u, 0x00000005u, 0x0004003bu, 0x0000000bu, 0x0000000cu, 0x00000007u, 0x0003003eu, 0x0000000cu,
    0x00000012u, 0x0004003du, 0x00000019u, 0x0000001du, 0x0000001cu, 0x00050041u, 0x0000001eu, 0x0000001fu,
    0x0000000cu, 0x0000001du, 0x0004003du, 0x00000007u, 0x00000020u, 0x0000001fu, 0x00050041u, 0x00000025u,
    0x00000026u, 0x00000023u, 0x00000024u, 0x0004003du, 0x00000006u, 0x00000027u, 0x00000026u, 0x00050051u,
    0x00000006u, 0x00000029u, 0x00000020u, 0x00000000u, 0x00050051u, 0x00000006u, 0x0000002au, 0x00000020u,
    0x00000001u, 0x00070050u, 0x00000013u, 0x0000002bu, 0x00000029u, 0x0000002au, 0x00000027u, 0x00000028u,
    0x00050041u, 0x0000002cu, 0x0000002du, 0x00000018u, 0x0000001au, 0x0003003eu, 0x0000002du, 0x0000002bu,
    0x000100fdu, 0x00010038u,
};
constexpr std::uint32_t fragment_words[] = {
    0x07230203u, 0x00010000u, 0x0008000bu, 0x00000012u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0006000bu,
    0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu, 0x00000000u, 0x00000001u,
    0x0006000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u, 0x00000009u, 0x00030010u, 0x00000004u,
    0x00000007u, 0x00030003u, 0x00000002u, 0x000001c2u, 0x00040005u, 0x00000004u, 0x6e69616du, 0x00000000u,
    0x00060005u, 0x00000009u, 0x7074756fu, 0x635f7475u, 0x726f6c6fu, 0x00000000u, 0x00040005u, 0x0000000au,
    0x75706e49u, 0x00007374u, 0x00050006u, 0x0000000au, 0x00000000u, 0x6f6c6f63u, 0x00000072u, 0x00050006u,
    0x0000000au, 0x00000001u, 0x74706564u, 0x00000068u, 0x00050005u, 0x0000000cu, 0x75706e69u, 0x61645f74u,
    0x00006174u, 0x00040047u, 0x00000009u, 0x0000001eu, 0x00000000u, 0x00030047u, 0x0000000au, 0x00000002u,
    0x00050048u, 0x0000000au, 0x00000000u, 0x00000023u, 0x00000000u, 0x00050048u, 0x0000000au, 0x00000001u,
    0x00000023u, 0x00000010u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
    0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u, 0x00000008u,
    0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u, 0x0004001eu, 0x0000000au,
    0x00000007u, 0x00000006u, 0x00040020u, 0x0000000bu, 0x00000009u, 0x0000000au, 0x0004003bu, 0x0000000bu,
    0x0000000cu, 0x00000009u, 0x00040015u, 0x0000000du, 0x00000020u, 0x00000001u, 0x0004002bu, 0x0000000du,
    0x0000000eu, 0x00000000u, 0x00040020u, 0x0000000fu, 0x00000009u, 0x00000007u, 0x00050036u, 0x00000002u,
    0x00000004u, 0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x00050041u, 0x0000000fu, 0x00000010u,
    0x0000000cu, 0x0000000eu, 0x0004003du, 0x00000007u, 0x00000011u, 0x00000010u, 0x0003003eu, 0x00000009u,
    0x00000011u, 0x000100fdu, 0x00010038u,
};

PFN_vkCmdBeginRendering real_begin = nullptr;
PFN_vkCmdEndRendering real_end = nullptr;
std::uint32_t begins = 0, ends = 0;
void VKAPI_CALL count_begin(VkCommandBuffer cb, const VkRenderingInfo* info) {
    ++begins;
    real_begin(cb, info);
}
void VKAPI_CALL count_end(VkCommandBuffer cb) {
    ++ends;
    real_end(cb);
}

struct Fixture {
    vktest::Device& d;
    VkImage color{}, depth{};
    VkImageView color_view{}, depth_view{};
    VkBuffer readback{};
    void* mapped = nullptr;
    VkShaderModule vs{}, fs{};
    VkQueryPool timestamps{};
    GraphicsPipeline graphics;
    ResourceBinding color_binding, depth_binding;
    VkPhysicalDeviceProperties properties{};
    std::uint32_t timestamp_bits = 0;

    ~Fixture() {
        if (graphics.pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, graphics.pipeline, nullptr);
        if (graphics.layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(d.device, graphics.layout, nullptr);
        for (auto module : {vs, fs})
            if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(d.device, module, nullptr);
        for (auto view : {color_view, depth_view})
            if (view) d.fn<PFN_vkDestroyImageView>("vkDestroyImageView")(d.device, view, nullptr);
        if (timestamps) d.fn<PFN_vkDestroyQueryPool>("vkDestroyQueryPool")(d.device, timestamps, nullptr);
    }

    bool initialize() {
        d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
        timestamp_bits = d.families[d.family].timestampValidBits;
        color = d.image(color_format, side, side, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        depth = d.image(depth_format, side, side, 1, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
        readback = d.buffer(pixels * (sizeof(float) * 4 + sizeof(float)), VK_BUFFER_USAGE_TRANSFER_DST_BIT, &mapped);
        if (!color || !depth || !readback || !mapped) return false;
        const auto view = [&](VkImage image, VkFormat format, VkImageAspectFlags aspects, VkImageView& out) {
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = image;
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = format;
            info.subresourceRange = {aspects, 0, 1, 0, 1};
            return d.fn<PFN_vkCreateImageView>("vkCreateImageView")(d.device, &info, nullptr, &out) == VK_SUCCESS;
        };
        if (!view(color, color_format, VK_IMAGE_ASPECT_COLOR_BIT, color_view) ||
            !view(depth, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT, depth_view)) return false;
        color_binding.image = color;
        color_binding.format = color_format;
        color_binding.width = color_binding.height = side;
        color_binding.color_attachment = true;
        depth_binding.image = depth;
        depth_binding.format = depth_format;
        depth_binding.width = depth_binding.height = side;
        depth_binding.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
        const auto shader = [&](auto& words, VkShaderModule& out) {
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize = sizeof(words);
            info.pCode = words;
            return d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &out) == VK_SUCCESS;
        };
        if (!shader(vertex_words, vs) || !shader(fragment_words, fs)) return false;
        graphics.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkPushConstantRange push{graphics.push_stages, 0, sizeof(ShaderUserData)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(d.device, &layout, nullptr, &graphics.layout) != VK_SUCCESS)
            return false;
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (unsigned k = 0; k < 2; ++k) {
            stages[k].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[k].stage = k ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
            stages[k].module = k ? fs : vs;
            stages[k].pName = "main";
        }
        VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        input.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = blend.alphaBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        color_blend.attachmentCount = 1;
        color_blend.pAttachments = &blend;
        const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = std::size(dynamic_states);
        dynamic.pDynamicStates = dynamic_states;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &color_format;
        rendering.depthAttachmentFormat = depth_format;
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
        pipeline.stageCount = 2;
        pipeline.pStages = stages;
        pipeline.pVertexInputState = &vertex;
        pipeline.pInputAssemblyState = &input;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &ds;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = graphics.layout;
        if (d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(d.device, VK_NULL_HANDLE, 1, &pipeline,
                nullptr, &graphics.pipeline) != VK_SUCCESS) return false;
        graphics.color_formats = {color_format};
        graphics.depth_format = depth_format;
        graphics.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (timestamp_bits) {
            VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            query.queryType = VK_QUERY_TYPE_TIMESTAMP;
            query.queryCount = 2;
            if (d.fn<PFN_vkCreateQueryPool>("vkCreateQueryPool")(d.device, &query, nullptr, &timestamps) != VK_SUCCESS)
                return false;
        }
        real_begin = d.fn<PFN_vkCmdBeginRendering>("vkCmdBeginRendering");
        real_end = d.fn<PFN_vkCmdEndRendering>("vkCmdEndRendering");
        return true;
    }

    ReplayFunctions functions() {
        ReplayFunctions vk;
#define FN(field, type, name) vk.field = d.fn<type>(name)
        FN(pipeline_barrier, PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
        FN(bind_pipeline, PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
        FN(push_constants, PFN_vkCmdPushConstants, "vkCmdPushConstants");
        FN(bind_vertex_buffers2, PFN_vkCmdBindVertexBuffers2, "vkCmdBindVertexBuffers2");
        FN(bind_index_buffer, PFN_vkCmdBindIndexBuffer, "vkCmdBindIndexBuffer");
        FN(set_viewport, PFN_vkCmdSetViewport, "vkCmdSetViewport");
        FN(set_scissor, PFN_vkCmdSetScissor, "vkCmdSetScissor");
        FN(set_primitive_topology, PFN_vkCmdSetPrimitiveTopology, "vkCmdSetPrimitiveTopology");
        FN(set_primitive_restart_enable, PFN_vkCmdSetPrimitiveRestartEnable, "vkCmdSetPrimitiveRestartEnable");
        FN(set_stencil_reference, PFN_vkCmdSetStencilReference, "vkCmdSetStencilReference");
        FN(set_blend_constants, PFN_vkCmdSetBlendConstants, "vkCmdSetBlendConstants");
        FN(draw, PFN_vkCmdDraw, "vkCmdDraw");
        FN(draw_indexed, PFN_vkCmdDrawIndexed, "vkCmdDrawIndexed");
#undef FN
        vk.begin_rendering = count_begin;
        vk.end_rendering = count_end;
        vk.graphics_pipeline = [&](std::uintptr_t id) { return id == 1 ? &graphics : nullptr; };
        vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer&, VkDeviceSize&) { return false; };
        vk.image_view = [&](const ResourceBinding& binding, const VkImageSubresourceRange&) {
            return binding.image == color ? color_view : binding.image == depth ? depth_view : VK_NULL_HANDLE;
        };
        return vk;
    }

    void barrier(VkCommandBuffer cb, VkImage image, VkImageAspectFlags aspect, VkImageLayout before,
                 VkImageLayout after, VkAccessFlags source_access, VkAccessFlags target_access,
                 VkPipelineStageFlags source_stage, VkPipelineStageFlags target_stage) {
        VkImageMemoryBarrier memory{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        memory.srcAccessMask = source_access;
        memory.dstAccessMask = target_access;
        memory.oldLayout = before;
        memory.newLayout = after;
        memory.srcQueueFamilyIndex = memory.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memory.image = image;
        memory.subresourceRange = {aspect, 0, 1, 0, 1};
        d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, source_stage, target_stage, 0, 0, nullptr, 0, nullptr,
                                                               1, &memory);
    }

    void clear(VkCommandBuffer cb) {
        d.initial_layout(cb, color, VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        d.initial_layout(cb, depth, VK_IMAGE_ASPECT_DEPTH_BIT, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const VkImageSubresourceRange color_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        const VkClearColorValue zero{};
        const VkClearDepthStencilValue depth_clear{1, 0};
        d.fn<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(cb, color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero,
                                                             1, &color_range);
        d.fn<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(cb, depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                                           &depth_clear, 1, &depth_range);
        barrier(cb, color, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        barrier(cb, depth, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);
    }

    void read(VkCommandBuffer cb) {
        barrier(cb, color, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        barrier(cb, depth, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        d.read_back(cb, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, side, side, 1, readback, 0);
        d.read_back(cb, depth, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, side, side, 1, readback,
                    pixels * sizeof(float) * 4);
    }
};

float depth_at(std::uint32_t draw) { return std::array{0.25f, 0.75f, 0.125f}[(draw / pixels) % 3]; }
std::array<float, 4> color_at(std::uint32_t draw) {
    std::array<float, 4> color{0, 0, 0, 0.5f};
    color[(draw / pixels) % 3] = 1;
    return color;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

bool run_case(Fixture& f, std::uint32_t count, std::uint32_t run_length, bool benchmark) {
    CommandList list;
    auto views = std::make_shared<DrawViews>();
    views->render_target_count = 1;
    views->render_targets[0] = {1, 0, 0, 1};
    std::vector<std::array<float, 4>> expected(pixels);
    std::vector<float> expected_depth(pixels, 1);
    for (std::uint32_t i = 0; i < count; ++i) {
        // A global UAV barrier must finish a rendering scope. Identical barriers in baseline and optimized
        // replay measure short compatible runs as well as the best case where the whole list is compatible.
        if (run_length && i && i % run_length == 0 && !list.uav_barrier({0})) return false;
        Draw draw;
        draw.pipeline = 1;
        draw.views = views;
        draw.depth = {2, 0, 0, 1};
        draw.viewport = {0, 0, float(side), float(side), 0, 1};
        const auto pixel = i * 37 % pixels, x = pixel % side, y = pixel / side;
        draw.scissor = {std::int32_t(x), std::int32_t(y), std::int32_t(x + 1), std::int32_t(y + 1)};
        draw.topology = 4;
        draw.count = 3;
        draw.instances = 1;
        const auto color = color_at(i);
        const auto depth = depth_at(i);
        for (unsigned k = 0; k < 4; ++k) draw.user_data[k] = std::bit_cast<std::uint32_t>(color[k]);
        draw.user_data[4] = std::bit_cast<std::uint32_t>(depth);
        if (!list.draw(std::move(draw))) return false;
        if (depth <= expected_depth[pixel]) {
            for (unsigned k = 0; k < 3; ++k) expected[pixel][k] = color[k] * 0.5f + expected[pixel][k] * 0.5f;
            expected[pixel][3] = 0.5f;
            expected_depth[pixel] = depth;
        }
    }
    if (!list.close()) return false;
    auto vk = f.functions();
    const auto resolve = [&](ResourceId id) { return id == 1 ? &f.color_binding : id == 2 ? &f.depth_binding : nullptr; };
    PreparedList prepared;
    const auto ready = prepare(list, resolve, vk, prepared);
    if (ready != ReplayResult::success || prepared.draw_count != count) {
        std::fprintf(stderr, "prepare failed: result=%d draws=%zu, expected=%u\n", int(ready), prepared.draw_count, count);
        return false;
    }
    std::vector<double> emit_ms, frame_ms, gpu_ms;
    const unsigned passes = benchmark ? 10 : 1, warmup = benchmark ? 3 : 0;
    bool ok = true;
    for (unsigned pass = 0; pass < passes; ++pass) {
        begins = ends = 0;
        double emit_time = 0;
        const auto start = Clock::now();
        if (!f.d.run([&](VkCommandBuffer cb) {
            f.clear(cb);
            if (f.timestamps) {
                // Finish the common clear/transition work before the interval starts. These execution stages
                // cover both earlier writes and the following timestamp, independently of the CPU record time.
                f.d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 0, nullptr);
                f.d.fn<PFN_vkCmdResetQueryPool>("vkCmdResetQueryPool")(cb, f.timestamps, 0, 2);
                f.d.fn<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, f.timestamps, 0);
            }
            const auto emit_start = Clock::now();
            emit(prepared, cb, vk);
            emit_time = std::chrono::duration<double, std::milli>(Clock::now() - emit_start).count();
            if (f.timestamps)
                f.d.fn<PFN_vkCmdWriteTimestamp>("vkCmdWriteTimestamp")(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, f.timestamps, 1);
            f.read(cb);
            return true;
        })) return false;
        const auto frame_time = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (pass >= warmup) {
            emit_ms.push_back(emit_time);
            frame_ms.push_back(frame_time);
            if (f.timestamps) {
                std::array<std::uint64_t, 2> values{};
                if (f.d.fn<PFN_vkGetQueryPoolResults>("vkGetQueryPoolResults")(f.d.device, f.timestamps, 0, 2,
                    sizeof(values), values.data(), sizeof(values[0]), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
                    return false;
                const auto mask = f.timestamp_bits == 64 ? UINT64_MAX : (std::uint64_t(1) << f.timestamp_bits) - 1;
                gpu_ms.push_back(double((values[1] - values[0]) & mask) * f.properties.limits.timestampPeriod / 1e6);
            }
        }
        const auto* color = static_cast<const float*>(f.mapped);
        const auto* depth = color + pixels * 4;
        for (unsigned p = 0; p < pixels; ++p) {
            for (unsigned k = 0; k < 4; ++k) {
                if (!std::isfinite(color[p * 4 + k]) || std::abs(color[p * 4 + k] - expected[p][k]) > 1e-6f) {
                    std::fprintf(stderr, "draws=%u pixel=%u component=%u got=%g expected=%g\n", count, p, k,
                                 double(color[p * 4 + k]), double(expected[p][k]));
                    return false;
                }
            }
            if (depth[p] != expected_depth[p]) {
                std::fprintf(stderr, "draws=%u depth pixel=%u got=%g expected=%g\n", count, p, double(depth[p]),
                             double(expected_depth[p]));
                return false;
            }
        }
        const auto expected_scopes = run_length ? (count + run_length - 1) / run_length : 1u;
        if (begins != expected_scopes || ends != begins) ok = false;
    }
    std::printf("draws=%u run_length=%u scopes=%u/%u pixel-depth-blend=PASS emit_ms=%.4f frame_ms=%.4f gpu_ms=%.4f\n",
                count, run_length ? run_length : count, begins, ends, median(emit_ms), median(frame_ms),
                gpu_ms.empty() ? 0.0 : median(gpu_ms));
    if (!ok) std::fprintf(stderr, "expected %u rendering scope(s), got %u/%u\n",
                         run_length ? (count + run_length - 1) / run_length : 1u, begins, ends);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--bench"))) {
        std::fprintf(stderr, "usage: vk_rendering_scope_test [--bench]\n");
        return 2;
    }
    vktest::Device device;
    const auto status = device.initialize();
    if (status != vktest::DeviceStatus::ready) return status == vktest::DeviceStatus::unavailable ? 77 : 1;
    VkFormatProperties format{};
    device.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
        device.physical, color_format, &format);
    if (!(format.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) ||
        !device.depth_format_usable(depth_format)) {
        std::printf("SKIP: RGBA32F blending or D32 attachment unavailable\n");
        return 77;
    }
    VkPhysicalDeviceProperties properties{};
    device.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(device.physical, &properties);
    if (properties.limits.maxPushConstantsSize < sizeof(ShaderUserData)) {
        std::printf("SKIP: the replay's 144-byte push constants are unavailable\n");
        return 77;
    }
    Fixture fixture{device};
    if (!fixture.initialize()) return 1;
    bool ok = run_case(fixture, 4096, 0, false);
    ok = run_case(fixture, 4096, 2048, false) && ok;
    if (argc == 2) {
        for (const auto count : {1000u, 10000u})
            for (const auto run_length : {1u, 4u, 16u, 64u, 0u})
                ok = run_case(fixture, count, run_length, true) && ok;
    }
    return ok && device.validation_errors() == 0 ? 0 : 1;
}
