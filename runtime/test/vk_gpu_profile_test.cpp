// LEONIDA_GPU_PROFILE (runtime/src/vulkan_gpu_profile.h): known dispatches and draws go through VulkanQueue with the
// profile on, and each pipeline must come back under its name with its call count, its runs and a GPU time that is
// above zero and grows with its work; the submission's span covers what the pipelines were given. Then the same
// short dispatches run with the profile off and on, and the difference per timestamp is printed (not checked). No
// game data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation; any validation error fails it.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;
using Kind = VulkanQueue::QueueKind;

constexpr std::uint32_t kGroups = 64;  // 64 x 64 invocations a dispatch
constexpr std::uint32_t kHeavyTurns = 20000, kLightTurns = 300, kVertexTurns = 4000;
constexpr std::uint32_t kVertices = 3 * 2048;
constexpr std::uintptr_t kHeavy = 0x1001, kLight = 0x1002, kDraw = 0x2001;

std::vector<std::uint32_t> read_words(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const std::streamoff bytes = file.tellg();
    if (bytes <= 0 || bytes % 4 != 0) return {};
    std::vector<std::uint32_t> words(static_cast<std::size_t>(bytes) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), bytes)) return {};
    return words;
}

struct Fixture {
    vktest::Device& d;
    VkShaderModule cs = VK_NULL_HANDLE, vs = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout compute_layout = VK_NULL_HANDLE, graphics_layout = VK_NULL_HANDLE;
    VkPipeline compute = VK_NULL_HANDLE, graphics = VK_NULL_HANDLE;

    ~Fixture() {
        if (graphics) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, graphics, nullptr);
        if (compute) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(d.device, compute, nullptr);
        const auto destroy_layout = d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
        if (graphics_layout) destroy_layout(d.device, graphics_layout, nullptr);
        if (compute_layout) destroy_layout(d.device, compute_layout, nullptr);
        if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(d.device, pool, nullptr);
        if (set_layout)
            d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(d.device, set_layout, nullptr);
        const auto destroy_shader = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
        if (vs) destroy_shader(d.device, vs, nullptr);
        if (cs) destroy_shader(d.device, cs, nullptr);
    }

    VkShaderModule module(const char* path) {
        const auto words = read_words(path);
        if (words.empty()) return VK_NULL_HANDLE;
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = words.size() * 4;
        info.pCode = words.data();
        VkShaderModule m = VK_NULL_HANDLE;
        return d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &m) == VK_SUCCESS
                   ? m : VK_NULL_HANDLE;
    }

    bool create(const char* compute_path, const char* vertex_path, VkBuffer results) {
        cs = module(compute_path);
        vs = module(vertex_path);
        if (!cs || !vs) return false;
        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                                   nullptr};
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 1;
        set_info.pBindings = &binding;
        if (d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(d.device, &set_info, nullptr,
                                                                                 &set_layout) != VK_SUCCESS)
            return false;
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &size;
        if (d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(d.device, &pool_info, nullptr, &pool) !=
            VK_SUCCESS)
            return false;
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = pool;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &set_layout;
        if (d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(d.device, &alloc, &set) != VK_SUCCESS)
            return false;
        const VkDescriptorBufferInfo buffer{results, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer;
        d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(d.device, 1, &write, 0, nullptr);

        const VkPushConstantRange compute_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderUserData)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set_layout;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &compute_range;
        const auto create_layout = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
        if (create_layout(d.device, &layout_info, nullptr, &compute_layout) != VK_SUCCESS) return false;
        const VkPushConstantRange vertex_range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShaderUserData)};
        layout_info.setLayoutCount = 0;
        layout_info.pSetLayouts = nullptr;
        layout_info.pPushConstantRanges = &vertex_range;
        if (create_layout(d.device, &layout_info, nullptr, &graphics_layout) != VK_SUCCESS) return false;

        VkComputePipelineCreateInfo compute_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        compute_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                              VK_SHADER_STAGE_COMPUTE_BIT, cs, "main"};
        compute_info.layout = compute_layout;
        if (d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(d.device, VK_NULL_HANDLE, 1, &compute_info,
                                                                           nullptr, &compute) != VK_SUCCESS)
            return false;

        // A vertex stage only, no attachments: the replay sizes the rendering by the viewport.
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                                    VK_SHADER_STAGE_VERTEX_BIT, vs, "main"};
        const VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        const VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        const VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        // What the replay sets at every draw (emit_commands).
        const VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT,          VK_DYNAMIC_STATE_SCISSOR,
                                         VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
                                         VK_DYNAMIC_STATE_STENCIL_REFERENCE,  VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = std::uint32_t(std::size(states));
        dynamic.pDynamicStates = states;
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rendering};
        info.stageCount = 1;
        info.pStages = &stage;
        info.pVertexInputState = &vi;
        info.pInputAssemblyState = &ia;
        info.pViewportState = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &ms;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.pDynamicState = &dynamic;
        info.layout = graphics_layout;
        return d.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(d.device, VK_NULL_HANDLE, 1, &info,
                                                                                nullptr, &graphics) == VK_SUCCESS;
    }
};

Dispatch dispatch(std::uintptr_t pipeline, std::uint32_t turns, std::uint32_t groups = kGroups) {
    Dispatch d{};
    d.pipeline = pipeline;
    d.user_data[0] = turns;
    d.x = groups;
    d.y = d.z = 1;
    return d;
}

Draw draw() {
    Draw d{};
    d.pipeline = kDraw;
    d.user_data[0] = kVertexTurns;
    d.topology = 4;  // D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
    d.viewport = {0, 0, 64, 64, 0, 1};
    d.scissor = {0, 0, 64, 64};
    d.count = kVertices;
    d.instances = 1;
    return d;
}

const gpu_profile::Profiler::Entry* find(const std::vector<gpu_profile::Profiler::Entry>& entries, const char* label) {
    for (const auto& e : entries)
        if (e.label == label) return &e;
    return nullptr;
}

// Waits until the fence thread has read `count` submissions of the direct queue.
bool resolved(VulkanContext& context, std::uint64_t count) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (context.gpu_profile().queue(0).submissions < count) {
        if (std::chrono::steady_clock::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

int check_labels() {
    gpu_profile::Key key;
    key.kind = gpu_profile::Kind::mesh;
    key.names = {"a/b/mesh_main", nullptr, "c/pixel_main"};
    EXPECT(gpu_profile::label(key) == "ms mesh_main ps pixel_main");
    key.kind = gpu_profile::Kind::graphics;
    key.names = {"vertex_main", "geometry_main", nullptr};
    EXPECT(gpu_profile::label(key) == "vs vertex_main gs geometry_main");
    key.kind = gpu_profile::Kind::compute;
    key.names = {"x/compute_main", nullptr, nullptr};
    EXPECT(gpu_profile::label(key) == "cs compute_main");
    key.names = {};
    key.id = 0xabc;
    EXPECT(gpu_profile::label(key) == "pso 0xabc");
    std::printf("labels: by stage names, else by id\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: vk_gpu_profile_test <gpu_profile_spin.comp.spv> <gpu_profile_spin.vert.spv>\n");
        return 2;
    }
    if (check_labels()) return 1;
    gpu_profile::set_enabled_for_test(true);  // before the context, which loads the query functions only then

    vktest::Device d;
    if (!d.create()) {
        std::printf("SKIP: no usable Vulkan device\n");
        return 77;
    }
    const std::uint32_t bits = d.families[d.family].timestampValidBits;
    VkPhysicalDeviceProperties properties{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
    if (!bits || properties.limits.timestampPeriod <= 0) {
        std::printf("SKIP: the graphics family writes no timestamps\n");
        return 77;
    }
    std::printf("device: %s, timestamps %u bits, %.3f ns a tick\n", properties.deviceName, bits,
                properties.limits.timestampPeriod);

    void* mapped = nullptr;
    const VkBuffer results = d.buffer(VkDeviceSize(kGroups) * 64 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    EXPECT(results);
    Fixture fixture{d};
    EXPECT(fixture.create(argv[1], argv[2], results));

    VulkanContext context(d.device_proc, d.device, true);
    EXPECT(context.loaded());
    context.set_timestamp_period(properties.limits.timestampPeriod);
    std::vector<std::uint32_t> family_bits;
    for (const auto& family : d.families) family_bits.push_back(family.timestampValidBits);
    context.set_timestamp_bits(family_bits);
    // One VkPipeline under two PSO ids and names: what each does is its user data's loop count.
    ComputePipeline compute{fixture.compute, fixture.compute_layout, 0, fixture.set};
    context.add_pipeline(kHeavy, compute, "profile_test/heavy");
    context.add_pipeline(kLight, compute, "profile_test/light");
    GraphicsPipeline graphics;
    graphics.pipeline = fixture.graphics;
    graphics.layout = fixture.graphics_layout;
    graphics.push_stages = VK_SHADER_STAGE_VERTEX_BIT;
    const CheckpointShaders draw_names{"profile_test/spin_vertices", nullptr, nullptr};
    context.add_graphics_pipeline(kDraw, graphics, &draw_names);
    VulkanQueue direct(context, d.queue, d.family, Kind::direct);
    EXPECT(direct.ok());

    // heavy x3, a UAV barrier, light x2, draw x4, heavy x1: heavy in two runs, the others in one each.
    CommandList list;
    for (int i = 0; i < 3; ++i) EXPECT(list.dispatch(dispatch(kHeavy, kHeavyTurns)));
    EXPECT(list.uav_barrier({0}));
    for (int i = 0; i < 2; ++i) EXPECT(list.dispatch(dispatch(kLight, kLightTurns)));
    for (int i = 0; i < 4; ++i) EXPECT(list.draw(draw()));
    EXPECT(list.dispatch(dispatch(kHeavy, kHeavyTurns)));
    EXPECT(list.close());
    const CommandList* lists[] = {&list};
    EXPECT(direct.execute(lists) == ReplayResult::success);
    EXPECT(direct.idle());
    EXPECT(resolved(context, 1));

    const auto entries = context.gpu_profile().snapshot();
    const auto queue = context.gpu_profile().queue(0);
    for (const auto& e : entries)
        std::printf("  %-28s %-8s %3llu calls in %llu runs, %.4f ms, %.4f ms per call\n", e.label.c_str(),
                    gpu_profile::kind_name(e.key.kind), static_cast<unsigned long long>(e.calls),
                    static_cast<unsigned long long>(e.runs), e.ms, e.ms / double(e.calls));
    std::printf("  queue: %llu submissions, %llu timestamps, busy %.4f ms, span %.4f ms, pipelines %.4f ms, "
                "other %.4f ms\n",
                static_cast<unsigned long long>(queue.submissions), static_cast<unsigned long long>(queue.stamps),
                queue.busy_ms, queue.span_ms, queue.pipelines_ms, queue.other_ms);
    EXPECT(entries.size() == 3);
    const auto* heavy = find(entries, "cs heavy");
    const auto* light = find(entries, "cs light");
    const auto* vertices = find(entries, "vs spin_vertices");
    EXPECT(heavy && light && vertices);
    EXPECT(heavy->key.kind == gpu_profile::Kind::compute && light->key.kind == gpu_profile::Kind::compute);
    EXPECT(vertices->key.kind == gpu_profile::Kind::graphics);
    EXPECT(heavy->calls == 4 && heavy->runs == 2);
    EXPECT(light->calls == 2 && light->runs == 1);
    EXPECT(vertices->calls == 4 && vertices->runs == 1);
    for (const auto* e : {heavy, light, vertices}) EXPECT(e->ms > 0 && e->ms < 2000);
    // 66 times the loop: well over twice the time per call, whatever the fixed cost of a dispatch is here.
    EXPECT(heavy->ms / 4 > 2 * (light->ms / 2));
    // Start, before heavy, after its first run (the barrier), before light, the draws and heavy, end: 7, all read.
    EXPECT(queue.submissions == 1 && queue.unread == 0 && queue.dropped == 0 && queue.stamps == 7);
    // Consecutive intervals: the pipelines and the other work add up to the span.
    EXPECT(queue.pipelines_ms <= queue.span_ms + 1e-6);
    EXPECT(queue.pipelines_ms + queue.other_ms <= queue.span_ms * 1.0001 + 1e-6);
    EXPECT(queue.pipelines_ms + queue.other_ms >= queue.span_ms * 0.9999 - 1e-6);
    context.gpu_profile().report_now();  // the lines' format, on stderr

    // The cost: 2048 one-group dispatches alternating between two pipelines (a run boundary, so a timestamp, at each),
    // submitted and waited for with the profile off and on; the best of five each.
    CommandList churn;
    for (int i = 0; i < 2048; ++i) EXPECT(churn.dispatch(dispatch(i % 2 ? kLight : kHeavy, 1, 1)));
    EXPECT(churn.close());
    const CommandList* churn_lists[] = {&churn};
    const auto best = [&](bool on) {
        gpu_profile::set_enabled_for_test(on);
        double least = 1e30;
        for (int round = 0; round < 5; ++round) {
            const auto start = std::chrono::steady_clock::now();
            if (direct.execute(churn_lists) != ReplayResult::success || !direct.idle()) return -1.0;
            least = std::min(least, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                              start).count());
        }
        return least;
    };
    const double off_ms = best(false), on_ms = best(true);
    EXPECT(off_ms >= 0 && on_ms >= 0);
    gpu_profile::set_enabled_for_test(true);
    EXPECT(resolved(context, 5));  // report_now() started a new window
    std::printf("cost: 2048 dispatches %.3f ms off, %.3f ms on (%.2f us per timestamp, submit to idle)\n", off_ms,
                on_ms, (on_ms - off_ms) * 1000.0 / 2049.0);

    if (d.validation) {
        std::printf("validation errors: %u\n", d.validation_errors());
        EXPECT(d.validation_errors() == 0);
    }
    std::printf("vk_gpu_profile_test: OK\n");
    return 0;
}
