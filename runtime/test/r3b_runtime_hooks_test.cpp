// Round 3B runtime hooks test:
//   1. LiteralSamplerSpec builds valid VkSpecializationInfo entries with SpecId base + i, 0xFFFFFFFF fallback,
//      and uint32 overflow protection (CPU-only).
//   2. vulkan_replay emits a pre-dispatch COMPUTE->COMPUTE SHADER_WRITE -> SHADER_READ|SHADER_WRITE memory barrier
//      iff ComputePipeline::reads_glc_buffer == true (direct and indirect dispatch), and fires
//      ReplayFunctions::on_buffer_write + PreparedList::wrote_buffer iff writes_buffer == true (CPU-only).
//   3. Live Vulkan test (when a GPU is present): enables vulkanMemoryModel + vulkanMemoryModelDeviceScope,
//      compiles a hand-written SPIR-V 1.5 compute shader declaring OpCapability VulkanMemoryModel/DeviceScope
//      and two OpSpecConstant uint32s, specializes them via LiteralSamplerSpec (one live BindlessSet sampler index
//      and one 0xFFFFFFFF fallback), dispatches through VulkanQueue with reads_glc_buffer=true, verifies
//      on_buffer_write_submitted fires iff writes_buffer==true, and reads back the specialized values from the GPU.
// No game data.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#include "../src/literal_sampler_spec.h"
#include "../src/title_mirror.h"
#include "../src/title_texture_refresh.h"
#include "../src/vulkan_backend.h"
#include "../src/vulkan_bindless.h"
#include "../src/vulkan_replay.h"
#include "vk_test_device.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what, int line) {
    if (ok) return;
    std::fprintf(stderr, "FAIL (line %d): %s\n", line, what);
    ++failures;
}

#define CHECK(expr) check((expr), #expr, __LINE__)

template <class T>
T fake_handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

// ---------------------------------------------------------------------------------------------------------------
// Part 1: LiteralSamplerSpec unit checks (CPU-only)

void test_literal_sampler_spec() {
    leonida::LiteralSamplerSpec spec;
    CHECK(!spec.active() && spec.ptr() == nullptr);

    // Empty slice leaves spec inactive.
    const std::vector<std::uint32_t> empty;
    spec.build(100u, empty);
    CHECK(!spec.active() && spec.ptr() == nullptr);

    // Normal specialization with a bound index, a fallback sentinel, and another bound index.
    const std::uint32_t indices[3] = {5u, leonida::LiteralSamplerSpec::kFallbackIndex, 42u};
    spec.build(100u, indices);
    CHECK(spec.active() && spec.ptr() != nullptr);
    const VkSpecializationInfo* info = spec.ptr();
    CHECK(info->mapEntryCount == 3);
    CHECK(info->dataSize == 3 * sizeof(std::uint32_t));
    CHECK(info->pMapEntries != nullptr && info->pData != nullptr);
    for (std::uint32_t i = 0; i < 3; ++i) {
        CHECK(info->pMapEntries[i].constantID == 100u + i);
        CHECK(info->pMapEntries[i].offset == i * sizeof(std::uint32_t));
        CHECK(info->pMapEntries[i].size == sizeof(std::uint32_t));
    }
    const auto* vals = static_cast<const std::uint32_t*>(info->pData);
    CHECK(vals[0] == 5u && vals[1] == 0xFFFFFFFFu && vals[2] == 42u);

    // Boundary: SpecId UINT32_MAX with 1 entry is valid; with 2 entries overflows and safely clears.
    const std::uint32_t one[1] = {7u};
    spec.build(std::numeric_limits<std::uint32_t>::max(), one);
    CHECK(spec.active() && spec.ptr()->pMapEntries[0].constantID == std::numeric_limits<std::uint32_t>::max());

    const std::uint32_t two[2] = {7u, 8u};
    spec.build(std::numeric_limits<std::uint32_t>::max(), two);
    CHECK(!spec.active() && spec.ptr() == nullptr);
}

// ---------------------------------------------------------------------------------------------------------------
// Part 2: Replay reads_glc_buffer barrier + writes_buffer hook checks (CPU-only)

struct RecordedBarrier {
    VkPipelineStageFlags src_stage = 0;
    VkPipelineStageFlags dst_stage = 0;
    VkAccessFlags src_access = 0;
    VkAccessFlags dst_access = 0;
    std::size_t order = 0;
};

struct ReplayLog {
    std::size_t seq = 0;
    std::vector<RecordedBarrier> barriers;
    std::vector<std::size_t> dispatches;
    std::vector<std::size_t> indirect_dispatches;
    std::vector<std::size_t> begin_renderings;
    std::vector<std::size_t> draws;
    std::size_t buffer_writes = 0;

    void reset() {
        seq = 0;
        barriers.clear();
        dispatches.clear();
        indirect_dispatches.clear();
        begin_renderings.clear();
        draws.clear();
        buffer_writes = 0;
    }
};

ReplayLog g_log;

void VKAPI_PTR mock_barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags,
                            std::uint32_t mem_count, const VkMemoryBarrier* mem, std::uint32_t,
                            const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
    RecordedBarrier b{};
    b.src_stage = src;
    b.dst_stage = dst;
    if (mem_count && mem) {
        b.src_access = mem[0].srcAccessMask;
        b.dst_access = mem[0].dstAccessMask;
    }
    b.order = ++g_log.seq;
    g_log.barriers.push_back(b);
}

void VKAPI_PTR mock_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
    g_log.dispatches.push_back(++g_log.seq);
}

void VKAPI_PTR mock_dispatch_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize) {
    g_log.indirect_dispatches.push_back(++g_log.seq);
}

void VKAPI_PTR mock_begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {
    g_log.begin_renderings.push_back(++g_log.seq);
}

void VKAPI_PTR mock_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    g_log.draws.push_back(++g_log.seq);
}

void test_replay_hooks() {
    using namespace leonida;

    std::map<std::uintptr_t, ComputePipeline> compute;
    std::map<std::uintptr_t, GraphicsPipeline> graphics;

    ComputePipeline cs_ro{};
    cs_ro.layout = fake_handle<VkPipelineLayout>(1);
    cs_ro.pipeline = fake_handle<VkPipeline>(0x100);
    cs_ro.reads_glc_buffer = false;
    cs_ro.writes_buffer = false;
    compute[1] = cs_ro;

    ComputePipeline cs_glc_rw{};
    cs_glc_rw.layout = fake_handle<VkPipelineLayout>(1);
    cs_glc_rw.pipeline = fake_handle<VkPipeline>(0x101);
    cs_glc_rw.reads_glc_buffer = true;
    cs_glc_rw.writes_buffer = true;
    compute[2] = cs_glc_rw;

    ComputePipeline cs_glc_ro{};
    cs_glc_ro.layout = fake_handle<VkPipelineLayout>(1);
    cs_glc_ro.pipeline = fake_handle<VkPipeline>(0x102);
    cs_glc_ro.reads_glc_buffer = true;
    cs_glc_ro.writes_buffer = false;
    compute[3] = cs_glc_ro;

    ComputePipeline cs_rw_noglc{};
    cs_rw_noglc.layout = fake_handle<VkPipelineLayout>(1);
    cs_rw_noglc.pipeline = fake_handle<VkPipeline>(0x103);
    cs_rw_noglc.reads_glc_buffer = false;
    cs_rw_noglc.writes_buffer = true;
    compute[4] = cs_rw_noglc;

    ComputePipeline cs_skipped{};
    cs_skipped.layout = fake_handle<VkPipelineLayout>(1);
    cs_skipped.skipped = true;
    cs_skipped.writes_buffer = true;
    cs_skipped.build = +[](void*) -> VkPipeline { return VK_NULL_HANDLE; };
    compute[5] = cs_skipped;

    GraphicsPipeline gfx_ro{};
    gfx_ro.layout = fake_handle<VkPipelineLayout>(2);
    gfx_ro.pipeline = fake_handle<VkPipeline>(0x200);
    gfx_ro.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    gfx_ro.vertex_slots = 1;
    gfx_ro.writes_buffer = false;
    graphics[10] = gfx_ro;

    GraphicsPipeline gfx_rw = gfx_ro;
    gfx_rw.pipeline = fake_handle<VkPipeline>(0x201);
    gfx_rw.writes_buffer = true;
    graphics[11] = gfx_rw;

    GraphicsPipeline gfx_glc = gfx_ro;
    gfx_glc.pipeline = fake_handle<VkPipeline>(0x202);
    gfx_glc.reads_glc_buffer = true;
    gfx_glc.writes_buffer = false;
    graphics[12] = gfx_glc;

    ReplayFunctions vk{};
    vk.pipeline_barrier = &mock_barrier;
    vk.bind_pipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    vk.push_constants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                            const void*) {};
    vk.dispatch = &mock_dispatch;
    vk.dispatch_indirect = &mock_dispatch_indirect;
    vk.begin_rendering = &mock_begin_rendering;
    vk.end_rendering = +[](VkCommandBuffer) {};
    vk.set_viewport = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {};
    vk.set_scissor = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {};
    vk.set_primitive_topology = +[](VkCommandBuffer, VkPrimitiveTopology) {};
    vk.set_primitive_restart_enable = +[](VkCommandBuffer, VkBool32) {};
    vk.set_stencil_reference = +[](VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {};
    vk.set_blend_constants = +[](VkCommandBuffer, const float*) {};
    vk.bind_index_buffer = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {};
    vk.bind_vertex_buffers2 = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*, const VkDeviceSize*,
                                  const VkDeviceSize*, const VkDeviceSize*) {};
    vk.draw = &mock_draw;
    vk.draw_indexed = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t, std::uint32_t) {};
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) { return fake_handle<VkImageView>(8); };
    vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer& buffer, VkDeviceSize& offset) {
        buffer = fake_handle<VkBuffer>(9);
        offset = 0;
        return true;
    };
    vk.compute_pipeline = [&compute](std::uintptr_t id) -> const ComputePipeline* {
        const auto it = compute.find(id);
        return it == compute.end() ? nullptr : &it->second;
    };
    vk.graphics_pipeline = [&graphics](std::uintptr_t id) -> const GraphicsPipeline* {
        const auto it = graphics.find(id);
        return it == graphics.end() ? nullptr : &it->second;
    };
    vk.on_buffer_write = +[](void* ctx) { ++static_cast<ReplayLog*>(ctx)->buffer_writes; };
    vk.on_buffer_write_context = &g_log;

    constexpr ResourceId kColorId = 1;
    constexpr ResourceId kArgBufId = 2;
    const auto resolve = [](ResourceId id) -> const ResourceBinding* {
        static const ResourceBinding color = [] {
            ResourceBinding b{};
            b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
            b.format = VK_FORMAT_R8G8B8A8_UNORM;
            b.width = b.height = 16;
            b.image = fake_handle<VkImage>(10);
            return b;
        }();
        static const ResourceBinding arg_buf = [] {
            ResourceBinding b{};
            b.buffer = fake_handle<VkBuffer>(11);
            b.size = 256;
            b.indirect_arguments = true;
            return b;
        }();
        if (id == kColorId) return &color;
        if (id == kArgBufId) return &arg_buf;
        return nullptr;
    };

    const auto make_draw = [&](std::uintptr_t pso) {
        Draw d{};
        d.pipeline = pso;
        d.topology = 4;
        d.count = 3;
        d.instances = 1;
        auto views = std::make_shared<DrawViews>();
        views->render_targets[0] = {kColorId, 0, 0, 1};
        views->render_target_count = 1;
        views->vertex_buffers[0].va = 0x1000;
        views->vertex_buffers[0].size = 64;
        d.views = views;
        return d;
    };

    const auto run_list = [&](const CommandList& list, PreparedList& out) {
        g_log.reset();
        CHECK(prepare(list, resolve, vk, out) == ReplayResult::success);
        emit(out, fake_handle<VkCommandBuffer>(99), vk);
    };

    // 2a: Direct dispatch with reads_glc_buffer=false, writes_buffer=false.
    {
        CommandList list;
        Dispatch d{};
        d.pipeline = 1;
        d.x = d.y = d.z = 1;
        CHECK(list.dispatch(d) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(!prep.wrote_buffer);
        CHECK(g_log.barriers.empty());
        CHECK(g_log.dispatches.size() == 1);
        CHECK(g_log.buffer_writes == 0);
    }

    // 2b: Direct dispatch with reads_glc_buffer=true, writes_buffer=true (direct queue: src_stage = shader_stages).
    {
        CommandList list;
        Dispatch d{};
        d.pipeline = 2;
        d.x = d.y = d.z = 1;
        CHECK(list.dispatch(d) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(prep.wrote_buffer);
        CHECK(g_log.barriers.size() == 1);
        CHECK(g_log.dispatches.size() == 1);
        CHECK(g_log.barriers[0].order < g_log.dispatches[0]);
        CHECK(g_log.barriers[0].src_stage == shader_stages);
        CHECK(g_log.barriers[0].dst_stage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        CHECK(g_log.barriers[0].src_access == VK_ACCESS_SHADER_WRITE_BIT);
        CHECK(g_log.barriers[0].dst_access == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
        CHECK(g_log.buffer_writes == 1);
    }

    // 2b-compute: Dedicated compute queue restricts src_stage to COMPUTE_SHADER_BIT (no ALL_GRAPHICS_BIT).
    {
        vk.queue_stages = detail::queue_stages_compute;
        CommandList list;
        Dispatch d{};
        d.pipeline = 2;
        d.x = d.y = d.z = 1;
        CHECK(list.dispatch(d) && list.close());
        PreparedList prep;
        run_list(list, prep);
        vk.queue_stages = 0;
        CHECK(g_log.barriers.size() == 1);
        CHECK(g_log.barriers[0].src_stage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        CHECK(g_log.barriers[0].dst_stage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    // 2c: Indirect dispatch (2 records) with reads_glc_buffer=true, writes_buffer=false.
    {
        CommandList list;
        ExecuteIndirect ind{};
        ind.kind = IndirectKind::dispatch;
        ind.max_count = 2;
        ind.argument_buffer = kArgBufId;
        ind.argument_offset = 0;
        ind.stride = sizeof(VkDispatchIndirectCommand);
        ind.pipeline = 3;
        CHECK(list.execute_indirect(ind) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(!prep.wrote_buffer);
        CHECK(g_log.indirect_dispatches.size() == 2);
        CHECK(g_log.barriers.size() == 2);
        CHECK(g_log.barriers[0].order < g_log.indirect_dispatches[0]);
        CHECK(g_log.barriers[1].order > g_log.indirect_dispatches[0] &&
              g_log.barriers[1].order < g_log.indirect_dispatches[1]);
        CHECK(g_log.barriers[0].src_stage == shader_stages);
        CHECK(g_log.barriers[0].dst_stage == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        CHECK(g_log.barriers[0].src_access == VK_ACCESS_SHADER_WRITE_BIT);
        CHECK(g_log.barriers[0].dst_access == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT));
        CHECK(g_log.buffer_writes == 0);
    }

    // 2d: Indirect dispatch with reads_glc_buffer=false, writes_buffer=true.
    {
        CommandList list;
        ExecuteIndirect ind{};
        ind.kind = IndirectKind::dispatch;
        ind.max_count = 1;
        ind.argument_buffer = kArgBufId;
        ind.argument_offset = 0;
        ind.stride = sizeof(VkDispatchIndirectCommand);
        ind.pipeline = 4;
        CHECK(list.execute_indirect(ind) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(prep.wrote_buffer);
        CHECK(g_log.barriers.empty());
        CHECK(g_log.indirect_dispatches.size() == 1);
        CHECK(g_log.buffer_writes == 1);
    }

    // 2e: Skipped compute dispatch does NOT fire on_buffer_write.
    {
        CommandList list;
        Dispatch d{};
        d.pipeline = 5;
        d.x = d.y = d.z = 1;
        CHECK(list.dispatch(d) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(!prep.wrote_buffer);
        CHECK(g_log.dispatches.empty());
        CHECK(g_log.buffer_writes == 0);
    }

    // 2f: Graphics draw with writes_buffer=false, writes_buffer=true, and reads_glc_buffer=true.
    {
        CommandList list;
        CHECK(list.draw(make_draw(10)) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(!prep.wrote_buffer);
        CHECK(g_log.draws.size() == 1);
        CHECK(g_log.buffer_writes == 0);
        // No GLC memory barrier (only image layout transition barriers have src_access != VK_ACCESS_SHADER_WRITE_BIT).
        bool saw_glc = false;
        for (const auto& b : g_log.barriers)
            if (b.src_access == VK_ACCESS_SHADER_WRITE_BIT && b.dst_stage == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT)
                saw_glc = true;
        CHECK(!saw_glc);
    }
    {
        CommandList list;
        CHECK(list.draw(make_draw(11)) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(prep.wrote_buffer);
        CHECK(g_log.draws.size() == 1);
        CHECK(g_log.buffer_writes == 1);
    }
    {
        CommandList list;
        CHECK(list.draw(make_draw(12)) && list.close());
        PreparedList prep;
        run_list(list, prep);
        CHECK(!prep.wrote_buffer);
        CHECK(g_log.begin_renderings.size() == 1);
        CHECK(g_log.draws.size() == 1);
        CHECK(g_log.buffer_writes == 0);
        bool saw_glc_before_rp = false;
        for (const auto& b : g_log.barriers) {
            if (b.src_stage == shader_stages && b.dst_stage == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT &&
                b.src_access == VK_ACCESS_SHADER_WRITE_BIT &&
                b.dst_access == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) &&
                b.order < g_log.begin_renderings[0]) {
                saw_glc_before_rp = true;
            }
        }
        CHECK(saw_glc_before_rp);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Part 3: Live GPU test (VulkanMemoryModel + LiteralSamplerSpec + reads_glc_buffer + on_buffer_write_submitted)

// Hand-assembled SPIR-V 1.5 compute shader:
//   - Optional OpCapability VulkanMemoryModel + VulkanMemoryModelDeviceScope (with OpMemoryModel Logical Vulkan)
//   - layout(constant_id = 100) const uint kSpec0 = 0xFFFFFFFFu;
//   - layout(constant_id = 101) const uint kSpec1 = 0xFFFFFFFFu;
//   - layout(set = 0, binding = 0) buffer Out { uint vals[2]; } out_buf;
//   - void main() { out_buf.vals[0] = kSpec0; out_buf.vals[1] = kSpec1; }
std::vector<std::uint32_t> build_spec_test_spirv(bool use_vulkan_memory_model) {
    std::vector<std::uint32_t> w;
    const auto emit = [&](std::uint16_t op, std::initializer_list<std::uint32_t> args) {
        w.push_back((std::uint32_t(args.size() + 1) << 16) | op);
        for (std::uint32_t a : args) w.push_back(a);
    };

    // IDs
    constexpr std::uint32_t kVoid = 1;
    constexpr std::uint32_t kFnType = 2;
    constexpr std::uint32_t kU32 = 3;
    constexpr std::uint32_t kConst2 = 4;
    constexpr std::uint32_t kArrU32 = 5;
    constexpr std::uint32_t kStruct = 6;
    constexpr std::uint32_t kPtrStruct = 7;
    constexpr std::uint32_t kVarBuf = 8;
    constexpr std::uint32_t kPtrU32 = 9;
    constexpr std::uint32_t kConst0 = 10;
    constexpr std::uint32_t kConst1 = 11;
    constexpr std::uint32_t kSpec0 = 12;
    constexpr std::uint32_t kSpec1 = 13;
    constexpr std::uint32_t kMain = 14;
    constexpr std::uint32_t kEntryLabel = 15;
    constexpr std::uint32_t kPtr0 = 16;
    constexpr std::uint32_t kPtr1 = 17;
    constexpr std::uint32_t kBound = 18;

    // Header: magic, SPIR-V 1.5, generator 0, bound, schema 0
    w.push_back(0x07230203u);
    w.push_back(0x00010500u);
    w.push_back(0u);
    w.push_back(kBound);
    w.push_back(0u);

    // Capabilities
    emit(17, {1u});  // OpCapability Shader
    if (use_vulkan_memory_model) {
        emit(17, {5345u});  // OpCapability VulkanMemoryModel
        emit(17, {5346u});  // OpCapability VulkanMemoryModelDeviceScope
    }

    // Memory model: Logical (0), Vulkan (3) or GLSL450 (1)
    emit(14, {0u, use_vulkan_memory_model ? 3u : 1u});

    // OpEntryPoint GLCompute(5) %main "main" %var_buf
    // "main\0" packed little-endian: 0x6e69616du, 0x00000000u
    emit(15, {5u, kMain, 0x6e69616du, 0u, kVarBuf});
    // OpExecutionMode %main LocalSize(17) 1 1 1
    emit(16, {kMain, 17u, 1u, 1u, 1u});

    // Decorations
    emit(71, {kArrU32, 6u, 4u});         // ArrayStride 4
    emit(72, {kStruct, 0u, 35u, 0u});    // MemberDecorate Offset 0
    emit(71, {kStruct, 2u});             // Block
    emit(71, {kVarBuf, 34u, 0u});        // DescriptorSet 0
    emit(71, {kVarBuf, 33u, 0u});        // Binding 0
    emit(71, {kSpec0, 1u, 100u});        // SpecId 100
    emit(71, {kSpec1, 1u, 101u});        // SpecId 101

    // Types & constants
    emit(19, {kVoid});                   // OpTypeVoid
    emit(33, {kFnType, kVoid});          // OpTypeFunction
    emit(21, {kU32, 32u, 0u});           // OpTypeInt 32 0
    emit(43, {kU32, kConst2, 2u});       // OpConstant 2
    emit(28, {kArrU32, kU32, kConst2});  // OpTypeArray u32[2]
    emit(30, {kStruct, kArrU32});        // OpTypeStruct { u32[2] }
    emit(32, {kPtrStruct, 12u, kStruct});  // OpTypePointer StorageBuffer(12)
    emit(59, {kPtrStruct, kVarBuf, 12u});  // OpVariable StorageBuffer
    emit(32, {kPtrU32, 12u, kU32});      // OpTypePointer StorageBuffer u32
    emit(43, {kU32, kConst0, 0u});       // OpConstant 0
    emit(43, {kU32, kConst1, 1u});       // OpConstant 1
    emit(50, {kU32, kSpec0, 0xFFFFFFFFu});  // OpSpecConstant 0xFFFFFFFF
    emit(50, {kU32, kSpec1, 0xFFFFFFFFu});  // OpSpecConstant 0xFFFFFFFF

    // Function main()
    emit(54, {kVoid, kMain, 0u, kFnType});
    emit(248, {kEntryLabel});
    emit(65, {kPtrU32, kPtr0, kVarBuf, kConst0, kConst0});  // &out_buf.vals[0]
    emit(62, {kPtr0, kSpec0});
    emit(65, {kPtrU32, kPtr1, kVarBuf, kConst0, kConst1});  // &out_buf.vals[1]
    emit(62, {kPtr1, kSpec1});
    emit(253, {});  // OpReturn
    emit(56, {});   // OpFunctionEnd
    return w;
}

void test_gpu_live() {
    vktest::Device dev;
    dev.request_buffer_device_address = true;
    const auto status = dev.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::puts("SKIP: no Vulkan device for live Round 3B GPU test");
        return;
    }
    CHECK(status == vktest::DeviceStatus::ready);
    if (status != vktest::DeviceStatus::ready) return;

    std::printf("r3b GPU: vulkanMemoryModel=%d vulkanMemoryModelDeviceScope=%d bindless=%d\n",
                int(dev.vulkan_memory_model), int(dev.vulkan_memory_model_device_scope), int(dev.bindless));

    const auto vkCreateShaderModule = dev.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    const auto vkDestroyShaderModule = dev.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    const auto vkCreateDescriptorSetLayout = dev.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout");
    const auto vkDestroyDescriptorSetLayout = dev.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout");
    const auto vkCreateDescriptorPool = dev.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
    const auto vkDestroyDescriptorPool = dev.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
    const auto vkAllocateDescriptorSets = dev.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    const auto vkUpdateDescriptorSets = dev.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    const auto vkCreatePipelineLayout = dev.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    const auto vkDestroyPipelineLayout = dev.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    const auto vkCreateComputePipelines = dev.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    const auto vkDestroyPipeline = dev.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");
    const auto vkCreateSampler = dev.fn<PFN_vkCreateSampler>("vkCreateSampler");
    const auto vkDestroySampler = dev.fn<PFN_vkDestroySampler>("vkDestroySampler");

    // 1. Bind a real sampler into BindlessSet (when bindless is supported) to get a live non-zero sampler index.
    std::uint32_t expected_sampler_index = 7u;
    VkSampler test_sampler = VK_NULL_HANDLE;
    std::unique_ptr<leonida::BindlessSet> bindless;
    if (dev.bindless) {
        VkPhysicalDeviceMemoryProperties mem_props{};
        dev.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(dev.physical,
                                                                                                        &mem_props);
        leonida::BindlessFeatures bf{};
        bf.update_after_bind = true;
        bf.partially_bound = dev.partially_bound;
        bindless = std::make_unique<leonida::BindlessSet>(dev.device_proc, dev.device, mem_props, bf);
        CHECK(bindless->ok());
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        CHECK(vkCreateSampler(dev.device, &sci, nullptr, &test_sampler) == VK_SUCCESS);
        expected_sampler_index = bindless->add_sampler(test_sampler);
        CHECK(expected_sampler_index != 0 && expected_sampler_index != leonida::LiteralSamplerSpec::kFallbackIndex);
    }

    // 2. Output SSBO + descriptor set.
    void* mapped = nullptr;
    const VkBuffer out_buf = dev.buffer(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    CHECK(out_buf != VK_NULL_HANDLE && mapped != nullptr);
    if (!out_buf || !mapped) return;
    std::memset(mapped, 0xCC, 256);

    const VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsl_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl_ci.bindingCount = 1;
    dsl_ci.pBindings = &b0;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorSetLayout(dev.device, &dsl_ci, nullptr, &dsl) == VK_SUCCESS);

    const VkDescriptorPoolSize pool_sz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dp_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dp_ci.maxSets = 1;
    dp_ci.poolSizeCount = 1;
    dp_ci.pPoolSizes = &pool_sz;
    VkDescriptorPool dp = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorPool(dev.device, &dp_ci, nullptr, &dp) == VK_SUCCESS);

    VkDescriptorSetAllocateInfo ds_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ds_ai.descriptorPool = dp;
    ds_ai.descriptorSetCount = 1;
    ds_ai.pSetLayouts = &dsl;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    CHECK(vkAllocateDescriptorSets(dev.device, &ds_ai, &ds) == VK_SUCCESS);

    const VkDescriptorBufferInfo dbi{out_buf, 0, 256};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = ds;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev.device, 1, &wds, 0, nullptr);

    const VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(leonida::ShaderUserData)};
    VkPipelineLayoutCreateInfo pl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl_ci.setLayoutCount = 1;
    pl_ci.pSetLayouts = &dsl;
    pl_ci.pushConstantRangeCount = 1;
    pl_ci.pPushConstantRanges = &pcr;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    CHECK(vkCreatePipelineLayout(dev.device, &pl_ci, nullptr, &layout) == VK_SUCCESS);

    // 3. Build SPIR-V module (with VulkanMemoryModel + VulkanMemoryModelDeviceScope when supported) and
    //    specialize SpecId 100 -> expected_sampler_index, SpecId 101 -> 0xFFFFFFFF fallback via LiteralSamplerSpec.
    const bool use_vmm = dev.vulkan_memory_model && dev.vulkan_memory_model_device_scope;
    const auto spv = build_spec_test_spirv(use_vmm);
    VkShaderModuleCreateInfo sm_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm_ci.codeSize = spv.size() * sizeof(std::uint32_t);
    sm_ci.pCode = spv.data();
    VkShaderModule mod = VK_NULL_HANDLE;
    CHECK(vkCreateShaderModule(dev.device, &sm_ci, nullptr, &mod) == VK_SUCCESS);

    VkPipeline pipe = VK_NULL_HANDLE;
    {
        leonida::LiteralSamplerSpec spec;
        const std::uint32_t spec_vals[2] = {expected_sampler_index, leonida::LiteralSamplerSpec::kFallbackIndex};
        spec.build(100u, spec_vals);
        CHECK(spec.active());
        VkComputePipelineCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp_ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                       mod, "main", spec.ptr()};
        cp_ci.layout = layout;
        CHECK(vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &pipe) == VK_SUCCESS);
    }  // spec destroyed here: pipeline creation must have copied all specialization data
    vkDestroyShaderModule(dev.device, mod, nullptr);

    // 4. Execute through VulkanContext + VulkanQueue and verify on_buffer_write_submitted + GPU output.
    leonida::VulkanContext ctx(dev.device_proc, dev.device, true, dev.conditional_rendering,
                               dev.buffer_device_address);
    CHECK(ctx.loaded());
    std::vector<std::uint64_t> write_tickets;
    ctx.set_on_buffer_write_submitted([&](std::uint64_t ticket) { write_tickets.push_back(ticket); });

    leonida::ComputePipeline p_ro{};
    p_ro.pipeline = pipe;
    p_ro.layout = layout;
    p_ro.bindless_set = ds;
    p_ro.reads_glc_buffer = true;
    p_ro.writes_buffer = false;
    ctx.add_pipeline(1001, p_ro);

    leonida::ComputePipeline p_rw = p_ro;
    p_rw.writes_buffer = true;
    ctx.add_pipeline(1002, p_rw);

    leonida::VulkanQueue queue(ctx, dev.queue, dev.family, leonida::VulkanQueue::QueueKind::direct);
    CHECK(queue.ok());

    // Submission 1: writes_buffer=false -> on_buffer_write_submitted must NOT fire.
    {
        leonida::CommandList list;
        leonida::Dispatch d{};
        d.pipeline = 1001;
        d.x = d.y = d.z = 1;
        CHECK(list.dispatch(d) && list.close());
        const leonida::CommandList* lists[1] = {&list};
        CHECK(queue.execute(lists) == leonida::ReplayResult::success);
        CHECK(write_tickets.empty());
    }

    // Submission 2: writes_buffer=true, reads_glc_buffer=true -> on_buffer_write_submitted MUST fire with ticket > 0.
    {
        leonida::CommandList list;
        leonida::Dispatch d{};
        d.pipeline = 1002;
        d.x = d.y = d.z = 1;
        CHECK(list.dispatch(d) && list.close());
        const leonida::CommandList* lists[1] = {&list};
        CHECK(queue.execute(lists) == leonida::ReplayResult::success);
        CHECK(write_tickets.size() == 1 && write_tickets[0] > 0);
    }

    CHECK(queue.idle());

    const auto* out_words = static_cast<const std::uint32_t*>(mapped);
    CHECK(out_words[0] == expected_sampler_index);
    CHECK(out_words[1] == leonida::LiteralSamplerSpec::kFallbackIndex);

    vkDestroyPipeline(dev.device, pipe, nullptr);
    vkDestroyPipelineLayout(dev.device, layout, nullptr);
    vkDestroyDescriptorPool(dev.device, dp, nullptr);
    vkDestroyDescriptorSetLayout(dev.device, dsl, nullptr);
    bindless.reset();
    if (test_sampler) vkDestroySampler(dev.device, test_sampler, nullptr);
    CHECK(dev.validation_errors() == 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Part 4: PassiveWriteBackTracker (overlapping dual-queue seq pruning + split/non-snowballing) & TitleMirror::range_sum
//         bit-for-bit parity with mirror_range_sum_bytes() (CPU-only)

void test_passive_write_back_tracker() {
    using leonida::PassiveWriteBackTracker;
    using Range = std::pair<std::uint64_t, std::uint64_t>;

    const auto collect_promote = [](PassiveWriteBackTracker& tr, std::uint64_t id) {
        std::vector<Range> out;
        tr.promote(id, [&](std::uint64_t b, std::uint64_t e) { out.emplace_back(b, e); });
        return out;
    };

    // 4a. Single queue: promote clears everything immediately.
    {
        PassiveWriteBackTracker tr;
        const std::uint64_t id1 = tr.begin_execute();
        tr.note(0x1000, 0x3000);
        const auto p1 = collect_promote(tr, id1);
        CHECK(p1.size() == 1 && p1[0] == Range(0x1000, 0x3000));
        CHECK(tr.empty() && tr.bytes() == 0);
        tr.end_execute(id1);
    }

    // 4b. Read-only submission followed by writes_buffer submission: ranges survive end_execute until promoted.
    {
        PassiveWriteBackTracker tr;
        const std::uint64_t ro = tr.begin_execute();
        tr.note(0x1000, 0x2000);
        tr.end_execute(ro);  // did not write buffers -> NOT promoted
        CHECK(!tr.empty() && tr.bytes() == 0x1000);

        const std::uint64_t rw = tr.begin_execute();
        tr.note(0x4000, 0x6000);
        const auto p = collect_promote(tr, rw);
        CHECK(p.size() == 2 && p[0] == Range(0x1000, 0x2000) && p[1] == Range(0x4000, 0x6000));
        CHECK(tr.empty() && tr.bytes() == 0);
        tr.end_execute(rw);
    }

    // 4c. Continuous overlapping dual-queue (direct + compute) over 100 submissions:
    //     Stale ranges from submission k MUST be evicted on k's promote (never re-promoted on k+2..99!),
    //     while concurrent newly noted ranges from submission k+1 MUST NOT be lost!
    {
        PassiveWriteBackTracker tr;
        constexpr std::uint64_t kRangeSize = 0x10000;  // 64 KiB each
        constexpr std::uint64_t kBase = 0x10000000;
        std::uint64_t prev_id = tr.begin_execute();
        tr.note(kBase, kBase + kRangeSize);  // sub 0

        std::uint64_t total_promoted_bytes = 0;
        for (std::size_t step = 1; step < 100; ++step) {
            const std::uint64_t next_id = tr.begin_execute();
            const std::uint64_t next_addr = kBase + step * 2 * kRangeSize;
            tr.note(next_addr, next_addr + kRangeSize);  // concurrent submission notes its own range

            // Previous submission promotes while `next_id` is STILL in flight:
            const auto promoted = collect_promote(tr, prev_id);
            tr.end_execute(prev_id);

            // Must have promoted only ranges relevant to prev_id (never accumulating ranges from steps <= step - 2!).
            CHECK(promoted.size() <= (step == 2 ? 3u : 2u));
            for (const auto& [b, e] : promoted) total_promoted_bytes += (e - b);
            // At step == 1 both R0 and R1 are held for next_id (2); for step >= 2 only next_id's freshly noted range remains!
            const std::size_t expected_left = (step == 1) ? 2u : 1u;
            CHECK(tr.size() == expected_left && tr.bytes() == expected_left * kRangeSize);
            prev_id = next_id;
        }
        // Final submission promotes its own range and clears completely.
        const auto last = collect_promote(tr, prev_id);
        tr.end_execute(prev_id);
        CHECK(last.size() == 1 && last[0] == Range(kBase + 99 * 2 * kRangeSize, kBase + 99 * 2 * kRangeSize + kRangeSize));
        CHECK(tr.empty() && tr.bytes() == 0);
        // Without retire_id pruning, 100 overlapping submits re-promote ~5050 * 64 KiB; with retire_id each range is
        // promoted at most twice (exactly 200 * 64 KiB across 100 overlapping dual-queue steps).
        CHECK(total_promoted_bytes == 199 * kRangeSize);
    }

    // 4d. Pre-existing note(R) + concurrent begin(B) -> begin(A) -> promote(A) -> promote(B):
    //     BOTH A and B MUST emit R so B's later GPU write is never dropped, a newer queue C started after A's promote
    //     MUST NOT re-promote R, and R MUST be erased immediately after B promotes!
    {
        PassiveWriteBackTracker tr;
        tr.note(0x10000, 0x20000);                      // pre-existing R
        const std::uint64_t id_b = tr.begin_execute();  // B starts (id 1)
        const std::uint64_t id_a = tr.begin_execute();  // A starts (id 2)
        // A promotes first while B is still in flight: A emits R and stamps R.retire_id = id_b (1).
        const auto pa = collect_promote(tr, id_a);
        tr.end_execute(id_a);
        CHECK(pa.size() == 1 && pa[0] == Range(0x10000, 0x20000));
        CHECK(tr.size() == 1 && tr.bytes() == 0x10000);  // still held for B

        // A third queue C starts AFTER A's promote (id 3 > retire_id 1): C must NOT re-promote R!
        const std::uint64_t id_c = tr.begin_execute();
        const auto pc = collect_promote(tr, id_c);
        CHECK(pc.empty());
        CHECK(tr.size() == 1);  // still held for B

        // Now B promotes: B MUST emit R (id_b <= retire_id), and R is immediately erased because min_other (3) > 1!
        const auto pb = collect_promote(tr, id_b);
        tr.end_execute(id_b);
        CHECK(pb.size() == 1 && pb[0] == Range(0x10000, 0x20000));
        CHECK(tr.empty() && tr.bytes() == 0);
        tr.end_execute(id_c);
    }

    // 4e. Read-only overlapping queue ending without promote still prunes already-promoted entries on end_execute(),
    //     while preserving freshly noted unpromoted entries.
    {
        PassiveWriteBackTracker tr;
        tr.note(0x10000, 0x20000);                      // R_old
        const std::uint64_t ro = tr.begin_execute();    // read-only queue (id 1)
        const std::uint64_t rw = tr.begin_execute();    // writes_buffer queue (id 2)
        const auto p_rw = collect_promote(tr, rw);
        tr.end_execute(rw);
        CHECK(p_rw.size() == 1 && p_rw[0] == Range(0x10000, 0x20000));
        CHECK(tr.size() == 1);  // retained until ro ends

        // ro notes a new range R_new (retire_id = 0) and finishes without promoting: R_old MUST be pruned on
        // end_execute(ro), while unpromoted R_new MUST survive for a future writes_buffer submission!
        tr.note(0x30000, 0x34000);
        tr.end_execute(ro);
        CHECK(tr.size() == 1 && tr.bytes() == 0x4000);
        const auto p_later = collect_promote(tr, 0);
        CHECK(p_later.size() == 1 && p_later[0] == Range(0x30000, 0x34000));
        CHECK(tr.empty());
    }

    // 4f. Partial overlap split + adjacent non-snowballing across retire watermarks:
    {
        PassiveWriteBackTracker tr;
        tr.note(0x10000, 0x20000);  // old 64 KiB range (retire_id = 0)
        const std::uint64_t id_a = tr.begin_execute();  // id 1
        const std::uint64_t id_b = tr.begin_execute();  // id 2
        // Queue A promotes first: stamps [0x10000, 0x20000) with retire_id = id_b (2).
        const auto pa = collect_promote(tr, id_a);
        tr.end_execute(id_a);
        CHECK(pa.size() == 1 && pa[0] == Range(0x10000, 0x20000));

        // Queue C starts (id 3 > 2), notes a 4 KiB sub-range inside [0x10000, 0x20000) AND an adjacent 4 KiB range:
        // left [0x10000, 0x14000) and right [0x15000, 0x20000) keep retire_id = 2, while new ranges get retire_id = 0!
        const std::uint64_t id_c = tr.begin_execute();
        tr.note(0x14000, 0x15000);
        tr.note(0x20000, 0x21000);
        // Queue C promotes: MUST emit ONLY the 8 KiB new ranges (NOT the old 60 KiB with retire_id = 2 < 3)!
        const auto pc = collect_promote(tr, id_c);
        tr.end_execute(id_c);
        CHECK(pc.size() == 2 && pc[0] == Range(0x14000, 0x15000) && pc[1] == Range(0x20000, 0x21000));

        // Queue B promotes last (id 2): emits everything still relevant to B and clears ALL!
        const auto pb = collect_promote(tr, id_b);
        tr.end_execute(id_b);
        CHECK(pb.size() == 4);
        CHECK(tr.empty() && tr.bytes() == 0);
    }

    // 4g. FIFO cap eviction evicts strictly oldest seq first.
    {
        PassiveWriteBackTracker tr(4, 0x4000);  // max 4 ranges, 16 KiB
        tr.note(0x1000, 0x2000);  // seq 1 (4 KiB)
        tr.note(0x3000, 0x4000);  // seq 2 (4 KiB)
        tr.note(0x5000, 0x6000);  // seq 3 (4 KiB)
        tr.note(0x7000, 0x8000);  // seq 4 (4 KiB)
        tr.note(0x9000, 0xB000);  // seq 5 (8 KiB) -> evicts seq 1 and seq 2
        CHECK(tr.size() == 3 && tr.bytes() == 0x4000);
        const auto p = collect_promote(tr, 0);
        CHECK(p.size() == 3 && p[0] == Range(0x5000, 0x6000) && p[1] == Range(0x7000, 0x8000) &&
              p[2] == Range(0x9000, 0xB000));
    }
}

void test_mirror_range_sum_parity() {
    constexpr std::uint64_t kPage = leonida::TitleMirror::kPage;
    std::vector<std::uint8_t> unit_mem(leonida::TitleMirror::kUnitBytes, 0);
    std::vector<std::uint8_t> page0(kPage), page1(kPage);
    for (std::size_t i = 0; i < kPage; ++i) {
        page0[i] = static_cast<std::uint8_t>((i * 29u + 3u) & 0xffu);
        page1[i] = static_cast<std::uint8_t>((i * 53u + 17u) & 0xffu);
    }
    leonida::TitleMirrorHooks hooks{};
    hooks.make_chunk = [](std::uint64_t, std::uint64_t, VkBuffer& out) {
        out = fake_handle<VkBuffer>(0x500);
        return true;
    };
    hooks.make_unit = [&](VkDeviceMemory& out, std::uint8_t*& host) {
        out = fake_handle<VkDeviceMemory>(0x600);
        host = unit_mem.data();
        return true;
    };
    hooks.bind = [](const std::vector<VkSparseBufferMemoryBindInfo>&, std::uint64_t) { return true; };

    leonida::TitleMirror mirror(std::move(hooks));
    constexpr std::uint64_t kBase = 0x10000000;
    CHECK(mirror.cover(kBase, kBase + 2 * kPage));
    mirror.map(kBase, page0.data(), nullptr);
    mirror.map(kBase + kPage, page1.data(), nullptr);

    std::vector<std::uint8_t> contiguous(2 * kPage);
    std::memcpy(contiguous.data(), page0.data(), kPage);
    std::memcpy(contiguous.data() + kPage, page1.data(), kPage);

    // Verify exact bit-for-bit equality between TitleMirror::range_sum() and mirror_range_sum_bytes() across
    // sub-page, full-page, and cross-page 256B-aligned ranges!
    const std::pair<std::uint64_t, std::uint64_t> cases[] = {
        {0, 256},
        {256, 4096},
        {0x4000, 0x8000},
        {0, kPage},
        {kPage - 0x2000, 0x12000},  // crosses tile 0 -> tile 1
        {0, 2 * kPage},
    };
    for (const auto& [off, len] : cases) {
        std::uint64_t mirror_h = 0;
        CHECK(mirror.range_sum(kBase + off, len, mirror_h));
        const std::uint64_t cpu_h = leonida::mirror_range_sum_bytes(contiguous.data() + off, len);
        CHECK(mirror_h != 0 && mirror_h == cpu_h);
    }
}

}  // namespace

int main() {
    test_literal_sampler_spec();
    test_replay_hooks();
    test_gpu_live();
    test_passive_write_back_tracker();
    test_mirror_range_sum_parity();
    if (failures) {
        std::fprintf(stderr, "r3b_runtime_hooks_test: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("r3b_runtime_hooks_test: OK");
    return 0;
}
