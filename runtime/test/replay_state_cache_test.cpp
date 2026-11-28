// Stage 5b Replay State Cache test (runtime/src/vulkan_replay.h, runtime/src/vulkan_backend.h):
//   1. CPU mock correctness: verifies redundant bind/push/dynamic-state elision across repeated draws & dispatches,
//      minimal dirty [lo, hi] 4-byte push-constant sub-range updates, full re-push on layout/stage switch and
//      graphics<->compute interleaving, dynamic state disturbance on mesh / !depth_bounds / !vertex_slots /
//      viewport_with_count pipeline transitions, cross-list caching on a shared ReplayStateCache, and compute+push
//      cache invalidation across every internal helper pass (capture_predicate, emit_indirect_clamp,
//      emit_quad_indices, emit_counted_dispatch, emit_predicated_copy, depth24_run, resolve_binary_occlusion,
//      sum_occlusion, sum_statistics, xdxr_record).
//   2. Throughput benchmark: replays 5,000 draws in a tight loop and verifies < 1.0 us/draw.
//   3. Live Vulkan validation test (when a GPU is present; run with LEONIDA_VK_VALIDATION=1): runs real graphics
//      draws (with dynamic viewport/scissor/topology/restart/stencil/blend/vertex/index), partial push-constant
//      updates, interleaved compute dispatches, and a clear pass through VulkanContext + VulkanQueue, reading back
//      both the color target and SSBO to verify bit-exact GPU execution with zero validation errors.
// No game data.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#include "../src/vulkan_backend.h"
#include "../src/vulkan_bindless.h"
#include "../src/vulkan_replay.h"
#include "vk_test_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace leonida;

int g_failures = 0;

void check_impl(bool ok, const char* expr, int line) {
    if (ok) return;
    std::fprintf(stderr, "FAIL (line %d): %s\n", line, expr);
    ++g_failures;
}

#define CHECK(expr) check_impl((expr), #expr, __LINE__)

template<typename H>
H fake_handle(std::uintptr_t v) {
    return reinterpret_cast<H>(v);
}

std::uint32_t float_bits(float f) {
    std::uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// ---------------------------------------------------------------------------------------------------------------
// Part 1: CPU mock correctness & invalidation coverage

struct PushCall {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderStageFlags stages = 0;
    std::uint32_t offset = 0;
    std::uint32_t size = 0;
    std::array<std::uint32_t, ReplayStateCache::kPushWords> words{};
};

struct MockCounts {
    std::size_t bind_gfx = 0;
    std::size_t bind_comp = 0;
    std::size_t desc_gfx = 0;
    std::size_t desc_comp = 0;
    std::size_t push = 0;
    std::size_t viewport = 0;
    std::size_t scissor = 0;
    std::size_t viewport_count = 0;
    std::size_t scissor_count = 0;
    std::size_t topology = 0;
    std::size_t restart = 0;
    std::size_t stencil_front = 0;
    std::size_t stencil_back = 0;
    std::size_t blend = 0;
    std::size_t depth_bounds = 0;
    std::size_t vertex = 0;
    std::size_t index = 0;
    std::size_t index_robust = 0;
    std::size_t draws = 0;
    std::size_t dispatches = 0;
    std::size_t begin_rendering = 0;
    std::size_t end_rendering = 0;
    std::size_t barriers = 0;
    std::vector<PushCall> pushes;

    void reset() { *this = MockCounts{}; }
};

MockCounts g_mc;

void VKAPI_PTR mc_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint bp, VkPipeline) {
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) ++g_mc.bind_gfx;
    else ++g_mc.bind_comp;
}
void VKAPI_PTR mc_bind_desc(VkCommandBuffer, VkPipelineBindPoint bp, VkPipelineLayout, std::uint32_t, std::uint32_t,
                            const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) ++g_mc.desc_gfx;
    else ++g_mc.desc_comp;
}
void VKAPI_PTR mc_push(VkCommandBuffer, VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset,
                       std::uint32_t size, const void* values) {
    ++g_mc.push;
    PushCall call{};
    call.layout = layout;
    call.stages = stages;
    call.offset = offset;
    call.size = size;
    if (values && offset + size <= sizeof(ShaderUserData) && (offset % 4 == 0) && (size % 4 == 0))
        std::memcpy(call.words.data() + offset / 4, values, size);
    g_mc.pushes.push_back(call);
}
void VKAPI_PTR mc_viewport(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) { ++g_mc.viewport; }
void VKAPI_PTR mc_scissor(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) { ++g_mc.scissor; }
void VKAPI_PTR mc_viewport_cnt(VkCommandBuffer, std::uint32_t, const VkViewport*) { ++g_mc.viewport_count; }
void VKAPI_PTR mc_scissor_cnt(VkCommandBuffer, std::uint32_t, const VkRect2D*) { ++g_mc.scissor_count; }
void VKAPI_PTR mc_topology(VkCommandBuffer, VkPrimitiveTopology) { ++g_mc.topology; }
void VKAPI_PTR mc_restart(VkCommandBuffer, VkBool32) { ++g_mc.restart; }
void VKAPI_PTR mc_stencil(VkCommandBuffer, VkStencilFaceFlags face, std::uint32_t) {
    if (face & VK_STENCIL_FACE_FRONT_BIT) ++g_mc.stencil_front;
    if (face & VK_STENCIL_FACE_BACK_BIT) ++g_mc.stencil_back;
}
void VKAPI_PTR mc_blend(VkCommandBuffer, const float*) { ++g_mc.blend; }
void VKAPI_PTR mc_depth_bounds(VkCommandBuffer, float, float) { ++g_mc.depth_bounds; }
void VKAPI_PTR mc_vertex(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*, const VkDeviceSize*,
                         const VkDeviceSize*, const VkDeviceSize*) {
    ++g_mc.vertex;
}
void VKAPI_PTR mc_index(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) { ++g_mc.index; }
void VKAPI_PTR mc_index_robust(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, VkIndexType) {
    ++g_mc.index_robust;
}
void VKAPI_PTR mc_draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) { ++g_mc.draws; }
void VKAPI_PTR mc_draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                               std::uint32_t) {
    ++g_mc.draws;
}
void VKAPI_PTR mc_draw_mesh(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) { ++g_mc.draws; }
void VKAPI_PTR mc_draw_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize, std::uint32_t, std::uint32_t) { ++g_mc.draws; }
void VKAPI_PTR mc_dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) { ++g_mc.dispatches; }
void VKAPI_PTR mc_dispatch_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize) { ++g_mc.dispatches; }

struct MockEnv {
    static constexpr ResourceId kColor = 1;
    static constexpr ResourceId kBuf = 2;
    static constexpr ResourceId kPred = 3;
    static constexpr ResourceId kTexD24 = 4;
    static constexpr ResourceId kResults = 5;
    static constexpr QueryHeapId kOccHeap = 10;
    static constexpr std::uint64_t kGeomVa = 0x100000000ull;

    std::map<ResourceId, ResourceBinding> res;
    std::map<std::uintptr_t, GraphicsPipeline> gfx;
    std::map<std::uintptr_t, ComputePipeline> comp;
    QueryPool occ_pool{};
    ReplayFunctions vk{};

    MockEnv() {
        ResourceBinding color{};
        color.image = fake_handle<VkImage>(0x100);
        color.format = VK_FORMAT_R8G8B8A8_UNORM;
        color.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        color.width = color.height = 64;
        color.color_attachment = true;
        res[kColor] = color;

        ResourceBinding buf{};
        buf.buffer = fake_handle<VkBuffer>(0x200);
        buf.size = 1 << 20;
        buf.indirect_arguments = true;
        res[kBuf] = buf;

        ResourceBinding pred{};
        pred.buffer = fake_handle<VkBuffer>(0x201);
        pred.size = 4096;
        res[kPred] = pred;

        ResourceBinding d24{};
        d24.image = fake_handle<VkImage>(0x101);
        d24.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        d24.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        d24.width = d24.height = 16;
        d24.depth24_as_float = true;
        d24.depth24 = true;
        res[kTexD24] = d24;

        ResourceBinding results{};
        results.buffer = fake_handle<VkBuffer>(0x202);
        results.size = 4096;
        res[kResults] = results;

        occ_pool = {fake_handle<VkQueryPool>(0x300), d3d12_query_heap::occlusion, 16, true,
                    std::vector<bool>(16, false)};

        const auto gfx_layout = fake_handle<VkPipelineLayout>(0x500);
        const auto comp_layout = fake_handle<VkPipelineLayout>(0x501);
        const auto bindless = fake_handle<VkDescriptorSet>(0x600);

        GraphicsPipeline g1{};
        g1.pipeline = fake_handle<VkPipeline>(0x700);
        g1.layout = gfx_layout;
        g1.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        g1.bindless_set = bindless;
        g1.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
        g1.vertex_slots = 0b1;
        g1.depth_bounds = true;
        gfx[1] = g1;

        GraphicsPipeline g2 = g1;
        g2.pipeline = fake_handle<VkPipeline>(0x701);
        g2.depth_bounds = false;  // disturbs depth_bounds when bound
        gfx[2] = g2;

        GraphicsPipeline g_vp_cnt = g1;
        g_vp_cnt.pipeline = fake_handle<VkPipeline>(0x702);
        g_vp_cnt.viewport_with_count = true;  // disturbs fixed viewport/scissor
        gfx[3] = g_vp_cnt;

        GraphicsPipeline g_mesh = g1;
        g_mesh.pipeline = fake_handle<VkPipeline>(0x703);
        g_mesh.mesh = true;
        g_mesh.vertex_slots = 0;
        g_mesh.push_stages = VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT;
        gfx[4] = g_mesh;

        ComputePipeline c1{};
        c1.pipeline = fake_handle<VkPipeline>(0x800);
        c1.layout = comp_layout;
        c1.bindless_set = bindless;
        comp[1] = c1;

        ComputePipeline c2 = c1;
        c2.pipeline = fake_handle<VkPipeline>(0x801);
        comp[2] = c2;

        vk.heap_base = 0x10000000ull;
        vk.pipeline_barrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                  std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                  std::uint32_t, const VkImageMemoryBarrier*) { ++g_mc.barriers; };
        vk.clear_color_image = +[](VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, std::uint32_t,
                                   const VkImageSubresourceRange*) {};
        vk.begin_rendering = +[](VkCommandBuffer, const VkRenderingInfo*) { ++g_mc.begin_rendering; };
        vk.end_rendering = +[](VkCommandBuffer) { ++g_mc.end_rendering; };
        vk.copy_buffer = +[](VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {};
        vk.copy_buffer_to_image = +[](VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                      const VkBufferImageCopy*) {};
        vk.copy_image_to_buffer = +[](VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                      const VkBufferImageCopy*) {};
        vk.copy_image = +[](VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, std::uint32_t,
                            const VkImageCopy*) {};
        vk.draw_indirect_first_instance = true;
        vk.fill_buffer = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {};
        vk.begin_conditional_rendering = +[](VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {};
        vk.end_conditional_rendering = +[](VkCommandBuffer) {};
        vk.reset_query_pool = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) {};
        vk.begin_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, VkQueryControlFlags) {};
        vk.end_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t) {};
        vk.write_timestamp = +[](VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, std::uint32_t) {};
        vk.copy_query_pool_results = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                         VkDeviceSize, VkDeviceSize, VkQueryResultFlags) {};
        vk.bind_pipeline = &mc_bind_pipeline;
        vk.bind_descriptor_sets = &mc_bind_desc;
        vk.push_constants = &mc_push;
        vk.set_viewport = &mc_viewport;
        vk.set_scissor = &mc_scissor;
        vk.set_viewport_with_count = &mc_viewport_cnt;
        vk.set_scissor_with_count = &mc_scissor_cnt;
        vk.set_primitive_topology = &mc_topology;
        vk.set_primitive_restart_enable = &mc_restart;
        vk.set_stencil_reference = &mc_stencil;
        vk.set_blend_constants = &mc_blend;
        vk.set_depth_bounds = &mc_depth_bounds;
        vk.bind_vertex_buffers2 = &mc_vertex;
        vk.bind_index_buffer = &mc_index;
        vk.bind_robust_index_view = &mc_index_robust;
        vk.draw = &mc_draw;
        vk.draw_indexed = &mc_draw_indexed;
        vk.draw_mesh_tasks = &mc_draw_mesh;
        vk.draw_indirect = &mc_draw_indirect;
        vk.dispatch = &mc_dispatch;
        vk.dispatch_indirect = &mc_dispatch_indirect;
        vk.zero_buffer = fake_handle<VkBuffer>(0x210);
        vk.zero_size = 65536;
        vk.quad_indices = fake_handle<VkBuffer>(0x211);
        vk.quad_index_count = 64;
        vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) {
            return fake_handle<VkImageView>(0x400);
        };
        vk.buffer_at = [](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
            if (va < kGeomVa || va >= kGeomVa + (16ull << 20)) return false;
            buffer = fake_handle<VkBuffer>(0x220);
            offset = va - kGeomVa;
            size = (std::min)(size, std::uint64_t(kGeomVa + (16ull << 20) - va));
            return true;
        };
        vk.graphics_pipeline = [this](std::uintptr_t id) -> const GraphicsPipeline* {
            const auto it = gfx.find(id);
            return it == gfx.end() ? nullptr : &it->second;
        };
        vk.compute_pipeline = [this](std::uintptr_t id) -> const ComputePipeline* {
            const auto it = comp.find(id);
            return it == comp.end() ? nullptr : &it->second;
        };
        vk.query_pool = [this](QueryHeapId id) -> QueryPool* { return id == kOccHeap ? &occ_pool : nullptr; };
        vk.prepare_predicate = [](VkBuffer src, VkDeviceSize off, PredicateCapture& out) {
            out = {src, off, fake_handle<VkBuffer>(0x230), 0, VK_NULL_HANDLE};
            return true;
        };
        vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) {};
        vk.resolve_binary_occlusion = [](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t, VkBuffer,
                                         VkDeviceSize) {};
        vk.copy_scratch = [](VkDeviceSize, VkBuffer& b, VkDeviceSize& off) {
            b = fake_handle<VkBuffer>(0x240);
            off = 0;
            return true;
        };
        vk.prepare_indirect_clamp = [](VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, std::uint32_t,
                                       std::uint32_t stride, std::uint32_t max_count, ClampedIndirect& out) {
            out.buffer = fake_handle<VkBuffer>(0x241);
            out.offset = 0;
            out.records = max_count;
            out.stride = stride;
            return true;
        };
        vk.emit_indirect_clamp = [](VkCommandBuffer, const ClampedIndirect&) {};
        vk.prepare_quad_indices = [](VkBuffer, VkDeviceSize, VkIndexType, std::uint64_t, std::uint32_t quads,
                                     VkBuffer scratch, VkDeviceSize scratch_off, QuadIndexExpansion& out) {
            out.buffer = scratch;
            out.offset = scratch_off;
            out.quads = quads;
            return true;
        };
        vk.emit_quad_indices = [](VkCommandBuffer, const QuadIndexExpansion&) {};
        vk.depth24_prepare = [](Depth24Job& job, std::vector<Depth24Job>& parts) {
            job.scratch = fake_handle<VkBuffer>(0x242);
            job.scratch_offset = 0;
            parts.push_back(job);
            return true;
        };
        vk.depth24_run = [](VkCommandBuffer, const Depth24Job&) {};
    }

    const ResourceBinding* resolve(ResourceId id) const {
        const auto it = res.find(id);
        return it == res.end() ? nullptr : &it->second;
    }

    Draw base_draw(std::uintptr_t pso = 1, bool indexed = true) const {
        Draw d{};
        d.pipeline = pso;
        auto views = std::make_shared<DrawViews>();
        views->render_targets[0] = {kColor, 0, 0, 1};
        views->render_target_count = 1;
        views->vertex_buffers[0] = {kGeomVa, 4096, 16};
        d.views = std::move(views);
        if (indexed) d.index = {kGeomVa + 4096, 1024, 57};  // R16_UINT
        d.topology = 4;  // TRIANGLELIST
        d.viewport = {0, 0, 64, 64, 0, 1};
        d.scissor = {0, 0, 64, 64};
        d.blend_factor = {1, 0.5f, 0.25f, 1};
        d.stencil_front = 3;
        d.stencil_back = 7;
        d.depth_bounds = {0.1f, 0.9f};
        d.indexed = indexed;
        d.count = 6;
        d.instances = 1;
        d.user_data[0] = 0x1111u;
        d.user_data[5] = 0x5555u;
        return d;
    }
};

void test_cpu_elision_and_invalidation() {
    MockEnv env;
    const auto resolve = [&env](ResourceId id) { return env.resolve(id); };
    const VkCommandBuffer cb = fake_handle<VkCommandBuffer>(0x1);

    // 1a: 5 identical indexed draws -> state emitted ONCE, 4 subsequent draws elide every redundant bind/push/dynamic.
    {
        CommandList list;
        const Draw d = env.base_draw(1, true);
        for (int k = 0; k < 5; ++k) CHECK(list.draw(d));
        CHECK(list.close());
        PreparedList prep;
        CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);
        g_mc.reset();
        emit(prep, cb, env.vk);
        CHECK(g_mc.draws == 5);
        CHECK(g_mc.begin_rendering == 1 && g_mc.end_rendering == 1);
        CHECK(g_mc.barriers == 1);
        CHECK(g_mc.bind_gfx == 1);
        CHECK(g_mc.desc_gfx == 1);
        CHECK(g_mc.push == 1 && g_mc.pushes[0].offset == 0 && g_mc.pushes[0].size == sizeof(ShaderUserData));
        CHECK(g_mc.viewport == 1 && g_mc.scissor == 1);
        CHECK(g_mc.topology == 1 && g_mc.restart == 1);
        CHECK(g_mc.stencil_front == 1 && g_mc.stencil_back == 1);
        CHECK(g_mc.blend == 1 && g_mc.depth_bounds == 1);
        CHECK(g_mc.vertex == 1 && g_mc.index == 1);
    }

    // 1b: Partial push-constant dirty range [lo, hi] + individual dynamic state mutations always re-emit.
    {
        CommandList list;
        Draw d = env.base_draw(1, true);
        CHECK(list.draw(d));
        // Mutate ONLY s[5] -> push [20, 4]
        d.user_data[5] = 0x6666u;
        CHECK(list.draw(d));
        // Mutate s[2] and s[4] -> minimal contiguous range [8, 12]
        d.user_data[2] = 0x2222u;
        d.user_data[4] = 0x4444u;
        CHECK(list.draw(d));
        // Mutate viewport, scissor, stencil_back only, blend, depth_bounds, index offset, vertex offset
        d.viewport.width = 32;
        d.scissor.right = 32;
        d.stencil_back = 9;
        d.blend_factor[2] = 0.75f;
        d.depth_bounds[1] = 0.8f;
        d.index.va += 256;
        auto views2 = std::make_shared<DrawViews>(*d.views);
        views2->vertex_buffers[0].va += 128;
        d.views = views2;
        CHECK(list.draw(d));
        CHECK(list.close());

        PreparedList prep;
        CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);
        g_mc.reset();
        emit(prep, cb, env.vk);
        CHECK(g_mc.draws == 4);
        CHECK(g_mc.bind_gfx == 1 && g_mc.desc_gfx == 1);
        CHECK(g_mc.push == 3);
        CHECK(g_mc.pushes[0].offset == 0 && g_mc.pushes[0].size == 144);
        CHECK(g_mc.pushes[1].offset == 20 && g_mc.pushes[1].size == 4 && g_mc.pushes[1].words[5] == 0x6666u);
        CHECK(g_mc.pushes[2].offset == 8 && g_mc.pushes[2].size == 12 && g_mc.pushes[2].words[2] == 0x2222u &&
              g_mc.pushes[2].words[4] == 0x4444u);
        CHECK(g_mc.viewport == 2 && g_mc.scissor == 2);
        CHECK(g_mc.stencil_front == 1 && g_mc.stencil_back == 2);
        CHECK(g_mc.blend == 2 && g_mc.depth_bounds == 2);
        CHECK(g_mc.vertex == 2 && g_mc.index == 2);
    }

    // 1c: Interleaved graphics <-> compute: push-constant bank is shared so full 144B re-pushes on switch,
    //     while graphics & compute pipeline/descriptor bindings remain independently cached.
    {
        CommandList list;
        const Draw d = env.base_draw(1, false);
        Dispatch c{};
        c.pipeline = 1;
        c.x = c.y = c.z = 1;
        c.user_data[0] = 0xAAu;
        CHECK(list.draw(d));
        CHECK(list.dispatch(c));
        CHECK(list.draw(d));      // identical draw after compute: MUST re-push full 144B, skips gfx bind/desc/dyn
        CHECK(list.dispatch(c));  // identical dispatch after gfx: MUST re-push full 144B, skips compute bind/desc
        c.user_data[3] = 0xBBu;
        CHECK(list.dispatch(c));  // consecutive compute: partial push [12, 4]
        CHECK(list.close());

        PreparedList prep;
        CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);
        g_mc.reset();
        emit(prep, cb, env.vk);
        CHECK(g_mc.draws == 2 && g_mc.dispatches == 3);
        CHECK(g_mc.bind_gfx == 1 && g_mc.desc_gfx == 1);
        CHECK(g_mc.bind_comp == 1 && g_mc.desc_comp == 1);
        CHECK(g_mc.push == 5);
        CHECK(g_mc.pushes[0].size == 144 && g_mc.pushes[1].size == 144 && g_mc.pushes[2].size == 144 &&
              g_mc.pushes[3].size == 144);
        CHECK(g_mc.pushes[4].offset == 12 && g_mc.pushes[4].size == 4 && g_mc.pushes[4].words[3] == 0xBBu);
    }

    // 1d: Pipeline dynamic-state disturbance: !depth_bounds, viewport_with_count, and mesh pipelines disturb the
    //     dynamic states they don't declare so the subsequent regular draw re-emits them.
    {
        CommandList list;
        const Draw d1 = env.base_draw(1, false);  // depth_bounds=true, fixed vp, vertex slot 0
        const Draw d2 = env.base_draw(2, false);  // depth_bounds=false -> disturbs depth_bounds
        const Draw d3 = env.base_draw(3, false);  // viewport_with_count=true -> disturbs fixed vp/sc
        Draw d4 = env.base_draw(4, false);        // mesh=true -> disturbs topology, restart, vertex slots
        d4.mesh = true;
        d4.first = 1;  // mesh dispatch group count Z = 1 (non-zero)
        CHECK(list.draw(d1));
        CHECK(list.draw(d2));
        CHECK(list.draw(d1));  // must re-emit depth_bounds
        CHECK(list.draw(d3));  // switches to with_count
        CHECK(list.draw(d1));  // must re-emit fixed viewport/scissor
        CHECK(list.draw(d4));  // mesh: disturbs topology, restart, vertex_buffers
        CHECK(list.draw(d1));  // must re-emit topology, restart, vertex_buffers
        CHECK(list.close());

        PreparedList prep;
        CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);
        g_mc.reset();
        emit(prep, cb, env.vk);
        CHECK(g_mc.draws == 7);
        CHECK(g_mc.depth_bounds == 2);  // d1 (#1), d1 (#3 after d2 disturbed depth_bounds)
        CHECK(g_mc.viewport == 2 && g_mc.viewport_count == 1);  // d1 (#1), d3 (#4 with_count), d1 (#5 after d3)
        CHECK(g_mc.topology == 2 && g_mc.restart == 2 && g_mc.vertex == 2);  // initial + after mesh disturbed them
    }

    // 1e: Helper pass invalidation: capture_predicate, emit_indirect_clamp, emit_quad_indices, depth24_run, and
    //     resolve_binary_occlusion MUST invalidate compute pipeline, compute desc, and shared push constants!
    {
        const auto check_helper_invalidates = [&](const char* name, auto&& inject_helper) {
            CommandList list;
            Dispatch c{};
            c.pipeline = 1;
            c.x = c.y = c.z = 1;
            c.user_data[0] = 0x77u;
            CHECK(list.dispatch(c));
            inject_helper(list);
            CHECK(list.dispatch(c));  // identical dispatch after helper MUST re-bind compute, desc, and full push!
            CHECK(list.close());
            PreparedList prep;
            const ReplayResult pr = prepare(list, resolve, env.vk, prep);
            CHECK(pr == ReplayResult::success);
            if (pr != ReplayResult::success) return;
            QueryReplay qr;
            CHECK(qr.plan(prep) == ReplayResult::success);
            g_mc.reset();
            emit(prep, cb, env.vk);
            const bool ok = g_mc.bind_comp == 2 && g_mc.desc_comp == 2 && g_mc.push >= 2 &&
                            g_mc.pushes.back().offset == 0 && g_mc.pushes.back().size == 144;
            if (!ok) {
                std::fprintf(stderr, "FAIL helper invalidation (%s): bind_comp=%zu desc_comp=%zu push=%zu last_sz=%u\n",
                             name, g_mc.bind_comp, g_mc.desc_comp, g_mc.push,
                             g_mc.pushes.empty() ? 0u : g_mc.pushes.back().size);
                ++g_failures;
            }
        };

        check_helper_invalidates("capture_predicate", [](CommandList& l) {
            CHECK(l.set_predication({MockEnv::kPred, 0, d3d12_predication::equal_zero}));
            CHECK(l.set_predication({0, 0, 0}));
        });
        check_helper_invalidates("emit_indirect_clamp", [&env](CommandList& l) {
            Draw d = env.base_draw(1, false);
            IndirectDrawArgs ind{};
            ind.argument_buffer = MockEnv::kBuf;
            ind.stride = 16;
            ind.max_count = 1;
            CHECK(l.draw_indirect(std::move(d), ind));
        });
        check_helper_invalidates("emit_quad_indices", [&env](CommandList& l) {
            Draw d = env.base_draw(1, true);
            d.topology = topology_quad_list;
            d.count = 8;
            CHECK(l.draw(d));
        });
        check_helper_invalidates("depth24_run", [](CommandList& l) {
            CopyTexture cp{};
            cp.dst = {MockEnv::kTexD24, false, 0, 0, 0, 0, 0, 0};
            cp.src = {MockEnv::kBuf, true, 0, 0, 16, 16, 1, 256};
            CHECK(l.copy_texture(cp));
        });
        check_helper_invalidates("resolve_binary_occlusion", [](CommandList& l) {
            CHECK(l.begin_query({MockEnv::kOccHeap, d3d12_query::binary_occlusion, 0}));
            CHECK(l.end_query({MockEnv::kOccHeap, d3d12_query::binary_occlusion, 0}));
            CHECK(l.resolve_queries({MockEnv::kOccHeap, d3d12_query::binary_occlusion, 0, 1, MockEnv::kResults, 0}));
        });
    }

    // 1f: Cross-list ReplayStateCache sharing (as VulkanQueue::execute_now does across PreparedLists in one CB).
    {
        CommandList l1, l2;
        const Draw d = env.base_draw(1, true);
        CHECK(l1.draw(d) && l1.close());
        CHECK(l2.draw(d) && l2.close());
        PreparedList p1, p2;
        CHECK(prepare(l1, resolve, env.vk, p1) == ReplayResult::success);
        CHECK(prepare(l2, resolve, env.vk, p2) == ReplayResult::success);
        ReplayStateCache shared;
        g_mc.reset();
        emit(p1, cb, env.vk, nullptr, nullptr, &shared);
        emit(p2, cb, env.vk, nullptr, nullptr, &shared);
        CHECK(g_mc.draws == 2 && g_mc.bind_gfx == 1 && g_mc.desc_gfx == 1 && g_mc.push == 1 && g_mc.index == 1);
    }

    std::puts("replay_state_cache CPU mock elision & invalidation: ok");
}

// ---------------------------------------------------------------------------------------------------------------
// Part 2: 5,000-draw throughput check (< 1.0 us/draw)

void test_throughput_5000_draws() {
    MockEnv env;
    // Swap to zero-overhead no-op callbacks (without vector push_back on pushes) for clean timing.
    env.vk.pipeline_barrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                  std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                  std::uint32_t, const VkImageMemoryBarrier*) {};
    env.vk.begin_rendering = +[](VkCommandBuffer, const VkRenderingInfo*) {};
    env.vk.end_rendering = +[](VkCommandBuffer) {};
    env.vk.bind_pipeline = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {};
    env.vk.bind_descriptor_sets = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t,
                                      std::uint32_t, const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {};
    env.vk.push_constants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                                const void*) {};
    env.vk.set_viewport = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkViewport*) {};
    env.vk.set_scissor = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkRect2D*) {};
    env.vk.set_primitive_topology = +[](VkCommandBuffer, VkPrimitiveTopology) {};
    env.vk.set_primitive_restart_enable = +[](VkCommandBuffer, VkBool32) {};
    env.vk.set_stencil_reference = +[](VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {};
    env.vk.set_blend_constants = +[](VkCommandBuffer, const float*) {};
    env.vk.set_depth_bounds = +[](VkCommandBuffer, float, float) {};
    env.vk.bind_vertex_buffers2 = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*,
                                      const VkDeviceSize*, const VkDeviceSize*, const VkDeviceSize*) {};
    env.vk.bind_index_buffer = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {};
    env.vk.draw_indexed = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                              std::uint32_t) {};

    constexpr std::uint32_t kDraws = 5000;
    CommandList list;
    Draw d = env.base_draw(1, true);
    for (std::uint32_t i = 0; i < kDraws; ++i) {
        // Realistic mix: same pipeline/viewport/scissor, 1 push-constant word changes every draw.
        d.user_data[0] = i;
        CHECK(list.draw(d));
    }
    CHECK(list.close());

    PreparedList prep;
    const auto resolve = [&env](ResourceId id) { return env.resolve(id); };
    CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);

    const VkCommandBuffer cb = fake_handle<VkCommandBuffer>(0x1);
    for (int w = 0; w < 5; ++w) emit(prep, cb, env.vk);

    constexpr int kReps = 21;
    std::vector<double> us_per_draw;
    us_per_draw.reserve(kReps);
    for (int r = 0; r < kReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        emit(prep, cb, env.vk);
        const auto t1 = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        us_per_draw.push_back(us / double(kDraws));
    }
    std::sort(us_per_draw.begin(), us_per_draw.end());
    const double median_us = us_per_draw[us_per_draw.size() / 2];
    std::printf("replay_state_cache throughput (%u draws): %.4f us/draw (%.2f ms total)\n", kDraws, median_us,
                median_us * double(kDraws) / 1000.0);
    CHECK(median_us < 1.0);
}

// ---------------------------------------------------------------------------------------------------------------
// Part 3: Live Vulkan GPU validation test (real graphics + compute + partial push + clear + interleave)

// Hand-assembled SPIR-V 1.3 vertex shader:
//   layout(location = 0) in vec2 in_pos;
//   void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
std::vector<std::uint32_t> build_vs_spirv() {
    std::vector<std::uint32_t> w;
    const auto op = [&](std::uint16_t code, std::initializer_list<std::uint32_t> args) {
        w.push_back((std::uint32_t(args.size() + 1) << 16) | code);
        for (std::uint32_t a : args) w.push_back(a);
    };
    constexpr std::uint32_t kVoid = 1, kFn = 2, kF32 = 3, kVec2 = 4, kVec4 = 5;
    constexpr std::uint32_t kPtrInVec2 = 6, kInPos = 7;
    constexpr std::uint32_t kPerVertex = 8, kPtrOutPV = 9, kOutPV = 10, kPtrOutVec4 = 11;
    constexpr std::uint32_t kI32 = 12, kI0 = 13, kF0 = 14, kF1 = 15;
    constexpr std::uint32_t kMain = 16, kLabel = 17, kLoadPos = 18, kX = 19, kY = 20, kPos4 = 21, kPtrPos = 22;
    constexpr std::uint32_t kBound = 23;

    w.insert(w.end(), {0x07230203u, 0x00010300u, 0u, kBound, 0u});
    op(17, {1u});                                       // OpCapability Shader
    op(14, {0u, 1u});                                   // OpMemoryModel Logical GLSL450
    op(15, {0u, kMain, 0x6e69616du, 0u, kOutPV, kInPos});  // OpEntryPoint Vertex %main "main"
    op(72, {kPerVertex, 0u, 11u, 0u});                  // MemberDecorate PerVertex 0 BuiltIn Position(0)
    op(71, {kPerVertex, 2u});                           // Decorate PerVertex Block
    op(71, {kInPos, 30u, 0u});                          // Decorate in_pos Location 0

    op(19, {kVoid});
    op(33, {kFn, kVoid});
    op(22, {kF32, 32u});
    op(23, {kVec2, kF32, 2u});
    op(23, {kVec4, kF32, 4u});
    op(32, {kPtrInVec2, 1u, kVec2});                    // Input(1)
    op(59, {kPtrInVec2, kInPos, 1u});
    op(30, {kPerVertex, kVec4});
    op(32, {kPtrOutPV, 3u, kPerVertex});                // Output(3)
    op(59, {kPtrOutPV, kOutPV, 3u});
    op(32, {kPtrOutVec4, 3u, kVec4});
    op(21, {kI32, 32u, 1u});
    op(43, {kI32, kI0, 0u});
    op(43, {kF32, kF0, float_bits(0.0f)});
    op(43, {kF32, kF1, float_bits(1.0f)});

    op(54, {kVoid, kMain, 0u, kFn});
    op(248, {kLabel});
    op(61, {kVec2, kLoadPos, kInPos});
    op(81, {kF32, kX, kLoadPos, 0u});
    op(81, {kF32, kY, kLoadPos, 1u});
    op(80, {kVec4, kPos4, kX, kY, kF0, kF1});
    op(65, {kPtrOutVec4, kPtrPos, kOutPV, kI0});
    op(62, {kPtrPos, kPos4});
    op(253, {});
    op(56, {});
    return w;
}

// Hand-assembled SPIR-V 1.3 fragment shader:
//   layout(push_constant) uniform PC { float r, g, b, a; uint pad[4]; float scale; } pc;
//   layout(location = 0) out vec4 out_color;
//   void main() { out_color = vec4(pc.r * pc.scale, pc.g * pc.scale, pc.b * pc.scale, pc.a); }
// Reads BOTH word 0..3 (offset 0..15) AND word 8 (offset 32) so partial push updates to either region are verified!
std::vector<std::uint32_t> build_fs_spirv() {
    std::vector<std::uint32_t> w;
    const auto op = [&](std::uint16_t code, std::initializer_list<std::uint32_t> args) {
        w.push_back((std::uint32_t(args.size() + 1) << 16) | code);
        for (std::uint32_t a : args) w.push_back(a);
    };
    constexpr std::uint32_t kVoid = 1, kFn = 2, kF32 = 3, kVec4 = 4, kI32 = 5;
    constexpr std::uint32_t kPtrOutVec4 = 6, kOutColor = 7;
    constexpr std::uint32_t kPcStruct = 8, kPtrPcStruct = 9, kPcVar = 10, kPtrPcF32 = 11;
    constexpr std::uint32_t kI0 = 12, kI1 = 13, kI2 = 14, kI3 = 15, kI4 = 16;
    constexpr std::uint32_t kMain = 17, kLabel = 18;
    constexpr std::uint32_t kPr = 19, kPg = 20, kPb = 21, kPa = 22, kPs = 23;
    constexpr std::uint32_t kR = 24, kG = 25, kB = 26, kA = 27, kS = 28;
    constexpr std::uint32_t kRs = 29, kGs = 30, kBs = 31, kCol = 32;
    constexpr std::uint32_t kBound = 33;

    w.insert(w.end(), {0x07230203u, 0x00010300u, 0u, kBound, 0u});
    op(17, {1u});                                       // OpCapability Shader
    op(14, {0u, 1u});                                   // OpMemoryModel Logical GLSL450
    op(15, {4u, kMain, 0x6e69616du, 0u, kOutColor});    // OpEntryPoint Fragment %main "main"
    op(16, {kMain, 7u});                                // OriginUpperLeft(7)
    op(71, {kOutColor, 30u, 0u});                       // Location 0
    op(71, {kPcStruct, 2u});                            // Block
    op(72, {kPcStruct, 0u, 35u, 0u});                   // Offset 0  (user_data[0]: r)
    op(72, {kPcStruct, 1u, 35u, 4u});                   // Offset 4  (user_data[1]: g)
    op(72, {kPcStruct, 2u, 35u, 8u});                   // Offset 8  (user_data[2]: b)
    op(72, {kPcStruct, 3u, 35u, 12u});                  // Offset 12 (user_data[3]: a)
    op(72, {kPcStruct, 4u, 35u, 32u});                  // Offset 32 (user_data[8]: scale)

    op(19, {kVoid});
    op(33, {kFn, kVoid});
    op(22, {kF32, 32u});
    op(23, {kVec4, kF32, 4u});
    op(21, {kI32, 32u, 1u});
    op(32, {kPtrOutVec4, 3u, kVec4});                   // Output(3)
    op(59, {kPtrOutVec4, kOutColor, 3u});
    op(30, {kPcStruct, kF32, kF32, kF32, kF32, kF32});
    op(32, {kPtrPcStruct, 9u, kPcStruct});              // PushConstant(9)
    op(59, {kPtrPcStruct, kPcVar, 9u});
    op(32, {kPtrPcF32, 9u, kF32});
    op(43, {kI32, kI0, 0u});
    op(43, {kI32, kI1, 1u});
    op(43, {kI32, kI2, 2u});
    op(43, {kI32, kI3, 3u});
    op(43, {kI32, kI4, 4u});

    op(54, {kVoid, kMain, 0u, kFn});
    op(248, {kLabel});
    op(65, {kPtrPcF32, kPr, kPcVar, kI0});
    op(65, {kPtrPcF32, kPg, kPcVar, kI1});
    op(65, {kPtrPcF32, kPb, kPcVar, kI2});
    op(65, {kPtrPcF32, kPa, kPcVar, kI3});
    op(65, {kPtrPcF32, kPs, kPcVar, kI4});
    op(61, {kF32, kR, kPr});
    op(61, {kF32, kG, kPg});
    op(61, {kF32, kB, kPb});
    op(61, {kF32, kA, kPa});
    op(61, {kF32, kS, kPs});
    op(133, {kF32, kRs, kR, kS});                       // OpFMul
    op(133, {kF32, kGs, kG, kS});
    op(133, {kF32, kBs, kB, kS});
    op(80, {kVec4, kCol, kRs, kGs, kBs, kA});
    op(62, {kOutColor, kCol});
    op(253, {});
    op(56, {});
    return w;
}

// Hand-assembled SPIR-V 1.3 compute shader:
//   layout(set = 0, binding = 0) buffer Out { uint vals[4]; } out_buf;
//   layout(push_constant) uniform PC { uint slot; uint val; uint pad[3]; uint xor_k; } pc;
//   void main() { out_buf.vals[pc.slot] = pc.val ^ pc.xor_k; }
std::vector<std::uint32_t> build_cs_spirv() {
    std::vector<std::uint32_t> w;
    const auto op = [&](std::uint16_t code, std::initializer_list<std::uint32_t> args) {
        w.push_back((std::uint32_t(args.size() + 1) << 16) | code);
        for (std::uint32_t a : args) w.push_back(a);
    };
    constexpr std::uint32_t kVoid = 1, kFn = 2, kU32 = 3, kI32 = 4, kC4 = 5, kArr = 6, kBufS = 7;
    constexpr std::uint32_t kPtrBufS = 8, kBufVar = 9, kPtrBufU32 = 10;
    constexpr std::uint32_t kPcS = 11, kPtrPcS = 12, kPcVar = 13, kPtrPcU32 = 14;
    constexpr std::uint32_t kI0 = 15, kI1 = 16, kI2 = 17;
    constexpr std::uint32_t kMain = 18, kLabel = 19;
    constexpr std::uint32_t kPslot = 20, kPval = 21, kPxor = 22, kSlot = 23, kVal = 24, kXor = 25, kRes = 26, kDst = 27;
    constexpr std::uint32_t kBound = 28;

    w.insert(w.end(), {0x07230203u, 0x00010300u, 0u, kBound, 0u});
    op(17, {1u});                                       // OpCapability Shader
    op(14, {0u, 1u});                                   // OpMemoryModel Logical GLSL450
    op(15, {5u, kMain, 0x6e69616du, 0u});               // OpEntryPoint GLCompute %main "main"
    op(16, {kMain, 17u, 1u, 1u, 1u});                   // LocalSize 1 1 1

    op(71, {kArr, 6u, 4u});                             // ArrayStride 4
    op(72, {kBufS, 0u, 35u, 0u});
    op(71, {kBufS, 2u});                                // Block
    op(71, {kBufVar, 34u, 0u});                         // DescriptorSet 0
    op(71, {kBufVar, 33u, 0u});                         // Binding 0
    op(71, {kPcS, 2u});                                 // Block
    op(72, {kPcS, 0u, 35u, 0u});                        // Offset 0  (user_data[0]: slot)
    op(72, {kPcS, 1u, 35u, 4u});                        // Offset 4  (user_data[1]: val)
    op(72, {kPcS, 2u, 35u, 20u});                       // Offset 20 (user_data[5]: xor_k)

    op(19, {kVoid});
    op(33, {kFn, kVoid});
    op(21, {kU32, 32u, 0u});
    op(21, {kI32, 32u, 1u});
    op(43, {kU32, kC4, 4u});
    op(28, {kArr, kU32, kC4});
    op(30, {kBufS, kArr});
    op(32, {kPtrBufS, 12u, kBufS});                     // StorageBuffer(12)
    op(59, {kPtrBufS, kBufVar, 12u});
    op(32, {kPtrBufU32, 12u, kU32});
    op(30, {kPcS, kU32, kU32, kU32});
    op(32, {kPtrPcS, 9u, kPcS});                        // PushConstant(9)
    op(59, {kPtrPcS, kPcVar, 9u});
    op(32, {kPtrPcU32, 9u, kU32});
    op(43, {kI32, kI0, 0u});
    op(43, {kI32, kI1, 1u});
    op(43, {kI32, kI2, 2u});

    op(54, {kVoid, kMain, 0u, kFn});
    op(248, {kLabel});
    op(65, {kPtrPcU32, kPslot, kPcVar, kI0});
    op(65, {kPtrPcU32, kPval, kPcVar, kI1});
    op(65, {kPtrPcU32, kPxor, kPcVar, kI2});
    op(61, {kU32, kSlot, kPslot});
    op(61, {kU32, kVal, kPval});
    op(61, {kU32, kXor, kPxor});
    op(198, {kU32, kRes, kVal, kXor});                  // OpBitwiseXor
    op(65, {kPtrBufU32, kDst, kBufVar, kI0, kSlot});
    op(62, {kDst, kRes});
    op(253, {});
    op(56, {});
    return w;
}

void test_gpu_live_validation() {
    vktest::Device dev;
    const auto status = dev.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::puts("SKIP: no Vulkan device for live replay_state_cache GPU test");
        return;
    }
    CHECK(status == vktest::DeviceStatus::ready);
    if (status != vktest::DeviceStatus::ready) return;

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
    const auto vkCreateGraphicsPipelines = dev.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines");
    const auto vkCreateComputePipelines = dev.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    const auto vkDestroyPipeline = dev.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");

    constexpr std::uint32_t kW = 8, kH = 8, kPitch = 256;
    const VkImage color_img =
        dev.image(VK_FORMAT_R8G8B8A8_UNORM, kW, kH, 1,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    void* rb_pixels = nullptr;
    const VkBuffer rb_buf = dev.buffer(kH * kPitch, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb_pixels);
    void* ssbo_mapped = nullptr;
    const VkBuffer ssbo = dev.buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &ssbo_mapped);
    void* vb_mapped = nullptr;
    const VkBuffer vb = dev.buffer(256, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb_mapped);
    void* ib_mapped = nullptr;
    const VkBuffer ib = dev.buffer(256, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &ib_mapped);
    CHECK(color_img && rb_buf && rb_pixels && ssbo && ssbo_mapped && vb && vb_mapped && ib && ib_mapped);
    if (!color_img || !rb_buf || !ssbo || !vb || !ib) return;

    std::memset(rb_pixels, 0, kH * kPitch);
    std::memset(ssbo_mapped, 0, 64);

    // Full-screen quad vertices (4 vertices, vec2) + two index slices (same triangles at offset 0 and offset 16).
    const float verts[8] = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
    std::memcpy(vb_mapped, verts, sizeof(verts));
    const std::uint16_t indices[12] = {0, 1, 2, 0, 2, 3, 0, 0, 0, 1, 2, 3};
    std::memcpy(ib_mapped, indices, sizeof(indices));

    // Descriptor set 0 for compute SSBO (and empty set layout for graphics).
    const VkDescriptorSetLayoutBinding b0{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dsl_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl_ci.bindingCount = 1;
    dsl_ci.pBindings = &b0;
    VkDescriptorSetLayout comp_dsl = VK_NULL_HANDLE;
    CHECK(vkCreateDescriptorSetLayout(dev.device, &dsl_ci, nullptr, &comp_dsl) == VK_SUCCESS);

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
    ds_ai.pSetLayouts = &comp_dsl;
    VkDescriptorSet comp_ds = VK_NULL_HANDLE;
    CHECK(vkAllocateDescriptorSets(dev.device, &ds_ai, &comp_ds) == VK_SUCCESS);

    const VkDescriptorBufferInfo dbi{ssbo, 0, 64};
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet = comp_ds;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &dbi;
    vkUpdateDescriptorSets(dev.device, 1, &wds, 0, nullptr);

    // Pipeline layouts.
    const VkPushConstantRange gfx_pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                      sizeof(ShaderUserData)};
    VkPipelineLayoutCreateInfo gpl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    gpl_ci.pushConstantRangeCount = 1;
    gpl_ci.pPushConstantRanges = &gfx_pcr;
    VkPipelineLayout gfx_layout = VK_NULL_HANDLE;
    CHECK(vkCreatePipelineLayout(dev.device, &gpl_ci, nullptr, &gfx_layout) == VK_SUCCESS);

    const VkPushConstantRange comp_pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderUserData)};
    VkPipelineLayoutCreateInfo cpl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    cpl_ci.setLayoutCount = 1;
    cpl_ci.pSetLayouts = &comp_dsl;
    cpl_ci.pushConstantRangeCount = 1;
    cpl_ci.pPushConstantRanges = &comp_pcr;
    VkPipelineLayout comp_layout = VK_NULL_HANDLE;
    CHECK(vkCreatePipelineLayout(dev.device, &cpl_ci, nullptr, &comp_layout) == VK_SUCCESS);

    // Shader modules.
    const auto vs_spv = build_vs_spirv();
    const auto fs_spv = build_fs_spirv();
    const auto cs_spv = build_cs_spirv();
    const auto make_mod = [&](const std::vector<std::uint32_t>& spv) {
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = spv.size() * sizeof(std::uint32_t);
        ci.pCode = spv.data();
        VkShaderModule m = VK_NULL_HANDLE;
        CHECK(vkCreateShaderModule(dev.device, &ci, nullptr, &m) == VK_SUCCESS);
        return m;
    };
    const VkShaderModule vs_mod = make_mod(vs_spv);
    const VkShaderModule fs_mod = make_mod(fs_spv);
    const VkShaderModule cs_mod = make_mod(cs_spv);

    // Graphics pipeline (dynamic rendering, dynamic viewport/scissor/topology/restart/stencil/blend/vertex_stride).
    VkPipelineShaderStageCreateInfo stages[2]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
    };
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_mod;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_mod;
    stages[1].pName = "main";

    VkVertexInputBindingDescription vbind{0, 8, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription vattr{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &vbind;
    vi.vertexAttributeDescriptionCount = 1;
    vi.pVertexAttributeDescriptions = &vattr;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb_state{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb_state.attachmentCount = 1;
    cb_state.pAttachments = &cba;

    const VkDynamicState dyn_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
        VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE,
    };
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = static_cast<std::uint32_t>(std::size(dyn_states));
    dyn.pDynamicStates = dyn_states;

    const VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkPipelineRenderingCreateInfo rend{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rend.colorAttachmentCount = 1;
    rend.pColorAttachmentFormats = &color_fmt;

    VkGraphicsPipelineCreateInfo gp_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rend};
    gp_ci.stageCount = 2;
    gp_ci.pStages = stages;
    gp_ci.pVertexInputState = &vi;
    gp_ci.pInputAssemblyState = &ia;
    gp_ci.pViewportState = &vp;
    gp_ci.pRasterizationState = &rs;
    gp_ci.pMultisampleState = &ms;
    gp_ci.pColorBlendState = &cb_state;
    gp_ci.pDynamicState = &dyn;
    gp_ci.layout = gfx_layout;
    VkPipeline gfx_pipe = VK_NULL_HANDLE;
    CHECK(vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &gp_ci, nullptr, &gfx_pipe) == VK_SUCCESS);

    // Compute pipeline.
    VkComputePipelineCreateInfo cp_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp_ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cp_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_ci.stage.module = cs_mod;
    cp_ci.stage.pName = "main";
    cp_ci.layout = comp_layout;
    VkPipeline comp_pipe = VK_NULL_HANDLE;
    CHECK(vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &cp_ci, nullptr, &comp_pipe) == VK_SUCCESS);

    // Second graphics pipeline with depth-only, and third with depth+stencil (tests depth_aspects scope split on GPU).
    const VkImage depth_img =
        dev.image(VK_FORMAT_D32_SFLOAT_S8_UINT, kW, kH, 1,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkPipeline gfx_depth_pipe = VK_NULL_HANDLE;
    VkPipeline gfx_depth_stencil_pipe = VK_NULL_HANDLE;
    {
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineRenderingCreateInfo rend_d = rend;
        rend_d.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkGraphicsPipelineCreateInfo gp_d = gp_ci;
        gp_d.pNext = &rend_d;
        gp_d.pDepthStencilState = &ds;
        CHECK(vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &gp_d, nullptr, &gfx_depth_pipe) == VK_SUCCESS);

        VkPipelineRenderingCreateInfo rend_ds = rend_d;
        rend_ds.stencilAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkGraphicsPipelineCreateInfo gp_ds = gp_d;
        gp_ds.pNext = &rend_ds;
        CHECK(vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &gp_ds, nullptr, &gfx_depth_stencil_pipe) ==
              VK_SUCCESS);
    }

    if (gfx_pipe && comp_pipe && depth_img && gfx_depth_pipe && gfx_depth_stencil_pipe) {
        VulkanContext ctx(dev.device_proc, dev.device, true, dev.conditional_rendering, dev.buffer_device_address);
        CHECK(ctx.loaded());

        constexpr ResourceId kRtId = 1, kVbId = 2, kIbId = 3, kRbId = 4, kDsId = 5;
        constexpr std::uint64_t kVbVa = 0x10000ull, kIbVa = 0x20000ull;
        ResourceBinding rt_b{};
        rt_b.image = color_img;
        rt_b.format = VK_FORMAT_R8G8B8A8_UNORM;
        rt_b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        rt_b.width = kW;
        rt_b.height = kH;
        rt_b.color_attachment = true;
        ctx.add_resource(kRtId, rt_b, VK_IMAGE_LAYOUT_UNDEFINED);

        ResourceBinding ds_b{};
        ds_b.image = depth_img;
        ds_b.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        ds_b.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        ds_b.width = kW;
        ds_b.height = kH;
        ctx.add_resource(kDsId, ds_b, VK_IMAGE_LAYOUT_UNDEFINED);

        ResourceBinding vb_b{};
        vb_b.buffer = vb;
        vb_b.size = 256;
        vb_b.va = kVbVa;
        ctx.add_resource(kVbId, vb_b);

        ResourceBinding ib_b{};
        ib_b.buffer = ib;
        ib_b.size = 256;
        ib_b.va = kIbVa;
        ctx.add_resource(kIbId, ib_b);

        ResourceBinding rb_b{};
        rb_b.buffer = rb_buf;
        rb_b.size = kH * kPitch;
        rb_b.host_readable = true;
        ctx.add_resource(kRbId, rb_b);

        GraphicsPipeline gp{};
        gp.pipeline = gfx_pipe;
        gp.layout = gfx_layout;
        gp.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        gp.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
        gp.vertex_slots = 0b1;
        ctx.add_graphics_pipeline(100, gp);

        GraphicsPipeline gp_glc = gp;
        gp_glc.reads_glc_buffer = true;
        ctx.add_graphics_pipeline(101, gp_glc);

        GraphicsPipeline gp_d = gp;
        gp_d.pipeline = gfx_depth_pipe;
        gp_d.depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        gp_d.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
        ctx.add_graphics_pipeline(102, gp_d);

        GraphicsPipeline gp_ds = gp_d;
        gp_ds.pipeline = gfx_depth_stencil_pipe;
        gp_ds.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        ctx.add_graphics_pipeline(103, gp_ds);

        ComputePipeline cp{};
        cp.pipeline = comp_pipe;
        cp.layout = comp_layout;
        cp.bindless_set = comp_ds;
        ctx.add_pipeline(200, cp);

        VulkanQueue queue(ctx, dev.queue, dev.family, VulkanQueue::QueueKind::direct);
        CHECK(queue.ok());

        // List 1: Clear -> Draw 1 (red, left half) -> Draw 2 (identical, elided) -> Draw 3 (partial push: change ONLY
        //         g=1.0 -> yellow on top-right) -> Compute 1 & Compute 2 (partial push) -> Draw 4 (identical color to
        //         Draw 3 but on bottom-right, proving compute invalidated push constants and Draw 4 re-pushed!).
        CommandList list1;
        CHECK(list1.transition({kRtId, all_subresources, d3d12_state::common, d3d12_state::render_target}));
        CHECK(list1.clear_render_target({kRtId, 0, 0, 1}, {0.0f, 0.0f, 1.0f, 1.0f}));

        auto make_quad = [&](float r, float g, float b, float scale, std::int32_t x0, std::int32_t y0, std::int32_t x1,
                             std::int32_t y1) {
            Draw d{};
            d.pipeline = 100;
            auto views = std::make_shared<DrawViews>();
            views->render_targets[0] = {kRtId, 0, 0, 1};
            views->render_target_count = 1;
            views->vertex_buffers[0] = {kVbVa, 256, 8};
            d.views = std::move(views);
            d.index = {kIbVa, 12, 57};  // R16_UINT
            d.indexed = true;
            d.topology = 4;
            d.viewport = {0.0f, 0.0f, float(kW), float(kH), 0.0f, 1.0f};
            d.scissor = {x0, y0, x1, y1};
            d.blend_factor = {1, 1, 1, 1};
            d.count = 6;
            d.instances = 1;
            d.user_data[0] = float_bits(r);
            d.user_data[1] = float_bits(g);
            d.user_data[2] = float_bits(b);
            d.user_data[3] = float_bits(1.0f);
            d.user_data[8] = float_bits(scale);
            return d;
        };

        // Top-left [0..4, 0..4): red (1, 0, 0, 1) * 1.0 = (255, 0, 0, 255)
        const Draw d_red = make_quad(1.0f, 0.0f, 0.0f, 1.0f, 0, 0, 4, 4);
        CHECK(list1.draw(d_red));
        CHECK(list1.draw(d_red));  // identical draw (state elided)

        // Top-right [4..8, 0..4): change ONLY user_data[1] (g = 1.0) & scissor -> yellow (255, 255, 0, 255)
        const Draw d_yellow = make_quad(1.0f, 1.0f, 0.0f, 1.0f, 4, 0, 8, 4);
        CHECK(list1.draw(d_yellow));

        // Interleaved compute dispatches:
        //   slot 0: 0x12340000 ^ 0x000000A5 = 0x123400A5
        //   slot 1 (partial push: slot + val change, xor_k unchanged): 0x56780000 ^ 0x000000A5 = 0x567800A5
        //   slot 2 (partial push: slot + xor_k change, val unchanged): 0x56780000 ^ 0x0000005A = 0x5678005A
        Dispatch c{};
        c.pipeline = 200;
        c.x = c.y = c.z = 1;
        c.user_data[0] = 0;
        c.user_data[1] = 0x12340000u;
        c.user_data[5] = 0x000000A5u;
        CHECK(list1.dispatch(c));
        c.user_data[0] = 1;
        c.user_data[1] = 0x56780000u;
        CHECK(list1.dispatch(c));
        c.user_data[0] = 2;
        c.user_data[5] = 0x0000005Au;
        CHECK(list1.dispatch(c));

        // Bottom-right [4..8, 4..8): SAME user_data as d_yellow after compute -> MUST re-push full push constants!
        const Draw d_yellow_br = make_quad(1.0f, 1.0f, 0.0f, 1.0f, 4, 4, 8, 8);
        CHECK(list1.draw(d_yellow_br));

        // Bottom-left [0..4, 4..8): change ONLY user_data[8] (scale = 0.0 -> black (0, 0, 0, 255))!
        const Draw d_black_bl = make_quad(1.0f, 1.0f, 0.0f, 0.0f, 0, 4, 4, 8);
        CHECK(list1.draw(d_black_bl));

        // Consecutive reads_glc_buffer draw on the same RT (must close open scope for ALL_GRAPHICS_BIT barrier),
        // followed by depth-only -> depth+stencil draws on the same D32S8 target (must split on depth_aspects!).
        Draw d_glc_br = make_quad(1.0f, 1.0f, 0.0f, 1.0f, 4, 4, 8, 8);
        d_glc_br.pipeline = 101;  // reads_glc_buffer = true
        CHECK(list1.draw(d_glc_br));

        CHECK(list1.transition({kDsId, all_subresources, d3d12_state::common, d3d12_state::depth_write}));
        Draw d_depth_only = make_quad(1.0f, 0.0f, 0.0f, 1.0f, 0, 0, 4, 4);
        d_depth_only.pipeline = 102;  // depth_aspects = DEPTH_BIT
        d_depth_only.depth = {kDsId, 0, 0, 1};
        CHECK(list1.draw(d_depth_only));
        Draw d_depth_stencil = make_quad(1.0f, 0.0f, 0.0f, 1.0f, 0, 0, 4, 4);
        d_depth_stencil.pipeline = 103;  // depth_aspects = DEPTH_BIT | STENCIL_BIT
        d_depth_stencil.depth = {kDsId, 0, 0, 1};
        CHECK(list1.draw(d_depth_stencil));
        CHECK(list1.draw(d_depth_stencil));  // merged into same depth+stencil scope

        // Read back color target into rb_buf.
        CHECK(list1.transition({kRtId, all_subresources, d3d12_state::render_target, d3d12_state::copy_source}));
        CopyTexture readback{};
        readback.dst = {kRbId, true, 0, 0, kW, kH, 1, kPitch};
        readback.src = {kRtId, false, 0, 0, 0, 0, 0, 0};
        CHECK(list1.copy_texture(readback));
        CHECK(list1.close());

        const CommandList* batch[1] = {&list1};
        CHECK(queue.execute(batch) == ReplayResult::success);
        CHECK(queue.idle());

        // Verify SSBO compute outputs.
        const auto* out_u32 = static_cast<const std::uint32_t*>(ssbo_mapped);
        CHECK(out_u32[0] == 0x123400A5u);
        CHECK(out_u32[1] == 0x567800A5u);
        CHECK(out_u32[2] == 0x5678005Au);

        // Verify color quadrants:
        //   TL (1,1): red (255,0,0,255)
        //   TR (6,1): yellow (255,255,0,255)
        //   BR (6,6): yellow (255,255,0,255)  <-- proves graphics push constants re-pushed after compute!
        //   BL (1,6): black (0,0,0,255)       <-- proves partial push update to user_data[8] (scale=0.0)!
        const auto* px = static_cast<const std::uint8_t*>(rb_pixels);
        const auto check_px = [&](std::uint32_t x, std::uint32_t y, std::uint8_t r, std::uint8_t g, std::uint8_t b,
                                  std::uint8_t a) {
            const std::size_t off = std::size_t(y) * kPitch + std::size_t(x) * 4;
            const bool ok = px[off] == r && px[off + 1] == g && px[off + 2] == b && px[off + 3] == a;
            if (!ok)
                std::fprintf(stderr, "FAIL pixel (%u,%u): got (%u,%u,%u,%u) want (%u,%u,%u,%u)\n", x, y,
                             unsigned(px[off]), unsigned(px[off + 1]), unsigned(px[off + 2]), unsigned(px[off + 3]),
                             unsigned(r), unsigned(g), unsigned(b), unsigned(a));
            CHECK(ok);
        };
        check_px(1, 1, 255, 0, 0, 255);
        check_px(6, 1, 255, 255, 0, 255);
        check_px(6, 6, 255, 255, 0, 255);
        check_px(1, 6, 0, 0, 0, 255);
        std::puts("replay_state_cache live GPU validation (graphics + partial push + compute interleave): ok");
    }

    if (gfx_depth_stencil_pipe) vkDestroyPipeline(dev.device, gfx_depth_stencil_pipe, nullptr);
    if (gfx_depth_pipe) vkDestroyPipeline(dev.device, gfx_depth_pipe, nullptr);
    if (gfx_pipe) vkDestroyPipeline(dev.device, gfx_pipe, nullptr);
    if (comp_pipe) vkDestroyPipeline(dev.device, comp_pipe, nullptr);
    if (vs_mod) vkDestroyShaderModule(dev.device, vs_mod, nullptr);
    if (fs_mod) vkDestroyShaderModule(dev.device, fs_mod, nullptr);
    if (cs_mod) vkDestroyShaderModule(dev.device, cs_mod, nullptr);
    if (gfx_layout) vkDestroyPipelineLayout(dev.device, gfx_layout, nullptr);
    if (comp_layout) vkDestroyPipelineLayout(dev.device, comp_layout, nullptr);
    if (dp) vkDestroyDescriptorPool(dev.device, dp, nullptr);
    if (comp_dsl) vkDestroyDescriptorSetLayout(dev.device, comp_dsl, nullptr);
    CHECK(dev.validation_errors() == 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Part 4: Batch-parallel synchronous pipeline compile (collect_unbuilt_pipelines + set_batch_compile_runner +
//         closed/cancel teardown race safety)

struct MiniCompilePool {
    struct Entry {
        const void* owner = nullptr;
        std::function<void()> fn;
    };
    std::mutex mutex;
    std::condition_variable cv;
    std::condition_variable done_cv;
    std::deque<Entry> queue;
    std::vector<const void*> running_owners;
    std::vector<std::thread> threads;
    bool closed = false;

    explicit MiniCompilePool(std::size_t n) {
        threads.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            threads.emplace_back([this] {
                for (;;) {
                    Entry entry;
                    {
                        std::unique_lock lock(mutex);
                        cv.wait(lock, [&] { return closed || !queue.empty(); });
                        if (queue.empty()) return;
                        entry = std::move(queue.front());
                        queue.pop_front();
                        running_owners.push_back(entry.owner);
                    }
                    entry.fn();
                    {
                        std::lock_guard lock(mutex);
                        const auto it = std::find(running_owners.begin(), running_owners.end(), entry.owner);
                        if (it != running_owners.end()) running_owners.erase(it);
                    }
                    done_cv.notify_all();
                }
            });
        }
    }
    ~MiniCompilePool() {
        close_and_drop();
        for (auto& t : threads)
            if (t.joinable()) t.join();
    }
    bool submit(const void* owner, std::function<void()> fn) {
        std::lock_guard lock(mutex);
        if (closed) return false;
        queue.push_back({owner, std::move(fn)});
        cv.notify_one();
        return true;
    }
    // Mirrors PipelineCompiler::cancel(owner): drops unstarted queued jobs for `owner` outside the lock and waits
    // only if `owner` is currently running (never waits on unrelated busy workers).
    std::size_t cancel(const void* owner) {
        std::vector<Entry> dropped;
        {
            std::unique_lock lock(mutex);
            for (;;) {
                std::deque<Entry> kept;
                for (auto& e : queue) {
                    if (e.owner == owner) dropped.push_back(std::move(e));
                    else kept.push_back(std::move(e));
                }
                queue.swap(kept);
                if (std::find(running_owners.begin(), running_owners.end(), owner) == running_owners.end()) break;
                done_cv.wait(lock);
            }
        }
        return dropped.size();  // destroys dropped captures outside the lock
    }
    void close_and_drop() {
        std::deque<Entry> dropped;
        {
            std::lock_guard lock(mutex);
            closed = true;
            dropped.swap(queue);
        }
        cv.notify_all();
        dropped.clear();  // mirrors PipelineCompiler::drop_locked(): destroys queued lambdas without running them
    }
    void drop_queued() {
        std::deque<Entry> dropped;
        {
            std::lock_guard lock(mutex);
            dropped.swap(queue);
        }
        dropped.clear();
    }
};

// Mirrors the exact work-stealing BatchState + RAII HelperDone + WaitHelpers (with cancel) runner used by Device.
VulkanContext::BatchCompileFn make_pool_batch_runner(MiniCompilePool& pool, std::size_t max_helpers) {
    return [&pool, max_helpers](std::size_t count, const std::function<void(std::size_t)>& job) {
        if (!count) return;
        if (count == 1) {
            job(0);
            return;
        }
        struct BatchState {
            std::atomic<std::size_t> next{0};
            std::size_t count = 0;
            const std::function<void(std::size_t)>* job = nullptr;
            std::vector<std::uint8_t> owner_tags;
            std::mutex mutex;
            std::condition_variable cv;
            std::size_t pending_helpers = 0;
        };
        auto run_work = [](BatchState& s) {
            for (;;) {
                const std::size_t i = s.next.fetch_add(1, std::memory_order_relaxed);
                if (i >= s.count) break;
                (*s.job)(i);
            }
        };
        struct HelperDone {
            std::shared_ptr<BatchState> state;
            explicit HelperDone(std::shared_ptr<BatchState> s) : state(std::move(s)) {
                std::lock_guard lock(state->mutex);
                ++state->pending_helpers;
            }
            ~HelperDone() {
                if (!state) return;
                std::lock_guard lock(state->mutex);
                if (--state->pending_helpers == 0) state->cv.notify_all();
            }
        };
        auto state = std::make_shared<BatchState>();
        state->count = count;
        state->job = &job;
        const std::size_t helpers = (std::min)(count - 1, max_helpers);
        state->owner_tags.resize(helpers, 0);
        {
            struct WaitHelpers {
                MiniCompilePool& pool;
                std::shared_ptr<BatchState> state;
                std::size_t submitted_helpers = 0;
                ~WaitHelpers() {
                    for (std::size_t k = 0; k < submitted_helpers; ++k)
                        pool.cancel(&state->owner_tags[k]);
                    std::unique_lock lock(state->mutex);
                    state->cv.wait(lock, [&] { return state->pending_helpers == 0; });
                }
            } wait_guard{pool, state, 0};
            for (std::size_t k = 0; k < helpers; ++k) {
                auto done = std::make_shared<HelperDone>(state);
                if (!pool.submit(&state->owner_tags[k], [done = std::move(done), run_work] { run_work(*done->state); }))
                    break;
                ++wait_guard.submitted_helpers;
            }
            run_work(*state);
        }
    };
}

void test_batch_parallel_sync_pipeline_compile() {
    // 4a: collect_unbuilt_pipelines() deduplicates graphics (pso, unbound_mask, rectangle_stage) and compute across lists.
    {
        MockEnv env;
        GraphicsPipeline unbuilt_a = env.gfx[1];
        unbuilt_a.pipeline = VK_NULL_HANDLE;
        unbuilt_a.color_formats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
        unbuilt_a.depth_format = VK_FORMAT_D32_SFLOAT;
        unbuilt_a.variant = [](std::uint32_t) { return fake_handle<VkPipeline>(0x910); };
        unbuilt_a.built = [](std::uint32_t) -> VkPipeline { return VK_NULL_HANDLE; };
        env.gfx[10] = unbuilt_a;

        GraphicsPipeline unbuilt_b = env.gfx[1];
        unbuilt_b.pipeline = VK_NULL_HANDLE;
        unbuilt_b.rectangle_variant_topology = 35u;
        unbuilt_b.variant = [](std::uint32_t) { return fake_handle<VkPipeline>(0x911); };
        unbuilt_b.built = [](std::uint32_t) -> VkPipeline { return VK_NULL_HANDLE; };
        env.gfx[11] = unbuilt_b;

        ComputePipeline unbuilt_c = env.comp[1];
        unbuilt_c.pipeline = VK_NULL_HANDLE;
        unbuilt_c.build = +[](void*) { return fake_handle<VkPipeline>(0x912); };
        unbuilt_c.built = +[](void*) -> VkPipeline { return VK_NULL_HANDLE; };
        env.comp[10] = unbuilt_c;

        CommandList l1, l2;
        // Already-built pipeline (id=1): must NOT be collected.
        CHECK(l1.draw(env.base_draw(1, false)));
        // Unbuilt PSO 10 with slot 0 bound, slot 1 unbound, depth unbound -> flags = (1<<1) | unbound_depth = 0x102.
        Draw da1 = env.base_draw(10, false);
        CHECK(l1.draw(da1));
        CHECK(l1.draw(da1));  // duplicate in same list -> deduplicated
        // Unbuilt PSO 10 with both color slots bound, depth unbound -> flags = unbound_depth = 0x100.
        Draw da2 = env.base_draw(10, false);
        auto views2 = std::make_shared<DrawViews>(*da2.views);
        views2->render_targets[1] = {MockEnv::kColor, 0, 0, 1};
        views2->render_target_count = 2;
        da2.views = std::move(views2);
        CHECK(l1.draw(da2));
        // Unbuilt PSO 11 with rectangle-list variant topology -> flags = rectangle_stage.
        Draw db = env.base_draw(11, false);
        db.topology = 35u;
        CHECK(l2.draw(db));
        CHECK(l2.draw(da1));  // duplicate across second list -> deduplicated
        // Unbuilt compute pipeline 10 (twice -> deduplicated).
        Dispatch c{};
        c.pipeline = 10;
        c.x = c.y = c.z = 1;
        CHECK(l2.dispatch(c));
        CHECK(l2.dispatch(c));
        CHECK(l1.close() && l2.close());

        const CommandList* batch[2] = {&l1, &l2};
        PipelineCompiles compiles;
        collect_unbuilt_pipelines(batch, env.vk, compiles);
        CHECK(compiles.size() == 4);  // 3 graphics variants + 1 compute
        VkPipeline out = VK_NULL_HANDLE;
        CHECK(!compiles.lookup(&env.gfx[1], nullptr, 0u, out));
        CHECK(compiles.lookup(&env.gfx[10], nullptr, (1u << 1) | GraphicsPipeline::unbound_depth, out));
        CHECK(compiles.lookup(&env.gfx[10], nullptr, GraphicsPipeline::unbound_depth, out));
        CHECK(compiles.lookup(&env.gfx[11], nullptr, GraphicsPipeline::rectangle_stage, out));
        CHECK(compiles.lookup(nullptr, &env.comp[10], 0u, out));
    }

    // 4b: Closed pool and mid-batch cancel/drop_queued race safety (zero deadlocks, 100% job completion).
    {
        // Closed pool: every submit returns false -> calling thread finishes all jobs cleanly.
        MiniCompilePool closed_pool(4);
        closed_pool.close_and_drop();
        auto closed_runner = make_pool_batch_runner(closed_pool, 4);
        constexpr std::size_t kClosedJobs = 32;
        std::array<std::atomic<unsigned>, kClosedJobs> closed_hits{};
        for (auto& h : closed_hits) h.store(0, std::memory_order_relaxed);
        closed_runner(kClosedJobs, [&](std::size_t i) { closed_hits[i].fetch_add(1, std::memory_order_relaxed); });
        for (std::size_t i = 0; i < kClosedJobs; ++i) CHECK(closed_hits[i].load() == 1);

        // Concurrent mid-batch drop_queued / close_and_drop while helpers are active: zero deadlocks, exact-once runs.
        for (int iter = 0; iter < 50; ++iter) {
            MiniCompilePool pool(4);
            auto runner = make_pool_batch_runner(pool, 4);
            constexpr std::size_t kJobs = 48;
            std::array<std::atomic<unsigned>, kJobs> hits{};
            for (auto& h : hits) h.store(0, std::memory_order_relaxed);
            std::atomic<bool> started{false};
            std::thread dropper([&] {
                while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
                if (iter % 2 == 0) pool.drop_queued();
                else pool.close_and_drop();
            });
            runner(kJobs, [&](std::size_t i) {
                started.store(true, std::memory_order_release);
                hits[i].fetch_add(1, std::memory_order_relaxed);
                if (i < 4) std::this_thread::sleep_for(std::chrono::microseconds(50));
            });
            dropper.join();
            for (std::size_t i = 0; i < kJobs; ++i) CHECK(hits[i].load() == 1);
        }

        // 4b-busy: All pool workers blocked on unrelated long jobs — calling thread must finish all batch jobs and
        //          cancel unstarted queued helpers immediately without stalling behind the busy workers!
        {
            MiniCompilePool busy_pool(2);
            std::mutex gate_mu;
            std::condition_variable gate_cv;
            bool release_unrelated = false;
            std::atomic<unsigned> blocked_workers{0};
            int unrelated_owner_a = 0, unrelated_owner_b = 0;
            const auto blocker = [&] {
                blocked_workers.fetch_add(1, std::memory_order_acq_rel);
                std::unique_lock lock(gate_mu);
                gate_cv.wait(lock, [&] { return release_unrelated; });
            };
            CHECK(busy_pool.submit(&unrelated_owner_a, blocker));
            CHECK(busy_pool.submit(&unrelated_owner_b, blocker));
            while (blocked_workers.load(std::memory_order_acquire) < 2) std::this_thread::yield();

            auto runner = make_pool_batch_runner(busy_pool, 4);
            constexpr std::size_t kFastJobs = 8;
            std::array<std::atomic<unsigned>, kFastJobs> fast_hits{};
            for (auto& h : fast_hits) h.store(0, std::memory_order_relaxed);
            const auto t0 = std::chrono::steady_clock::now();
            runner(kFastJobs, [&](std::size_t i) { fast_hits[i].fetch_add(1, std::memory_order_relaxed); });
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
            for (std::size_t i = 0; i < kFastJobs; ++i) CHECK(fast_hits[i].load() == 1);
            CHECK(elapsed_ms < 250);

            {
                std::lock_guard lock(gate_mu);
                release_unrelated = true;
            }
            gate_cv.notify_all();
        }
    }

    // 4c: Live VulkanQueue::execute() batch-parallel pre-compile of 4 unbuilt graphics PSOs (with duplicates):
    //     compiles concurrently (max_in_flight >= 2), builds each PSO once, executes all draws with zero skips.
    {
        vktest::Device dev;
        const auto status = dev.initialize();
        if (status == vktest::DeviceStatus::unavailable) {
            std::puts("SKIP: no Vulkan device for live batch-parallel pipeline compile test");
            return;
        }
        CHECK(status == vktest::DeviceStatus::ready);
        if (status != vktest::DeviceStatus::ready) return;

        const auto vkCreateShaderModule = dev.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule");
        const auto vkDestroyShaderModule = dev.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
        const auto vkCreatePipelineLayout = dev.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
        const auto vkDestroyPipelineLayout = dev.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
        const auto vkCreateGraphicsPipelines = dev.fn<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines");
        const auto vkDestroyPipeline = dev.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");

        constexpr std::uint32_t kW = 8, kH = 8, kPitch = 256;
        const VkImage color_img =
            dev.image(VK_FORMAT_R8G8B8A8_UNORM, kW, kH, 1,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        void* rb_pixels = nullptr;
        const VkBuffer rb_buf = dev.buffer(kH * kPitch, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &rb_pixels);
        void* vb_mapped = nullptr;
        const VkBuffer vb = dev.buffer(256, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &vb_mapped);
        CHECK(color_img && rb_buf && rb_pixels && vb && vb_mapped);
        if (!color_img || !rb_buf || !vb) return;
        std::memset(rb_pixels, 0, kH * kPitch);
        const float verts[12] = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
        std::memcpy(vb_mapped, verts, sizeof(verts));

        const VkPushConstantRange gfx_pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                          sizeof(ShaderUserData)};
        VkPipelineLayoutCreateInfo gpl_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        gpl_ci.pushConstantRangeCount = 1;
        gpl_ci.pPushConstantRanges = &gfx_pcr;
        VkPipelineLayout gfx_layout = VK_NULL_HANDLE;
        CHECK(vkCreatePipelineLayout(dev.device, &gpl_ci, nullptr, &gfx_layout) == VK_SUCCESS);

        const auto vs_spv = build_vs_spirv();
        const auto fs_spv = build_fs_spirv();
        const auto make_mod = [&](const std::vector<std::uint32_t>& spv) {
            VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            ci.codeSize = spv.size() * sizeof(std::uint32_t);
            ci.pCode = spv.data();
            VkShaderModule m = VK_NULL_HANDLE;
            CHECK(vkCreateShaderModule(dev.device, &ci, nullptr, &m) == VK_SUCCESS);
            return m;
        };
        const VkShaderModule vs_mod = make_mod(vs_spv);
        const VkShaderModule fs_mod = make_mod(fs_spv);

        const auto build_real_gfx_pipe = [&]() -> VkPipeline {
            VkPipelineShaderStageCreateInfo stages[2]{
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
            };
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vs_mod;
            stages[0].pName = "main";
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fs_mod;
            stages[1].pName = "main";

            VkVertexInputBindingDescription vbind{0, 8, VK_VERTEX_INPUT_RATE_VERTEX};
            VkVertexInputAttributeDescription vattr{0, 0, VK_FORMAT_R32G32_SFLOAT, 0};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            vi.vertexBindingDescriptionCount = 1;
            vi.pVertexBindingDescriptions = &vbind;
            vi.vertexAttributeDescriptionCount = 1;
            vi.pVertexAttributeDescriptions = &vattr;

            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            vp.viewportCount = 1;
            vp.scissorCount = 1;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            rs.polygonMode = VK_POLYGON_MODE_FILL;
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
            rs.lineWidth = 1.0f;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState cba{};
            cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                 VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cb_state{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            cb_state.attachmentCount = 1;
            cb_state.pAttachments = &cba;
            const VkDynamicState dyn_states[] = {
                VK_DYNAMIC_STATE_VIEWPORT,           VK_DYNAMIC_STATE_SCISSOR,
                VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
                VK_DYNAMIC_STATE_STENCIL_REFERENCE,  VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE,
            };
            VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dyn.dynamicStateCount = static_cast<std::uint32_t>(std::size(dyn_states));
            dyn.pDynamicStates = dyn_states;
            const VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
            VkPipelineRenderingCreateInfo rend{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
            rend.colorAttachmentCount = 1;
            rend.pColorAttachmentFormats = &color_fmt;
            VkGraphicsPipelineCreateInfo gp_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rend};
            gp_ci.stageCount = 2;
            gp_ci.pStages = stages;
            gp_ci.pVertexInputState = &vi;
            gp_ci.pInputAssemblyState = &ia;
            gp_ci.pViewportState = &vp;
            gp_ci.pRasterizationState = &rs;
            gp_ci.pMultisampleState = &ms;
            gp_ci.pColorBlendState = &cb_state;
            gp_ci.pDynamicState = &dyn;
            gp_ci.layout = gfx_layout;
            VkPipeline pipe = VK_NULL_HANDLE;
            CHECK(vkCreateGraphicsPipelines(dev.device, VK_NULL_HANDLE, 1, &gp_ci, nullptr, &pipe) == VK_SUCCESS);
            return pipe;
        };

        std::array<VkPipeline, 4> created_pipes{};
        {
            VulkanContext ctx(dev.device_proc, dev.device, true, dev.conditional_rendering, dev.buffer_device_address);
            CHECK(ctx.loaded());
            MiniCompilePool pool(4);
            ctx.set_batch_compile_runner(make_pool_batch_runner(pool, 4));

            constexpr ResourceId kRtId = 1, kVbId = 2, kRbId = 3;
            constexpr std::uint64_t kVbVa = 0x10000ull;
            ResourceBinding rt_b{};
            rt_b.image = color_img;
            rt_b.format = VK_FORMAT_R8G8B8A8_UNORM;
            rt_b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
            rt_b.width = kW;
            rt_b.height = kH;
            rt_b.color_attachment = true;
            ctx.add_resource(kRtId, rt_b, VK_IMAGE_LAYOUT_UNDEFINED);

            ResourceBinding vb_b{};
            vb_b.buffer = vb;
            vb_b.size = 256;
            vb_b.va = kVbVa;
            ctx.add_resource(kVbId, vb_b);

            ResourceBinding rb_b{};
            rb_b.buffer = rb_buf;
            rb_b.size = kH * kPitch;
            rb_b.host_readable = true;
            ctx.add_resource(kRbId, rb_b);

            std::atomic<unsigned> in_flight{0};
            std::atomic<unsigned> max_in_flight{0};
            std::array<std::atomic<unsigned>, 4> compile_calls{};
            for (auto& c : compile_calls) c.store(0, std::memory_order_relaxed);
            std::mutex pipe_mu;

            for (std::size_t idx = 0; idx < 4; ++idx) {
                GraphicsPipeline gp{};
                gp.pipeline = VK_NULL_HANDLE;
                gp.layout = gfx_layout;
                gp.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
                gp.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
                gp.vertex_slots = 0b1;
                gp.built = [&, idx](std::uint32_t) -> VkPipeline {
                    std::lock_guard lock(pipe_mu);
                    return created_pipes[idx];
                };
                gp.variant = [&, idx](std::uint32_t) -> VkPipeline {
                    compile_calls[idx].fetch_add(1, std::memory_order_relaxed);
                    const unsigned cur = in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
                    unsigned prev = max_in_flight.load(std::memory_order_relaxed);
                    while (cur > prev && !max_in_flight.compare_exchange_weak(prev, cur, std::memory_order_acq_rel)) {}
                    // Brief wait so at least 2 threads overlap concurrently inside variant().
                    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
                    while (max_in_flight.load(std::memory_order_acquire) < 2 &&
                           std::chrono::steady_clock::now() < until)
                        std::this_thread::yield();
                    VkPipeline p = build_real_gfx_pipe();
                    {
                        std::lock_guard lock(pipe_mu);
                        created_pipes[idx] = p;
                    }
                    in_flight.fetch_sub(1, std::memory_order_acq_rel);
                    return p;
                };
                ctx.add_graphics_pipeline(300 + idx, gp);
            }

            VulkanQueue queue(ctx, dev.queue, dev.family, VulkanQueue::QueueKind::direct);
            CHECK(queue.ok());

            auto make_quad = [&](std::uintptr_t pso, float r, float g, float b, std::int32_t x0, std::int32_t y0,
                                 std::int32_t x1, std::int32_t y1) {
                Draw d{};
                d.pipeline = pso;
                auto views = std::make_shared<DrawViews>();
                views->render_targets[0] = {kRtId, 0, 0, 1};
                views->render_target_count = 1;
                views->vertex_buffers[0] = {kVbVa, 256, 8};
                d.views = std::move(views);
                d.indexed = false;
                d.topology = 4;
                d.viewport = {0.0f, 0.0f, float(kW), float(kH), 0.0f, 1.0f};
                d.scissor = {x0, y0, x1, y1};
                d.blend_factor = {1, 1, 1, 1};
                d.count = 6;
                d.instances = 1;
                d.user_data[0] = float_bits(r);
                d.user_data[1] = float_bits(g);
                d.user_data[2] = float_bits(b);
                d.user_data[3] = float_bits(1.0f);
                d.user_data[8] = float_bits(1.0f);
                return d;
            };

            // Two lists across one execute() batch using all 4 unbuilt PSOs (with duplicate draws).
            CommandList list1, list2;
            CHECK(list1.transition({kRtId, all_subresources, d3d12_state::common, d3d12_state::render_target}));
            CHECK(list1.draw(make_quad(300, 1.0f, 0.0f, 0.0f, 0, 0, 4, 4)));
            CHECK(list1.draw(make_quad(300, 1.0f, 0.0f, 0.0f, 0, 0, 4, 4)));  // duplicate PSO 300
            CHECK(list1.draw(make_quad(301, 0.0f, 1.0f, 0.0f, 4, 0, 8, 4)));
            CHECK(list1.close());

            CHECK(list2.draw(make_quad(302, 0.0f, 0.0f, 1.0f, 0, 4, 4, 8)));
            CHECK(list2.draw(make_quad(303, 1.0f, 1.0f, 0.0f, 4, 4, 8, 8)));
            CHECK(list2.draw(make_quad(301, 0.0f, 1.0f, 0.0f, 4, 0, 8, 4)));  // duplicate across list boundary
            CHECK(list2.transition({kRtId, all_subresources, d3d12_state::render_target, d3d12_state::copy_source}));
            CopyTexture readback{};
            readback.dst = {kRbId, true, 0, 0, kW, kH, 1, kPitch};
            readback.src = {kRtId, false, 0, 0, 0, 0, 0, 0};
            CHECK(list2.copy_texture(readback));
            CHECK(list2.close());

            const CommandList* batch[2] = {&list1, &list2};
            CHECK(queue.execute(batch) == ReplayResult::success);
            CHECK(queue.idle());

            CHECK(max_in_flight.load(std::memory_order_acquire) >= 2);
            for (std::size_t idx = 0; idx < 4; ++idx) CHECK(compile_calls[idx].load() == 1);

            const auto* px = static_cast<const std::uint8_t*>(rb_pixels);
            const auto check_px = [&](std::uint32_t x, std::uint32_t y, std::uint8_t r, std::uint8_t g, std::uint8_t b,
                                      std::uint8_t a) {
                const std::size_t off = std::size_t(y) * kPitch + std::size_t(x) * 4;
                CHECK(px[off] == r && px[off + 1] == g && px[off + 2] == b && px[off + 3] == a);
            };
            check_px(1, 1, 255, 0, 0, 255);
            check_px(6, 1, 0, 255, 0, 255);
            check_px(1, 6, 0, 0, 255, 255);
            check_px(6, 6, 255, 255, 0, 255);
            ctx.set_batch_compile_runner({});
        }

        for (VkPipeline p : created_pipes)
            if (p) vkDestroyPipeline(dev.device, p, nullptr);
        if (vs_mod) vkDestroyShaderModule(dev.device, vs_mod, nullptr);
        if (fs_mod) vkDestroyShaderModule(dev.device, fs_mod, nullptr);
        if (gfx_layout) vkDestroyPipelineLayout(dev.device, gfx_layout, nullptr);
        CHECK(dev.validation_errors() == 0);
        std::puts("batch-parallel sync pipeline compile (collect_unbuilt, concurrency, closed/drop race, GPU): ok");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Part 5: Dynamic rendering scope batching, LayoutTracker no-op elision, PrepareViewCache & PreparedList reuse

void test_rendering_scope_batching_and_fast_paths() {
    MockEnv env;
    constexpr ResourceId kColor2 = 100;
    constexpr ResourceId kColorArray = 101;
    constexpr ResourceId kColorGeneral = 102;
    constexpr ResourceId kDepth = 103;
    constexpr ResourceId kDepth2 = 104;

    ResourceBinding color2 = env.res[MockEnv::kColor];
    color2.image = fake_handle<VkImage>(0x110);
    color2.mip_levels = 2;
    env.res[kColor2] = color2;

    ResourceBinding color_arr = env.res[MockEnv::kColor];
    color_arr.image = fake_handle<VkImage>(0x111);
    color_arr.array_layers = 4;
    env.res[kColorArray] = color_arr;

    ResourceBinding color_gen = env.res[MockEnv::kColor];
    color_gen.image = fake_handle<VkImage>(0x112);
    color_gen.simultaneous_access = true;
    env.res[kColorGeneral] = color_gen;

    ResourceBinding depth{};
    depth.image = fake_handle<VkImage>(0x113);
    depth.format = VK_FORMAT_D24_UNORM_S8_UINT;
    depth.aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    depth.width = depth.height = 64;
    env.res[kDepth] = depth;

    ResourceBinding depth2 = depth;
    depth2.image = fake_handle<VkImage>(0x114);
    env.res[kDepth2] = depth2;

    // Distinct image views per (image, mip, layer, layerCount, format) so view changes are observable.
    std::size_t view_calls = 0;
    env.vk.image_view = [&view_calls](const ResourceBinding& r, const VkImageSubresourceRange& range) {
        ++view_calls;
        std::uintptr_t h = reinterpret_cast<std::uintptr_t>(r.image) * 1315423911ull;
        h ^= std::uintptr_t(range.baseMipLevel) << 8;
        h ^= std::uintptr_t(range.baseArrayLayer) << 12;
        h ^= std::uintptr_t(range.layerCount) << 16;
        h ^= std::uintptr_t(r.format) << 20;
        return fake_handle<VkImageView>(h ? h : 1);
    };

    // Additional pipelines for depth_aspects, reads_glc_buffer, multi-RT, and attachment-less draws.
    GraphicsPipeline g_depth_only = env.gfx[1];
    g_depth_only.pipeline = fake_handle<VkPipeline>(0x710);
    g_depth_only.depth_format = VK_FORMAT_D24_UNORM_S8_UINT;
    g_depth_only.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    env.gfx[10] = g_depth_only;

    GraphicsPipeline g_depth_stencil = g_depth_only;
    g_depth_stencil.pipeline = fake_handle<VkPipeline>(0x711);
    g_depth_stencil.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    env.gfx[11] = g_depth_stencil;

    GraphicsPipeline g_glc = env.gfx[1];
    g_glc.pipeline = fake_handle<VkPipeline>(0x712);
    g_glc.reads_glc_buffer = true;
    env.gfx[12] = g_glc;

    GraphicsPipeline g_mrt2 = env.gfx[1];
    g_mrt2.pipeline = fake_handle<VkPipeline>(0x713);
    g_mrt2.color_formats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    g_mrt2.depth_format = VK_FORMAT_D24_UNORM_S8_UINT;
    g_mrt2.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    env.gfx[13] = g_mrt2;

    GraphicsPipeline g_no_att = env.gfx[1];
    g_no_att.pipeline = fake_handle<VkPipeline>(0x714);
    g_no_att.color_formats.clear();
    env.gfx[14] = g_no_att;

    const auto resolve = [&env](ResourceId id) { return env.resolve(id); };
    const VkCommandBuffer cb = fake_handle<VkCommandBuffer>(0x1);

    // 5a: Consecutive MRT + DSV draws sharing targets (even across different pipelines) merge into 1 scope + 1 barrier;
    //     PrepareViewCache calls vk.image_view only once per unique attachment view across all draws.
    {
        CommandList list;
        Draw d = env.base_draw(13, true);
        auto v = std::make_shared<DrawViews>(*d.views);
        v->render_targets[0] = {MockEnv::kColor, 0, 0, 1};
        v->render_targets[1] = {kColor2, 0, 0, 1};
        v->render_target_count = 2;
        d.views = v;
        d.depth = {kDepth, 0, 0, 1};
        for (int k = 0; k < 6; ++k) {
            d.user_data[0] = std::uint32_t(k);
            CHECK(list.draw(d));
        }
        CHECK(list.close());

        view_calls = 0;
        PreparedList prep;
        CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);
        CHECK(view_calls == 3);  // 2 color views + 1 depth view cached across all 6 draws (instead of 18)

        g_mc.reset();
        emit(prep, cb, env.vk);
        CHECK(g_mc.draws == 6);
        CHECK(g_mc.begin_rendering == 1 && g_mc.end_rendering == 1);
        CHECK(g_mc.barriers == 1);  // coalesced single attachment barrier for 2 RTs + 1 DSV
    }

    // 5b: Changing color target, mip/view, depth_aspects, layers, or area closes and reopens rendering scope.
    {
        // Color target change (kColor -> kColor2)
        CommandList list;
        Draw d1 = env.base_draw(1, false);
        Draw d2 = env.base_draw(1, false);
        auto v2 = std::make_shared<DrawViews>(*d2.views);
        v2->render_targets[0] = {kColor2, 0, 0, 1};
        d2.views = v2;
        // Same kColor2, different mip (mip 1 -> different view, range, and area 32x32)
        Draw d3 = env.base_draw(1, false);
        auto v3 = std::make_shared<DrawViews>(*d3.views);
        v3->render_targets[0] = {kColor2, 1, 0, 1};
        d3.views = v3;
        // Different layerCount (1 layer -> 2 layers on kColorArray)
        Draw d4 = env.base_draw(1, false);
        auto v4 = std::make_shared<DrawViews>(*d4.views);
        v4->render_targets[0] = {kColorArray, 0, 0, 1};
        d4.views = v4;
        Draw d5 = env.base_draw(1, false);
        auto v5 = std::make_shared<DrawViews>(*d5.views);
        v5->render_targets[0] = {kColorArray, 0, 0, 2};
        d5.views = v5;
        // Different depth_aspects on the SAME depth target (DEPTH -> DEPTH|STENCIL)
        Draw d6 = env.base_draw(10, false);
        d6.depth = {kDepth, 0, 0, 1};
        Draw d7 = env.base_draw(11, false);
        d7.depth = {kDepth, 0, 0, 1};
        // Attachment-less draws with different viewport areas (64x64 -> 32x32)
        Draw d8 = env.base_draw(14, false);
        auto v8 = std::make_shared<DrawViews>(*d8.views);
        v8->render_target_count = 0;
        d8.views = v8;
        d8.viewport = {0, 0, 64, 64, 0, 1};
        Draw d9 = env.base_draw(14, false);
        auto v9 = std::make_shared<DrawViews>(*d9.views);
        v9->render_target_count = 0;
        d9.views = v9;
        d9.viewport = {0, 0, 32, 32, 0, 1};
        // Different layout (OPTIMAL vs simultaneous_access GENERAL)
        Draw d10 = env.base_draw(1, false);
        auto v10 = std::make_shared<DrawViews>(*d10.views);
        v10->render_targets[0] = {kColorGeneral, 0, 0, 1};
        d10.views = v10;

        CHECK(list.draw(d1) && list.draw(d2) && list.draw(d3) && list.draw(d4) && list.draw(d5));
        CHECK(list.draw(d6) && list.draw(d7) && list.draw(d8) && list.draw(d9) && list.draw(d10));
        CHECK(list.close());

        PreparedList prep;
        CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);
        g_mc.reset();
        emit(prep, cb, env.vk);
        CHECK(g_mc.draws == 10);
        CHECK(g_mc.begin_rendering == 10 && g_mc.end_rendering == 10);
    }

    // 5c: Exact event ordering at mandatory scope-closing boundaries:
    //     reads_glc_buffer, indirect_clamp, quad_indices, non-draw (dispatch/barrier/clear), query, and predication.
    {
        static std::vector<std::string> ev;
        ev.clear();
        ReplayFunctions trace_vk = env.vk;
        trace_vk.begin_rendering = +[](VkCommandBuffer, const VkRenderingInfo*) { ev.push_back("begin_rp"); };
        trace_vk.end_rendering = +[](VkCommandBuffer) { ev.push_back("end_rp"); };
        trace_vk.draw = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
            ev.push_back("draw");
        };
        trace_vk.draw_indexed = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                    std::uint32_t) { ev.push_back("draw_idx"); };
        trace_vk.draw_indirect = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, std::uint32_t, std::uint32_t) {
            ev.push_back("draw_ind");
        };
        trace_vk.pipeline_barrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags dst,
                                        VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                        const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
            if (dst == VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT) ev.push_back("glc_barrier");
        };
        trace_vk.emit_indirect_clamp = [](VkCommandBuffer, const ClampedIndirect&) { ev.push_back("clamp"); };
        trace_vk.emit_quad_indices = [](VkCommandBuffer, const QuadIndexExpansion&) { ev.push_back("quad_expand"); };
        trace_vk.dispatch = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {
            ev.push_back("dispatch");
        };
        trace_vk.begin_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, VkQueryControlFlags) {
            ev.push_back("begin_q");
        };
        trace_vk.end_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t) { ev.push_back("end_q"); };
        trace_vk.begin_conditional_rendering = +[](VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
            ev.push_back("begin_pred");
        };
        trace_vk.end_conditional_rendering = +[](VkCommandBuffer) { ev.push_back("end_pred"); };

        CommandList list;
        const Draw d = env.base_draw(1, false);
        const Draw d_glc = env.base_draw(12, false);
        Draw d_quad = env.base_draw(1, true);
        d_quad.topology = topology_quad_list;
        d_quad.count = 8;
        IndirectDrawArgs ind{};
        ind.argument_buffer = MockEnv::kBuf;
        ind.stride = 16;
        ind.max_count = 1;
        Dispatch c{};
        c.pipeline = 1;
        c.x = c.y = c.z = 1;

        // 2 merged draws -> reads_glc_buffer -> quad_expand -> indirect_clamp -> dispatch -> query -> predication
        CHECK(list.draw(d));
        CHECK(list.draw(d));
        CHECK(list.draw(d_glc));
        CHECK(list.draw(d_quad));
        CHECK(list.draw_indirect(Draw{d}, ind));
        CHECK(list.dispatch(c));
        CHECK(list.begin_query({MockEnv::kOccHeap, d3d12_query::occlusion, 0}));
        CHECK(list.draw(d));
        CHECK(list.draw(d));
        CHECK(list.end_query({MockEnv::kOccHeap, d3d12_query::occlusion, 0}));
        CHECK(list.set_predication({MockEnv::kPred, 0, d3d12_predication::equal_zero}));
        CHECK(list.draw(d));
        CHECK(list.draw(d));
        CHECK(list.close());

        PreparedList prep;
        CHECK(prepare(list, resolve, trace_vk, prep) == ReplayResult::success);
        QueryReplay qr;
        CHECK(qr.plan(prep) == ReplayResult::success);
        emit(prep, cb, trace_vk);

        const std::vector<std::string> expected = {
            "begin_rp", "draw", "draw", "end_rp",
            "glc_barrier", "begin_rp", "draw", "end_rp",
            "quad_expand", "begin_rp", "draw_idx", "end_rp",
            "clamp", "begin_rp", "draw_ind", "end_rp",
            "dispatch",
            "begin_q", "begin_rp", "draw", "draw", "end_rp", "end_q",
            "begin_pred", "begin_rp", "draw", "draw", "end_rp", "end_pred",
        };
        if (ev != expected) {
            std::fprintf(stderr, "FAIL scope boundary event order (got %zu vs expected %zu):\n  got: ", ev.size(),
                         expected.size());
            for (const auto& s : ev) std::fprintf(stderr, "%s ", s.c_str());
            std::fprintf(stderr, "\n  want: ");
            for (const auto& s : expected) std::fprintf(stderr, "%s ", s.c_str());
            std::fprintf(stderr, "\n");
            ++g_failures;
        }
    }

    // 5d: No-op LayoutTracker::set() elision (0 journal entries when unchanged; exact rollback when changed).
    {
        LayoutTracker tracker(true);
        ResourceBinding img{};
        img.image = fake_handle<VkImage>(0x120);
        img.format = VK_FORMAT_R8G8B8A8_UNORM;
        img.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        img.mip_levels = 3;
        img.array_layers = 2;
        tracker.track(42, img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        tracker.begin();
        CHECK(tracker.journal_size() == 0);
        // No-op set on whole image and on single subresource must NOT journal anything.
        tracker.set(42, {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS},
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        tracker.set(42, {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 1, 1}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        CHECK(tracker.journal_size() == 0);

        // Actual change on mip 1 layer 0 journals once.
        tracker.set(42, {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1}, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHECK(tracker.journal_size() == 1);
        CHECK(tracker.layout(42, 1) == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHECK(tracker.layout(42, 0) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // Rollback restores mip 1 layer 0 to COLOR_ATTACHMENT_OPTIMAL.
        tracker.rollback();
        CHECK(tracker.journal_size() == 0);
        CHECK(tracker.layout(42, 1) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

        // Commit persists the change.
        tracker.begin();
        tracker.set(42, {VK_IMAGE_ASPECT_COLOR_BIT, 2, 1, 1, 1}, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        CHECK(tracker.journal_size() == 1);
        tracker.commit();
        CHECK(tracker.journal_size() == 0);
        CHECK(tracker.layout(42, 5) == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }

    // 5e: PreparedList reuse across mixed lists (draws -> copies/clears -> draws -> shorter list) has no stale state.
    {
        PreparedList reused;
        CommandList l1, l2, l3;
        const Draw d = env.base_draw(1, true);
        CHECK(l1.draw(d) && l1.draw(d) && l1.draw(d) && l1.close());
        CHECK(l2.transition({MockEnv::kColor, all_subresources, d3d12_state::common, d3d12_state::render_target}));
        CHECK(l2.clear_render_target({MockEnv::kColor, 0, 0, 1}, {0, 0, 0, 1}));
        CHECK(l2.copy_buffer({MockEnv::kBuf, 0, MockEnv::kResults, 0, 64}));
        CHECK(l2.draw(d));
        CHECK(l2.close());
        CHECK(l3.draw(d) && l3.close());

        CHECK(prepare(l1, resolve, env.vk, reused) == ReplayResult::success);
        CHECK(reused.commands.size() == 3 && reused.draw_count == 3);
        CHECK(prepare(l2, resolve, env.vk, reused) == ReplayResult::success);
        CHECK(reused.commands.size() == 4 && reused.draw_count == 1);
        CHECK(!reused.commands[2].buffer_copies.empty());
        CHECK(reused.commands[3].buffer_copies.empty());
        CHECK(prepare(l3, resolve, env.vk, reused) == ReplayResult::success);
        CHECK(reused.commands.size() == 1 && reused.draw_count == 1);
        CHECK(reused.commands[0].buffer_copies.empty());
        g_mc.reset();
        emit(reused, cb, env.vk);
        CHECK(g_mc.draws == 1 && g_mc.begin_rendering == 1 && g_mc.end_rendering == 1);
    }

    std::puts("rendering scope batching, LayoutTracker no-op elision & prepare fast path: ok");
}

// ---------------------------------------------------------------------------------------------------------------
// Part 6: Round 8 WS3 replay floor cuts: CommandList::close() cached summary, EmitScratch reuse,
//         VulkanContext::unbuilt_sync_pipelines_ atomic fast-path + pass-0 fallback, and BindlessSet clean atomics.
void test_r8_replay_floor_cuts() {
    // 6a: CommandList::close() caches predication_captures, runs_shaders, has_rt_ds_clear, and reset() clears them.
    {
        CommandList l;
        CHECK(l.close());
        CHECK(l.commands().empty() && l.predication_captures() == 0 && !l.runs_shaders() && !l.has_rt_ds_clear());

        CHECK(l.reset());
        CHECK(l.uav_barrier({}));
        CHECK(l.copy_buffer({1, 0, 2, 0, 64}));
        CHECK(l.set_predication({3, 0, d3d12_predication::equal_zero}));
        CHECK(l.set_predication({0, 0, 0}));  // turning predication off does NOT increment captures
        CHECK(l.set_predication({3, 8, d3d12_predication::not_equal_zero}));
        CHECK(l.close());
        CHECK(l.predication_captures() == 2 && !l.runs_shaders() && !l.has_rt_ds_clear());

        CHECK(l.reset());
        CHECK(l.predication_captures() == 0 && !l.runs_shaders() && !l.has_rt_ds_clear());
        TargetView tv{};
        tv.resource = 1;
        tv.layer_count = 1;
        CHECK(l.clear_render_target(tv, ClearColor{}));
        Dispatch d{};
        d.pipeline = 1;
        d.x = d.y = d.z = 1;
        CHECK(l.dispatch(d));
        CHECK(l.close());
        CHECK(l.predication_captures() == 0 && l.runs_shaders() && l.has_rt_ds_clear());

        CHECK(l.reset());
        CHECK(l.clear_depth_stencil(tv, clear_depth, 1.0f, 0));
        CHECK(l.close());
        CHECK(!l.runs_shaders() && l.has_rt_ds_clear());

        CHECK(l.reset());
        ExecuteIndirect ei{};
        ei.kind = IndirectKind::dispatch;
        ei.argument_buffer = 2;
        ei.stride = 16;
        ei.max_count = 1;
        CHECK(l.execute_indirect(ei));
        CHECK(l.close());
        CHECK(l.runs_shaders() && !l.has_rt_ds_clear());
    }

    // 6b: EmitScratch reuse across repeated emit() calls preserves vector capacity without reallocating.
    {
        MockEnv env;
        const auto resolve = [&env](ResourceId id) { return env.resolve(id); };
        const VkCommandBuffer cb = fake_handle<VkCommandBuffer>(0x2);

        CommandList list;
        const Draw d = env.base_draw(1, false);
        CHECK(list.draw(d) && list.close());
        PreparedList prep;
        CHECK(prepare(list, resolve, env.vk, prep) == ReplayResult::success);

        EmitScratch scratch;
        scratch.many_barriers.reserve(8);
        const void* orig_ptr = scratch.many_barriers.data();
        const std::size_t orig_cap = scratch.many_barriers.capacity();
        g_mc.reset();
        ReplayStateCache cache;
        emit(prep, cb, env.vk, nullptr, nullptr, &cache, &scratch);
        emit(prep, cb, env.vk, nullptr, nullptr, &cache, &scratch);
        CHECK(g_mc.draws == 2);
        CHECK(scratch.many_barriers.data() == orig_ptr && scratch.many_barriers.capacity() == orig_cap);
    }

    // 6c & 6d: Live Vulkan checks for unbuilt_sync_pipelines_ atomic fast-path (+ pass-0 fallback) and BindlessSet
    //          lock-free clean atomics (needs_prepare / current_set / needs_commit).
    vktest::Device dev;
    if (!dev.create()) {
        std::puts("r8 replay floor cuts (CPU): ok (GPU part skipped: no Vulkan device)");
        return;
    }

    // 6c: VulkanContext unbuilt_sync_pipelines_ atomic fast-path.
    {
        VulkanContext ctx(dev.device_proc, dev.device, true, dev.conditional_rendering, dev.buffer_device_address);
        CHECK(ctx.loaded());
        CHECK(!ctx.has_unbuilt_sync_pipelines());

        // Pre-built pipeline does NOT arm unbuilt_sync_pipelines_.
        ComputePipeline prebuilt{};
        prebuilt.pipeline = fake_handle<VkPipeline>(0x900);
        ctx.add_pipeline(1, prebuilt);
        CHECK(!ctx.has_unbuilt_sync_pipelines());

        // Unbuilt synchronous compute pipeline arms unbuilt_sync_pipelines_; building clears it back to 0.
        bool is_built = false;
        ComputePipeline lazy{};
        lazy.build_context = &is_built;
        lazy.built = +[](void* ctx) -> VkPipeline {
            return *static_cast<bool*>(ctx) ? fake_handle<VkPipeline>(0x901) : VK_NULL_HANDLE;
        };
        lazy.build = +[](void* ctx) -> VkPipeline {
            *static_cast<bool*>(ctx) = true;
            return fake_handle<VkPipeline>(0x901);
        };
        ctx.add_pipeline(2, lazy);
        CHECK(ctx.has_unbuilt_sync_pipelines());
        ctx.note_sync_pipelines_built(1);
        CHECK(!ctx.has_unbuilt_sync_pipelines());

        // Unused unbuilt pipeline clears after kWarmEmptyScans consecutive empty scans with no new registrations.
        ctx.add_pipeline(3, lazy);
        CHECK(ctx.has_unbuilt_sync_pipelines());
        const std::uint64_t gen = ctx.sync_pipeline_gen();
        for (int i = 0; i < 16; ++i) ctx.note_empty_pipeline_scan(gen);
        CHECK(!ctx.has_unbuilt_sync_pipelines());
    }

    // 6d: BindlessSet lock-free clean atomics (needs_prepare / current_set / needs_commit).
    {
        VkPhysicalDeviceMemoryProperties mem_props{};
        const auto get_mem = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            dev.instance_proc(dev.instance, "vkGetPhysicalDeviceMemoryProperties"));
        CHECK(get_mem != nullptr);
        get_mem(dev.physical, &mem_props);

        BindlessFeatures features{};
        features.update_after_bind = false;
        features.partially_bound = false;
        BindlessSet set(dev.device_proc, dev.device, mem_props, features);
        CHECK(set.ok());
        CHECK(set.needs_prepare());
        CHECK(set.current_set() == set.set() && set.current_set() != VK_NULL_HANDLE);
        CHECK(!set.needs_commit());

        const auto create_pool = dev.fn<PFN_vkCreateCommandPool>("vkCreateCommandPool");
        const auto destroy_pool = dev.fn<PFN_vkDestroyCommandPool>("vkDestroyCommandPool");
        const auto alloc_cb = dev.fn<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
        const auto begin_cb = dev.fn<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer");
        const auto end_cb = dev.fn<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
        const auto queue_submit = dev.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
        const auto queue_wait = dev.fn<PFN_vkQueueWaitIdle>("vkQueueWaitIdle");

        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = dev.family;
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        CHECK(create_pool(dev.device, &pci, nullptr, &pool) == VK_SUCCESS);
        VkCommandBuffer cb = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        CHECK(alloc_cb(dev.device, &cai, &cb) == VK_SUCCESS);

        auto record_and_submit = [&](auto&& fn) {
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            CHECK(begin_cb(cb, &bi) == VK_SUCCESS);
            fn(cb);
            CHECK(end_cb(cb) == VK_SUCCESS);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            CHECK(queue_submit(dev.queue, 1, &si, VK_NULL_HANDLE) == VK_SUCCESS);
            CHECK(queue_wait(dev.queue) == VK_SUCCESS);
        };

        // First flush: initializes fallbacks & uploads initial table -> needs_commit becomes true until result reported.
        record_and_submit([&](VkCommandBuffer c) {
            set.record_fallback_init(c);
            CHECK(set.flush(c, 1));
            CHECK(set.needs_commit());
        });
        set.table_upload_result(true);
        CHECK(!set.needs_commit());
        CHECK(!set.needs_prepare());
        CHECK(set.current_set() == set.set());

        // Inserting a key dirties needs_prepare(); flushing + committing clears it back to false.
        DescriptorKey key{};
        key[0] = 0x12345678u;
        CHECK(set.insert_key(key, 0, 0));
        CHECK(set.needs_prepare());
        record_and_submit([&](VkCommandBuffer c) {
            CHECK(set.flush(c, 2));
            CHECK(set.needs_commit());
        });
        set.table_upload_result(true);
        CHECK(!set.needs_commit());
        CHECK(!set.needs_prepare());

        // Replacing an existing key replaces the generation (updates current_set() and keeps needs_prepare() true
        // while a retired generation waits for release_retired()).
        const VkDescriptorSet old_set = set.current_set();
        CHECK(set.insert_key(key, 1, 0));
        CHECK(set.needs_prepare());
        record_and_submit([&](VkCommandBuffer c) {
            CHECK(set.flush(c, 3));
        });
        set.table_upload_result(true);
        CHECK(set.current_set() == set.set() && set.current_set() != old_set);
        CHECK(set.has_retired() && set.needs_prepare());  // retired generation still pending release
        set.release_retired(3);
        CHECK(!set.has_retired() && !set.needs_prepare());

        // Erasing a key dirties needs_prepare() again.
        CHECK(set.erase_key(key));
        CHECK(set.needs_prepare());
        record_and_submit([&](VkCommandBuffer c) {
            CHECK(set.flush(c, 4));
        });
        set.table_upload_result(true);
        CHECK(!set.needs_prepare());

        destroy_pool(dev.device, pool, nullptr);
        CHECK(dev.validation_errors() == 0);
    }

    std::puts("r8 replay floor cuts (CommandList summary, EmitScratch, unbuilt_sync_pipelines_, BindlessSet atomics): ok");
}

}  // namespace

int main() {
    test_cpu_elision_and_invalidation();
    test_throughput_5000_draws();
    test_gpu_live_validation();
    test_batch_parallel_sync_pipeline_compile();
    test_rendering_scope_batching_and_fast_paths();
    test_r8_replay_floor_cuts();
    if (g_failures) {
        std::fprintf(stderr, "replay_state_cache_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::puts("replay_state_cache_test: ALL PASS");
    return 0;
}
