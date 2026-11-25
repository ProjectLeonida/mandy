// CPU regression for replay's consecutive draw rendering scopes. Synthetic prepared resources and recording
// Vulkan functions check draw state/order and scope boundaries without a driver, GPU or game files.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {
using namespace leonida;

template<class T> T handle(std::uintptr_t value) { return reinterpret_cast<T>(value); }

bool ok = true;
void check(bool condition, const char* description) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", description); ok = false; }
}

struct DrawState {
    VkPipeline pipeline{};
    VkDescriptorSet descriptors{};
    ShaderUserData data{};
    std::vector<VkViewport> viewports;
    std::vector<VkRect2D> scissors;
    VkPrimitiveTopology topology{};
    VkBool32 restart{};
    std::uint32_t front = 0, back = 0;
    std::array<float, 4> blend{};
    VkBuffer vertex{}, index{};
    VkDeviceSize vertex_offset = 0, vertex_size = 0, stride = 0, index_offset = 0;
    VkIndexType index_type{};
    std::string kind;
    std::uint32_t count = 0, instances = 0, first = 0, first_instance = 0;
    std::int32_t base = 0;
};

struct Rendering {
    VkRect2D area{};
    std::uint32_t layers = 0;
    std::vector<VkRenderingAttachmentInfo> colors;
    VkRenderingAttachmentInfo depth{}, stencil{};
    bool has_depth = false, has_stencil = false;
};

struct Recorder {
    static Recorder* current;
    bool rendering = false;
    DrawState state;
    std::vector<DrawState> draws;
    std::vector<Rendering> scopes;
    std::vector<std::string> events;
    std::size_t ends = 0;

    void outside(const char* event) {
        check(!rendering, "barriers, helpers and non-draw commands execute outside rendering");
        events.emplace_back(event);
    }
    void draw(const char* kind, std::uint32_t count, std::uint32_t instances, std::uint32_t first,
              std::int32_t base, std::uint32_t first_instance) {
        check(rendering, "every draw executes inside rendering");
        state.kind = kind;
        state.count = count; state.instances = instances; state.first = first;
        state.base = base; state.first_instance = first_instance;
        draws.push_back(state);
        events.emplace_back(kind);
    }

    ReplayFunctions functions() {
        ReplayFunctions vk{};
        vk.begin_rendering = +[](VkCommandBuffer, const VkRenderingInfo* info) {
            auto& r = *current;
            check(!r.rendering, "rendering scopes never nest");
            r.rendering = true;
            Rendering scope{};
            scope.area = info->renderArea; scope.layers = info->layerCount;
            if (info->colorAttachmentCount)
                scope.colors.assign(info->pColorAttachments, info->pColorAttachments + info->colorAttachmentCount);
            if (info->pDepthAttachment) { scope.has_depth = true; scope.depth = *info->pDepthAttachment; }
            if (info->pStencilAttachment) { scope.has_stencil = true; scope.stencil = *info->pStencilAttachment; }
            r.scopes.push_back(std::move(scope));
            r.events.emplace_back("begin");
        };
        vk.end_rendering = +[](VkCommandBuffer) {
            auto& r = *current;
            check(r.rendering, "end rendering has a matching begin");
            r.rendering = false; ++r.ends; r.events.emplace_back("end");
        };
        vk.pipeline_barrier = +[](VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                  std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                  std::uint32_t, const VkImageMemoryBarrier*) { current->outside("barrier"); };
        vk.pipeline_barrier2 = +[](VkCommandBuffer, const VkDependencyInfo*) { current->outside("barrier2"); };
        vk.bind_pipeline = +[](VkCommandBuffer, VkPipelineBindPoint point, VkPipeline pipeline) {
            if (point == VK_PIPELINE_BIND_POINT_COMPUTE) current->outside("bind_compute");
            else current->state.pipeline = pipeline;
        };
        vk.bind_descriptor_sets = +[](VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t,
                                      std::uint32_t count, const VkDescriptorSet* sets, std::uint32_t,
                                      const std::uint32_t*) {
            check(count == 1, "one bindless descriptor set is bound");
            current->state.descriptors = sets[0];
        };
        vk.push_constants = +[](VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t offset,
                                std::uint32_t size, const void* data) {
            check(offset == 0 && size == sizeof(ShaderUserData), "draw user-data block stays complete");
            std::memcpy(&current->state.data, data, sizeof(ShaderUserData));
        };
        vk.set_viewport = +[](VkCommandBuffer, std::uint32_t first, std::uint32_t count, const VkViewport* values) {
            check(first == 0, "viewports start at zero");
            current->state.viewports.assign(values, values + count);
        };
        vk.set_scissor = +[](VkCommandBuffer, std::uint32_t first, std::uint32_t count, const VkRect2D* values) {
            check(first == 0, "scissors start at zero");
            current->state.scissors.assign(values, values + count);
        };
        vk.set_viewport_with_count = +[](VkCommandBuffer, std::uint32_t count, const VkViewport* values) {
            current->state.viewports.assign(values, values + count);
        };
        vk.set_scissor_with_count = +[](VkCommandBuffer, std::uint32_t count, const VkRect2D* values) {
            current->state.scissors.assign(values, values + count);
        };
        vk.set_primitive_topology = +[](VkCommandBuffer, VkPrimitiveTopology value) { current->state.topology = value; };
        vk.set_primitive_restart_enable = +[](VkCommandBuffer, VkBool32 value) { current->state.restart = value; };
        vk.set_stencil_reference = +[](VkCommandBuffer, VkStencilFaceFlags face, std::uint32_t value) {
            if (face & VK_STENCIL_FACE_FRONT_BIT) current->state.front = value;
            if (face & VK_STENCIL_FACE_BACK_BIT) current->state.back = value;
        };
        vk.set_blend_constants = +[](VkCommandBuffer, const float* values) {
            std::copy_n(values, 4, current->state.blend.begin());
        };
        vk.bind_vertex_buffers2 = +[](VkCommandBuffer, std::uint32_t, std::uint32_t count, const VkBuffer* buffers,
                                     const VkDeviceSize* offsets, const VkDeviceSize* sizes, const VkDeviceSize* strides) {
            check(count == 1, "vertex buffer binding is recorded");
            auto& s = current->state;
            s.vertex = buffers[0]; s.vertex_offset = offsets[0]; s.vertex_size = sizes[0]; s.stride = strides[0];
        };
        vk.bind_index_buffer = +[](VkCommandBuffer, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
            current->state.index = buffer; current->state.index_offset = offset; current->state.index_type = type;
        };
        vk.draw = +[](VkCommandBuffer, std::uint32_t count, std::uint32_t instances, std::uint32_t first,
                      std::uint32_t first_instance) { current->draw("draw", count, instances, first, 0, first_instance); };
        vk.draw_indexed = +[](VkCommandBuffer, std::uint32_t count, std::uint32_t instances, std::uint32_t first,
                              std::int32_t base, std::uint32_t first_instance) {
            current->draw("indexed", count, instances, first, base, first_instance);
        };
        vk.draw_mesh_tasks = +[](VkCommandBuffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
            current->draw("mesh", x, y, z, 0, 0);
        };
        vk.draw_indirect = +[](VkCommandBuffer, VkBuffer, VkDeviceSize, std::uint32_t count, std::uint32_t stride) {
            current->draw("indirect", count, stride, 0, 0, 0);
        };
        vk.copy_buffer = +[](VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {
            current->outside("copy");
        };
        vk.dispatch = +[](VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) { current->outside("dispatch"); };
        vk.clear_color_image = +[](VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                   std::uint32_t, const VkImageSubresourceRange*) { current->outside("clear_image"); };
        vk.clear_attachments = +[](VkCommandBuffer, std::uint32_t, const VkClearAttachment*, std::uint32_t,
                                   const VkClearRect*) {
            check(current->rendering, "rectangle clear owns a rendering scope");
            current->events.emplace_back("clear_rect");
        };
        vk.reset_query_pool = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, std::uint32_t) { current->outside("reset_query"); };
        vk.begin_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t, VkQueryControlFlags) { current->outside("begin_query"); };
        vk.end_query = +[](VkCommandBuffer, VkQueryPool, std::uint32_t) { current->outside("end_query"); };
        vk.emit_quad_indices = [](VkCommandBuffer, const QuadIndexExpansion&) { current->outside("quad_helper"); };
        vk.emit_indirect_clamp = [](VkCommandBuffer, const ClampedIndirect&) { current->outside("clamp_helper"); };
        vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) { current->outside("capture_predicate"); };
        vk.begin_conditional_rendering = +[](VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
            current->outside("begin_predicate");
        };
        vk.end_conditional_rendering = +[](VkCommandBuffer) { current->outside("end_predicate"); };
        vk.begin_label = +[](VkCommandBuffer, const VkDebugUtilsLabelEXT*) { current->outside("begin_label"); };
        vk.end_label = +[](VkCommandBuffer) { current->outside("end_label"); };
        vk.insert_label = +[](VkCommandBuffer, const VkDebugUtilsLabelEXT*) { current->outside("insert_label"); };
        vk.quad_indices = handle<VkBuffer>(90);
        vk.quad_index_count = 12;
        vk.heap_base = 0x12340000;
        return vk;
    }
};
Recorder* Recorder::current = nullptr;

struct Fixture {
    Recorder recorder;
    ReplayFunctions vk = recorder.functions();
    GraphicsPipeline graphics{}, other{};
    QueryPool query{};
    CommandList list;
    PreparedList prepared;

    Fixture() {
        graphics.pipeline = handle<VkPipeline>(10);
        graphics.layout = handle<VkPipelineLayout>(11);
        graphics.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        graphics.bindless_set = handle<VkDescriptorSet>(12);
        graphics.code_base = 0x87650000;
        graphics.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        graphics.strip_cut = 1;
        other = graphics;
        query.pool = handle<VkQueryPool>(80);
        prepared.list = &list;
        prepared.statistics_pool = query.pool;
        prepared.statistics_mesh = {0, 0};
    }

    static detail::PreparedDraw::Attachment attachment(ResourceId id, bool depth = false) {
        detail::PreparedDraw::Attachment a{};
        a.id = id;
        a.binding.image = handle<VkImage>(100 + id);
        a.binding.format = depth ? VK_FORMAT_D32_SFLOAT_S8_UINT : VK_FORMAT_R8G8B8A8_UNORM;
        a.binding.aspects = depth ? depth_stencil_aspects : VK_IMAGE_ASPECT_COLOR_BIT;
        a.binding.width = 640; a.binding.height = 480;
        a.binding.array_layers = 4; a.binding.mip_levels = 2;
        a.range = {a.binding.aspects, 0, 1, 0, 1};
        a.view = handle<VkImageView>(200 + id);
        return a;
    }

    static Draw draw(std::uint32_t n = 0) {
        Draw d{};
        d.pipeline = 1; d.count = 3 + n; d.instances = 2; d.first = n; d.first_instance = 4 + n;
        d.topology = 4; // triangle list
        d.viewport = {float(n), float(n + 1), 320, 240, 0, 1};
        d.scissor = {std::int32_t(n), std::int32_t(n + 1), 300, 200};
        d.stencil_front = 10 + n; d.stencil_back = 20 + n;
        d.blend_factor = {float(n), 0.25f, 0.5f, 1};
        d.user_data.fill(0x100 + n);
        return d;
    }

    std::size_t add_draw(Draw draw = Fixture::draw(), const IndirectDrawArgs* indirect = nullptr) {
        check(indirect ? list.draw_indirect(std::move(draw), *indirect) : list.draw(draw), "draw records");
        const auto index = prepared.commands.size();
        auto& command = prepared.commands.emplace_back();
        command.draw = std::uint32_t(prepared.draws.size());
        auto& d = prepared.draws.emplace_back();
        d.graphics = &graphics; d.pipeline = graphics.pipeline;
        d.colors = {attachment(1)}; d.depth = attachment(2, true);
        d.area = {640, 480}; d.layers = 1;
        d.vertex_buffers = {handle<VkBuffer>(30)};
        d.vertex_offsets = {0}; d.vertex_sizes = {1024}; d.vertex_strides = {16};
        d.index_buffer = handle<VkBuffer>(31); d.index_offset = 32; d.index_type = VK_INDEX_TYPE_UINT16;
        d.indirect_buffer = handle<VkBuffer>(32);
        return index;
    }

    detail::PreparedDraw& at(std::size_t command) { return prepared.draws[prepared.commands[command].draw]; }
    detail::Prepared& add_command() { return prepared.commands.emplace_back(); }
    void emit() {
        if (!list.closed()) check(list.close(), "list closes");
        check(list.commands().size() == prepared.commands.size(), "prepared commands match the list");
        Recorder::current = &recorder;
        leonida::emit(prepared, VK_NULL_HANDLE, vk);
        check(!recorder.rendering && recorder.ends == recorder.scopes.size(), "list ends with all rendering scopes closed");
    }
};

std::vector<std::string> actions(const Recorder& r) {
    std::vector<std::string> out;
    for (const auto& event : r.events)
        if (event != "barrier" && event != "barrier2" && event != "bind_compute") out.push_back(event);
    return out;
}

void compatible_state() {
    Fixture f;
    f.add_draw(Fixture::draw());
    Draw second = Fixture::draw(1);
    second.indexed = true; second.topology = 5; second.base_vertex = -7;
    const auto id = f.add_draw(second);
    auto& d = f.at(id);
    d.vertex_buffers[0] = handle<VkBuffer>(40); d.vertex_offsets[0] = 128; d.vertex_sizes[0] = 512;
    d.vertex_strides[0] = 24; d.index_buffer = handle<VkBuffer>(41); d.index_offset = 64;
    f.emit();
    check(f.recorder.scopes.size() == 1, "consecutive compatible direct and indexed draws share one scope");
    check(actions(f.recorder) == std::vector<std::string>{"begin", "draw", "indexed", "end"}, "draw order is unchanged");
    check(std::count(f.recorder.events.begin(), f.recorder.events.end(), "barrier") == 2,
          "shared color and depth attachments are ordered once before the run");
    check(f.recorder.draws.size() == 2, "both draws are emitted");
    if (f.recorder.draws.size() != 2 || f.recorder.scopes.empty()) return;
    const auto& a = f.recorder.draws[0]; const auto& b = f.recorder.draws[1];
    check(a.data.s[0] == 0x100 && b.data.s[31] == 0x101 && b.data.heap_base == f.vk.heap_base &&
          b.data.code_base == f.graphics.code_base, "user data and address bases are pushed for each draw");
    check(a.viewports[0].x == 0 && b.viewports[0].x == 1 && b.viewports[0].y == 242 &&
          b.viewports[0].height == -240 && b.scissors[0].offset.x == 1 && b.scissors[0].offset.y == 2,
          "viewport and scissor changes reach the second draw");
    check(a.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST && b.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP &&
          !a.restart && b.restart, "topology and primitive restart change inside the shared scope");
    check(b.front == 11 && b.back == 21 && b.blend == second.blend_factor,
          "both stencil references and blend constants update for every draw");
    check(b.pipeline == f.graphics.pipeline && b.descriptors == f.graphics.bindless_set &&
          b.vertex == handle<VkBuffer>(40) && b.vertex_offset == 128 && b.vertex_size == 512 && b.stride == 24 &&
          b.index == handle<VkBuffer>(41) && b.index_offset == 64 && b.index_type == VK_INDEX_TYPE_UINT16,
          "pipeline, descriptors, vertex and index bindings remain per draw");
    check(b.count == second.count && b.instances == second.instances && b.first == second.first &&
          b.base == -7 && b.first_instance == second.first_instance, "indexed draw arguments preserve bases and instances");
    const auto& scope = f.recorder.scopes[0];
    check(scope.colors.size() == 1 && scope.colors[0].imageView == Fixture::attachment(1).view &&
          scope.colors[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD && scope.colors[0].storeOp == VK_ATTACHMENT_STORE_OP_STORE &&
          scope.has_depth && scope.has_stencil && scope.depth.imageView == scope.stencil.imageView &&
          scope.area.extent.width == 640 && scope.area.extent.height == 480 && scope.layers == 1,
          "merged rendering keeps the original attachment load/store, depth/stencil, extent and layers");
}

void changed_targets() {
    using Change = std::function<void(Fixture&, detail::PreparedDraw&)>;
    const std::vector<std::pair<const char*, Change>> changes = {
        {"different color image", [](auto&, auto& d) { d.colors[0] = Fixture::attachment(3); }},
        {"different color mip view", [](auto&, auto& d) { d.colors[0].view = handle<VkImageView>(301); d.colors[0].range.baseMipLevel = 1; }},
        {"different depth view", [](auto&, auto& d) { d.depth.view = handle<VkImageView>(302); }},
        {"unbound color slot", [](auto&, auto& d) { d.colors[0] = {}; }},
        {"additional MRT slot", [](auto&, auto& d) { d.colors.push_back(Fixture::attachment(3)); }},
        {"different color layout", [](auto&, auto& d) { d.colors[0].binding.simultaneous_access = true; }},
        {"different rendering width", [](auto&, auto& d) { d.area.width = 320; }},
        {"different rendering height", [](auto&, auto& d) { d.area.height = 240; }},
        {"different array layer count", [](auto&, auto& d) { d.layers = 2; d.colors[0].range.layerCount = 2; }},
        {"different array layer slice", [](auto&, auto& d) { d.colors[0].range.baseArrayLayer = 1; d.colors[0].view = handle<VkImageView>(303); }},
        {"different pipeline variant", [](auto&, auto& d) { d.pipeline = handle<VkPipeline>(50); }},
        {"different depth/stencil pipeline", [](auto& f, auto& d) { f.other.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT; d.graphics = &f.other; }},
        {"no depth attachment", [](auto&, auto& d) { d.depth = {}; }},
    };
    for (const auto& [description, change] : changes) {
        Fixture f;
        f.add_draw();
        const auto b = f.add_draw(); change(f, f.at(b));
        const auto c = f.add_draw(); f.at(c) = f.at(b);
        f.emit();
        check(f.recorder.scopes.size() == 2, description);
        check(actions(f.recorder) == std::vector<std::string>{"begin", "draw", "end", "begin", "draw", "draw", "end"},
              "a changed rendering configuration closes the old scope and begins a compatible new run");
    }
    Fixture mrt;
    for (int n = 0; n < 2; ++n) {
        auto& d = mrt.at(mrt.add_draw());
        d.colors = n ? std::vector{Fixture::attachment(3), Fixture::attachment(1)}
                     : std::vector{Fixture::attachment(1), Fixture::attachment(3)};
    }
    mrt.emit();
    check(mrt.recorder.scopes.size() == 2 && mrt.recorder.scopes[1].colors[0].imageView == Fixture::attachment(3).view,
          "MRT attachment order is significant");
}

void attachmentless_and_multiple_viewports() {
    Fixture f;
    f.graphics.viewport_with_count = true;
    f.graphics.depth_aspects = 0;
    f.vk.graphics_pipeline = [&](std::uintptr_t) -> const GraphicsPipeline* { return &f.graphics; };
    f.vk.buffer_at = [](std::uint64_t, std::uint64_t&, VkBuffer&, VkDeviceSize&) { return false; };
    f.vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) { return VkImageView{}; };
    for (std::uint32_t n = 0; n < 3; ++n) {
        auto draw = Fixture::draw(n);
        draw.viewport = {0, 0, 320, 240, 0, 1};
        auto views = std::make_shared<DrawViewports>();
        views->viewport_count = views->scissor_count = 2;
        views->viewports[0] = draw.viewport;
        views->viewports[1] = n == 2 ? Viewport{400, 20, 400, 32, 0, 1} : Viewport{10 + float(n), 20, 64, 32, 0, 1};
        views->scissors[0] = draw.scissor; views->scissors[1] = {10, 20, 74, 52};
        draw.viewports = views;
        check(f.list.draw(draw), "attachmentless draw records");
    }
    check(f.list.close(), "attachmentless list closes");
    check(prepare(f.list, [](ResourceId) -> const ResourceBinding* { return nullptr; }, f.vk, f.prepared) == ReplayResult::success,
          "attachmentless rendering extents prepare from real viewport snapshots");
    f.emit();
    check(f.recorder.scopes.size() == 2 && f.recorder.scopes[0].area.extent.width == 320 &&
          f.recorder.scopes[1].area.extent.width == 800,
          "attachmentless draws split when their viewport-derived rendering extent changes");
    check(f.recorder.draws.size() == 3 && f.recorder.draws[1].viewports.size() == 2 &&
          f.recorder.draws[1].viewports[1].x == 11 && f.recorder.draws[1].scissors.size() == 2,
          "multiple viewport/scissor snapshots update inside a shared scope");
}

void non_draw_boundaries() {
    for (int kind = 0; kind < 4; ++kind) {
        Fixture f;
        f.add_draw();
        auto& p = f.add_command();
        if (kind == 0) check(f.list.uav_barrier({0}), "UAV barrier records");
        if (kind == 1) {
            check(f.list.copy_buffer({5, 0, 6, 0, 32}), "copy records");
            p.binding.buffer = handle<VkBuffer>(60); p.source.buffer = handle<VkBuffer>(61);
            p.buffer_copies = {{0, 0, 32}};
        }
        if (kind == 2) {
            check(f.list.dispatch({2, {}, 1, 1, 1}), "dispatch records");
            p.pipeline.pipeline = handle<VkPipeline>(62); p.pipeline.layout = f.graphics.layout;
        }
        if (kind == 3) {
            check(f.list.clear_render_target({1, 0, 0, 1}, {0, 0, 0, 1}), "clear records");
            p.binding = Fixture::attachment(1).binding; p.range = Fixture::attachment(1).range;
            p.clear_format = p.binding.format;
        }
        f.add_draw(); f.add_draw(); f.emit();
        check(f.recorder.scopes.size() == 2 && f.recorder.draws.size() == 3,
              "barrier, copy, dispatch and image clear each split consecutive draw runs");
    }
    Fixture rect;
    rect.add_draw();
    check(rect.list.clear_render_target({1, 0, 0, 1}, {0, 0, 0, 1}), "rectangle clear records");
    auto& p = rect.add_command();
    p.binding = Fixture::attachment(1).binding; p.range = Fixture::attachment(1).range;
    p.view = Fixture::attachment(1).view; p.clear_format = p.binding.format;
    p.rects = {{{{0, 0}, {16, 16}}, 0, 1}};
    rect.add_draw(); rect.emit();
    check(actions(rect.recorder) == std::vector<std::string>{"begin", "draw", "end", "begin", "clear_rect", "end", "begin", "draw", "end"},
          "rectangle clears own a separate balanced rendering scope");
}

void ineligible_draws() {
    for (int kind = 0; kind < 6; ++kind) {
        Fixture f;
        f.add_draw();
        Draw draw = Fixture::draw();
        IndirectDrawArgs arguments{7, 0, 16, 1};
        if (kind == 2) { draw.indexed = true; draw.topology = topology_quad_list; draw.count = 8; }
        const auto b = f.add_draw(draw, kind == 1 ? &arguments : nullptr);
        auto& p = f.prepared.commands[b]; auto& d = f.at(b);
        if (kind == 0) d.skipped = true;
        if (kind == 2) { d.quad_indices.buffer = handle<VkBuffer>(70); d.quad_indices.quads = 2; }
        if (kind == 3) p.indirect_clamp.buffer = handle<VkBuffer>(71);
        if (kind == 4) p.statistics_end = 0;
        if (kind == 5) p.statistics_begin = 1;
        f.add_draw(); f.add_draw(); f.emit();
        check(f.recorder.scopes.size() == (kind == 0 ? 2u : 3u),
              "skipped, indirect, helper and statistics-endpoint draws cannot carry rendering into the next draw");
        auto expected = std::vector<std::string>{"begin", "draw", "end"};
        if (kind != 0) {
            if (kind == 2) expected.emplace_back("quad_helper");
            if (kind == 3) expected.emplace_back("clamp_helper");
            if (kind == 4) expected.emplace_back("end_query");
            if (kind == 5) { expected.emplace_back("reset_query"); expected.emplace_back("begin_query"); }
            expected.emplace_back("begin"); expected.emplace_back(kind == 1 ? "indirect" : kind == 2 ? "indexed" : "draw");
            expected.emplace_back("end");
        }
        expected.insert(expected.end(), {"begin", "draw", "draw", "end"});
        check(actions(f.recorder) == expected, "ineligible draw closes before helpers and immediately after its draw");
    }
    for (bool mesh : {false, true}) {
        Fixture f;
        f.graphics.mesh = mesh;
        for (int i = 0; i < 2; ++i) {
            auto draw = Fixture::draw(); draw.mesh = mesh; draw.first = 1;
            if (!mesh) { draw.topology = topology_quad_list; draw.count = 8; }
            f.add_draw(draw);
        }
        f.emit();
        check(f.recorder.scopes.size() == 1 && f.recorder.draws.size() == 2,
              "direct mesh and non-indexed quad draws without helpers can share a scope");
    }
}

void queries_predicates_labels_and_list_end() {
    Fixture f;
    f.add_draw();
    check(f.list.begin_query({1, d3d12_query::occlusion, 0}), "query begins"); f.add_command();
    f.prepared.queries.push_back({}); f.prepared.queries.back().command = 1; f.prepared.queries.back().heap = &f.query;
    f.add_draw(); f.add_draw();
    check(f.list.end_query({1, d3d12_query::occlusion, 0}), "query ends"); f.add_command();
    f.prepared.queries.push_back({}); f.prepared.queries.back().command = 4; f.prepared.queries.back().heap = &f.query;
    f.add_draw(); f.emit();
    check(actions(f.recorder) == std::vector<std::string>{"begin", "draw", "end", "reset_query", "begin_query", "begin", "draw", "draw", "end", "end_query", "begin", "draw", "end"},
          "query commands end rendering before query resets and endpoints");

    Fixture predicate;
    predicate.add_draw();
    check(predicate.list.set_predication({8, 0, d3d12_predication::equal_zero}), "predicate records");
    predicate.add_command().predicate.scratch = handle<VkBuffer>(81);
    predicate.add_draw(); predicate.add_draw(); predicate.emit();
    check(actions(predicate.recorder) == std::vector<std::string>{"begin", "draw", "end", "capture_predicate", "begin_predicate", "begin", "draw", "draw", "end", "end_predicate"},
          "predicate capture and final conditional-rendering closure execute outside rendering");

    Fixture label;
    check(label.list.marker(DebugMarker::Kind::begin, "synthetic pass"), "label records"); label.add_command();
    label.add_draw(); label.add_draw(); label.emit();
    check(actions(label.recorder) == std::vector<std::string>{"begin_label", "begin", "draw", "draw", "end", "end_label"},
          "final rendering scope closes before unbalanced labels are closed");
    label.emit();
    check(label.recorder.scopes.size() == 2 && label.recorder.ends == 2,
          "emitting another list never reuses a scope from the previous emit call");
}
}

int main() {
    compatible_state();
    changed_targets();
    attachmentless_and_multiple_viewports();
    non_draw_boundaries();
    ineligible_draws();
    queries_predicates_labels_and_list_end();
    if (ok) std::puts("replay rendering scopes: PASS (synthetic CPU recording)");
    return ok ? 0 : 1;
}
