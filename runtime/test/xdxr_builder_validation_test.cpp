// Unit tests for XdxrBuilder pre-validation in src/xdxr_build.h (no Vulkan instance, no GPU).
// Mismatched geometry sums, null device addresses, uncomputed layouts, and 32-bit overflow counts must fail before
// recording any command into the Vulkan command buffer. No dispatch has more than XdxrBuilder::max_groups (65535)
// workgroups along X: larger steps (copies and refits over 16 MiB, builds of over 4M elements) are split with
// vkCmdDispatchBase into consecutive ranges of groups.
#include "../src/xdxr_build.h"

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <vector>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace {

template<typename Handle>
Handle fake_handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(value);
    else
        return static_cast<Handle>(value);
}

struct MockRecorder {
    unsigned binds = 0;
    unsigned pushes = 0;
    unsigned dispatches = 0;
    unsigned barriers = 0;
    std::uint32_t last_group_count_x = 0;
    std::uint32_t last_copy_count = 0;
    std::uint32_t max_group_count_x = 0;
    // One entry per vkCmdPushConstants: the groups its dispatches cover, and whether they were consecutive from 0.
    std::vector<std::uint64_t> groups_per_push;
    bool contiguous = true;
    VkPipelineCreateFlags pipeline_flags = 0;

    void reset() {
        const VkPipelineCreateFlags flags = pipeline_flags;
        *this = MockRecorder{};
        pipeline_flags = flags;
    }
    void record(std::uint32_t base, std::uint32_t x) {
        ++dispatches;
        last_group_count_x = x;
        max_group_count_x = std::max(max_group_count_x, x);
        if (groups_per_push.empty() || base != groups_per_push.back()) contiguous = false;
        if (!groups_per_push.empty()) groups_per_push.back() += x;
    }
    bool empty() const { return !binds && !pushes && !dispatches && !barriers; }
};

MockRecorder g_rec;

VKAPI_ATTR VkResult VKAPI_CALL mock_create_shader_module(VkDevice, const VkShaderModuleCreateInfo*,
                                                         const VkAllocationCallbacks*, VkShaderModule* out) {
    *out = fake_handle<VkShaderModule>(1);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_shader_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_pipeline_layout(VkDevice, const VkPipelineLayoutCreateInfo*,
                                                           const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = fake_handle<VkPipelineLayout>(2);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL mock_create_compute_pipelines(VkDevice, VkPipelineCache, std::uint32_t count,
                                                             const VkComputePipelineCreateInfo* info,
                                                             const VkAllocationCallbacks*, VkPipeline* out) {
    for (std::uint32_t i = 0; i < count; ++i) out[i] = fake_handle<VkPipeline>(3 + i);
    if (count) g_rec.pipeline_flags = info[0].flags;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mock_destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}

VKAPI_ATTR void VKAPI_CALL mock_cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) { ++g_rec.binds; }

VKAPI_ATTR void VKAPI_CALL mock_cmd_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                                                   std::uint32_t, std::uint32_t size, const void* values) {
    ++g_rec.pushes;
    g_rec.groups_per_push.push_back(0);
    if (size >= 80 && values) {
        std::uint32_t mode = 0, count = 0;
        std::memcpy(&mode, static_cast<const std::uint8_t*>(values) + 48, 4);
        std::memcpy(&count, static_cast<const std::uint8_t*>(values) + 52, 4);
        if (mode == 12) g_rec.last_copy_count = count;
    }
}

VKAPI_ATTR void VKAPI_CALL mock_cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t, std::uint32_t) {
    g_rec.record(0, x);
}
VKAPI_ATTR void VKAPI_CALL mock_cmd_dispatch_base(VkCommandBuffer, std::uint32_t base_x, std::uint32_t base_y,
                                                  std::uint32_t base_z, std::uint32_t x, std::uint32_t y,
                                                  std::uint32_t z) {
    if (base_y || base_z || y != 1 || z != 1) g_rec.contiguous = false;
    g_rec.record(base_x, x);
}

VKAPI_ATTR void VKAPI_CALL mock_cmd_pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                                     VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*,
                                                     std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                                                     const VkImageMemoryBarrier*) {
    ++g_rec.barriers;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_get_device_proc_addr(VkDevice, const char* name) {
    if (!name) return nullptr;
    if (std::strcmp(name, "vkCreateShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_create_shader_module);
    if (std::strcmp(name, "vkDestroyShaderModule") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_shader_module);
    if (std::strcmp(name, "vkCreatePipelineLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_create_pipeline_layout);
    if (std::strcmp(name, "vkDestroyPipelineLayout") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline_layout);
    if (std::strcmp(name, "vkCreateComputePipelines") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_create_compute_pipelines);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_destroy_pipeline);
    if (std::strcmp(name, "vkCmdBindPipeline") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_bind_pipeline);
    if (std::strcmp(name, "vkCmdPushConstants") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_push_constants);
    if (std::strcmp(name, "vkCmdDispatch") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_dispatch);
    if (std::strcmp(name, "vkCmdDispatchBase") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_dispatch_base);
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0) return reinterpret_cast<PFN_vkVoidFunction>(mock_cmd_pipeline_barrier);
    return nullptr;
}

}  // namespace

int main() {
    using namespace leonida;

    const VkDevice dev = fake_handle<VkDevice>(0x1000);
    const VkCommandBuffer cb = fake_handle<VkCommandBuffer>(0x2000);
    XdxrBuilder builder(mock_get_device_proc_addr, dev);
    CHECK(builder.loaded());
    CHECK((g_rec.pipeline_flags & VK_PIPELINE_CREATE_DISPATCH_BASE_BIT) != 0);   // vkCmdDispatchBase needs it

    XdxrLayout blas1;
    CHECK(blas1.compute(1, false));
    XdxrLayout blas4;
    CHECK(blas4.compute(4, false));
    XdxrLayout tlas0;
    CHECK(tlas0.compute(0, true));
    XdxrLayout tlas4;
    CHECK(tlas4.compute(4, true));
    const XdxrLayout uncomputed{};

    const std::uint64_t dst = 0x100000;
    const std::uint64_t scratch = 0x200000;
    const std::uint64_t instances = 0x300000;

    // 1. Fewer triangles than layout.elements must fail cleanly without recording any commands.
    g_rec.reset();
    CHECK(!builder.record_blas(cb, blas4, {XdxrTriangles{.triangle_count = 3}}, dst, scratch));
    CHECK(g_rec.empty());

    // 2. More triangles than layout.elements must fail cleanly without recording any commands.
    g_rec.reset();
    CHECK(!builder.record_blas(cb, blas4,
                               {XdxrTriangles{.triangle_count = 2}, XdxrTriangles{.triangle_count = 3}},
                               dst, scratch));
    CHECK(g_rec.empty());

    // 3. Sums that would wrap uint32_t back to layout.elements must be rejected before any dispatch.
    g_rec.reset();
    CHECK(!builder.record_blas(cb, blas1,
                               {XdxrTriangles{.triangle_count = 0x80000001u},
                                XdxrTriangles{.triangle_count = 0x80000000u}},
                               dst, scratch));
    CHECK(g_rec.empty());

    // 4. Null addresses, uncomputed layouts, and non-empty TLAS with null instances record nothing.
    g_rec.reset();
    CHECK(!builder.record_blas(cb, blas1, {XdxrTriangles{.triangle_count = 1}}, 0, scratch));
    CHECK(!builder.record_blas(cb, blas1, {XdxrTriangles{.triangle_count = 1}}, dst, 0));
    CHECK(!builder.record_blas(cb, uncomputed, {}, dst, scratch));
    CHECK(!builder.record_tlas(cb, tlas4, instances, 0, scratch));
    CHECK(!builder.record_tlas(cb, tlas4, instances, dst, 0));
    CHECK(!builder.record_tlas(cb, tlas4, 0, dst, scratch));
    CHECK(!builder.record_tlas(cb, uncomputed, instances, dst, scratch));
    CHECK(g_rec.empty());

    // 5. Empty TLAS with null instances pointer is valid and records a build.
    g_rec.reset();
    CHECK(builder.record_tlas(cb, tlas0, 0, dst, scratch));
    CHECK(g_rec.binds > 0 && g_rec.dispatches > 0);

    // 6. Valid multi-geometry BLAS records dispatches.
    g_rec.reset();
    CHECK(builder.record_blas(cb, blas4,
                              {XdxrTriangles{.triangle_count = 1}, XdxrTriangles{.triangle_count = 3}},
                              dst, scratch));
    CHECK(g_rec.binds > 0 && g_rec.dispatches > 0 && g_rec.pushes == g_rec.dispatches);

    CHECK(g_rec.max_group_count_x <= XdxrBuilder::max_groups && g_rec.contiguous);
    // 7. A large record_copy (one word a thread) computes its group count without 32-bit overflow, 16,777,216
    //    groups, and records it as consecutive dispatches of at most 65535 groups after one push.
    g_rec.reset();
    CHECK(builder.record_copy(cb, dst, dst + 0x10000, 0xffffffffull));
    CHECK(g_rec.last_copy_count == 0x3fffffffu && g_rec.pushes == 1);
    CHECK(g_rec.groups_per_push.size() == 1 && g_rec.groups_per_push[0] == (0x3fffffffull + 63u) / 64u);
    CHECK(g_rec.dispatches == 257 && g_rec.max_group_count_x == XdxrBuilder::max_groups && g_rec.contiguous);
    CHECK(g_rec.last_group_count_x == 16777216u - 256u * 65535u);
    // 8. Exactly 65535 groups (a copy of 65535 * 256 bytes) is still one plain dispatch; one group more splits.
    g_rec.reset();
    CHECK(builder.record_copy(cb, dst, dst + 0x10000, 65535ull * 256));
    CHECK(g_rec.dispatches == 1 && g_rec.last_group_count_x == 65535u && g_rec.contiguous);
    g_rec.reset();
    CHECK(builder.record_copy(cb, dst, dst + 0x10000, 65535ull * 256 + 4));
    CHECK(g_rec.dispatches == 2 && g_rec.groups_per_push[0] == 65536u && g_rec.last_group_count_x == 1u &&
          g_rec.contiguous);
    // 9. A refit into another buffer copies the source first, one word a thread: over 16 MiB its copy is split too,
    //    as is every other step of a large build.
    XdxrLayout big;
    CHECK(big.compute(5000000, false) && big.size > 65535ull * 256);
    g_rec.reset();
    CHECK(builder.record_blas(cb, big, {XdxrTriangles{.triangle_count = 5000000}}, dst, scratch, dst + (1ull << 32)));
    CHECK(g_rec.max_group_count_x <= XdxrBuilder::max_groups && g_rec.contiguous);
    bool split_copy = false;
    for (const std::uint64_t groups : g_rec.groups_per_push) split_copy |= groups == (big.size / 4 + 63) / 64;
    CHECK(split_copy);
    g_rec.reset();
    CHECK(builder.record_blas(cb, big, {XdxrTriangles{.triangle_count = 5000000}}, dst, scratch));
    CHECK(g_rec.max_group_count_x <= XdxrBuilder::max_groups && g_rec.contiguous);

    std::puts("xdxr_builder_validation: all checks passed");
    return 0;
}
