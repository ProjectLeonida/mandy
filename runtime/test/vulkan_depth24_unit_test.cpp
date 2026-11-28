// Pure-CPU unit tests for runtime/src/vulkan_depth24.h (Depth24Converter):
//   - null PFN_vkGetDeviceProcAddr / VK_NULL_HANDLE device guards in load() and destructor;
//   - 64-bit overflow guards in prepare() (row/slice/span/offset/scratch sizing, max_range <= slack);
//   - exact O(1) slice and row partitioning parity for both D24 and rgb96 layouts;
//   - atomic rollback of parts and caller scratch fields when make_part() fails mid-loop;
//   - per-semaphore cached vkGetSemaphoreCounterValue queries, error handling, spare reuse cap,
//     ticket-0 immediate reclaim, and in-place compaction in collect() and forget();
//   - overflow-safe dispatch workgroup counts in run().
// No GPU, Vulkan loader, or game files required.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_depth24.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

template<typename Handle>
Handle make_handle(std::uint64_t id) {
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(id));
}

struct MockVk {
    std::uint64_t next_handle = 1;
    int create_buffer_calls = 0;
    int destroy_buffer_calls = 0;
    int free_memory_calls = 0;
    int create_pool_calls = 0;
    int destroy_pool_calls = 0;
    int alloc_set_calls = 0;
    int fail_alloc_set_at = -1;
    int fail_create_pool_at = -1;
    int sem_queries = 0;
    std::unordered_map<VkSemaphore, std::uint64_t> sem_values;
    std::unordered_map<VkSemaphore, VkResult> sem_results;
    std::vector<std::array<std::uint32_t, 3>> dispatches;
};

MockVk* g_vk = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL m_create_shader(VkDevice, const VkShaderModuleCreateInfo*,
                                               const VkAllocationCallbacks*, VkShaderModule* out) {
    *out = make_handle<VkShaderModule>(++g_vk->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_shader(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_dsl(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                            const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    *out = make_handle<VkDescriptorSetLayout>(++g_vk->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_dsl(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pl(VkDevice, const VkPipelineLayoutCreateInfo*,
                                           const VkAllocationCallbacks*, VkPipelineLayout* out) {
    *out = make_handle<VkPipelineLayout>(++g_vk->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pl(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pipe(VkDevice, VkPipelineCache, std::uint32_t,
                                             const VkComputePipelineCreateInfo*,
                                             const VkAllocationCallbacks*, VkPipeline* out) {
    *out = make_handle<VkPipeline>(++g_vk->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pipe(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_pool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                             const VkAllocationCallbacks*, VkDescriptorPool* out) {
    ++g_vk->create_pool_calls;
    if (g_vk->fail_create_pool_at == g_vk->create_pool_calls) return VK_ERROR_OUT_OF_POOL_MEMORY;
    *out = make_handle<VkDescriptorPool>(++g_vk->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {
    ++g_vk->destroy_pool_calls;
}
VKAPI_ATTR VkResult VKAPI_CALL m_alloc_sets(VkDevice, const VkDescriptorSetAllocateInfo* info,
                                            VkDescriptorSet* out) {
    ++g_vk->alloc_set_calls;
    if (g_vk->fail_alloc_set_at > 0 && g_vk->alloc_set_calls >= g_vk->fail_alloc_set_at)
        return VK_ERROR_OUT_OF_POOL_MEMORY;
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i)
        out[i] = make_handle<VkDescriptorSet>(++g_vk->next_handle);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_update_sets(VkDevice, std::uint32_t, const VkWriteDescriptorSet*,
                                         std::uint32_t, const VkCopyDescriptorSet*) {}
VKAPI_ATTR VkResult VKAPI_CALL m_create_buf(VkDevice, const VkBufferCreateInfo* info,
                                            const VkAllocationCallbacks*, VkBuffer* out) {
    ++g_vk->create_buffer_calls;
    *out = make_handle<VkBuffer>(info->size ? info->size : 1);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_destroy_buf(VkDevice, VkBuffer, const VkAllocationCallbacks*) {
    ++g_vk->destroy_buffer_calls;
}
VKAPI_ATTR void VKAPI_CALL m_mem_req(VkDevice, VkBuffer buf, VkMemoryRequirements* req) {
    *req = {};
    req->size = static_cast<VkDeviceSize>(reinterpret_cast<std::uintptr_t>(buf));
    req->alignment = 256;
    req->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL m_bind_buf(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL m_free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {
    ++g_vk->free_memory_calls;
}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_pipe(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout,
                                           std::uint32_t, std::uint32_t, const VkDescriptorSet*,
                                           std::uint32_t, const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags,
                                      std::uint32_t, std::uint32_t, const void*) {}
VKAPI_ATTR void VKAPI_CALL m_cmd_dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    g_vk->dispatches.push_back({x, y, z});
}
VKAPI_ATTR VkResult VKAPI_CALL m_get_sem(VkDevice, VkSemaphore sem, std::uint64_t* out) {
    ++g_vk->sem_queries;
    const auto res_it = g_vk->sem_results.find(sem);
    const VkResult res = res_it != g_vk->sem_results.end() ? res_it->second : VK_SUCCESS;
    if (out) {
        const auto val_it = g_vk->sem_values.find(sem);
        *out = val_it != g_vk->sem_values.end() ? val_it->second : 0;
    }
    return res;
}
VKAPI_ATTR VkResult VKAPI_CALL m_wait_idle(VkDevice) { return VK_SUCCESS; }

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mock_gdpa(VkDevice, const char* name) {
#define MATCH(sym, fn) if (std::strcmp(name, sym) == 0) return reinterpret_cast<PFN_vkVoidFunction>(fn)
    MATCH("vkCreateShaderModule", m_create_shader);
    MATCH("vkDestroyShaderModule", m_destroy_shader);
    MATCH("vkCreateDescriptorSetLayout", m_create_dsl);
    MATCH("vkDestroyDescriptorSetLayout", m_destroy_dsl);
    MATCH("vkCreatePipelineLayout", m_create_pl);
    MATCH("vkDestroyPipelineLayout", m_destroy_pl);
    MATCH("vkCreateComputePipelines", m_create_pipe);
    MATCH("vkDestroyPipeline", m_destroy_pipe);
    MATCH("vkCreateDescriptorPool", m_create_pool);
    MATCH("vkDestroyDescriptorPool", m_destroy_pool);
    MATCH("vkAllocateDescriptorSets", m_alloc_sets);
    MATCH("vkUpdateDescriptorSets", m_update_sets);
    MATCH("vkCreateBuffer", m_create_buf);
    MATCH("vkDestroyBuffer", m_destroy_buf);
    MATCH("vkGetBufferMemoryRequirements", m_mem_req);
    MATCH("vkBindBufferMemory", m_bind_buf);
    MATCH("vkFreeMemory", m_free_mem);
    MATCH("vkCmdBindPipeline", m_cmd_bind_pipe);
    MATCH("vkCmdBindDescriptorSets", m_cmd_bind_sets);
    MATCH("vkCmdPushConstants", m_cmd_push);
    MATCH("vkCmdDispatch", m_cmd_dispatch);
    MATCH("vkGetSemaphoreCounterValue", m_get_sem);
    MATCH("vkDeviceWaitIdle", m_wait_idle);
#undef MATCH
    return nullptr;
}

VkDeviceMemory mock_alloc(const VkMemoryRequirements&) {
    return make_handle<VkDeviceMemory>(++g_vk->next_handle);
}

}  // namespace

int main() {
    using namespace leonida;
    const VkDevice dev = make_handle<VkDevice>(1);
    const VkBuffer app_buf = make_handle<VkBuffer>(100);
    const VkBuffer ext_scratch = make_handle<VkBuffer>(200);
    const VkCommandBuffer cb = make_handle<VkCommandBuffer>(1);

    // 1. Null get or null device fails load() cleanly without dereferencing get or crashing in destructor.
    {
        MockVk state;
        g_vk = &state;
        Depth24Converter null_get(nullptr, dev, mock_alloc);
        CHECK(!null_get.loaded());
        Depth24Converter null_dev(mock_gdpa, VK_NULL_HANDLE, mock_alloc);
        CHECK(!null_dev.loaded());
        Depth24Job job{};
        job.app = app_buf;
        job.width = job.height = job.slices = 1;
        job.app_row_words = job.app_slice_words = 1;
        job.app_size = 16;
        std::vector<Depth24Job> parts;
        CHECK(!null_get.prepare(job, parts));
        null_get.run(cb, job);
        CHECK(state.dispatches.empty());
    }

    // 2. Overflow guards & max_range <= slack rejection before allocating any scratch buffer.
    {
        MockVk state;
        g_vk = &state;
        Depth24Converter tiny_range(mock_gdpa, dev, mock_alloc, 256);
        CHECK(tiny_range.loaded());
        Depth24Job ok_job{};
        ok_job.app = app_buf;
        ok_job.width = ok_job.height = ok_job.slices = 1;
        ok_job.app_row_words = ok_job.app_slice_words = 1;
        ok_job.app_size = 64;
        std::vector<Depth24Job> parts;
        CHECK(!tiny_range.prepare(ok_job, parts));
        CHECK(state.create_buffer_calls == 0);

        Depth24Converter conv(mock_gdpa, dev, mock_alloc, 1u << 20);
        // (height - 1) * app_row_words + aw overflow in rgb96 mode
        Depth24Job ov = ok_job;
        ov.rgb96 = true;
        ov.width = 0xffffffffu;
        ov.height = 0xffffffffu;
        ov.app_row_words = 0xffffffffu;
        ov.app_slice_words = 0xffffffffu;
        ov.app_size = UINT64_MAX;
        CHECK(!conv.prepare(ov, parts));

        // span_words * 4 overflow
        ov = ok_job;
        ov.width = 4;
        ov.height = 4;
        ov.slices = 0xffffffffu;
        ov.app_row_words = 4;
        ov.app_slice_words = 0xffffffffu;
        ov.app_size = UINT64_MAX;
        CHECK(!conv.prepare(ov, parts));

        // app_offset + span_bytes overflow
        ov = ok_job;
        ov.app_offset = UINT64_MAX - 3;
        ov.app_size = UINT64_MAX;
        CHECK(!conv.prepare(ov, parts));

        // Caller scratch_offset misaligned or overflowing
        ov = ok_job;
        ov.scratch = ext_scratch;
        ov.scratch_offset = 2;
        CHECK(!conv.prepare(ov, parts));
        ov.scratch_offset = UINT64_MAX - 3;  // 0xfffffffffffffffc (4-aligned, + 4 wraps)
        CHECK(!conv.prepare(ov, parts));
        CHECK(state.create_buffer_calls == 0);
    }

    // 3. Exact partitioning parity (D24 pitched 1 KiB -> 6 parts; rgb96 pitched 1 KiB -> 15 parts).
    {
        MockVk state;
        g_vk = &state;
        Depth24Converter tiny(mock_gdpa, dev, mock_alloc, 1024);
        std::vector<Depth24Job> parts;

        Depth24Job d24{};
        d24.app = app_buf;
        d24.scratch = ext_scratch;
        d24.app_offset = 1100;
        d24.app_size = 16384;
        d24.width = 37;
        d24.height = 5;
        d24.slices = 3;
        d24.app_row_words = 64;
        d24.app_slice_words = 400;
        CHECK(tiny.prepare(d24, parts));
        CHECK(parts.size() == 6);
        CHECK(parts[0].height == 3 && parts[1].height == 2);

        Depth24Job rgb{};
        rgb.rgb96 = true;
        rgb.app = app_buf;
        rgb.scratch = ext_scratch;
        rgb.app_offset = 1100;
        rgb.app_size = 16384;
        rgb.width = 37;
        rgb.height = 5;
        rgb.slices = 3;
        rgb.app_row_words = 128;
        rgb.app_slice_words = 700;
        CHECK(tiny.prepare(rgb, parts));
        CHECK(parts.size() == 15);
        for (const auto& p : parts) CHECK(p.height == 1 && p.slices == 1);
    }

    // 4. Atomic rollback of parts and whole.scratch when descriptor allocation fails mid-loop.
    {
        MockVk state;
        state.fail_alloc_set_at = 3;  // fail on 3rd part
        g_vk = &state;
        Depth24Converter tiny(mock_gdpa, dev, mock_alloc, 1024);
        Depth24Job d24{};
        d24.app = app_buf;
        d24.app_offset = 0;
        d24.app_size = 16384;
        d24.width = 37;
        d24.height = 5;
        d24.slices = 3;
        d24.app_row_words = 64;
        d24.app_slice_words = 400;
        std::vector<Depth24Job> parts(1);
        CHECK(!tiny.prepare(d24, parts));
        CHECK(parts.empty());
        CHECK(d24.scratch == VK_NULL_HANDLE && d24.scratch_offset == 0);
    }

    // 5. Per-semaphore cached queries + O(N) compaction + error handling + spare reuse in collect() & forget().
    {
        MockVk state;
        g_vk = &state;
        const VkSemaphore sem_a = make_handle<VkSemaphore>(10);
        const VkSemaphore sem_b = make_handle<VkSemaphore>(20);
        const VkSemaphore sem_lost = make_handle<VkSemaphore>(30);
        state.sem_values[sem_a] = 5;
        state.sem_values[sem_b] = 1;
        state.sem_values[sem_lost] = UINT64_MAX;
        state.sem_results[sem_lost] = VK_ERROR_DEVICE_LOST;

        Depth24Converter conv(mock_gdpa, dev, mock_alloc, 1u << 20);
        const auto submit_one = [&](VkSemaphore sem, std::uint64_t ticket) {
            Depth24Job j{};
            j.app = app_buf;
            j.width = 16;
            j.height = 16;
            j.slices = 1;
            j.app_row_words = 16;
            j.app_slice_words = 256;
            j.app_size = 4096;
            std::vector<Depth24Job> parts;
            CHECK(conv.prepare(j, parts));
            conv.retire(sem, ticket);
            return 0;
        };

        // 6 retired batches on sem_a (tickets 1..6), 2 on sem_b (1, 2), 3 on sem_lost (1..3), 1 ticket-0 batch.
        for (std::uint64_t t = 1; t <= 6; ++t) CHECK(submit_one(sem_a, t) == 0);
        CHECK(submit_one(sem_b, 1) == 0);
        CHECK(submit_one(sem_b, 2) == 0);
        for (std::uint64_t t = 1; t <= 3; ++t) CHECK(submit_one(sem_lost, t) == 0);
        CHECK(submit_one(VK_NULL_HANDLE, 0) == 0);

        // Single collect() must query each distinct semaphore at most ONCE (3 total, not 11!).
        state.sem_queries = 0;
        conv.collect();
        CHECK(state.sem_queries == 3);
        // Completed: 5 on sem_a + 1 on sem_b + 1 ticket-0 = 7 pools destroyed; spare cap 4 keeps 4 chunks and destroys 3.
        CHECK(state.destroy_pool_calls == 7);
        CHECK(state.destroy_buffer_calls == 3);
        // sem_lost failed with DEVICE_LOST even though *out == UINT64_MAX: none of its 3 batches were reclaimed.
        // Next prepare() reuses a spare chunk without creating a new VkBuffer.
        const int bufs_before = state.create_buffer_calls;
        CHECK(submit_one(sem_a, 7) == 0);
        CHECK(state.create_buffer_calls == bufs_before);

        // forget(sem_lost) releases all 3 batches on sem_lost cleanly.
        conv.forget(sem_lost);
        CHECK(state.destroy_pool_calls == 10);
        CHECK(state.destroy_buffer_calls == 6);
    }

    // 6. Overflow-safe workgroup dispatch counts in run() for width near UINT32_MAX.
    {
        MockVk state;
        g_vk = &state;
        Depth24Converter conv(mock_gdpa, dev, mock_alloc, 1u << 20);
        // Trigger pipeline creation via a tiny valid prepare().
        Depth24Job init{};
        init.app = app_buf;
        init.scratch = ext_scratch;
        init.width = init.height = init.slices = 1;
        init.app_row_words = init.app_slice_words = 1;
        init.app_size = 16;
        std::vector<Depth24Job> parts;
        CHECK(conv.prepare(init, parts));

        Depth24Job huge{};
        huge.width = UINT32_MAX;
        huge.height = UINT32_MAX - 1;
        huge.slices = 2;
        huge.set = parts[0].set;
        conv.run(cb, huge);
        CHECK(state.dispatches.size() == 1);
        CHECK(state.dispatches[0][0] == 0x20000000u);
        CHECK(state.dispatches[0][1] == 0x20000000u);
        CHECK(state.dispatches[0][2] == 2u);
    }

    std::puts("vulkan_depth24_unit_test: all checks passed");
    return 0;
}
