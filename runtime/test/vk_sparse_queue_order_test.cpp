#include "../src/vulkan_backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using namespace leonida;

template<class Handle>
Handle handle(std::uintptr_t value) {
    if constexpr (std::is_pointer_v<Handle>) return reinterpret_cast<Handle>(value);
    else return static_cast<Handle>(value);
}

struct Submission {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize memory_offset = 0;
    VkDeviceSize resource_offset = 0;
    VkSemaphore fence = VK_NULL_HANDLE;
    std::vector<VkSemaphore> waits;
    std::vector<std::uint64_t> values;

    bool waits_for(VkSemaphore semaphore, std::uint64_t value) const {
        for (std::size_t i = 0; i < waits.size(); ++i)
            if (waits[i] == semaphore && values[i] == value) return true;
        return false;
    }
};

std::vector<Submission> submissions;

PFN_vkVoidFunction VKAPI_CALL no_proc(VkDevice, const char*) { return nullptr; }

VkResult VKAPI_CALL create_pool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*,
                                VkCommandPool* out) {
    *out = handle<VkCommandPool>(1);
    return VK_SUCCESS;
}

VkResult VKAPI_CALL create_semaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*,
                                     VkSemaphore* out) {
    *out = handle<VkSemaphore>(1);
    return VK_SUCCESS;
}

void VKAPI_CALL destroy_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
void VKAPI_CALL destroy_semaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {}
void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {}
VkResult VKAPI_CALL idle(VkQueue) { return VK_SUCCESS; }

VkResult VKAPI_CALL completed(VkDevice, VkSemaphore, std::uint64_t* out) {
    *out = 0;
    return VK_SUCCESS;
}

void record_waits(Submission& submission, const void* next, std::uint32_t count, const VkSemaphore* waits) {
    if (!count) return;
    const auto& timeline = *static_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
    submission.waits.assign(waits, waits + count);
    submission.values.assign(timeline.pWaitSemaphoreValues, timeline.pWaitSemaphoreValues + count);
}

VkResult VKAPI_CALL queue_submit(VkQueue, std::uint32_t, const VkSubmitInfo* info, VkFence) {
    auto& submission = submissions.emplace_back();
    record_waits(submission, info->pNext, info->waitSemaphoreCount, info->pWaitSemaphores);
    if (info->signalSemaphoreCount > 1) submission.fence = info->pSignalSemaphores[1];
    return VK_SUCCESS;
}

VkResult VKAPI_CALL bind_sparse(VkQueue, std::uint32_t, const VkBindSparseInfo* info, VkFence) {
    auto& submission = submissions.emplace_back();
    record_waits(submission, info->pNext, info->waitSemaphoreCount, info->pWaitSemaphores);
    const auto& buffer = info->pBufferBinds[0];
    submission.buffer = buffer.buffer;
    submission.memory = buffer.pBinds[0].memory;
    submission.memory_offset = buffer.pBinds[0].memoryOffset;
    submission.resource_offset = buffer.pBinds[0].resourceOffset;
    return VK_SUCCESS;
}

void install_calls(VulkanContext& context) {
    context.vk.vkCreateCommandPool = create_pool;
    context.vk.vkDestroyCommandPool = destroy_pool;
    context.vk.vkCreateSemaphore = create_semaphore;
    context.vk.vkDestroySemaphore = destroy_semaphore;
    context.vk.vkDestroyBuffer = destroy_buffer;
    context.vk.vkQueueWaitIdle = idle;
    context.vk.vkGetSemaphoreCounterValue = completed;
    context.vk.vkQueueSubmit = queue_submit;
    context.vk.vkQueueBindSparse = bind_sparse;
}

VkBuffer add_sparse_buffer(VulkanContext& context, ResourceId id, std::uintptr_t buffer_id) {
    auto sparse = std::make_shared<SparseResource>(context.device(), context.vk);
    sparse->buffer = handle<VkBuffer>(buffer_id);
    sparse->tile_count = 2;
    sparse->memory_types = 1;
    ResourceBinding binding{};
    binding.buffer = sparse->buffer;
    binding.size = 2 * tile_size_in_bytes;
    binding.sparse_buffer = true;
    context.add_sparse_resource(id, binding, std::move(sparse));
    return binding.buffer;
}

}

int main() {
    VulkanContext context(no_proc, handle<VkDevice>(1), false);
    install_calls(context);
    VulkanQueue queue(context, handle<VkQueue>(1), 0);
    EXPECT(queue.ok());

    constexpr ResourceId source = 1, target = 2;
    const auto source_buffer = add_sparse_buffer(context, source, 1);
    const auto target_buffer = add_sparse_buffer(context, target, 2);
    const auto memory = handle<VkDeviceMemory>(1);
    auto heap = std::make_shared<TileHeapAllocation>(TileHeapAllocation{memory, 0, 2 * tile_size_in_bytes, 0});
    const auto get_heap = [&](bool, std::uint32_t) { return heap; };
    const auto progress = handle<VkSemaphore>(1);
    const auto previous = handle<VkSemaphore>(2), gate = handle<VkSemaphore>(3), done = handle<VkSemaphore>(4);
    context.holds().add_fence(gate, 0);
    const std::array first{TileRegion{0, 1}};
    std::array regions = first;
    std::array ranges{TileRange{TileRangeMode::sequential, 1, 1}};

    queue.wait(gate, 1);
    EXPECT(queue.signal(previous, 1));
    EXPECT(queue.update_tiles(source, regions, ranges, get_heap));
    EXPECT(queue.copy_tile_mappings(target, regions, source, regions));
    EXPECT(queue.signal(done, 1));
    EXPECT(submissions.empty());
    regions[0].first = 1;
    ranges[0].first = 0;
    heap.reset();
    context.holds().signaled(gate, 1);
    context.holds().pump_ready();
    EXPECT(submissions.size() == 4);
    EXPECT(submissions[0].fence == previous && submissions[0].waits_for(gate, 1));
    EXPECT(submissions[1].buffer == source_buffer && submissions[1].memory == memory);
    EXPECT(submissions[2].buffer == target_buffer && submissions[2].memory == memory);
    EXPECT(submissions[1].memory_offset == tile_size_in_bytes && submissions[1].resource_offset == 0);
    EXPECT(submissions[2].memory_offset == tile_size_in_bytes && submissions[2].resource_offset == 0);
    EXPECT(submissions[1].waits_for(progress, 1) && submissions[2].waits_for(progress, 2));
    EXPECT(submissions[3].fence == done);
    EXPECT(submissions[3].waits_for(progress, 3));

    submissions.clear();
    queue.wait(gate, 2);
    const std::array unmap{TileRange{TileRangeMode::unmap, 0, 1}};
    EXPECT(queue.update_tiles(source, first, unmap, get_heap));
    EXPECT(queue.copy_tile_mappings(target, first, source, first));
    EXPECT(submissions.empty());
    context.holds().signaled(gate, 2);
    context.holds().pump_ready();
    EXPECT(submissions.size() == 2);
    EXPECT(submissions[0].buffer == source_buffer && submissions[0].memory == VK_NULL_HANDLE);
    EXPECT(submissions[0].waits_for(gate, 2));
    EXPECT(submissions[1].buffer == target_buffer && submissions[1].memory == VK_NULL_HANDLE);

    submissions.clear();
    queue.wait(gate, 3);
    EXPECT(queue.copy_tile_mappings(target, first, source, first));
    EXPECT(queue.signal(done, 2));
    EXPECT(submissions.empty());
    context.holds().signaled(gate, 3);
    context.holds().pump_ready();
    EXPECT(submissions.size() == 2);
    EXPECT(submissions[0].buffer == VK_NULL_HANDLE && submissions[0].waits_for(gate, 3));
    EXPECT(submissions[1].fence == done);

    submissions.clear();
    heap = std::make_shared<TileHeapAllocation>(TileHeapAllocation{memory, 0, 2 * tile_size_in_bytes, 0});
    queue.wait(gate, 4);
    EXPECT(queue.update_tiles(source, first, ranges, get_heap));
    context.remove_resource(source);
    add_sparse_buffer(context, source, 3);
    context.holds().signaled(gate, 4);
    context.holds().pump_ready();
    EXPECT(submissions.size() == 1);
    EXPECT(submissions[0].buffer == VK_NULL_HANDLE && submissions[0].waits_for(gate, 4));

    std::puts("vk_sparse_queue_order: OK");
    return 0;
}
