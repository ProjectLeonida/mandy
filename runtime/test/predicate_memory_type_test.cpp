// Memory types for the predicate reducer's buffer (src/vulkan_predicate.h, PredicateReducer::batch) and the present
// composer's target (src/vulkan_present_compose.h), both chosen through memory_type_order and allocate_first_fitting
// (src/vulkan_memory.h). Following kevincommits' finding (#1608, Finding 19, item 4): a DEVICE_LOCAL type whose heap
// is smaller than the allocation is passed over (VUID-vkAllocateMemory-pAllocateInfo-01713, the 256 MiB BAR heap of
// a GPU without Resizable BAR), a type the driver refuses is followed by the next, a non-DEVICE_LOCAL type is the
// last resort, and a memoryTypeCount above VK_MAX_MEMORY_TYPES is never indexed past 32. No GPU: the device is fakes.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_predicate.h"
#include "../src/vulkan_memory.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

template<typename Handle>
Handle handle(std::uint64_t id) {
    return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(id));
}

struct Mock {
    std::uint32_t type_bits = 0x7u;
    std::vector<std::uint32_t> refuse;   // types whose vkAllocateMemory fails
    std::vector<std::uint32_t> attempts;  // every type asked, in order
    std::uint32_t allocated_type = UINT32_MAX;
};
Mock mock;

VKAPI_ATTR VkResult VKAPI_CALL ok_create(VkDevice, const void*, const VkAllocationCallbacks*, std::uint64_t* out) {
    *out = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                             VkBuffer* out) {
    *out = handle<VkBuffer>(info->size);  // the size, for the requirements below
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL requirements(VkDevice, VkBuffer buffer, VkMemoryRequirements* r) {
    r->size = static_cast<VkDeviceSize>(reinterpret_cast<std::uintptr_t>(buffer));
    r->alignment = 256;
    r->memoryTypeBits = mock.type_bits;
}
VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*,
                                        VkDeviceMemory* out) {
    mock.attempts.push_back(info->memoryTypeIndex);
    for (const auto t : mock.refuse)
        if (t == info->memoryTypeIndex) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    mock.allocated_type = info->memoryTypeIndex;
    *out = handle<VkDeviceMemory>(0x100 + info->memoryTypeIndex);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice, VkPipelineCache, std::uint32_t count,
                                                const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                VkPipeline* out) {
    for (std::uint32_t i = 0; i < count; ++i) out[i] = handle<VkPipeline>(2);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL allocate_sets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) out[i] = handle<VkDescriptorSet>(3);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL bind_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL nothing() {}

PFN_vkVoidFunction VKAPI_CALL get_proc(VkDevice, const char* name) {
    const std::string n = name;
    if (n == "vkCreateBuffer") return reinterpret_cast<PFN_vkVoidFunction>(&create_buffer);
    if (n == "vkGetBufferMemoryRequirements") return reinterpret_cast<PFN_vkVoidFunction>(&requirements);
    if (n == "vkAllocateMemory") return reinterpret_cast<PFN_vkVoidFunction>(&allocate);
    if (n == "vkCreateComputePipelines") return reinterpret_cast<PFN_vkVoidFunction>(&create_pipelines);
    if (n == "vkAllocateDescriptorSets") return reinterpret_cast<PFN_vkVoidFunction>(&allocate_sets);
    if (n == "vkBindBufferMemory") return reinterpret_cast<PFN_vkVoidFunction>(&bind_memory);
    // vkCreateDescriptorSetLayout, vkCreatePipelineLayout, vkCreateShaderModule, vkCreateDescriptorPool: the handle is
    // the fourth argument.
    if (n.rfind("vkCreate", 0) == 0) return reinterpret_cast<PFN_vkVoidFunction>(&ok_create);
    return reinterpret_cast<PFN_vkVoidFunction>(&nothing);  // destroy, free, update, the vkCmd* entries
}

// A GPU without Resizable BAR: type 0 DEVICE_LOCAL | HOST_VISIBLE in a small heap (here 1 KiB for the BAR's 256 MiB),
// type 1 DEVICE_LOCAL in the large VRAM heap, type 2 HOST_VISIBLE system memory.
VkPhysicalDeviceMemoryProperties memory_layout() {
    VkPhysicalDeviceMemoryProperties m{};
    m.memoryHeapCount = 3;
    m.memoryHeaps[0] = {1024, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    m.memoryHeaps[1] = {VkDeviceSize(8) << 30, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    m.memoryHeaps[2] = {VkDeviceSize(16) << 30, 0};
    m.memoryTypeCount = 3;
    m.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0};
    m.memoryTypes[1] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 1};
    m.memoryTypes[2] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 2};
    return m;
}

// A predicate batch of `count` slots (16 bytes each): the type its buffer memory came from, UINT32_MAX for none.
std::uint32_t batch_type(std::uint64_t count, const std::vector<std::uint32_t>& refuse,
                         const VkPhysicalDeviceMemoryProperties& m = memory_layout()) {
    mock.refuse = refuse;
    mock.attempts.clear();
    mock.allocated_type = UINT32_MAX;
    leonida::PredicateReducer reducer(&get_proc, handle<VkDevice>(1), m, VkDeviceSize(1) << 30);
    const auto batch = reducer.batch(count);
    return batch && batch->memory ? mock.allocated_type : UINT32_MAX;
}

}  // namespace

int main() {
    // 128 slots are 2 KiB: the 1 KiB BAR heap cannot hold them, so its type is never asked.
    check(batch_type(128, {}) == 1 && mock.attempts.front() == 1,
          "a buffer larger than the BAR heap goes to the VRAM type without asking the BAR type");
    // 16 slots are 256 bytes: the BAR type is first, and when the driver refuses it the VRAM type is next.
    check(batch_type(16, {0}) == 1 && mock.attempts == std::vector<std::uint32_t>({0, 1}),
          "a refused DEVICE_LOCAL type is followed by the next one");
    // Both DEVICE_LOCAL types refused: system memory is the last resort.
    check(batch_type(16, {0, 1}) == 2 && mock.attempts == std::vector<std::uint32_t>({0, 1, 2}),
          "with every DEVICE_LOCAL type refused, a host type is used");
    check(batch_type(16, {0, 1, 2}) == UINT32_MAX, "with every type refused, no batch");
    // Only types the buffer accepts are asked.
    mock.type_bits = 0x4u;
    check(batch_type(16, {}) == 2 && mock.attempts == std::vector<std::uint32_t>({2}),
          "a type outside memoryTypeBits is never asked");
    mock.type_bits = 0x7u;

    // A memoryTypeCount past VK_MAX_MEMORY_TYPES (a broken driver report) is read up to 32 types and no further.
    {
        auto m = memory_layout();
        m.memoryTypeCount = VK_MAX_MEMORY_TYPES + 8;
        for (std::uint32_t i = 3; i < VK_MAX_MEMORY_TYPES; ++i) m.memoryTypes[i] = {0, 2};
        const auto order = leonida::memory_type_order(m, 0xffffffffu, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 64);
        bool in_range = !order.empty();
        for (const auto t : order) in_range = in_range && t < VK_MAX_MEMORY_TYPES;
        check(in_range && order.front() == 0 && order.size() == VK_MAX_MEMORY_TYPES,
              "a memoryTypeCount above 32 is clamped to VK_MAX_MEMORY_TYPES");
    }

    // The composer's picker: DEVICE_LOCAL first, then any type, through allocate_first_fitting.
    {
        const auto m = memory_layout();
        const auto order = leonida::memory_type_order(m, 0x7u, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 4096);
        check(order == std::vector<std::uint32_t>({1, 2}), "a 4 KiB image: VRAM, then system memory (not the BAR)");
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.allocationSize = 4096;
        mock.refuse = {1};
        mock.attempts.clear();
        const VkDeviceMemory memory = leonida::allocate_first_fitting(&allocate, handle<VkDevice>(1), info, order);
        check(memory == handle<VkDeviceMemory>(0x102) && mock.attempts == std::vector<std::uint32_t>({1, 2}),
              "allocate_first_fitting moves on to system memory when VRAM is refused");
        mock.refuse = {1, 2};
        check(!leonida::allocate_first_fitting(&allocate, handle<VkDevice>(1), info, order),
              "allocate_first_fitting returns no memory when every type is refused");
        check(!leonida::allocate_first_fitting(nullptr, handle<VkDevice>(1), info, order),
              "allocate_first_fitting without vkAllocateMemory returns no memory");
    }

    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
