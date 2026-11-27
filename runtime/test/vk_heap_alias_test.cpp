// A console descriptor heap read in place through the GPU address arena (docs/d3d12_vulkan.md section 3.4): the
// title's heap pages have one GPU home in the page mirror (src/title_mirror.h: sparse buffers at heap_base + the
// title's address), but 25,865 of the corpus's 26,503 shaders keep only the low 32 bits of a descriptor-table pointer,
// so the heap must also be readable at heap_base + an offset below 4 GiB, inside the sparse arena. This checks the
// mechanism on a real device, with no shader and no game data: one VkDeviceMemory bound into two sparse aliased
// buffers (a "mirror chunk" at the title's offset and the "arena" at the heap's offset) shows the same bytes through
// both - what the CPU writes into the home, and what the GPU writes through either buffer - and the arena's pages after
// the heap, which nothing is bound to, read zero (the null-slot guard without memory). Unbinding the mirror's view
// leaves the arena's view intact.
// Exit code 77 (skipped) on a device without sparse aliased buffers with strict residency. LEONIDA_VK_VALIDATION=1 runs
// it under VK_LAYER_KHRONOS_validation.
#define VK_NO_PROTOTYPES
#include "../src/arena_backing.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr VkDeviceSize kMiB = 1 << 20;
constexpr VkDeviceSize kBuffer = 64 * kMiB;
constexpr VkDeviceSize kTitle = 40 * kMiB;  // where the heap's pages lie in the mirror chunk
constexpr VkDeviceSize kHeap = 8 * kMiB;    // where shaders read the heap in the arena
constexpr std::uint32_t kPages = 4;

#define FUNCTIONS(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) \
    X(vkDestroyInstance)
#define DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkBindBufferMemory) X(vkQueueBindSparse) \
    X(vkQueueWaitIdle) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCmdPipelineBarrier) \
    X(vkQueueSubmit) X(vkDeviceWaitIdle) X(vkResetCommandBuffer)
#define DECLARE(name) PFN_##name name = nullptr;
FUNCTIONS(DECLARE)
DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE

struct Run {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    ~Run() {
        if (device) {
            vkDeviceWaitIdle(device);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

int test() {
#ifdef _WIN32
    HMODULE library = LoadLibraryA("vulkan-1.dll");
    auto get_instance = library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                                      reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")))
                                : nullptr;
#else
    void* library = dlopen("libvulkan.so.1", RTLD_NOW);
    auto get_instance =
        library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr")) : nullptr;
#endif
    if (!get_instance) {
        std::puts("SKIP: no Vulkan loader");
        return 77;
    }
    Run run;
    auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(get_instance(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* validation = std::getenv("LEONIDA_VK_VALIDATION");
    if (validation && *validation == '1') {
        instance_info.enabledLayerCount = 1;
        instance_info.ppEnabledLayerNames = &layer;
    }
    if (!create_instance || create_instance(&instance_info, nullptr, &run.instance) != VK_SUCCESS) {
        std::puts("SKIP: no Vulkan instance");
        return 77;
    }
#define LOAD(name) name = reinterpret_cast<PFN_##name>(get_instance(run.instance, #name));
    FUNCTIONS(LOAD)
#undef LOAD
    std::uint32_t count = 1;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    vkEnumeratePhysicalDevices(run.instance, &count, &physical);  // the first: as the loader orders them
    if (!physical) {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    std::fprintf(stderr, "device: %s (driver 0x%x)\n", properties.deviceName, properties.driverVersion);
    if (!features.features.sparseBinding || !features.features.sparseResidencyBuffer ||
        !features.features.sparseResidencyAliased || !properties.sparseProperties.residencyNonResidentStrict ||
        properties.limits.sparseAddressSpaceSize < 2 * kBuffer) {
        std::puts("SKIP: no sparse aliased buffers with strict residency on this device");
        return 77;
    }
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    std::uint32_t work = UINT32_MAX;
    for (std::uint32_t f = 0; f < family_count && work == UINT32_MAX; ++f)
        if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) work = f;
    if (work == UINT32_MAX) {
        std::puts("SKIP: no compute queue");
        return 77;
    }
    std::vector<std::uint32_t> used(family_count, 0);
    used[work] = 1;
    const auto bind_queue = leonida::pick_arena_bind_queue(families.data(), family_count, used.data(), 16);
    if (bind_queue.family == UINT32_MAX) {
        std::puts("SKIP: no queue to spare for the binds");
        return 77;
    }
    const float priorities[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo queues[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    queues[0].queueFamilyIndex = work;
    queues[0].queueCount = bind_queue.family == work ? 2 : 1;
    queues[0].pQueuePriorities = priorities;
    queues[1].queueFamilyIndex = bind_queue.family;
    queues[1].queueCount = 1;
    queues[1].pQueuePriorities = priorities;
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    enable.features.sparseBinding = VK_TRUE;
    enable.features.sparseResidencyBuffer = VK_TRUE;
    enable.features.sparseResidencyAliased = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable};
    device_info.queueCreateInfoCount = bind_queue.family == work ? 1 : 2;
    device_info.pQueueCreateInfos = queues;
    EXPECT(vkCreateDevice(physical, &device_info, nullptr, &run.device) == VK_SUCCESS);
    auto get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_instance(run.instance, "vkGetDeviceProcAddr"));
#define LOAD(name) name = reinterpret_cast<PFN_##name>(get_device(run.device, #name));
    DEVICE_FUNCTIONS(LOAD)
#undef LOAD
    const VkDevice device = run.device;
    VkQueue work_queue = VK_NULL_HANDLE, binds = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, work, 0, &work_queue);
    vkGetDeviceQueue(device, bind_queue.family, bind_queue.index, &binds);

    // The arena and a mirror chunk, both sparse and aliased, as the device makes them.
    VkBufferCreateInfo sparse_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    sparse_info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                        VK_BUFFER_CREATE_SPARSE_ALIASED_BIT;
    sparse_info.size = kBuffer;
    sparse_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer arena = VK_NULL_HANDLE, mirror = VK_NULL_HANDLE;
    EXPECT(vkCreateBuffer(device, &sparse_info, nullptr, &arena) == VK_SUCCESS);
    EXPECT(vkCreateBuffer(device, &sparse_info, nullptr, &mirror) == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, arena, &requirements);
    const VkDeviceSize page = requirements.alignment;
    EXPECT(page && kTitle % page == 0 && kHeap % page == 0);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    const auto pick_type = [&](std::uint32_t bits) {
        const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (const VkMemoryPropertyFlags want : {host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, host})
            for (std::uint32_t t = 0; t < memory.memoryTypeCount; ++t)
                if ((bits >> t & 1) && (memory.memoryTypes[t].propertyFlags & want) == want) return t;
        return UINT32_MAX;
    };
    // The heap pages' one home.
    VkMemoryAllocateInfo home_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    home_info.allocationSize = page * kPages;
    home_info.memoryTypeIndex = pick_type(requirements.memoryTypeBits);
    EXPECT(home_info.memoryTypeIndex != UINT32_MAX);
    VkDeviceMemory home = VK_NULL_HANDLE;
    EXPECT(vkAllocateMemory(device, &home_info, nullptr, &home) == VK_SUCCESS);
    void* home_host = nullptr;
    EXPECT(vkMapMemory(device, home, 0, VK_WHOLE_SIZE, 0, &home_host) == VK_SUCCESS);
    auto* words = static_cast<std::uint32_t*>(home_host);

    const auto bind = [&](VkBuffer buffer, VkDeviceSize at, VkDeviceMemory memory_or_null) {
        VkSparseMemoryBind b{};
        b.resourceOffset = at;
        b.size = page * kPages;
        b.memory = memory_or_null;
        const VkSparseBufferMemoryBindInfo buffer_bind{buffer, 1, &b};
        VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        info.bufferBindCount = 1;
        info.pBufferBinds = &buffer_bind;
        return vkQueueBindSparse(binds, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(binds) == VK_SUCCESS;
    };
    EXPECT(bind(mirror, kTitle, home));
    EXPECT(bind(arena, kHeap, home));

    // Readback.
    VkBufferCreateInfo readback_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    readback_info.size = 3 * page;
    readback_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer readback = VK_NULL_HANDLE;
    EXPECT(vkCreateBuffer(device, &readback_info, nullptr, &readback) == VK_SUCCESS);
    VkMemoryRequirements readback_requirements{};
    vkGetBufferMemoryRequirements(device, readback, &readback_requirements);
    VkMemoryAllocateInfo readback_allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    readback_allocate.allocationSize = readback_requirements.size;
    readback_allocate.memoryTypeIndex = pick_type(readback_requirements.memoryTypeBits);
    VkDeviceMemory readback_memory = VK_NULL_HANDLE;
    EXPECT(vkAllocateMemory(device, &readback_allocate, nullptr, &readback_memory) == VK_SUCCESS);
    EXPECT(vkBindBufferMemory(device, readback, readback_memory, 0) == VK_SUCCESS);
    void* readback_host = nullptr;
    EXPECT(vkMapMemory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &readback_host) == VK_SUCCESS);
    auto* got = static_cast<std::uint32_t*>(readback_host);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = work;
    VkCommandPool pool = VK_NULL_HANDLE;
    EXPECT(vkCreateCommandPool(device, &pool_info, nullptr, &pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    EXPECT(vkAllocateCommandBuffers(device, &command_info, &commands) == VK_SUCCESS);
    // Records `body`, submits and waits.
    const auto run_commands = [&](auto&& body) {
        vkResetCommandBuffer(commands, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commands, &begin);
        body();
        if (vkEndCommandBuffer(commands) != VK_SUCCESS) return false;
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount = 1;
        info.pCommandBuffers = &commands;
        return vkQueueSubmit(work_queue, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS &&
               vkQueueWaitIdle(work_queue) == VK_SUCCESS;
    };
    const auto transfer_barrier = [&] {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                             nullptr, 0, nullptr);
    };

    // The CPU writes descriptor-like words into the home: the arena's view and the mirror's show them; the arena page
    // after the heap reads zero.
    const std::size_t per_page = std::size_t(page / 4);
    for (std::size_t i = 0; i < per_page * kPages; ++i) words[i] = std::uint32_t(i * 2654435761u) | 1u;
    std::memset(got, 0xff, std::size_t(3 * page));
    EXPECT(run_commands([&] {
        const VkBufferCopy from_arena[2] = {{kHeap + page, 0, page}, {kHeap + kPages * page, page, page}};
        vkCmdCopyBuffer(commands, arena, readback, 2, from_arena);
        const VkBufferCopy from_mirror{kTitle + page, 2 * page, page};
        vkCmdCopyBuffer(commands, mirror, readback, 1, &from_mirror);
    }));
    for (std::size_t i = 0; i < per_page; ++i) {
        EXPECT(got[i] == words[per_page + i]);
        EXPECT(got[per_page + i] == 0);  // the guard: nothing bound
        EXPECT(got[2 * per_page + i] == words[per_page + i]);
    }

    // The GPU writes through the mirror's view: the arena's view and the CPU's mapping show it; and the other way.
    EXPECT(run_commands([&] {
        vkCmdFillBuffer(commands, mirror, kTitle + 2 * page + 256, 256, 0xabcdef01u);
        vkCmdFillBuffer(commands, arena, kHeap + 3 * page, 256, 0x01fedcbau);
        transfer_barrier();
        const VkBufferCopy copies[2] = {{kHeap + 2 * page + 256, 0, 256}, {kHeap + 3 * page, 256, 256}};
        vkCmdCopyBuffer(commands, arena, readback, 1, &copies[0]);
        const VkBufferCopy from_mirror{kTitle + 3 * page, 256, 256};
        vkCmdCopyBuffer(commands, mirror, readback, 1, &from_mirror);
    }));
    for (std::size_t i = 0; i < 64; ++i) {
        EXPECT(got[i] == 0xabcdef01u && words[2 * per_page + 64 + i] == 0xabcdef01u);
        EXPECT(got[64 + i] == 0x01fedcbau && words[3 * per_page + i] == 0x01fedcbau);
    }

    // The page leaves the mirror (the title unmapped that view): the arena's view still shows the home, the mirror's
    // reads zero.
    EXPECT(bind(mirror, kTitle, VK_NULL_HANDLE));
    std::memset(got, 0xff, std::size_t(3 * page));
    EXPECT(run_commands([&] {
        const VkBufferCopy from_arena{kHeap, 0, page};
        vkCmdCopyBuffer(commands, arena, readback, 1, &from_arena);
        const VkBufferCopy from_mirror{kTitle, page, page};
        vkCmdCopyBuffer(commands, mirror, readback, 1, &from_mirror);
    }));
    for (std::size_t i = 0; i < per_page; ++i) {
        EXPECT(got[i] == words[i]);
        EXPECT(got[per_page + i] == 0);
    }

    EXPECT(bind(arena, kHeap, VK_NULL_HANDLE));
    vkDeviceWaitIdle(device);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, readback, nullptr);
    vkFreeMemory(device, readback_memory, nullptr);
    vkDestroyBuffer(device, arena, nullptr);
    vkDestroyBuffer(device, mirror, nullptr);
    vkFreeMemory(device, home, nullptr);
    std::printf("vk_heap_alias_test: ok (one home in a mirror chunk and in the arena, %llu KiB pages)\n",
                static_cast<unsigned long long>(page >> 10));
    return 0;
}

}  // namespace

int main() { return test(); }
