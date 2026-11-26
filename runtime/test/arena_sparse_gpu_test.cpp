// The GPU address arena's backing blocks on a real device (src/arena_backing.h): a sparse arena of 256 MiB with
// 40 MiB allocated has about 40 MiB bound; what the CPU writes through the blocks' mappings is what the GPU reads at
// those arena offsets; a block whose allocations are all released stays bound while a submission that read it is
// in flight, and is unbound (reading zero) only once that submission has completed. A bind made while the work queue
// is blocked completes: the arena binds on a queue of its own. No shaders, no game data.
// Exit code 77 (skipped) on a device without sparse buffers, without residencyNonResidentStrict, or without a queue
// to spare for the binds: the runtime keeps one allocation behind its arena there. LEONIDA_VK_VALIDATION=1 runs it
// under VK_LAYER_KHRONOS_validation.
#define VK_NO_PROTOTYPES
#include "../src/arena_backing.h"
#include "../src/arena_blocks.h"

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
constexpr VkDeviceSize kArena = 256 * kMiB;
constexpr VkDeviceSize kHead = 64 * 1024;

#define FUNCTIONS(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) \
    X(vkDestroyInstance)
#define DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkBindBufferMemory) X(vkQueueBindSparse) \
    X(vkQueueWaitIdle) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdCopyBuffer) X(vkQueueSubmit) X(vkCreateSemaphore) \
    X(vkDestroySemaphore) X(vkSignalSemaphore) X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) \
    X(vkDeviceWaitIdle) X(vkResetCommandBuffer)
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
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    std::fprintf(stderr, "device: %s (driver 0x%x)\n", properties.deviceName, properties.driverVersion);
    if (!features.features.sparseBinding || !features.features.sparseResidencyBuffer ||
        !properties.sparseProperties.residencyNonResidentStrict || !v12.timelineSemaphore ||
        !v12.bufferDeviceAddress || properties.limits.sparseAddressSpaceSize < kArena) {
        std::puts("SKIP: no sparse buffers with strict residency on this device (the arena stays one allocation)");
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
        std::puts("SKIP: no queue to spare for the arena's binds (the runtime keeps one allocation)");
        return 77;
    }
    std::fprintf(stderr, "work queue family %u, bind queue family %u index %u\n", work, bind_queue.family,
                 bind_queue.index);
    const float priorities[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo queues[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    queues[0].queueFamilyIndex = work;
    queues[0].queueCount = bind_queue.family == work ? 2 : 1;
    queues[0].pQueuePriorities = priorities;
    queues[1].queueFamilyIndex = bind_queue.family;
    queues[1].queueCount = 1;
    queues[1].pQueuePriorities = priorities;
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.timelineSemaphore = VK_TRUE;
    enable12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &enable12};
    enable.features.sparseBinding = VK_TRUE;
    enable.features.sparseResidencyBuffer = VK_TRUE;
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

    // The arena's buffer as the runtime makes it, sparse.
    VkBufferCreateInfo arena_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    arena_info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
    arena_info.size = kArena;
    arena_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer arena = VK_NULL_HANDLE;
    EXPECT(vkCreateBuffer(device, &arena_info, nullptr, &arena) == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, arena, &requirements);
    EXPECT(leonida::kArenaBlockBytes % requirements.alignment == 0);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    const auto pick_type = [&](std::uint32_t bits) {
        const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (const VkMemoryPropertyFlags want : {host | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, host})
            for (std::uint32_t t = 0; t < memory.memoryTypeCount; ++t)
                if ((bits >> t & 1) && (memory.memoryTypes[t].propertyFlags & want) == want) return t;
        return UINT32_MAX;
    };
    const std::uint32_t type = pick_type(requirements.memoryTypeBits);
    EXPECT(type != UINT32_MAX);

    std::uint64_t memory_objects = 0;
    leonida::ArenaBackingHooks hooks;
    hooks.make = [&](VkDeviceSize bytes, VkDeviceMemory& out, std::uint8_t*& host) {
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocate.allocationSize = bytes;
        allocate.memoryTypeIndex = type;
        void* mapped = nullptr;
        if (vkAllocateMemory(device, &allocate, nullptr, &out) != VK_SUCCESS) return false;
        if (vkMapMemory(device, out, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
            vkFreeMemory(device, out, nullptr);
            return false;
        }
        host = static_cast<std::uint8_t*>(mapped);
        ++memory_objects;
        return true;
    };
    hooks.destroy = [&](VkDeviceMemory m) {
        vkFreeMemory(device, m, nullptr);
        --memory_objects;
    };
    hooks.bind = [&](const VkSparseMemoryBind* b, std::uint32_t n) {
        const VkSparseBufferMemoryBindInfo buffer{arena, n, b};
        VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        info.bufferBindCount = 1;
        info.pBufferBinds = &buffer;
        return vkQueueBindSparse(binds, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(binds) == VK_SUCCESS;
    };
    leonida::ArenaBacking backing;
    EXPECT(backing.init_sparse(kArena, leonida::kArenaBlockBytes, std::move(hooks)));
    leonida::ArenaBlocks blocks;
    blocks.reset(kHead, kArena);
    EXPECT(backing.acquire(0, kHead, true));
    const auto allocate = [&](VkDeviceSize asked, VkDeviceSize& out) {
        const VkDeviceSize bytes = backing.footprint(asked);
        if (!blocks.allocate(bytes, backing.alignment(asked, 256),
                             [&](std::uint64_t lo, std::uint64_t hi) { return backing.keep_out(lo, hi); }, out))
            return false;
        if (backing.acquire(out, bytes)) return true;
        blocks.release(out, bytes);
        return false;
    };

    // 40 MiB allocated: 100 of 256 KiB and 3 of 5 MiB.
    struct Live { VkDeviceSize offset, bytes; };
    std::vector<Live> live;
    for (int i = 0; i < 100; ++i) {
        VkDeviceSize at = 0;
        EXPECT(allocate(256 * 1024, at));
        live.push_back({at, 256 * 1024});
    }
    for (int i = 0; i < 3; ++i) {
        VkDeviceSize at = 0;
        EXPECT(allocate(5 * kMiB, at));
        live.push_back({at, 5 * kMiB});
    }
    const VkDeviceSize allocated = 40 * kMiB;
    const VkDeviceSize bound = backing.stats().bound;
    std::fprintf(stderr, "arena %llu MiB, %llu MiB allocated, %llu MiB bound in %llu memory objects\n",
                 (unsigned long long)(kArena >> 20), (unsigned long long)(allocated >> 20),
                 (unsigned long long)(bound >> 20), (unsigned long long)memory_objects);
    // About what is allocated: the small ones pack into whole blocks, the large ones round up to whole blocks.
    EXPECT(bound >= allocated && bound <= allocated + 6 * kMiB);

    // The CPU writes through the blocks' mappings; the GPU reads the same bytes at the arena offsets.
    const Live small = live[37], large = live[101];
    for (const Live& l : {small, large}) {
        auto* words = reinterpret_cast<std::uint32_t*>(backing.host(l.offset, l.bytes));
        EXPECT(words != nullptr);
        for (VkDeviceSize i = 0; i < l.bytes / 4; ++i) words[i] = std::uint32_t(l.offset / 4 + i) ^ 0x5a5a5a5au;
    }
    VkBufferCreateInfo readback_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    readback_info.size = 2 * 4096;
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
    VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline};
    VkSemaphore gate = VK_NULL_HANDLE, done = VK_NULL_HANDLE;  // the host opens gate; done is the retire ticket
    EXPECT(vkCreateSemaphore(device, &semaphore_info, nullptr, &gate) == VK_SUCCESS);
    EXPECT(vkCreateSemaphore(device, &semaphore_info, nullptr, &done) == VK_SUCCESS);
    const auto record_copies = [&]() {
        vkResetCommandBuffer(commands, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commands, &begin);
        const VkBufferCopy copies[2] = {{small.offset, 0, 4096}, {large.offset + 3 * kMiB, 4096, 4096}};
        vkCmdCopyBuffer(commands, arena, readback, 2, copies);
        return vkEndCommandBuffer(commands) == VK_SUCCESS;
    };
    const auto submit = [&](std::uint64_t wait, std::uint64_t signal) {
        VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        values.waitSemaphoreValueCount = wait ? 1 : 0;
        values.pWaitSemaphoreValues = &wait;
        values.signalSemaphoreValueCount = 1;
        values.pSignalSemaphoreValues = &signal;
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO, &values};
        info.waitSemaphoreCount = wait ? 1 : 0;
        info.pWaitSemaphores = &gate;
        info.pWaitDstStageMask = &stage;
        info.commandBufferCount = 1;
        info.pCommandBuffers = &commands;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &done;
        return vkQueueSubmit(work_queue, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS;
    };
    const auto completed = [&] {
        std::uint64_t value = 0;
        vkGetSemaphoreCounterValue(device, done, &value);
        return value;
    };

    // Submission 1 reads both allocations, but waits for the host first: it is in flight.
    std::memset(got, 0, 2 * 4096);
    EXPECT(record_copies());
    EXPECT(submit(1, 1));
    // Every allocation is released while it may still read them: their blocks are pending with ticket 1, and a
    // reclaim with what has completed (0) unbinds none of them.
    for (const Live& l : live) {
        const VkDeviceSize bytes = backing.footprint(l.bytes);
        EXPECT(blocks.release(l.offset, bytes) == leonida::ArenaBlocks::Fault::none);
        backing.release(l.offset, bytes, 1);
    }
    EXPECT(completed() == 0);
    EXPECT(backing.reclaim(completed()) == 0);
    EXPECT(backing.stats().bound == bound);
    // A new allocation binds a block now, on the arena's queue, while the work queue is blocked.
    VkDeviceSize fresh = 0;
    EXPECT(allocate(3 * kMiB, fresh));
    EXPECT(backing.stats().bound == bound + 4 * kMiB);

    // Open the gate: the submission reads what the CPU wrote, from memory that was still bound.
    VkSemaphoreSignalInfo open{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    open.semaphore = gate;
    open.value = 1;
    EXPECT(vkSignalSemaphore(device, &open) == VK_SUCCESS);
    VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    const std::uint64_t one = 1;
    wait.semaphoreCount = 1;
    wait.pSemaphores = &done;
    wait.pValues = &one;
    EXPECT(vkWaitSemaphores(device, &wait, 5'000'000'000ull) == VK_SUCCESS);
    EXPECT(got[0] == (std::uint32_t(small.offset / 4) ^ 0x5a5a5a5au));
    EXPECT(got[1023] == (std::uint32_t(small.offset / 4 + 1023) ^ 0x5a5a5a5au));
    EXPECT(got[1024] == (std::uint32_t((large.offset + 3 * kMiB) / 4) ^ 0x5a5a5a5au));

    // Ticket 1 has completed: the released blocks are unbound and freed; the head and the fresh one stay.
    const VkDeviceSize unbound = backing.reclaim(completed());
    std::fprintf(stderr, "after the submission retired: %llu MiB unbound, %llu MiB bound, %llu memory objects\n",
                 (unsigned long long)(unbound >> 20), (unsigned long long)(backing.stats().bound >> 20),
                 (unsigned long long)memory_objects);
    EXPECT(backing.stats().bound == leonida::kArenaBlockBytes + 4 * kMiB);
    EXPECT(!backing.host(large.offset, 4) && memory_objects == 2);
    // The GPU reads zero there now (residencyNonResidentStrict), and no fault.
    std::memset(got, 0xff, 2 * 4096);
    EXPECT(record_copies());
    EXPECT(submit(0, 2));
    const std::uint64_t two = 2;
    wait.pValues = &two;
    EXPECT(vkWaitSemaphores(device, &wait, 5'000'000'000ull) == VK_SUCCESS);
    EXPECT(got[1024] == 0 && got[2047] == 0);

    vkDeviceWaitIdle(device);
    backing.clear();
    vkDestroySemaphore(device, gate, nullptr);
    vkDestroySemaphore(device, done, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, readback, nullptr);
    vkFreeMemory(device, readback_memory, nullptr);
    vkDestroyBuffer(device, arena, nullptr);
    return 0;
}

}  // namespace

int main() {
    const int result = test();
    if (!result) std::fprintf(stderr, "arena_sparse_gpu: ok\n");
    return result;
}
