// What a page-sized sparse bind costs on this device: vkQueueBindSparse of N 64 KiB pages (contiguous in the buffer,
// scattered over one memory pool, as a remap of title pages would bind them), the time the call takes on the CPU and
// the time until its timeline semaphore is signalled, for N = 1 .. 4096; the same for unbinds; and the same page of
// memory bound at two buffer offsets (an aliased physical page). A measurement for the page mirror's layout, not a
// pass/fail test: it prints a table and exits 0 (77 when the device has no sparse buffers). Pick the device with
// MESA_VK_DEVICE_SELECT. No shaders, no game data.
#define VK_NO_PROTOTYPES
#include "../src/arena_backing.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

constexpr VkDeviceSize kPage = 64 * 1024;
constexpr VkDeviceSize kBuffer = VkDeviceSize(3) << 30;  // a mirror chunk: 3 GiB
constexpr VkDeviceSize kPool = 256ull << 20;             // the frames: 4,096 pages

#define FUNCTIONS(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) \
    X(vkDestroyInstance)
#define DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkQueueBindSparse) X(vkCreateSemaphore) X(vkDestroySemaphore) \
    X(vkWaitSemaphores) X(vkDeviceWaitIdle) X(vkGetBufferDeviceAddress)
#define DECLARE(name) PFN_##name name = nullptr;
FUNCTIONS(DECLARE)
DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE

using Clock = std::chrono::steady_clock;
double us(Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }

}  // namespace

int main() {
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
    if (!get_instance) return std::puts("SKIP: no Vulkan loader"), 77;
    VkInstance instance = VK_NULL_HANDLE;
    auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(get_instance(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    if (!create_instance || create_instance(&instance_info, nullptr, &instance) != VK_SUCCESS)
        return std::puts("SKIP: no Vulkan instance"), 77;
#define LOAD(name) name = reinterpret_cast<PFN_##name>(get_instance(instance, #name));
    FUNCTIONS(LOAD)
#undef LOAD
    std::uint32_t count = 1;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    vkEnumeratePhysicalDevices(instance, &count, &physical);
    if (!physical) return std::puts("SKIP: no Vulkan device"), 77;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    std::printf("device: %s (driver 0x%x); sparseAddressSpaceSize %llu GiB, maxBufferSize via 3 GiB chunk, "
                "aliased %d, strict %d\n",
                properties.deviceName, properties.driverVersion,
                static_cast<unsigned long long>(properties.limits.sparseAddressSpaceSize >> 30),
                int(features.features.sparseResidencyAliased), int(properties.sparseProperties.residencyNonResidentStrict));
    if (!features.features.sparseBinding || !features.features.sparseResidencyBuffer || !v12.timelineSemaphore)
        return std::puts("SKIP: no sparse buffers"), 77;
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    std::vector<std::uint32_t> used(family_count, 0);
    const auto bind_queue = leonida::pick_arena_bind_queue(families.data(), family_count, used.data(), 16);
    if (bind_queue.family == UINT32_MAX) return std::puts("SKIP: no sparse queue"), 77;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue.queueFamilyIndex = bind_queue.family;
    queue.queueCount = bind_queue.index + 1;
    std::vector<float> priorities(queue.queueCount, priority);
    queue.pQueuePriorities = priorities.data();
    VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.timelineSemaphore = VK_TRUE;
    enable12.bufferDeviceAddress = v12.bufferDeviceAddress;
    enable12.bufferDeviceAddressCaptureReplay = v12.bufferDeviceAddressCaptureReplay;
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &enable12};
    enable.features.sparseBinding = VK_TRUE;
    enable.features.sparseResidencyBuffer = VK_TRUE;
    enable.features.sparseResidencyAliased = features.features.sparseResidencyAliased;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue;
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS) return std::puts("no device"), 1;
    auto get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_instance(instance, "vkGetDeviceProcAddr"));
#define LOAD(name) name = reinterpret_cast<PFN_##name>(get_device(device, #name));
    DEVICE_FUNCTIONS(LOAD)
#undef LOAD
    VkQueue binds = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, bind_queue.family, bind_queue.index, &binds);

    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                        (features.features.sparseResidencyAliased ? VK_BUFFER_CREATE_SPARSE_ALIASED_BIT : 0);
    buffer_info.size = kBuffer;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) return std::puts("no 3 GiB sparse buffer"), 1;
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    std::uint32_t type = UINT32_MAX;
    for (std::uint32_t t = 0; t < memory.memoryTypeCount && type == UINT32_MAX; ++t)
        if ((requirements.memoryTypeBits >> t & 1) &&
            (memory.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            type = t;
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = kPool;
    allocate.memoryTypeIndex = type;
    VkDeviceMemory pool = VK_NULL_HANDLE;
    if (type == UINT32_MAX || vkAllocateMemory(device, &allocate, nullptr, &pool) != VK_SUCCESS)
        return std::puts("no pool memory"), 1;
    std::printf("page alignment %llu; bind queue family %u index %u\n",
                static_cast<unsigned long long>(requirements.alignment), bind_queue.family, bind_queue.index);

    VkSemaphoreTypeCreateInfo timeline_type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline_type};
    VkSemaphore timeline = VK_NULL_HANDLE;
    vkCreateSemaphore(device, &semaphore_info, nullptr, &timeline);
    std::uint64_t value = 0;

    std::mt19937 random(7);
    const std::uint64_t pool_pages = kPool / kPage, buffer_pages = kBuffer / kPage;
    // One call of `n` binds; returns {cpu us, until-signalled us}.
    std::vector<VkSparseMemoryBind> list;
    // memory: 0 scattered pool pages, 1 one page for all, 2 the pool's pages in order (a run of one allocation)
    const auto bind = [&](std::size_t n, bool unbind, std::uint64_t first, bool scattered_buffer, int same_memory) {
        list.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint64_t at = scattered_buffer ? random() % buffer_pages : (first + i) % buffer_pages;
            list[i] = {at * kPage, kPage, unbind ? VK_NULL_HANDLE : pool,
                       unbind ? 0
                       : same_memory == 1 ? 0
                       : same_memory == 2 ? ((first + i) % pool_pages) * kPage
                                          : (random() % pool_pages) * kPage,
                       0};
        }
        const VkSparseBufferMemoryBindInfo buffer_bind{buffer, std::uint32_t(n), list.data()};
        VkTimelineSemaphoreSubmitInfo signal_value{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        const std::uint64_t next = ++value;
        signal_value.signalSemaphoreValueCount = 1;
        signal_value.pSignalSemaphoreValues = &next;
        VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &signal_value};
        info.bufferBindCount = 1;
        info.pBufferBinds = &buffer_bind;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &timeline;
        const auto t0 = Clock::now();
        const VkResult r = vkQueueBindSparse(binds, 1, &info, VK_NULL_HANDLE);
        const auto t1 = Clock::now();
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &timeline;
        wait.pValues = &next;
        vkWaitSemaphores(device, &wait, UINT64_MAX);
        const auto t2 = Clock::now();
        if (r != VK_SUCCESS) std::printf("  bind failed: %d\n", int(r));
        return std::pair<double, double>{us(t1 - t0), us(t2 - t0)};
    };
    const auto row = [&](const char* what, std::size_t n, bool unbind, bool scattered, int same, int rounds) {
        std::vector<double> cpu, done;
        for (int k = 0; k < rounds; ++k) {
            const auto [c, d] = bind(n, unbind, std::uint64_t(k) * n, scattered, same);
            cpu.push_back(c);
            done.push_back(d);
        }
        std::sort(cpu.begin(), cpu.end());
        std::sort(done.begin(), done.end());
        const double c = cpu[cpu.size() / 2], d = done[done.size() / 2];
        std::printf("%-34s %5zu pages: call %9.1f us (%7.2f us/page), signalled after %9.1f us (%7.2f us/page)\n",
                    what, n, c, c / double(n), d, d / double(n));
    };
    row("warm-up", 64, false, false, 0, 4);
    for (const std::size_t n : {std::size_t(1), std::size_t(16), std::size_t(64), std::size_t(256), std::size_t(1024),
                                std::size_t(4096)}) {
        row("bind, contiguous VAs", n, false, false, 0, n >= 1024 ? 5 : 21);
        row("bind, scattered VAs", n, false, true, 0, n >= 1024 ? 5 : 21);
        row("bind, VAs and memory both in order", n, false, false, 2, n >= 1024 ? 5 : 21);
        row("unbind, contiguous VAs", n, true, false, 0, n >= 1024 ? 5 : 21);
    }
    if (features.features.sparseResidencyAliased) row("aliased: one page at many VAs", 256, false, false, 1, 21);
    // Placement: can a sparse buffer be made at a device address of our choosing (bufferDeviceAddressCaptureReplay,
    // VkBufferOpaqueCaptureAddressCreateInfo)? If so, a mirror chunk could sit at heap_base + the title's address,
    // and a V# or pointer the title wrote into its own memory would reach it unchanged.
    std::printf("bufferDeviceAddressCaptureReplay %d\n", int(v12.bufferDeviceAddressCaptureReplay));
    if (v12.bufferDeviceAddress && v12.bufferDeviceAddressCaptureReplay && vkGetBufferDeviceAddress) {
        const auto place = [&](std::uint64_t want, VkDeviceSize bytes) {
            VkBufferOpaqueCaptureAddressCreateInfo at{VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO};
            at.opaqueCaptureAddress = want;
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &at};
            info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                         VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
            info.size = bytes;
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            VkBuffer made = VK_NULL_HANDLE;
            const VkResult r = vkCreateBuffer(device, &info, nullptr, &made);
            std::uint64_t got = 0;
            if (r == VK_SUCCESS) {
                VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                address.buffer = made;
                got = vkGetBufferDeviceAddress(device, &address);
            }
            std::printf("  sparse %llu MiB at 0x%016llx: %s, address 0x%016llx\n",
                        static_cast<unsigned long long>(bytes >> 20), static_cast<unsigned long long>(want),
                        r == VK_SUCCESS ? (got == want ? "placed" : "made elsewhere") : "refused",
                        static_cast<unsigned long long>(got));
            return made;
        };
        // A heap base low in the upper half, and chunks at heap_base + title addresses seen under Wine.
        const std::uint64_t heap = 0xffff800100000000ull;
        std::vector<VkBuffer> placed;
        for (const std::uint64_t title : {0ull, 0x0000000600000000ull, 0x00000075e0000000ull, 0x000000d340000000ull,
                                          0x00007fb3a0000000ull, 0x00007fb7a0000000ull, 0x00007ffe00000000ull})
            placed.push_back(place(heap + title, VkDeviceSize(3) << 30));
        for (VkBuffer b : placed)
            if (b) vkDestroyBuffer(device, b, nullptr);
    }
    // A remap's rate: 2,000 pages a second in batches of 32 (the logs' queue slot 31: tens of pages a call).
    {
        const auto t0 = Clock::now();
        for (int k = 0; k < 63; ++k) bind(32, false, std::uint64_t(k) * 32, true, 0);
        std::printf("remap load: 63 calls of 32 scattered pages in %.1f ms\n", us(Clock::now() - t0) / 1000.0);
    }
    bind(std::size_t(buffer_pages > 65536 ? 65536 : buffer_pages), true, 0, false, 0);
    vkDeviceWaitIdle(device);
    vkDestroySemaphore(device, timeline, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
