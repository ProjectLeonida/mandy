// The page mirror on a real device (src/title_mirror.h): chunks made at heap_base + a title address (a sparse "heap"
// buffer stands in for the GPU address arena, whose address is heap_base), pages bound to one home per physical page,
// and a compute shader that reads a V# out of a buffer and loads through it as the translated shaders do
// (glsl/cs_title_mirror_read.comp). Checked:
//   1. a V# whose base is a mirrored title address reads that page's bytes; one at another title address of the same
//      physical page (an alias) reads the same; one at a page with nothing bound reads zero, with no fault;
//   2. a remap (a page given another physical page, as CopyPageMappingsBatchX does) is a rebind: the next dispatch,
//      which waits for the batch's value (the ordering rule), reads the new page; a write to a physical page, once
//      dirty() names a page that shows it, reaches every alias with one upload; an unmap reads zero again;
//   3. a home no page binds is freed only once the ticket passed to map() has completed.
// Exit 77 without sparse aliased strict residency, bufferDeviceAddressCaptureReplay or glslangValidator (llvmpipe: the
// runtime keeps the old paths there, the page mirror is off). No game data.
#define VK_NO_PROTOTYPES
#include "../src/arena_backing.h"
#include "../src/title_mirror.h"
#include "../src/title_mirror_space.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
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

#define FUNCTIONS(X) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) \
    X(vkDestroyInstance)
#define DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkBindBufferMemory) X(vkQueueBindSparse) \
    X(vkQueueSubmit) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkWaitSemaphores) X(vkDeviceWaitIdle) \
    X(vkGetBufferDeviceAddress) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdPushConstants) X(vkCmdDispatch) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkResetCommandBuffer)
#define DECLARE(name) PFN_##name name = nullptr;
FUNCTIONS(DECLARE)
DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE

constexpr std::uint64_t kPage = leonida::TitleMirror::kPage;

std::vector<std::uint32_t> load_spirv(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto bytes = static_cast<std::size_t>(file.tellg());
    std::vector<std::uint32_t> words(bytes / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
    return words;
}

struct Push {
    std::uint64_t heap_base, vsharps, out_words;
    std::uint32_t count, pad;
};

int test() {
#ifndef LEONIDA_TEST_TITLE_MIRROR_SPV
    std::puts("SKIP: no glslangValidator at build time");
    return 77;
#else
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
    // Declared before the cleanup, which uses them.
    std::vector<VkBuffer> chunks;
    std::vector<VkDeviceMemory> units;
    std::unique_ptr<leonida::TitleMirror> mirror;
    struct Cleanup {
        VkInstance instance;
        VkDevice device = VK_NULL_HANDLE;
        std::vector<std::function<void()>> undo;
        ~Cleanup() {
            if (device) vkDeviceWaitIdle(device);
            for (auto it = undo.rbegin(); it != undo.rend(); ++it) (*it)();
            if (device) vkDestroyDevice(device, nullptr);
            vkDestroyInstance(instance, nullptr);
        }
    } run{instance};
    std::uint32_t count = 1;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    vkEnumeratePhysicalDevices(instance, &count, &physical);
    if (!physical) return std::puts("SKIP: no Vulkan device"), 77;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v12};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    std::fprintf(stderr, "device: %s\n", properties.deviceName);
    if (!features.features.sparseBinding || !features.features.sparseResidencyBuffer ||
        !features.features.sparseResidencyAliased || !properties.sparseProperties.residencyNonResidentStrict ||
        !v12.timelineSemaphore || !v12.bufferDeviceAddress || !v12.bufferDeviceAddressCaptureReplay ||
        !features.features.shaderInt64)
        return std::puts("SKIP: no aliased strict sparse buffers with capture-replay addresses (page mirror off)"), 77;
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    std::uint32_t work = UINT32_MAX;
    for (std::uint32_t f = 0; f < family_count && work == UINT32_MAX; ++f)
        if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) work = f;
    std::vector<std::uint32_t> used(family_count, 0);
    if (work != UINT32_MAX) used[work] = 1;
    const auto bind_queue = leonida::pick_arena_bind_queue(families.data(), family_count, used.data(), 16);
    if (work == UINT32_MAX || bind_queue.family == UINT32_MAX) return std::puts("SKIP: no queue for the binds"), 77;
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
    enable12.bufferDeviceAddressCaptureReplay = VK_TRUE;
    VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &enable12};
    enable.features.sparseBinding = VK_TRUE;
    enable.features.sparseResidencyBuffer = VK_TRUE;
    enable.features.sparseResidencyAliased = VK_TRUE;
    enable.features.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enable};
    device_info.queueCreateInfoCount = bind_queue.family == work ? 1 : 2;
    device_info.pQueueCreateInfos = queues;
    EXPECT(vkCreateDevice(physical, &device_info, nullptr, &run.device) == VK_SUCCESS);
    auto get_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(get_instance(instance, "vkGetDeviceProcAddr"));
#define LOAD(name) name = reinterpret_cast<PFN_##name>(get_device(run.device, #name));
    DEVICE_FUNCTIONS(LOAD)
#undef LOAD
    const VkDevice device = run.device;
    VkQueue work_queue = VK_NULL_HANDLE, binds = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, work, 0, &work_queue);
    vkGetDeviceQueue(device, bind_queue.family, bind_queue.index, &binds);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    const auto address_of = [&](VkBuffer b) {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = b;
        return vkGetBufferDeviceAddress(device, &info);
    };
    const auto sparse_info = [&](VkDeviceSize bytes) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT |
                     VK_BUFFER_CREATE_SPARSE_ALIASED_BIT;
        info.size = bytes;
        info.usage = usage;
        return info;
    };
    // The "arena": a sparse buffer whose address is heap_base, as the runtime's sparse arena is.
    const VkBufferCreateInfo heap_info = sparse_info(64ull << 20);
    VkBuffer heap = VK_NULL_HANDLE;
    EXPECT(vkCreateBuffer(device, &heap_info, nullptr, &heap) == VK_SUCCESS);
    run.undo.push_back([device, heap] { vkDestroyBuffer(device, heap, nullptr); });
    const std::uint64_t heap_base = address_of(heap);
    VkMemoryRequirements sparse_requirements{};
    vkGetBufferMemoryRequirements(device, heap, &sparse_requirements);
    std::fprintf(stderr, "heap_base 0x%llx\n", static_cast<unsigned long long>(heap_base));

    VkSemaphoreTypeCreateInfo timeline_type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline_type};
    VkSemaphore timeline = VK_NULL_HANDLE, done = VK_NULL_HANDLE;
    EXPECT(vkCreateSemaphore(device, &semaphore_info, nullptr, &timeline) == VK_SUCCESS);
    EXPECT(vkCreateSemaphore(device, &semaphore_info, nullptr, &done) == VK_SUCCESS);
    run.undo.push_back([device, timeline, done] {
        vkDestroySemaphore(device, timeline, nullptr);
        vkDestroySemaphore(device, done, nullptr);
    });

    leonida::TitleMirrorHooks hooks;
    hooks.make_chunk = [&](std::uint64_t title_base, std::uint64_t bytes, VkBuffer& out) {
        VkBufferOpaqueCaptureAddressCreateInfo at{VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO};
        at.opaqueCaptureAddress = heap_base + title_base;
        VkBufferCreateInfo info = sparse_info(bytes);
        info.flags |= VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT;
        info.pNext = &at;
        if (vkCreateBuffer(device, &info, nullptr, &out) != VK_SUCCESS) return false;
        if (address_of(out) != heap_base + title_base) {
            vkDestroyBuffer(device, out, nullptr);
            return false;
        }
        chunks.push_back(out);
        return true;
    };
    hooks.make_unit = [&](VkDeviceMemory& out, std::uint8_t*& host) {
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t t = 0; t < memory.memoryTypeCount && type == UINT32_MAX; ++t)
            if ((sparse_requirements.memoryTypeBits >> t & 1) && (memory.memoryTypes[t].propertyFlags & want) == want)
                type = t;
        if (type == UINT32_MAX) return false;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = leonida::TitleMirror::kUnitBytes;
        allocate.memoryTypeIndex = type;
        void* mapped = nullptr;
        if (vkAllocateMemory(device, &allocate, nullptr, &out) != VK_SUCCESS) return false;
        if (vkMapMemory(device, out, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) return false;
        host = static_cast<std::uint8_t*>(mapped);
        units.push_back(out);
        return true;
    };
    hooks.free_unit = [&](VkDeviceMemory m) { vkFreeMemory(device, m, nullptr); };
    hooks.bind = [&](const std::vector<VkSparseBufferMemoryBindInfo>& list, std::uint64_t value) {
        VkTimelineSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        signal.signalSemaphoreValueCount = 1;
        signal.pSignalSemaphoreValues = &value;
        VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &signal};
        info.bufferBindCount = std::uint32_t(list.size());
        info.pBufferBinds = list.data();
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &timeline;
        return vkQueueBindSparse(binds, 1, &info, VK_NULL_HANDLE) == VK_SUCCESS;
    };
    mirror = std::make_unique<leonida::TitleMirror>(std::move(hooks));
    run.undo.push_back([&mirror, &chunks, device] {
        vkDeviceWaitIdle(device);
        mirror.reset();
        for (VkBuffer c : chunks) vkDestroyBuffer(device, c, nullptr);
    });

    // A reservation of the title's in the mirror's range, and two physical pages (host memory, as xmem's sections are).
    const std::uint64_t t0 = leonida::title_mirror_space::kLow;
    EXPECT(mirror->cover(t0, t0 + 64 * kPage) && mirror->covers(t0) && !mirror->covers(t0 + leonida::TitleMirror::kChunk));
    std::vector<std::uint32_t> a(kPage / 4), b(kPage / 4);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = 0xA0000000u + std::uint32_t(i);
        b[i] = 0xB0000000u + std::uint32_t(i);
    }
    auto* page_a = reinterpret_cast<std::uint8_t*>(a.data());
    auto* page_b = reinterpret_cast<std::uint8_t*>(b.data());
    std::uint64_t ticket = 1;
    mirror->map(t0, page_a, nullptr, ticket);              // page 0: A
    mirror->map(t0 + 3 * kPage, page_a, nullptr, ticket);  // page 3: A again (an alias)
    mirror->map(t0 + kPage, page_b, nullptr, ticket);      // page 1: B; page 2: nothing
    EXPECT(mirror->counters().homes == 2 && mirror->counters().uploads == 2);
    EXPECT(mirror->flush(1) && mirror->last_batch() == 3);

    // The V#s, in a buffer of their own (the title's memory a shader reads them from), and the output.
    const auto host_buffer = [&](VkDeviceSize bytes, VkBuffer& out, void*& mapped) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        if (vkCreateBuffer(device, &info, nullptr, &out) != VK_SUCCESS) return false;
        VkMemoryRequirements r{};
        vkGetBufferMemoryRequirements(device, out, &r);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t t = 0; t < memory.memoryTypeCount && type == UINT32_MAX; ++t)
            if ((r.memoryTypeBits >> t & 1) && (memory.memoryTypes[t].propertyFlags & want) == want) type = t;
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocate.allocationSize = r.size;
        allocate.memoryTypeIndex = type;
        VkDeviceMemory m = VK_NULL_HANDLE;
        if (type == UINT32_MAX || vkAllocateMemory(device, &allocate, nullptr, &m) != VK_SUCCESS) return false;
        run.undo.push_back([device, out, m] {
            vkDestroyBuffer(device, out, nullptr);
            vkFreeMemory(device, m, nullptr);
        });
        return vkBindBufferMemory(device, out, m, 0) == VK_SUCCESS &&
               vkMapMemory(device, m, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS;
    };
    constexpr std::uint32_t kViews = 4;
    VkBuffer vsharps = VK_NULL_HANDLE, out = VK_NULL_HANDLE;
    void* vsharp_words = nullptr;
    void* out_words = nullptr;
    EXPECT(host_buffer(kViews * 16, vsharps, vsharp_words) && host_buffer(kViews * 4, out, out_words));
    // Title-written V#s: base = the title address (48 bits), stride 0, records 0x1000, as the title builds them.
    const std::uint64_t bases[kViews] = {t0, t0 + 3 * kPage, t0 + kPage, t0 + 2 * kPage};
    auto* w = static_cast<std::uint32_t*>(vsharp_words);
    for (std::uint32_t i = 0; i < kViews; ++i) {
        w[i * 4 + 0] = std::uint32_t(bases[i]);
        w[i * 4 + 1] = std::uint32_t(bases[i] >> 32) & 0xffffu;
        w[i * 4 + 2] = 0x1000;
        w[i * 4 + 3] = 0;
    }

    const std::vector<std::uint32_t> code = load_spirv(LEONIDA_TEST_TITLE_MIRROR_SPV);
    EXPECT(!code.empty());
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size() * 4;
    module_info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    EXPECT(vkCreateShaderModule(device, &module_info, nullptr, &module) == VK_SUCCESS);
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    EXPECT(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout) == VK_SUCCESS);
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                           module, "main", nullptr};
    pipeline_info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    EXPECT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = work;
    VkCommandPool pool = VK_NULL_HANDLE;
    EXPECT(vkCreateCommandPool(device, &pool_info, nullptr, &pool) == VK_SUCCESS);
    run.undo.push_back([device, module, layout, pipeline, pool] {
        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, layout, nullptr);
        vkDestroyShaderModule(device, module, nullptr);
    });
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    EXPECT(vkAllocateCommandBuffers(device, &command_info, &commands) == VK_SUCCESS);

    std::uint64_t done_value = 0;
    // One dispatch that waits for the mirror's bind value `wait` (the ordering rule), then the four reads.
    const auto dispatch = [&](std::uint64_t wait, std::uint32_t* result) {
        vkResetCommandBuffer(commands, 0);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(commands, &begin);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        const Push push{heap_base, address_of(vsharps), address_of(out), kViews, 0};
        vkCmdPushConstants(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(commands, 1, 1, 1);
        vkEndCommandBuffer(commands);
        const std::uint64_t signal = ++done_value;
        VkTimelineSemaphoreSubmitInfo values{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        values.waitSemaphoreValueCount = 1;
        values.pWaitSemaphoreValues = &wait;
        values.signalSemaphoreValueCount = 1;
        values.pSignalSemaphoreValues = &signal;
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, &values};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &timeline;
        submit.pWaitDstStageMask = &stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commands;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &done;
        if (vkQueueSubmit(work_queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) return false;
        VkSemaphoreWaitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &done;
        wait_info.pValues = &signal;
        if (vkWaitSemaphores(device, &wait_info, 5'000'000'000ull) != VK_SUCCESS) return false;
        std::memcpy(result, out_words, kViews * 4);
        return true;
    };

    // 1. The page, its alias, another page, and a page with nothing bound.
    std::uint32_t r[kViews]{};
    EXPECT(dispatch(1, r));
    std::fprintf(stderr, "read 0x%08x 0x%08x 0x%08x 0x%08x\n", r[0], r[1], r[2], r[3]);
    EXPECT(r[0] == a[0x10] && r[1] == a[0x10] && r[2] == b[0x10] && r[3] == 0);

    // 2. A remap of page 0 to B, a write to A seen through its alias with one upload, and an unmap of page 1.
    mirror->map(t0, page_b, nullptr, ++ticket);
    a[0x10] = 0xA11A5E00u;
    const std::uint64_t uploads = mirror->counters().uploads;
    EXPECT(mirror->dirty(t0 + 3 * kPage) && mirror->upload() == 1 && mirror->counters().uploads == uploads + 1);
    EXPECT(mirror->flush(2));
    EXPECT(dispatch(2, r));
    EXPECT(r[0] == b[0x10] && r[1] == 0xA11A5E00u && r[2] == b[0x10] && r[3] == 0);
    mirror->map(t0 + kPage, nullptr, nullptr, ++ticket);
    EXPECT(mirror->flush(3));
    EXPECT(dispatch(3, r));
    EXPECT(r[0] == b[0x10] && r[1] == 0xA11A5E00u && r[2] == 0 && r[3] == 0);

    // 3. A unbound at page 3 too: its home stays unstamped (never freed even by retire(UINT64_MAX)) until flush()
    //    publishes the sparse unbind and stamp_freed() assigns the ticket; then it waits for that ticket to retire.
    const std::uint64_t freed = mirror->counters().homes_freed;
    mirror->map(t0 + 3 * kPage, nullptr, nullptr);
    EXPECT(mirror->counters().homes == 1 && mirror->unstamped_freed_count() == 1);
    mirror->retire(UINT64_MAX);
    EXPECT(mirror->counters().homes_freed == freed && mirror->unstamped_freed_count() == 1);
    EXPECT(mirror->flush(4));
    const std::uint64_t stamp = ++ticket;
    mirror->stamp_freed(stamp);
    EXPECT(mirror->unstamped_freed_count() == 0);
    mirror->retire(stamp - 1);
    EXPECT(mirror->counters().homes_freed == freed);
    mirror->retire(stamp);
    EXPECT(mirror->counters().homes_freed == freed + 1);
    EXPECT(dispatch(4, r) && r[0] == b[0x10] && r[1] == 0);

    // 4. Parallel batch upload (>= kParallelThreshold): 64 distinct pages mapped, mutated, uploaded via internal
    //    worker pool into mapped Vulkan unit memory, and verified on the GPU through title-written V#s.
    constexpr std::size_t kBatchPages = 64;
    std::vector<std::uint32_t> batch_frames(kBatchPages * (kPage / 4));
    for (std::size_t p = 0; p < kBatchPages; ++p) {
        std::uint32_t* page_words = batch_frames.data() + p * (kPage / 4);
        for (std::size_t i = 0; i < kPage / 4; ++i)
            page_words[i] = 0xC0000000u | (std::uint32_t(p) << 16) | std::uint32_t(i);
        mirror->map(t0 + (4 + p) * kPage, reinterpret_cast<std::uint8_t*>(page_words), nullptr, ++ticket);
    }
    EXPECT(mirror->flush(5));
    // Mutate word 0x10 of every page in the 64-page batch and upload in parallel.
    for (std::size_t p = 0; p < kBatchPages; ++p) {
        batch_frames[p * (kPage / 4) + 0x10] = 0xD0000000u | (std::uint32_t(p) << 16) | 0x10u;
        EXPECT(mirror->dirty(t0 + (4 + p) * kPage));
    }
    const std::uint64_t batch_uploads_before = mirror->counters().uploads;
    EXPECT(mirror->upload() == kBatchPages);
    EXPECT(mirror->counters().uploads == batch_uploads_before + kBatchPages);
    EXPECT(mirror->last_upload_bytes() == kBatchPages * 256);
    // Point the 4 V#s at 4 pages spanning the parallel batch (0, 17, 41, 63) and verify GPU reads.
    const std::size_t sample_pages[kViews] = {0, 17, 41, 63};
    for (std::uint32_t i = 0; i < kViews; ++i) {
        const std::uint64_t va = t0 + (4 + sample_pages[i]) * kPage;
        w[i * 4 + 0] = std::uint32_t(va);
        w[i * 4 + 1] = std::uint32_t(va >> 32) & 0xffffu;
    }
    EXPECT(dispatch(5, r));
    for (std::uint32_t i = 0; i < kViews; ++i) {
        const std::uint32_t expected = 0xD0000000u | (std::uint32_t(sample_pages[i]) << 16) | 0x10u;
        EXPECT(r[i] == expected);
    }

    std::printf("title mirror: a title-written V# read through a buffer reaches its page at heap_base + its address, "
                "its alias the same home, an unbound page zero; a remap, a write and an unmap reach the next dispatch "
                "that waits for the bind; 64-page parallel upload verified on GPU (%zu chunk, %zu unit)\n",
                chunks.size(), units.size());
    return 0;
#endif
}

}  // namespace

int main() { return test(); }
