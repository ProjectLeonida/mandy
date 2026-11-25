// RT-344: AMD's PC FidelityFX upscaler through src/ffx_vk_backend on a Vulkan device, on procedural inputs (no game
// data). Windows program (on Linux it runs under Wine, whose Vulkan reaches the native driver).
//
//   LEONIDA_FFX_DIR=<FidelityFX SDK>\Kits\FidelityFX\signedbin  ffx_vk_dispatch_test
//
// Two cases, each on its own context: 1280x720 -> 1920x1080, the one configuration the console's DLL builds
// (docs/d3d12_vulkan.md A.3), and 1920x1080 -> 3840x2160 (4K, Performance). An RGBA16F checkerboard of 32-pixel squares
// (1.0 / 0.0), depth 0 (far, inverted) and zero motion go through 30 upscales with a Halton(2,3) jitter, using the
// game's measured context flags (HDR, inverted depth, auto exposure). Then every output value must be finite and the
// centre of every output square near its input value. Prints the provider the loader chose for the GPU and driver
// (FSR 4.1 or 3.1, docs/d3d12_vulkan.md A.3) and the time per upscale. Exits 77 without a Vulkan device or the loader DLL.
//
// LEONIDA_VK_VALIDATION=1 runs it under VK_LAYER_KHRONOS_validation, and any validation error fails it, except the
// ones known_provider_error names: those are in AMD's FSR 4.1.2 shaders (this test creates no shader module and
// dispatches nothing itself), measured on an RX 9070 XT under RADV, and are printed and counted but not failed.
#include "../src/ffx_vk_backend.h"
#include "vk_device_pick.h"
#include "vulkan_loader.h"

#include <windows.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <vector>

namespace {

constexpr uint32_t SQUARE = 32, FRAMES = 30, WARMUP = 5;
struct Case {
    uint32_t rw, rh, ow, oh;   // render and upscale sizes; integer ratios of SQUARE pixels
};
constexpr Case cases[] = {{1280, 720, 1920, 1080}, {1920, 1080, 3840, 2160}};

void on_message(uint32_t type, const wchar_t* message) { std::printf("  ffx message (%u): %ls\n", type, message); }

float half_to_float(uint16_t h) {
    const uint32_t sign = h >> 15, exp = (h >> 10) & 31, mant = h & 1023;
    const float v = exp == 0    ? std::ldexp(float(mant), -24)
                    : exp == 31 ? (mant ? NAN : INFINITY)
                                : std::ldexp(float(mant | 1024), int(exp) - 25);
    return sign ? -v : v;
}

float halton(int i, int base) {
    float f = 1, r = 0;
    for (; i > 0; i /= base) {
        f /= float(base);
        r += f * float(i % base);
    }
    return r;
}

// AMD's FSR 4.1.2 Vulkan shaders on RDNA4 (RADV, Mesa 26.3.1; #4999). They declare SPV_NV_cooperative_matrix2, which
// the device does not offer; spirv-val rejects an OpFConvert on cooperative-matrix types and an OpStore to an image;
// and three storage images (rw_reprojected_color, rw_history_color, rw_mlsr_output_color) get a 3-component
// OpImageWrite on an RGBA16F view. Matched by VUID and text, so the same VUID from anything else still fails.
bool known_provider_error(const VkDebugUtilsMessengerCallbackDataEXT* data) {
    const char* id = data->pMessageIdName ? data->pMessageIdName : "";
    const char* text = data->pMessage ? data->pMessage : "";
    auto is = [&](const char* vuid, const char* needle) {
        return std::strcmp(id, vuid) == 0 && std::strstr(text, needle) != nullptr;
    };
    return is("VUID-VkShaderModuleCreateInfo-pCode-08742", "SPV_NV_cooperative_matrix2") ||
           is("VUID-VkShaderModuleCreateInfo-pCode-08740", "CooperativeMatrixConversionsNV") ||
           is("VUID-VkShaderModuleCreateInfo-pCode-08737", "Expected Use of Matrix type and Result Type to be identical") ||
           is("VUID-StandaloneSpirv-OpTypeImage-06924", "Cannot store to OpTypeImage") ||
           is("VUID-vkCmdDispatch-OpImageWrite-08795", "OpImageWrite Texel operand only contains 3 components");
}

unsigned validation_errors = 0, provider_errors = 0;

VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT types,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (!(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ||
        !(types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        return VK_FALSE;
    const bool provider = known_provider_error(data);
    ++(provider ? provider_errors : validation_errors);
    std::printf("validation%s: %s\n", provider ? " (AMD's FSR 4 shaders, not counted)" : "",
                data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

#define CHECK(x)                                                                \
    do {                                                                        \
        VkResult r_ = (x);                                                      \
        if (r_ != VK_SUCCESS) {                                                 \
            std::printf("FAIL %s -> %d (line %d)\n", #x, int(r_), __LINE__);    \
            return 1;                                                           \
        }                                                                       \
    } while (0)

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string error;
    if (leonida::ffx_vk::Upscaler probe; !probe.load(error)) {
        std::printf("SKIP %s\n", error.c_str());
        return 77;
    }
    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) {
        std::printf("SKIP no Vulkan loader\n");
        return 77;
    }
    VkInstance inst = VK_NULL_HANDLE;
    auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    const char* validation_env = std::getenv("LEONIDA_VK_VALIDATION");
    const bool validation = validation_env && std::strcmp(validation_env, "0") != 0;
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* debug_utils = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT messenger{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messenger.pfnUserCallback = on_validation;
    if (validation) {
        auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
            gipa(nullptr, "vkEnumerateInstanceLayerProperties"));
        uint32_t n = 0;
        enumerate(&n, nullptr);
        std::vector<VkLayerProperties> layers(n);
        enumerate(&n, layers.data());
        bool found = false;
        for (const auto& l : layers) found = found || std::strcmp(l.layerName, layer) == 0;
        if (!found) {
            std::printf("FAIL LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer);
            return 1;
        }
        ici.pNext = &messenger;   // messages from instance creation itself
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &debug_utils;
    }
    if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) {
        std::printf("SKIP no Vulkan instance\n");
        return 77;
    }
    if (validation) {
        auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            gipa(inst, "vkCreateDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerEXT handle;
        if (!create_messenger || create_messenger(inst, &messenger, nullptr, &handle) != VK_SUCCESS) {
            std::printf("FAIL vkCreateDebugUtilsMessengerEXT\n");
            return 1;
        }
    }
#define LOADI(name) auto name = reinterpret_cast<PFN_##name>(gipa(inst, #name))
    LOADI(vkEnumeratePhysicalDevices); LOADI(vkGetPhysicalDeviceProperties); LOADI(vkGetPhysicalDeviceFeatures2);
    LOADI(vkGetPhysicalDeviceQueueFamilyProperties); LOADI(vkGetPhysicalDeviceMemoryProperties);
    LOADI(vkCreateDevice); LOADI(vkGetDeviceProcAddr); LOADI(vkEnumerateDeviceExtensionProperties);
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(inst, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(inst, &count, devices.data());
    std::vector<VkPhysicalDeviceProperties> candidates(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) vkGetPhysicalDeviceProperties(devices[i], &candidates[i]);
    const std::size_t chosen = leonida::vk_test::pick_device(candidates);   // honours LEONIDA_VK_DEVICE
    if (chosen >= devices.size()) {
        leonida::vk_test::report_no_device();
        return 77;
    }
    const VkPhysicalDevice pd = devices[chosen];
    const VkPhysicalDeviceProperties& props = candidates[chosen];
    std::printf("device: %s\n", props.deviceName);

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &families, nullptr);
    std::vector<VkQueueFamilyProperties> qf(families);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &families, qf.data());
    uint32_t family = families;
    for (uint32_t i = 0; i < families && family == families; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) family = i;
    if (family == families) {
        std::printf("SKIP no graphics+compute queue\n");
        return 77;
    }
    // Every core feature the device has, and the backend's wanted extensions it has with all their features.
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    if (props.apiVersion < VK_API_VERSION_1_3) f12.pNext = nullptr;
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrices{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT float8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    void* const feature_of[] = {&derivatives, &matrices, &float8};   // in wanted_device_extensions order
    uint32_t available_count = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &available_count, nullptr);
    std::vector<VkExtensionProperties> available(available_count);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &available_count, available.data());
    std::vector<const char*> extensions;
    for (std::size_t i = 0; i < std::size(leonida::ffx_vk::wanted_device_extensions); ++i) {
        const char* name = leonida::ffx_vk::wanted_device_extensions[i];
        for (const auto& e : available) {
            if (std::strcmp(e.extensionName, name) != 0) continue;
            extensions.push_back(name);
            auto* feature = static_cast<VkBaseOutStructure*>(feature_of[i]);   // prepend to the query chain
            feature->pNext = static_cast<VkBaseOutStructure*>(f2.pNext);
            f2.pNext = feature;
        }
    }
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    std::printf("device extensions:");
    for (const char* name : extensions) std::printf(" %s", name);
    std::printf("%s\n", extensions.empty() ? " none of the wanted ones" : "");
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(extensions.size());
    dci.ppEnabledExtensionNames = extensions.data();
    VkDevice dev = VK_NULL_HANDLE;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define LOADD(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev, #name))
    LOADD(vkGetDeviceQueue); LOADD(vkCreateImage); LOADD(vkGetImageMemoryRequirements); LOADD(vkAllocateMemory);
    LOADD(vkBindImageMemory); LOADD(vkCreateBuffer); LOADD(vkGetBufferMemoryRequirements); LOADD(vkBindBufferMemory);
    LOADD(vkMapMemory); LOADD(vkCreateCommandPool); LOADD(vkAllocateCommandBuffers); LOADD(vkBeginCommandBuffer);
    LOADD(vkEndCommandBuffer); LOADD(vkCmdPipelineBarrier); LOADD(vkCmdCopyBufferToImage);
    LOADD(vkCmdCopyImageToBuffer); LOADD(vkCmdClearColorImage); LOADD(vkQueueSubmit); LOADD(vkQueueWaitIdle);
    LOADD(vkResetCommandBuffer);
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto memory_type = [&](uint32_t bits, VkMemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
        return mp.memoryTypeCount;
    };
    auto allocate = [&](const VkMemoryRequirements& mr, VkMemoryPropertyFlags flags, VkDeviceMemory& memory) {
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, flags);
        return ai.memoryTypeIndex < mp.memoryTypeCount && vkAllocateMemory(dev, &ai, nullptr, &memory) == VK_SUCCESS;
    };
    auto make_image = [&](uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImage& image) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {w, h, 1};
        ci.mipLevels = ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = usage;
        VkMemoryRequirements mr;
        VkDeviceMemory memory;
        if (vkCreateImage(dev, &ci, nullptr, &image) != VK_SUCCESS) return false;
        vkGetImageMemoryRequirements(dev, image, &mr);
        return allocate(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory) &&
               vkBindImageMemory(dev, image, memory, 0) == VK_SUCCESS;
    };
    auto make_buffer = [&](VkDeviceSize size, VkBuffer& buffer, void** mapped) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkMemoryRequirements mr;
        VkDeviceMemory memory;
        if (vkCreateBuffer(dev, &bi, nullptr, &buffer) != VK_SUCCESS) return false;
        vkGetBufferMemoryRequirements(dev, buffer, &mr);
        return allocate(mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory) &&
               vkBindBufferMemory(dev, buffer, memory, 0) == VK_SUCCESS &&
               vkMapMemory(dev, memory, 0, size, 0, mapped) == VK_SUCCESS;
    };

#define CHECK_CASE(x)                                                           \
    do {                                                                        \
        VkResult r_ = (x);                                                      \
        if (r_ != VK_SUCCESS) {                                                 \
            std::printf("FAIL %s -> %d (line %d)\n", #x, int(r_), __LINE__);    \
            return false;                                                       \
        }                                                                       \
    } while (0)
    // One case on a fresh context and fresh images; true when its output is right.
    auto run_case = [&](const Case& c) -> bool {
        const uint32_t RW = c.rw, RH = c.rh, OW = c.ow, OH = c.oh;
        std::printf("case %ux%u -> %ux%u\n", RW, RH, OW, OH);
        const VkImageUsageFlags input = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        VkImage color, depth, motion, output;
        VkBuffer upload, readback;
        void* upload_map;
        void* readback_map;
        if (!make_image(RW, RH, VK_FORMAT_R16G16B16A16_SFLOAT, input, color) ||
            !make_image(RW, RH, VK_FORMAT_R32_SFLOAT, input, depth) ||
            !make_image(RW, RH, VK_FORMAT_R16G16_SFLOAT, input, motion) ||
            !make_image(OW, OH, VK_FORMAT_R16G16B16A16_SFLOAT,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, output) ||
            !make_buffer(VkDeviceSize(RW) * RH * 8, upload, &upload_map) ||
            !make_buffer(VkDeviceSize(OW) * OH * 8, readback, &readback_map)) {
            std::printf("FAIL image or buffer creation\n");
            return false;
        }
        auto* texel = static_cast<uint16_t*>(upload_map);
        for (uint32_t y = 0; y < RH; ++y)
            for (uint32_t x = 0; x < RW; ++x) {
                uint16_t* p = texel + (std::size_t(y) * RW + x) * 4;
                p[0] = p[1] = p[2] = ((x / SQUARE + y / SQUARE) & 1) ? 0x3C00 : 0;   // 1.0 or 0.0
                p[3] = 0x3C00;
            }

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = family;
        VkCommandPool pool;
        CHECK_CASE(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cb;
        CHECK_CASE(vkAllocateCommandBuffers(dev, &cai, &cb));
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        auto barrier = [&](VkImage image, VkImageLayout from, VkImageLayout to) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            b.oldLayout = from;
            b.newLayout = to;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = image;
            b.subresourceRange = range;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &b);
        };
        const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        auto submit = [&]() {
            vkEndCommandBuffer(cb);
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            VkResult r = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
            if (r == VK_SUCCESS) r = vkQueueWaitIdle(queue);
            vkResetCommandBuffer(cb, 0);
            return r;
        };

        // Inputs in SHADER_READ_ONLY_OPTIMAL (the COMPUTE_READ state), the output in GENERAL (UNORDERED_ACCESS).
        vkBeginCommandBuffer(cb, &begin);
        for (VkImage image : {color, depth, motion})
            barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy up{};
        up.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        up.imageExtent = {RW, RH, 1};
        vkCmdCopyBufferToImage(cb, upload, color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &up);
        const VkClearColorValue zero{};
        vkCmdClearColorImage(cb, depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        vkCmdClearColorImage(cb, motion, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        for (VkImage image : {color, depth, motion})
            barrier(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        barrier(output, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        CHECK_CASE(submit());

        ffx::CreateContextDescUpscale create{{ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE, nullptr},
                                             ffx::UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | ffx::UPSCALE_ENABLE_DEPTH_INVERTED |
                                                 ffx::UPSCALE_ENABLE_AUTO_EXPOSURE | ffx::UPSCALE_ENABLE_DEBUG_CHECKING,
                                             {RW, RH}, {OW, OH}, on_message};
        leonida::ffx_vk::Upscaler upscaler;
        if (!upscaler.load(error)) {
            std::printf("FAIL %s\n", error.c_str());
            return false;
        }
        const ffx::ReturnCode made = upscaler.create({dev, pd, vkGetDeviceProcAddr}, create);
        if (made != ffx::RETURN_OK) {
            std::printf("FAIL ffxCreateContext on the Vulkan backend: %s\n", ffx::return_code_name(made));
            return false;
        }
        std::printf("provider: %s\n", upscaler.provider().c_str());

        auto resource = [](VkImage image, uint32_t w, uint32_t h, uint32_t format, uint32_t usage, uint32_t state) {
            ffx::Resource r{};
            r.resource = reinterpret_cast<void*>(image);
            r.description = {ffx::RESOURCE_TYPE_TEXTURE2D, format, w, h, 1, 1, 0, usage};
            r.state = state;
            return r;
        };
        ffx::DispatchDescUpscale d{};
        d.header.type = ffx::DISPATCH_DESC_TYPE_UPSCALE;
        d.color = resource(color, RW, RH, ffx::SURFACE_FORMAT_R16G16B16A16_FLOAT, ffx::RESOURCE_USAGE_READ_ONLY,
                           ffx::RESOURCE_STATE_COMPUTE_READ);
        d.depth = resource(depth, RW, RH, ffx::SURFACE_FORMAT_R32_FLOAT, ffx::RESOURCE_USAGE_READ_ONLY,
                           ffx::RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = resource(motion, RW, RH, ffx::SURFACE_FORMAT_R16G16_FLOAT, ffx::RESOURCE_USAGE_READ_ONLY,
                                   ffx::RESOURCE_STATE_COMPUTE_READ);
        d.output = resource(output, OW, OH, ffx::SURFACE_FORMAT_R16G16B16A16_FLOAT, ffx::RESOURCE_USAGE_UAV,
                            ffx::RESOURCE_STATE_UNORDERED_ACCESS);
        d.motionVectorScale = {float(RW), float(RH)};
        d.renderSize = {RW, RH};
        d.upscaleSize = {OW, OH};
        d.frameTimeDelta = 16.6f;
        d.preExposure = 1.0f;
        d.cameraNear = 1e30f;   // inverted depth: near and far swap
        d.cameraFar = 0.1f;
        d.cameraFovAngleVertical = 1.0f;
        d.viewSpaceToMetersFactor = 1.0f;
        LARGE_INTEGER frequency, t0, t1;
        QueryPerformanceFrequency(&frequency);
        double timed_ms = 0;
        for (uint32_t frame = 0; frame < FRAMES; ++frame) {
            d.jitterOffset = {halton(int(frame % 8) + 1, 2) - 0.5f, halton(int(frame % 8) + 1, 3) - 0.5f};
            d.reset = frame == 0;
            vkBeginCommandBuffer(cb, &begin);
            d.commandList = cb;
            const ffx::ReturnCode rc = upscaler.dispatch(d);
            if (rc != ffx::RETURN_OK) {
                std::printf("FAIL ffxDispatch, frame %u: %s\n", frame, ffx::return_code_name(rc));
                return false;
            }
            QueryPerformanceCounter(&t0);
            CHECK_CASE(submit());
            QueryPerformanceCounter(&t1);
            if (frame >= WARMUP) timed_ms += double(t1.QuadPart - t0.QuadPart) * 1000.0 / double(frequency.QuadPart);
        }
        std::printf("%u upscales %ux%u -> %ux%u, submit and wait %.2f ms each after %u warm-up frames\n", FRAMES, RW, RH,
                    OW, OH, timed_ms / (FRAMES - WARMUP), WARMUP);

        vkBeginCommandBuffer(cb, &begin);
        barrier(output, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy down{};
        down.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        down.imageExtent = {OW, OH, 1};
        vkCmdCopyImageToBuffer(cb, output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &down);
        CHECK_CASE(submit());

        const auto* out = static_cast<const uint16_t*>(readback_map);
        std::size_t non_finite = 0;
        for (std::size_t i = 0; i < std::size_t(OW) * OH * 4; ++i) non_finite += !std::isfinite(half_to_float(out[i]));
        const uint32_t square = SQUARE * OW / RW;
        uint32_t right = 0, squares = 0;
        for (uint32_t sy = 0; (sy + 1) * square <= OH; ++sy)
            for (uint32_t sx = 0; (sx + 1) * square <= OW; ++sx, ++squares) {
                const float want = ((sx + sy) & 1) ? 1.0f : 0.0f;
                const std::size_t centre = (std::size_t(sy) * square + square / 2) * OW + sx * square + square / 2;
                bool close = true;
                for (int c = 0; c < 3; ++c) close = close && std::fabs(half_to_float(out[centre * 4 + c]) - want) < 0.1f;
                right += close;
            }
        std::printf("non-finite output values: %zu; square centres within 0.1 of the input: %u of %u\n", non_finite,
                    right, squares);
        return non_finite == 0 && right == squares;
    };
    bool ok = true;
    for (const Case& c : cases) ok = run_case(c) && ok;
    if (validation)
        std::printf("validation errors: %u (and %u in AMD's FSR 4 shaders, not counted)\n", validation_errors,
                    provider_errors);
    ok = ok && validation_errors == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
