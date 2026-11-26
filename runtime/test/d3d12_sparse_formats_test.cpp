// Real reserved-image admission, reported tiling, and native WARP comparisons for standard color formats.
#include "d3d12_sparse_formats_fixture.h"
#include "host_wine.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace sparse_test {

bool sparse_image(ID3D12Resource* resource) {
    return rt::d3d12::is_sparse_image(resource);
}

D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT format, UINT width, UINT height) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

bool linear_buffer(ID3D12Device& device, UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                   ComPtr<ID3D12Resource>& resource) {
    const D3D12_HEAP_PROPERTIES heap{type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(&resource)));
}

void transition(ID3D12GraphicsCommandList& list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list.ResourceBarrier(1, &barrier);
}

bool submit_and_wait(ID3D12Device& device, ID3D12CommandQueue& queue, ID3D12GraphicsCommandList& list, bool layer) {
    ComPtr<ID3D12Fence> done;
    if (FAILED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))) || FAILED(list.Close())) return false;
    ID3D12CommandList* lists[] = {&list};
    queue.ExecuteCommandLists(1, lists);
    const bool replayed = !layer || rt::d3d12::last_execute_result(&queue) == leonida::ReplayResult::success;
    if (FAILED(queue.Signal(done.Get(), 1))) return false;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool completed = SUCCEEDED(done->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    CloseHandle(event);
    return completed && done->GetCompletedValue() == 1 && SUCCEEDED(device.GetDeviceRemovedReason()) && replayed;
}

}  // namespace sparse_test

namespace {

using sparse_test::FormatCase;

struct Format {
    DXGI_FORMAT dxgi;
    VkFormat backing;
    UINT bytes;
};

// DXGI enums remain distinct cases even when two flagless textures have the same Vulkan storage format.
constexpr Format formats[] = {
    {DXGI_FORMAT_R32G32B32A32_FLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, 16},
    {DXGI_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_UINT, 16},
    {DXGI_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_SINT, 16},
    {DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8},
    {DXGI_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UNORM, 8},
    {DXGI_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_UINT, 8},
    {DXGI_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_SNORM, 8},
    {DXGI_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_SINT, 8},
    {DXGI_FORMAT_R32G32_FLOAT, VK_FORMAT_R32G32_SFLOAT, 8},
    {DXGI_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_UINT, 8},
    {DXGI_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_SINT, 8},
    {DXGI_FORMAT_R10G10B10A2_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4},
    {DXGI_FORMAT_R10G10B10A2_UINT, VK_FORMAT_A2B10G10R10_UINT_PACK32, 4},
    {DXGI_FORMAT_R11G11B10_FLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4},
    {DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, 4},
    {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, VK_FORMAT_R8G8B8A8_SRGB, 4},
    {DXGI_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_UINT, 4},
    {DXGI_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_SNORM, 4},
    {DXGI_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SINT, 4},
    {DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, 4},
    {DXGI_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UNORM, 4},
    {DXGI_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_UINT, 4},
    {DXGI_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_SNORM, 4},
    {DXGI_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_SINT, 4},
    {DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, 4},
    {DXGI_FORMAT_R32_UINT, VK_FORMAT_R32_UINT, 4},
    {DXGI_FORMAT_R32_SINT, VK_FORMAT_R32_SINT, 4},
    {DXGI_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_UNORM, 2},
    {DXGI_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_UINT, 2},
    {DXGI_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_SNORM, 2},
    {DXGI_FORMAT_R8G8_SINT, VK_FORMAT_R8G8_SINT, 2},
    {DXGI_FORMAT_R16_FLOAT, VK_FORMAT_R16_SFLOAT, 2},
    {DXGI_FORMAT_R16_UNORM, VK_FORMAT_R16_UNORM, 2},
    {DXGI_FORMAT_R16_UINT, VK_FORMAT_R16_UINT, 2},
    {DXGI_FORMAT_R16_SNORM, VK_FORMAT_R16_SNORM, 2},
    {DXGI_FORMAT_R16_SINT, VK_FORMAT_R16_SINT, 2},
    {DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, 1},
    {DXGI_FORMAT_R8_UINT, VK_FORMAT_R8_UINT, 1},
    {DXGI_FORMAT_R8_SNORM, VK_FORMAT_R8_SNORM, 1},
    {DXGI_FORMAT_R8_SINT, VK_FORMAT_R8_SINT, 1},
    {DXGI_FORMAT_R9G9B9E5_SHAREDEXP, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 4},
    {DXGI_FORMAT_B5G6R5_UNORM, VK_FORMAT_R5G6B5_UNORM_PACK16, 2},
    {DXGI_FORMAT_B5G5R5A1_UNORM, VK_FORMAT_A1R5G5B5_UNORM_PACK16, 2},
    {DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, 4},
    {DXGI_FORMAT_B8G8R8X8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, 4},
    {DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4},
    {DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, VK_FORMAT_B8G8R8A8_SRGB, 4},
    {DXGI_FORMAT_B8G8R8X8_UNORM_SRGB, VK_FORMAT_B8G8R8A8_SRGB, 4},
    {DXGI_FORMAT_B4G4R4A4_UNORM, VK_FORMAT_A4R4G4B4_UNORM_PACK16, 2},
    {DXGI_FORMAT_A4B4G4R4_UNORM, VK_FORMAT_R4G4B4A4_UNORM_PACK16, 2},
};

FormatCase format_case(const Format& f, bool clipped) {
    // Standard 64 KiB D3D12 single-sample tile dimensions in texels.
    const UINT w = f.bytes <= 2 ? 256 : f.bytes <= 8 ? 128 : 64;
    const UINT h = f.bytes == 1 ? 256 : f.bytes <= 4 ? 128 : 64;
    return {f.dxgi, f.bytes, w, h, 2 * w - UINT(clipped), 2 * h - UINT(clipped)};
}

ComPtr<ID3D12Device> create_warp() {
    ComPtr<ID3D12Device> device;
    if (hostwine::host_is_wine()) return device;
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) return {};
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
    DXGI_ADAPTER_DESC1 desc{};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts))) ||
        opts.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2 || FAILED(adapter->GetDesc1(&desc))) return {};
    std::printf("Native D3D12 WARP: %ls\n", desc.Description);
    return device;
}

bool tiling_matches(ID3D12Device& device, ID3D12Resource& resource, const FormatCase& c) {
    UINT total = 0, count = 1;
    D3D12_PACKED_MIP_INFO mips{};
    D3D12_TILE_SHAPE tile{};
    D3D12_SUBRESOURCE_TILING sub{};
    device.GetResourceTiling(&resource, &total, &mips, &tile, &count, 0, &sub);
    return total == 4 && count == 1 && mips.NumStandardMips == 1 && !mips.NumPackedMips &&
           !mips.NumTilesForPackedMips && tile.WidthInTexels == c.tile_width && tile.HeightInTexels == c.tile_height &&
           tile.DepthInTexels == 1 && sub.WidthInTiles == 2 && sub.HeightInTiles == 2 && sub.DepthInTiles == 1 &&
           sub.StartTileIndexInOverallResource == 0;
}

template<typename Proc>
Proc instance_proc(const rt::d3d12::VulkanDeviceInfo& info, const char* name) {
    return reinterpret_cast<Proc>(info.get_instance_proc(info.instance, name));
}

// A committed fallback is only a skip if an independent native Vulkan query explains it.
struct SparseCaps {
    const rt::d3d12::VulkanDeviceInfo& info;
    leonida::VulkanDeviceFunctions vk{};
    PFN_vkGetPhysicalDeviceImageFormatProperties image_properties = nullptr;
    std::vector<std::uint32_t> families;
    std::uint32_t local_types = 0;
    bool supported = false;

    bool query_device() {
        const auto get_device = instance_proc<PFN_vkGetDeviceProcAddr>(info, "vkGetDeviceProcAddr");
        image_properties = instance_proc<PFN_vkGetPhysicalDeviceImageFormatProperties>(info, "vkGetPhysicalDeviceImageFormatProperties");
        const auto get_properties = instance_proc<PFN_vkGetPhysicalDeviceProperties>(info, "vkGetPhysicalDeviceProperties");
        const auto get_features = instance_proc<PFN_vkGetPhysicalDeviceFeatures>(info, "vkGetPhysicalDeviceFeatures");
        const auto get_memory = instance_proc<PFN_vkGetPhysicalDeviceMemoryProperties>(info, "vkGetPhysicalDeviceMemoryProperties");
        const auto get_queues = instance_proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(info, "vkGetPhysicalDeviceQueueFamilyProperties");
        if (!get_device || !image_properties || !get_properties || !get_features || !get_memory || !get_queues || !vk.load(get_device, info.device))
            return false;
        VkPhysicalDeviceProperties props{};
        get_properties(info.physical, &props);
        VkPhysicalDeviceMemoryProperties memory{};
        get_memory(info.physical, &memory);
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) local_types |= 1u << i;
        std::uint32_t count = 0;
        get_queues(info.physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        get_queues(info.physical, &count, queues.data());
        if (info.queue_family >= count) return false;
        VkPhysicalDeviceFeatures features{};
        get_features(info.physical, &features);
        supported = features.sparseBinding && features.sparseResidencyImage2D && features.sparseResidencyAliased &&
                    props.sparseProperties.residencyNonResidentStrict && (queues[info.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
        if (supported && !(info.features.sparseBinding && info.features.sparseResidencyImage2D && info.features.sparseResidencyAliased)) {
            std::fputs("Standalone device did not enable its supported sparse image features\n", stderr);
            return false;
        }
        for (const auto f : {info.queue_family, info.compute_family, info.copy_family})
            if (f != UINT32_MAX && std::find(families.begin(), families.end(), f) == families.end()) families.push_back(f);
        return true;
    }

    VkResult query_image(const FormatCase& c, VkFormat backing) const {
        VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        create.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
        create.imageType = VK_IMAGE_TYPE_2D;
        create.format = backing;
        create.extent = {c.width, c.height, 1};
        create.mipLevels = create.arrayLayers = 1;
        create.samples = VK_SAMPLE_COUNT_1_BIT;
        create.tiling = VK_IMAGE_TILING_OPTIMAL;
        create.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        leonida::share_across_queue_families(create, families);
        VkImageFormatProperties props{};
        VkResult result = image_properties(info.physical, backing, create.imageType, create.tiling, create.usage, create.flags, &props);
        if (result != VK_SUCCESS) return result;
        if (c.width > props.maxExtent.width || c.height > props.maxExtent.height) return VK_ERROR_FORMAT_NOT_SUPPORTED;
        VkImage image{};
        result = vk.vkCreateImage(info.device, &create, nullptr, &image);
        if (result != VK_SUCCESS) return result;
        VkMemoryRequirements memory{};
        vk.vkGetImageMemoryRequirements(info.device, image, &memory);
        std::uint32_t count = 0;
        vk.vkGetImageSparseMemoryRequirements(info.device, image, &count, nullptr);
        VkSparseImageMemoryRequirements req{};
        if (count == 1) vk.vkGetImageSparseMemoryRequirements(info.device, image, &count, &req);
        vk.vkDestroyImage(info.device, image, nullptr);
        const auto& p = req.formatProperties;
        const bool standard = count == 1 && p.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
            !(p.flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT) && p.imageGranularity.width == c.tile_width &&
            p.imageGranularity.height == c.tile_height && p.imageGranularity.depth == 1;
        return standard && !req.imageMipTailSize && req.imageMipTailFirstLod && memory.alignment == 65536 &&
               (memory.memoryTypeBits & local_types) ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
};

bool negative_cases(ID3D12Device& device, const SparseCaps& caps) {
    const auto base = sparse_test::texture_desc(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 256);
    auto check = [&](const char* name, const D3D12_RESOURCE_DESC& desc, const D3D12_CLEAR_VALUE* clear, HRESULT expected) {
        ComPtr<ID3D12Resource> resource;
        const HRESULT hr = device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, clear, IID_PPV_ARGS(&resource));
        if (hr == expected && (FAILED(hr) ? !resource : resource && !rt::d3d12::is_sparse_image(resource.Get()))) return true;
        std::fprintf(stderr, "%s: expected hr %#lx and no sparse image, got %#lx\n", name, static_cast<unsigned long>(expected), static_cast<unsigned long>(hr));
        return false;
    };
    for (const auto format : {DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_A8_UNORM}) {
        auto desc = base;
        desc.Format = format;
        if (!check("excluded format", desc, nullptr, format == DXGI_FORMAT_A8_UNORM ? E_INVALIDARG : S_OK)) return false;
    }
    // The original RGBA8 path used Vulkan's mip-tail decision even when D3D tiling reports this mip as packed.
    const FormatCase tiny{DXGI_FORMAT_R8G8B8A8_UNORM, 4, 128, 128, 1, 1};
    const VkResult tiny_support = caps.query_image(tiny, VK_FORMAT_R8G8B8A8_UNORM);
    const auto tiny_desc = sparse_test::texture_desc(tiny.format, tiny.width, tiny.height);
    ComPtr<ID3D12Resource> rgba8;
    if ((tiny_support != VK_SUCCESS && tiny_support != VK_ERROR_FORMAT_NOT_SUPPORTED) ||
        FAILED(device.CreateReservedResource(&tiny_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rgba8))) ||
        !rgba8 || rt::d3d12::is_sparse_image(rgba8.Get()) != (tiny_support == VK_SUCCESS)) {
        std::fprintf(stderr, "legacy 1x1 RGBA8 backing disagrees with native sparse requirements (%d)\n", int(tiny_support));
        return false;
    }
    std::printf("legacy 1x1 RGBA8: %s\n", tiny_support == VK_SUCCESS ? "real sparse" : "committed");
    if (!check("logical packed mip", sparse_test::texture_desc(DXGI_FORMAT_R32_UINT, 1, 1), nullptr, S_OK)) return false;
    auto desc = base;
    desc.MipLevels = 2;
    if (!check("multi-mip", desc, nullptr, S_OK)) return false;
    desc = base;
    desc.DepthOrArraySize = 2;
    if (!check("array", desc, nullptr, S_OK)) return false;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = desc.Format;
    if (!check("clear without RT/DS", base, &clear, E_INVALIDARG)) return false;
    desc = base;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (!check("render target", desc, nullptr, S_OK) || !check("optimized clear", desc, &clear, S_OK)) return false;
    ComPtr<ID3D12Device10> d10;
    ComPtr<ID3D12Resource> castable;
    const DXGI_FORMAT view = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    if (FAILED(device.QueryInterface(IID_PPV_ARGS(&d10))) ||
        FAILED(d10->CreateReservedResource2(&base, D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 1, &view,
                                            IID_PPV_ARGS(&castable))) || !castable || rt::d3d12::is_sparse_image(castable.Get())) {
        std::fputs("explicit castable list did not use committed backing\n", stderr);
        return false;
    }
    return true;
}

bool transfers_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_B5G6R5_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R11G11B10_FLOAT:
        return true;
    default: return false;
    }
}

struct Admitted {
    FormatCase c;
    bool warp;
};

bool mapping_formats(const std::vector<Admitted>& admitted, bool warp) {
    for (auto needed : {DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R16_UINT,
                        DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R32G32B32A32_UINT}) {
        if (std::count_if(admitted.begin(), admitted.end(), [&](const Admitted& a) {
                return a.c.format == needed && (!warp || a.warp);
            }) != 2) return false;
    }
    return true;
}

}  // namespace

int main() {
    ComPtr<ID3D12Device> device;
    const HRESULT init = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device));
    if (FAILED(init)) {
        // Standalone returns E_FAIL both without a compatible driver and for setup failures.
        vktest::Device probe;
        if (probe.initialize() == vktest::DeviceStatus::unavailable) {
            std::puts("SKIP: no compatible Vulkan device");
            return 77;
        }
        std::fprintf(stderr, "Vulkan standalone initialization failed, hr %#lx\n", static_cast<unsigned long>(init));
        return 1;
    }
    const auto* info = rt::d3d12::vulkan_device_info(device.Get());
    if (!info) return 1;
    SparseCaps caps{*info};
    if (!caps.query_device()) return 1;
    if (!caps.supported) {
        std::puts("GPU lacks strict aliased sparse 2D residency on the selected direct queue");
        return 77;
    }
    auto warp = create_warp();
    if (!warp) std::puts("Native WARP unavailable; Vulkan checks still run");
    unsigned skips = 0, committed = 0, failures = 0, native = 0;
    std::vector<Admitted> admitted;
    for (const Format& f : formats) {
        bool warp_support = false;
        if (warp) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{f.dxgi};
            const HRESULT hr = warp->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support));
            if (FAILED(hr) && hr != E_INVALIDARG) {
                std::fprintf(stderr, "WARP format support DXGI %u failed, hr %#lx\n", UINT(f.dxgi), static_cast<unsigned long>(hr));
                return 1;
            }
            warp_support = SUCCEEDED(hr) && (support.Support2 & D3D12_FORMAT_SUPPORT2_TILED);
            if (!warp_support) std::printf("SKIP WARP DXGI %u: format lacks tiled support\n", UINT(f.dxgi));
        }
        for (bool clipped : {false, true}) {
            const FormatCase c = format_case(f, clipped);
            const auto desc = sparse_test::texture_desc(f.dxgi, c.width, c.height);
            if (warp_support) {
                ComPtr<ID3D12Resource> reference;
                if (FAILED(warp->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&reference))) ||
                    !reference || !tiling_matches(*warp.Get(), *reference.Get(), c)) {
                    std::fprintf(stderr, "DXGI %u %ux%u: WARP standard tiling mismatch\n", UINT(f.dxgi), c.width, c.height);
                    ++failures;
                } else ++native;
            }
            ComPtr<ID3D12Resource> resource;
            const HRESULT hr = device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource));
            if (SUCCEEDED(hr) && resource && rt::d3d12::is_sparse_image(resource.Get())) {
                if (!tiling_matches(*device.Get(), *resource.Get(), c)) {
                    std::fprintf(stderr, "DXGI %u %ux%u: Vulkan standard tiling mismatch\n", UINT(f.dxgi), c.width, c.height);
                    ++failures;
                }
                admitted.push_back({c, warp_support});
                continue;
            }
            const VkResult physical = caps.query_image(c, f.backing);
            bool fallback_ok = SUCCEEDED(hr) && resource;
            if (physical == VK_ERROR_FORMAT_NOT_SUPPORTED && hr == E_INVALIDARG) {
                // If this ordinary image can be created, the reserved API owes us a committed fallback.
                auto ordinary = desc;
                ordinary.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
                const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                                 D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
                ComPtr<ID3D12Resource> control;
                const HRESULT ordinary_hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &ordinary,
                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&control));
                fallback_ok = ordinary_hr == E_INVALIDARG && !control;
            }
            if (physical == VK_ERROR_FORMAT_NOT_SUPPORTED && fallback_ok) {
                std::printf("SKIP DXGI %u %ux%u: no compatible Vulkan sparse image\n", UINT(f.dxgi), c.width, c.height);
                ++skips;
            } else {
                std::fprintf(stderr, "DXGI %u %ux%u: expected real sparse backing, got %s (hr %#lx, native Vulkan %d)\n",
                    UINT(f.dxgi), c.width, c.height, resource ? "committed" : "no resource", static_cast<unsigned long>(hr), int(physical));
                if (SUCCEEDED(hr) && resource) ++committed;
                else ++failures;
            }
        }
    }
    const unsigned total = unsigned(std::size(formats)) * 2;
    std::printf("admission: %zu/%u real sparse, %u hardware skips, %u unexpected committed; WARP tilings %u/%u\n",
                admitted.size(), total, skips, committed, native, total);
    if (failures || committed || !negative_cases(*device.Get(), caps)) return 1;
    if (admitted.empty()) return 77;
    unsigned transfers = 0, native_transfers = 0;
    for (const Admitted& a : admitted) {
        if (!transfers_format(a.c.format)) continue;
        if (!sparse_test::test_transfers(*device.Get(), a.warp ? warp.Get() : nullptr, a.c)) return 1;
        ++transfers;
        if (a.warp) ++native_transfers;
    }
    const bool mappings = mapping_formats(admitted, false), native_mappings = mapping_formats(admitted, true);
    if (mappings && !sparse_test::test_mappings(*device.Get(), false)) return 1;
    if (native_mappings && !sparse_test::test_mappings(*warp.Get(), true)) return 1;
    std::printf("transfer fixtures: %u Vulkan, %u WARP; mapping fixtures: %d Vulkan, %d WARP\n",
                transfers, native_transfers, int(mappings), int(native_mappings));
    return 0;
}
