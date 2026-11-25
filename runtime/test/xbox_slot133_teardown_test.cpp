// Xbox device slot 133 (CreatePlacedResourceX) textures and device teardown (RT-SLOT133-STANDIN-LEAK).
//
// A texture placed through slot 133 is a committed texture whose GPU VA is the title's address
// (create_buffer_at_host_address, docs/d3d12_vulkan.md). Once it is released and then the device, nothing of the
// device may stay behind: under LEONIDA_VK_VALIDATION=1 a device that never went leaves its Vulkan objects at
// vkDestroyDevice (VUID-vkDestroyDevice-device-05137, an error the validation callback counts). So the error count
// is read only after the Vulkan device is gone, which the other slot 133 tests do not do.
//
// The ~60 objects reported for this path were the test's, not the layer's: `placed = {}` on a d3d12test::Ref copied a
// null pointer over the reference without releasing it, so the texture, and through it the device, lived on. Ref now
// releases on that assignment, and this test drops its objects the same way.
//
// The shapes are the ones the finding named: a 1280x720 format-118 (console D16S8) depth target without unordered
// access and a plain RGBA8 texture, plus an RGBA8 render target and a buffer through the same slot. No game files.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include <cstdio>
#include <memory>

using namespace d3d12test;
#define VERIFY(expr) do { if (!(expr)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {
using Place = HRESULT(WINAPI*)(void*, UINT64, const D3D12_RESOURCE_DESC*, UINT, const D3D12_CLEAR_VALUE*, const IID*,
                               void**);

// One slot 133 placement on fresh title pages, released again. `gpu_va` is what the texture must report.
int place(ID3D12Device* xbox, const D3D12_RESOURCE_DESC& desc, const D3D12_CLEAR_VALUE* clear, const char* what) {
    constexpr SIZE_T pages = 16u << 20;  // past each shape's size on any GPU
    void* va = VirtualAlloc(nullptr, pages, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    VERIFY(va);
    const unsigned long before = rt::d3d12_xs::device_proxy_ref_count(xbox);
    Ref<ID3D12Resource> placed;
    const HRESULT hr = reinterpret_cast<Place>(slot(xbox, 133))(xbox, reinterpret_cast<UINT64>(va), &desc,
                                                                0xFFFFFFFFu, clear, &__uuidof(ID3D12Resource),
                                                                reinterpret_cast<void**>(&placed.p));
    const bool made = SUCCEEDED(hr) && placed.p;
    if (!made) std::fprintf(stderr, "slot 133 %s: hr=0x%08x\n", what, unsigned(hr));
    // The resource's GetDevice answers the proxy, so it holds a reference to it while it lives, and gives it back.
    Ref<ID3D12Device> parent;
    const bool device_ok = made && SUCCEEDED(placed->GetDevice(IID_PPV_ARGS(&parent.p))) && parent.p == xbox;
    const unsigned long during = rt::d3d12_xs::device_proxy_ref_count(xbox);
    parent = {};
    // `= {}` is how a test drops an object mid-test: it must release it (d3d12test::Ref's move assignment). It used
    // to copy a null pointer over the reference, and the texture, and through it the device, outlived the test.
    placed = {};
    VERIFY(!placed.p && !parent.p);
    const unsigned long after = rt::d3d12_xs::device_proxy_ref_count(xbox);
    VERIFY(!made || during == before + 2);  // the texture's reference and GetDevice's
    VirtualFree(va, 0, MEM_RELEASE);
    VERIFY(made);
    VERIFY(device_ok);
    if (after != before)
        std::fprintf(stderr, "slot 133 %s: the proxy had %lu references before and keeps %lu after release\n", what,
                     before, after);
    VERIFY(after == before);
    std::printf("slot 133 %s: placed and released, device references %lu -> %lu\n", what, before, after);
    return 0;
}

int run(vktest::Device& gpu) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.bindless = gpu.bindless;
    info.bindless_partially_bound = gpu.partially_bound;
    info.get_instance_proc = gpu.instance_proc;
    info.instance = gpu.instance;
    info.physical = gpu.physical;
    info.device = gpu.device;
    info.queue = gpu.queue;
    info.queue_family = gpu.family;
    info.buffer_device_address = true;
    ID3D12Device* dev{};
    VERIFY(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev))));
    ID3D12Device* proxy{};
    VERIFY(SUCCEEDED(rt::d3d12_xs::create_device_proxy(dev, __uuidof(ID3D12Device), reinterpret_cast<void**>(&proxy),
                                                       true)));
    rt::d3d12::set_device_outer(dev, proxy);  // the proxy owns `dev` from here
    VERIFY(rt::d3d12_xs::device_proxy_ref_count(proxy) == 1);

    D3D12_RESOURCE_DESC depth{};
    depth.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depth.Width = 1280;
    depth.Height = 720;
    depth.DepthOrArraySize = depth.MipLevels = 1;
    depth.SampleDesc.Count = 1;
    depth.Format = static_cast<DXGI_FORMAT>(118);  // the console's D16S8
    depth.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
    depth.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE depth_clear{};
    depth_clear.Format = depth.Format;
    depth_clear.DepthStencil.Depth = 1;
    VERIFY(place(proxy, depth, &depth_clear, "1280x720 format-118 depth target without UAV") == 0);

    D3D12_RESOURCE_DESC color = depth;
    color.Width = color.Height = 256;
    color.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    color.Flags = D3D12_RESOURCE_FLAG_NONE;
    VERIFY(place(proxy, color, nullptr, "256x256 RGBA8 texture") == 0);

    color.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE color_clear{};
    color_clear.Format = color.Format;
    VERIFY(place(proxy, color, &color_clear, "256x256 RGBA8 render target") == 0);

    D3D12_RESOURCE_DESC buffer = buffer_desc(65536);
    buffer.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(1);  // the Xbox row-major word
    VERIFY(place(proxy, buffer, nullptr, "64 KiB buffer") == 0);

    VERIFY(rt::d3d12_xs::device_proxy_ref_count(proxy) == 1);
    VERIFY(proxy->Release() == 0);
    return 0;
}
}  // namespace

int main() {
    {
        auto gpu = std::make_unique<vktest::Device>();
        const auto status = gpu->initialize();
        if (status == vktest::DeviceStatus::unavailable) return 77;
        VERIFY(status == vktest::DeviceStatus::ready);
        VERIFY(run(*gpu) == 0);
    }  // vkDestroyDevice: a device object still alive is reported here, under validation
    if (vktest::validation_error_count() != 0) {
        std::fprintf(stderr, "%u validation errors, counted through vkDestroyDevice\n",
                     vktest::validation_error_count());
        return 1;
    }
    std::printf("slot 133 placements leave nothing of the device behind at vkDestroyDevice\n");
    return 0;
}
