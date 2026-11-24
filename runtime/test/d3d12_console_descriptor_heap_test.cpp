// Console Xbox CreateDescriptorHeap (issue #313 comment 7107265058) passes a 16-byte PC
// D3D12_DESCRIPTOR_HEAP_DESC followed by address at +0x10 and size at +0x18. The title asserts
// GetCPUDescriptorHandleForHeapStart() == address. This tests the explicit Leonida console-heap
// creation path used by the Xbox device proxy slot 14. It does not cover title-memory aliasing /
// section 3.3 backing.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <cstdint>
#include <cstdio>

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T** put() {
        if (p) {
            p->Release();
            p = nullptr;
        }
        return &p;
    }
    T* get() const { return p; }
    T* operator->() const { return p; }
};

bool check_console_addressed_heap(ID3D12Device* device) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 8;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    desc.NodeMask = 0;
    const D3D12_GPU_VIRTUAL_ADDRESS address = 0x112233440000ull;
    const std::uint64_t size = 8ull * 32ull;

    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = rt::d3d12::create_console_descriptor_heap(
        device, &desc, address, size, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "create_console_descriptor_heap failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return false;
    }
    const auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    if (cpu.ptr != static_cast<SIZE_T>(address)) {
        std::fprintf(stderr, "CPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(cpu.ptr), static_cast<unsigned long long>(address));
        return false;
    }
    const auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
    if (gpu.ptr != address) {
        std::fprintf(stderr, "GPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(gpu.ptr), static_cast<unsigned long long>(address));
        return false;
    }
    const D3D12_DESCRIPTOR_HEAP_DESC got = heap->GetDesc();
    if (got.Type != desc.Type || got.NumDescriptors != desc.NumDescriptors || got.Flags != desc.Flags ||
        got.NodeMask != 1) {
        std::fprintf(stderr, "GetDesc mismatch\n");
        return false;
    }
    return true;
}

bool check_xs_proxy_slot14(ID3D12Device* real_device) {
    Ref<ID3D12Device> proxy;
    real_device->AddRef();  // create_device_proxy takes ownership of one reference on success
    if (FAILED(rt::d3d12_xs::create_device_proxy(real_device, IID_PPV_ARGS(&proxy.p))) || !proxy.p) {
        real_device->Release();  // the caller keeps its reference on failure
        std::fprintf(stderr, "create_device_proxy failed\n");
        return false;
    }

    struct ConsoleDescriptorHeapDesc {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        std::uint64_t address = 0;
        std::uint64_t size = 0;
    } console{};
    console.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    console.desc.NumDescriptors = 4;
    console.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    console.address = 0x556677880000ull;
    console.size = 4ull * 32ull;

    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = proxy->CreateDescriptorHeap(&console.desc, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "proxy CreateDescriptorHeap(console) failed: 0x%08lx\n",
                     static_cast<unsigned long>(hr));
        return false;
    }
    // The proxy's own GetCPUDescriptorHandleForHeapStart answers the console (RAX) convention; this test is a PC
    // caller, so read the wrapped heap's PC method. The address proves slot 14 passed the console VA through.
    ID3D12DescriptorHeap* real = rt::d3d12_xs::heap_proxy_real_heap(heap.p);
    if (!real) {
        std::fprintf(stderr, "heap proxy has no real heap\n");
        return false;
    }
    const auto cpu = real->GetCPUDescriptorHandleForHeapStart();
    if (cpu.ptr != static_cast<SIZE_T>(console.address)) {
        std::fprintf(stderr, "proxy CPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(cpu.ptr),
                     static_cast<unsigned long long>(console.address));
        return false;
    }

    // An ordinary one-descriptor desc followed by the words 1 and 32 (what d3d12_draw_test's RTV desc had after it
    // on the stack under MSVC RelWithDebInfo): the size matches 1 * 32, but 0x1 is no heap address, so slot 14 has
    // to create an ordinary heap instead of handing out 0x1 as the heap start.
    ConsoleDescriptorHeapDesc stray{};
    stray.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    stray.desc.NumDescriptors = 1;
    stray.address = 1;
    stray.size = 32;
    Ref<ID3D12DescriptorHeap> stray_heap;
    const HRESULT stray_hr = proxy->CreateDescriptorHeap(&stray.desc, IID_PPV_ARGS(stray_heap.put()));
    ID3D12DescriptorHeap* stray_real = SUCCEEDED(stray_hr) && stray_heap.get()
                                           ? rt::d3d12_xs::heap_proxy_real_heap(stray_heap.p) : nullptr;
    if (!stray_real) {
        std::fprintf(stderr, "proxy CreateDescriptorHeap(stray trailing words) failed: 0x%08lx\n",
                     static_cast<unsigned long>(stray_hr));
        return false;
    }
    const auto stray_cpu = stray_real->GetCPUDescriptorHandleForHeapStart();
    if (stray_cpu.ptr == 0 || stray_cpu.ptr == static_cast<SIZE_T>(stray.address)) {
        std::fprintf(stderr, "stray trailing words were taken as a console heap: CPU start 0x%llx\n",
                     static_cast<unsigned long long>(stray_cpu.ptr));
        return false;
    }
    return true;
}

bool check_normal_heap_unchanged(ID3D12Device* device) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 4;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "CreateDescriptorHeap(normal) failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return false;
    }
    const auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    if (cpu.ptr == 0) {
        std::fprintf(stderr, "normal heap CPU start is null\n");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    Ref<ID3D12Device> device;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(device.put()))) || !device.get()) {
        std::fprintf(stderr, "create_vulkan_device failed\n");
        return 1;
    }

    if (!check_normal_heap_unchanged(device.get())) return 1;
    if (!check_console_addressed_heap(device.get())) return 1;
    if (!check_xs_proxy_slot14(device.get())) return 1;
    std::fprintf(stderr, "ok\n");
    return 0;
}
