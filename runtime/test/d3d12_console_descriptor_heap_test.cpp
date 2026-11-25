// Console Xbox CreateDescriptorHeap (issue #313 comment 7107265058) passes a 16-byte PC
// D3D12_DESCRIPTOR_HEAP_DESC followed by address at +0x10 and size at +0x18. The title asserts
// GetCPUDescriptorHandleForHeapStart() == address. This tests the explicit Leonida console-heap
// creation path used by the Xbox device proxy slot 14. It does not cover title-memory aliasing /
// section 3.3 backing.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/title_page_watch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// The pages a test write watch made read-only, and its opener: the D3D12 layer's slot-133 shadows protect the title's
// pages the same way and give them back through leonida::kernel_writes.
std::uintptr_t g_watched_begin = 0, g_watched_end = 0;
void open_watched(std::uintptr_t from, std::uintptr_t to) {
    DWORD old = 0;
    if (from < g_watched_end && to > g_watched_begin)
        VirtualProtect(reinterpret_cast<void*>(g_watched_begin), g_watched_end - g_watched_begin, PAGE_READWRITE, &old);
}

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

bool check_xs_proxy_slot14(ID3D12Device* real_device, bool expect_console) {
    void* proxy_raw = nullptr;
    // The proxy takes one reference to the device on success; the caller's Ref keeps its own.
    real_device->AddRef();
    if (FAILED(rt::d3d12_xs::create_device_proxy(real_device, __uuidof(ID3D12Device), &proxy_raw)) || !proxy_raw) {
        real_device->Release();
        std::fprintf(stderr, "create_device_proxy failed\n");
        return false;
    }
    Ref<ID3D12Device> proxy;
    proxy.p = static_cast<ID3D12Device*>(proxy_raw);

    struct ConsoleDescriptorHeapDesc {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        std::uint64_t address = 0;
        std::uint64_t size = 0;
    } console{};
    console.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    console.desc.NumDescriptors = 4;
    console.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    alignas(64) unsigned char storage[4096]{};
    console.address = reinterpret_cast<std::uintptr_t>(storage);
    console.size = console.desc.NumDescriptors * real_device->GetDescriptorHandleIncrementSize(console.desc.Type);
    if (console.size > sizeof(storage)) return false;
    D3D12_DESCRIPTOR_HEAP_DESC ordinary = console.desc;
    Ref<ID3D12DescriptorHeap> ordinary_heap;
    if (FAILED(proxy->CreateDescriptorHeap(&ordinary, IID_PPV_ARGS(ordinary_heap.put())))) return false;

    auto check_start = [&](const char* label, bool uses_console_address) {
        Ref<ID3D12DescriptorHeap> heap;
        const HRESULT hr = proxy->CreateDescriptorHeap(&console.desc, IID_PPV_ARGS(heap.put()));
        if (FAILED(hr) || !heap.get()) {
            std::fprintf(stderr, "%s: CreateDescriptorHeap failed: 0x%08lx\n", label, static_cast<unsigned long>(hr));
            return false;
        }
        // Xbox heap slot 9 returns the handle in RAX rather than through a hidden PC return pointer.
        using GetCpuHandleFn = D3D12_CPU_DESCRIPTOR_HANDLE(WINAPI*)(ID3D12DescriptorHeap*);
        void** const vtbl = *reinterpret_cast<void***>(heap.get());
        const auto cpu = reinterpret_cast<GetCpuHandleFn>(vtbl[9])(heap.get());
        if (!cpu.ptr || (cpu.ptr == console.address) != uses_console_address) {
            std::fprintf(stderr, "%s: expected %s handle, got 0x%llx (console address 0x%llx)\n",
                         label, uses_console_address ? "console" : "ordinary",
                         static_cast<unsigned long long>(cpu.ptr), static_cast<unsigned long long>(console.address));
            return false;
        }
        return true;
    };
    if (!expect_console) return check_start("host device", false);
    if (!check_start("writable backing", true)) return false;
    ++console.size;
    if (!check_start("size mismatch", false)) return false;
    --console.size;

    void* read_only = VirtualAlloc(nullptr, console.size, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY);
    if (!read_only) return false;
    console.address = reinterpret_cast<std::uintptr_t>(read_only);
    const bool ok = check_start("read-only backing", false);
    VirtualFree(read_only, 0, MEM_RELEASE);
    if (!ok) return false;

    // Writable memory a write watch has made read-only (the retail title places a slot-133 buffer over the memory it
    // carves its heap from, and that buffer's shadow watches it) is still the title's writable memory.
    void* watched = VirtualAlloc(nullptr, console.size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    DWORD old = 0;
    if (!watched || !VirtualProtect(watched, console.size, PAGE_READONLY, &old)) return false;
    g_watched_begin = reinterpret_cast<std::uintptr_t>(watched);
    g_watched_end = g_watched_begin + console.size;
    leonida::kernel_writes::set_opener(&open_watched);
    console.address = g_watched_begin;
    const bool watched_ok = check_start("watched backing", true);
    leonida::kernel_writes::set_opener(nullptr);
    VirtualFree(watched, 0, MEM_RELEASE);
    return watched_ok;
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

void* g_next_backing = nullptr;
void* test_heap_alloc(std::size_t, std::uint64_t) {
    void* p = g_next_backing;
    g_next_backing = nullptr;
    return p;
}
void test_heap_free(void*, std::uint64_t) {}

bool check_mapped_title_heap_sync_and_copy(ID3D12Device* real_device) {
    struct VirtualMem {
        void* p = nullptr;
        ~VirtualMem() { if (p) VirtualFree(p, 0, MEM_RELEASE); }
    };
    struct AllocGuard {
        ~AllocGuard() {
            g_next_backing = nullptr;
            rt::d3d12_xs::set_descriptor_heap_allocator(nullptr, nullptr);
        }
    };
    VirtualMem sv_mem{VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)};
    if (!sv_mem.p) return false;
    AllocGuard alloc_guard;

    real_device->AddRef();
    Ref<ID3D12Device> proxy;
    if (FAILED(rt::d3d12_xs::create_device_proxy(real_device, IID_PPV_ARGS(proxy.put()), false)) || !proxy.get()) {
        real_device->Release();
        std::fprintf(stderr, "create_device_proxy failed\n");
        return false;
    }
    const UINT stride = real_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (stride != 32) {
        std::fprintf(stderr, "unexpected CBV_SRV_UAV descriptor stride %u\n", stride);
        return false;
    }

    // Obtain three distinct, valid CBV descriptor records from the layer.
    D3D12_DESCRIPTOR_HEAP_DESC staging_desc{};
    staging_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    staging_desc.NumDescriptors = 4;
    staging_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    Ref<ID3D12DescriptorHeap> staging;
    if (FAILED(real_device->CreateDescriptorHeap(&staging_desc, IID_PPV_ARGS(staging.put()))) || !staging.get()) {
        std::fprintf(stderr, "CreateDescriptorHeap(staging) failed\n");
        return false;
    }
    const SIZE_T staging_base = staging->GetCPUDescriptorHandleForHeapStart().ptr;
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv0{0x10000ull, 256u};
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv1{0x20000ull, 512u};
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv2{0x30000ull, 768u};
    real_device->CreateConstantBufferView(&cbv0, D3D12_CPU_DESCRIPTOR_HANDLE{staging_base});
    real_device->CreateConstantBufferView(&cbv1, D3D12_CPU_DESCRIPTOR_HANDLE{staging_base + stride});
    real_device->CreateConstantBufferView(&cbv2, D3D12_CPU_DESCRIPTOR_HANDLE{staging_base + 2 * stride});
    std::vector<std::uint8_t> bytes0(stride), bytes1(stride), bytes2(stride);
    std::memcpy(bytes0.data(), reinterpret_cast<const void*>(staging_base), stride);
    std::memcpy(bytes1.data(), reinterpret_cast<const void*>(staging_base + stride), stride);
    std::memcpy(bytes2.data(), reinterpret_cast<const void*>(staging_base + 2 * stride), stride);
    if (bytes0 == bytes1 || bytes1 == bytes2) {
        std::fprintf(stderr, "staging CBVs were not distinct\n");
        return false;
    }

    constexpr UINT kSlots = 8;
    const SIZE_T heap_bytes = static_cast<SIZE_T>(kSlots) * stride;

    // 1. Pre-fill slot 0 BEFORE CreateDescriptorHeap(shader-visible) with custom XMem backing:
    //    must sync into the GPU-arena real heap on the very first ExecuteCommandLists.
    std::memset(sv_mem.p, 0, heap_bytes);
    std::memcpy(sv_mem.p, bytes0.data(), stride);

    rt::d3d12_xs::set_descriptor_heap_allocator(&test_heap_alloc, &test_heap_free);
    g_next_backing = sv_mem.p;
    D3D12_DESCRIPTOR_HEAP_DESC sv_desc{};
    sv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    sv_desc.NumDescriptors = kSlots;
    sv_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Ref<ID3D12DescriptorHeap> sv_heap;
    if (FAILED(proxy->CreateDescriptorHeap(&sv_desc, IID_PPV_ARGS(sv_heap.put()))) || !sv_heap.get()) {
        std::fprintf(stderr, "proxy CreateDescriptorHeap(custom shader-visible) failed\n");
        return false;
    }
    auto* sv_real = static_cast<ID3D12DescriptorHeap*>(rt::d3d12_xs::real_object_behind_proxy(sv_heap.p));
    if (!sv_real) {
        std::fprintf(stderr, "custom shader-visible heap has no real heap\n");
        return false;
    }

    D3D12_COMMAND_QUEUE_DESC qdesc{};
    qdesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    if (FAILED(proxy->CreateCommandQueue(&qdesc, IID_PPV_ARGS(queue.put()))) || !queue.get()) {
        std::fprintf(stderr, "proxy CreateCommandQueue failed\n");
        return false;
    }

    // First submit: pre-filled slot 0 must reach sv_real immediately.
    queue->ExecuteCommandLists(0, nullptr);
    const SIZE_T sv_real_start = sv_real->GetCPUDescriptorHandleForHeapStart().ptr;
    if (std::memcmp(reinterpret_cast<const void*>(sv_real_start), bytes0.data(), stride) != 0) {
        std::fprintf(stderr, "pre-filled custom shader-visible heap slot 0 did not sync on first submit\n");
        return false;
    }

    // Post-creation direct CPU write to slot 1: must sync on next submit.
    std::memcpy(static_cast<std::uint8_t*>(sv_mem.p) + stride, bytes1.data(), stride);
    queue->ExecuteCommandLists(0, nullptr);
    if (std::memcmp(reinterpret_cast<const void*>(sv_real_start + stride), bytes1.data(), stride) != 0) {
        std::fprintf(stderr, "post-creation direct CPU write to slot 1 did not sync on submit\n");
        return false;
    }

    // 2. Direct CPU writes into custom backing as source of CopyDescriptorsSimple and CopyDescriptors
    //    (before any ExecuteCommandLists): must reach an ordinary destination heap on the CPU timeline.
    std::memcpy(static_cast<std::uint8_t*>(sv_mem.p) + 2 * stride, bytes2.data(), stride);
    std::memcpy(static_cast<std::uint8_t*>(sv_mem.p) + 3 * stride, bytes0.data(), stride);

    Ref<ID3D12DescriptorHeap> ordinary_dst;
    if (FAILED(real_device->CreateDescriptorHeap(&staging_desc, IID_PPV_ARGS(ordinary_dst.put()))) ||
        !ordinary_dst.get()) {
        std::fprintf(stderr, "CreateDescriptorHeap(ordinary_dst) failed\n");
        return false;
    }
    const SIZE_T dst_start = ordinary_dst->GetCPUDescriptorHandleForHeapStart().ptr;
    const SIZE_T src_start = reinterpret_cast<SIZE_T>(sv_mem.p);

    proxy->CopyDescriptorsSimple(1, D3D12_CPU_DESCRIPTOR_HANDLE{dst_start},
                                 D3D12_CPU_DESCRIPTOR_HANDLE{src_start + 2 * stride},
                                 D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (std::memcmp(reinterpret_cast<const void*>(dst_start), bytes2.data(), stride) != 0) {
        std::fprintf(stderr, "CopyDescriptorsSimple from custom-backed heap missed direct CPU write\n");
        return false;
    }

    const D3D12_CPU_DESCRIPTOR_HANDLE dst_range{dst_start + stride};
    const D3D12_CPU_DESCRIPTOR_HANDLE src_range{src_start + 3 * stride};
    const UINT one = 1;
    proxy->CopyDescriptors(1, &dst_range, &one, 1, &src_range, &one, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (std::memcmp(reinterpret_cast<const void*>(dst_start + stride), bytes0.data(), stride) != 0) {
        std::fprintf(stderr, "CopyDescriptors from custom-backed heap missed direct CPU write\n");
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
    Ref<ID3D12Device> host;
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(host.put())))) return 1;
    if (!check_xs_proxy_slot14(host.get(), false)) return 1;
    if (!check_xs_proxy_slot14(device.get(), true)) return 1;
    if (!check_mapped_title_heap_sync_and_copy(device.get())) return 1;
    std::fprintf(stderr, "ok\n");
    return 0;
}
