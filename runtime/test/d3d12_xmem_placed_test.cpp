// Xbox device slot 133 (CreatePlacedResourceX) on xmem physical pages that were mapped but never touched. xmem
// commits physical pages on first touch, so such a range is still reserved when the title places a buffer on it, and
// the Vulkan layer's import (rt::d3d12::create_buffer_at_host_address) refuses reserved pages and answers with its
// fallback buffer: a shadow that copies only committed pages, so a reserved page reads as zeros and a write to it
// faults. With rt::xmem::commit_physical_range registered as the slot's
// pages hook (rt::d3d12_xs::set_placed_at_va_pages_hook, as gta6rt does), the pages are committed first: the buffer's
// GPU VA is the window itself and a GPU copy reads what the CPU writes there afterwards.
// Needs VK_EXT_external_memory_host (skipped otherwise). No shaders or game data.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../shims/xmem.h"

#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)

namespace {

const char* const who = "xmem slot-133 buffer";

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, std::uint64_t*, std::uint64_t*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, const std::uint64_t*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, std::uint64_t, const std::uint64_t*);
using SlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t);
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;
constexpr std::uint64_t kPages = 4;

bool state_is(const void* address, DWORD state) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) && info.State == state;
}

// Slot 133 as the title calls it: (va, desc, layout, clear, riid, out).
HRESULT place(void* device, void* va, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** out) {
    void* const* vtable = *static_cast<void* const* const*>(device);
    const GUID iid = __uuidof(ID3D12Resource);
    const std::uint64_t r = reinterpret_cast<SlotFn>(vtable[133])(
        device, reinterpret_cast<std::uint64_t>(va), reinterpret_cast<std::uint64_t>(&desc), 0xFFFFFFFFull, 0,
        reinterpret_cast<std::uint64_t>(&iid), reinterpret_cast<std::uint64_t>(out));
    return static_cast<HRESULT>(static_cast<std::uint32_t>(r));
}

int gpu_reads_cpu_writes(ID3D12Device* real, ID3D12Resource* buffer, const void* pages, UINT64 size) {
    D3D12_RESOURCE_DESC desc = buffer->GetDesc();
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    EXPECT(SUCCEEDED(real->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(real->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(real->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(real->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    list->CopyBufferRegion(readback.p, 0, buffer, 0, size);
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(SUCCEEDED(real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, done)));
    EXPECT(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
    CloseHandle(done);
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const bool same = std::memcmp(mapped, pages, static_cast<size_t>(size)) == 0;
    readback->Unmap(0, nullptr);
    EXPECT(same);
    return 0;
}

int run(ID3D12Device* real, void* device, bool eager) {
    const auto virtual_alloc = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    const auto allocate =
        reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    const auto free_pages =
        reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    const auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    EXPECT(virtual_alloc && allocate && free_pages && map);

    constexpr UINT64 size = kPages * kPage64k;
    auto* const window =
        static_cast<std::uint8_t*>(virtual_alloc(nullptr, size, MEM_RESERVE | kXmem64kPages, 0x23, PAGE_READWRITE));
    EXPECT(window != nullptr);
    std::uint64_t pages[kPages]{};
    std::uint64_t count = kPages;
    EXPECT(allocate(0x280, &count, pages) == TRUE && count == kPages);
    EXPECT(map(window, kPages, pages) == window);
    EXPECT(state_is(window, eager ? MEM_COMMIT : MEM_RESERVE));  // verify the selected physical-page mapper

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x1);  // the Xbox row-major word
    const auto window_va = reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(window);

    // Without the hook the reserved pages are refused and the fallback answers. Its GPU VA is the title's address
    // either way: the imported buffer is bound to the window (create_host_buffer: va_ = start), and the shadow keeps
    // that VA too so the descriptors and views the title builds from it still find the buffer (89261c7ed,
    // d3d12_existing_heap_test.cpp). The reserved state is what shows the import refused these pages.
    rt::d3d12_xs::set_placed_at_va_pages_hook(nullptr);
    {
        Ref<ID3D12Resource> fallback;
        EXPECT(place(device, window, desc, &fallback.p) == S_OK && fallback.p);
        if (eager) {
            EXPECT(fallback->GetGPUVirtualAddress() == window_va && state_is(window, MEM_COMMIT));
            for (UINT64 i = 0; i < size; ++i) window[i] = static_cast<std::uint8_t>(i * 17u + 29u);
            if (gpu_reads_cpu_writes(real, fallback.p, window, size)) return 1;
        } else {
            EXPECT(fallback->GetGPUVirtualAddress() == window_va);
            EXPECT(state_is(window, MEM_RESERVE));
        }
    }
    std::printf("%s: physical pages without the hook: %s\n", who, eager ? "already committed and GPU-visible" : "still reserved");

    // With it the range is committed first and imported.
    rt::d3d12_xs::set_placed_at_va_pages_hook(&rt::xmem::commit_physical_range);
    {
        Ref<ID3D12Resource> buffer;
        EXPECT(place(device, window, desc, &buffer.p) == S_OK && buffer.p);
        EXPECT(state_is(window, MEM_COMMIT) && state_is(window + size - 1, MEM_COMMIT));
        EXPECT(buffer->GetGPUVirtualAddress() == window_va);
        for (UINT64 i = 0; i < size; ++i) window[i] = static_cast<std::uint8_t>(i * 131u + 7u);
        if (gpu_reads_cpu_writes(real, buffer.p, window, size)) return 1;
    }
    rt::d3d12_xs::set_placed_at_va_pages_hook(nullptr);
    std::printf("%s: with commit_physical_range: %llu bytes committed, GPU VA is the window, a GPU copy reads the "
                "CPU's writes\n", who, static_cast<unsigned long long>(size));

    EXPECT(map(window, kPages, nullptr) == window);
    EXPECT(free_pages(count, pages) == TRUE);
    return 0;
}

int linear_texture_mapping(ID3D12Device* dev, ID3D12Device* xbox, bool private_backing) {
    const auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll","XMemVirtualAlloc"));
    const auto alloc = reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll","XMemAllocatePhysicalPages"));
    const auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll","XMemMapPhysicalPages"));
    const auto release = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll","XMemFreePhysicalPages"));
    EXPECT(reserve && alloc && map && release);
    std::uint64_t count=1,page{};
    void* window = private_backing ? VirtualAlloc(nullptr,kPage64k,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE)
                                   : reserve(nullptr,kPage64k,MEM_RESERVE|kXmem64kPages,0x23,PAGE_READWRITE);
    EXPECT(window);
    if (!private_backing) EXPECT(alloc(0x280,&count,&page) && map(window,1,&page)==window);
    rt::d3d12::set_device_outer(dev,xbox);
    rt::d3d12_xs::set_texture_page_provider(&rt::xmem::pin_mapped_data);
    constexpr UINT w=9,h=8;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width=w;desc.Height=h;desc.MipLevels=desc.DepthOrArraySize=1;desc.SampleDesc.Count=1;
    desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    desc.Layout=static_cast<D3D12_TEXTURE_LAYOUT>(0x100);
    Ref<ID3D12Resource> tex,rb;
    EXPECT(place(xbox,window,desc,&tex.p)==S_OK && tex->GetGPUVirtualAddress()==reinterpret_cast<UINT64>(window));
    constexpr UINT pitch=256;
    D3D12_RANGE empty{0,0};void* cpu=nullptr;
    EXPECT(tex->Map(0,&empty,&cpu)==S_OK && cpu && ((cpu!=window)==private_backing));
    std::array<float,4> pixel{};
    for(UINT y=0;y<h;++y) for(UINT x=0;x<w;++x) {
        for(UINT c=0;c<4;++c) pixel[c]=float(y*w*4+x*4+c)/512.0f;
        std::memcpy(static_cast<std::uint8_t*>(cpu)+y*pitch+x*16,pixel.data(),16);
    }
    D3D12_HEAP_PROPERTIES read_heap{D3D12_HEAP_TYPE_READBACK};
    const auto buf=d3d12test::buffer_desc(h*pitch);
    EXPECT(SUCCEEDED(dev->CreateCommittedResource(&read_heap,D3D12_HEAP_FLAG_NONE,&buf,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&rb.p))));
    D3D12_COMMAND_QUEUE_DESC q{};
    Ref<ID3D12CommandQueue> queue;Ref<ID3D12CommandAllocator> commands;Ref<ID3D12GraphicsCommandList> list;Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(dev->CreateCommandQueue(&q,IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&commands.p))));
    EXPECT(SUCCEEDED(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,commands.p,nullptr,IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence.p))));
    const auto submit=[&](UINT64 value) {
        if(FAILED(list->Close())) return false;
        ID3D12CommandList* batch[]{list.p};queue->ExecuteCommandLists(1,batch);
        if(FAILED(queue->Signal(fence.p,value))) return false;
        const auto end=GetTickCount64()+6000;
        while(fence->GetCompletedValue()<value && GetTickCount64()<end) Sleep(1);
        return fence->GetCompletedValue()==value;
    };
    auto barrier=d3d12test::transition(tex.p,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1,&barrier);
    D3D12_TEXTURE_COPY_LOCATION from{},to{};from.pResource=tex.p;to.pResource=rb.p;
    to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;to.PlacedFootprint={0,{desc.Format,w,h,1,pitch}};
    list->CopyTextureRegion(&to,0,0,0,&from,nullptr);
    EXPECT(submit(1));
    void* bytes=nullptr;EXPECT(SUCCEEDED(rb->Map(0,nullptr,&bytes)));
    for(UINT y=0;y<h;++y) EXPECT(std::memcmp(static_cast<const std::uint8_t*>(bytes)+y*pitch,
                                             static_cast<const std::uint8_t*>(cpu)+y*pitch,w*16)==0);
    rb->Unmap(0,nullptr);
    tex->Unmap(0,&empty);
    Ref<ID3D12DescriptorHeap> rt;
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV,1};
    EXPECT(SUCCEEDED(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&rt.p))));
    const auto handle=rt->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(tex.p,nullptr,handle);
    EXPECT(SUCCEEDED(commands->Reset()) && SUCCEEDED(list->Reset(commands.p,nullptr)));
    barrier=d3d12test::transition(tex.p,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
    list->ResourceBarrier(1,&barrier);
    const float rgba[]{.25f,.5f,.75f,1.0f};
    list->ClearRenderTargetView(handle,rgba,0,nullptr);
    EXPECT(submit(2));
    D3D12_RANGE read{0,h*pitch};cpu=nullptr;
    EXPECT(tex->Map(0,&read,&cpu)==S_OK && cpu && ((cpu!=window)==private_backing));
    for(UINT y=0;y<h;++y) for(UINT x=0;x<w;++x)
        EXPECT(std::memcmp(static_cast<const std::uint8_t*>(cpu)+y*pitch+x*16,rgba,16)==0);
    tex->Unmap(0,&empty);
    if (!private_backing) {
        EXPECT(map(window,1,nullptr)==window && release(1,&page));
        cpu=reinterpret_cast<void*>(1);
        EXPECT(FAILED(tex->Map(0,&empty,&cpu)) && !cpu);
    } else {
        const auto* original=static_cast<const std::uint8_t*>(window);
        for(UINT i=0;i<kPage64k;++i) EXPECT(original[i]==0);
        EXPECT(VirtualFree(window,0,MEM_RELEASE));
    }
    std::printf("linear placed texture: %u GPU texels match writes during Map, %u mapped texels match the GPU clear, released placement refused\n",w*h,w*h);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool eager = argc == 2 && std::strcmp(argv[1], "--eager-mapping") == 0;
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.host_pointer_alignment) {
        std::printf("%s: skipped, no VK_EXT_external_memory_host\n", who);
        return 77;
    }
    int failures = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.host_pointer_alignment = vk.host_pointer_alignment;
        Ref<ID3D12Device> real;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&real.p)))) return 1;
        real->AddRef();  // the proxy takes one reference
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12_xs::create_device_proxy(real.p, __uuidof(ID3D12Device),
                                                     reinterpret_cast<void**>(&device.p)))) {
            real->Release();
            return 1;
        }
        if (argc==2 && (std::strcmp(argv[1],"--linear-map")==0 || std::strcmp(argv[1],"--linear-staging")==0))
            failures+=linear_texture_mapping(real.p,device.p,std::strcmp(argv[1],"--linear-staging")==0);
        else failures += run(real.p, device.p, eager);
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}
