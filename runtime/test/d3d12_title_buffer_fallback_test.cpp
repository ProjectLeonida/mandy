// A vertex or index view that names the title's own memory gets a buffer made over those pages (make_title_buffer,
// TitleVaAliases::want_buffer), the way slot 133 makes one: the pages imported, or a shadow copied from them. When
// neither can be made, slot 133's last resort is an ordinary DEFAULT buffer, which has neither the title's VA nor its
// bytes. A view must not keep such a buffer: it does not cover the view, so the next IASetVertexBuffers of the same
// view would make another, each holding device memory and a device reference until the device goes, until
// LEONIDA_TITLE_BUFFERS_MB is used up and no new title view is resolved for the rest of the session.
//   - The arena is filled first (LEONIDA_ARENA_MB=16, so a few 1 MiB buffers do it) and host-visible allocations are
//     refused (the test wraps vkAllocateMemory behind the instance-proc pointer it hands the layer), with
//     LEONIDA_SLOT133_IMPORT=0: no import and no shadow can be made, but a device-local DEFAULT buffer can.
//   - The same 2 MiB view of title memory bound 16 times, as vertex and index views, keeps no buffer (every buffer
//     made over title memory holds a device reference, and the device's count does not grow) and is tried once (the
//     buffers of its size created by the first bind are all there are).
//   - Once the retry delay has passed, the next bind tries the view again (as many buffers of its size created as by
//     the first bind) and the binds after it do not, and still nothing is kept.
// Needs a device-local memory type that is not host-visible (skipped otherwise). Synthetic data only. Set
// LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/title_va_alias.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

// The Vulkan calls the layer makes, seen through the function pointers it loads.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkAllocateMemory real_allocate = nullptr;
PFN_vkCreateBuffer real_create_buffer = nullptr;
VkPhysicalDeviceMemoryProperties memory{};
std::atomic<bool> refuse_host_visible{false};
std::atomic<VkDeviceSize> watched_size{0};
std::atomic<unsigned> watched_buffers{0};  // buffers of watched_size created

VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice device, const VkMemoryAllocateInfo* info,
                                        const VkAllocationCallbacks* allocator, VkDeviceMemory* out) {
    if (refuse_host_visible.load() && info->memoryTypeIndex < memory.memoryTypeCount &&
        (memory.memoryTypes[info->memoryTypeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    return real_allocate(device, info, allocator, out);
}

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice device, const VkBufferCreateInfo* info,
                                             const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
    if (info->size == watched_size.load()) ++watched_buffers;
    return real_create_buffer(device, info, allocator, buffer);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkAllocateMemory") == 0) {
        real_allocate = reinterpret_cast<PFN_vkAllocateMemory>(real_device_proc(device, name));
        return real_allocate ? reinterpret_cast<PFN_vkVoidFunction>(&allocate) : nullptr;
    }
    if (std::strcmp(name, "vkCreateBuffer") == 0) {
        real_create_buffer = reinterpret_cast<PFN_vkCreateBuffer>(real_device_proc(device, name));
        return real_create_buffer ? reinterpret_cast<PFN_vkVoidFunction>(&create_buffer) : nullptr;
    }
    return real_device_proc(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
        real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real_instance_proc(instance, name));
        return real_device_proc ? reinterpret_cast<PFN_vkVoidFunction>(&device_proc) : nullptr;
    }
    return real_instance_proc(instance, name);
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

// The device's reference count: every buffer the device keeps over title memory holds one.
ULONG references(ID3D12Device* device) {
    device->AddRef();
    return device->Release();
}

}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    real_instance_proc = vk.instance_proc;
    reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        vk.instance_proc(vk.instance, "vkGetPhysicalDeviceMemoryProperties"))(vk.physical, &memory);
    bool device_only = false;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        device_only |= (memory.memoryTypes[i].propertyFlags &
                        (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ==
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (!device_only) {
        std::fprintf(stderr, "SKIP: every device-local memory type is host-visible\n");
        return 77;
    }

    constexpr UINT title_bytes = 2u << 20;
    auto& aliases = leonida::title_va_aliases();
    void* pages = VirtualAlloc(nullptr, title_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    EXPECT(pages);
    const auto title_va = reinterpret_cast<std::uintptr_t>(pages);
    aliases.add_title_memory(title_va, title_bytes);
    int result = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = &instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        const std::uint64_t arena = rt::d3d12::device_arena_size(device.p);
        if (!arena) {
            std::fprintf(stderr, "SKIP: the device has no GPU address arena\n");
            result = 77;
        } else {
            // Fill the arena: a 1 MiB buffer that is not given an arena VA found no room there, so a 2 MiB shadow
            // will not find any either.
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            const D3D12_RESOURCE_DESC filler_desc = buffer_desc(1u << 20);
            std::vector<Ref<ID3D12Resource>> fillers(arena >> 20);
            bool full = false;
            for (auto& filler : fillers) {
                EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &filler_desc,
                                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                 IID_PPV_ARGS(&filler.p))));
                if ((full = filler->GetGPUVirtualAddress() >= arena)) break;
            }
            EXPECT(full);

            Ref<ID3D12CommandAllocator> allocator;
            Ref<ID3D12GraphicsCommandList> list;
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            const D3D12_VERTEX_BUFFER_VIEW vertices{title_va, title_bytes, 16};
            const D3D12_INDEX_BUFFER_VIEW indices{title_va, title_bytes, DXGI_FORMAT_R16_UINT};
            const auto bind = [&](unsigned i) {
                if (i & 1) list->IASetIndexBuffer(&indices);
                else list->IASetVertexBuffers(0, 1, &vertices);
            };

            // No import, no shadow: nothing over the title's pages is kept, and the view is tried once.
            const ULONG before = references(device.p);
            watched_size = title_bytes;
            refuse_host_visible = true;
            bind(0);
            const unsigned first = watched_buffers.load();
            for (unsigned i = 1; i < 16; ++i) bind(i);
            const ULONG refused = references(device.p);
            const unsigned tried = watched_buffers.load();
            std::printf("16 binds with no import or shadow: %lu buffers kept, %u buffers of the view's size created "
                        "(%u by the first bind)\n",
                        refused - before, tried, first);
            EXPECT(first >= 1);
            EXPECT(refused == before);
            EXPECT(tried == first);
            EXPECT(aliases.shader_va(title_va) == title_va);

            // After the retry delay the next bind tries the view again, once. Host-visible memory is still refused
            // here: a buffer the device keeps over title memory holds a reference to it, so the device would never be
            // destroyed, and its Vulkan objects would be reported as leaked under the validation layer.
            Sleep(1500);
            bind(0);
            const unsigned retried = watched_buffers.load();
            for (unsigned i = 1; i < 8; ++i) bind(i);
            const unsigned again = watched_buffers.load();
            refuse_host_visible = false;
            const ULONG after = references(device.p);
            std::printf("after the retry delay: %u buffers of the view's size created by the next bind, %u by 7 "
                        "further binds, %lu buffers kept\n",
                        retried - tried, again - retried, after - before);
            EXPECT(retried == tried + first);
            EXPECT(again == retried);
            EXPECT(after == before);
            EXPECT(SUCCEEDED(list->Close()));
        }
    }
    aliases.remove_title_memory(title_va);
    VirtualFree(pages, 0, MEM_RELEASE);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    if (!result) std::printf("ok\n");
    return result;
}
