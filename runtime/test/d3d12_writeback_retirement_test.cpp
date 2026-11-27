// A held queue keeps the global retirement prefix behind >64 completed submissions on another queue.
// Title-shadow writeback must wait for that prefix, without blocking the queue that can release it.
// Synthetic buffers and transfer commands only: no shaders or game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/title_va_alias.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); return 1; } } while (0)

namespace {
constexpr std::uint32_t kBatches = 130;
constexpr std::uint32_t kStride = 4096;  // separate tracker pages; range-coalescing behavior is outside this test
constexpr std::uint32_t kBytes = ((kBatches + 15) / 16) * 65536;
template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

D3D12_RESOURCE_DESC buffer_desc() {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kBytes;
    desc.Height = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

// Declared after the D3D12 objects: even a failed assertion releases the native queue before their teardown waits.
struct Gate {
    vktest::Device& vk;
    VkSemaphore semaphore = VK_NULL_HANDLE;
    bool released = false;
    template<class T> T proc(const char* name) { return reinterpret_cast<T>(vk.device_proc(vk.device, name)); }
    bool release() {
        if (!semaphore || released) return true;
        VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        signal.semaphore = semaphore;
        signal.value = 1;
        released = proc<PFN_vkSignalSemaphore>("vkSignalSemaphore")(vk.device, &signal) == VK_SUCCESS;
        return released;
    }
    ~Gate() {
        if (semaphore) {
            release();
            proc<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(vk.device);
            proc<PFN_vkDestroySemaphore>("vkDestroySemaphore")(vk.device, semaphore, nullptr);
        }
    }
};

int run(vktest::Device& vk) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.command_queues = vk.queues;
    info.buffer_device_address = true;
    // No host imports: the title buffer must have a shadow, regardless of driver import support.
    info.host_pointer_alignment = 0;
    Ref<ID3D12Device> device;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
    Ref<ID3D12CommandQueue> held, running;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&held.p))));
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&running.p))));
    Ref<ID3D12CommandAllocator> held_allocator, allocator;
    Ref<ID3D12GraphicsCommandList> held_list, list;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&held_allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, held_allocator.p, nullptr,
                                             IID_PPV_ARGS(&held_list.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(event);
    struct Event { HANDLE value; ~Event() { CloseHandle(value); } } close_event{event};
    struct Pages {
        void* value = VirtualAlloc(nullptr, kBytes, MEM_RESERVE | MEM_COMMIT | MEM_TOP_DOWN, PAGE_READWRITE);
        Pages() {
            if (value) leonida::title_va_aliases().add_title_memory(reinterpret_cast<std::uint64_t>(value), kBytes);
        }
        ~Pages() {
            if (value) {
                leonida::title_va_aliases().remove_title_memory(reinterpret_cast<std::uint64_t>(value));
                VirtualFree(value, 0, MEM_RELEASE);
            }
        }
    } pages;
    CHECK(pages.value);
    auto* words = static_cast<std::uint32_t*>(pages.value);
    const auto va = reinterpret_cast<std::uint64_t>(pages.value);
    Ref<ID3D12Resource> title, upload, other;
    const auto desc = buffer_desc();
    CHECK(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, pages.value, desc, 0xFFFFFFFFu,
                                                           IID_PPV_ARGS(&title.p))));
    CHECK(rt::d3d12::title_shadow_buffer(title.p));
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD}, readback_heap{D3D12_HEAP_TYPE_READBACK};
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                   IID_PPV_ARGS(&upload.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&other.p))));
    constexpr std::uint32_t expected = 0x12345678;
    void* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, &mapped)));
    for (std::uint32_t i = 0; i < kBatches; ++i) {
        const std::uint32_t word = expected + i;
        std::memcpy(static_cast<std::uint8_t*>(mapped) + kStride * i, &word, sizeof(word));
    }
    upload->Unmap(0, nullptr);

    // Block queue 0 at a real timeline semaphore, outside D3D12's host-held console waits.
    Gate gate{vk};
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
    CHECK(gate.proc<PFN_vkCreateSemaphore>("vkCreateSemaphore")(vk.device, &create, nullptr, &gate.semaphore) == VK_SUCCESS);
    const std::uint64_t value = 1;
    VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues = &value;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo wait{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
    wait.waitSemaphoreCount = 1;
    wait.pWaitSemaphores = &gate.semaphore;
    wait.pWaitDstStageMask = &stage;
    CHECK(gate.proc<PFN_vkQueueSubmit>("vkQueueSubmit")(vk.queues[0], 1, &wait, VK_NULL_HANDLE) == VK_SUCCESS);
    held_list->CopyBufferRegion(other.p, 0, upload.p, 0, sizeof(expected));
    CHECK(held_list->Close() == S_OK);
    ID3D12CommandList* held_lists[] = {held_list.p};
    held->ExecuteCommandLists(1, held_lists);
    CHECK(rt::d3d12::last_execute_result(held.p) == leonida::ReplayResult::success);

    CHECK(list->Close() == S_OK);
    for (std::uint64_t pass = 1; pass <= kBatches; ++pass) {
        const auto offset = kStride * (pass - 1);  // distinct pages: dropping the oldest ranges must also fail
        CHECK(allocator->Reset() == S_OK);
        CHECK(list->Reset(allocator.p, nullptr) == S_OK);
        list->CopyBufferRegion(title.p, offset, upload.p, offset, sizeof(expected));
        CHECK(list->Close() == S_OK);
        // The notification a bound title-memory V# would make; the transfer supplies its synthetic GPU write.
        leonida::title_va_aliases().note_reached(va + offset, sizeof(expected));
        ID3D12CommandList* lists[] = {list.p};
        running->ExecuteCommandLists(1, lists);
        CHECK(rt::d3d12::last_execute_result(running.p) == leonida::ReplayResult::success);
        CHECK(SUCCEEDED(running->Signal(fence.p, pass)));
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(pass, event)));
        CHECK(WaitForSingleObject(event, 5000) == WAIT_OBJECT_0);
        for (std::uint32_t i = 0; i < kBatches; ++i)
            if (words[(kStride / 4) * i] != 0) {
                std::fprintf(stderr, "premature title writeback on refresh %llu (first queue still held)\n",
                             static_cast<unsigned long long>(pass));
                return 1;
            }
    }
    CHECK(gate.release());
    CHECK(gate.proc<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(vk.device) == VK_SUCCESS);
    // No new range is noted: a refresh must drain the retained backlog even without a current write.
    CHECK(allocator->Reset() == S_OK);
    CHECK(list->Reset(allocator.p, nullptr) == S_OK);
    CHECK(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    running->ExecuteCommandLists(1, lists);
    CHECK(rt::d3d12::last_execute_result(running.p) == leonida::ReplayResult::success);
    std::uint32_t matched = 0;
    for (std::uint32_t i = 0; i < kBatches; ++i) matched += words[(kStride / 4) * i] == expected + i;
    std::printf("retired writeback results: %u/%u matched\n", matched, kBatches);
    CHECK(matched == kBatches);
    std::puts("130 pending writeback batches stayed fenced and drained after release");
    return 0;
}
}  // namespace

int main() {
    vktest::Device vk;
    vk.queue_count_wanted = 2;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (vk.queues.size() < 2) {
        std::puts("SKIP: needs two graphics-family Vulkan queues");
        return 77;
    }
    const int result = run(vk);
    return result || vk.validation_errors() ? 1 : 0;
}
