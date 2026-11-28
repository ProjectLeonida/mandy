// The list values' write fallbacks (runtime/src/value_write_sink.h), on a Vulkan device and the queue and list proxies,
// with xmem linked in as in gta6rt. No game data.
// - A write whose address is a GPU address arena offset (a default-heap buffer) lands in the arena, read back by a GPU
//   copy; one in the arena's reserved start is dropped, never stored at that number in host memory.
// - A write to a page of xmem's that the title made PAGE_NOACCESS (its VirtualProtect, which xmem's fault handler and
//   kernel_writes' reopen leave as the title set it), where the host store faults, lands through xmem's own mapping
//   of the physical page.
// - A write to ordinary title memory lands as before, through neither fallback.
// LEONIDA_GPU_LABELS=1 throughout: the arena and xmem's pages are label targets too, and the lists here are tracked by
// the list values, so no write may be handed to the label path (one writer per list).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/d3d12_xs_list_values.h"
#include "../shims/shims.h"
#include "../shims/xmem.h"
#include "../src/title_page_watch.h"
#include "../src/title_texture_memory.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #x);                                                       \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

using namespace d3d12test;
namespace xs = rt::d3d12_xs;

namespace {
using Create = HRESULT(WINAPI *)(void *, const void *, ID3D12CommandAllocator *, ID3D12PipelineState *, const IID *,
                                 void **);
using Put = void(WINAPI *)(void *, std::uintptr_t, std::uint64_t, std::uint32_t, std::uint32_t);
using XMemVirtualAllocFn = void *(WINAPI *)(void *, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI *)(std::uint64_t, std::uint64_t *, std::uint64_t *);
using XMemFreePhysicalPagesFn = BOOL(WINAPI *)(std::uint64_t, const std::uint64_t *);
using XMemMapPhysicalPagesFn = void *(WINAPI *)(void *, std::uint64_t, const std::uint64_t *);
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;

struct Writes {
    std::uintptr_t address;
    std::uint64_t value;
    bool wide;
};

// One tracked list with the writes, submitted through the queue proxy; true once every write has landed or failed.
bool submit(ID3D12Device *device, ID3D12CommandQueue *queue, ID3D12CommandAllocator *allocator,
            std::initializer_list<Writes> writes) {
    std::uint64_t desc[4]{};
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(reinterpret_cast<Create>(slot(device, 73))(device, desc, allocator, nullptr,
                                                          &__uuidof(ID3D12GraphicsCommandList),
                                                          reinterpret_cast<void **>(&list.p))) ||
        !xs::is_list_proxy(list.p))
        return false;
    for (const Writes &w : writes)
        reinterpret_cast<Put>(slot(list.p, w.wide ? 72 : 71))(list.p, w.address, w.value, 0, 0);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList *submitted[1] = {list.p};
    queue->ExecuteCommandLists(1, submitted);
    Ref<ID3D12Fence> done;
    return SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))) &&
           SUCCEEDED(queue->Signal(done.p, 1)) && SUCCEEDED(done->SetEventOnCompletion(1, nullptr)) &&
           xs::list_values_idle(4000);
}

// The first `bytes` of a default-heap buffer, through a GPU copy into a readback buffer.
bool read_back(ID3D12Device *real, ID3D12Resource *buffer, void *out, UINT64 bytes) {
    D3D12_RESOURCE_DESC desc = buffer->GetDesc();
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(real->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))) ||
        FAILED(real->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(real->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(real->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                       IID_PPV_ARGS(&list.p))) ||
        FAILED(real->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    list->CopyBufferRegion(readback.p, 0, buffer, 0, bytes);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList *lists[1] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;
    void *mapped = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
    if (FAILED(readback->Map(0, &range, &mapped)) || !mapped) return false;
    std::memcpy(out, mapped, static_cast<std::size_t>(bytes));
    readback->Unmap(0, nullptr);
    return true;
}
}  // namespace

int main() {
    _putenv_s("LEONIDA_GPU_LABELS", "1");
    xs::set_list_values_enabled(true);
    vktest::Device gpu;
    auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    CHECK(status == vktest::DeviceStatus::ready);
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gpu.instance_proc;
    info.instance = gpu.instance;
    info.physical = gpu.physical;
    info.device = gpu.device;
    info.queue = gpu.queue;
    info.queue_family = gpu.family;
    info.buffer_device_address = true;
    ID3D12Device *inner = nullptr;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    CHECK(
        SUCCEEDED(xs::create_device_proxy(inner, __uuidof(ID3D12Device), reinterpret_cast<void **>(&device.p), true)));
    rt::d3d12::set_device_outer(inner, device.p);
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    CHECK(SUCCEEDED(inner->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));

    // 1. Ordinary title memory: the host store, as before; neither fallback.
    auto *plain = static_cast<volatile LONG64 *>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(plain);
    // The title's memory, as xmem registers it in gta6rt: an address below the arena's size is the title's there.
    leonida::title_va_aliases().add_title_memory(reinterpret_cast<std::uint64_t>(plain), 4096);
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p,
                 {{reinterpret_cast<std::uintptr_t>(plain), 11, false},
                  {reinterpret_cast<std::uintptr_t>(plain + 1), 0x500000006ull, true}}));
    auto stats = xs::list_value_stats();
    CHECK(stats.writes_landed == 2 && stats.writes_failed == 0);
    CHECK(stats.writes_arena == 0 && stats.writes_title_pages == 0 && stats.labels_handed == 0);
    CHECK(InterlockedCompareExchange64(plain, 0, 0) == 11 && InterlockedCompareExchange64(plain + 1, 0, 0) ==
                                                                 0x500000006ll);
    VirtualFree(const_cast<LONG64 *>(plain), 0, MEM_RELEASE);
    std::puts("a write to the title's own memory lands through the host store");

    // 2. An arena offset: a default-heap buffer's VA. The value lands in the arena, where the GPU reads it.
    D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = kPage64k;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> buffer;
    CHECK(SUCCEEDED(inner->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                   D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer.p))));
    const std::uint64_t va = buffer->GetGPUVirtualAddress();
    CHECK(va && va < (std::uint64_t{1} << 32));  // an arena offset, not a host address
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p, {{va, 41, false}, {va + 8, 0x300000004ull, true}}));
    stats = xs::list_value_stats();
    CHECK(stats.labels_handed == 0);
    CHECK(stats.writes_landed == 2 && stats.writes_arena == 2 && stats.writes_failed == 0);
    std::uint64_t words[2]{};
    CHECK(read_back(inner, buffer.p, words, sizeof(words)));
    CHECK(static_cast<std::uint32_t>(words[0]) == 41 && words[1] == 0x300000004ull);
    // The arena's reserved start (null region, zero page, quad indices) is the arena's, and shaders read it: a write
    // there is dropped, and never stored at that number in host memory.
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p, {{0x100, 99, false}}));
    stats = xs::list_value_stats();
    CHECK(stats.writes_landed == 0 && stats.writes_failed == 1 && stats.writes_arena == 0);
    std::puts("a write to an arena offset lands in the arena; one in its reserved start is dropped");

    // 3. A page of xmem's that the title made PAGE_NOACCESS through its VirtualProtect: the host store faults there
    // (xmem's fault handler and reopen_unowned leave a page the title protected), and the value lands through xmem's
    // own mapping of the page.
    auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    auto allocate =
        reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    auto free_pages = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    using ProtectFn = BOOL(WINAPI *)(void *, SIZE_T, DWORD, PDWORD);
    auto title_protect = reinterpret_cast<ProtectFn>(rt::shims::find("kernel32.dll", "VirtualProtect"));
    CHECK(reserve && allocate && map && free_pages && title_protect);
    auto *window =
        static_cast<std::uint8_t *>(reserve(nullptr, kPage64k, MEM_RESERVE | kXmem64kPages, 0, PAGE_READWRITE));
    CHECK(window);
    std::uint64_t page = 0, count = 1;
    CHECK(allocate(0x280, &count, &page) && count == 1);
    CHECK(map(window, 1, &page) == window);
    window[0] = 1;  // committed, as a page the title used
    DWORD old = 0;
    CHECK(title_protect(window, kPage64k, PAGE_NOACCESS, &old));
    // What the list values did before the fallback: the host store into the title's view, which refuses it.
    CHECK(!leonida::kernel_writes::store(window + 0x80, 1, 4));
    xs::reset_list_value_stats();
    CHECK(submit(device.p, queue.p, allocator.p,
                 {{reinterpret_cast<std::uintptr_t>(window) + 0x40, 0x5151, false},
                  {reinterpret_cast<std::uintptr_t>(window) + 0x48, 0x700000008ull, true}}));
    stats = xs::list_value_stats();
    CHECK(stats.labels_handed == 0);
    CHECK(stats.writes_landed == 2 && stats.writes_title_pages == 2 && stats.writes_failed == 0);
    const std::uintptr_t host = rt::xmem::physical_page_scratch(page);
    CHECK(host);
    std::uint32_t low = 0;
    std::uint64_t wide = 0;
    std::memcpy(&low, reinterpret_cast<const void *>(host + 0x40), sizeof(low));
    std::memcpy(&wide, reinterpret_cast<const void *>(host + 0x48), sizeof(wide));
    CHECK(low == 0x5151 && wide == 0x700000008ull);
    map(window, 1, nullptr);
    CHECK(free_pages(1, &page));
    std::puts("a write to a PAGE_NOACCESS page of xmem's lands through xmem's own mapping");

    // 4. Round 10 WS3: retroactive GPU-to-GPU wait relay (Wait-before-Signal across two queues -> relayed onto
    //    writer's written fence with zero unmet gate releases, and cycle preservation when after_gate != 0).
    {
        // Console list slot 74: wait64(address, comparison, reference, flags).
        using WaitOp = void(WINAPI *)(void *, std::uintptr_t, std::uint32_t, std::uint64_t, std::uint32_t);
        D3D12_COMMAND_QUEUE_DESC compute_desc{};
        compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        Ref<ID3D12CommandQueue> compute_q;
        Ref<ID3D12CommandAllocator> compute_alloc;
        CHECK(SUCCEEDED(device->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&compute_q.p))));
        CHECK(SUCCEEDED(inner->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&compute_alloc.p))));

        auto *slot_mem =
            static_cast<volatile LONG64 *>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        CHECK(slot_mem);
        slot_mem[0] = 0;
        slot_mem[1] = 0;
        leonida::title_va_aliases().add_title_memory(reinterpret_cast<std::uint64_t>(slot_mem), 4096);

        // 4a. Direct submits Wait(slot_mem[0] >= 1) BEFORE Compute submits Write(slot_mem[0] = 1).
        xs::reset_list_value_stats();
        std::uint64_t desc[4]{};
        std::uint64_t compute_list_desc[4]{static_cast<std::uint64_t>(D3D12_COMMAND_LIST_TYPE_COMPUTE)};
        Ref<ID3D12GraphicsCommandList> wait_list, write_list;
        CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(device.p, desc, allocator.p, nullptr,
                                                                     &__uuidof(ID3D12GraphicsCommandList),
                                                                     reinterpret_cast<void **>(&wait_list.p))));
        reinterpret_cast<WaitOp>(slot(wait_list.p, 74))(
            wait_list.p, reinterpret_cast<std::uintptr_t>(slot_mem), 7 /*>=*/, 1, 0);
        CHECK(SUCCEEDED(wait_list->Close()));
        ID3D12CommandList *wl[1] = {wait_list.p};
        queue->ExecuteCommandLists(1, wl);

        Ref<ID3D12Fence> direct_done;
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&direct_done.p))));
        CHECK(SUCCEEDED(queue->Signal(direct_done.p, 1)));

        // Now Compute submits the matching write: Pump::add_writes() retroactively relays Direct's gate!
        CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(device.p, compute_list_desc, compute_alloc.p,
                                                                     nullptr, &__uuidof(ID3D12GraphicsCommandList),
                                                                     reinterpret_cast<void **>(&write_list.p))));
        reinterpret_cast<Put>(slot(write_list.p, 72))(
            write_list.p, reinterpret_cast<std::uintptr_t>(slot_mem), 1, 0, 0);
        CHECK(SUCCEEDED(write_list->Close()));
        ID3D12CommandList *wrl[1] = {write_list.p};
        compute_q->ExecuteCommandLists(1, wrl);

        CHECK(SUCCEEDED(direct_done->SetEventOnCompletion(1, nullptr)));
        CHECK(xs::list_values_idle(4000));
        stats = xs::list_value_stats();
        CHECK(stats.waits_relayed >= 1);
        CHECK(stats.gates_released == 0);
        CHECK(InterlockedCompareExchange64(slot_mem, 0, 0) == 1);

        // 4b. Cycle preservation (after_gate != 0): Direct waits on slot_mem[0] >= 10 then writes slot_mem[1] = 10;
        //     Compute waits on slot_mem[1] >= 10 then writes slot_mem[0] = 10. Neither write group has after_gate == 0,
        //     so neither is retroactively relayed into a GPU cycle and list_value_progress breaks the cycle cleanly.
        xs::reset_list_value_stats();
        Ref<ID3D12GraphicsCommandList> cyc_a, cyc_b;
        CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(device.p, desc, allocator.p, nullptr,
                                                                     &__uuidof(ID3D12GraphicsCommandList),
                                                                     reinterpret_cast<void **>(&cyc_a.p))));
        reinterpret_cast<WaitOp>(slot(cyc_a.p, 74))(
            cyc_a.p, reinterpret_cast<std::uintptr_t>(slot_mem), 7 /*>=*/, 10, 0);
        reinterpret_cast<Put>(slot(cyc_a.p, 72))(
            cyc_a.p, reinterpret_cast<std::uintptr_t>(slot_mem + 1), 10, 0, 0);
        CHECK(SUCCEEDED(cyc_a->Close()));
        CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(device.p, compute_list_desc, compute_alloc.p,
                                                                     nullptr, &__uuidof(ID3D12GraphicsCommandList),
                                                                     reinterpret_cast<void **>(&cyc_b.p))));
        reinterpret_cast<WaitOp>(slot(cyc_b.p, 74))(
            cyc_b.p, reinterpret_cast<std::uintptr_t>(slot_mem + 1), 7 /*>=*/, 10, 0);
        reinterpret_cast<Put>(slot(cyc_b.p, 72))(
            cyc_b.p, reinterpret_cast<std::uintptr_t>(slot_mem), 10, 0, 0);
        CHECK(SUCCEEDED(cyc_b->Close()));

        ID3D12CommandList *cal[1] = {cyc_a.p};
        ID3D12CommandList *cbl[1] = {cyc_b.p};
        queue->ExecuteCommandLists(1, cal);
        compute_q->ExecuteCommandLists(1, cbl);
        Ref<ID3D12Fence> cyc_done_a, cyc_done_b;
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cyc_done_a.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&cyc_done_b.p))));
        CHECK(SUCCEEDED(queue->Signal(cyc_done_a.p, 1)));
        CHECK(SUCCEEDED(compute_q->Signal(cyc_done_b.p, 1)));
        CHECK(SUCCEEDED(cyc_done_a->SetEventOnCompletion(1, nullptr)));
        CHECK(SUCCEEDED(cyc_done_b->SetEventOnCompletion(1, nullptr)));
        CHECK(xs::list_values_idle(4000));
        stats = xs::list_value_stats();
        CHECK(stats.waits_relayed == 0);
        CHECK(stats.released_cycle >= 1);

        VirtualFree(const_cast<LONG64 *>(slot_mem), 0, MEM_RELEASE);
        std::puts("retroactive Wait-before-Signal relay and after_gate!=0 cycle preservation ok");
    }

    // 4c. Round 10b WS-A (A1/A2/A3/A4): bare write groups force CPU visibility signal (no retro-relay, no
    //     signal_gpu_relay); exact-value/width/timestamp checks downgrade non-matching landed writes to cpu_met;
    //     stale wait_met re-check prevents premature retro-relay; relay_wait returning false preserves the gate.
    {
        namespace lv = rt::d3d12_xs::list_values;
        using WaitOp = void(WINAPI *)(void *, std::uintptr_t, std::uint32_t, std::uint64_t, std::uint32_t);

        // Unit checks for Pump::match_landed() (A1 bare + A2 exact address/width/value/timestamp).
        {
            const lv::Wait w64{0x1000, 10, 7 /*>=*/, true};
            const lv::Wait w32{0x1000, 10, 7 /*>=*/, false};
            const lv::Pump::LandedRange exact_ok[] = {{0x1000, 10, 8, true, false}};
            const lv::Pump::LandedRange bare_ok[] = {{0x1000, 10, 8, true, true}};
            const lv::Pump::LandedRange low_val[] = {{0x1000, 5, 8, true, false}};
            const lv::Pump::LandedRange wrong_width[] = {{0x1000, 10, 4, true, false}};
            const lv::Pump::LandedRange wrong_addr[] = {{0x1004, 10, 4, true, false}};
            const lv::Pump::LandedRange timestamp_wr[] = {{0x1000, 10, 8, false, false}};
            const lv::Pump::LandedRange overwritten_low[] = {{0x1000, 10, 8, true, false}, {0x1000, 5, 8, true, false}};
            const lv::Pump::LandedRange bare_then_exec[] = {{0x1000, 10, 8, true, true}, {0x1000, 11, 8, true, false}};
            const lv::Pump::LandedRange unrelated[] = {{0x2000, 10, 8, true, false}};

            CHECK(lv::Pump::match_landed(w64, exact_ok) == lv::Pump::LandedMatch::gpu_met);
            CHECK(lv::Pump::match_landed(w64, bare_ok) == lv::Pump::LandedMatch::bare);
            CHECK(lv::Pump::match_landed(w64, low_val) == lv::Pump::LandedMatch::val_mismatch);
            CHECK(lv::Pump::match_landed(w64, wrong_width) == lv::Pump::LandedMatch::val_mismatch);
            CHECK(lv::Pump::match_landed(w32, exact_ok) == lv::Pump::LandedMatch::val_mismatch);
            CHECK(lv::Pump::match_landed(w64, wrong_addr) == lv::Pump::LandedMatch::val_mismatch);
            CHECK(lv::Pump::match_landed(w64, timestamp_wr) == lv::Pump::LandedMatch::val_mismatch);
            CHECK(lv::Pump::match_landed(w64, overwritten_low) == lv::Pump::LandedMatch::val_mismatch);
            CHECK(lv::Pump::match_landed(w64, bare_then_exec) == lv::Pump::LandedMatch::bare);
            CHECK(lv::Pump::match_landed(w64, unrelated) == lv::Pump::LandedMatch::none);
        }

        D3D12_COMMAND_QUEUE_DESC compute_desc{};
        compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        Ref<ID3D12CommandQueue> compute_q;
        Ref<ID3D12CommandAllocator> compute_alloc;
        CHECK(SUCCEEDED(device->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&compute_q.p))));
        CHECK(SUCCEEDED(inner->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&compute_alloc.p))));

        auto *mem =
            static_cast<volatile LONG64 *>(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        CHECK(mem);
        leonida::title_va_aliases().add_title_memory(reinterpret_cast<std::uint64_t>(mem), 4096);

        static std::atomic<std::uint64_t> gpu_relay_calls{0};
        struct HookGuard {
            ~HookGuard() {
                lv::signal_gpu_relay_hook() = nullptr;
                lv::relay_wait_hook() = nullptr;
            }
        } hook_guard;
        lv::signal_gpu_relay_hook() = [](ID3D12Fence *f, std::uint64_t v) -> HRESULT {
            gpu_relay_calls.fetch_add(1, std::memory_order_relaxed);
            return rt::d3d12::fence_signal_gpu_relay(f, v);
        };

        std::uint64_t direct_desc[4]{};
        std::uint64_t compute_list_desc[4]{static_cast<std::uint64_t>(D3D12_COMMAND_LIST_TYPE_COMPUTE)};

        // A1 live test: bare pre-write group (`[write64(mem[0]=1); wait64(mem[1]>=1)]` as first list of execute)
        // releases held compute consumer -> NOT retro-relayed, NOT signaled via signal_gpu_relay, increments bare
        // downgrade counter.
        mem[0] = 0;
        mem[1] = 0;
        gpu_relay_calls.store(0, std::memory_order_relaxed);
        xs::reset_list_value_stats();
        {
            Ref<ID3D12GraphicsCommandList> wait_list, bare_list;
            CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(
                device.p, compute_list_desc, compute_alloc.p, nullptr, &__uuidof(ID3D12GraphicsCommandList),
                reinterpret_cast<void **>(&wait_list.p))));
            reinterpret_cast<WaitOp>(slot(wait_list.p, 74))(wait_list.p, reinterpret_cast<std::uintptr_t>(mem),
                                                            7 /*>=*/, 1, 0);
            CHECK(SUCCEEDED(wait_list->Close()));
            ID3D12CommandList *wl[1] = {wait_list.p};
            compute_q->ExecuteCommandLists(1, wl);
            Ref<ID3D12Fence> compute_done;
            CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&compute_done.p))));
            CHECK(SUCCEEDED(compute_q->Signal(compute_done.p, 1)));

            CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(
                device.p, direct_desc, allocator.p, nullptr, &__uuidof(ID3D12GraphicsCommandList),
                reinterpret_cast<void **>(&bare_list.p))));
            // Pre-write before an unmet wait on the first list -> ValueSubmission::flush() runs with group_.empty()!
            reinterpret_cast<Put>(slot(bare_list.p, 72))(bare_list.p, reinterpret_cast<std::uintptr_t>(mem), 1, 0, 0);
            reinterpret_cast<WaitOp>(slot(bare_list.p, 74))(bare_list.p, reinterpret_cast<std::uintptr_t>(mem + 1),
                                                            7 /*>=*/, 1, 0);
            CHECK(SUCCEEDED(bare_list->Close()));
            ID3D12CommandList *bl[1] = {bare_list.p};
            queue->ExecuteCommandLists(1, bl);
            Ref<ID3D12Fence> direct_done;
            CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&direct_done.p))));
            CHECK(SUCCEEDED(queue->Signal(direct_done.p, 1)));

            CHECK(SUCCEEDED(compute_done->SetEventOnCompletion(1, nullptr)));
            CHECK(gpu_relay_calls.load(std::memory_order_relaxed) == 0);
            CHECK(lv::counters().gpu_relay_downgraded_bare.load(std::memory_order_relaxed) >= 1);
            CHECK(xs::list_value_stats().waits_relayed == 0);

            InterlockedExchange64(mem + 1, 1);
            CHECK(SUCCEEDED(direct_done->SetEventOnCompletion(1, nullptr)));
            CHECK(xs::list_values_idle(4000));
        }

        // A2 live test: non-bare 32-bit write meeting a 64-bit wait AND non-bare GPU timestamp write meeting a 64-bit
        // wait both downgrade to cpu_met (no signal_gpu_relay, increments gpu_relay_downgraded_val).
        for (int kind = 0; kind < 2; ++kind) {
            mem[0] = 0;
            gpu_relay_calls.store(0, std::memory_order_relaxed);
            xs::reset_list_value_stats();
            Ref<ID3D12GraphicsCommandList> wait_list, wr_list;
            CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(
                device.p, direct_desc, allocator.p, nullptr, &__uuidof(ID3D12GraphicsCommandList),
                reinterpret_cast<void **>(&wait_list.p))));
            reinterpret_cast<WaitOp>(slot(wait_list.p, 74))(wait_list.p, reinterpret_cast<std::uintptr_t>(mem),
                                                            7 /*>=*/, 1, 0);
            CHECK(SUCCEEDED(wait_list->Close()));
            ID3D12CommandList *wl[1] = {wait_list.p};
            queue->ExecuteCommandLists(1, wl);
            Ref<ID3D12Fence> done;
            CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))));
            CHECK(SUCCEEDED(queue->Signal(done.p, 1)));

            CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(
                device.p, compute_list_desc, compute_alloc.p, nullptr, &__uuidof(ID3D12GraphicsCommandList),
                reinterpret_cast<void **>(&wr_list.p))));
            if (kind == 0) {
                // 32-bit write (slot 71) overlapping 64-bit wait -> width mismatch downgrade.
                reinterpret_cast<Put>(slot(wr_list.p, 71))(wr_list.p, reinterpret_cast<std::uintptr_t>(mem), 1, 0, 0);
            } else {
                // GPU timestamp write (slot 72, flags=1) -> exact_value == false downgrade.
                reinterpret_cast<Put>(slot(wr_list.p, 72))(wr_list.p, reinterpret_cast<std::uintptr_t>(mem), 0, 0, 1);
            }
            CHECK(SUCCEEDED(wr_list->Close()));
            ID3D12CommandList *wrl[1] = {wr_list.p};
            compute_q->ExecuteCommandLists(1, wrl);

            CHECK(SUCCEEDED(done->SetEventOnCompletion(1, nullptr)));
            CHECK(xs::list_values_idle(4000));
            CHECK(gpu_relay_calls.load(std::memory_order_relaxed) == 0);
            CHECK(lv::counters().gpu_relay_downgraded_val.load(std::memory_order_relaxed) >= 1);
        }

        // A4 live test: when relay_wait returns false (nothing substituted), apply_retroactive_relays preserves the
        // gate so Pump still opens it when the write lands.
        mem[0] = 0;
        gpu_relay_calls.store(0, std::memory_order_relaxed);
        xs::reset_list_value_stats();
        lv::relay_wait_hook() = [](ID3D12Fence *, std::uint64_t, std::span<const lv::RelayTarget>) -> bool {
            return false;
        };
        {
            Ref<ID3D12GraphicsCommandList> wait_list, wr_list;
            CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(
                device.p, direct_desc, allocator.p, nullptr, &__uuidof(ID3D12GraphicsCommandList),
                reinterpret_cast<void **>(&wait_list.p))));
            reinterpret_cast<WaitOp>(slot(wait_list.p, 74))(wait_list.p, reinterpret_cast<std::uintptr_t>(mem),
                                                            7 /*>=*/, 1, 0);
            CHECK(SUCCEEDED(wait_list->Close()));
            ID3D12CommandList *wl[1] = {wait_list.p};
            queue->ExecuteCommandLists(1, wl);
            Ref<ID3D12Fence> done;
            CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p))));
            CHECK(SUCCEEDED(queue->Signal(done.p, 1)));

            CHECK(SUCCEEDED(reinterpret_cast<Create>(slot(device.p, 73))(
                device.p, compute_list_desc, compute_alloc.p, nullptr, &__uuidof(ID3D12GraphicsCommandList),
                reinterpret_cast<void **>(&wr_list.p))));
            reinterpret_cast<Put>(slot(wr_list.p, 72))(wr_list.p, reinterpret_cast<std::uintptr_t>(mem), 1, 0, 0);
            CHECK(SUCCEEDED(wr_list->Close()));
            ID3D12CommandList *wrl[1] = {wr_list.p};
            compute_q->ExecuteCommandLists(1, wrl);

            CHECK(SUCCEEDED(done->SetEventOnCompletion(1, nullptr)));
            CHECK(xs::list_values_idle(4000));
            stats = xs::list_value_stats();
            CHECK(stats.waits_relayed == 0);
            CHECK(stats.gates_met >= 1);
            CHECK(stats.gates_released == 0);
            CHECK(gpu_relay_calls.load(std::memory_order_relaxed) >= 1);  // non-bare exact match still uses vis-0!
        }

        VirtualFree(const_cast<LONG64 *>(mem), 0, MEM_RELEASE);
        std::puts("WS-A bare downgrade, exact-value/width/timestamp downgrade, and relay_wait false preservation ok");
    }

    // 5. Round 10 WS3: TextureVqCache negative caching & hoisted host_stop over fragmented VMAs.
    {
        constexpr SIZE_T kSpan = 4 * kPage64k;
        auto *region = static_cast<std::uint8_t *>(VirtualAlloc(nullptr, kSpan, MEM_RESERVE, PAGE_READWRITE));
        CHECK(region);
        const auto base = reinterpret_cast<std::uintptr_t>(region);
        leonida::TextureVqCache cache;

        // Reserved (uncommitted) pool caches negative readability across probes in the same region.
        CHECK(leonida::texture_memory_readable_bytes(base, kPage64k, nullptr, &cache) == 0);
        CHECK(!cache.readable && cache.lo <= base && cache.hi >= base + kSpan);
        CHECK(leonida::texture_memory_readable_bytes(base + kPage64k, kPage64k, nullptr, &cache) == 0);
        CHECK(!cache.readable);

        // Commit all 4 tiles, notify title_memory_map, and fragment alternating RO/RW 64 KiB VMAs.
        CHECK(VirtualAlloc(region, kSpan, MEM_COMMIT, PAGE_READWRITE) == region);
        leonida::title_memory_map::changed(base, base + kSpan);
        DWORD old_prot = 0;
        CHECK(VirtualProtect(region, kPage64k, PAGE_READONLY, &old_prot));
        CHECK(VirtualProtect(region + 2 * kPage64k, kPage64k, PAGE_READONLY, &old_prot));
        CHECK(leonida::texture_memory_readable_bytes(base, kSpan, nullptr, &cache) == kSpan);
        CHECK(cache.readable);

        // PAGE_NOACCESS in the middle stops readability and populates negative cache entry.
        CHECK(VirtualProtect(region + 2 * kPage64k, kPage64k, PAGE_NOACCESS, &old_prot));
        leonida::title_memory_map::changed(base + 2 * kPage64k, base + 3 * kPage64k);
        CHECK(leonida::texture_memory_readable_bytes(base, kSpan, nullptr, &cache) == 2 * kPage64k);
        CHECK(!cache.readable && cache.lo == base + 2 * kPage64k && cache.hi == base + 3 * kPage64k);

        VirtualFree(region, 0, MEM_RELEASE);
        std::puts("TextureVqCache negative caching and fragmented VMA readability ok");
    }

    CHECK(gpu.validation_errors() == 0);
    return 0;
}
