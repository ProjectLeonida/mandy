// Placed buffers in ID3D12Device3::OpenExistingHeapFromAddress existing heaps (RT-EXISTING-HEAP-PLACED):
//   1. CreatePlacedResource / CreatePlacedResource1 / CreatePlacedResource2 validation (including the
//      SHARED_CROSS_ADAPTER -> ALLOW_CROSS_ADAPTER requirement and castable-list/layout checks) compared
//      case-by-case with the live host D3D12 runtime and WARP; plus E_NOTIMPL for placed textures.
//   2. Non-zeroing preservation of pre-filled VirtualAlloc pages, Map() returning base + offset, Map() error checks,
//      GetDesc().Flags preservation, GetHeapProperties parity (CUSTOM / WRITE_BACK and WRITE_COMBINE / L0, node masks
//      1, flags 0x421), and GetGPUVirtualAddress offset stride.
//   3. Bidirectional zero-copy CPU<->GPU coherence (CPU write -> GPU readback, GPU upload -> direct host RAM read,
//      placed-to-placed copy inside the same heap) with buffer_device_address both disabled and enabled (GPU arena).
//   4. Overlapping placed buffer aliasing, releasing ID3D12Heap before GPU work on placed buffers, and 64-iteration
//      alloc/open/place/copy/release/VirtualFree stress (specifically catching Intel WDDM asynchronous unpin bugs).
//   5. Forwarding through the Xbox-ABI device proxy (rt::d3d12_xs::create_device_proxy).
// Skips (77) when no usable Vulkan GPU or VK_EXT_external_memory_host is available. No shaders or game data.
#include "host_adapter.h"
#include "host_wine.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <dxgi1_4.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define EXPECT(condition)                                                                      \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition);               \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

#define EXPECT_HR(expr, want)                                                                  \
    do {                                                                                       \
        const HRESULT got_ = (expr);                                                           \
        if (got_ != (want)) {                                                                  \
            std::fprintf(stderr, "%s line %d: %s = 0x%08lX, expected 0x%08lX\n", who, __LINE__, \
                         #expr, static_cast<unsigned long>(got_), static_cast<unsigned long>(want)); \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

struct VirtualRange {
    void* ptr = nullptr;
    SIZE_T size = 0;

    bool alloc(SIZE_T bytes, DWORD protect = PAGE_READWRITE) {
        reset();
        ptr = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, protect);
        size = ptr ? bytes : 0;
        return ptr != nullptr;
    }
    ~VirtualRange() { reset(); }
    void reset() {
        if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);
        ptr = nullptr;
        size = 0;
    }
    std::uint8_t* bytes() const { return static_cast<std::uint8_t*>(ptr); }
};

constexpr SIZE_T k64KiB = 0x10000;
constexpr SIZE_T k256KiB = 4 * k64KiB;

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC existing_buffer_desc(UINT64 bytes, D3D12_RESOURCE_FLAGS extra = D3D12_RESOURCE_FLAG_NONE) {
    return buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | extra);
}

D3D12_RESOURCE_DESC1 buffer_desc1(UINT64 bytes, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC1 d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC1 existing_buffer_desc1(UINT64 bytes, D3D12_RESOURCE_FLAGS extra = D3D12_RESOURCE_FLAG_NONE) {
    return buffer_desc1(bytes, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | extra);
}

D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

void fill_pattern(void* dst, std::size_t size, std::uint8_t seed) {
    auto* out = static_cast<std::uint8_t*>(dst);
    for (std::size_t i = 0; i < size; ++i) {
        std::uint8_t v = static_cast<std::uint8_t>((i * 131u + seed * 37u + 19u) & 0xFFu);
        if (v == 0) v = 0xA5u;  // strictly non-zero so uninitialized zero-fill is always caught
        out[i] = v;
    }
}

bool pattern_matches(const void* ptr, std::size_t size, std::uint8_t seed, std::size_t* mismatch_at = nullptr) {
    const auto* in = static_cast<const std::uint8_t*>(ptr);
    for (std::size_t i = 0; i < size; ++i) {
        std::uint8_t want = static_cast<std::uint8_t>((i * 131u + seed * 37u + 19u) & 0xFFu);
        if (want == 0) want = 0xA5u;
        if (in[i] != want) {
            if (mismatch_at) *mismatch_at = i;
            return false;
        }
    }
    return true;
}

// Reusable direct-queue copy helper for verifying CPU<->GPU coherence without repeating boilerplate.
struct CopyRunner {
    ID3D12Device* device = nullptr;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    HANDLE event = nullptr;
    UINT64 fence_value = 0;

    ~CopyRunner() {
        if (event) CloseHandle(event);
    }

    int init(ID3D12Device* dev, const char* who) {
        device = dev;
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        EXPECT_HR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p)), S_OK);
        EXPECT_HR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p)), S_OK);
        EXPECT_HR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                            IID_PPV_ARGS(&list.p)),
                  S_OK);
        EXPECT_HR(list->Close(), S_OK);
        EXPECT_HR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)), S_OK);
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        EXPECT(event != nullptr);
        return 0;
    }

    int copy_region(ID3D12Resource* dst, UINT64 dst_off, ID3D12Resource* src, UINT64 src_off, UINT64 bytes,
                    const char* who) {
        EXPECT_HR(allocator->Reset(), S_OK);
        EXPECT_HR(list->Reset(allocator.p, nullptr), S_OK);
        list->CopyBufferRegion(dst, dst_off, src, src_off, bytes);
        EXPECT_HR(list->Close(), S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const UINT64 target = ++fence_value;
        EXPECT_HR(queue->Signal(fence.p, target), S_OK);
        EXPECT_HR(fence->SetEventOnCompletion(target, event), S_OK);
        EXPECT(WaitForSingleObject(event, 10000) == WAIT_OBJECT_0);
        return 0;
    }
};

struct ValidationCase {
    const char* name;
    HRESULT hr = S_OK;
    bool has_heap_props = false;
    D3D12_HEAP_PROPERTIES heap_props{};
    D3D12_HEAP_FLAGS heap_flags = D3D12_HEAP_FLAG_NONE;
};

struct ValidationSuite {
    bool has_device8 = false;
    bool has_device10 = false;
    std::vector<ValidationCase> cases;
};

ValidationSuite run_validation_suite(ID3D12Device* device, const char* who) {
    ValidationSuite suite;
    Ref<ID3D12Device3> dev3;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dev3.p))) || !dev3.p) return suite;
    Ref<ID3D12Device8> dev8;
    suite.has_device8 = SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dev8.p))) && dev8.p != nullptr;
    Ref<ID3D12Device10> dev10;
    suite.has_device10 = SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dev10.p))) && dev10.p != nullptr;

    VirtualRange range;
    if (!range.alloc(k256KiB, PAGE_READWRITE)) return suite;
    Ref<ID3D12Heap> heap;
    if (FAILED(dev3->OpenExistingHeapFromAddress(range.ptr, IID_PPV_ARGS(&heap.p))) || !heap.p) return suite;

    const auto record = [&](const char* name, HRESULT hr, ID3D12Resource* res) {
        ValidationCase c{name, hr};
        if (SUCCEEDED(hr) && res) {
            if (SUCCEEDED(res->GetHeapProperties(&c.heap_props, &c.heap_flags))) c.has_heap_props = true;
            res->Release();
        }
        suite.cases.push_back(c);
    };
    const auto place0 = [&](const char* name, UINT64 offset, const D3D12_RESOURCE_DESC& desc,
                            D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear, bool provide_out = true) {
        ID3D12Resource* res = nullptr;
        const HRESULT hr = device->CreatePlacedResource(
            heap.p, offset, &desc, state, clear, __uuidof(ID3D12Resource),
            provide_out ? reinterpret_cast<void**>(&res) : nullptr);
        record(name, hr, res);
    };

    place0("v0: buffer offset 0 COMMON", 0, existing_buffer_desc(k64KiB), D3D12_RESOURCE_STATE_COMMON, nullptr);
    place0("v0: UAV buffer offset 64 KiB COPY_SOURCE", k64KiB,
           existing_buffer_desc(k64KiB, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS), D3D12_RESOURCE_STATE_COPY_SOURCE,
           nullptr);
    place0("v0: buffer without ALLOW_CROSS_ADAPTER", 0, buffer_desc(k64KiB), D3D12_RESOURCE_STATE_COMMON, nullptr);
    place0("v0: unaligned offset 4096", 4096, existing_buffer_desc(k64KiB), D3D12_RESOURCE_STATE_COMMON, nullptr);
    place0("v0: out-of-bounds offset+width", 3 * k64KiB, existing_buffer_desc(2 * k64KiB), D3D12_RESOURCE_STATE_COMMON,
           nullptr);
    place0("v0: oversized width > heap", 0, existing_buffer_desc(2 * k256KiB), D3D12_RESOURCE_STATE_COMMON, nullptr);
    place0("v0: zero-width buffer", 0, existing_buffer_desc(0), D3D12_RESOURCE_STATE_COMMON, nullptr);

    D3D12_CLEAR_VALUE cv{};
    cv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    place0("v0: non-null clear value on buffer", 0, existing_buffer_desc(k64KiB), D3D12_RESOURCE_STATE_COMMON, &cv);
    place0("v0: valid args with null ppvResource", 0, existing_buffer_desc(k64KiB), D3D12_RESOURCE_STATE_COMMON, nullptr,
           false);

    if (suite.has_device8) {
        {
            const D3D12_RESOURCE_DESC1 d1 = existing_buffer_desc1(k64KiB);
            ID3D12Resource* res = nullptr;
            const HRESULT hr = dev8->CreatePlacedResource1(heap.p, k64KiB, &d1, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                           IID_PPV_ARGS(&res));
            record("v1: buffer offset 64 KiB COMMON", hr, res);
        }
        {
            const D3D12_RESOURCE_DESC1 d1_no = buffer_desc1(k64KiB);
            ID3D12Resource* res = nullptr;
            const HRESULT hr = dev8->CreatePlacedResource1(heap.p, k64KiB, &d1_no, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                           IID_PPV_ARGS(&res));
            record("v1: buffer without ALLOW_CROSS_ADAPTER", hr, res);
        }
    }
    if (suite.has_device10) {
        const D3D12_RESOURCE_DESC1 d1 = existing_buffer_desc1(k64KiB);
        {
            ID3D12Resource* res = nullptr;
            const HRESULT hr = dev10->CreatePlacedResource2(
                heap.p, 2 * k64KiB, &d1, D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, 0, nullptr, IID_PPV_ARGS(&res));
            record("v2: buffer offset 128 KiB UNDEFINED", hr, res);
        }
        {
            const D3D12_RESOURCE_DESC1 d1_no = buffer_desc1(k64KiB);
            ID3D12Resource* res = nullptr;
            const HRESULT hr = dev10->CreatePlacedResource2(
                heap.p, 2 * k64KiB, &d1_no, D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, 0, nullptr, IID_PPV_ARGS(&res));
            record("v2: buffer without ALLOW_CROSS_ADAPTER", hr, res);
        }
        {
            ID3D12Resource* res = nullptr;
            const HRESULT hr = dev10->CreatePlacedResource2(
                heap.p, 2 * k64KiB, &d1, D3D12_BARRIER_LAYOUT_COMMON, nullptr, 0, nullptr, IID_PPV_ARGS(&res));
            record("v2: buffer with BARRIER_LAYOUT_COMMON", hr, res);
        }
        {
            const DXGI_FORMAT cast[] = {DXGI_FORMAT_R8G8B8A8_UNORM};
            ID3D12Resource* res = nullptr;
            const HRESULT hr = dev10->CreatePlacedResource2(
                heap.p, 0, &d1, D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, 1, cast, IID_PPV_ARGS(&res));
            record("v2: buffer with castable format list", hr, res);
        }
    }
    std::printf("%s: recorded %zu existing-heap CreatePlacedResource cases\n", who, suite.cases.size());
    return suite;
}

int compare_validation_suites(const ValidationSuite& ours, const ValidationSuite& host, const char* host_name) {
    const char* who = "parity";
    EXPECT(!ours.cases.empty());
    size_t compared = 0;
    for (const auto& oc : ours.cases) {
        const auto it = std::find_if(host.cases.begin(), host.cases.end(),
                                     [&](const ValidationCase& hc) { return std::strcmp(hc.name, oc.name) == 0; });
        if (it == host.cases.end()) continue;
        if (oc.hr != it->hr) {
            std::fprintf(stderr, "vulkan vs %s: \"%s\" hr 0x%08lX != 0x%08lX\n", host_name, oc.name,
                         static_cast<unsigned long>(oc.hr), static_cast<unsigned long>(it->hr));
            return 1;
        }
        if (SUCCEEDED(oc.hr) && oc.has_heap_props && it->has_heap_props) {
            const bool props_match =
                oc.heap_props.Type == it->heap_props.Type &&
                oc.heap_props.CPUPageProperty == it->heap_props.CPUPageProperty &&
                oc.heap_props.MemoryPoolPreference == it->heap_props.MemoryPoolPreference &&
                oc.heap_props.CreationNodeMask == it->heap_props.CreationNodeMask &&
                oc.heap_props.VisibleNodeMask == it->heap_props.VisibleNodeMask &&
                oc.heap_flags == it->heap_flags;
            if (!props_match) {
                std::fprintf(stderr,
                             "vulkan vs %s: \"%s\" GetHeapProperties differs "
                             "(type=%d cpu=%d pool=%d flags=0x%x vs type=%d cpu=%d pool=%d flags=0x%x)\n",
                             host_name, oc.name, int(oc.heap_props.Type), int(oc.heap_props.CPUPageProperty),
                             int(oc.heap_props.MemoryPoolPreference), unsigned(oc.heap_flags),
                             int(it->heap_props.Type), int(it->heap_props.CPUPageProperty),
                             int(it->heap_props.MemoryPoolPreference), unsigned(it->heap_flags));
                return 1;
            }
        }
        ++compared;
    }
    if (ours.has_device8 && host.has_device8 && ours.has_device10 && host.has_device10) {
        EXPECT(compared == 15);
    } else {
        EXPECT(compared >= 9);
    }
    std::printf("vulkan vs %s: %zu existing-heap CreatePlacedResource & GetHeapProperties cases match\n", host_name,
                compared);
    return 0;
}

int check_metadata_map_and_wc(ID3D12Device* device, const char* who) {
    Ref<ID3D12Device3> dev3;
    EXPECT_HR(device->QueryInterface(IID_PPV_ARGS(&dev3.p)), S_OK);
    Ref<ID3D12Device8> dev8;
    EXPECT_HR(device->QueryInterface(IID_PPV_ARGS(&dev8.p)), S_OK);
    Ref<ID3D12Device10> dev10;
    EXPECT_HR(device->QueryInterface(IID_PPV_ARGS(&dev10.p)), S_OK);

    {
        VirtualRange range;
        EXPECT(range.alloc(k256KiB, PAGE_READWRITE));
        fill_pattern(range.ptr, k256KiB, 0x31);

        Ref<ID3D12Heap> heap;
        EXPECT_HR(dev3->OpenExistingHeapFromAddress(range.ptr, IID_PPV_ARGS(&heap.p)), S_OK);

        // Placed textures in existing heaps remain explicitly E_NOTIMPL for now.
        {
            const D3D12_RESOURCE_DESC td = texture_desc(64, 64, D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
            Ref<ID3D12Resource> tex;
            EXPECT_HR(device->CreatePlacedResource(heap.p, 0, &td, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                   IID_PPV_ARGS(&tex.p)),
                      E_NOTIMPL);
            EXPECT(tex.p == nullptr);
        }

        Ref<ID3D12Resource> bufs[4];
        const D3D12_RESOURCE_DESC d0 = existing_buffer_desc(k64KiB);
        const D3D12_RESOURCE_DESC1 d1 = existing_buffer_desc1(k64KiB);
        EXPECT_HR(device->CreatePlacedResource(heap.p, 0 * k64KiB, &d0, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&bufs[0].p)),
                  S_OK);
        EXPECT_HR(dev8->CreatePlacedResource1(heap.p, 1 * k64KiB, &d1, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&bufs[1].p)),
                  S_OK);
        EXPECT_HR(dev10->CreatePlacedResource2(heap.p, 2 * k64KiB, &d1, D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, 0,
                                               nullptr, IID_PPV_ARGS(&bufs[2].p)),
                  S_OK);
        EXPECT_HR(device->CreatePlacedResource(heap.p, 3 * k64KiB, &d0, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&bufs[3].p)),
                  S_OK);

        D3D12_GPU_VIRTUAL_ADDRESS vas[4]{};
        for (size_t i = 0; i < 4; ++i) {
            EXPECT((bufs[i]->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER) != 0);

            void* mapped = nullptr;
            EXPECT_HR(bufs[i]->Map(0, nullptr, &mapped), S_OK);
            EXPECT(mapped == range.bytes() + i * k64KiB);
            EXPECT(std::memcmp(mapped, range.bytes() + i * k64KiB, k64KiB) == 0);
            EXPECT(pattern_matches(mapped, k64KiB, 0x31));
            bufs[i]->Unmap(0, nullptr);

            // Map() invalid argument contracts.
            EXPECT_HR(bufs[i]->Map(1, nullptr, &mapped), E_INVALIDARG);
            const D3D12_RANGE oob{0, k64KiB + 1};
            EXPECT_HR(bufs[i]->Map(0, &oob, &mapped), E_INVALIDARG);

            D3D12_HEAP_PROPERTIES hp{};
            D3D12_HEAP_FLAGS hf = D3D12_HEAP_FLAG_NONE;
            EXPECT_HR(bufs[i]->GetHeapProperties(&hp, &hf), S_OK);
            EXPECT(hp.Type == D3D12_HEAP_TYPE_CUSTOM);
            EXPECT(hp.CPUPageProperty == D3D12_CPU_PAGE_PROPERTY_WRITE_BACK);
            EXPECT(hp.MemoryPoolPreference == D3D12_MEMORY_POOL_L0);
            EXPECT(hp.CreationNodeMask == 1 && hp.VisibleNodeMask == 1);
            EXPECT(hf == (D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
                          D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS));

            vas[i] = bufs[i]->GetGPUVirtualAddress();
            EXPECT(vas[i] != 0);
        }
        EXPECT(vas[1] - vas[0] == k64KiB);
        EXPECT(vas[2] - vas[1] == k64KiB);
        EXPECT(vas[3] - vas[2] == k64KiB);
    }

    // Write-combine existing heap: metadata, Map() preservation, and clean device teardown.
    {
        VirtualRange wc_range;
        EXPECT(wc_range.alloc(k64KiB, PAGE_READWRITE | PAGE_WRITECOMBINE));
        MEMORY_BASIC_INFORMATION mbi{};
        EXPECT(VirtualQuery(wc_range.ptr, &mbi, sizeof(mbi)) == sizeof(mbi));
        fill_pattern(wc_range.ptr, k64KiB, 0x57);

        Ref<ID3D12Heap> heap;
        EXPECT_HR(dev3->OpenExistingHeapFromAddress(wc_range.ptr, IID_PPV_ARGS(&heap.p)), S_OK);
        const D3D12_RESOURCE_DESC d = existing_buffer_desc(k64KiB);
        Ref<ID3D12Resource> placed;
        EXPECT_HR(device->CreatePlacedResource(heap.p, 0, &d, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&placed.p)),
                  S_OK);
        void* mapped = nullptr;
        EXPECT_HR(placed->Map(0, nullptr, &mapped), S_OK);
        EXPECT(mapped == wc_range.ptr);
        EXPECT(pattern_matches(mapped, k64KiB, 0x57));
        placed->Unmap(0, nullptr);

        D3D12_HEAP_PROPERTIES hp{};
        D3D12_HEAP_FLAGS hf = D3D12_HEAP_FLAG_NONE;
        EXPECT_HR(placed->GetHeapProperties(&hp, &hf), S_OK);
        const D3D12_CPU_PAGE_PROPERTY want_prop = (mbi.Protect & PAGE_WRITECOMBINE)
                                                      ? D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
                                                      : D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        EXPECT(hp.Type == D3D12_HEAP_TYPE_CUSTOM && hp.CPUPageProperty == want_prop &&
               hp.MemoryPoolPreference == D3D12_MEMORY_POOL_L0);
    }

    std::printf("%s: Map(), non-zeroing, GPU VA stride, GetHeapProperties, E_NOTIMPL texture & WC verified\n", who);
    return 0;
}

int check_coherence_aliasing_and_stress(ID3D12Device* device, const char* who) {
    CopyRunner runner;
    if (runner.init(device, who)) return 1;

    Ref<ID3D12Device3> dev3;
    EXPECT_HR(device->QueryInterface(IID_PPV_ARGS(&dev3.p)), S_OK);

    const D3D12_RESOURCE_DESC d64_comm = buffer_desc(k64KiB);
    const D3D12_RESOURCE_DESC d64 = existing_buffer_desc(k64KiB);
    const D3D12_HEAP_PROPERTIES upload_hp{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_hp{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> upload, readback;
    EXPECT_HR(device->CreateCommittedResource(&upload_hp, D3D12_HEAP_FLAG_NONE, &d64_comm,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p)),
              S_OK);
    EXPECT_HR(device->CreateCommittedResource(&readback_hp, D3D12_HEAP_FLAG_NONE, &d64_comm,
                                              D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p)),
              S_OK);

    // 3. Bidirectional CPU<->GPU coherence and placed-to-placed GPU copy within the same existing heap.
    {
        VirtualRange range;
        EXPECT(range.alloc(2 * k64KiB, PAGE_READWRITE));
        Ref<ID3D12Heap> heap;
        EXPECT_HR(dev3->OpenExistingHeapFromAddress(range.ptr, IID_PPV_ARGS(&heap.p)), S_OK);
        Ref<ID3D12Resource> buf_a, buf_b;
        EXPECT_HR(device->CreatePlacedResource(heap.p, 0, &d64, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&buf_a.p)),
                  S_OK);
        EXPECT_HR(device->CreatePlacedResource(heap.p, k64KiB, &d64, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&buf_b.p)),
                  S_OK);

        // CPU -> GPU via raw host pointer after creation
        fill_pattern(range.bytes(), k64KiB, 0x11);
        if (runner.copy_region(readback.p, 0, buf_a.p, 0, k64KiB, who)) return 1;
        void* rb_ptr = nullptr;
        EXPECT_HR(readback->Map(0, nullptr, &rb_ptr), S_OK);
        EXPECT(pattern_matches(rb_ptr, k64KiB, 0x11));
        readback->Unmap(0, nullptr);

        // GPU -> CPU direct read from raw host pages
        void* up_ptr = nullptr;
        EXPECT_HR(upload->Map(0, nullptr, &up_ptr), S_OK);
        fill_pattern(up_ptr, k64KiB, 0x22);
        upload->Unmap(0, nullptr);
        if (runner.copy_region(buf_a.p, 0, upload.p, 0, k64KiB, who)) return 1;
        EXPECT(pattern_matches(range.bytes(), k64KiB, 0x22));

        // Placed -> placed inside the same existing heap, then read directly from host RAM
        fill_pattern(range.bytes(), k64KiB, 0x33);
        std::memset(range.bytes() + k64KiB, 0, k64KiB);
        if (runner.copy_region(buf_b.p, 0, buf_a.p, 0, k64KiB, who)) return 1;
        EXPECT(pattern_matches(range.bytes() + k64KiB, k64KiB, 0x33));
    }

    // 4a. Overlapping aliased placed buffers: [0..128 KiB) and [64 KiB..128 KiB).
    {
        VirtualRange range;
        EXPECT(range.alloc(2 * k64KiB, PAGE_READWRITE));
        Ref<ID3D12Heap> heap;
        EXPECT_HR(dev3->OpenExistingHeapFromAddress(range.ptr, IID_PPV_ARGS(&heap.p)), S_OK);
        const D3D12_RESOURCE_DESC d128 = existing_buffer_desc(2 * k64KiB);
        Ref<ID3D12Resource> wide, hi;
        EXPECT_HR(device->CreatePlacedResource(heap.p, 0, &d128, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&wide.p)),
                  S_OK);
        EXPECT_HR(device->CreatePlacedResource(heap.p, k64KiB, &d64, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&hi.p)),
                  S_OK);

        fill_pattern(range.bytes() + k64KiB, k64KiB, 0x44);
        if (runner.copy_region(readback.p, 0, wide.p, k64KiB, k64KiB, who)) return 1;
        void* rb_ptr = nullptr;
        EXPECT_HR(readback->Map(0, nullptr, &rb_ptr), S_OK);
        EXPECT(pattern_matches(rb_ptr, k64KiB, 0x44));
        readback->Unmap(0, nullptr);

        void* up_ptr = nullptr;
        EXPECT_HR(upload->Map(0, nullptr, &up_ptr), S_OK);
        fill_pattern(up_ptr, k64KiB, 0x55);
        upload->Unmap(0, nullptr);
        if (runner.copy_region(hi.p, 0, upload.p, 0, k64KiB, who)) return 1;
        EXPECT(pattern_matches(range.bytes() + k64KiB, k64KiB, 0x55));
    }

    // 4b. Release ID3D12Heap BEFORE GPU copies on 4 placed buffers; placed resources keep imported VkDeviceMemory alive.
    {
        VirtualRange range;
        EXPECT(range.alloc(k256KiB, PAGE_READWRITE));
        Ref<ID3D12Resource> bufs[4];
        {
            Ref<ID3D12Heap> heap;
            EXPECT_HR(dev3->OpenExistingHeapFromAddress(range.ptr, IID_PPV_ARGS(&heap.p)), S_OK);
            for (size_t i = 0; i < 4; ++i) {
                EXPECT_HR(device->CreatePlacedResource(heap.p, i * k64KiB, &d64, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                       IID_PPV_ARGS(&bufs[i].p)),
                          S_OK);
            }
        }  // heap released here
        for (size_t i = 0; i < 4; ++i) {
            const std::uint8_t seed = static_cast<std::uint8_t>(0x60 + i);
            fill_pattern(range.bytes() + i * k64KiB, k64KiB, seed);
            if (runner.copy_region(readback.p, 0, bufs[i].p, 0, k64KiB, who)) return 1;
            void* rb_ptr = nullptr;
            EXPECT_HR(readback->Map(0, nullptr, &rb_ptr), S_OK);
            EXPECT(pattern_matches(rb_ptr, k64KiB, seed));
            readback->Unmap(0, nullptr);
        }
        void* up_ptr = nullptr;
        EXPECT_HR(upload->Map(0, nullptr, &up_ptr), S_OK);
        fill_pattern(up_ptr, k64KiB, 0x77);
        upload->Unmap(0, nullptr);
        if (runner.copy_region(bufs[3].p, 0, upload.p, 0, k64KiB, who)) return 1;
        EXPECT(pattern_matches(range.bytes() + 3 * k64KiB, k64KiB, 0x77));
    }  // buffers released here -> immediate VirtualFree on scope exit

    // 4c. 64-iteration alloc/open/place/copy/verify/release/VirtualFree stress loop.
    for (int iter = 0; iter < 64; ++iter) {
        VirtualRange range;
        EXPECT(range.alloc(k64KiB, PAGE_READWRITE));
        const std::uint8_t seed = static_cast<std::uint8_t>((iter * 7 + 3) & 0xFF);
        fill_pattern(range.ptr, k64KiB, seed);
        Ref<ID3D12Heap> heap;
        EXPECT_HR(dev3->OpenExistingHeapFromAddress(range.ptr, IID_PPV_ARGS(&heap.p)), S_OK);
        Ref<ID3D12Resource> placed;
        EXPECT_HR(device->CreatePlacedResource(heap.p, 0, &d64, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&placed.p)),
                  S_OK);
        heap.reset();
        if (runner.copy_region(readback.p, 0, placed.p, 0, k64KiB, who)) return 1;
        void* rb_ptr = nullptr;
        EXPECT_HR(readback->Map(0, nullptr, &rb_ptr), S_OK);
        EXPECT(pattern_matches(rb_ptr, k64KiB, seed));
        readback->Unmap(0, nullptr);
        placed.reset();
        range.reset();  // immediately frees pages while queue continues running
    }

    std::printf("%s: bidirectional coherence, aliasing, heap lifetime & 64-iter stress passed\n", who);
    return 0;
}

int check_xbox_proxy(const vktest::Device& vk) {
    const char* who = "xbox proxy";
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.host_pointer_alignment = vk.host_pointer_alignment;
    info.buffer_device_address = vk.buffer_device_address;

    Ref<ID3D12Device> real;
    EXPECT_HR(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&real.p)), S_OK);
    real->AddRef();  // create_device_proxy takes one reference on success
    Ref<ID3D12Device> proxy;
    if (FAILED(rt::d3d12_xs::create_device_proxy(real.p, __uuidof(ID3D12Device),
                                                 reinterpret_cast<void**>(&proxy.p)))) {
        real->Release();
        return 1;
    }

    // Slot 48 on the Xbox proxy is CreateCommandQueueX rather than PC OpenExistingHeapFromAddress; open the heap on
    // the underlying real device and exercise CreatePlacedResource (slot 29) + Map + GPU copy through the proxy.
    Ref<ID3D12Device3> real3;
    EXPECT_HR(real->QueryInterface(IID_PPV_ARGS(&real3.p)), S_OK);
    VirtualRange range;
    EXPECT(range.alloc(2 * k64KiB, PAGE_READWRITE));
    fill_pattern(range.ptr, 2 * k64KiB, 0x89);

    Ref<ID3D12Heap> heap;
    EXPECT_HR(real3->OpenExistingHeapFromAddress(range.ptr, IID_PPV_ARGS(&heap.p)), S_OK);
    const D3D12_RESOURCE_DESC d64 = existing_buffer_desc(k64KiB);
    Ref<ID3D12Resource> placed;
    EXPECT_HR(proxy->CreatePlacedResource(heap.p, k64KiB, &d64, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                          IID_PPV_ARGS(&placed.p)),
              S_OK);
    void* mapped = nullptr;
    EXPECT_HR(placed->Map(0, nullptr, &mapped), S_OK);
    EXPECT(mapped == range.bytes() + k64KiB);
    EXPECT(pattern_matches(mapped, k64KiB, 0x89));
    placed->Unmap(0, nullptr);

    CopyRunner runner;
    if (runner.init(proxy.p, who)) return 1;
    const D3D12_RESOURCE_DESC d64_comm = buffer_desc(k64KiB);
    const D3D12_HEAP_PROPERTIES readback_hp{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> readback;
    EXPECT_HR(proxy->CreateCommittedResource(&readback_hp, D3D12_HEAP_FLAG_NONE, &d64_comm,
                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p)),
              S_OK);
    fill_pattern(range.bytes() + k64KiB, k64KiB, 0x9B);
    if (runner.copy_region(readback.p, 0, placed.p, 0, k64KiB, who)) return 1;
    void* rb_ptr = nullptr;
    EXPECT_HR(readback->Map(0, nullptr, &rb_ptr), S_OK);
    EXPECT(pattern_matches(rb_ptr, k64KiB, 0x9B));
    readback->Unmap(0, nullptr);

    std::printf("%s: CreatePlacedResource (slot 29) on existing heap + Map + GPU readback passed\n", who);
    return 0;
}

int run_vulkan_mode(bool enable_bda) {
    const char* mode_name = enable_bda ? "vulkan (bda+arena)" : "vulkan (no-bda)";
    const char* who = mode_name;
    vktest::Device vk;
    vk.request_buffer_device_address = enable_bda;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.host_pointer_alignment) {
        std::printf("%s: skipped, no VK_EXT_external_memory_host\n", mode_name);
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
        info.buffer_device_address = vk.buffer_device_address;

        Ref<ID3D12Device> ours;
        EXPECT_HR(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)), S_OK);
        const ValidationSuite vulkan_suite = run_validation_suite(ours.p, mode_name);
        failures += vulkan_suite.cases.empty() ? 1 : 0;
        failures += check_metadata_map_and_wc(ours.p, mode_name);
        failures += check_coherence_aliasing_and_stress(ours.p, mode_name);

        if (!enable_bda && !hostwine::host_is_wine()) {
            Ref<IDXGIFactory4> factory;
            if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
                for (const bool warp : {false, true}) {
                    Ref<IDXGIAdapter> adapter;
                    Ref<ID3D12Device> host;
                    std::string adapter_name;
                    if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
                    if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
                    const std::string host_who = "host " + adapter_name;
                    const ValidationSuite host_suite = run_validation_suite(host.p, host_who.c_str());
                    failures += compare_validation_suites(vulkan_suite, host_suite, host_who.c_str());
                }
            }
        }
        if (enable_bda) failures += check_xbox_proxy(vk);
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%s: %u validation errors\n", mode_name, vk.validation_errors());
        return 1;
    }
    return failures ? 1 : 0;
}

}  // namespace

int main() {
    for (const bool enable_bda : {false, true}) {
        const int rc = run_vulkan_mode(enable_bda);
        if (rc != 0) return rc;
    }
    return 0;
}
