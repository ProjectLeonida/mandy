// ID3D12GraphicsCommandList4 render passes (D3D-27) on the Vulkan-backed list, compared byte for byte with Windows
// WARP through the D3D12 API on both sides: CLEAR and PRESERVE beginning accesses on two render targets, the depth
// and stencil planes cleared by their own accesses, an MSAA RESOLVE ending access into a rectangle of a larger
// target, a pass suspended in one list and resumed in the next, and the list errors Close reports.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

constexpr UINT kSize = 4;  // every target is 4 x 4
constexpr DXGI_FORMAT kColor = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT kDepth = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;

// What a case leaves behind: the bytes of every subresource it reads back, and the HRESULTs it records.
struct Result {
    std::vector<std::vector<unsigned char>> images;
    std::vector<HRESULT> results;
};

// One queue, fence and readback path per case. `list` is the first list; `second()` makes another one, for the
// suspend/resume case.
struct Context {
    ID3D12Device* device = nullptr;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator, allocator2;
    Ref<ID3D12GraphicsCommandList4> list, list2;
    Ref<ID3D12Fence> fence;
    UINT64 fence_value = 0;
    std::deque<Ref<ID3D12Resource>> readbacks;  // a deque: Ref is not movable, and the buffers must not move
    struct Pending {
        ID3D12Resource* readback;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
        UINT bytes;
        UINT64 size;
    };
    std::vector<Pending> pending;

    bool create(ID3D12Device* d) {
        device = d;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                 IID_PPV_ARGS(&list.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        return true;
    }
    bool second() {
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator2.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator2.p, nullptr,
                                                 IID_PPV_ARGS(&list2.p))));
        return true;
    }

    bool texture(DXGI_FORMAT format, UINT samples, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                 Ref<ID3D12Resource>& out) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = desc.Height = kSize;
        desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.Format = format;
        desc.SampleDesc.Count = samples;
        desc.Flags = flags;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                       IID_PPV_ARGS(&out.p))));
        return true;
    }

    // A descriptor heap of `count` RTVs or DSVs; handle(i) is its i-th slot.
    struct Heap {
        Ref<ID3D12DescriptorHeap> heap;
        UINT increment = 0;
        D3D12_CPU_DESCRIPTOR_HANDLE handle(UINT i) const {
            auto h = heap.p->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * increment;
            return h;
        }
    };
    bool heap(D3D12_DESCRIPTOR_HEAP_TYPE type, UINT count, Heap& out) {
        const D3D12_DESCRIPTOR_HEAP_DESC desc{type, count, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&out.heap.p))));
        out.increment = device->GetDescriptorHandleIncrementSize(type);
        return true;
    }

    static void barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                        D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = resource;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter = after;
        l->ResourceBarrier(1, &b);
    }

    // Records a copy of one subresource (already in COPY_SOURCE) to a new readback buffer; finish() reads it.
    bool read(ID3D12GraphicsCommandList* l, ID3D12Resource* resource, UINT subresource, UINT bytes) {
        const auto desc = resource->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 total = 0;
        device->GetCopyableFootprints(&desc, subresource, 1, 0, &footprint, nullptr, nullptr, &total);
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = total;
        buffer.Height = buffer.SampleDesc.Count = 1;
        buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        readbacks.emplace_back();
        CHECK(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readbacks.back().p))));
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = resource;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = subresource;
        dst.pResource = readbacks.back().p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        pending.push_back({readbacks.back().p, footprint, bytes, total});
        return true;
    }

    // Closes and executes the lists in one ExecuteCommandLists, waits, and appends every pending readback.
    bool finish(Result& result, bool both = false) {
        CHECK(SUCCEEDED(list->Close()));
        if (both) CHECK(SUCCEEDED(list2->Close()));
        ID3D12CommandList* lists[] = {list.p, list2.p};
        queue->ExecuteCommandLists(both ? 2 : 1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, ++fence_value)));
        Event event;
        CHECK(event.handle);
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(fence_value, event.handle)));
        CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
        for (const auto& p : pending) {
            void* mapped = nullptr;
            const D3D12_RANGE range{0, SIZE_T(p.size)};
            CHECK(SUCCEEDED(p.readback->Map(0, &range, &mapped)) && mapped);
            std::vector<unsigned char> pixels;
            for (UINT row = 0; row < p.footprint.Footprint.Height; ++row) {
                const auto* start = static_cast<unsigned char*>(mapped) + row * p.footprint.Footprint.RowPitch;
                pixels.insert(pixels.end(), start, start + p.footprint.Footprint.Width * p.bytes);
            }
            const D3D12_RANGE none{};
            p.readback->Unmap(0, &none);
            result.images.push_back(std::move(pixels));
        }
        pending.clear();
        return true;
    }
};

D3D12_RENDER_PASS_BEGINNING_ACCESS begin_access(D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE type,
                                                DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN,
                                                std::array<float, 4> color = {}, float depth = 0, UINT8 stencil = 0) {
    D3D12_RENDER_PASS_BEGINNING_ACCESS access{};
    access.Type = type;
    if (type == D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR) {
        access.Clear.ClearValue.Format = format;
        if (format == kDepth) {
            access.Clear.ClearValue.DepthStencil = {depth, stencil};
        } else {
            std::memcpy(access.Clear.ClearValue.Color, color.data(), sizeof(float) * 4);
        }
    }
    return access;
}

D3D12_RENDER_PASS_ENDING_ACCESS end_access(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE type) {
    D3D12_RENDER_PASS_ENDING_ACCESS access{};
    access.Type = type;
    return access;
}

constexpr auto kBeginPreserve = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE;
constexpr auto kBeginClear = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
constexpr auto kBeginNoAccess = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS;
constexpr auto kEndPreserve = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
constexpr auto kEndNoAccess = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS;
constexpr auto kEndResolve = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE;
constexpr auto kRT = D3D12_RESOURCE_STATE_RENDER_TARGET;
constexpr auto kCopySrc = D3D12_RESOURCE_STATE_COPY_SOURCE;

// Two colour targets cleared red and blue before the pass; the pass CLEARs the first to green and PRESERVEs the
// second. Expected: green, blue.
bool clear_and_preserve(ID3D12Device* device, Result& result) {
    Context c;
    CHECK(c.create(device));
    Ref<ID3D12Resource> rt[2];
    for (auto& r : rt) CHECK(c.texture(kColor, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, kRT, r));
    Context::Heap rtv;
    CHECK(c.heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, rtv));
    for (UINT i = 0; i < 2; ++i) device->CreateRenderTargetView(rt[i].p, nullptr, rtv.handle(i));
    const FLOAT red[4]{1, 0, 0, 1}, blue[4]{0, 0, 1, 1};
    c.list->ClearRenderTargetView(rtv.handle(0), red, 0, nullptr);
    c.list->ClearRenderTargetView(rtv.handle(1), blue, 0, nullptr);
    D3D12_RENDER_PASS_RENDER_TARGET_DESC targets[2]{};
    targets[0] = {rtv.handle(0), begin_access(kBeginClear, kColor, {0, 1, 0, 1}), end_access(kEndPreserve)};
    targets[1] = {rtv.handle(1), begin_access(kBeginPreserve), end_access(kEndPreserve)};
    c.list->BeginRenderPass(2, targets, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
    c.list->EndRenderPass();
    for (auto& r : rt) {
        Context::barrier(c.list.p, r.p, kRT, kCopySrc);
        CHECK(c.read(c.list.p, r.p, 0, 4));
    }
    return c.finish(result);
}

// A D32_FLOAT_S8X24 target cleared to depth 1, stencil 3 before the passes. Pass 1 CLEARs depth to 0.25 and
// PRESERVEs stencil; with `second`, pass 2 leaves depth (NO_ACCESS) and CLEARs stencil to 9. Expected:
// 0.25 / 3 after pass 1, 0.25 / 9 after pass 2.
bool depth_and_stencil_planes(ID3D12Device* device, bool second, Result& result) {
    Context c;
    CHECK(c.create(device));
    Ref<ID3D12Resource> ds;
    CHECK(c.texture(kDepth, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE, ds));
    Context::Heap dsv;
    CHECK(c.heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, dsv));
    device->CreateDepthStencilView(ds.p, nullptr, dsv.handle(0));
    c.list->ClearDepthStencilView(dsv.handle(0), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 3, 0,
                                  nullptr);
    D3D12_RENDER_PASS_DEPTH_STENCIL_DESC pass1{};
    pass1.cpuDescriptor = dsv.handle(0);
    pass1.DepthBeginningAccess = begin_access(kBeginClear, kDepth, {}, 0.25f, 0);
    pass1.StencilBeginningAccess = begin_access(kBeginPreserve);
    pass1.DepthEndingAccess = end_access(kEndPreserve);
    pass1.StencilEndingAccess = end_access(kEndPreserve);
    c.list->BeginRenderPass(0, nullptr, &pass1, D3D12_RENDER_PASS_FLAG_NONE);
    c.list->EndRenderPass();
    if (second) {
        D3D12_RENDER_PASS_DEPTH_STENCIL_DESC pass2{};
        pass2.cpuDescriptor = dsv.handle(0);
        pass2.DepthBeginningAccess = begin_access(kBeginNoAccess);
        pass2.StencilBeginningAccess = begin_access(kBeginClear, kDepth, {}, 0, 9);
        pass2.DepthEndingAccess = end_access(kEndNoAccess);
        pass2.StencilEndingAccess = end_access(kEndPreserve);
        c.list->BeginRenderPass(0, nullptr, &pass2, D3D12_RENDER_PASS_FLAG_NONE);
        c.list->EndRenderPass();
    }
    Context::barrier(c.list.p, ds.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, kCopySrc);
    CHECK(c.read(c.list.p, ds.p, 0, 4));  // the depth plane, R32_FLOAT
    CHECK(c.read(c.list.p, ds.p, 1, 1));  // the stencil plane, R8_UINT
    return c.finish(result);
}

// A 4x MSAA colour target CLEARed in the pass and RESOLVEd at its end into the 2 x 2 rectangle at (1, 1) of a
// single-sample target cleared blue before. With `suspend`, the pass is split: it suspends in the first list and
// resumes in the second, whose CLEAR (red) and the first part's RESOLVE are the ones D3D12 does not run. Expected:
// the clear colour in the rectangle, blue around it.
bool msaa_resolve(ID3D12Device* device, bool suspend, Result& result) {
    Context c;
    CHECK(c.create(device));
    if (suspend) CHECK(c.second());
    Ref<ID3D12Resource> msaa, dst;
    CHECK(c.texture(kColor, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, kRT, msaa));
    CHECK(c.texture(kColor, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, kRT, dst));
    Context::Heap rtv;
    CHECK(c.heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, rtv));
    device->CreateRenderTargetView(msaa.p, nullptr, rtv.handle(0));
    device->CreateRenderTargetView(dst.p, nullptr, rtv.handle(1));
    const FLOAT blue[4]{0, 0, 1, 1};
    c.list->ClearRenderTargetView(rtv.handle(1), blue, 0, nullptr);
    Context::barrier(c.list.p, dst.p, kRT, D3D12_RESOURCE_STATE_RESOLVE_DEST);

    D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS sub{0, 0, 1, 1, {0, 0, 2, 2}};
    const auto pass = [&](ID3D12GraphicsCommandList4* l, std::array<float, 4> color, D3D12_RENDER_PASS_FLAGS flags) {
        D3D12_RENDER_PASS_RENDER_TARGET_DESC target{};
        target.cpuDescriptor = rtv.handle(0);
        target.BeginningAccess = begin_access(kBeginClear, kColor, color);
        target.EndingAccess = end_access(kEndResolve);
        auto& resolve = target.EndingAccess.Resolve;
        resolve.pSrcResource = msaa.p;
        resolve.pDstResource = dst.p;
        resolve.SubresourceCount = 1;
        resolve.pSubresourceParameters = &sub;
        resolve.Format = kColor;
        resolve.ResolveMode = D3D12_RESOLVE_MODE_AVERAGE;
        resolve.PreserveResolveSource = FALSE;
        l->BeginRenderPass(1, &target, nullptr, flags);
        l->EndRenderPass();
    };
    // 0.2 / 0.4 / 0.6 / 0.8 are exact enough in UNORM8 that every sample averages to the same byte.
    ID3D12GraphicsCommandList4* last = c.list.p;
    if (suspend) {
        pass(c.list.p, {0.2f, 0.4f, 0.6f, 0.8f}, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
        pass(c.list2.p, {1, 0, 0, 1}, D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
        last = c.list2.p;
    } else {
        pass(c.list.p, {0.2f, 0.4f, 0.6f, 0.8f}, D3D12_RENDER_PASS_FLAG_NONE);
    }
    Context::barrier(last, dst.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, kCopySrc);
    CHECK(c.read(last, dst.p, 0, 4));
    return c.finish(result, suspend);
}

// What Close returns, each on a list of its own: after EndRenderPass with no pass begun, with a pass left open,
// after a nested BeginRenderPass, and for one pass begun and ended.
bool list_errors(ID3D12Device* device, Result& result) {
    Context c;
    CHECK(c.create(device));
    Ref<ID3D12Resource> rt;
    CHECK(c.texture(kColor, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, kRT, rt));
    Context::Heap rtv;
    CHECK(c.heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, rtv));
    device->CreateRenderTargetView(rt.p, nullptr, rtv.handle(0));
    D3D12_RENDER_PASS_RENDER_TARGET_DESC target{rtv.handle(0), begin_access(kBeginPreserve), end_access(kEndPreserve)};
    const auto close_after = [&](int begins, int ends, Result& r) {
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList4> list;
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                 IID_PPV_ARGS(&list.p))));
        for (int i = 0; i < begins; ++i) list->BeginRenderPass(1, &target, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
        for (int i = 0; i < ends; ++i) list->EndRenderPass();
        r.results.push_back(list->Close());
        return true;
    };
    CHECK(close_after(0, 1, result));  // EndRenderPass without a pass
    CHECK(close_after(1, 0, result));  // a pass left open
    CHECK(close_after(2, 1, result));  // a nested BeginRenderPass
    CHECK(close_after(1, 1, result));  // a valid pass
    return true;
}

std::string hex(const std::vector<unsigned char>& bytes) {
    std::string out;
    char buffer[4];
    for (unsigned char b : bytes) {
        std::snprintf(buffer, sizeof(buffer), "%02x", b);
        out += buffer;
    }
    return out;
}

std::vector<unsigned char> repeat(std::initializer_list<unsigned char> pixel, UINT count) {
    std::vector<unsigned char> out;
    for (UINT i = 0; i < count; ++i) out.insert(out.end(), pixel);
    return out;
}

std::vector<unsigned char> depth_bytes(float value) {
    std::vector<unsigned char> out(kSize * kSize * 4);
    for (UINT i = 0; i < kSize * kSize; ++i) std::memcpy(out.data() + i * 4, &value, 4);
    return out;
}

// The blue 4 x 4 target with the 2 x 2 rectangle at (1, 1) in `inside`.
std::vector<unsigned char> resolved(std::initializer_list<unsigned char> inside) {
    std::vector<unsigned char> out;
    for (UINT y = 0; y < kSize; ++y) {
        for (UINT x = 0; x < kSize; ++x) {
            const bool in = x >= 1 && x < 3 && y >= 1 && y < 3;
            if (in) out.insert(out.end(), inside);
            else out.insert(out.end(), {0, 0, 255, 255});
        }
    }
    return out;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const unsigned errors_before = vk.validation_errors();
    int failures = 0;
    {
        Ref<ID3D12Device> ours, host;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
            FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            std::puts("WARP reference unavailable: differential test skipped");
            return 77;
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5{};
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options5, sizeof(options5))) &&
            options5.RenderPassesTier != D3D12_RENDER_PASS_TIER_0) {
            std::fprintf(stderr, "RenderPassesTier is %d, this emulation is Tier 0\n", int(options5.RenderPassesTier));
            ++failures;
        }

        using Case = bool (*)(ID3D12Device*, Result&);
        struct Named {
            const char* name;
            Case run;
            std::vector<std::vector<unsigned char>> expected;  // empty: compared with WARP only
        };
        const auto green = repeat({0, 255, 0, 255}, kSize * kSize), blue = repeat({0, 0, 255, 255}, kSize * kSize);
        const auto resolve_colour = resolved({51, 102, 153, 204});
        const Named cases[] = {
            {"clear_and_preserve", clear_and_preserve, {green, blue}},
            {"depth_clear_stencil_preserve",
             [](ID3D12Device* d, Result& r) { return depth_and_stencil_planes(d, false, r); },
             {depth_bytes(0.25f), repeat({3}, kSize * kSize)}},
            {"stencil_clear_depth_no_access",
             [](ID3D12Device* d, Result& r) { return depth_and_stencil_planes(d, true, r); },
             {depth_bytes(0.25f), repeat({9}, kSize * kSize)}},
            {"msaa_resolve_region", [](ID3D12Device* d, Result& r) { return msaa_resolve(d, false, r); },
             {resolve_colour}},
            {"suspend_resume", [](ID3D12Device* d, Result& r) { return msaa_resolve(d, true, r); }, {resolve_colour}},
            {"list_errors", list_errors, {}},
        };
        for (const auto& item : cases) {
            Result from_host, from_ours;
            if (!item.run(host.p, from_host)) {
                std::fprintf(stderr, "%s: the WARP run failed\n", item.name);
                ++failures;
                continue;
            }
            if (!item.run(ours.p, from_ours)) {
                std::fprintf(stderr, "%s: the Vulkan-backed run failed\n", item.name);
                ++failures;
                continue;
            }
            bool ok = from_host.images == from_ours.images && from_host.results == from_ours.results;
            if (!item.expected.empty() && from_host.images != item.expected) {
                std::fprintf(stderr, "%s: WARP differs from the expected bytes; the case is not testing what it says\n",
                             item.name);
                ok = false;
            }
            for (size_t i = 0; !ok && i < from_host.images.size() && i < from_ours.images.size(); ++i) {
                if (from_host.images[i] == from_ours.images[i]) continue;
                std::fprintf(stderr, "%s image %zu:\n  WARP %s\n  ours %s\n", item.name, i,
                             hex(from_host.images[i]).c_str(), hex(from_ours.images[i]).c_str());
            }
            for (size_t i = 0; !ok && i < from_host.results.size() && i < from_ours.results.size(); ++i) {
                std::fprintf(stderr, "%s result %zu: WARP 0x%08lx, ours 0x%08lx\n", item.name, i,
                             static_cast<unsigned long>(from_host.results[i]),
                             static_cast<unsigned long>(from_ours.results[i]));
            }
            std::printf("%s %s\n", ok ? "PASS" : "FAIL", item.name);
            if (!ok) ++failures;
        }
    }
    const unsigned errors = vk.validation_errors() - errors_before;
    if (errors) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", errors);
        ++failures;
    }
    std::printf("%s\n", failures ? "render pass test FAILED" : "render pass test passed");
    return failures ? 1 : 0;
}
