// D3D12 render passes (ID3D12GraphicsCommandList4::BeginRenderPass / EndRenderPass, D3D-27) on the layer, compared
// with Windows' own D3D12 over WARP (no debug layer). Two groups:
//   data   each case records a render pass whose beginning and ending accesses write the targets (CLEAR on colour,
//          depth and stencil separately, PRESERVE, a RESOLVE ending into a region and into the whole texture,
//          SUSPENDING/RESUMING), runs it, copies every plane into a readback buffer and compares the bytes with WARP.
//   close  each case records a few calls on a fresh list and only closes it; the layer is held to WARP's verdict
//          (Close succeeds or fails; both codes are printed, as in d3d12_close_parity_test).
// The layer emulates render passes on its own commands (RenderPassesTier 0), which is what the PC runtime does for a
// driver at tier 0, so WARP is the reference for both the bytes and the recording rules. No game data or shaders.
// Needs a Vulkan GPU (exit 77 without one) and Microsoft's WARP (exit 77 under Wine, whose host D3D12 is vkd3d).
#include "vk_test_device.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "    line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
// Owns one reference. Movable, not copyable, so a vector of them can grow without releasing twice.
template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(Ref&& other) noexcept : p(other.p) { other.p = nullptr; }
    Ref& operator=(Ref&&) = delete;
    Ref(const Ref&) = delete;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

constexpr UINT kWidth = 5, kHeight = 3;  // odd sizes expose a wrong treatment of footprint padding
constexpr DXGI_FORMAT kRgba = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT kFloat = DXGI_FORMAT_R32_FLOAT;
constexpr DXGI_FORMAT kDepth = DXGI_FORMAT_D24_UNORM_S8_UINT;

D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, UINT samples = 1) {
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = kWidth;
    texture.Height = kHeight;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = format;
    texture.SampleDesc.Count = samples;
    texture.Flags = flags;
    return texture;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* image, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

// One subresource copied into a readback buffer, read after the list has run.
struct Plane {
    Ref<ID3D12Resource> buffer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT64 total = 0;
    unsigned bytes = 0;  // per texel in this plane
};

// Everything one case creates on one device: a direct queue and list, the textures and their views.
struct Scene {
    ID3D12Device* device = nullptr;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList4> list4;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    UINT rtv_size = 0;
    std::vector<Ref<ID3D12Resource>> textures;
    std::vector<Plane> planes;

    bool create(ID3D12Device* d, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT) {
        device = d;
        const D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, type, allocator.p, nullptr, IID_PPV_ARGS(&list.p))));
        CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list4.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC rtv{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 9, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC dsv{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&rtv_heap.p))));
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&dsv, IID_PPV_ARGS(&dsv_heap.p))));
        rtv_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        textures.reserve(16);
        return true;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE rtv(UINT slot) const {
        D3D12_CPU_DESCRIPTOR_HANDLE handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += SIZE_T(slot) * rtv_size;
        return handle;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dsv() const { return dsv_heap->GetCPUDescriptorHandleForHeapStart(); }

    // A texture in `state`; a render target gets an RTV in `slot`, a depth texture the one DSV.
    ID3D12Resource* texture(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                            UINT samples = 1, int slot = -1) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        const D3D12_RESOURCE_DESC desc = texture_desc(format, flags, samples);
        textures.emplace_back();
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                   IID_PPV_ARGS(&textures.back().p))))
            return nullptr;
        ID3D12Resource* resource = textures.back().p;
        if (slot >= 0) device->CreateRenderTargetView(resource, nullptr, rtv(UINT(slot)));
        if (flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) device->CreateDepthStencilView(resource, nullptr, dsv());
        return resource;
    }

    // Records the copy of one subresource of a 1-sample texture (in COPY_SOURCE) into a new readback buffer.
    bool copy_out(ID3D12Resource* image, DXGI_FORMAT format, UINT subresource, unsigned bytes) {
        planes.emplace_back();
        Plane& plane = planes.back();
        plane.bytes = bytes;
        const D3D12_RESOURCE_DESC texture = texture_desc(format, D3D12_RESOURCE_FLAG_NONE);
        device->GetCopyableFootprints(&texture, subresource, 1, 0, &plane.placed, nullptr, nullptr, &plane.total);
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = plane.total;
        buffer.Height = buffer.SampleDesc.Count = 1;
        buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        CHECK(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&plane.buffer.p))));
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = image;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = subresource;
        dst.pResource = plane.buffer.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = plane.placed;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        return true;
    }

    // Runs the list and reads every plane back, tightly packed.
    bool execute(std::vector<std::vector<unsigned char>>& out) {
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
        Event event;
        CHECK(event.handle);
        CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
        CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
        for (Plane& plane : planes) {
            void* mapped = nullptr;
            const D3D12_RANGE range{0, SIZE_T(plane.total)};
            CHECK(SUCCEEDED(plane.buffer->Map(0, &range, &mapped)) && mapped);
            std::vector<unsigned char> pixels(std::size_t(kWidth) * kHeight * plane.bytes);
            for (UINT y = 0; y < kHeight; ++y)
                std::memcpy(pixels.data() + std::size_t(y) * kWidth * plane.bytes,
                            static_cast<unsigned char*>(mapped) + std::size_t(y) * plane.placed.Footprint.RowPitch,
                            std::size_t(kWidth) * plane.bytes);
            const D3D12_RANGE none{};
            plane.buffer->Unmap(0, &none);
            out.push_back(std::move(pixels));
        }
        return true;
    }
};

D3D12_RENDER_PASS_BEGINNING_ACCESS begin_clear(DXGI_FORMAT format, const std::array<float, 4>& color) {
    D3D12_RENDER_PASS_BEGINNING_ACCESS access{};
    access.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
    access.Clear.ClearValue.Format = format;
    std::memcpy(access.Clear.ClearValue.Color, color.data(), sizeof(float) * 4);
    return access;
}
D3D12_RENDER_PASS_BEGINNING_ACCESS begin_clear_depth(float depth, UINT8 stencil) {
    D3D12_RENDER_PASS_BEGINNING_ACCESS access{};
    access.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
    access.Clear.ClearValue.Format = kDepth;
    access.Clear.ClearValue.DepthStencil = {depth, stencil};
    return access;
}
D3D12_RENDER_PASS_BEGINNING_ACCESS begin(D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE type) {
    D3D12_RENDER_PASS_BEGINNING_ACCESS access{};
    access.Type = type;
    return access;
}
D3D12_RENDER_PASS_ENDING_ACCESS end(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE type) {
    D3D12_RENDER_PASS_ENDING_ACCESS access{};
    access.Type = type;
    return access;
}
D3D12_RENDER_PASS_RENDER_TARGET_DESC target(D3D12_CPU_DESCRIPTOR_HANDLE handle,
                                            const D3D12_RENDER_PASS_BEGINNING_ACCESS& first,
                                            const D3D12_RENDER_PASS_ENDING_ACCESS& last) {
    return {handle, first, last};
}
const auto kPreserve = begin(D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE);
const auto kKeep = end(D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE);

constexpr std::array<float, 4> kBefore{0.25f, 0.5f, 0.75f, 1.0f};
constexpr std::array<float, 4> kCleared{1.0f, 0.0f, 0.5f, 0.25f};
constexpr std::array<float, 4> kSecond{0.0f, 1.0f, 0.125f, 0.5f};
constexpr std::array<float, 4> kFloatBefore{-3.5f, 0, 0, 0};

// ---- data cases ---------------------------------------------------------------------------------------------------

// Colour target 0 is cleared by the pass, colour target 1 keeps what a clear before the pass wrote; the depth plane
// is cleared by the pass while the stencil plane keeps its value, or the other way round.
bool colour_and_depth(Scene& s, bool clear_depth_plane) {
    ID3D12Resource* rt0 = s.texture(kRgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_RENDER_TARGET, 1, 0);
    ID3D12Resource* rt1 = s.texture(kFloat, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_RENDER_TARGET, 1, 1);
    ID3D12Resource* ds = s.texture(kDepth, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    CHECK(rt0 && rt1 && ds);
    s.list->ClearRenderTargetView(s.rtv(0), kBefore.data(), 0, nullptr);
    s.list->ClearRenderTargetView(s.rtv(1), kFloatBefore.data(), 0, nullptr);
    s.list->ClearDepthStencilView(s.dsv(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5f, 7, 0, nullptr);
    const D3D12_RENDER_PASS_RENDER_TARGET_DESC targets[] = {
        target(s.rtv(0), begin_clear(kRgba, kCleared), kKeep),
        target(s.rtv(1), kPreserve, kKeep),
    };
    D3D12_RENDER_PASS_DEPTH_STENCIL_DESC depth{};
    depth.cpuDescriptor = s.dsv();
    depth.DepthBeginningAccess = clear_depth_plane ? begin_clear_depth(0.25f, 0x5A) : kPreserve;
    depth.StencilBeginningAccess = clear_depth_plane ? kPreserve : begin_clear_depth(0.0f, 0xA5);
    depth.DepthEndingAccess = depth.StencilEndingAccess = kKeep;
    s.list4->BeginRenderPass(2, targets, &depth, D3D12_RENDER_PASS_FLAG_NONE);
    s.list4->EndRenderPass();
    transition(s.list.p, rt0, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(s.list.p, rt1, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(s.list.p, ds, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    return s.copy_out(rt0, kRgba, 0, 4) && s.copy_out(rt1, kFloat, 0, 4) && s.copy_out(ds, kDepth, 0, 4) &&
           s.copy_out(ds, kDepth, 1, 1);
}
bool depth_cleared(Scene& s) { return colour_and_depth(s, true); }
bool stencil_cleared(Scene& s) { return colour_and_depth(s, false); }

// Two passes on the same target in one list: the second pass's clear is the one left.
bool two_passes(Scene& s) {
    ID3D12Resource* rt = s.texture(kRgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                   D3D12_RESOURCE_STATE_RENDER_TARGET, 1, 0);
    CHECK(rt);
    const auto first = target(s.rtv(0), begin_clear(kRgba, kCleared), kKeep);
    const auto second = target(s.rtv(0), begin_clear(kRgba, kSecond), kKeep);
    s.list4->BeginRenderPass(1, &first, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
    s.list4->EndRenderPass();
    s.list4->BeginRenderPass(1, &second, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
    s.list4->EndRenderPass();
    transition(s.list.p, rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    return s.copy_out(rt, kRgba, 0, 4);
}

// A suspended pass resumed later in the list: what each pass's beginning access leaves.
bool suspend_resume(Scene& s) {
    ID3D12Resource* rt = s.texture(kRgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                   D3D12_RESOURCE_STATE_RENDER_TARGET, 1, 0);
    CHECK(rt);
    const auto first = target(s.rtv(0), begin_clear(kRgba, kCleared), kKeep);
    const auto second = target(s.rtv(0), begin_clear(kRgba, kSecond), kKeep);
    s.list4->BeginRenderPass(1, &first, nullptr, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
    s.list4->EndRenderPass();
    s.list4->BeginRenderPass(1, &second, nullptr, D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
    s.list4->EndRenderPass();
    transition(s.list.p, rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    return s.copy_out(rt, kRgba, 0, 4);
}

// A 4-sample target cleared by the pass and resolved by its ending access into a 1-sample texture: into a region
// (source rectangle and destination offset), or the whole texture.
bool resolve(Scene& s, bool region) {
    ID3D12Resource* msaa = s.texture(kRgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                     D3D12_RESOURCE_STATE_RENDER_TARGET, 4, 0);
    ID3D12Resource* dst = s.texture(kRgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_RENDER_TARGET, 1, 1);
    CHECK(msaa && dst);
    s.list->ClearRenderTargetView(s.rtv(1), kBefore.data(), 0, nullptr);
    transition(s.list.p, dst, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS where{};
    if (region) {
        where.DstX = 1;
        where.DstY = 1;
        where.SrcRect = {0, 0, 3, 2};
    } else {
        where.SrcRect = {0, 0, LONG(kWidth), LONG(kHeight)};
    }
    D3D12_RENDER_PASS_ENDING_ACCESS last{};
    last.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE;
    last.Resolve.pSrcResource = msaa;
    last.Resolve.pDstResource = dst;
    last.Resolve.SubresourceCount = 1;
    last.Resolve.pSubresourceParameters = &where;
    last.Resolve.Format = kRgba;
    last.Resolve.ResolveMode = D3D12_RESOLVE_MODE_AVERAGE;
    last.Resolve.PreserveResolveSource = region ? FALSE : TRUE;
    const auto desc = target(s.rtv(0), begin_clear(kRgba, kCleared), last);
    s.list4->BeginRenderPass(1, &desc, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
    s.list4->EndRenderPass();
    transition(s.list.p, dst, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    return s.copy_out(dst, kRgba, 0, 4);
}
bool resolve_region(Scene& s) { return resolve(s, true); }
bool resolve_whole(Scene& s) { return resolve(s, false); }

struct DataCase {
    const char* name;
    bool (*record)(Scene&);
};
const DataCase kDataCases[] = {
    {"colour CLEAR + PRESERVE, depth CLEAR, stencil PRESERVE", depth_cleared},
    {"colour CLEAR + PRESERVE, depth PRESERVE, stencil CLEAR", stencil_cleared},
    {"two passes on one target", two_passes},
    {"SUSPENDING then RESUMING pass", suspend_resume},
    {"RESOLVE ending into a region", resolve_region},
    {"RESOLVE ending into the whole texture", resolve_whole},
};

// Close's result and every plane, or false when the device could not set the case up.
bool run_data(ID3D12Device* device, const DataCase& c, HRESULT& closed, std::vector<std::vector<unsigned char>>& out) {
    Scene s;
    CHECK(s.create(device));
    CHECK(c.record(s));
    closed = s.list->Close();
    if (FAILED(closed)) return true;  // nothing to run
    return s.execute(out);
}

// ---- close cases --------------------------------------------------------------------------------------------------

struct CloseScene {
    Scene s;
    ID3D12Resource* rt = nullptr;
    ID3D12Resource* other = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* buffer = nullptr;
    ID3D12Resource* buffer2 = nullptr;
    D3D12_RENDER_PASS_RENDER_TARGET_DESC desc{};

    ID3D12Resource* make_buffer() {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc_buffer{};
        desc_buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc_buffer.Width = 256;
        desc_buffer.Height = desc_buffer.SampleDesc.Count = 1;
        desc_buffer.DepthOrArraySize = desc_buffer.MipLevels = 1;
        desc_buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        s.textures.emplace_back();
        if (FAILED(s.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc_buffer,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                     IID_PPV_ARGS(&s.textures.back().p))))
            return nullptr;
        return s.textures.back().p;
    }
};

using Recorder = std::function<void(CloseScene&)>;
struct CloseCase {
    const char* name;
    Recorder record;
    bool bundle = false;
};

void pass(CloseScene& c, D3D12_RENDER_PASS_FLAGS flags = D3D12_RENDER_PASS_FLAG_NONE) {
    c.s.list4->BeginRenderPass(1, &c.desc, nullptr, flags);
}

const CloseCase kCloseCases[] = {
    {"Begin, End", [](CloseScene& c) { pass(c); c.s.list4->EndRenderPass(); }},
    {"Begin without End", [](CloseScene& c) { pass(c); }},
    {"End without Begin", [](CloseScene& c) { c.s.list4->EndRenderPass(); }},
    {"Begin, Begin, End", [](CloseScene& c) { pass(c); pass(c); c.s.list4->EndRenderPass(); }},
    {"Begin, End, End", [](CloseScene& c) { pass(c); c.s.list4->EndRenderPass(); c.s.list4->EndRenderPass(); }},
    {"Begin with 9 render targets", [](CloseScene& c) {
         D3D12_RENDER_PASS_RENDER_TARGET_DESC nine[9];
         for (auto& t : nine) t = c.desc;
         c.s.list4->BeginRenderPass(9, nine, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
         c.s.list4->EndRenderPass();
     }},
    {"Begin with no targets, End", [](CloseScene& c) {
         c.s.list4->BeginRenderPass(0, nullptr, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
         c.s.list4->EndRenderPass();
     }},
    {"OMSetRenderTargets inside a pass", [](CloseScene& c) {
         pass(c);
         const auto handle = c.s.rtv(0);
         c.s.list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
         c.s.list4->EndRenderPass();
     }},
    {"ClearRenderTargetView inside a pass", [](CloseScene& c) {
         pass(c);
         c.s.list->ClearRenderTargetView(c.s.rtv(0), kBefore.data(), 0, nullptr);
         c.s.list4->EndRenderPass();
     }},
    {"ResourceBarrier inside a pass", [](CloseScene& c) {
         pass(c);
         transition(c.s.list.p, c.other, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
         c.s.list4->EndRenderPass();
     }},
    {"CopyResource inside a pass", [](CloseScene& c) {
         pass(c);
         c.s.list->CopyResource(c.other, c.rt);
         c.s.list4->EndRenderPass();
     }},
    {"CopyTextureRegion inside a pass", [](CloseScene& c) {
         pass(c);
         D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
         dst.pResource = c.other;
         src.pResource = c.rt;
         dst.Type = src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
         c.s.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
         c.s.list4->EndRenderPass();
     }},
    {"ClearDepthStencilView inside a pass", [](CloseScene& c) {
         pass(c);
         c.s.list->ClearDepthStencilView(c.s.dsv(), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
         c.s.list4->EndRenderPass();
     }},
    {"CopyBufferRegion inside a pass", [](CloseScene& c) {
         pass(c);
         c.s.list->CopyBufferRegion(c.buffer2, 0, c.buffer, 0, 256);
         c.s.list4->EndRenderPass();
     }},
    {"ResolveSubresourceRegion inside a pass", [](CloseScene& c) {
         pass(c);
         ID3D12GraphicsCommandList1* list1 = nullptr;
         if (SUCCEEDED(c.s.list->QueryInterface(IID_PPV_ARGS(&list1)))) {
             list1->ResolveSubresourceRegion(c.other, 0, 0, 0, c.rt, 0, nullptr, kRgba, D3D12_RESOLVE_MODE_AVERAGE);
             list1->Release();
         }
         c.s.list4->EndRenderPass();
     }},
    {"DiscardResource inside a pass", [](CloseScene& c) {
         pass(c);
         c.s.list->DiscardResource(c.rt, nullptr);
         c.s.list4->EndRenderPass();
     }},
    {"ResolveSubresource inside a pass", [](CloseScene& c) {
         pass(c);
         c.s.list->ResolveSubresource(c.other, 0, c.rt, 0, kRgba);
         c.s.list4->EndRenderPass();
     }},
    {"ResourceBarrier between SUSPENDING and RESUMING", [](CloseScene& c) {
         pass(c, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
         c.s.list4->EndRenderPass();
         transition(c.s.list.p, c.other, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
         pass(c, D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
         c.s.list4->EndRenderPass();
     }},
    {"SUSPENDING, then a pass that does not resume", [](CloseScene& c) {
         pass(c, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
         c.s.list4->EndRenderPass();
         pass(c);
         c.s.list4->EndRenderPass();
     }},
    {"SUSPENDING, then OMSetRenderTargets", [](CloseScene& c) {
         pass(c, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
         c.s.list4->EndRenderPass();
         const auto handle = c.s.rtv(0);
         c.s.list->OMSetRenderTargets(1, &handle, FALSE, nullptr);
     }},
    {"a copy after a pass ends", [](CloseScene& c) {
         pass(c);
         c.s.list4->EndRenderPass();
         c.s.list->CopyResource(c.other, c.rt);
     }},
    {"Begin, End, Reset-free second pass with RESUMING and SUSPENDING", [](CloseScene& c) {
         pass(c, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
         c.s.list4->EndRenderPass();
         pass(c, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS | D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
         c.s.list4->EndRenderPass();
         pass(c, D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
         c.s.list4->EndRenderPass();
     }},
    {"RESUMING with no suspended pass", [](CloseScene& c) {
         pass(c, D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
         c.s.list4->EndRenderPass();
     }},
    {"SUSPENDING and never resumed", [](CloseScene& c) {
         pass(c, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
         c.s.list4->EndRenderPass();
     }},
    {"SUSPENDING, a copy, RESUMING", [](CloseScene& c) {
         pass(c, D3D12_RENDER_PASS_FLAG_SUSPENDING_PASS);
         c.s.list4->EndRenderPass();
         c.s.list->CopyResource(c.other, c.rt);
         pass(c, D3D12_RENDER_PASS_FLAG_RESUMING_PASS);
         c.s.list4->EndRenderPass();
     }},
    {"Begin, End in a bundle", [](CloseScene& c) { pass(c); c.s.list4->EndRenderPass(); }, true},
};

HRESULT run_close(ID3D12Device* device, const CloseCase& c) {
    CloseScene scene;
    const auto type = c.bundle ? D3D12_COMMAND_LIST_TYPE_BUNDLE : D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (!scene.s.create(device, type)) return E_UNEXPECTED;
    scene.rt = scene.s.texture(kRgba, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_STATE_RENDER_TARGET, 1, 0);
    scene.other = scene.s.texture(kRgba, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
    scene.depth = scene.s.texture(kDepth, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, D3D12_RESOURCE_STATE_DEPTH_WRITE);
    scene.buffer = scene.make_buffer();
    scene.buffer2 = scene.make_buffer();
    if (!scene.rt || !scene.other || !scene.depth || !scene.buffer || !scene.buffer2) return E_UNEXPECTED;
    scene.desc = target(scene.s.rtv(0), begin_clear(kRgba, kCleared), kKeep);
    c.record(scene);
    return scene.s.list->Close();
}

void report(const char* name, std::size_t plane, const std::vector<unsigned char>& actual,
            const std::vector<unsigned char>& expected) {
    for (std::size_t b = 0; b < actual.size() && b < expected.size(); ++b) {
        if (actual[b] != expected[b]) {
            std::printf("DIFF  %s: plane %zu byte %zu: layer %02x, WARP %02x\n", name, plane, b, unsigned(actual[b]),
                        unsigned(expected[b]));
            return;
        }
    }
    std::printf("DIFF  %s: plane %zu sizes %zu and %zu\n", name, plane, actual.size(), expected.size());
}
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // every line reaches the log, even if a case crashes the process
    if (hostwine::host_is_wine()) return 77;  // the reference is Microsoft's runtime over WARP
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    const unsigned errors_before = vk.validation_errors();
    unsigned matched = 0, differ = 0;
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
            std::fprintf(stderr, "WARP reference unavailable: differential test cannot run\n");
            return 1;
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
        if (SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options))))
            std::printf("WARP RenderPassesTier %d\n", int(options.RenderPassesTier));

        for (const DataCase& c : kDataCases) {
            HRESULT want_close = E_UNEXPECTED, got_close = E_UNEXPECTED;
            std::vector<std::vector<unsigned char>> expected, actual;
            if (!run_data(host.p, c, want_close, expected)) {
                std::printf("SKIP  %s: WARP could not set it up\n", c.name);
                continue;
            }
            const bool ran = run_data(ours.p, c, got_close, actual);
            bool same = ran && SUCCEEDED(want_close) == SUCCEEDED(got_close) && actual.size() == expected.size();
            if (!ran) std::printf("FAIL  %s: the layer could not set it up\n", c.name);
            else if (SUCCEEDED(want_close) != SUCCEEDED(got_close))
                std::printf("DIFF  %s: Close layer 0x%08lx, WARP 0x%08lx\n", c.name, unsigned long(got_close),
                            unsigned long(want_close));
            for (std::size_t p = 0; same && p < expected.size(); ++p) {
                if (actual[p] != expected[p]) {
                    report(c.name, p, actual[p], expected[p]);
                    same = false;
                }
            }
            if (same) {
                ++matched;
                std::printf("ok    %s (%zu planes)\n", c.name, expected.size());
            } else {
                ++differ;
            }
        }
        for (const CloseCase& c : kCloseCases) {
            const HRESULT want = run_close(host.p, c);
            const HRESULT got = run_close(ours.p, c);
            const bool same = SUCCEEDED(want) == SUCCEEDED(got);
            std::printf("%s  Close after %s: layer 0x%08lx, WARP 0x%08lx\n", same ? "ok  " : "DIFF", c.name,
                        unsigned long(got), unsigned long(want));
            if (same) ++matched;
            else ++differ;
        }
    }  // Destroy the D3D12 objects before checking the Vulkan validation counter.
    std::printf("render passes: %u matched WARP, %u differ\n", matched, differ);
    if (differ) return 1;
    if (vk.validation_errors() != errors_before) return 1;
    return 0;
}
