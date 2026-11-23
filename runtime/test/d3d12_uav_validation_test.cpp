// CreateUnorderedAccessView validation for buffers and textures: checks that invalid flags, stride, element
// counts, typeless/sRGB/depth-stencil/BC formats, dimension mismatches, and out-of-range subresource slices leave
// the descriptor slot zeroed without leaking bindless entries, while valid views still encode and clear on the GPU.
// Synthetic inputs only; no shaders or game data.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../src/gfx10_descriptors.h"

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace g = leonida::gfx10;

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

struct Vsharp {
    std::uint64_t address = 0;
    std::uint32_t stride = 0, records = 0, format = 0, oob = 0;
};

Vsharp decode_vsharp(const void* ptr, unsigned dword_offset = 0) {
    std::uint32_t w[4]{};
    std::memcpy(w, static_cast<const std::uint32_t*>(ptr) + dword_offset, sizeof(w));
    return {
        (std::uint64_t(w[1] & 0xffffu) << 32) | w[0],
        (w[1] >> 16) & 0x3fffu,
        w[2],
        (w[3] >> 12) & 0x7fu,
        (w[3] >> 28) & 3u,
    };
}

bool is_null_slot(const void* ptr, UINT step) {
    const auto* bytes = static_cast<const std::uint8_t*>(ptr);
    for (UINT i = 0; i < step; ++i)
        if (bytes[i] != 0) return false;
    return true;
}

bool decode_tsharp(const void* ptr, g::Image& out, std::uint32_t (&raw)[8]) {
    std::memcpy(raw, ptr, sizeof(raw));
    g::ImageWords words{};
    std::memcpy(words.data(), raw, sizeof(raw));
    return g::is_image(words) && g::decode(words, out);
}

D3D12_RESOURCE_DESC make_buffer_desc(UINT64 width, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = width;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = flags;
    return d;
}

D3D12_RESOURCE_DESC make_tex_desc(D3D12_RESOURCE_DIMENSION dim, DXGI_FORMAT fmt, UINT64 width, UINT height,
                                  UINT16 depth_or_array, UINT16 mips, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dim;
    d.Width = width;
    d.Height = dim == D3D12_RESOURCE_DIMENSION_TEXTURE1D ? 1u : height;
    d.DepthOrArraySize = depth_or_array;
    d.MipLevels = mips;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = flags;
    return d;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC make_buf_uav(DXGI_FORMAT fmt, UINT64 first, UINT count, UINT stride,
                                              UINT64 counter_offset = 0,
                                              D3D12_BUFFER_UAV_FLAGS flags = D3D12_BUFFER_UAV_FLAG_NONE) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format = fmt;
    d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer = {first, count, stride, counter_offset, flags};
    return d;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {res, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return b;
}

bool wait_fence(ID3D12Fence* fence, UINT64 value) {
    if (fence->GetCompletedValue() >= value) return true;
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const bool ok = SUCCEEDED(fence->SetEventOnCompletion(value, ev)) && WaitForSingleObject(ev, 10000) == WAIT_OBJECT_0;
    CloseHandle(ev);
    return ok;
}

}  // namespace

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.buffer_device_address || !vk.bindless) return 77;

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.bindless = vk.bindless;
    info.bindless_partially_bound = vk.partially_bound;
    info.image_view_min_lod = vk.image_view_min_lod;
    info.robust_image_access = vk.robust_image_access;
    info.sliced_3d_uav = vk.sliced_3d_uav;

    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
    EXPECT(rt::d3d12::bindless_set(device.p) != VK_NULL_HANDLE);
    VkPhysicalDeviceProperties props{};
    if (const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            vk.instance_proc(vk.instance, "vkGetPhysicalDeviceProperties")))
        get_props(vk.physical, &props);
    std::printf("device: %s (validation %s)\n", props.deviceName, vk.validation ? "on" : "off");

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    EXPECT(step == rt::d3d12::descriptor_size);

    D3D12_DESCRIPTOR_HEAP_DESC cpu_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 20,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC gpu_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 16,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&cpu_heap_desc, IID_PPV_ARGS(&cpu_heap.p))));
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&gpu_heap_desc, IID_PPV_ARGS(&gpu_heap.p))));

    const SIZE_T cpu_base = cpu_heap->GetCPUDescriptorHandleForHeapStart().ptr;
    const auto cpu_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{cpu_base + SIZE_T(i) * step}; };
    const auto slot_ptr = [&](UINT i) { return reinterpret_cast<void*>(cpu_slot(i).ptr); };
    const auto expect_rejected = [&](ID3D12Resource* res, ID3D12Resource* counter,
                                     const D3D12_UNORDERED_ACCESS_VIEW_DESC* desc, const char* label) -> bool {
        const auto room_before = rt::d3d12::bindless_room(device.p);
        std::memset(slot_ptr(0), 0xA5, step);
        device->CreateUnorderedAccessView(res, counter, desc, cpu_slot(0));
        if (!is_null_slot(slot_ptr(0), step)) {
            std::fprintf(stderr, "FAIL: expected null descriptor for invalid UAV (%s)\n", label);
            return false;
        }
        const auto room_after = rt::d3d12::bindless_room(device.p);
        if (room_after.images_2d != room_before.images_2d) {
            std::fprintf(stderr, "FAIL: invalid UAV (%s) consumed a bindless 2D image slot\n", label);
            return false;
        }
        return true;
    };

    // ---- Buffer UAV validation -------------------------------------------------------------------------------------
    Ref<ID3D12Resource> uav_buf, plain_buf, counter_buf;
    const auto uav_buf_desc = make_buffer_desc(4096, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto plain_buf_desc = make_buffer_desc(4096, D3D12_RESOURCE_FLAG_NONE);
    const auto counter_buf_desc = make_buffer_desc(8192, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &uav_buf_desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&uav_buf.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &plain_buf_desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                     IID_PPV_ARGS(&plain_buf.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &counter_buf_desc,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                     IID_PPV_ARGS(&counter_buf.p))));

    // Positive controls: structured (with and without counter), raw, typed.
    {
        const auto structured = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 64, 16);
        device->CreateUnorderedAccessView(uav_buf.p, nullptr, &structured, cpu_slot(1));
        const Vsharp v = decode_vsharp(slot_ptr(1));
        EXPECT(v.address == uav_buf->GetGPUVirtualAddress() && v.stride == 16 && v.records == 64 && v.oob == 0);

        const auto structured_ctr = make_buf_uav(DXGI_FORMAT_UNKNOWN, 1, 32, 16, 4096);
        device->CreateUnorderedAccessView(uav_buf.p, counter_buf.p, &structured_ctr, cpu_slot(2));
        const Vsharp vc = decode_vsharp(slot_ptr(2));
        const Vsharp ctr = decode_vsharp(slot_ptr(2), 4);
        EXPECT(vc.address == uav_buf->GetGPUVirtualAddress() + 16 && vc.stride == 16 && vc.records == 32);
        EXPECT(ctr.address == counter_buf->GetGPUVirtualAddress() + 4096 && ctr.stride == 0 && ctr.records == 4 &&
               ctr.oob == 3);

        const auto raw = make_buf_uav(DXGI_FORMAT_R32_TYPELESS, 4, 32, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW);
        device->CreateUnorderedAccessView(uav_buf.p, nullptr, &raw, cpu_slot(3));
        const Vsharp vr = decode_vsharp(slot_ptr(3));
        EXPECT(vr.address == uav_buf->GetGPUVirtualAddress() + 16 && vr.stride == 0 && vr.records == 128 &&
               vr.oob == 3);

        const auto typed = make_buf_uav(DXGI_FORMAT_R32_UINT, 2, 16, 0);
        device->CreateUnorderedAccessView(uav_buf.p, nullptr, &typed, cpu_slot(4));
        const Vsharp vt = decode_vsharp(slot_ptr(4));
        EXPECT(vt.address == uav_buf->GetGPUVirtualAddress() + 8 && vt.stride == 4 && vt.records == 16 &&
               vt.format == g::format_from_vulkan(VK_FORMAT_R32_UINT) && vt.oob == 1);
    }

    // Invalid buffer UAVs.
    {
        const auto bad_flag_struct = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 16, 0, D3D12_BUFFER_UAV_FLAGS(2));
        EXPECT(expect_rejected(uav_buf.p, nullptr, &bad_flag_struct, "buffer unknown flag bit on structured"));

        const auto bad_flag_raw = make_buf_uav(DXGI_FORMAT_R32_TYPELESS, 0, 16, 0, 0, D3D12_BUFFER_UAV_FLAGS(3));
        EXPECT(expect_rejected(uav_buf.p, nullptr, &bad_flag_raw, "buffer unknown flag bit on raw"));

        const auto bad_flag_typed = make_buf_uav(DXGI_FORMAT_R32_UINT, 0, 16, 0, 0, D3D12_BUFFER_UAV_FLAGS(0x80000000));
        EXPECT(expect_rejected(uav_buf.p, nullptr, &bad_flag_typed, "buffer high unknown flag bit on typed"));

        const auto raw_with_unknown = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 16, 0, D3D12_BUFFER_UAV_FLAG_RAW);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &raw_with_unknown, "buffer RAW flag with DXGI_FORMAT_UNKNOWN"));

        const auto raw_with_stride = make_buf_uav(DXGI_FORMAT_R32_TYPELESS, 0, 16, 4, 0, D3D12_BUFFER_UAV_FLAG_RAW);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &raw_with_stride, "buffer raw with non-zero StructureByteStride"));

        const auto typed_with_stride = make_buf_uav(DXGI_FORMAT_R32_UINT, 0, 16, 4);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &typed_with_stride, "buffer typed with non-zero StructureByteStride"));

        const auto raw_with_typed = make_buf_uav(DXGI_FORMAT_R32_UINT, 0, 16, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &raw_with_typed, "buffer RAW flag with typed format"));

        const auto zero_struct = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 0, 16);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &zero_struct, "buffer structured NumElements == 0"));

        const auto zero_raw = make_buf_uav(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &zero_raw, "buffer raw NumElements == 0"));

        const auto zero_typed = make_buf_uav(DXGI_FORMAT_R32_UINT, 0, 0, 0);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &zero_typed, "buffer typed NumElements == 0"));

        const auto zero_stride = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 0);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &zero_stride, "buffer structured StructureByteStride == 0"));

        const auto huge_stride = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 1, 0x4000);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &huge_stride, "buffer structured stride > 0x3fff"));

        const auto oob_elem = make_buf_uav(DXGI_FORMAT_R32_UINT, 1024, 1, 0);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &oob_elem, "buffer FirstElement + NumElements past end"));

        const auto wrap_elem = make_buf_uav(DXGI_FORMAT_R32_UINT, UINT64_MAX, 1, 0);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &wrap_elem, "buffer FirstElement UINT64_MAX wrap"));

        const auto valid_typed = make_buf_uav(DXGI_FORMAT_R32_UINT, 0, 16, 0);
        EXPECT(expect_rejected(plain_buf.p, nullptr, &valid_typed, "buffer without ALLOW_UNORDERED_ACCESS"));
        EXPECT(expect_rejected(uav_buf.p, nullptr, nullptr, "buffer with null desc"));

        D3D12_UNORDERED_ACCESS_VIEW_DESC wrong_dim{};
        wrong_dim.Format = DXGI_FORMAT_R32_UINT;
        wrong_dim.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        EXPECT(expect_rejected(uav_buf.p, nullptr, &wrong_dim, "buffer with TEXTURE2D ViewDimension"));

        // Counter validation edge cases.
        const auto ctr_on_raw = make_buf_uav(DXGI_FORMAT_R32_TYPELESS, 0, 16, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW);
        EXPECT(expect_rejected(uav_buf.p, counter_buf.p, &ctr_on_raw, "counter on raw buffer UAV"));
        EXPECT(expect_rejected(uav_buf.p, counter_buf.p, &valid_typed, "counter on typed buffer UAV"));

        const auto ctr_valid_desc = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 16, 0);
        EXPECT(expect_rejected(uav_buf.p, plain_buf.p, &ctr_valid_desc, "counter buffer without ALLOW_UNORDERED_ACCESS"));

        const auto ctr_misaligned = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 16, 4);
        EXPECT(expect_rejected(uav_buf.p, counter_buf.p, &ctr_misaligned, "counter offset not 4096-aligned"));

        const auto ctr_oob = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 16, 8192);
        EXPECT(expect_rejected(uav_buf.p, counter_buf.p, &ctr_oob, "counter offset at end of buffer"));

        const auto ctr_offset_no_res = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 16, 4096);
        EXPECT(expect_rejected(uav_buf.p, nullptr, &ctr_offset_no_res, "non-zero CounterOffsetInBytes with null counter"));
    }

    // ---- Texture UAV validation ------------------------------------------------------------------------------------
    Ref<ID3D12Resource> tex1d, tex1d_arr, tex2d, tex2d_arr, tex2d_typeless, tex2d_bgra_typeless, tex2d_r32_typeless,
        tex2d_r9g9b9e5, tex2d_no_uav, tex3d, tex2d_depth;
    const auto create_tex = [&](Ref<ID3D12Resource>& out, D3D12_RESOURCE_DIMENSION dim, DXGI_FORMAT fmt, UINT64 w,
                                UINT h, UINT16 depth_or_arr, UINT16 mips, D3D12_RESOURCE_FLAGS flags) {
        const auto d = make_tex_desc(dim, fmt, w, h, depth_or_arr, mips, flags);
        return SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &d,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&out.p)));
    };
    EXPECT(create_tex(tex1d, D3D12_RESOURCE_DIMENSION_TEXTURE1D, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 1, 1, 2,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex1d_arr, D3D12_RESOURCE_DIMENSION_TEXTURE1D, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 1, 2, 2,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 3,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d_arr, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16, 4, 3,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d_typeless, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_TYPELESS, 16, 16, 1, 2,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d_bgra_typeless, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_B8G8R8A8_TYPELESS, 16, 16,
                      1, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d_r32_typeless, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R32_TYPELESS, 16, 16, 1, 1,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d_r9g9b9e5, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 16, 16, 1, 1,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d_no_uav, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, 1,
                      D3D12_RESOURCE_FLAG_NONE));
    EXPECT(create_tex(tex3d, D3D12_RESOURCE_DIMENSION_TEXTURE3D, DXGI_FORMAT_R8G8B8A8_UNORM, 8, 8, 4, 2,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
    EXPECT(create_tex(tex2d_depth, D3D12_RESOURCE_DIMENSION_TEXTURE2D, DXGI_FORMAT_D32_FLOAT, 16, 16, 1, 1,
                      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));

    // Positive controls for texture UAVs.
    {
        g::Image img{};
        std::uint32_t raw[8]{};

        // Default desc == nullptr on concrete 1D, 1DArray, 2D, 2DArray, 3D, and R9G9B9E5.
        device->CreateUnorderedAccessView(tex1d.p, nullptr, nullptr, cpu_slot(5));
        EXPECT(decode_tsharp(slot_ptr(5), img, raw) && img.type == g::ImageType::image_1d && img.first_mip == 0 &&
               img.mip_count == 1 && img.layer_count == 1);
        if (vk.robust_image_access) EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        device->CreateUnorderedAccessView(tex1d_arr.p, nullptr, nullptr, cpu_slot(15));
        EXPECT(decode_tsharp(slot_ptr(15), img, raw) && img.type == g::ImageType::image_1d_array &&
               img.first_mip == 0 && img.mip_count == 1 && img.first_layer == 0 && img.layer_count == 2);
        if (vk.robust_image_access) EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        device->CreateUnorderedAccessView(tex3d.p, nullptr, nullptr, cpu_slot(16));
        EXPECT(decode_tsharp(slot_ptr(16), img, raw) && img.type == g::ImageType::image_3d && img.first_mip == 0 &&
               img.mip_count == 1 && img.depth == 4);
        if (vk.robust_image_access) EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        device->CreateUnorderedAccessView(tex2d.p, nullptr, nullptr, cpu_slot(6));
        EXPECT(decode_tsharp(slot_ptr(6), img, raw) && img.type == g::ImageType::image_2d && img.first_mip == 0 &&
               img.mip_count == 1 && img.layer_count == 1);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        device->CreateUnorderedAccessView(tex2d_arr.p, nullptr, nullptr, cpu_slot(7));
        EXPECT(decode_tsharp(slot_ptr(7), img, raw) && img.type == g::ImageType::image_2d_array && img.first_mip == 0 &&
               img.mip_count == 1 && img.first_layer == 0 && img.layer_count == 4);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        device->CreateUnorderedAccessView(tex2d_r9g9b9e5.p, nullptr, nullptr, cpu_slot(8));
        EXPECT(decode_tsharp(slot_ptr(8), img, raw) && img.type == g::ImageType::image_2d);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        // Explicit TEXTURE1D on 1D array (slice 0, mip 1) preserves d3d12_texture1d_test behaviour.
        D3D12_UNORDERED_ACCESS_VIEW_DESC u1d{};
        u1d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        u1d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        u1d.Texture1D.MipSlice = 1;
        device->CreateUnorderedAccessView(tex1d_arr.p, nullptr, &u1d, cpu_slot(9));
        EXPECT(decode_tsharp(slot_ptr(9), img, raw) && img.type == g::ImageType::image_1d && img.first_mip == 1 &&
               img.layer_count == 1);

        // Explicit TEXTURE2DARRAY with subrange and with ArraySize == UINT_MAX (~0u).
        D3D12_UNORDERED_ACCESS_VIEW_DESC u2da{};
        u2da.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        u2da.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        u2da.Texture2DArray = {1, 1, 2, 0};
        device->CreateUnorderedAccessView(tex2d_arr.p, nullptr, &u2da, cpu_slot(10));
        EXPECT(decode_tsharp(slot_ptr(10), img, raw) && img.type == g::ImageType::image_2d_array &&
               img.first_mip == 1 && img.first_layer == 1 && img.layer_count == 2);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        u2da.Texture2DArray = {0, 2, UINT_MAX, 0};
        device->CreateUnorderedAccessView(tex2d_arr.p, nullptr, &u2da, cpu_slot(11));
        EXPECT(decode_tsharp(slot_ptr(11), img, raw) && img.type == g::ImageType::image_2d_array &&
               img.first_layer == 2 && img.layer_count == 2);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);

        // Concrete typed UAV on a TYPELESS texture.
        D3D12_UNORDERED_ACCESS_VIEW_DESC u_typed{};
        u_typed.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        u_typed.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        u_typed.Texture2D = {0, 0};
        device->CreateUnorderedAccessView(tex2d_typeless.p, nullptr, &u_typed, cpu_slot(12));
        EXPECT(decode_tsharp(slot_ptr(12), img, raw) && img.type == g::ImageType::image_2d &&
               img.format == g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UINT));
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, raw) != 0);
    }

    // Invalid texture UAVs.
    {
        // Null desc on TYPELESS texture must be rejected (D3D12 requires an explicit concrete format).
        EXPECT(expect_rejected(tex2d_typeless.p, nullptr, nullptr, "null desc on R8G8B8A8_TYPELESS texture"));
        EXPECT(expect_rejected(tex2d_r32_typeless.p, nullptr, nullptr, "null desc on R32_TYPELESS texture"));

        // Explicit TYPELESS format in UAV desc must be rejected.
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        d.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
        EXPECT(expect_rejected(tex2d_typeless.p, nullptr, &d, "explicit TYPELESS format on typeless texture"));
        EXPECT(expect_rejected(tex2d.p, nullptr, &d, "explicit TYPELESS format on concrete texture"));

        // Explicit sRGB format in UAV desc must be rejected (no sRGB UAVs in D3D12 / no STORAGE_IMAGE_BIT in Vulkan).
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        EXPECT(expect_rejected(tex2d_typeless.p, nullptr, &d, "R8G8B8A8_UNORM_SRGB on R8G8B8A8_TYPELESS"));
        d.Format = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        EXPECT(expect_rejected(tex2d_bgra_typeless.p, nullptr, &d, "B8G8R8A8_UNORM_SRGB on B8G8R8A8_TYPELESS"));

        // Depth-stencil, block-compressed, and cross-family formats must be rejected.
        d.Format = DXGI_FORMAT_D32_FLOAT;
        EXPECT(expect_rejected(tex2d_r32_typeless.p, nullptr, &d, "D32_FLOAT view format on R32_TYPELESS UAV"));
        d.Format = DXGI_FORMAT_BC1_UNORM;
        EXPECT(expect_rejected(tex2d_typeless.p, nullptr, &d, "BC1_UNORM view format on R8G8B8A8_TYPELESS UAV"));
        d.Format = DXGI_FORMAT_R32_FLOAT;
        EXPECT(expect_rejected(tex2d.p, nullptr, &d, "cross-family R32_FLOAT on R8G8B8A8_UNORM without castable list"));
        EXPECT(expect_rejected(tex2d_depth.p, nullptr, nullptr, "depth-stencil resource without ALLOW_UNORDERED_ACCESS"));

        // Out-of-range MipSlice, PlaneSlice, FirstArraySlice, and ArraySize (must NOT silently clamp!).
        d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        d.Texture2D = {3, 0};  // tex2d has 3 mips (0..2)
        EXPECT(expect_rejected(tex2d.p, nullptr, &d, "TEXTURE2D MipSlice >= MipLevels"));

        d.Texture2D = {0, 1};
        EXPECT(expect_rejected(tex2d.p, nullptr, &d, "TEXTURE2D non-zero PlaneSlice on color texture"));

        d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        d.Texture1D = {2};  // tex1d has 2 mips (0..1)
        EXPECT(expect_rejected(tex1d.p, nullptr, &d, "TEXTURE1D MipSlice >= MipLevels"));

        d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {0, 1, 4, 0};  // 1 + 4 > 4 slices: must reject, not clamp
        EXPECT(expect_rejected(tex2d_arr.p, nullptr, &d, "TEXTURE2DARRAY FirstArraySlice + ArraySize > DepthOrArraySize"));

        d.Texture2DArray = {0, 3, 2, 0};  // 3 + 2 > 4 slices
        EXPECT(expect_rejected(tex2d_arr.p, nullptr, &d, "TEXTURE2DARRAY ArraySize past end of array"));

        d.Texture2DArray = {0, 0, 0, 0};
        EXPECT(expect_rejected(tex2d_arr.p, nullptr, &d, "TEXTURE2DARRAY ArraySize == 0"));

        d.Texture2DArray = {0, 4, 1, 0};
        EXPECT(expect_rejected(tex2d_arr.p, nullptr, &d, "TEXTURE2DARRAY FirstArraySlice >= DepthOrArraySize"));

        d.Texture2DArray = {3, 0, 1, 0};
        EXPECT(expect_rejected(tex2d_arr.p, nullptr, &d, "TEXTURE2DARRAY MipSlice >= MipLevels"));

        d.Texture2DArray = {0, 0, 1, 1};
        EXPECT(expect_rejected(tex2d_arr.p, nullptr, &d, "TEXTURE2DARRAY non-zero PlaneSlice on color texture"));

        // Dimension mismatches.
        d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        EXPECT(expect_rejected(tex2d.p, nullptr, &d, "TEXTURE1D view on TEXTURE2D resource"));

        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        EXPECT(expect_rejected(tex1d.p, nullptr, &d, "TEXTURE2D view on TEXTURE1D resource"));
        EXPECT(expect_rejected(tex3d.p, nullptr, &d, "TEXTURE2D view on TEXTURE3D resource"));

        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray = {0, 0, 1, 0};
        EXPECT(expect_rejected(tex1d.p, nullptr, &d, "TEXTURE2DARRAY view on TEXTURE1D resource"));

        d = make_buf_uav(DXGI_FORMAT_R8G8B8A8_UNORM, 0, 16, 0);
        EXPECT(expect_rejected(tex2d.p, nullptr, &d, "BUFFER view on TEXTURE2D resource"));

        d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION(99);
        EXPECT(expect_rejected(tex2d.p, nullptr, &d, "unknown ViewDimension enum"));

        // Out-of-range 1DArray / 3D UAV subranges and counter/flag rejections.
        d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
        d.Texture1DArray = {0, 0, 0};
        EXPECT(expect_rejected(tex1d_arr.p, nullptr, &d, "TEXTURE1DARRAY ArraySize == 0"));

        d.Texture1DArray = {0, 1, 2};
        EXPECT(expect_rejected(tex1d_arr.p, nullptr, &d, "TEXTURE1DARRAY FirstArraySlice + ArraySize > 2"));

        d = {};
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        d.Texture3D = {0, 0, 0};
        EXPECT(expect_rejected(tex3d.p, nullptr, &d, "TEXTURE3D WSize == 0"));

        d.Texture3D = {2, 0, 1};
        EXPECT(expect_rejected(tex3d.p, nullptr, &d, "TEXTURE3D MipSlice >= MipLevels"));

        d.Texture3D = {0, 4, 1};
        EXPECT(expect_rejected(tex3d.p, nullptr, &d, "TEXTURE3D FirstWSlice >= Depth"));

        EXPECT(expect_rejected(tex2d_no_uav.p, nullptr, nullptr, "texture without ALLOW_UNORDERED_ACCESS"));
        EXPECT(expect_rejected(tex2d.p, counter_buf.p, nullptr, "texture UAV with counter resource"));

        // Counter resource cannot be a texture even for a structured buffer UAV.
        const auto valid_struct = make_buf_uav(DXGI_FORMAT_UNKNOWN, 0, 16, 16, 0);
        EXPECT(expect_rejected(uav_buf.p, tex2d.p, &valid_struct, "buffer structured UAV with texture counter"));
    }

    // ---- GPU verification: valid buffer & typeless->typed texture UAV clears read back exact bytes -----------------
    {
        Ref<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        Ref<ID3D12CommandAllocator> alloc;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        const auto buf_view = make_buf_uav(DXGI_FORMAT_R32_UINT, 0, 16, 0);
        D3D12_UNORDERED_ACCESS_VIEW_DESC tex_view{};
        tex_view.Format = DXGI_FORMAT_R8G8B8A8_UINT;
        tex_view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        tex_view.Texture2D = {0, 0};

        const D3D12_CPU_DESCRIPTOR_HANDLE cpu_buf_h = cpu_slot(13);
        const D3D12_CPU_DESCRIPTOR_HANDLE cpu_tex_h = cpu_slot(14);
        const SIZE_T gpu_cpu_base = gpu_heap->GetCPUDescriptorHandleForHeapStart().ptr;
        const UINT64 gpu_base = gpu_heap->GetGPUDescriptorHandleForHeapStart().ptr;
        const D3D12_CPU_DESCRIPTOR_HANDLE gpu_buf_cpu{gpu_cpu_base + SIZE_T(0) * step};
        const D3D12_CPU_DESCRIPTOR_HANDLE gpu_tex_cpu{gpu_cpu_base + SIZE_T(1) * step};
        const D3D12_GPU_DESCRIPTOR_HANDLE gpu_buf_h{gpu_base + UINT64(0) * step};
        const D3D12_GPU_DESCRIPTOR_HANDLE gpu_tex_h{gpu_base + UINT64(1) * step};

        device->CreateUnorderedAccessView(uav_buf.p, nullptr, &buf_view, cpu_buf_h);
        device->CreateUnorderedAccessView(uav_buf.p, nullptr, &buf_view, gpu_buf_cpu);
        device->CreateUnorderedAccessView(tex2d_typeless.p, nullptr, &tex_view, cpu_tex_h);
        device->CreateUnorderedAccessView(tex2d_typeless.p, nullptr, &tex_view, gpu_tex_cpu);

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT64 tex_rb_bytes = 0;
        const D3D12_RESOURCE_DESC tdesc = tex2d_typeless->GetDesc();
        device->GetCopyableFootprints(&tdesc, 0, 1, 0, &fp, nullptr, nullptr, &tex_rb_bytes);

        Ref<ID3D12Resource> buf_rb, tex_rb;
        const auto buf_rb_desc = make_buffer_desc(64, D3D12_RESOURCE_FLAG_NONE);
        const auto tex_rb_desc = make_buffer_desc(tex_rb_bytes, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buf_rb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&buf_rb.p))));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &tex_rb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&tex_rb.p))));

        D3D12_RESOURCE_BARRIER to_uav[2] = {
            transition(uav_buf.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            transition(tex2d_typeless.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        };
        list->ResourceBarrier(2, to_uav);
        ID3D12DescriptorHeap* heaps[] = {gpu_heap.p};
        list->SetDescriptorHeaps(1, heaps);
        const UINT buf_val[4] = {0x13579BDFu, 0, 0, 0};
        const UINT tex_val[4] = {0x11u, 0x22u, 0x33u, 0x44u};
        list->ClearUnorderedAccessViewUint(gpu_buf_h, cpu_buf_h, uav_buf.p, buf_val, 0, nullptr);
        list->ClearUnorderedAccessViewUint(gpu_tex_h, cpu_tex_h, tex2d_typeless.p, tex_val, 0, nullptr);
        D3D12_RESOURCE_BARRIER to_src[2] = {
            transition(uav_buf.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            transition(tex2d_typeless.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        };
        list->ResourceBarrier(2, to_src);
        list->CopyBufferRegion(buf_rb.p, 0, uav_buf.p, 0, 64);
        D3D12_TEXTURE_COPY_LOCATION dst_loc{tex_rb.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst_loc.PlacedFootprint = fp;
        D3D12_TEXTURE_COPY_LOCATION src_loc{tex2d_typeless.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        list->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
        EXPECT(SUCCEEDED(list->Close()));

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(wait_fence(fence.p, 1));

        void* mapped = nullptr;
        EXPECT(SUCCEEDED(buf_rb->Map(0, nullptr, &mapped)) && mapped);
        const auto* words = static_cast<const std::uint32_t*>(mapped);
        for (UINT i = 0; i < 16; ++i) EXPECT(words[i] == 0x13579BDFu);
        buf_rb->Unmap(0, nullptr);

        mapped = nullptr;
        EXPECT(SUCCEEDED(tex_rb->Map(0, nullptr, &mapped)) && mapped);
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        for (UINT y = 0; y < 16; ++y) {
            const auto* row = base + SIZE_T(y) * fp.Footprint.RowPitch;
            for (UINT x = 0; x < 16; ++x) {
                EXPECT(row[x * 4 + 0] == 0x11u && row[x * 4 + 1] == 0x22u &&
                       row[x * 4 + 2] == 0x33u && row[x * 4 + 3] == 0x44u);
            }
        }
        tex_rb->Unmap(0, nullptr);
    }

    std::printf("d3d12_uav_validation_test: PASS (buffer & texture UAV validation + GPU clear readback)\n");
    return 0;
}
