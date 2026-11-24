// Slot-133 aliases share GPU contents and retain individual COM identities.
#pragma once
#include "d3d12_copy_compare.h"

namespace placeddepthtest {

inline bool check_alias_lifetime(ID3D12Device* device, ID3D12Device* xbox) {
    using copycmp::Ptr;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 256;
    desc.Height = 128;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    const auto allocation = device->GetResourceAllocationInfo(0, 1, &desc);
    if (!allocation.Alignment || allocation.SizeInBytes == UINT64_MAX || allocation.SizeInBytes <= allocation.Alignment)
        return false;
    struct Pages {
        void* ptr;
        ~Pages() { if (ptr) VirtualFree(ptr, 0, MEM_RELEASE); }
    } pages{VirtualAlloc(nullptr, SIZE_T(allocation.SizeInBytes * 2), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)};
    if (!pages.ptr) return false;
    const UINT64 address = reinterpret_cast<UINT64>(pages.ptr);
    auto xbox_desc = desc;
    xbox_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    xbox_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
    using Create = HRESULT(WINAPI*)(void*, UINT64, const D3D12_RESOURCE_DESC*, UINT,
                                    const D3D12_CLEAR_VALUE*, REFIID, void**);
    const auto create = reinterpret_cast<Create>((*reinterpret_cast<void***>(xbox))[133]);
    const auto place = [&](UINT64 va, const D3D12_RESOURCE_DESC& shape, Ptr<ID3D12Resource>& result,
                           const D3D12_CLEAR_VALUE* clear = nullptr) {
        return create(xbox, va, &shape, UINT_MAX, clear, IID_PPV_ARGS(&result.p));
    };
    Ptr<ID3D12Resource> first, alias, next, invalid;
    if (FAILED(place(address, xbox_desc, first)) || FAILED(place(address, xbox_desc, alias)) ||
        FAILED(place(address + allocation.SizeInBytes, xbox_desc, next))) return false;
    if (first.p == alias.p || alias->GetGPUVirtualAddress() != address ||
        next->GetGPUVirtualAddress() != address + allocation.SizeInBytes) return false;
    // Independent COM private data helps catch accidental reuse of the object itself as the alias.
    constexpr GUID cookie_key{0x650cf350, 0xa07e, 0x428d, {0xb4, 0x14, 0xdc, 0xaf, 0xc6, 0x97, 0xe8, 0x4b}};
    const UINT cookie = 12;
    UINT bytes = 0;
    if (FAILED(first->SetPrivateData(cookie_key, sizeof(cookie), &cookie)) ||
        alias->GetPrivateData(cookie_key, &bytes, nullptr) != DXGI_ERROR_NOT_FOUND) return false;
    auto smaller = xbox_desc;
    smaller.Width /= 2;
    D3D12_CLEAR_VALUE optimized{};
    optimized.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    optimized.DepthStencil = {1.0f, 0};
    if (place(address, xbox_desc, invalid, &optimized) != E_NOTIMPL || invalid.p) return false;
    if (place(address + allocation.Alignment, xbox_desc, invalid) != E_NOTIMPL || invalid.p ||
        place(address + 1, xbox_desc, invalid) != E_INVALIDARG || invalid.p) return false;

    Ptr<ID3D12DescriptorHeap> heap;
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p)))) return false;
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
    dsv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(first.p, &dsv, handle);
    copycmp::Run commands;
    if (!commands.init(device, true)) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layouts[2]{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 2, 0, layouts, nullptr, nullptr, &total);
    Ptr<ID3D12Resource> readback;
    if (!commands.buffer(D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_STATE_COPY_DEST, readback)) return false;
    commands.begin();
    commands.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                                         0.375f, 0xb7, 0, nullptr);
    D3D12_RESOURCE_BARRIER activate{};
    activate.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    activate.Aliasing = {first.p, alias.p};
    commands.list->ResourceBarrier(1, &activate);
    if (!commands.submit().empty()) return false;
    first.reset();
    commands.begin();
    commands.barrier(alias.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT plane = 0; plane < 2; ++plane) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = alias.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = plane;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = layouts[plane];
        commands.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (!commands.submit().empty()) return false;
    unsigned char* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return false;
    bool valid = true;
    for (UINT y = 0; y < desc.Height && valid; ++y) {
        for (UINT x = 0; x < desc.Width; ++x) {
            float depth;
            std::memcpy(&depth, mapped + layouts[0].Offset + y * layouts[0].Footprint.RowPitch + x * sizeof(depth),
                        sizeof(depth));
            if (depth != 0.375f || mapped[layouts[1].Offset + y * layouts[1].Footprint.RowPitch + x] != 0xb7) {
                std::fprintf(stderr, "slot-133 alias: wrong readback at (%u, %u)\n", x, y);
                valid = false;
                break;
            }
        }
    }
    readback->Unmap(0, nullptr);
    alias.reset();
    return valid && SUCCEEDED(place(address, smaller, invalid)) && invalid->GetGPUVirtualAddress() == address;
}

inline bool check_same_base_shapes(ID3D12Device* device, ID3D12Device* xbox) {
    using copycmp::Ptr;
    D3D12_RESOURCE_DESC small_desc{};
    small_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    small_desc.Width = 256;
    small_desc.Height = 128;
    small_desc.DepthOrArraySize = small_desc.MipLevels = 1;
    small_desc.SampleDesc.Count = 1;
    small_desc.Format = DXGI_FORMAT_R16_TYPELESS;
    small_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    auto large_desc = small_desc;
    large_desc.Width = 384;
    large_desc.Height = 256;
    large_desc.DepthOrArraySize = 3;
    const auto small_size = device->GetResourceAllocationInfo(0, 1, &small_desc);
    const auto large_size = device->GetResourceAllocationInfo(0, 1, &large_desc);
    if (!small_size.Alignment || small_size.Alignment != large_size.Alignment ||
        !small_size.SizeInBytes || large_size.SizeInBytes == UINT64_MAX ||
        large_size.SizeInBytes <= small_size.SizeInBytes) return false;
    struct Pages {
        void* address;
        ~Pages() { if (address) VirtualFree(address, 0, MEM_RELEASE); }
    } pages{VirtualAlloc(nullptr, SIZE_T(small_size.Alignment + large_size.SizeInBytes + small_size.SizeInBytes),
                         MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)};
    if (!pages.address) return false;
    const UINT64 base_address = reinterpret_cast<UINT64>(pages.address) + small_size.Alignment;
    using Create = HRESULT(WINAPI*)(void*, UINT64, const D3D12_RESOURCE_DESC*, D3D12_BARRIER_LAYOUT,
                                    const D3D12_CLEAR_VALUE*, REFIID, void**);
    const auto create = reinterpret_cast<Create>((*reinterpret_cast<void***>(xbox))[133]);
    const auto place = [&](UINT64 address, D3D12_RESOURCE_DESC desc, Ptr<ID3D12Resource>& out) {
        desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        return create(xbox, address, &desc, D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&out.p));
    };
    Ptr<ID3D12Resource> first, first_reader, second, second_reader, adjacent, refused;
    if (FAILED(place(base_address, small_desc, first)) ||
        FAILED(place(base_address + small_size.SizeInBytes, small_desc, adjacent)) ||
        place(base_address, large_desc, refused) != E_NOTIMPL || refused.p) return false;
    adjacent.reset();
    if (FAILED(place(base_address, large_desc, second)) || FAILED(place(base_address, large_desc, second_reader)) ||
        FAILED(place(base_address, small_desc, first_reader)) ||
        first.p == first_reader.p || second.p == second_reader.p || first.p == second.p ||
        first->GetGPUVirtualAddress() != base_address || second->GetGPUVirtualAddress() != base_address ||
        first->GetDesc().Width != small_desc.Width || second->GetDesc().Width != large_desc.Width ||
        second->GetDesc().DepthOrArraySize != large_desc.DepthOrArraySize ||
        place(base_address + small_size.SizeInBytes, small_desc, refused) != E_NOTIMPL || refused.p ||
        place(base_address - small_size.Alignment, large_desc, refused) != E_NOTIMPL || refused.p ||
        FAILED(place(base_address + large_size.SizeInBytes, small_desc, adjacent))) return false;

    copycmp::Run commands;
    if (!commands.init(device, true)) return false;
    const auto initialize_and_check = [&](ID3D12Resource* before, ID3D12Resource* writer, ID3D12Resource* reader,
                                          const D3D12_RESOURCE_DESC& desc, D3D12_RESOURCE_STATES previous,
                                          UINT16 seed, UINT16 updated) {
        const UINT layers = desc.DepthOrArraySize;
        const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, layers};
        const D3D12_DESCRIPTOR_HEAP_DESC uav_cpu_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1};
        auto uav_gpu_desc = uav_cpu_desc;
        uav_gpu_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Ptr<ID3D12DescriptorHeap> dsv, uav_cpu, uav_gpu;
        if (FAILED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv.p))) ||
            FAILED(device->CreateDescriptorHeap(&uav_cpu_desc, IID_PPV_ARGS(&uav_cpu.p))) ||
            FAILED(device->CreateDescriptorHeap(&uav_gpu_desc, IID_PPV_ARGS(&uav_gpu.p)))) return false;
        const auto dsv_start = dsv->GetCPUDescriptorHandleForHeapStart().ptr;
        const auto dsv_step = device->GetDescriptorHandleIncrementSize(dsv_desc.Type);
        const auto uav = uav_cpu->GetCPUDescriptorHandleForHeapStart();
        D3D12_UNORDERED_ACCESS_VIEW_DESC writable{};
        writable.Format = DXGI_FORMAT_R16_UINT;
        writable.ViewDimension = layers == 1 ? D3D12_UAV_DIMENSION_TEXTURE2D : D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        if (layers > 1) writable.Texture2DArray = {0, 0, layers, 0};
        device->CreateUnorderedAccessView(writer, nullptr, &writable, uav);
        device->CopyDescriptorsSimple(1, uav_gpu->GetCPUDescriptorHandleForHeapStart(), uav, uav_cpu_desc.Type);

        std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> footprints(layers);
        UINT64 bytes = 0;
        device->GetCopyableFootprints(&desc, 0, layers, 0, footprints.data(), nullptr, nullptr, &bytes);
        Ptr<ID3D12Resource> readback;
        if (!bytes || !commands.buffer(D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_STATE_COPY_DEST, readback))
            return false;
        commands.begin();
        D3D12_RESOURCE_BARRIER activate{};
        activate.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        activate.Aliasing = {before, writer};
        commands.list->ResourceBarrier(1, &activate);
        commands.barrier(writer, previous, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        for (UINT layer = 0; layer < layers; ++layer) {
            D3D12_DEPTH_STENCIL_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_D16_UNORM;
            view.ViewDimension = layers == 1 ? D3D12_DSV_DIMENSION_TEXTURE2D : D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            if (layers > 1) view.Texture2DArray = {0, layer, 1};
            const D3D12_CPU_DESCRIPTOR_HANDLE handle{dsv_start + layer * dsv_step};
            device->CreateDepthStencilView(writer, &view, handle);
            commands.list->ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH, float(seed) / 65535.0f, 0, 0, nullptr);
        }
        commands.barrier(writer, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* visible[] = {uav_gpu.p};
        commands.list->SetDescriptorHeaps(1, visible);
        const UINT values[] = {updated, 0, 0, 0};
        const D3D12_RECT rect{1, 1, 2, 2};
        commands.list->ClearUnorderedAccessViewUint(uav_gpu->GetGPUDescriptorHandleForHeapStart(), uav, writer,
                                                    values, 1, &rect);
        activate.Aliasing = {writer, reader};
        commands.list->ResourceBarrier(1, &activate);
        commands.barrier(reader, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        for (UINT layer = 0; layer < layers; ++layer) {
            const auto src = copycmp::sub_loc(reader, layer);
            const auto dst = copycmp::foot_loc(readback.p, footprints[layer]);
            commands.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        if (const auto error = commands.submit(); !error.empty()) {
            std::fprintf(stderr, "slot-133 different-shape alias: %s\n", error.c_str());
            return false;
        }
        unsigned char* mapped = nullptr;
        if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) || !mapped) return false;
        bool matched = true;
        for (const auto& footprint : footprints) {
            for (UINT y = 0; y < desc.Height && matched; ++y) {
                for (UINT x = 0; x < desc.Width; ++x) {
                    UINT16 actual = 0;
                    std::memcpy(&actual, mapped + footprint.Offset + y * footprint.Footprint.RowPitch + x * 2,
                                sizeof(actual));
                    const UINT16 expected = x == 1 && y == 1 ? updated : seed;
                    if (actual != expected) {
                        std::fprintf(stderr, "slot-133 different-shape alias: pixel (%u,%u) got %u, expected %u\n",
                                     x, y, unsigned(actual), unsigned(expected));
                        matched = false;
                        break;
                    }
                }
            }
        }
        readback->Unmap(0, nullptr);
        return matched;
    };
    if (!initialize_and_check(nullptr, first.p, first_reader.p, small_desc, D3D12_RESOURCE_STATE_COMMON, 257, 0x4567) ||
        !initialize_and_check(first_reader.p, second.p, second_reader.p, large_desc, D3D12_RESOURCE_STATE_COMMON, 514,
                              0x9234) ||
        !initialize_and_check(second_reader.p, first.p, first_reader.p, small_desc, D3D12_RESOURCE_STATE_COPY_SOURCE, 771,
                              0x789a)) return false;
    std::printf("slot-133 same-base depth/UAV shapes: switches, independent backing and exact-alias readback passed\n");
    return true;
}

} // namespace placeddepthtest
