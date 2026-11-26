// Exercise mapping changes independently of the format-admission oracle. All snapshots are ordinary
// row-major CopyTextureRegion readbacks, so a wrong tile grid in both mapping and CopyTiles is observable.
#include "d3d12_sparse_formats_fixture.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace sparse_test {
namespace {
using Microsoft::WRL::ComPtr;
using Coord = D3D12_TILED_RESOURCE_COORDINATE;
constexpr UINT tile_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
constexpr D3D12_TILE_REGION_SIZE one{1, FALSE, 0, 0, 0};
constexpr D3D12_TILE_REGION_SIZE column{2, TRUE, 1, 2, 1};

struct PageFormat {
    DXGI_FORMAT format;
    UINT width, height, bytes;  // texels per tile and bytes per texel
};

void alias(ID3D12GraphicsCommandList& list, ID3D12Resource* before, ID3D12Resource* after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    barrier.Aliasing = {before, after};
    list.ResourceBarrier(1, &barrier);
}

std::uint8_t payload(UINT page, UINT byte) {
    return static_cast<std::uint8_t>(11 + page * 53 + byte * 7 + (byte >> 7) * 13 + (byte >> 12));
}

bool mapping_case(ID3D12Device& device, bool native, PageFormat src, PageFormat dst) {
    // Each image has the same texel dimensions. R16 and RGBA32 therefore have different numbers
    // of tiles per row (2 and 8). Copy a column, including its second row, using each image's grid.
    constexpr UINT width = 512, height = 256;
    const Coord origin{}, src_top{1, 0, 0, 0}, src_bottom{1, 1, 0, 0};
    const Coord dst_top{width / dst.width - 1, 0, 0, 0}, dst_bottom{dst_top.X, 1, 0, 0}, witness{0, 1, 0, 0};
    const bool compatible = src.bytes == dst.bytes;
    ComPtr<ID3D12Resource> a, b, peer, upload, readback;
    const auto a_desc = texture_desc(src.format, width, height), b_desc = texture_desc(dst.format, width, height);
    if (FAILED(device.CreateReservedResource(&a_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a))) ||
        FAILED(device.CreateReservedResource(&b_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b))) ||
        FAILED(device.CreateReservedResource(&b_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&peer))))
        return false;
    const D3D12_HEAP_DESC heap_desc{4 * tile_bytes, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1}, 0, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES};
    ComPtr<ID3D12Heap> heap;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    // Snapshot indices name the resource state being checked. The initially-unmapped and unmapped
    // snapshots have independent slots with a nonzero sentinel.
    enum Snapshot : UINT { initial, inherited, recycled, disjoint, unmapped, remapped, snapshots };
    if (FAILED(device.CreateHeap(&heap_desc, IID_PPV_ARGS(&heap))) ||
        FAILED(device.CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))) ||
        FAILED(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        !linear_buffer(device, 4 * tile_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload) ||
        !linear_buffer(device, snapshots * tile_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback))
        return false;
    const D3D12_RANGE no_read{0, 0};
    void* data = nullptr;
    if (FAILED(upload->Map(0, &no_read, &data))) return false;
    for (UINT page = 0; page != 4; ++page)
        for (UINT i = 0; i != tile_bytes; ++i)
            static_cast<std::uint8_t*>(data)[page * tile_bytes + i] = payload(page, i);
    upload->Unmap(0, nullptr);
    if (FAILED(readback->Map(0, &no_read, &data))) return false;
    std::memset(data, 0xcd, snapshots * tile_bytes);
    readback->Unmap(0, nullptr);

    const auto map = [&](ID3D12Resource* image, const Coord& coordinate, UINT page) {
        queue->UpdateTileMappings(image, 1, &coordinate, &one, heap.Get(), 1, nullptr, &page, nullptr,
                                  D3D12_TILE_MAPPING_FLAG_NONE);
    };
    const auto write = [&](ID3D12Resource* image, const Coord& coordinate, UINT pattern) {
        transition(*list.Get(), image, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyTiles(image, &coordinate, &one, upload.Get(), pattern * tile_bytes,
                        D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
        transition(*list.Get(), image, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    };
    const auto read = [&](ID3D12Resource* image, PageFormat fmt, const Coord& coord, Snapshot slot) {
        D3D12_TEXTURE_COPY_LOCATION from{image}, to{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = {UINT64(slot) * tile_bytes, {fmt.format, fmt.width, fmt.height, 1, fmt.width * fmt.bytes}};
        const D3D12_BOX box{coord.X * fmt.width, coord.Y * fmt.height, 0, (coord.X + 1) * fmt.width,
                            (coord.Y + 1) * fmt.height, 1};
        transition(*list.Get(), image, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyTextureRegion(&to, 0, 0, 0, &from, &box);
        transition(*list.Get(), image, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    };
    const auto advance = [&] {
        return submit_and_wait(device, *queue.Get(), *list.Get(), !native) && SUCCEEDED(list->Reset(allocator.Get(), nullptr));
    };

    read(b.Get(), dst, dst_bottom, initial);
    if (!advance()) return false;
    map(a.Get(), origin, 0);
    map(a.Get(), src_top, 3);
    map(a.Get(), src_bottom, 1);
    map(peer.Get(), origin, 2);
    map(peer.Get(), witness, 1);
    queue->CopyTileMappings(b.Get(), &dst_top, a.Get(), &src_top, &column, D3D12_TILE_MAPPING_FLAG_NONE);
    write(peer.Get(), origin, 2);
    alias(*list.Get(), peer.Get(), a.Get());
    write(a.Get(), origin, 0);
    write(a.Get(), src_bottom, 1);
    alias(*list.Get(), a.Get(), b.Get());
    if (compatible) read(b.Get(), dst, dst_bottom, inherited);
    // For unlike shapes the physical page can be recycled. Its old nonzero interpretation is undefined.
    // Write the *entire* destination tile before reading the same-format peer mapped directly to page 1.
    write(b.Get(), dst_bottom, 3);
    alias(*list.Get(), b.Get(), peer.Get());
    read(peer.Get(), dst, witness, recycled);
    alias(*list.Get(), peer.Get(), a.Get());
    read(a.Get(), src, origin, disjoint);
    if (!advance()) return false;

    const auto null_range = D3D12_TILE_RANGE_FLAG_NULL;
    queue->UpdateTileMappings(b.Get(), 1, &dst_bottom, &one, nullptr, 1, &null_range, nullptr, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    read(b.Get(), dst, dst_bottom, unmapped);
    if (!advance()) return false;

    // Page 2 was initialized through peer. Queue a remap and its read behind an unsignalled fence.
    ComPtr<ID3D12Fence> gate, done;
    if (FAILED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))) ||
        FAILED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)))) return false;
    alias(*list.Get(), peer.Get(), b.Get());
    read(b.Get(), dst, dst_bottom, remapped);
    if (FAILED(list->Close()) || FAILED(queue->Wait(gate.Get(), 1))) return false;
    map(b.Get(), dst_bottom, 2);
    ID3D12CommandList* batch[] = {list.Get()};
    queue->ExecuteCommandLists(1, batch);
    const bool held = SUCCEEDED(queue->Signal(done.Get(), 1)) && done->GetCompletedValue() == 0;
    const HRESULT released = gate->Signal(1);
    if (FAILED(released)) return false;
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;
    const bool finished = SUCCEEDED(done->SetEventOnCompletion(1, event)) &&
                          WaitForSingleObject(event, 15000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!held || !finished || FAILED(device.GetDeviceRemovedReason())) return false;

    if (FAILED(readback->Map(0, nullptr, &data))) return false;
    const auto* observed = static_cast<const std::uint8_t*>(data);
    const std::array<int, snapshots> expected_pages{-1, 1, 3, 0, -1, 2};
    bool matched = true;
    for (UINT n = 0; n < snapshots && matched; ++n) {
        if (n == inherited && !compatible) continue;
        for (UINT i = 0; i != tile_bytes; ++i) {
            const auto expected = expected_pages[n] < 0 ? 0 : payload(UINT(expected_pages[n]), i);
            if (observed[n * tile_bytes + i] != expected) {
                std::fprintf(stderr, "sparse mapping formats %u -> %u, snapshot %u byte %u: expected %u, got %u\n",
                             UINT(src.format), UINT(dst.format), n, i, UINT(expected), UINT(observed[n * tile_bytes + i]));
                matched = false;
                break;
            }
        }
    }
    readback->Unmap(0, &no_read);
    return matched;
}
}  // namespace

bool test_mappings(ID3D12Device& device, bool native) {
    constexpr PageFormat r32{DXGI_FORMAT_R32_UINT, 128, 128, 4}, rgba8{DXGI_FORMAT_R8G8B8A8_UINT, 128, 128, 4};
    constexpr PageFormat r16{DXGI_FORMAT_R16_UINT, 256, 128, 2}, rg8{DXGI_FORMAT_R8G8_UNORM, 256, 128, 2};
    constexpr PageFormat rgba32{DXGI_FORMAT_R32G32B32A32_UINT, 64, 64, 16};
    const PageFormat cases[][2] = {{r32, r32}, {r32, rgba8}, {rgba8, r32}, {r16, rg8}, {rg8, r16}, {r16, rgba32}};
    for (const auto& pair : cases) {
        if (!mapping_case(device, native, pair[0], pair[1])) {
            std::fprintf(stderr, "sparse mapping failed: %s, DXGI %u -> %u\n",
                         native ? "WARP" : "Vulkan", UINT(pair[0].format), UINT(pair[1].format));
            return false;
        }
    }
    std::printf("sparse mappings (%s): 6/6 cases, including 4 typed alias directions and a cross-shape page remap\n",
                native ? "WARP" : "Vulkan");
    return true;
}
}  // namespace sparse_test
