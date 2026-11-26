// Independent tile-major and row-major transfers through a single-mip, four-tile reserved texture.
#include "d3d12_sparse_formats_fixture.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace sparse_test {
namespace {
using Microsoft::WRL::ComPtr;
using Bytes = std::vector<std::uint8_t>;
constexpr UINT tile_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
constexpr std::uint8_t sentinel = 0xed;

std::uint8_t sample(UINT x, UINT y, UINT lane, UINT seed) {
    // Avoid both the padding sentinel and zero, and distinguish rows, tiles, and bytes within each texel.
    return static_cast<std::uint8_t>(1 + (x * 37 + y * 71 + x * y * 3 + lane * 17 + seed) % 211);
}

bool run_transfer(ID3D12Device& device, bool layer, const FormatCase& c, D3D12_COMMAND_LIST_TYPE type,
                  UINT tile_offset, Bytes& visible) {
    const auto desc = texture_desc(c.format, c.width, c.height);
    ComPtr<ID3D12Resource> image;
    if (FAILED(device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&image))) ||
        (layer && !sparse_image(image.Get()))) return false;
    const D3D12_HEAP_DESC heap_desc{4 * tile_bytes,
        {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1},
        0, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES};
    ComPtr<ID3D12Heap> heap;
    if (FAILED(device.CreateHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return false;

    // Use a padded ordinary footprint even on exact tile multiples. A tile slot always occupies 64 KiB.
    constexpr UINT row_offset = 4 * tile_bytes + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    constexpr UINT alignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    const UINT row_pitch = (c.width * c.bytes / alignment + 1) * alignment;
    const UINT buffer_bytes = row_offset + row_pitch * c.height + alignment;
    const UINT tile_pitch = c.tile_width * c.bytes;
    Bytes input(buffer_bytes, 0x94), expected(buffer_bytes, sentinel);
    const auto tiled = [&](UINT x, UINT y) {
        const UINT tile = (y / c.tile_height) * 2 + x / c.tile_width;
        return tile_offset + tile * tile_bytes + (y % c.tile_height) * tile_pitch + (x % c.tile_width) * c.bytes;
    };
    for (UINT y = 0; y < c.height; ++y) {
        for (UINT x = 0; x < c.width; ++x) {
            const UINT row = row_offset + y * row_pitch + x * c.bytes;
            const UINT slot = tiled(x, y);
            for (UINT b = 0; b < c.bytes; ++b) {
                input[slot + b] = expected[row + b] = sample(x, y, b, 13);
                input[row + b] = expected[slot + b] = sample(x, y, b, 97);
            }
        }
    }

    ComPtr<ID3D12Resource> upload, readback;
    if (!linear_buffer(device, buffer_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, upload) ||
        !linear_buffer(device, buffer_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, readback))
        return false;
    const D3D12_RANGE empty{};
    void* mapped = nullptr;
    if (FAILED(upload->Map(0, &empty, &mapped))) return false;
    std::memcpy(mapped, input.data(), input.size());
    upload->Unmap(0, nullptr);
    if (FAILED(readback->Map(0, &empty, &mapped))) return false;
    std::memset(mapped, sentinel, buffer_bytes);
    readback->Unmap(0, nullptr);

    ComPtr<ID3D12CommandQueue> direct, queue;
    const D3D12_COMMAND_QUEUE_DESC direct_desc{};
    if (FAILED(device.CreateCommandQueue(&direct_desc, IID_PPV_ARGS(&direct)))) return false;
    const D3D12_TILED_RESOURCE_COORDINATE origin{};
    const D3D12_TILE_REGION_SIZE all{4, TRUE, 2, 2, 1};
    const UINT first_page = 0, page_count = 4;
    direct->UpdateTileMappings(image.Get(), 1, &origin, &all, heap.Get(), 1, nullptr, &first_page, &page_count,
                               D3D12_TILE_MAPPING_FLAG_NONE);
    ComPtr<ID3D12Fence> handoff;
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT) {
        const D3D12_COMMAND_QUEUE_DESC qdesc{type};
        if (FAILED(device.CreateCommandQueue(&qdesc, IID_PPV_ARGS(&queue))) ||
            FAILED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&handoff))) ||
            FAILED(direct->Signal(handoff.Get(), 1)) || FAILED(queue->Wait(handoff.Get(), 1))) return false;
    } else {
        queue = direct;
    }
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    if (FAILED(device.CreateCommandAllocator(type, IID_PPV_ARGS(&allocator))) ||
        FAILED(device.CreateCommandList(0, type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)))) return false;
    D3D12_TEXTURE_COPY_LOCATION texture{image.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    D3D12_TEXTURE_COPY_LOCATION rows{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    rows.PlacedFootprint = {row_offset, {c.format, c.width, c.height, 1, row_pitch}};
    transition(*list.Get(), image.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyTiles(image.Get(), &origin, &all, upload.Get(), tile_offset,
                    D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
    transition(*list.Get(), image.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyTextureRegion(&rows, 0, 0, 0, &texture, nullptr);
    transition(*list.Get(), image.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    rows.pResource = upload.Get();
    list->CopyTextureRegion(&texture, 0, 0, 0, &rows, nullptr);
    transition(*list.Get(), image.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyTiles(image.Get(), &origin, &all, readback.Get(), tile_offset,
                    D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER);
    if (!submit_and_wait(device, *queue.Get(), *list.Get(), layer)) return false;

    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    const auto* data = static_cast<const std::uint8_t*>(mapped);
    // Native drivers may copy the out-of-image part of a partial tile. Ignore just those WARP bytes.
    // Vulkan's clipped transfer must preserve every padding byte, including the prefix at an odd offset.
    if (!layer) {
        for (UINT tile = 0; tile < 4; ++tile) {
            const UINT x0 = (tile % 2) * c.tile_width;
            for (UINT y = 0; y < c.tile_height; ++y) {
                const UINT covered = (tile / 2) * c.tile_height + y < c.height ?
                    std::min(c.tile_width, c.width - x0) * c.bytes : 0;
                const UINT row = tile_offset + tile * tile_bytes + y * tile_pitch;
                std::copy(data + row + covered, data + row + tile_pitch, expected.begin() + row + covered);
            }
        }
    }
    const auto mismatch = std::mismatch(expected.begin(), expected.end(), data);
    const bool matches = mismatch.first == expected.end();
    if (!matches) {
        std::fprintf(stderr, "DXGI %u %ux%u %s queue type %u offset %u: byte %zu got %#x, expected %#x\n",
                     UINT(c.format), c.width, c.height, layer ? "Vulkan" : "WARP", UINT(type), tile_offset,
                     std::size_t(mismatch.first - expected.begin()), unsigned(*mismatch.second),
                     unsigned(*mismatch.first));
    }
    visible.clear();
    for (UINT y = 0; y < c.height; ++y) {
        const auto* row = data + row_offset + y * row_pitch;
        visible.insert(visible.end(), row, row + c.width * c.bytes);
        for (UINT x0 = 0; x0 < c.width; x0 += c.tile_width) {
            const auto* slot = data + tiled(x0, y);
            visible.insert(visible.end(), slot, slot + std::min(c.width - x0, c.tile_width) * c.bytes);
        }
    }
    readback->Unmap(0, &empty);
    return matches;
}
}  // namespace

bool test_transfers(ID3D12Device& device, ID3D12Device* warp, const FormatCase& c) {
    Bytes native, observed;
    unsigned submissions = 0;
    for (const auto type : {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                            D3D12_COMMAND_LIST_TYPE_COPY}) {
        // Compare each queue with the same native queue, using an aligned WARP buffer offset.
        if (warp && !run_transfer(*warp, false, c, type, 0, native)) return false;
        const auto check = [&](UINT offset) {
            ++submissions;
            return run_transfer(device, true, c, type, offset, observed) && (!warp || native == observed);
        };
        const UINT staged_offset = c.bytes == 4 ? 1 : std::min(c.bytes, 4u);
        if (!check(0) || !check(staged_offset) || (c.bytes == 16 && !check(8))) return false;
    }
    std::printf("DXGI %u %ux%u: %u DIRECT/COMPUTE/COPY transfers matched both layouts (%zu visible bytes%s)\n",
                UINT(c.format), c.width, c.height, submissions, observed.size(), warp ? ", WARP matched" : "");
    return true;
}
}  // namespace sparse_test
