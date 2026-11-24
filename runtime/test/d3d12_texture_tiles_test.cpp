// D3D-285: compare synthetic sparse RGBA8 texture tiles on the Vulkan shim and native D3D12; at byte-unaligned
// CopyTiles offsets WARP also confirms the expectations.
#include "../shims/d3d12_com.h"
#include "host_wine.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {

using Microsoft::WRL::ComPtr;
constexpr UINT tile_width = 128, tile_height = 128, tiles = 4;
constexpr UINT64 tile_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
constexpr UINT64 image_bytes = tiles * tile_bytes;
constexpr UINT row_pitch = tile_width * sizeof(std::uint32_t);
constexpr UINT full_width = 2 * tile_width;
constexpr UINT snapshots = 3;
constexpr UINT64 row_major_offset = snapshots * image_bytes + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
constexpr auto to_tiled = D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE;
constexpr auto to_linear = D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER;

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

struct Gate {
    ID3D12Fence* fence;
    ~Gate() { if (fence) fence->Signal(1); }
};

D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

bool linear_buffer(ID3D12Device& device, UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                   ComPtr<ID3D12Resource>& resource) {
    const D3D12_HEAP_PROPERTIES heap{type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(&resource)));
}

std::uint32_t pixel(UINT tile, UINT x, UINT y) {
    return 0xcc000000u | ((tile + 1) << 16) | (y << 8) | x;
}

void transition(ID3D12GraphicsCommandList& list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list.ResourceBarrier(1, &barrier);
}

// The tiles' texels in the upload buffer, from `data` = the buffer at the CopyTiles offset.
void fill_tiles(std::uint8_t* data) {
    for (UINT tile = 0; tile < tiles; ++tile)
        for (UINT y = 0; y < tile_height; ++y)
            for (UINT x = 0; x < tile_width; ++x) {
                const auto value = pixel(tile, x, y);
                std::memcpy(data + tile * tile_bytes + y * row_pitch + x * sizeof(value), &value, sizeof(value));
            }
}

std::uint32_t load(const std::uint8_t* bytes, UINT64 offset) {
    std::uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

// Every readback texel run() checks: `visit(read, tile, x, y, offset, expected)` with its readback offset and the
// value the D3D12 rules put there. Stops at the first false.
template <typename Visit>
bool for_each_checked(UINT width, UINT height, UINT64 tile_copy_offset, Visit&& visit) {
    for (UINT tile = 0; tile < tiles; ++tile) {
        const UINT first_y = (tile / 2) * tile_height, first_x = (tile % 2) * tile_width;
        const UINT rows = std::min(tile_height, height - first_y), columns = std::min(tile_width, width - first_x);
        for (UINT y = 0; y < rows; ++y) {
            for (UINT x = 0; x < columns; ++x) {
                const auto tiled = tile_copy_offset + tile * tile_bytes + y * row_pitch + x * sizeof(std::uint32_t);
                if (!visit("initially unbound", tile, x, y, tiled, 0u)) return false;
                const UINT source_x = ((tile ^ 1) % 2) * tile_width;
                // The source never wrote the texels beyond its own clipped edge.
                if (source_x + x < width) {
                    const auto aliased = pixel(tile ^ 1, x, y);
                    const auto row_major = row_major_offset +
                                           ((first_y + y) * full_width + first_x + x) * sizeof(std::uint32_t);
                    if (!visit("alias", tile, x, y, image_bytes + tiled, aliased)) return false;
                    if (!visit("row-major alias", tile, x, y, row_major, aliased)) return false;
                }
                const auto remapped = tile < 2 ? pixel(0, x, y) : 0u;
                if (!visit("remap", tile, x, y, 2 * image_bytes + tiled, remapped)) return false;
            }
        }
    }
    return true;
}

// The readback buffer run() leaves if every CopyTiles moves whole 64 KiB tiles (clipped texels too) with its linear
// offset taken as `linear`, in both directions. `linear` == tile_copy_offset gives the D3D12 rules (main() checks
// that against the expectations above); tile_copy_offset & ~3 gives the AMD pattern described in main().
std::vector<std::uint8_t> modelled_readback(UINT width, UINT height, UINT64 tile_copy_offset, UINT64 linear) {
    std::vector<std::uint8_t> upload(tile_copy_offset + image_bytes), heap(image_bytes),
        readback(row_major_offset + image_bytes);
    fill_tiles(upload.data() + tile_copy_offset);
    // The heap tile behind each of a's tiles (UpdateTileMappings: the left column, then the right); b swaps the
    // columns (CopyTileMappings). Snapshot 0 reads a unmapped: zeros, as the buffer starts.
    constexpr UINT a_heap_tile[tiles] = {0, 2, 1, 3};
    const auto b_tile = [&](UINT tile) { return heap.data() + a_heap_tile[tile ^ 1] * tile_bytes; };
    for (UINT tile = 0; tile < tiles; ++tile)
        std::memcpy(heap.data() + a_heap_tile[tile] * tile_bytes, upload.data() + linear + tile * tile_bytes,
                    tile_bytes);
    for (UINT tile = 0; tile < tiles; ++tile)
        std::memcpy(readback.data() + linear + image_bytes + tile * tile_bytes, b_tile(tile), tile_bytes);
    for (UINT y = 0; y < height; ++y)  // CopyTextureRegion, at an aligned footprint
        for (UINT x = 0; x < width; ++x)
            std::memcpy(readback.data() + row_major_offset + (y * full_width + x) * sizeof(std::uint32_t),
                        b_tile((y / tile_height) * 2 + x / tile_width) + (y % tile_height) * row_pitch +
                            (x % tile_width) * sizeof(std::uint32_t),
                        sizeof(std::uint32_t));
    // The gated remap: b's top row reuses heap tile 0 (a's first tile), its bottom row is unmapped (zeros).
    for (UINT tile = 0; tile < tiles; ++tile)
        if (tile < 2)
            std::memcpy(readback.data() + linear + 2 * image_bytes + tile * tile_bytes, heap.data(), tile_bytes);
        else
            std::memset(readback.data() + linear + 2 * image_bytes + tile * tile_bytes, 0, tile_bytes);
    return readback;
}

// The model with the true offset must reproduce the expectations, or it cannot judge a native divergence.
bool model_matches_rules(UINT width, UINT height, UINT64 tile_copy_offset) {
    const auto model = modelled_readback(width, height, tile_copy_offset, tile_copy_offset);
    return for_each_checked(width, height, tile_copy_offset,
                            [&](const char* read, UINT tile, UINT x, UINT y, UINT64 offset, std::uint32_t expected) {
        if (load(model.data(), offset) == expected) return true;
        std::fprintf(stderr, "the CopyTiles model breaks the expectations: %s tile %u at (%u,%u)\n", read, tile, x, y);
        return false;
    });
}

// A native run at a byte-unaligned offset reads every texel quietly instead of stopping at the first that breaks the
// D3D12 rules: how many do, and how many differ from `model` (the readback modelled_readback gives).
struct Advisory {
    const std::vector<std::uint8_t>* model = nullptr;
    std::uint64_t texels = 0, differ = 0, unmodelled = 0;
    char first_unmodelled[160] = {};
};

// False when a call fails, or (unless `advisory`) a texel breaks the D3D12 rules.
bool run(ID3D12Device& device, UINT width, UINT height, UINT64 tile_copy_offset, std::vector<std::uint32_t>& result,
         bool layer, bool separate_copy_queue, bool& sparse, Advisory* advisory = nullptr) {
    ComPtr<ID3D12Resource> a, b, upload, readback;
    const auto desc = texture_desc(width, height);
    EXPECT(SUCCEEDED(device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a))));
    sparse = !layer || rt::d3d12::is_sparse_image(a.Get());
    if (!sparse) return true;
    EXPECT(SUCCEEDED(device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b))));
    D3D12_TILE_SHAPE shape{};
    D3D12_PACKED_MIP_INFO mips{};
    UINT tile_count = 0;
    device.GetResourceTiling(a.Get(), &tile_count, &mips, &shape, nullptr, 0, nullptr);
    EXPECT(tile_count == tiles && !mips.NumPackedMips && shape.WidthInTexels == tile_width &&
           shape.HeightInTexels == tile_height && shape.DepthInTexels == 1);

    const D3D12_HEAP_DESC heap_desc{
        image_bytes, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1},
        0, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES};
    ComPtr<ID3D12Heap> heap;
    EXPECT(SUCCEEDED(device.CreateHeap(&heap_desc, IID_PPV_ARGS(&heap))));
    EXPECT(linear_buffer(device, tile_copy_offset + image_bytes, D3D12_HEAP_TYPE_UPLOAD,
                         D3D12_RESOURCE_STATE_GENERIC_READ, upload));
    EXPECT(linear_buffer(device, row_major_offset + image_bytes, D3D12_HEAP_TYPE_READBACK,
                         D3D12_RESOURCE_STATE_COPY_DEST, readback));
    const D3D12_RANGE empty{0, 0};
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, &empty, &mapped)));
    fill_tiles(static_cast<std::uint8_t*>(mapped) + tile_copy_offset);
    upload->Unmap(0, nullptr);

    ComPtr<ID3D12Fence> gate, done;
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))));
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))));
    std::array<ComPtr<ID3D12CommandAllocator>, snapshots> allocators;
    std::array<ComPtr<ID3D12GraphicsCommandList>, snapshots> lists;
    for (UINT i = 0; i < snapshots; ++i) {
        const auto type = separate_copy_queue && i == 1 ? D3D12_COMMAND_LIST_TYPE_COPY : D3D12_COMMAND_LIST_TYPE_DIRECT;
        EXPECT(SUCCEEDED(device.CreateCommandAllocator(type, IID_PPV_ARGS(&allocators[i]))));
        EXPECT(SUCCEEDED(device.CreateCommandList(0, type, allocators[i].Get(), nullptr, IID_PPV_ARGS(&lists[i]))));
    }
    ComPtr<ID3D12CommandQueue> queue, copy_queue;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    EXPECT(SUCCEEDED(device.CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))));
    ComPtr<ID3D12Fence> handoff;
    if (separate_copy_queue) {
        const D3D12_COMMAND_QUEUE_DESC copy_desc{D3D12_COMMAND_LIST_TYPE_COPY};
        EXPECT(SUCCEEDED(device.CreateCommandQueue(&copy_desc, IID_PPV_ARGS(&copy_queue))));
        EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&handoff))));
    }
    const Event event;
    EXPECT(event.handle);
    const auto submit = [&](UINT i) {
        if (FAILED(lists[i]->Close())) return false;
        ID3D12CommandList* batch[] = {lists[i].Get()};
        auto* target = separate_copy_queue && i == 1 ? copy_queue.Get() : queue.Get();
        target->ExecuteCommandLists(1, batch);
        return SUCCEEDED(device.GetDeviceRemovedReason());
    };
    const auto wait_done = [&](UINT64 value) {
        return SUCCEEDED(queue->Signal(done.Get(), value)) &&
               SUCCEEDED(done->SetEventOnCompletion(value, event.handle)) &&
               WaitForSingleObject(event.handle, 15000) == WAIT_OBJECT_0 &&
               SUCCEEDED(device.GetDeviceRemovedReason());
    };

    const D3D12_TILED_RESOURCE_COORDINATE origin{}, right{1, 0, 0, 0}, bottom{0, 1, 0, 0};
    const D3D12_TILE_REGION_SIZE all{tiles, TRUE, 2, 2, 1}, column{2, TRUE, 1, 2, 1}, row{2, TRUE, 2, 1, 1};
    lists[0]->CopyTiles(a.Get(), &origin, &all, readback.Get(), tile_copy_offset, to_linear);
    EXPECT(submit(0));
    EXPECT(wait_done(1));

    // Box regions traverse a column with a gap between rows. Swap the two columns in the alias.
    const D3D12_TILED_RESOURCE_COORDINATE starts[] = {origin, right};
    const D3D12_TILE_REGION_SIZE column_regions[] = {column, column};
    const UINT heap_offset = 0;
    queue->UpdateTileMappings(a.Get(), 2, starts, column_regions, heap.Get(), 1, nullptr, &heap_offset, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    queue->CopyTileMappings(b.Get(), &right, a.Get(), &origin, &column, D3D12_TILE_MAPPING_FLAG_NONE);
    queue->CopyTileMappings(b.Get(), &origin, a.Get(), &right, &column, D3D12_TILE_MAPPING_FLAG_NONE);
    transition(*lists[1].Get(), a.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    lists[1]->CopyTiles(a.Get(), &origin, &all, upload.Get(), tile_copy_offset, to_tiled);
    transition(*lists[1].Get(), a.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    D3D12_RESOURCE_BARRIER alias{};
    alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    alias.Aliasing = {a.Get(), b.Get()};
    lists[1]->ResourceBarrier(1, &alias);
    lists[1]->CopyTiles(b.Get(), &origin, &all, readback.Get(), tile_copy_offset + image_bytes, to_linear);
    // An ordinary row-major copy also checks where the constrained CopyTiles path wrote each texel.
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = b.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {row_major_offset, {desc.Format, width, height, 1, 2 * row_pitch}};
    lists[1]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (separate_copy_queue) {
        // The copy must see the direct queue's mappings. The later direct read must see the copy's writes.
        EXPECT(SUCCEEDED(queue->Signal(handoff.Get(), 1)));
        EXPECT(SUCCEEDED(copy_queue->Wait(handoff.Get(), 1)));
    }
    EXPECT(submit(1));
    if (separate_copy_queue) {
        EXPECT(SUCCEEDED(copy_queue->Signal(handoff.Get(), 2)));
        EXPECT(SUCCEEDED(queue->Wait(handoff.Get(), 2)));
    }
    EXPECT(wait_done(2));

    Gate open_on_exit{gate.Get()};
    EXPECT(SUCCEEDED(queue->Wait(gate.Get(), 1)));
    const auto reuse = D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE;
    queue->UpdateTileMappings(b.Get(), 1, &origin, &row, heap.Get(), 1, &reuse, &heap_offset, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    const auto unmap = D3D12_TILE_RANGE_FLAG_NULL;
    queue->UpdateTileMappings(b.Get(), 1, &bottom, &row, nullptr, 1, &unmap, nullptr, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    // In-flight mappings retain their heap memory after the caller drops the heap object.
    if (layer) heap.Reset();
    lists[2]->CopyTiles(b.Get(), &origin, &all, readback.Get(), tile_copy_offset + 2 * image_bytes, to_linear);
    EXPECT(submit(2));
    EXPECT(SUCCEEDED(queue->Signal(done.Get(), 3)));
    EXPECT(done->GetCompletedValue() == 2);
    EXPECT(SUCCEEDED(gate->Signal(1)));
    open_on_exit.fence = nullptr;
    EXPECT(SUCCEEDED(done->SetEventOnCompletion(3, event.handle)));
    EXPECT(WaitForSingleObject(event.handle, 15000) == WAIT_OBJECT_0);

    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    const auto judge = [&](const char* read, UINT tile, UINT x, UINT y, UINT64 offset, std::uint32_t expected) {
        const auto actual = load(bytes, offset);
        if (advisory) {
            const auto modelled = load(advisory->model->data(), offset);
            ++advisory->texels;
            advisory->differ += actual != expected;
            if (actual != modelled && !advisory->unmodelled++)
                std::snprintf(advisory->first_unmodelled, sizeof(advisory->first_unmodelled),
                              "%s tile %u at (%u,%u): got %#x, expected %#x, modelled %#x", read, tile, x, y, actual,
                              expected, modelled);
        } else if (actual != expected) {
            std::fprintf(stderr, "%s tile %u at (%u,%u): got %#x, expected %#x\n", read, tile, x, y, actual, expected);
            return false;
        }
        if (actual == expected) result.push_back(actual);
        return true;
    };
    EXPECT(for_each_checked(width, height, tile_copy_offset, judge));
    readback->Unmap(0, &empty);
    return true;
}

// Under Wine and Proton the host D3D12 is vkd3d, and "WARP" is vkd3d-proton on the GPU: neither is compared.
ComPtr<ID3D12Device> native_device() {
    if (hostwine::host_is_wine()) return {};
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) != S_OK) return {};
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        ComPtr<ID3D12Device> device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) continue;
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
            options.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2) continue;
        std::printf("Native D3D12: %ls, tiled resources tier %u\n", desc.Description, unsigned(options.TiledResourcesTier));
        return device;
    }
}

ComPtr<ID3D12Device> warp_device() {
    if (hostwine::host_is_wine()) return {};
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter))) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2) {
        std::puts("No WARP with tiled resources tier 2: nothing confirms the expectations at byte-unaligned offsets");
        return {};
    }
    std::printf("WARP: tiled resources tier %u\n", unsigned(options.TiledResourcesTier));
    return device;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // progress in order with the failures on stderr
    ComPtr<ID3D12Device> vulkan;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&vulkan)))) {
        std::puts("Vulkan standalone device unavailable");
        return 77;
    }
    auto multiple_mips = texture_desc(256, 256);
    multiple_mips.MipLevels = 2;
    ComPtr<ID3D12Resource> fallback;
    if (FAILED(vulkan->CreateReservedResource(&multiple_mips, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&fallback))) ||
        rt::d3d12::is_sparse_image(fallback.Get())) {
        std::fputs("A texture with multiple mips did not use the committed fallback\n", stderr);
        return 1;
    }
    auto native = native_device();
    // Byte-unaligned CopyTiles offsets. CopyTiles documents no alignment for BufferStartOffsetInBytes, and WARP meets
    // the expectations at every offset tried (1-8), so at such an offset WARP runs too, to confirm them; the layer is
    // held to the expectations. AMD's Windows D3D12 driver (RX 7900 XT, 26.11.2) does not meet them: judging by the
    // texel values (the D3D12 debug layer was not run), it moves the tiles as if the offset were rounded down to a
    // multiple of 4, in both directions. The row-major copy then finds each texel 1-3 bytes early, and texels that
    // straddle a tile seam differ in the CopyTiles reads too. Such a native run is a NOTE only when every texel
    // matches that rounded-down model (modelled_readback); any other divergence fails, and so does any texel at a
    // 4-byte offset.
    const auto warp = warp_device();
    const auto* info = rt::d3d12::vulkan_device_info(vulkan.Get());
    const bool dedicated_copy = info && !info->copy_queues.empty() && info->copy_family != info->queue_family;
    struct ReplayCase {
        UINT width, height, linear_offset;
        bool separate_copy_queue;
    };
    // Clipped edges at a 4-byte offset keep native strict on partial tiles on every GPU.
    constexpr std::array cases{ReplayCase{256, 256, 4, false}, ReplayCase{256, 240, 4, false},
                               ReplayCase{240, 240, 4, false}, ReplayCase{256, 240, 1, false},
                               ReplayCase{240, 240, 3, false}, ReplayCase{256, 256, 4, true}};
    if (!dedicated_copy) std::puts("No separate Vulkan copy family for the cross-family case");
    for (const auto& [width, height, linear_offset, separate_copy_queue] : cases) {
        if (separate_copy_queue && !dedicated_copy) continue;
        bool sparse = false;
        std::vector<std::uint32_t> observed, expected, reference;
        if (!run(*vulkan.Get(), width, height, linear_offset, observed, true, separate_copy_queue, sparse)) {
            std::fprintf(stderr, "Vulkan replay failed: %u x %u, linear offset %u\n", width, height, linear_offset);
            return 1;
        }
        if (!sparse) {
            std::printf("Vulkan device does not support compatible sparse RGBA8 tiling for %u x %u\n", width, height);
            if (height == 256) return 77;
            continue;
        }
        std::printf("Vulkan replay: %u x %u, linear offset %u, initially unbound, aliased, row-major, gated remap%s\n",
                    width, height, linear_offset,
                    separate_copy_queue ? ", COPY-to-DIRECT handoff" : "");
        const bool byte_offset = linear_offset % 4 != 0;
        if (byte_offset && !model_matches_rules(width, height, linear_offset)) return 1;
        if (native) {
            const UINT rounded_offset = linear_offset & ~3u;
            std::vector<std::uint8_t> rounded;
            Advisory advisory;
            if (byte_offset) {
                rounded = modelled_readback(width, height, linear_offset, rounded_offset);
                advisory.model = &rounded;
            }
            if (!run(*native.Get(), width, height, linear_offset, expected, false, separate_copy_queue, sparse,
                     byte_offset ? &advisory : nullptr) ||
                (!advisory.differ && observed != expected)) {
                std::fprintf(stderr, "Native D3D12 failed: %u x %u, linear offset %u\n", width, height, linear_offset);
                return 1;
            }
            if (!advisory.differ) {
                std::printf("Native D3D12 matched all %zu verified pixels\n", observed.size());
            } else if (!advisory.unmodelled) {
                std::printf("NOTE: native D3D12 broke the expectations at %llu of %llu checked reads, every read as "
                            "CopyTiles would with linear offset %u rounded down to %u in both directions "
                            "(docs/testing.md, Known problems); not a failure\n",
                            static_cast<unsigned long long>(advisory.differ),
                            static_cast<unsigned long long>(advisory.texels), linear_offset, rounded_offset);
            } else {
                std::fprintf(stderr, "Native D3D12 failed: %u x %u, linear offset %u: %llu of %llu checked reads "
                             "break the expectations, and %llu differ from the rounded-down model; first: %s\n",
                             width, height, linear_offset, static_cast<unsigned long long>(advisory.differ),
                             static_cast<unsigned long long>(advisory.texels),
                             static_cast<unsigned long long>(advisory.unmodelled), advisory.first_unmodelled);
                return 1;
            }
        }
        if (byte_offset && warp) {
            if (!run(*warp.Get(), width, height, linear_offset, reference, false, separate_copy_queue, sparse)) {
                std::fprintf(stderr, "WARP failed: %u x %u, linear offset %u\n", width, height, linear_offset);
                return 1;
            }
            std::printf("WARP meets the expectations at linear offset %u (%zu pixels)\n", linear_offset,
                        reference.size());
        }
    }
    return 0;
}
