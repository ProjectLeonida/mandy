// BuildRaytracingAccelerationStructure on the standalone Vulkan device writes XDXR BVH4 records (docs/d3d12_vulkan.md
// 4.8, src/xdxr_build.h), the layout the game's translated traversal shaders read. A BLAS of two geometries (an
// unindexed triangle soup, and an R16-indexed grid with a 16-byte stride and a 3x4 transform) and a TLAS of three
// instances are built through the D3D12 API, sized by GetRaytracingAccelerationStructurePrebuildInfo, copied to
// readback buffers and written to <out dir>; the BLAS is then updated in place (PERFORM_UPDATE) after its soup moves,
// and written again as d3d12_blas_update; then its postbuild COMPACTED_SIZE and a TLAS rebuild's CURRENT_SIZE are
// checked against the prebuild sizes, and the BLAS is compacted (CopyRaytracingAccelerationStructure COMPACT) into a
// buffer of exactly that size, written as d3d12_blas_compacted as <case>.bin + <case>.json at their console GPU virtual addresses.
// Last, the BLAS again with MINIMIZE_MEMORY (BOX16 nodes: a smaller prebuild size and postbuild size), written as
// d3d12_blas_box16, then updated in place with the same flags (d3d12_blas_box16_update); and once more without
// ALLOW_UPDATE, so that its triangles are paired into TRI0/TRI1 leaves, one record a pair, with its own (larger)
// prebuild scratch (d3d12_blas_paired; the ALLOW_UPDATE builds must have no pairs, since a refit rewrites the leaves
// one triangle each). Then serialization: the SERIALIZATION
// postbuild info, SERIALIZE of the BLAS and the TLAS (header, identifier, BLAS pointers, bytes), DESERIALIZE of the
// BLAS to another address and of the TLAS with its serialized BLAS pointers rewritten to point there, written as
// d3d12_blas_deserialized and d3d12_tlas_deserialized.
// `python -I tools/xdxr_bvh.py check-build <out dir>` validates them and traces rays against brute force (ctest:
// d3d12_xdxr_rt + d3d12_xdxr_rt_check). Exit 77 without a Vulkan device, or with LEONIDA_RT_AS=khr. No game files.
#include "../shims/d3d12_com.h"
#include "../src/xdxr_build.h"   // xdxr_serialized_id: the identifier a serialized structure starts with

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
    T** put() { return &p; }
};

int fail(const char* what, HRESULT hr = S_OK) {
    std::fprintf(stderr, "FAIL: %s (hr 0x%08lx)\n", what, static_cast<unsigned long>(hr));
    return 1;
}

HRESULT buffer(ID3D12Device* device, D3D12_HEAP_TYPE heap, UINT64 size, D3D12_RESOURCE_STATES state,
               D3D12_RESOURCE_FLAGS flags, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size ? size : 1;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
}

std::string num(double x) {
    char s[64];
    std::snprintf(s, sizeof s, "%.9g", x);
    return s;
}

bool write_file(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), std::streamsize(bytes));
    return bool(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: d3d12_xdxr_rt_test <out dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const auto ext = entry.path().extension();
        if (ext == ".bin" || ext == ".json") std::filesystem::remove(entry.path(), ec);
    }

    // LEONIDA_RT_AS=khr builds #743's VkAccelerationStructureKHR instead, whose bytes are the driver's own.
    char rt_as[8] = {};
    std::size_t rt_as_length = 0;
    if (getenv_s(&rt_as_length, rt_as, "LEONIDA_RT_AS") == 0 && rt_as_length > 1 && std::strcmp(rt_as, "khr") == 0) {
        std::puts("SKIP: LEONIDA_RT_AS=khr");
        return 77;
    }
    Ref<ID3D12Device5> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(device.put())))) {
        std::puts("SKIP: no Vulkan D3D12 device");
        return 77;
    }

    // ---- geometry: a triangle soup, and an R16-indexed grid (stride 16) with a transform ----
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> spread(-4.0f, 4.0f), jitter(-0.6f, 0.6f);
    const UINT soup_tris = 300;
    std::vector<float> soup;
    for (UINT t = 0; t < soup_tris; ++t) {
        const float c[3] = {spread(rng), spread(rng), spread(rng)};
        for (int k = 0; k < 9; ++k) soup.push_back(c[k % 3] + jitter(rng));
    }
    const UINT n = 8;
    std::vector<float> grid;
    for (UINT j = 0; j <= n; ++j)
        for (UINT i = 0; i <= n; ++i) {
            grid.push_back(float(i) - n * 0.5f);
            grid.push_back(float(j) - n * 0.5f);
            grid.push_back(0.25f * float((i * 7 + j * 3) % 5));
            grid.push_back(9.0f);                                    // padding: the stride is 16
        }
    std::vector<std::uint16_t> indices;
    for (UINT j = 0; j < n; ++j)
        for (UINT i = 0; i < n; ++i) {
            const std::uint16_t a = std::uint16_t(j * (n + 1) + i);
            const std::uint16_t q[6] = {a, std::uint16_t(a + 1), std::uint16_t(a + n + 2), a, std::uint16_t(a + n + 2),
                                        std::uint16_t(a + n + 1)};
            indices.insert(indices.end(), q, q + 6);
        }
    const float transform[12] = {0, 2, 0, 1, 0, 0, 1, -2, 2, 0, 0, 3};   // exact in float

    const UINT64 soup_bytes = soup.size() * 4, grid_bytes = grid.size() * 4, index_bytes = indices.size() * 2;
    const UINT64 upload_size = 256 + soup_bytes + grid_bytes + index_bytes + 64 + 3 * 64;
    Ref<ID3D12Resource> upload;
    HRESULT hr = buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, upload_size, D3D12_RESOURCE_STATE_GENERIC_READ,
                        D3D12_RESOURCE_FLAG_NONE, upload.put());
    if (FAILED(hr)) return fail("upload buffer", hr);
    char* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload");
    UINT64 at = 0;
    auto place = [&](const void* data, UINT64 bytes, UINT64 align) {
        at = (at + align - 1) / align * align;
        std::memcpy(mapped + at, data, std::size_t(bytes));
        const UINT64 va = upload->GetGPUVirtualAddress() + at;
        at += bytes;
        return va;
    };
    const UINT64 soup_va = place(soup.data(), soup_bytes, 16);
    const UINT64 grid_va = place(grid.data(), grid_bytes, 16);
    const UINT64 index_va = place(indices.data(), index_bytes, 16);
    const UINT64 transform_va = place(transform, sizeof transform, 16);

    D3D12_RAYTRACING_GEOMETRY_DESC geometries[2]{};
    geometries[0].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometries[0].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometries[0].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometries[0].Triangles.VertexCount = soup_tris * 3;
    geometries[0].Triangles.VertexBuffer = {soup_va, 12};
    geometries[1].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geometries[1].Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geometries[1].Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geometries[1].Triangles.VertexCount = (n + 1) * (n + 1);
    geometries[1].Triangles.VertexBuffer = {grid_va, 16};
    geometries[1].Triangles.IndexFormat = DXGI_FORMAT_R16_UINT;
    geometries[1].Triangles.IndexCount = UINT(indices.size());
    geometries[1].Triangles.IndexBuffer = index_va;
    geometries[1].Triangles.Transform3x4 = transform_va;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blas_inputs{};
    blas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blas_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;   // updated below
    blas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blas_inputs.NumDescs = 2;
    blas_inputs.pGeometryDescs = geometries;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&blas_inputs, &blas_info);
    if (!blas_info.ResultDataMaxSizeInBytes) {
        std::puts("SKIP: the device does not build XDXR acceleration structures (no bufferDeviceAddress)");
        return 77;
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlas_inputs{};
    tlas_inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlas_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlas_inputs.NumDescs = 3;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlas_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&tlas_inputs, &tlas_info);
    if (!tlas_info.ResultDataMaxSizeInBytes) return fail("TLAS prebuild info is empty");

    const D3D12_RESOURCE_FLAGS uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES as_state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
    Ref<ID3D12Resource> blas, tlas, scratch, blas_rb, tlas_rb;
    const UINT64 scratch_size = blas_info.ScratchDataSizeInBytes > tlas_info.ScratchDataSizeInBytes
                                    ? blas_info.ScratchDataSizeInBytes : tlas_info.ScratchDataSizeInBytes;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, blas_info.ResultDataMaxSizeInBytes, as_state, uav, blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, tlas_info.ResultDataMaxSizeInBytes, as_state, uav, tlas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, scratch_size, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav,
                           scratch.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, blas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, blas_rb.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, tlas_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, tlas_rb.put())))
        return fail("acceleration-structure buffers", hr);
    if (blas->GetGPUVirtualAddress() % 256 || tlas->GetGPUVirtualAddress() % 256)
        return fail("acceleration structures must be 256-byte aligned");

    // ---- instances: identity, rotated + translated, and masked out ----
    const float c = std::cos(0.7f), s = std::sin(0.7f);
    const float xforms[3][12] = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
                                 {c, -s, 0, 9, s, c, 0, -1, 0, 0, 1, 0.5f},
                                 {2, 0, 0, -9, 0, 0.5f, 0, 2, 0, 0, 1.5f, 1}};
    const UINT masks[3] = {0xff, 0x01, 0x00};
    D3D12_RAYTRACING_INSTANCE_DESC instances[3]{};
    for (int i = 0; i < 3; ++i) {
        std::memcpy(instances[i].Transform, xforms[i], sizeof xforms[i]);
        instances[i].InstanceID = 20 + i;
        instances[i].InstanceMask = masks[i];
        instances[i].AccelerationStructure = blas->GetGPUVirtualAddress();
    }
    const UINT64 instances_va = place(instances, sizeof instances, 16);
    upload->Unmap(0, nullptr);
    tlas_inputs.InstanceDescs = instances_va;

    // ---- record: BLAS, barrier, TLAS, barrier, copies to readback ----
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList4> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(queue.put()))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(allocator.put()))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(list.put()))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.put()))))
        return fail("queue, allocator, list or fence");
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.Inputs = blas_inputs;
    build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = nullptr;                         // every UAV: the BLAS and the scratch
    list->ResourceBarrier(1, &barrier);
    build.Inputs = tlas_inputs;
    build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    // The layer keeps buffers in one memory, so a copy out of an acceleration structure reads its bytes as they are
    // (D3D12 itself only allows CopyRaytracingAccelerationStructure; this is a test of the layer's records).
    list->CopyBufferRegion(blas_rb.p, 0, blas.p, 0, blas_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(tlas_rb.p, 0, tlas.p, 0, tlas_info.ResultDataMaxSizeInBytes);
    if (FAILED(hr = list->Close())) return fail("Close", hr);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1))) return fail("Signal");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);   // reused for the update below
    const bool done = event && SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                      WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    if (!done) return fail("the builds did not complete");
    if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed", device->GetDeviceRemovedReason());

    // ---- dump ----
    auto dump = [&](ID3D12Resource* rb, UINT64 size, const std::string& name, const std::string& json) {
        void* p = nullptr;
        if (FAILED(rb->Map(0, nullptr, &p))) return false;
        std::uint32_t units = 0;
        std::memcpy(&units, static_cast<char*>(p) + 4, 4);    // header totalSizeIn64BNodes
        const UINT64 used = (UINT64(units) + 1) * 64;   // the field does not count the 64-byte header
        const bool ok = used && used <= size && write_file(dir + "/" + name + ".bin", p, std::size_t(used)) &&
                        write_file(dir + "/" + name + ".json", json.data(), json.size());
        rb->Unmap(0, nullptr);
        return ok;
    };
    // The BLAS's triangles as the reference sees them, for the soup's current positions.
    auto triangles_json = [&](const std::vector<float>& soup_now) {
        std::string tris = "[";
        bool first = true;
        auto add_tri = [&](const float v[3][3], int geometry, UINT primitive) {
            tris += first ? "\n  [" : ",\n  [";
            first = false;
            for (int k = 0; k < 3; ++k)
                tris += (k ? ", [" : "[") + num(v[k][0]) + ", " + num(v[k][1]) + ", " + num(v[k][2]) + "]";
            tris += ", " + std::to_string(geometry) + ", " + std::to_string(primitive) + "]";
        };
        for (UINT t = 0; t < soup_tris; ++t) {
            float v[3][3];
            for (int k = 0; k < 9; ++k) v[k / 3][k % 3] = soup_now[9 * t + k];
            add_tri(v, 0, t);
        }
        for (UINT t = 0; t < indices.size() / 3; ++t) {
            float v[3][3];
            for (int k = 0; k < 3; ++k) {
                const float* p = &grid[4 * indices[3 * t + k]];
                for (int r = 0; r < 3; ++r)
                    v[k][r] = transform[4 * r] * p[0] + transform[4 * r + 1] * p[1] + transform[4 * r + 2] * p[2] +
                              transform[4 * r + 3];
            }
            add_tri(v, 1, t);
        }
        return tris + "]";
    };
    const std::string tris = triangles_json(soup);
    // ALLOW_UPDATE: never paired, since a refit rewrites the leaves one triangle each ("pairs": false)
    if (!dump(blas_rb.p, blas_info.ResultDataMaxSizeInBytes, "d3d12_blas",
              "{\"kind\": \"blas\", \"pairs\": false, \"box16\": false, \"address\": " + std::to_string(blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + tris + "}\n"))
        return fail("BLAS readback");
    std::string inst = "[";
    for (int i = 0; i < 3; ++i) {
        inst += i ? ", {\"transform\": [" : "{\"transform\": [";
        for (int k = 0; k < 12; ++k) inst += (k ? ", " : "") + num(xforms[i][k]);
        inst += "], \"mask\": " + std::to_string(masks[i]) + ", \"blas\": 0}";
    }
    inst += "]";
    if (!dump(tlas_rb.p, tlas_info.ResultDataMaxSizeInBytes, "d3d12_tlas",
              "{\"kind\": \"tlas\", \"address\": " + std::to_string(tlas->GetGPUVirtualAddress()) +
                  ", \"bias\": 0, \"blas\": [\"d3d12_blas\"], \"instances\": " + inst + "}\n"))
        return fail("TLAS readback");

    // ---- PERFORM_UPDATE in place: the soup moves (every vertex), the grid stays; the BLAS is refitted ----
    std::vector<float> moved = soup;
    std::uniform_real_distribution<float> nudge(-1.0f, 1.0f);
    for (std::size_t v = 0; v < moved.size(); v += 3) {
        moved[v] = moved[v] * 1.3f + nudge(rng);
        moved[v + 1] = moved[v + 1] * 0.7f + nudge(rng);
        moved[v + 2] += nudge(rng);
    }
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload for the update");
    std::memcpy(mapped + (soup_va - upload->GetGPUVirtualAddress()), moved.data(), std::size_t(soup_bytes));
    upload->Unmap(0, nullptr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for the update");
    build.Inputs = blas_inputs;
    build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
    build.SourceAccelerationStructureData = blas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(blas_rb.p, 0, blas.p, 0, blas_info.ResultDataMaxSizeInBytes);
    if (FAILED(hr = list->Close())) return fail("Close (update)", hr);
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 2))) return fail("Signal (update)");
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool updated = event && SUCCEEDED(fence->SetEventOnCompletion(2, event)) &&
                         WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    if (!updated) return fail("the update did not complete");
    if (!dump(blas_rb.p, blas_info.ResultDataMaxSizeInBytes, "d3d12_blas_update",
              "{\"kind\": \"blas\", \"pairs\": false, \"address\": " + std::to_string(blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("BLAS readback after the update");

    // ---- postbuild info and compaction: EmitRaytracingAccelerationStructurePostbuildInfo for both structures, a
    // TLAS rebuild with a CURRENT_SIZE postbuild desc, then COMPACT the BLAS into a buffer of exactly its size ----
    auto submit = [&](UINT64 value) {
        if (FAILED(list->Close())) return false;
        queue->ExecuteCommandLists(1, lists);
        if (FAILED(queue->Signal(fence.p, value))) return false;
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        const bool ok = e && SUCCEEDED(fence->SetEventOnCompletion(value, e)) &&
                        WaitForSingleObject(e, 10000) == WAIT_OBJECT_0;
        if (e) CloseHandle(e);
        return ok;
    };
    Ref<ID3D12Resource> info, info_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, 24, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav, info.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, 24, D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, info_rb.put())))
        return fail("postbuild buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for postbuild");
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC compacted_desc{
        info->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE};
    const D3D12_GPU_VIRTUAL_ADDRESS both[2] = {blas->GetGPUVirtualAddress(), tlas->GetGPUVirtualAddress()};
    list->EmitRaytracingAccelerationStructurePostbuildInfo(&compacted_desc, 2, both);
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC current_desc{
        info->GetGPUVirtualAddress() + 16, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE};
    build = {};
    build.Inputs = tlas_inputs;
    build.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 1, &current_desc);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(info_rb.p, 0, info.p, 0, 24);
    if (!submit(3)) return fail("postbuild submission");
    UINT64 sizes[3] = {};
    {
        void* p = nullptr;
        if (FAILED(info_rb->Map(0, nullptr, &p))) return fail("Map postbuild");
        std::memcpy(sizes, p, sizeof sizes);
        info_rb->Unmap(0, nullptr);
    }
    // A build's size is the nodes it wrote, within the prebuild size (room for the most a tree of that many
    // elements can have), and it is already as small as it gets: COMPACTED_SIZE and CURRENT_SIZE are that size. The
    // same TLAS built twice has the same tree.
    auto fits = [](UINT64 size, UINT64 max) { return size && size % 64 == 0 && size <= max; };
    if (!fits(sizes[0], blas_info.ResultDataMaxSizeInBytes) || !fits(sizes[1], tlas_info.ResultDataMaxSizeInBytes) ||
        sizes[2] != sizes[1]) {
        std::fprintf(stderr, "postbuild sizes %llu %llu %llu, prebuild %llu %llu\n",
                     static_cast<unsigned long long>(sizes[0]), static_cast<unsigned long long>(sizes[1]),
                     static_cast<unsigned long long>(sizes[2]),
                     static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                     static_cast<unsigned long long>(tlas_info.ResultDataMaxSizeInBytes));
        return fail("postbuild sizes");
    }
    Ref<ID3D12Resource> compacted, compacted_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, sizes[0], as_state, uav, compacted.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, sizes[0], D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_FLAG_NONE, compacted_rb.put())))
        return fail("compacted buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for compaction");
    list->CopyRaytracingAccelerationStructure(compacted->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
                                              D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(compacted_rb.p, 0, compacted.p, 0, sizes[0]);
    if (!submit(4)) return fail("compaction submission");
    if (!dump(compacted_rb.p, sizes[0], "d3d12_blas_compacted",
              "{\"kind\": \"blas\", \"address\": " + std::to_string(compacted->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("compacted BLAS readback");

    // ---- MINIMIZE_MEMORY: the same BLAS with BOX16 nodes (half boxes, 64 bytes a node instead of 128). Prebuild
    // info must already count the smaller nodes, and an update with the same flags must refit them as BOX16 ----
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS box16_inputs = blas_inputs;
    box16_inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO box16_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&box16_inputs, &box16_info);
    if (!box16_info.ResultDataMaxSizeInBytes || box16_info.ResultDataMaxSizeInBytes >= blas_info.ResultDataMaxSizeInBytes)
        return fail("MINIMIZE_MEMORY prebuild size is not below the default one");
    Ref<ID3D12Resource> box16_blas, box16_rb;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, box16_info.ResultDataMaxSizeInBytes, as_state, uav,
                           box16_blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, box16_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, box16_rb.put())))
        return fail("MINIMIZE_MEMORY buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for MINIMIZE_MEMORY");
    build = {};
    build.Inputs = box16_inputs;
    build.DestAccelerationStructureData = box16_blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 1, &compacted_desc);   // its COMPACTED_SIZE, at info + 0
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(box16_rb.p, 0, box16_blas.p, 0, box16_info.ResultDataMaxSizeInBytes);
    list->CopyBufferRegion(info_rb.p, 0, info.p, 0, 8);
    if (!submit(5)) return fail("MINIMIZE_MEMORY submission");
    UINT64 box16_size = 0;
    {
        void* p = nullptr;
        if (FAILED(info_rb->Map(0, nullptr, &p))) return fail("Map postbuild (MINIMIZE_MEMORY)");
        std::memcpy(&box16_size, p, 8);
        info_rb->Unmap(0, nullptr);
    }
    // the nodes the build wrote and the spare unit, within the prebuild size and below the BOX32 build's own size
    if (!box16_size || box16_size % 64 || box16_size > box16_info.ResultDataMaxSizeInBytes || box16_size >= sizes[0])
        return fail("MINIMIZE_MEMORY postbuild size");
    if (!dump(box16_rb.p, box16_info.ResultDataMaxSizeInBytes, "d3d12_blas_box16",
              "{\"kind\": \"blas\", \"pairs\": false, \"box16\": true, \"address\": " +
                  std::to_string(box16_blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + triangles_json(moved) + "}\n"))
        return fail("MINIMIZE_MEMORY BLAS readback");
    // the soup moves back to where it started, and the BOX16 BLAS is refitted in place
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("Map upload (MINIMIZE_MEMORY)");
    std::memcpy(mapped + (soup_va - upload->GetGPUVirtualAddress()), soup.data(), std::size_t(soup_bytes));
    upload->Unmap(0, nullptr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for the BOX16 update");
    build.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    build.SourceAccelerationStructureData = box16_blas->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(box16_rb.p, 0, box16_blas.p, 0, box16_info.ResultDataMaxSizeInBytes);
    if (!submit(6)) return fail("BOX16 update submission");
    if (!dump(box16_rb.p, box16_info.ResultDataMaxSizeInBytes, "d3d12_blas_box16_update",
              "{\"kind\": \"blas\", \"pairs\": false, \"box16\": true, \"address\": " +
                  std::to_string(box16_blas->GetGPUVirtualAddress()) +
                  ", \"triangles\": " + tris + "}\n"))
        return fail("BOX16 BLAS readback after the update");
    std::printf("d3d12_xdxr_rt: MINIMIZE_MEMORY BLAS %llu bytes (BOX16)\n", static_cast<unsigned long long>(box16_size));

    // ---- a BLAS that will not be updated (no ALLOW_UPDATE): sibling triangles that share an edge are paired into
    // TRI0/TRI1 leaves, one 64-byte record a pair. The prebuild room is the same (a leaf per triangle) and the
    // scratch larger (the leaves wait there while the pairs are chosen); the build is 64 bytes smaller per pair ----
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS static_inputs = blas_inputs;
    static_inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO static_info{};
    device->GetRaytracingAccelerationStructurePrebuildInfo(&static_inputs, &static_info);
    if (static_info.ResultDataMaxSizeInBytes != blas_info.ResultDataMaxSizeInBytes)
        return fail("a BLAS without ALLOW_UPDATE has another prebuild size");
    if (static_info.ScratchDataSizeInBytes <= blas_info.ScratchDataSizeInBytes)
        return fail("a BLAS that pairs its triangles asks for no more scratch");
    Ref<ID3D12Resource> static_blas, static_rb, static_scratch;
    if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, static_info.ResultDataMaxSizeInBytes, as_state, uav,
                           static_blas.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, static_info.ResultDataMaxSizeInBytes,
                           D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, static_rb.put())) ||
        FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, static_info.ScratchDataSizeInBytes,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav, static_scratch.put())))
        return fail("static BLAS buffers", hr);
    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for the static BLAS");
    build = {};
    build.Inputs = static_inputs;
    build.DestAccelerationStructureData = static_blas->GetGPUVirtualAddress();
    build.ScratchAccelerationStructureData = static_scratch->GetGPUVirtualAddress();
    list->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(static_rb.p, 0, static_blas.p, 0, static_info.ResultDataMaxSizeInBytes);
    if (!submit(7)) return fail("static BLAS submission");
    if (!dump(static_rb.p, static_info.ResultDataMaxSizeInBytes, "d3d12_blas_paired",
              "{\"kind\": \"blas\", \"pairs\": true, \"address\": " +
                  std::to_string(static_blas->GetGPUVirtualAddress()) + ", \"triangles\": " + tris + "}\n"))
        return fail("static BLAS readback");
    {
        // One leaf record a pair: the element region (offsetToElements to offsetToNodes) holds a record per
        // triangle less one per TRI1 child, so the build is 64 bytes a pair smaller than unpaired.
        void* p = nullptr;
        if (FAILED(static_rb->Map(0, nullptr, &p))) return fail("Map the static BLAS");
        const auto* s = static_cast<const char*>(p);
        std::uint32_t units = 0, triangles = 0, nodes_off = 0, elements_off = 0, pairs = 0;
        std::memcpy(&units, s + 4, 4);
        std::memcpy(&triangles, s + 12, 4);
        std::memcpy(&nodes_off, s + 40, 4);
        std::memcpy(&elements_off, s + 52, 4);
        const UINT64 used = (UINT64(units) + 1) * 64;
        for (UINT64 o = nodes_off; o + 128 <= used; o += 128) {         // BOX32 nodes
            for (int c = 0; c < 4; ++c) {
                std::uint32_t ptr = 0;
                std::memcpy(&ptr, s + o + 4 * c, 4);
                if (ptr != 0xffffffffu && (ptr & 7u) == 1u) ++pairs;     // TRI1
            }
        }
        static_rb->Unmap(0, nullptr);
        if (!pairs || nodes_off < elements_off || (nodes_off - elements_off) / 64 + pairs != triangles) {
            std::fprintf(stderr, "paired BLAS: %u triangles, %u pairs, %u leaf records\n", triangles, pairs,
                         (nodes_off - elements_off) / 64);
            return fail("a paired BLAS does not keep one leaf record a pair");
        }
        std::printf("d3d12_xdxr_rt: paired BLAS %llu bytes, %u triangles in %u leaves (%u bytes saved)\n",
                    static_cast<unsigned long long>(used), triangles, triangles - pairs, 64 * pairs);
    }

    // ---- serialization: the SERIALIZATION postbuild info of the BLAS (as updated) and of the TLAS (as rebuilt for
    // CURRENT_SIZE), SERIALIZE both, DESERIALIZE the BLAS somewhere else, point the TLAS's serialized BLAS pointers
    // at that copy (what D3D12 lets an application do after moving its BLASes), and DESERIALIZE the TLAS. The copies
    // must be the same bytes but for the instances' BLAS addresses, and are dumped for the ray check ----
    {
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for serialization");
        Ref<ID3D12Resource> ser_info, ser_info_rb, tlas_now_rb;
        if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, 32, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, uav,
                               ser_info.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, 32, D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_FLAG_NONE, ser_info_rb.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, sizes[1], D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_FLAG_NONE, tlas_now_rb.put())))
            return fail("serialization info buffers", hr);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC ser_desc{
            ser_info->GetGPUVirtualAddress(), D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION};
        list->EmitRaytracingAccelerationStructurePostbuildInfo(&ser_desc, 2, both);
        list->ResourceBarrier(1, &barrier);
        list->CopyBufferRegion(ser_info_rb.p, 0, ser_info.p, 0, 32);
        list->CopyBufferRegion(compacted_rb.p, 0, blas.p, 0, sizes[0]);      // the BLAS's bytes now, to compare
        list->CopyBufferRegion(tlas_now_rb.p, 0, tlas.p, 0, sizes[1]);
        if (!submit(8)) return fail("serialization postbuild submission");
        D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER expect_hdr[2]{};
        UINT64 ser_sizes[4] = {};                       // BLAS: size, pointers; TLAS: size, pointers
        {
            void* p = nullptr;
            if (FAILED(ser_info_rb->Map(0, nullptr, &p))) return fail("Map serialization info");
            std::memcpy(ser_sizes, p, sizeof ser_sizes);
            ser_info_rb->Unmap(0, nullptr);
        }
        const UINT64 hdr = sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER);
        static_assert(sizeof(D3D12_SERIALIZED_RAYTRACING_ACCELERATION_STRUCTURE_HEADER) == 56, "D3D12 header size");
        // the header, one 8-byte BLAS pointer per instance of the TLAS (none for the BLAS), and the structure
        if (ser_sizes[0] != hdr + sizes[0] || ser_sizes[1] != 0 || ser_sizes[2] != hdr + 3 * 8 + sizes[1] ||
            ser_sizes[3] != 3) {
            std::fprintf(stderr, "SERIALIZATION info %llu %llu %llu %llu, structures %llu %llu\n",
                         static_cast<unsigned long long>(ser_sizes[0]), static_cast<unsigned long long>(ser_sizes[1]),
                         static_cast<unsigned long long>(ser_sizes[2]), static_cast<unsigned long long>(ser_sizes[3]),
                         static_cast<unsigned long long>(sizes[0]), static_cast<unsigned long long>(sizes[1]));
            return fail("SERIALIZATION postbuild info");
        }

        Ref<ID3D12Resource> ser_blas, ser_tlas, ser_blas_rb, ser_tlas_rb, blas2, tlas2, blas2_rb, tlas2_rb, patch;
        if (FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, ser_sizes[0], D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               uav, ser_blas.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, ser_sizes[2], D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               uav, ser_tlas.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, ser_sizes[0], D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_FLAG_NONE, ser_blas_rb.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, ser_sizes[2], D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_FLAG_NONE, ser_tlas_rb.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, sizes[0], as_state, uav, blas2.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_DEFAULT, sizes[1], as_state, uav, tlas2.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, sizes[0], D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_FLAG_NONE, blas2_rb.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_READBACK, sizes[1], D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_FLAG_NONE, tlas2_rb.put())) ||
            FAILED(hr = buffer(device.p, D3D12_HEAP_TYPE_UPLOAD, 3 * 8, D3D12_RESOURCE_STATE_GENERIC_READ,
                               D3D12_RESOURCE_FLAG_NONE, patch.put())))
            return fail("serialization buffers", hr);
        const UINT64 blas2_va = blas2->GetGPUVirtualAddress(), tlas2_va = tlas2->GetGPUVirtualAddress();
        if (blas2_va % 256 || tlas2_va % 256 || blas2_va == blas->GetGPUVirtualAddress())
            return fail("deserialized structures need their own 256-byte aligned addresses");
        {
            void* p = nullptr;
            if (FAILED(patch->Map(0, nullptr, &p))) return fail("Map pointer patch");
            const UINT64 moved_to[3] = {blas2_va, blas2_va, blas2_va};
            std::memcpy(p, moved_to, sizeof moved_to);
            patch->Unmap(0, nullptr);
        }
        // The serialized buffers go through the states D3D12 asks for: SERIALIZE writes them as UNORDERED_ACCESS, the
        // readback copies them as COPY_SOURCE, the pointer patch writes the TLAS's as COPY_DEST, and DESERIALIZE reads
        // them as NON_PIXEL_SHADER_RESOURCE. A UAV barrier alone does not order the patch's copy before DESERIALIZE.
        auto transition = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
            D3D12_RESOURCE_BARRIER t{};
            t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            t.Transition.pResource = r;
            t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            t.Transition.StateBefore = before;
            t.Transition.StateAfter = after;
            list->ResourceBarrier(1, &t);
        };
        if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return fail("Reset for SERIALIZE");
        list->CopyRaytracingAccelerationStructure(ser_blas->GetGPUVirtualAddress(), blas->GetGPUVirtualAddress(),
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE);
        list->CopyRaytracingAccelerationStructure(ser_tlas->GetGPUVirtualAddress(), tlas->GetGPUVirtualAddress(),
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_SERIALIZE);
        transition(ser_blas.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(ser_tlas.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(ser_blas_rb.p, 0, ser_blas.p, 0, ser_sizes[0]);
        list->CopyBufferRegion(ser_tlas_rb.p, 0, ser_tlas.p, 0, ser_sizes[2]);   // as SERIALIZE wrote it
        transition(ser_tlas.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(ser_tlas.p, hdr, patch.p, 0, 3 * 8);              // the TLAS's BLAS pointers
        transition(ser_blas.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(ser_tlas.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->CopyRaytracingAccelerationStructure(blas2_va, ser_blas->GetGPUVirtualAddress(),
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
        list->CopyRaytracingAccelerationStructure(tlas2_va, ser_tlas->GetGPUVirtualAddress(),
                                                  D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_DESERIALIZE);
        list->ResourceBarrier(1, &barrier);
        list->CopyBufferRegion(blas2_rb.p, 0, blas2.p, 0, sizes[0]);
        list->CopyBufferRegion(tlas2_rb.p, 0, tlas2.p, 0, sizes[1]);
        if (!submit(9)) return fail("serialize/deserialize submission");
        if (FAILED(device->GetDeviceRemovedReason())) return fail("device removed after serialization");

        auto read = [](ID3D12Resource* rb, UINT64 size) {
            std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
            void* p = nullptr;
            if (SUCCEEDED(rb->Map(0, nullptr, &p))) {
                std::memcpy(bytes.data(), p, bytes.size());
                rb->Unmap(0, nullptr);
            } else {
                bytes.clear();
            }
            return bytes;
        };
        const auto blas_now = read(compacted_rb.p, sizes[0]), tlas_now = read(tlas_now_rb.p, sizes[1]);
        const auto ser_b = read(ser_blas_rb.p, ser_sizes[0]), ser_t = read(ser_tlas_rb.p, ser_sizes[2]);
        const auto blas2_bytes = read(blas2_rb.p, sizes[0]), tlas2_bytes = read(tlas2_rb.p, sizes[1]);
        if (blas_now.empty() || tlas_now.empty() || ser_b.empty() || ser_t.empty() || blas2_bytes.empty() ||
            tlas2_bytes.empty())
            return fail("serialization readbacks");

        // the headers: this layer's identifier, the sizes the postbuild info gave, the pointer counts
        const UINT64 blas_va = blas->GetGPUVirtualAddress();
        for (int k = 0; k < 2; ++k) {
            const auto& ser = k ? ser_t : ser_b;
            std::memcpy(&expect_hdr[k], ser.data(), sizeof expect_hdr[k]);
            const auto& h = expect_hdr[k];
            if (std::memcmp(&h.DriverMatchingIdentifier, leonida::xdxr_serialized_id, sizeof(leonida::xdxr_serialized_id)) ||
                h.SerializedSizeInBytesIncludingHeader != ser_sizes[2 * k] || h.DeserializedSizeInBytes != sizes[k] ||
                h.NumBottomLevelAccelerationStructurePointersAfterHeader != ser_sizes[2 * k + 1])
                return fail(k ? "serialized TLAS header" : "serialized BLAS header");
            if (device->CheckDriverMatchingIdentifier(D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE,
                                                      &h.DriverMatchingIdentifier) !=
                D3D12_DRIVER_MATCHING_IDENTIFIER_COMPATIBLE_WITH_DEVICE)
                return fail("CheckDriverMatchingIdentifier does not accept the layer's own identifier");
        }
        D3D12_SERIALIZED_DATA_DRIVER_MATCHING_IDENTIFIER other = expect_hdr[0].DriverMatchingIdentifier;
        other.DriverOpaqueVersioningData[4] ^= 1;
        if (device->CheckDriverMatchingIdentifier(D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE, &other) !=
            D3D12_DRIVER_MATCHING_IDENTIFIER_INCOMPATIBLE_VERSION)
            return fail("CheckDriverMatchingIdentifier: another serialized layout version is not INCOMPATIBLE_VERSION");
        other = expect_hdr[0].DriverMatchingIdentifier;
        other.DriverOpaqueGUID.Data1 ^= 1;
        if (device->CheckDriverMatchingIdentifier(D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE, &other) !=
            D3D12_DRIVER_MATCHING_IDENTIFIER_UNRECOGNIZED)
            return fail("CheckDriverMatchingIdentifier: another GUID is not UNRECOGNIZED");
        // the TLAS's pointers, as SERIALIZE wrote them: the BLAS each instance names; then each structure unchanged
        for (int i = 0; i < 3; ++i) {
            UINT64 ptr = 0;
            std::memcpy(&ptr, ser_t.data() + hdr + 8 * i, 8);
            if (ptr != blas_va) return fail("serialized TLAS: an instance's BLAS pointer is not the BLAS's address");
        }
        if (std::memcmp(ser_b.data() + hdr, blas_now.data(), blas_now.size()) ||
            std::memcmp(ser_t.data() + hdr + 3 * 8, tlas_now.data(), tlas_now.size()))
            return fail("a serialized structure's bytes differ from the structure's");

        // the deserialized BLAS is the BLAS; the deserialized TLAS is the TLAS but for each INSTANCE record's BLAS
        // address (+60, in 256-byte units), which is now the deserialized BLAS's
        if (blas2_bytes != blas_now) return fail("the deserialized BLAS differs from the BLAS");
        std::uint32_t elements_off = 0;
        std::memcpy(&elements_off, tlas_now.data() + 52, 4);
        std::size_t patched = 0;
        for (std::size_t w = 0; w + 4 <= tlas_now.size(); w += 4) {
            std::uint32_t was = 0, now = 0;
            std::memcpy(&was, tlas_now.data() + w, 4);
            std::memcpy(&now, tlas2_bytes.data() + w, 4);
            const bool pointer = w >= elements_off && w < elements_off + 3 * 128 && (w - elements_off) % 128 == 60;
            if (pointer) {
                if (was != std::uint32_t(blas_va >> 8) || now != std::uint32_t(blas2_va >> 8)) {
                    std::fprintf(stderr, "instance word at +%zu: %08x (TLAS %08x), want %08x (BLAS %08x)\n", w, now,
                                 was, std::uint32_t(blas2_va >> 8), std::uint32_t(blas_va >> 8));
                    return fail("deserialized TLAS: an instance's BLAS address is not the moved BLAS's");
                }
                ++patched;
            } else if (was != now) {
                std::fprintf(stderr, "deserialized TLAS word at +%zu: %08x, was %08x\n", w, now, was);
                return fail("the deserialized TLAS differs from the TLAS outside the BLAS addresses");
            }
        }
        if (patched != 3) return fail("deserialized TLAS: not every instance's BLAS address was checked");

        if (!dump(blas2_rb.p, sizes[0], "d3d12_blas_deserialized",
                  "{\"kind\": \"blas\", \"pairs\": false, \"address\": " + std::to_string(blas2_va) +
                      ", \"triangles\": " + triangles_json(moved) + "}\n") ||
            !dump(tlas2_rb.p, sizes[1], "d3d12_tlas_deserialized",
                  "{\"kind\": \"tlas\", \"address\": " + std::to_string(tlas2_va) +
                      ", \"bias\": 0, \"blas\": [\"d3d12_blas_deserialized\"], \"instances\": " + inst + "}\n"))
            return fail("deserialized structures readback");
        std::printf("d3d12_xdxr_rt: serialized BLAS %llu bytes, TLAS %llu bytes (3 BLAS pointers); deserialized "
                    "with the TLAS's BLAS moved\n",
                    static_cast<unsigned long long>(ser_sizes[0]), static_cast<unsigned long long>(ser_sizes[2]));
    }
    std::printf("d3d12_xdxr_rt: BLAS %llu bytes (prebuild %llu), TLAS (prebuild %llu), written to %s\n",
                static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                static_cast<unsigned long long>(blas_info.ResultDataMaxSizeInBytes),
                static_cast<unsigned long long>(tlas_info.ResultDataMaxSizeInBytes), dir.c_str());
    return 0;
}
