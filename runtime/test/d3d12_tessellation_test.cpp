// Exercising one D3D12 PSO with 3- and 4-point patches verifies the Vulkan pipeline variants. The hull shader
// drops a patch if gl_PatchVerticesIn differs from the root constant, so a stale variant leaves a half unpainted.
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <array>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
static_assert(leonida::detail::patch_control_points(D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST) == 1);
static_assert(leonida::detail::patch_control_points(D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST) == 32);

int main() {
    vktest::Device gpu;
    const auto status = gpu.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    if (!gpu.enabled.tessellationShader) return 77;

    const auto library = std::filesystem::temp_directory_path() /
                         ("leonida_tessellation_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(library);
    const std::array stages{"vs", "hs", "ds", "ps"};
    const std::array paths{LEONIDA_TEST_VS_SPV, LEONIDA_TEST_HS_SPV, LEONIDA_TEST_DS_SPV, LEONIDA_TEST_PS_SPV};
    std::array<std::vector<std::uint8_t>, 4> containers;
    std::array<D3D12_SHADER_BYTECODE, 4> codes{};
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << '{';
        for (unsigned i = 0; i < stages.size(); ++i) {
            const auto program = fssm_program({i, 0xbf810000});
            containers[i] = shader_container(program);
            codes[i] = {containers[i].data(), containers[i].size()};
            const char* stage = stages[i];
            std::filesystem::copy_file(paths[i], library / (std::string(stage) + ".spv"),
                                       std::filesystem::copy_options::overwrite_existing);
            if (i) index << ',';
            index << '"' << stage << "\":{\"stage\":\"" << stage << "\",\"fssm_sha256\":\""
                  << leonida::sha256_hex(program) << "\",\"bindings\":[],\"push_constant_bytes\":144"
                  << (i == 3 ? ",\"outputs\":[0]}" : "}");
        }
        index << '}';
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gpu.instance_proc;
    info.instance = gpu.instance;
    info.physical = gpu.physical;
    info.device = gpu.device;
    info.queue = gpu.queue;
    info.queue_family = gpu.family;
    info.buffer_device_address = true;
    info.features = gpu.enabled;
    info.shader_dir = library.string();
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));

    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants = {0, 0, 1};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
    const auto blob = serialize_root_signature(root_desc);
    EXPECT(!blob.empty());
    Ref<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root.p;
    pso.VS = codes[0];
    pso.HS = codes[1];
    pso.DS = codes[2];
    pso.PS = codes[3];
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = ~0u;
    pso.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> tessellation, invalid;
    EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&tessellation.p))));
    auto wrong = pso;
    wrong.DS = {};
    EXPECT(device->CreateGraphicsPipelineState(&wrong, IID_PPV_ARGS(&invalid.p)) == E_INVALIDARG && !invalid.p);
    wrong = pso;
    wrong.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    EXPECT(device->CreateGraphicsPipelineState(&wrong, IID_PPV_ARGS(&invalid.p)) == E_INVALIDARG && !invalid.p);

    constexpr UINT size = 32, pitch = 256;
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC image{};
    image.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    image.Width = image.Height = size;
    image.DepthOrArraySize = image.MipLevels = 1;
    image.Format = pso.RTVFormats[0];
    image.SampleDesc.Count = 1;
    image.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const auto readback_desc = buffer_desc(size * pitch);
    Ref<ID3D12Resource> target, readback;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &image,
        D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&target.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));

    const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p,
                                               tessellation.p, IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    const float black[4] = {0, 0, 0, 1};
    list->ClearRenderTargetView(rtv, black, 0, nullptr);
    list->SetGraphicsRootSignature(root.p);
    const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
    list->RSSetViewports(1, &viewport);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    const auto draw_half = [&](D3D12_PRIMITIVE_TOPOLOGY topology, UINT count, const D3D12_RECT& scissor) {
        list->IASetPrimitiveTopology(topology);
        list->RSSetScissorRects(1, &scissor);
        list->SetGraphicsRoot32BitConstant(0, count, 0);
        list->DrawInstanced(count, 1, 0, 0);
    };
    draw_half(D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST, 3, {0, 0, size / 2, size});
    draw_half(D3D_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST, 4, {size / 2, 0, size, size});

    const auto barrier = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target.p;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {0, {image.Format, size, size, 1, pitch}};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[]{list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* pixels = static_cast<const std::uint8_t*>(mapped);
    constexpr std::array<std::uint8_t, 4> color{32, 160, 240, 255};
    for (UINT y = 0; y < size; ++y)
        for (UINT x = 0; x < size; ++x)
            EXPECT(std::memcmp(pixels + y * pitch + x * 4, color.data(), color.size()) == 0);
    readback->Unmap(0, nullptr);

    EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, tessellation.p)));
    list->SetGraphicsRootSignature(root.p);
    list->RSSetViewports(1, &viewport);
    draw_half(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, 3, {0, 0, size, size});
    EXPECT(list->Close() == S_OK);
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_draw);

    EXPECT(!gpu.validation_errors());
    std::filesystem::remove_all(library);
    return 0;
}
