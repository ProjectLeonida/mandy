// A T# the title writes into a console descriptor heap (Xbox slot 14) before its texture exists is bound when the
// texture appears, at the next submission, through the heap's slot records (docs/d3d12_vulkan.md section 3.4): no
// pass looks at waiting T#s again on its own. Through D3D12 alone, on the Xbox-ABI proxy over the Vulkan device:
//   - a probe texture gives the T# words of an 8x8 RGBA8 texture and the address the layer gives textures (the next
//     one is 256 bytes on, Resource::image_address);
//   - a console heap of 60,001 slots, written before the heap is made: 60,000 T#s naming addresses no texture will
//     ever have, then, last, the T# of the texture to come. All wait after the heap's first refresh;
//   - the texture is made at the address the T# names, filled by a copy, and sampled through that slot by the
//     test/glsl/cs_set0_sample.comp shader (d3d12_set0_compute_test) in the same submission. Its texels must come
//     back exactly. With a pass that tries waiting T#s round robin under a time budget, the last of 60,001 is not
//     reached in that submission and the shader reads the fallback image;
//   - the texture goes: its T# leaves the descriptor hash table at once, and the slot waits again.
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation (0 errors required). No game data: the program
// is two marker words and the module is GLSL written to the translator's contract.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_CS_SET0_SAMPLE_SPV
#error "LEONIDA_TEST_CS_SET0_SAMPLE_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

using Texel = std::array<std::uint8_t, 4>;
constexpr UINT texture_size = 8, texels = texture_size * texture_size, run_bytes = texels * 4;
constexpr UINT waiting = 60000;  // T#s that never find a texture, ahead of the one that will
constexpr UINT kDescriptor = 32;

Texel pattern(UINT x, UINT y) {
    return {std::uint8_t(200 - x * 20), std::uint8_t(30 + y * 25), std::uint8_t((x * 8 + y) * 3), 255};
}

// The T# words `words` with their base address replaced (dword 0: address bits 8..39, dword 1 bits 0..7: 40..47).
std::array<std::uint32_t, 8> at_address(std::array<std::uint32_t, 8> words, std::uint64_t address) {
    words[0] = std::uint32_t(address >> 8);
    words[1] = (words[1] & ~0xffu) | (std::uint32_t(address >> 40) & 0xffu);
    return words;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }

    const auto program = fssm_program({0x43530003, 0xbf810000});
    const auto container = shader_container(program);
    const testfiles::ScratchDir scratch("leonida_tsharp_refill_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SET0_SAMPLE_SPV),
                                library / "cs_set0_sample.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_set0_sample\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program)
              << "\", \"bindings\": [0, 1, 2], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": false, \"wave64\": false}}";
    }

    std::vector<std::uint8_t> results(run_bytes);
    std::array<std::uint32_t, 8> refill_words{};
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.image_view_min_lod = d.image_view_min_lod;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p), /*console_lists=*/false)));
        rt::d3d12::set_device_outer(inner, device.p);
        if (rt::d3d12::device_push_constants_need_fallback(inner)) {
            std::printf("d3d12_tsharp_refill_test: skipped, the PSO needs 144 push-constant bytes and this GPU has %u\n",
                        rt::d3d12::device_max_push_constants_size(inner));
            return 77;
        }

        // Root signature and PSO as in d3d12_set0_compute_test: s4 the view's offset in dwords, s5 the first output
        // element, s6:s7 the output, s8 the view table, s9 the sampler table.
        D3D12_DESCRIPTOR_RANGE view_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[4] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor = {0, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &view_range};
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[3].DescriptorTable = {1, &sampler_range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {4, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container.data(), container.size()};
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        // Every buffer first: from here on the only textures made are the probe and the one to come.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        Ref<ID3D12Resource> upload, output, readback;
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto output_desc = buffer_desc(results.size(), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &output_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&output.p))));
        const auto readback_desc = buffer_desc(results.size());
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        const D3D12_DESCRIPTOR_HEAP_DESC view_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&view_heap_desc, IID_PPV_ARGS(&view_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t view_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 9))(view_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        const std::uint64_t sampler_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 10))(sampler_heap.p);
        EXPECT(view_cpu && sampler_cpu && sampler_gpu < (1ull << 32));
        D3D12_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});

        // The probe: the T# words of an 8x8 RGBA8 texture, and where the layer put it.
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = texture_desc.Height = texture_size;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        Ref<ID3D12Resource> probe;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&probe.p))));
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D = {0, 1, 0, 0.0f};
        device->CreateShaderResourceView(probe.p, &srv, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(view_cpu)});
        std::array<std::uint32_t, 8> probe_words{};
        std::memcpy(probe_words.data(), reinterpret_cast<const void*>(view_cpu), sizeof(probe_words));
        const std::uint64_t probe_address = rt::d3d12::resource_image_address(probe.p);
        EXPECT(probe_address && (probe_words[0] == std::uint32_t(probe_address >> 8)));
        const std::uint64_t refill_address = probe_address + 256;
        refill_words = at_address(probe_words, refill_address);

        // The console heap, written before it is made: 60,000 T#s at addresses no texture will have (a range of the
        // address space the layer never gives textures, 256 apart), then the one to come, last.
        // After it, the same texture's T# as the title rebuilds it from 32 bits (run 58: 0x7f0000021400 for the
        // layer's 0x100021400, and the low 32 bits alone): both must wait for the texture and bind when it comes.
        constexpr UINT slots = waiting + 3;
        constexpr std::uint64_t bytes = std::uint64_t(slots) * kDescriptor;
        auto* title = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        EXPECT(title);
        for (UINT i = 0; i < waiting; ++i) {
            const auto words = at_address(probe_words, 0x7e0000000000ull + std::uint64_t(i) * 256);
            std::memcpy(title + std::size_t(i) * kDescriptor, words.data(), sizeof(words));
        }
        std::memcpy(title + std::size_t(waiting) * kDescriptor, refill_words.data(), sizeof(refill_words));
        const auto title_form = at_address(probe_words, 0x7f0000000000ull | (refill_address & 0xffffffffull));
        const auto low_form = at_address(probe_words, refill_address & 0xffffffffull);
        std::memcpy(title + std::size_t(waiting + 1) * kDescriptor, title_form.data(), sizeof(title_form));
        std::memcpy(title + std::size_t(waiting + 2) * kDescriptor, low_form.data(), sizeof(low_form));
        std::uint32_t title_key[8], low_key[8];
        std::memcpy(title_key, title_form.data(), sizeof(title_key));
        std::memcpy(low_key, low_form.data(), sizeof(low_key));
        const D3D12_DESCRIPTOR_HEAP_DESC console_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                                                      D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> console_heap;
        EXPECT(SUCCEEDED(rt::d3d12::create_console_descriptor_heap(device.p, &console_desc,
                                                                   reinterpret_cast<std::uint64_t>(title), bytes,
                                                                   IID_PPV_ARGS(&console_heap.p))));
        D3D12_DESCRIPTOR_HEAP_DESC got{};
        D3D12_GPU_VIRTUAL_ADDRESS shader_va = 0;
        EXPECT(rt::d3d12::descriptor_heap_info(console_heap.p, got, shader_va));
        const std::uint64_t refill_slot_va = shader_va + std::uint64_t(waiting) * kDescriptor;
        EXPECT(refill_slot_va < (1ull << 32));
        std::uint32_t key[8];
        std::memcpy(key, refill_words.data(), sizeof(key));
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) == 0);  // no texture yet: the slot waits
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, title_key) == 0);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, low_key) == 0);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        // One empty submission: the waiting T#s have had a refresh with nothing new (as a running title's would).
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) == 0);

        // The texture comes, at the address the T# names.
        Ref<ID3D12Resource> texture;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&texture.p))));
        EXPECT(rt::d3d12::resource_image_address(texture.p) == refill_address);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        EXPECT(footprint.Footprint.RowPitch * texture_size <= 4096);
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        for (UINT y = 0; y < texture_size; ++y)
            for (UINT x = 0; x < texture_size; ++x)
                std::memcpy(up + y * footprint.Footprint.RowPitch + x * 4, pattern(x, y).data(), 4);

        // One submission: fill the texture and sample it through the waiting slot.
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = texture.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = upload.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        D3D12_RESOURCE_BARRIER sampled =
            transition(texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &sampled);
        ID3D12DescriptorHeap* heaps[] = {view_heap.p, sampler_heap.p};
        list->SetDescriptorHeaps(2, heaps);
        list->SetComputeRootSignature(root.p);
        list->SetPipelineState(pso.p);
        const D3D12_GPU_VIRTUAL_ADDRESS output_va = output->GetGPUVirtualAddress();
        EXPECT(output_va < (1ull << 32));
        list->SetComputeRootUnorderedAccessView(1, output_va);
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{refill_slot_va});
        list->SetComputeRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu});
        const std::uint32_t offset_and_first[2] = {0, 0};
        list->SetComputeRoot32BitConstants(0, 2, offset_and_first, 0);
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER to_copy = transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        list->CopyBufferRegion(readback.p, 0, output.p, 0, results.size());
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 2)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2, nullptr)));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        std::memcpy(results.data(), mapped, results.size());
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) != 0);
        // The title's 32-bit forms of the same address were fixed at that submission too.
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, title_key) != 0);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, low_key) != 0);

        // The texture goes: its T# leaves the hash table at once (a shader reading the slot gets the fallback), and
        // the next submission takes the slot's record back to waiting.
        texture.reset();
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) == 0);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, title_key) == 0);
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, low_key) == 0);
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 3)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(3, nullptr)));
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, key) == 0);

        for (auto* r : {&probe, &upload, &output, &readback}) r->reset();
        pso.reset();
        root.reset();
        view_heap.reset();
        sampler_heap.reset();
        console_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        VirtualFree(title, 0, MEM_RELEASE);
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    unsigned mismatches = 0;
    for (UINT i = 0; i < texels; ++i) {
        const UINT x = i % texture_size, y = i / texture_size;
        const Texel want = pattern(x, y);
        const std::uint8_t* got = results.data() + i * 4;
        if (!std::memcmp(got, want.data(), 4)) continue;
        if (mismatches++ < 8)
            std::fprintf(stderr, "texel %u: got %u,%u,%u,%u want %u,%u,%u,%u\n", i, got[0], got[1], got[2], got[3],
                         want[0], want[1], want[2], want[3]);
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("T# refill: the last of %u waiting T#s sampled its texture, all %u texels, at the first submission "
                "after the texture appeared; its T# left the hash table when the texture went\n", waiting + 1, texels);
    return 0;
}
