// Real GPU readback of fitting and spilling blocks, partial setters, unchanged-signature rebinds, direct and
// indirect dispatch snapshots, repeated execution and two logical queues. No game files or host D3D12 needed
// beyond its root-signature serializer. LEONIDA_VK_VALIDATION=1 enables validation where installed.
#include "vk_test_device.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include "../src/title_va_alias.h"
#include <array>
#include <cstring>

#define EXPECT(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
using namespace d3d12test;

int main() {
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    if (!vk.buffer_device_address) return 77;
    const testfiles::ScratchDir scratch("leonida_root_arguments_test");
    const auto& dir = scratch.path();
    const auto fitting_program = fssm_program({0x1100u, 0x9812u, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    const auto spilling_program = fssm_program({0x1100u, 0x9813u, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_ROOT_SMALL_SPV), dir / "small.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_ROOT_LARGE_SPV), dir / "large.spv"));
    {
        std::ofstream index(dir / "index.json", std::ios::binary);
        index << "{";
        for (unsigned i = 0; i < 2; ++i)
            index << (i ? "," : "") << "\"" << (i ? "large" : "small")
                  << "\":{\"stage\":\"cs\",\"fssm_sha256\":\"" << leonida::sha256_hex(i ? spilling_program : fitting_program)
                  << "\",\"bindings\":[],\"push_constant_bytes\":144,\"local_size\":[1,1,1],"
                     "\"uses_code_base\":false,\"wave64\":false}";
        index << "}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    info.shader_dir = dir.string();
    Ref<ID3D12Device> dev;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev.p))));
    const auto buffer = [&](Ref<ID3D12Resource>& out, UINT bytes, D3D12_HEAP_TYPE type,
                            D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        const D3D12_HEAP_PROPERTIES heap{type};
        const auto desc = buffer_desc(bytes, flags);
        return SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                       IID_PPV_ARGS(&out.p)));
    };
    for (UINT count : {8u, 27u}) {
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        std::array<D3D12_ROOT_PARAMETER, 4> params{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, count};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &range};
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[3].Descriptor = {0, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {4, params.data(), 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto serialized = serialize_root_signature(root_desc);
        EXPECT(!serialized.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(dev->CreateRootSignature(0, serialized.data(), serialized.size(), IID_PPV_ARGS(&root.p))));
        const auto* layout = rt::d3d12::root_signature_layout(root.p);
        EXPECT(layout && layout->compute_ok && layout->block_dwords == count + 5);
        EXPECT(layout->parameters[2].table_descriptors == 1);
        EXPECT(layout->parameters[2].block_offset == count + 2);
        const auto code = shader_container(count == 8 ? fitting_program : spilling_program);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {code.data(), code.size()};
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(dev->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        const D3D12_INDIRECT_ARGUMENT_DESC arg{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
        const D3D12_COMMAND_SIGNATURE_DESC sig_desc{12, 1, &arg, 0};
        Ref<ID3D12CommandSignature> signature;
        EXPECT(SUCCEEDED(dev->CreateCommandSignature(&sig_desc, nullptr, IID_PPV_ARGS(&signature.p))));
        Ref<ID3D12Resource> indirect;
        EXPECT(buffer(indirect, 16, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(indirect->Map(0, nullptr, &mapped)));
        const UINT indirect_words[] = {2, 3, 4, 1}; // dimensions, then count
        std::memcpy(mapped, indirect_words, sizeof indirect_words);
        indirect->Unmap(0, nullptr);
        std::array<Ref<ID3D12CommandQueue>, 2> queues;
        std::array<Ref<ID3D12CommandAllocator>, 2> allocators;
        std::array<Ref<ID3D12GraphicsCommandList>, 2> lists;
        std::array<Ref<ID3D12Fence>, 2> fences;
        std::array<Ref<ID3D12Resource>, 2> outputs, readbacks;
        std::array<std::array<std::array<UINT, 64>, 2>, 2> expected{};
        for (UINT q = 0; q < 2; ++q) {
            const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
            EXPECT(SUCCEEDED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queues[q].p))));
            EXPECT(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocators[q].p))));
            EXPECT(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[q].p,
                                                   pso.p, IID_PPV_ARGS(&lists[q].p))));
            EXPECT(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fences[q].p))));
            EXPECT(buffer(outputs[q], 512, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
            EXPECT(buffer(readbacks[q], 512, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST));
            auto* list = lists[q].p;
            list->SetComputeRootSignature(root.p);
            for (UINT n = 0; n < 2; ++n) {
                auto& want = expected[q][n];
                for (UINT i = 0; i < count; ++i) want[i] = 0x1000 + q * 0x100 + n * 0x10 + i;
                list->SetComputeRoot32BitConstants(0, count, want.data(), 0);
                want[3] = 0xabcd + q + n;
                list->SetComputeRoot32BitConstant(0, want[3], 3);
                want[5] = 0x7123 + q + n;
                want[6] = 0x8123 + q + n;
                list->SetComputeRoot32BitConstants(0, 2, want.data() + 5, 5);
                const UINT64 cbv = 0x12345678abcdef00ull + q * 256 + n * 16;
                // Main zeros unbacked root views. Keep that negative control, then register a synthetic,
                // value-only alias to exercise both address halves. The shader never dereferences this CBV.
                const UINT64 shader_cbv = n ? 0x12abcdef00ull + q * 256 + n * 16 : 0;
                if (n) EXPECT(leonida::title_va_aliases().add(cbv, 65536, shader_cbv));
                list->SetComputeRootConstantBufferView(1, cbv);
                if (n) leonida::title_va_aliases().remove(cbv, shader_cbv);
                want[count] = UINT(shader_cbv);
                want[count + 1] = UINT(shader_cbv >> 32);
                want[count + 2] = 0x34567890 + q + n;
                list->SetComputeRootDescriptorTable(2, {want[count + 2]});
                const UINT64 dest = outputs[q]->GetGPUVirtualAddress() + n * 256;
                list->SetComputeRootUnorderedAccessView(3, dest);
                want[count + 3] = UINT(dest);
                want[count + 4] = UINT(dest >> 32);
                list->SetComputeRootSignature(root.p); // must preserve all values
                // The inline model: constants fit; with count 27 all later parameters are block-only.
                for (UINT i = 0; i < (count == 27 ? count : count + 5); ++i) want[32 + i] = want[i];
                want[60] = 2; want[61] = 3; want[62] = 4;
                if (!q) list->Dispatch(2, 3, 4);
                else list->ExecuteIndirect(signature.p, 1, indirect.p, 0, n ? indirect.p : nullptr, n ? 12 : 0);
            }
            list->SetComputeRoot32BitConstant(0, 0xdeadbeef, 0); // cannot change earlier snapshots
            const auto to_copy = transition(outputs[q].p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &to_copy);
            list->CopyBufferRegion(readbacks[q].p, 0, outputs[q].p, 0, 512);
            const auto back = transition(outputs[q].p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->ResourceBarrier(1, &back);
            EXPECT(SUCCEEDED(list->Close()));
        }
        for (UINT run = 1; run <= 2; ++run) {
            for (UINT q = 0; q < 2; ++q) {
                ID3D12CommandList* batch[] = {lists[q].p};
                queues[q]->ExecuteCommandLists(1, batch);
                EXPECT(rt::d3d12::last_execute_result(queues[q].p) == leonida::ReplayResult::success);
                EXPECT(SUCCEEDED(queues[q]->Signal(fences[q].p, run)));
            }
            for (UINT q = 0; q < 2; ++q) {
                HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
                EXPECT(done);
                const bool waited = SUCCEEDED(fences[q]->SetEventOnCompletion(run, done)) &&
                                    WaitForSingleObject(done, 20000) == WAIT_OBJECT_0;
                CloseHandle(done);
                EXPECT(waited);
                EXPECT(SUCCEEDED(readbacks[q]->Map(0, nullptr, &mapped)));
                const auto* words = static_cast<const UINT*>(mapped);
                // Only written words: the argument block, all 28 inline words, dimensions and padding.
                for (UINT n = 0; n < 2; ++n)
                    for (UINT i = 0; i < 64; ++i) {
                        if (i >= count + 5 && i < 32) continue;
                        if (words[n * 64 + i] != expected[q][n][i]) {
                            std::fprintf(stderr, "constants=%u run=%u queue=%u dispatch=%u word=%u: %08x != %08x\n",
                                         count, run, q, n, i, words[n * 64 + i], expected[q][n][i]);
                            return 1;
                        }
                    }
                readbacks[q]->Unmap(0, nullptr);
            }
        }
    }
    EXPECT(vk.validation_errors() == 0);
    std::puts("root arguments: fitting/spilling, direct/indirect/counted, two queues, repeat execution: OK");
    return 0;
}
