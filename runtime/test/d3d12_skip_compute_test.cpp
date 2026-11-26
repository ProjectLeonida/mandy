// LEONIDA_SKIP_COMPUTE (diagnostic, unset by default): a compute program whose library name contains one of the
// listed parts is not built, and the dispatches that name it, direct and indirect, are dropped while the rest of the
// command list runs on the GPU.
//
// Two synthetic programs, "count_keep" and "count_drop", share one GLSL module that counts its dispatches in the
// buffer its root UAV names. The variable is read once per process, so each case is a run of its own:
//   (no argument)  the variable is unset: both programs run and nothing is logged;
//   --skip-one     "absent,drop": only count_drop is skipped, and the dispatches around it still run;
//   --skip-all     "all": both are skipped.
// Both pipeline states are created in every case. A skipped program's buffer keeps the values it was seeded with.
// No game data. Set LEONIDA_VK_VALIDATION=1 to enable Vulkan core and synchronization validation.
#include "vk_test_device.h"
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <iterator>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_SKIP_COMPUTE_SPV
#error "LEONIDA_TEST_SKIP_COMPUTE_SPV must name the synthetic GLSL compute module"
#endif

using namespace d3d12test;
namespace {
constexpr UINT kStride = 20, kArgsOffset = 32, kCountOffset = 44, kRecords = 2;
constexpr UINT kSeed[2] = {100, 1000};  // what both output buffers hold before any dispatch

// The module adds 1 to word 0 for each group and the group's index + 1 to word 1.
void add_groups(std::array<UINT, 2>& words, UINT groups) {
    words[0] += groups;
    words[1] += groups * (groups + 1) / 2;
}

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// What the runtime writes to stderr between the constructor and finish(), which also passes it on to the real stderr.
class StderrCapture {
public:
    explicit StderrCapture(const std::filesystem::path& file) : path_(file) {
        std::fflush(stderr);
        std::FILE* out = std::fopen(path_.string().c_str(), "wb");
        saved_ = _dup(2);
        if (out && saved_ >= 0) _dup2(_fileno(out), 2);
        if (out) std::fclose(out);
    }
    std::string finish() {
        if (saved_ < 0) return {};
        std::fflush(stderr);
        _dup2(saved_, 2);
        _close(saved_);
        saved_ = -1;
        std::ifstream in(path_, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        std::fputs(text.c_str(), stderr);
        return text;
    }
    ~StderrCapture() { finish(); }

private:
    std::filesystem::path path_;
    int saved_ = -1;
};

// The lines of `text` that name the variable, and whether one of them names `program`.
unsigned skip_lines(const std::string& text, const char* program, bool& names_program) {
    unsigned lines = 0;
    names_program = false;
    for (std::size_t at = 0; at < text.size();) {
        const std::size_t end = std::min(text.find('\n', at), text.size());
        const std::string line = text.substr(at, end - at);
        if (line.find("LEONIDA_SKIP_COMPUTE") != std::string::npos) {
            ++lines;
            names_program = names_program || line.find(program) != std::string::npos;
        }
        at = end + 1;
    }
    return lines;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (argc > 2 || (!mode.empty() && mode != "--skip-one" && mode != "--skip-all")) {
        std::fprintf(stderr, "usage: d3d12_skip_compute_test [--skip-one | --skip-all]\n");
        return 2;
    }
    // Set before the device exists: the runtime reads the list at the first compute pipeline it creates.
    if (!SetEnvironmentVariableA("LEONIDA_SKIP_COMPUTE",
                                 mode == "--skip-one" ? "absent,drop" : mode == "--skip-all" ? "all" : nullptr))
        if (!mode.empty() || GetLastError() != ERROR_ENVVAR_NOT_FOUND) return fail("set LEONIDA_SKIP_COMPUTE");
    const bool keep_runs = mode != "--skip-all", drop_runs = mode.empty();

    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return fail("create the Vulkan test device");
    if (!vk.buffer_device_address) return 77;

    // Two programs with different hashes, each with its own name in the index and its own copy of the module.
    std::vector<std::uint8_t> keep_program = fssm_program({0x1100u, 0x51u, 0x4bu, 0x45u});
    std::vector<std::uint8_t> drop_program = fssm_program({0x1100u, 0x51u, 0x44u, 0x52u});
    for (UINT i = 0; i < 12; ++i) {
        put32(keep_program, 0xbf810000 + i);
        put32(drop_program, 0xbf810000 + i);
    }
    const testfiles::ScratchDir scratch("leonida_skip_compute");  // removed after the device that reads it
    const std::filesystem::path& fixture = scratch.path();
    for (const char* name : {"count_keep.spv", "count_drop.spv"})
        if (!testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_SKIP_COMPUTE_SPV), fixture / name))
            return fail("copy the module into the library folder");
    {
        const auto entry = [](const char* name, const std::vector<std::uint8_t>& program) {
            return std::string("\"") + name + "\":{\"stage\":\"cs\",\"fssm_sha256\":\"" + leonida::sha256_hex(program) +
                   "\",\"bindings\":[],\"push_constant_bytes\":144,\"local_size\":[1,1,1],"
                   "\"uses_code_base\":false,\"wave64\":false}";
        };
        std::ofstream index(fixture / "index.json", std::ios::binary);
        index << "{" << entry("count_keep", keep_program) << "," << entry("count_drop", drop_program) << "}";
    }
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = vk.buffer_device_address;
    info.shader_dir = fixture.string();
    Ref<ID3D12Device> dev;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&dev.p)))) return fail("create the layer's device");
    // The pipeline states declare the 144-byte push-constant block; a device under that (128, the Vulkan minimum,
    // as on some AMD iGPUs) refuses them, which is a device limit, not a wrong result: SKIP like the other
    // d3d12_* tests that check device_push_constants_need_fallback (77 is the ctest skip code here).
    if (rt::d3d12::device_push_constants_need_fallback(dev.p)) {
        std::printf("d3d12_skip_compute_test: skipped, the pipelines need 144 push-constant bytes and this GPU has %u\n",
                    rt::d3d12::device_max_push_constants_size(dev.p));
        return 77;
    }

    Ref<ID3D12RootSignature> root;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    param.Descriptor = {0, 0};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto serialized = serialize_root_signature(root_desc);
    if (serialized.empty() ||
        FAILED(dev->CreateRootSignature(0, serialized.data(), serialized.size(), IID_PPV_ARGS(&root.p))))
        return fail("create the root signature");
    // A skipped program still gets its pipeline state: the title must not see a failure it would not see otherwise.
    // Each skipped program is named in the log once, when its first dispatch is replayed; with the variable unset nothing is.
    const auto keep_code = shader_container(keep_program), drop_code = shader_container(drop_program);
    Ref<ID3D12PipelineState> keep, drop;
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.p;
    StderrCapture capture(fixture / "stderr.txt");
    desc.CS = {keep_code.data(), keep_code.size()};
    const HRESULT keep_created = dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&keep.p));
    desc.CS = {drop_code.data(), drop_code.size()};
    const HRESULT drop_created = dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&drop.p));
    if (FAILED(keep_created)) return fail("create count_keep");
    if (FAILED(drop_created)) return fail("create count_drop");

    const D3D12_INDIRECT_ARGUMENT_DESC argument{D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH};
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{kStride, 1, &argument, 0};
    Ref<ID3D12CommandSignature> signature;
    if (FAILED(dev->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature.p))))
        return fail("create the command signature");
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    Ref<ID3D12Resource> source, args, counts, keep_out, drop_out, readback;
    const auto create = [&](Ref<ID3D12Resource>& dest, D3D12_HEAP_PROPERTIES heap, D3D12_RESOURCE_STATES state,
                            D3D12_RESOURCE_FLAGS flags) {
        const auto buffer = buffer_desc(256, flags);
        return SUCCEEDED(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, state, nullptr,
                                                      IID_PPV_ARGS(&dest.p)));
    };
    if (!create(source, upload_heap, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE) ||
        !create(args, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE) ||
        !create(counts, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE) ||
        !create(keep_out, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        !create(drop_out, default_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) ||
        !create(readback, readback_heap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE))
        return fail("create the buffers");
    // The upload buffer: the seed of both outputs, two DISPATCH records (3 groups, then 1) and the count 2.
    std::array<UINT, 64> input{};
    input[0] = kSeed[0];
    input[1] = kSeed[1];
    const UINT record_groups[kRecords] = {3, 1};
    for (UINT i = 0; i < kRecords; ++i) {
        const UINT base = (kArgsOffset + i * kStride) / 4;
        input[base] = record_groups[i];
        input[base + 1] = input[base + 2] = 1;
    }
    input[40] = kRecords;  // source offset 160, copied to the count buffer at kCountOffset
    void* mapped = nullptr;
    if (FAILED(source->Map(0, nullptr, &mapped)) || !mapped) return fail("map the upload buffer");
    std::memcpy(mapped, input.data(), sizeof(input));
    source->Unmap(0, nullptr);

    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_COMPUTE};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(dev->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.p, keep.p,
                                      IID_PPV_ARGS(&list.p))) ||
        FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return fail("create the queue and the list");
    list->CopyBufferRegion(args.p, kArgsOffset, source.p, kArgsOffset, kRecords * kStride);
    list->CopyBufferRegion(counts.p, kCountOffset, source.p, 160, 4);
    list->CopyBufferRegion(keep_out.p, 0, source.p, 0, 8);
    list->CopyBufferRegion(drop_out.p, 0, source.p, 0, 8);
    const D3D12_RESOURCE_BARRIER ready[] = {
        transition(args.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(counts.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        transition(keep_out.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        transition(drop_out.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
    list->ResourceBarrier(4, ready);
    list->SetComputeRootSignature(root.p);
    std::array<UINT, 2> keep_want{kSeed[0], kSeed[1]}, drop_want{kSeed[0], kSeed[1]};
    // count_keep, direct.
    list->SetPipelineState(keep.p);
    list->SetComputeRootUnorderedAccessView(0, keep_out->GetGPUVirtualAddress());
    list->Dispatch(2, 1, 1);
    if (keep_runs) add_groups(keep_want, 2);
    // count_drop: direct, indirect, and indirect with a GPU count.
    list->SetPipelineState(drop.p);
    list->SetComputeRootUnorderedAccessView(0, drop_out->GetGPUVirtualAddress());
    list->Dispatch(2, 1, 1);
    list->ExecuteIndirect(signature.p, 1, args.p, kArgsOffset, nullptr, 0);
    list->ExecuteIndirect(signature.p, kRecords, args.p, kArgsOffset, counts.p, kCountOffset);
    if (drop_runs) {
        add_groups(drop_want, 2);
        add_groups(drop_want, record_groups[0]);
        for (const UINT groups : record_groups) add_groups(drop_want, groups);
    }
    // count_keep again: a dropped dispatch leaves the commands after it in place.
    list->SetPipelineState(keep.p);
    list->SetComputeRootUnorderedAccessView(0, keep_out->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    if (keep_runs) add_groups(keep_want, 1);
    const D3D12_RESOURCE_BARRIER done[] = {
        transition(keep_out.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(drop_out.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    list->ResourceBarrier(2, done);
    list->CopyBufferRegion(readback.p, 0, keep_out.p, 0, 8);
    list->CopyBufferRegion(readback.p, 8, drop_out.p, 0, 8);
    if (FAILED(list->Close())) return fail("close the list");
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success)
        return fail("the list is replayed, with or without the dropped dispatches");
    if (FAILED(queue->Signal(fence.p, 1))) return fail("signal the fence");
    const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return fail("create the event");
    const bool completed = SUCCEEDED(fence->SetEventOnCompletion(1, event)) &&
                           WaitForSingleObject(event, 20000) == WAIT_OBJECT_0;
    CloseHandle(event);
    if (!completed || FAILED(readback->Map(0, nullptr, &mapped)) || !mapped) return fail("read the results back");
    std::array<UINT, 2> keep_got{}, drop_got{};
    std::memcpy(keep_got.data(), mapped, sizeof(keep_got));
    std::memcpy(drop_got.data(), static_cast<const char*>(mapped) + 8, sizeof(drop_got));
    readback->Unmap(0, nullptr);

    const std::string log = capture.finish();
    bool names_keep = false, names_drop = false;
    const unsigned logged = skip_lines(log, "count_keep", names_keep);
    skip_lines(log, "count_drop", names_drop);
    const bool log_ok = logged == (keep_runs ? 0u : 1u) + (drop_runs ? 0u : 1u) && names_keep != keep_runs &&
                        names_drop != drop_runs;

    std::printf("LEONIDA_SKIP_COMPUTE %s: %u log line(s), count_keep %u/%u (expected %u/%u), count_drop %u/%u "
                "(expected %u/%u)\n",
                mode.empty() ? "unset" : mode == "--skip-one" ? "\"absent,drop\"" : "\"all\"", logged, keep_got[0],
                keep_got[1], keep_want[0], keep_want[1], drop_got[0], drop_got[1], drop_want[0], drop_want[1]);
    if (keep_got != keep_want) return fail(keep_runs ? "count_keep runs all its dispatches" : "count_keep is skipped");
    if (drop_got != drop_want) return fail(drop_runs ? "count_drop runs all its dispatches" : "count_drop is skipped");
    if (!log_ok) return fail("the log names exactly the skipped programs, once each");
    if (vk.validation_errors()) {
        std::fprintf(stderr, "validation errors: %u\n", vk.validation_errors());
        return 1;
    }
    return 0;
}
