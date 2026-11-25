// The pipeline compiler's workers (shims/d3d12_pipeline_compile.h) under LEONIDA_ASYNC_PIPELINES=1, through the D3D12
// layer on a GPU: one compute module of more than 128 KiB, which is a large build, and many small ones, each asked for by
// a dispatch (big_compute_module.h makes them; every dispatch stores a value the CPU computes too).
//   hold (the default)   the large module's build is parked on its worker by the compiler's job probe, after it counts as
//                        running and before it reaches the driver, so "a long compile in flight" is a state the test
//                        holds. With more than one worker every small pipeline is then built by the driver and its
//                        dispatch stores the right value, while the large one is still parked. With one worker
//                        (LEONIDA_COMPILE_THREADS=1) none is: they wait behind it, as before the pool. Let go, the
//                        large one builds, and every dispatch stores the right value.
//   measure <bytes> <n> [seed]
//                        nothing parked: a large module of <bytes> of SPIR-V and <n> small ones, with the time after
//                        which the small ones were all ready, the time for the large one, and the process's peak commit.
//                        For a run by hand (4194304 200: the driver takes tens of seconds over the large one), with
//                        a seed it has not seen, since a driver keeps what it compiled across runs.
// The warm-up is off, so every build is one a dispatch asked for. Self-contained: the SPIR-V is made in memory. No game
// data. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "big_compute_module.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "test_temp_dir.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_pipeline_compile.h"
#include "../src/shader_library.h"

#include <psapi.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using d3d12test::Ref;
using rt::d3d12::PipelineCompiler;
using Clock = std::chrono::steady_clock;

constexpr unsigned kWords = bigmodule::kLoadBase + bigmodule::kLoadWords;
constexpr unsigned kMaxSmall = bigmodule::kLoadBase - 1;  // each program stores the word of its number
constexpr std::size_t kSmallBytes = 12 * 1024;

#define CHECK(expr)                                                                  \
    do {                                                                             \
        if (!(expr)) {                                                               \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);     \
            return false;                                                            \
        }                                                                            \
    } while (0)

long long ms_since(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
}

// Program 0 is the large one; 1 to `small_count` are the small ones.
unsigned small_count = 64;
std::size_t big_bytes = 132 * 1024;
std::uint32_t seed = 20261125;  // other modules for another seed: ones the driver has not kept from a run before
std::vector<bigmodule::Program> programs;

std::vector<std::uint8_t> program(unsigned n) { return d3d12test::fssm_program({0x43504f00u, n, 0xbf810000u}); }

bool write_library(const std::filesystem::path& dir) {
    programs.resize(small_count + 1);
    std::ofstream index(dir / "index.json", std::ios::binary | std::ios::trunc);
    index << "{";
    for (unsigned n = 0; n <= small_count; ++n) {
        const std::string name = "cs" + std::to_string(n);
        const bigmodule::Words words = bigmodule::make(n ? kSmallBytes : big_bytes, seed + n, &programs[n]);
        CHECK((words.size() * 4 > 128 * 1024) == (n == 0));
        std::ofstream f(dir / (name + ".spv"), std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
        CHECK(bool(f));
        index << (n ? ",\n" : "") << "\"" << name << "\": {\"stage\": \"cs\", \"fssm_sha256\": \""
              << leonida::sha256_hex(program(n))
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"local_size\": [1, 1, 1], \"wave64\": false}";
    }
    index << "}";
    return bool(index);
}

struct Pipelines {
    Ref<ID3D12RootSignature> root;
    std::vector<ID3D12PipelineState*> states;
    ~Pipelines() {
        for (auto* s : states) s->Release();
    }
    bool create(ID3D12Device* dev) {
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants.Num32BitValues = 1;
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
        desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto blob = d3d12test::serialize_root_signature(desc);
        CHECK(SUCCEEDED(dev->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
        for (unsigned n = 0; n <= small_count; ++n) {
            const auto cs = d3d12test::shader_container(program(n));
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
            pd.pRootSignature = root.p;
            pd.CS = {cs.data(), cs.size()};
            ID3D12PipelineState* state = nullptr;
            CHECK(SUCCEEDED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&state))));
            states.push_back(state);
        }
        return true;
    }
};

// A queue and a list that dispatches the programs `first` to `last` and copies what they stored back.
struct Frame {
    unsigned first = 0, last = 0;
    Ref<ID3D12Resource> buffer, readback;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    UINT64 fence_value = 0;

    bool create(ID3D12Device* dev, const Pipelines& p, unsigned from, unsigned to) {
        first = from;
        last = to;
        const D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, back{D3D12_HEAP_TYPE_READBACK};
        const auto bd = d3d12test::buffer_desc(4 * kWords, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        CHECK(SUCCEEDED(dev->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &bd,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&buffer.p))));
        const auto rd = d3d12test::buffer_desc(4 * kWords);
        CHECK(SUCCEEDED(dev->CreateCommittedResource(&back, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nullptr, IID_PPV_ARGS(&readback.p))));
        const D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        CHECK(SUCCEEDED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        CHECK(SUCCEEDED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
        list->SetComputeRootSignature(p.root.p);
        list->SetComputeRootUnorderedAccessView(0, buffer->GetGPUVirtualAddress());
        for (unsigned n = first; n <= last; ++n) {
            list->SetPipelineState(p.states[n]);
            list->SetComputeRoot32BitConstant(1, n, 0);
            list->Dispatch(1, 1, 1);
        }
        const auto to_source =
            d3d12test::transition(buffer.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_source);
        list->CopyBufferRegion(readback.p, 0, buffer.p, 0, 4 * kWords);
        const auto back_again =
            d3d12test::transition(buffer.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1, &back_again);
        CHECK(SUCCEEDED(list->Close()));
        return true;
    }

    // Runs the list and returns how many of its programs stored what the CPU computes: a dispatch whose pipeline is not
    // ready is left out (LEONIDA_ASYNC_PIPELINES), and its word stays what it was.
    int stored() {
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.p, ++fence_value);
        fence->SetEventOnCompletion(fence_value, nullptr);
        void* mapped = nullptr;
        if (FAILED(readback->Map(0, nullptr, &mapped)) || !mapped) return -1;
        std::uint32_t words[kWords];
        std::memcpy(words, mapped, sizeof words);
        std::memset(mapped, 0xee, sizeof words);  // stale bytes cannot pass the next time
        readback->Unmap(0, nullptr);
        int right = 0;
        for (unsigned n = first; n <= last; ++n) right += words[n] == programs[n].run(n);
        return right;
    }
    int count() const { return int(last - first + 1); }

    // Runs the list again and again until every program stored its value.
    bool all_stored(unsigned ms) {
        const auto until = Clock::now() + std::chrono::milliseconds(ms);
        for (;;) {
            if (stored() == count()) return true;
            if (Clock::now() > until) return false;
            Sleep(10);
        }
    }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    ~Event() { CloseHandle(handle); }
    void set() const { SetEvent(handle); }
    bool wait(DWORD ms) const { return WaitForSingleObject(handle, ms) == WAIT_OBJECT_0; }
};

bool make_device(vktest::Device& vk, const std::string& library, Ref<ID3D12Device>& out) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.buffer_device_address = true;
    info.subgroup = vk.subgroup;
    info.features = vk.enabled;
    info.shader_dir = library;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&out.p))));
    return true;
}

bool hold(vktest::Device& vk, const std::string& library) {
    auto& compiler = PipelineCompiler::instance();
    const unsigned workers = compiler.threads();
    Event parked, proceed;
    std::atomic<const void*> target{nullptr};
    std::atomic<bool> let_go{false};
    {
        Ref<ID3D12Device> dev;
        CHECK(make_device(vk, library, dev));
        Pipelines pipelines;
        Frame large, smalls;
        CHECK(pipelines.create(dev.p));
        CHECK(large.create(dev.p, pipelines, 0, 0) && smalls.create(dev.p, pipelines, 1, small_count));
        const auto before = compiler.stats();
        const std::uint64_t before_completed = before.completed;
        target = pipelines.states[0];
        compiler.set_job_probe([&](const void* owner) {
            if (owner != target.load()) return;
            parked.set();
            proceed.wait(INFINITE);
        });
        // A failed check leaves through here too: the device's teardown waits for the build, which must not stay parked.
        struct Unpark {
            PipelineCompiler& compiler;
            Event& proceed;
            ~Unpark() {
                proceed.set();
                compiler.set_job_probe(nullptr);
            }
        } unpark{compiler, proceed};
        // A job counts as completed a moment after the pipeline it built can be used.
        const auto completed = [&](std::uint64_t jobs) {
            for (int i = 0; i < 1000 && compiler.stats().completed != before_completed + jobs; ++i) Sleep(5);
            return compiler.stats().completed == before_completed + jobs;
        };
        const auto start = Clock::now();
        CHECK(large.stored() == 0);  // not built: asked for, and its dispatch left out
        CHECK(parked.wait(30000));
        CHECK(smalls.stored() >= 0);  // asks for every small one
        CHECK(compiler.stats().submitted == before.submitted + small_count + 1);
        if (workers > 1) {
            // The other workers build them, in the driver, while the large one is in flight.
            CHECK(smalls.all_stored(120000));
            const auto took = ms_since(start);
            CHECK(!let_go && large.stored() == 0);
            CHECK(completed(small_count));
            std::printf("hold (%u workers): %u small pipelines built and dispatched in %lld ms with the large build in "
                        "flight\n", workers, small_count, took);
        } else {
            Sleep(500);
            CHECK(smalls.stored() == 0 && large.stored() == 0);
            CHECK(completed(0));
            std::printf("hold (1 worker): no small pipeline built with the large build in flight, as before the pool\n");
        }
        let_go = true;
        proceed.set();
        CHECK(large.all_stored(300000) && smalls.all_stored(300000));
        compiler.wait_idle();
        CHECK(completed(small_count + 1));
    }
    CHECK(!compiler.worker_running());  // the device's last release stopped and joined every worker
    return true;
}

bool measure(vktest::Device& vk, const std::string& library) {
    auto& compiler = PipelineCompiler::instance();
    {
        Ref<ID3D12Device> dev;
        CHECK(make_device(vk, library, dev));
        Pipelines pipelines;
        Frame large, smalls;
        CHECK(pipelines.create(dev.p));
        CHECK(large.create(dev.p, pipelines, 0, 0) && smalls.create(dev.p, pipelines, 1, small_count));
        const auto start = Clock::now();
        CHECK(large.stored() == 0);
        Sleep(50);  // the large build is the first a worker takes
        CHECK(smalls.stored() >= 0);
        CHECK(smalls.all_stored(1800000));
        const auto small_ms = ms_since(start);
        const bool large_ready_first = large.stored() == 1;
        CHECK(large.all_stored(1800000));
        const auto large_ms = ms_since(start);
        compiler.wait_idle();
        PROCESS_MEMORY_COUNTERS memory{sizeof memory};
        GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof memory);
        std::printf("measure: workers=%u large_bytes=%zu small=%u small_ready_ms=%lld large_ready_ms=%lld "
                    "large_ready_before_small=%d peak_commit_mib=%zu\n", compiler.threads(), big_bytes, small_count,
                    small_ms, large_ms, int(large_ready_first), std::size_t(memory.PeakPagefileUsage >> 20));
    }
    CHECK(!compiler.worker_running());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "hold";
    if (scenario == "measure") {
        if (argc > 2) big_bytes = std::strtoull(argv[2], nullptr, 10);
        if (argc > 3) small_count = unsigned(std::strtoul(argv[3], nullptr, 10));
        if (argc > 4) seed = std::uint32_t(std::strtoul(argv[4], nullptr, 10));
    } else if (scenario != "hold") {
        std::fprintf(stderr, "usage: d3d12_compile_pool_test [hold | measure <large SPIR-V bytes> <small modules> [seed]]\n");
        return 2;
    }
    if (small_count < 1 || small_count > kMaxSmall || big_bytes <= 128 * 1024) {
        std::fprintf(stderr, "1 to %u small modules, and more than 131072 bytes for the large one\n", kMaxSmall);
        return 2;
    }
    vktest::Device vk;
    vk.request_buffer_device_address = true;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    VkPhysicalDeviceProperties properties{};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(vk.physical, &properties);
    if (!vk.buffer_device_address || !vk.enabled.shaderInt64) {
        std::printf("SKIP: the GPU lacks bufferDeviceAddress or shaderInt64\n");
        return 77;
    }
    if (properties.limits.maxPushConstantsSize < 144) {
        std::printf("SKIP: maxPushConstantsSize %u, below the 144-byte block\n", properties.limits.maxPushConstantsSize);
        return 77;
    }
    _putenv_s("LEONIDA_ASYNC_PIPELINES", "1");
    _putenv_s("LEONIDA_STRICT_REPLAY", "0");
    _putenv_s("LEONIDA_LAZY_COMPUTE", "1");
    _putenv_s("LEONIDA_COMPUTE_WARMUP", "0");        // every build is one a dispatch asked for
    _putenv_s("LEONIDA_PIPELINE_CACHE_BYTES", "0");  // no session file in or out
    const testfiles::ScratchDir library("leonida_compile_pool_test");
    if (!write_library(library.path())) return 1;
    if (!(scenario == "measure" ? measure(vk, library.path().string()) : hold(vk, library.path().string()))) return 1;
    if (vk.validation_errors()) {
        std::fprintf(stderr, "FAIL: %u validation errors\n", vk.validation_errors());
        return 1;
    }
    std::printf("d3d12_compile_pool_test: passed on %s\n", properties.deviceName);
    return 0;
}
