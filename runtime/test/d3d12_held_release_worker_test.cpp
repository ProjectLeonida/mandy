// Held work released by another thread's Signal runs on the released queue's worker, not on the signalling thread
// (RT-RELEASER-HELD-REPLAY, src/vulkan_backend.h release_ready). The list values' releaser signals the gates of every
// queue from one thread; when that thread ran each queue's held replay itself, one queue's slow first-use pipeline
// builds held up every other queue's gates.
//
// LEONIDA_HELD_RUN_DELAY_MS stands in for a slow replay: each held operation waits that long before it runs. Two
// queues each hold a Signal behind a Wait; one thread signals both gates, one after the other, as the releaser does.
// With LEONIDA_RELEASE_WORKER=1 (the first ctest entry) both Signals return at once and the two queues' held work runs
// side by side; without it (the default, the second entry) the signalling thread runs each in turn, so the second
// queue waits for the first queue's replay. Order on one queue is kept either way. No shaders, no game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr unsigned kDelayMs = 1500;

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() {
        if (p) p->Release();
    }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { CloseHandle(h); }
    bool set_within(DWORD ms) const { return WaitForSingleObject(h, ms) == WAIT_OBJECT_0; }
};

double ms_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int check(ID3D12Device* device, bool worker) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    Ref<ID3D12CommandQueue> a, b;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&a.p))));
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&b.p))));
    Ref<ID3D12Fence> gate_a, gate_b, fa, fb;
    for (ID3D12Fence** f : {&gate_a.p, &gate_b.p, &fa.p, &fb.p})
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(f))));

    // Each queue holds two Signals behind its gate; the second must still land after the first (fence ends on 2).
    EXPECT(SUCCEEDED(a->Wait(gate_a.p, 1)));
    EXPECT(SUCCEEDED(a->Signal(fa.p, 1)));
    EXPECT(SUCCEEDED(a->Signal(fa.p, 2)));
    EXPECT(SUCCEEDED(b->Wait(gate_b.p, 1)));
    EXPECT(SUCCEEDED(b->Signal(fb.p, 1)));
    EXPECT(SUCCEEDED(b->Signal(fb.p, 2)));
    EXPECT(fa->GetCompletedValue() == 0 && fb->GetCompletedValue() == 0);

    Event a_done, b_done;
    EXPECT(SUCCEEDED(fa->SetEventOnCompletion(2, a_done.h)));
    EXPECT(SUCCEEDED(fb->SetEventOnCompletion(2, b_done.h)));

    // One thread opens both gates, as the releaser does.
    const auto start = std::chrono::steady_clock::now();
    EXPECT(SUCCEEDED(gate_a->Signal(1)));
    const double first_signal_ms = ms_since(start);
    EXPECT(SUCCEEDED(gate_b->Signal(1)));
    const double both_signals_ms = ms_since(start);
    EXPECT(b_done.set_within(20000));
    const double b_ms = ms_since(start);
    EXPECT(a_done.set_within(20000));
    const double a_ms = ms_since(start);
    std::printf("release %s: gate Signals returned after %.0f and %.0f ms; queue B's held work done at %.0f ms, "
                "queue A's at %.0f ms (each held operation waits %u ms)\n",
                worker ? "worker" : "on the signalling thread", first_signal_ms, both_signals_ms, b_ms, a_ms, kDelayMs);
    EXPECT(fa->GetCompletedValue() == 2 && fb->GetCompletedValue() == 2);  // in order on each queue

    if (worker) {
        // The Signals only ask the workers; the two queues' held Signals (2 x delay each) run side by side, so
        // neither waits for the other's (each would end near 4 x delay if they ran in turn).
        EXPECT(both_signals_ms < kDelayMs / 2.0);
        // The stand-in replay did take its time (two held operations per queue), so the Signals returned early
        // because they handed the work off, not because there was nothing to wait for.
        EXPECT(b_ms >= 2.0 * kDelayMs - 50 && a_ms >= 2.0 * kDelayMs - 50);
        EXPECT(b_ms < 3.0 * kDelayMs && a_ms < 3.0 * kDelayMs);
    } else {
        // The signalling thread runs A's held work (2 x delay) before it gets to B's gate.
        EXPECT(first_signal_ms >= 2.0 * kDelayMs - 50);
        EXPECT(b_ms >= 4.0 * kDelayMs - 100);
    }
    return 0;
}

// This process's CPU time (all threads), in ms.
double process_cpu_ms() {
    FILETIME created, exited, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0.0;
    const auto ticks = [](const FILETIME& t) { return (std::uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    return double(ticks(kernel) + ticks(user)) / 10000.0;
}

// A worker whose queue's front still waits goes idle. It used to re-arm its own release request from its pump on
// every pass, so a front waiting on an unsignalled fence kept the worker busy (a core's worth of CPU) for as long as
// the fence stayed unsignalled. With the release worker only.
int check_blocked_front_idles(ID3D12Device* device) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    Ref<ID3D12CommandQueue> q;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&q.p))));
    Ref<ID3D12Fence> gate1, gate2, other, f;
    for (ID3D12Fence** fence : {&gate1.p, &gate2.p, &other.p, &f.p})
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence))));
    EXPECT(SUCCEEDED(q->Wait(gate1.p, 1)));
    EXPECT(SUCCEEDED(q->Signal(f.p, 1)));
    EXPECT(SUCCEEDED(q->Wait(gate2.p, 1)));
    EXPECT(SUCCEEDED(q->Signal(f.p, 2)));

    // The first part runs on the worker (which starts for it); the front then waits on gate2, never signalled yet.
    Event first, second;
    EXPECT(SUCCEEDED(f->SetEventOnCompletion(1, first.h)));
    EXPECT(SUCCEEDED(gate1->Signal(1)));
    EXPECT(first.set_within(20000));
    EXPECT(SUCCEEDED(other->Signal(1)));  // a Signal nothing waits on: a pass over the holds, as any would make
    Sleep(200);                           // let the worker finish what that pass asked of it

    const double cpu_before = process_cpu_ms();
    const auto start = std::chrono::steady_clock::now();
    Sleep(1000);
    const double cpu_ms = process_cpu_ms() - cpu_before, wall_ms = ms_since(start);
    std::printf("blocked front: %.0f ms of CPU in %.0f ms while the front waits\n", cpu_ms, wall_ms);
    EXPECT(cpu_ms < 0.3 * wall_ms);
    EXPECT(f->GetCompletedValue() == 1);

    EXPECT(SUCCEEDED(f->SetEventOnCompletion(2, second.h)));
    EXPECT(SUCCEEDED(gate2->Signal(1)));
    EXPECT(second.set_within(20000));
    EXPECT(f->GetCompletedValue() == 2);
    return 0;
}

}  // namespace

int main() {
    // Read once, at the first release: set before any queue holds work.
    char delay[16];
    std::snprintf(delay, sizeof(delay), "%u", kDelayMs);
    _putenv_s("LEONIDA_HELD_RUN_DELAY_MS", delay);
    const char* on = std::getenv("LEONIDA_RELEASE_WORKER");
    const bool worker = on && std::strcmp(on, "1") == 0;

    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    ID3D12Device* raw = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&raw))));
    Ref<ID3D12Device> device;
    device.p = raw;
    EXPECT(check(device.p, worker) == 0);
    if (worker) EXPECT(check_blocked_front_idles(device.p) == 0);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("held release: %s\n", worker ? "two queues' held work released by one thread ran side by side, in "
                                              "order on each queue"
                                            : "the signalling thread ran each queue's held work in turn (as before)");
    return 0;
}
