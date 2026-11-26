// chad.dll: the console's audio DSP command queue. No PC has one, so this gives the title connections and streams
// whose commands complete at once with silent output: nothing the title submits is processed. xapu.dll, the APU
// command queue, is in xapu_queue.cpp.
//
// Read from the executable's call sites (counts in docs/runtime_imports.md), statically, never run:
//   XDspConnectWithMaximumStreamLimit(buffer, u32, u32, 0x80, 1, handle*)   1 site; success is a return of 0. The
//                                             buffer comes from the title's XMemAlloc and stays the title's.
//   XDspActivate(handle, desc*, out*, out*)   1 site; success is a return of 0, else the caller returns -1. The two
//                                             outputs are fields of the caller's object. The title later reads
//                                             dwords at +4 and +0xc through one of them with no null check, and
//                                             passes the other to XDspSubmitCommand.
//   XDspSubmitCommand(stream, cmd*, id*)      1 site; success is a return of 0. The title zeroes *id before the call
//                                             and afterwards waits until the status dword at +0xc is 0 and the one
//                                             at +4 equals *id.
//   XDspDeactivate, XDspDisconnect            1 site each, tail calls: the caller returns what they return.
//
// chad.dll's signatures, layouts, limits and error codes are facts recorded by Marvisak/VIctory (GPL-3.0-or-later,
// crates/victory-hle/src/modules/dsp.rs at commit ffe310e), which runs the retail executable through these calls; see
// docs/related_projects.md. Nothing of its code is copied. They fit every chad shape above:
//   HRESULT XDspConnectWithMaximumStreamLimit(void* base, uint32_t base_bytes, uint32_t ir_seconds,
//           uint32_t max_streams, uint32_t max_channels, DspConnection** out)
//                       E_INVALIDARG for a null base, base_bytes outside 0x10000..0x1F400000, ir_seconds above 0x80,
//                       max_streams above 0x100 or max_channels above 2; then E_POINTER for a null out.
//   HRESULT XDspActivate(DspConnection*, const DspStreamDesc*, const DspStatus** status, DspStream** stream)
//                       E_POINTER for a null status or stream out; E_INVALIDARG for a null connection or description
//                       or a description that breaks the rules below; E_OUTOFMEMORY once max_streams are active.
//                       The status block starts zeroed with result S_OK. A description is valid when:
//                       kind 1 (convolution): 1 or 2 channels, 0x200 or 0x400 frames, option 0x4 clear;
//                       any other kind: 1 channel, 0x200, 0x400 or 0x800 frames;
//                       and options 0x2 and 0x8 need 2 channels.
//   HRESULT XDspSubmitCommand(DspStream*, const DspCommand*, uint32_t* ticket)
//                       E_INVALIDARG for a null stream or command. The command completes at once: frames * channels
//                       floats of output (when not null) are written, the status block's submitted and completed
//                       words advance to the next ticket and its result is S_OK, and *ticket (when not null) gets
//                       that ticket. This is the wait above: completed (+4) equals *id and result (+0xc) is 0.
//   HRESULT XDspDeactivate(DspStream*), HRESULT XDspDisconnect(DspConnection*)   E_INVALIDARG for null.
// No convolution or other block transform is computed: the hardware's impulse response layout is not known, so the
// output of every command is zeros and reverb returns are dry.
//
// HYPOTHESIS: a connection or stream handle that is unknown or already released is refused with E_INVALIDARG, and
// XDspDisconnect with streams still active releases them, so their handles are then refused too. The title releases
// its streams before it disconnects; what the console does otherwise is not recorded.
#include <windows.h>
#include <intrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <set>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

#pragma pack(push, 1)
struct DspStreamDesc {
    uint32_t kind;  // 1: convolution
    uint32_t options;
    uint32_t frames;  // per block
    uint32_t channels;
    uint32_t ir_floats;
    const float* ir;  // impulse response, in the hardware's layout
};
#pragma pack(pop)
struct DspStatus {  // the block the title polls
    uint32_t submitted;
    uint32_t completed;  // ticket of the last completed command
    uint32_t reserved;
    HRESULT result;  // of that command
};
struct DspCommand {  // one block
    float gains[2];
    const float* input;
    float* output;
    uint64_t reserved;
};
static_assert(sizeof(DspStreamDesc) == 0x1C && offsetof(DspStreamDesc, ir) == 0x14, "packed description");
static_assert(sizeof(DspStatus) == 16 && offsetof(DspStatus, completed) == 4 && offsetof(DspStatus, result) == 0xC,
              "the dwords the title's wait reads");
static_assert(sizeof(DspCommand) == 32 && offsetof(DspCommand, output) == 0x10, "gains, input, output, reserved");

constexpr uint32_t kDspConvolution = 1;
constexpr uint32_t kDspStereoOptions = 0x2 | 0x8;  // need two channels
constexpr uint32_t kDspNotForConvolution = 0x4;
constexpr uint32_t kDspMinBaseBytes = 0x10000;
constexpr uint32_t kDspMaxBaseBytes = 0x1F400000;
constexpr uint32_t kDspMaxIrSeconds = 0x80;
constexpr uint32_t kDspMaxStreams = 0x100;
constexpr uint32_t kDspMaxChannels = 2;

struct DspStream;
struct DspConnection {
    uint32_t max_streams;
    std::set<DspStream*> streams;
};
struct DspStream {
    DspConnection* owner;
    uint32_t frames;
    uint32_t channels;
    DspStatus status;  // handed to the title by XDspActivate
};

// Every live handle, so a stale or doubled release is refused instead of freeing twice. One lock for all: the title
// submits one command per block, so there is nothing to contend.
std::mutex g_dsp_mutex;
std::set<DspConnection*> g_dsp_connections;
std::set<DspStream*> g_dsp_streams;

bool dsp_description_valid(const DspStreamDesc& d) {
    if ((d.options & kDspStereoOptions) && d.channels != 2) return false;
    if (d.kind == kDspConvolution) {
        if (d.options & kDspNotForConvolution) return false;
        return (d.channels == 1 || d.channels == 2) && (d.frames == 0x200 || d.frames == 0x400);
    }
    return d.channels == 1 && (d.frames == 0x200 || d.frames == 0x400 || d.frames == 0x800);
}

// The title polls the status block from other threads: each word is one interlocked store, result before completed.
void dsp_store(uint32_t& word, uint32_t value) {
    InterlockedExchange(reinterpret_cast<volatile LONG*>(&word), static_cast<LONG>(value));
}

void dsp_release_stream(DspStream* s) {  // under g_dsp_mutex
    s->owner->streams.erase(s);
    g_dsp_streams.erase(s);
    delete s;
}

// gta6rt --trace FILE: the first call of each function, with the caller, so the first boot shows what the title sends
// without a line per call.
void note(const char* api, uint64_t value, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::pair<std::string, uint64_t>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({api, value}).second) return;
    }
    rt::startup_trace::event(
        "audio_queue", std::string("\"api\":") + rt::startup_trace::quote(api) +
                           ",\"value\":" + rt::startup_trace::quote(rt::startup_trace::hex(value)) + ",\"caller\":" +
                           rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
}

HRESULT __stdcall XDspConnectWithMaximumStreamLimit(void* base, uint32_t base_bytes, uint32_t ir_seconds,
                                                    uint32_t max_streams, uint32_t max_channels,
                                                    DspConnection** handle) {
    note("XDspConnectWithMaximumStreamLimit", max_streams, _ReturnAddress());
    if (!base || base_bytes < kDspMinBaseBytes || base_bytes > kDspMaxBaseBytes || ir_seconds > kDspMaxIrSeconds ||
        max_streams > kDspMaxStreams || max_channels > kDspMaxChannels)
        return E_INVALIDARG;
    if (!handle) return E_POINTER;
    auto* c = new (std::nothrow) DspConnection{max_streams, {}};  // a throwing new would unwind into the title
    if (!c) return E_OUTOFMEMORY;
    try {
        const std::lock_guard lock(g_dsp_mutex);
        g_dsp_connections.insert(c);
    } catch (...) {
        delete c;
        return E_OUTOFMEMORY;
    }
    *handle = c;
    return S_OK;
}

HRESULT __stdcall XDspDisconnect(DspConnection* c) {
    note("XDspDisconnect", 0, _ReturnAddress());
    const std::lock_guard lock(g_dsp_mutex);
    if (!c || !g_dsp_connections.erase(c)) return E_INVALIDARG;
    while (!c->streams.empty()) dsp_release_stream(*c->streams.begin());
    delete c;
    return S_OK;
}

HRESULT __stdcall XDspActivate(DspConnection* c, const DspStreamDesc* description, const DspStatus** status,
                               DspStream** stream) {
    note("XDspActivate", description ? description->kind : ~0ull, _ReturnAddress());
    if (!status || !stream) return E_POINTER;
    if (!c || !description) return E_INVALIDARG;
    DspStreamDesc d;
    std::memcpy(&d, description, sizeof d);
    if (!dsp_description_valid(d)) return E_INVALIDARG;
    const std::lock_guard lock(g_dsp_mutex);
    if (!g_dsp_connections.count(c)) return E_INVALIDARG;
    if (c->streams.size() >= c->max_streams) return E_OUTOFMEMORY;
    auto* s = new (std::nothrow) DspStream{c, d.frames, d.channels, {0, 0, 0, S_OK}};
    if (!s) return E_OUTOFMEMORY;
    try {
        g_dsp_streams.insert(s);
        c->streams.insert(s);
    } catch (...) {
        g_dsp_streams.erase(s);
        delete s;
        return E_OUTOFMEMORY;
    }
    *status = &s->status;
    *stream = s;
    return S_OK;
}

HRESULT __stdcall XDspDeactivate(DspStream* s) {
    note("XDspDeactivate", 0, _ReturnAddress());
    const std::lock_guard lock(g_dsp_mutex);
    if (!s || !g_dsp_streams.count(s)) return E_INVALIDARG;
    dsp_release_stream(s);
    return S_OK;
}

// Completes at once with a block of silence: no convolution is computed.
HRESULT __stdcall XDspSubmitCommand(DspStream* s, const DspCommand* command, uint32_t* ticket) {
    note("XDspSubmitCommand", 0, _ReturnAddress());
    if (!s || !command) return E_INVALIDARG;
    DspCommand cmd;
    std::memcpy(&cmd, command, sizeof cmd);
    const std::lock_guard lock(g_dsp_mutex);
    if (!g_dsp_streams.count(s)) return E_INVALIDARG;
    if (cmd.output) std::memset(cmd.output, 0, size_t{s->frames} * s->channels * sizeof(float));
    const uint32_t next = s->status.submitted + 1;  // wraps like the 32-bit word it is
    dsp_store(s->status.submitted, next);
    dsp_store(reinterpret_cast<uint32_t&>(s->status.result), static_cast<uint32_t>(S_OK));
    dsp_store(s->status.completed, next);
    if (ticket) *ticket = next;
    return S_OK;
}

}  // namespace

SHIM_TABLE(chad,
    {"chad.dll", "XDspActivate", reinterpret_cast<void*>(&XDspActivate)},
    {"chad.dll", "XDspConnectWithMaximumStreamLimit", reinterpret_cast<void*>(&XDspConnectWithMaximumStreamLimit)},
    {"chad.dll", "XDspDeactivate", reinterpret_cast<void*>(&XDspDeactivate)},
    {"chad.dll", "XDspDisconnect", reinterpret_cast<void*>(&XDspDisconnect)},
    {"chad.dll", "XDspSubmitCommand", reinterpret_cast<void*>(&XDspSubmitCommand)},
);
