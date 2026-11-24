// Whether the host has a default audio render endpoint (docs/boot.md, "No audio output device").
//
// The retail title asks for the default render endpoint (IMMDeviceEnumerator::GetDefaultAudioEndpoint(eRender,
// eConsole)) once while it starts its audio. Without one it skips its audio device setup, runs on, and stops after the
// first-run screens: its per-frame audio update returns early while it still holds a lock, the frame thread waits on
// that lock for good, and the frame counter the loading code waits on never moves again. Nothing crashes and the
// loading screen keeps being drawn, so the cause is easy to miss. A console always has an output, so the runtime says
// so at start when the host has none; a Wine prefix with its audio drivers off, or no sound device, is the usual case.
//
// The probe runs on its own short-lived thread, in its own COM apartment, so the thread that later runs the title
// starts with no COM state of ours.
#pragma once

#include <windows.h>

#include <mmdeviceapi.h>

namespace rt::audio_probe {

enum class Endpoint {
    present,  // a default render endpoint exists
    missing,  // the enumerator answered, and there is no default render endpoint
    unknown,  // COM or the enumerator itself failed; nothing is reported
};

namespace detail {
struct ProbeResult {
    Endpoint endpoint = Endpoint::unknown;
    HRESULT hr = S_OK;  // GetDefaultAudioEndpoint's result when the enumerator was created
};

inline DWORD WINAPI probe_thread(void* param) {
    auto* out = static_cast<ProbeResult*>(param);
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(init)) return 0;
    IMMDeviceEnumerator* enumerator = nullptr;
    if (SUCCEEDED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                   reinterpret_cast<void**>(&enumerator))) &&
        enumerator) {
        IMMDevice* device = nullptr;
        out->hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
        out->endpoint = SUCCEEDED(out->hr) && device ? Endpoint::present : Endpoint::missing;
        if (device) device->Release();
        enumerator->Release();
    }
    CoUninitialize();
    return 0;
}
}  // namespace detail

// Asks the host once; waits at most `timeout_ms` for the answer (an audio service that hangs is reported as unknown).
inline Endpoint default_render_endpoint(HRESULT* hr = nullptr, DWORD timeout_ms = 5000) {
    // Heap-held so a probe thread that outlives the wait still writes into live memory; it is freed only after the
    // thread has finished, and leaked (a few bytes, once) when it has not.
    auto* result = new detail::ProbeResult;
    HANDLE thread = CreateThread(nullptr, 0, &detail::probe_thread, result, 0, nullptr);
    if (!thread) {
        delete result;
        return Endpoint::unknown;
    }
    const bool finished = WaitForSingleObject(thread, timeout_ms) == WAIT_OBJECT_0;
    CloseHandle(thread);
    if (!finished) return Endpoint::unknown;
    const detail::ProbeResult r = *result;
    delete result;
    if (hr) *hr = r.hr;
    return r.endpoint;
}

}  // namespace rt::audio_probe
