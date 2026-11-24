// Stack walk shared by the diagnostics (startup_trace.h and crash_log.h).
//
// The frames come from a CONTEXT, not from this thread's live stack, so a crash handler can walk the faulting
// thread's stack even when its own frame sits above a corrupt one. The image's .pdata must be registered
// (RtlAddFunctionTable, which the loader does for mapped images and Windows does for host images); a frame without
// an entry is treated as a leaf and its return address is read from [rsp].
//
// A corrupt context or unwind table can itself fault, so the walk runs under SEH and returns the frames it managed
// before the fault. It allocates nothing: the caller owns the output array.
#pragma once

#include <windows.h>

#include <cstdint>

namespace rt::stack_walk {

// Writes at most `max_frames` addresses, starting with the context's Rip, to `out`; returns how many.
inline void raw_frames(const CONTEXT& start_context, uint64_t* out, int max_frames, volatile int* written) {
    int n = 0;
    CONTEXT c = start_context;
    while (n < max_frames && c.Rip) {
        out[n++] = c.Rip;
        *written = n;
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(c.Rip, &image_base, nullptr);
        if (!fn) {  // a leaf function: the return address is at [rsp]
            if (!c.Rsp || IsBadReadPtr(reinterpret_cast<void*>(c.Rsp), sizeof(DWORD64))) break;
            c.Rip = *reinterpret_cast<const DWORD64*>(c.Rsp);
            c.Rsp += sizeof(DWORD64);
            continue;
        }
        void* handler_data = nullptr;
        DWORD64 frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, c.Rip, fn, &c, &handler_data, &frame, nullptr);
    }
}

inline int frames(const CONTEXT& start_context, uint64_t* out, int max_frames) {
    volatile int written = 0;
#ifdef _MSC_VER
    // Keep this frame free of C++ objects with destructors: /EHsc does not let __except catch an SEH fault there.
    __try {
        raw_frames(start_context, out, max_frames, &written);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    raw_frames(start_context, out, max_frames, &written);
#endif
    return written;
}

}  // namespace rt::stack_walk
