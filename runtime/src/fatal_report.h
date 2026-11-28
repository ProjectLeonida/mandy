#pragma once
// Where a fatal exception happened, in the title's own terms, for the line gta6rt prints before the process ends:
// the faulting address as an RVA of the title's image, the values near the stack pointer that point into the image
// (likely return addresses, innermost first), the general registers, and when rbx holds an object whose first
// qword points into the image, that vtable's RVA. A title's soft assert (`int3` after a report call, e.g. its
// out-of-memory report) is then traceable from the log alone. Reads go through ReadProcessMemory, so a bad stack
// pointer ends the walk instead of faulting again inside the handler.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <string>

namespace rt::fatal_report {

inline bool read_qword(std::uintptr_t at, std::uintptr_t& out) {
    SIZE_T got = 0;
    return at && ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at), &out, sizeof(out), &got) &&
           got == sizeof(out);
}

// `base`/`size`: the title's image (0 when not known yet). `max_slots` stack qwords are looked at, at most
// `max_found` image addresses listed.
inline std::string describe(const EXCEPTION_POINTERS* info, std::uintptr_t base, std::size_t size,
                            int max_slots = 512, int max_found = 12) {
    std::string out;
    char buf[512];
    const bool known = base && size;
    auto inside = [&](std::uintptr_t v) { return known && v >= base && v < base + size; };
    const std::uintptr_t at =
        info && info->ExceptionRecord ? reinterpret_cast<std::uintptr_t>(info->ExceptionRecord->ExceptionAddress) : 0;
    if (inside(at))
        std::snprintf(buf, sizeof(buf), "[rt] fault at title +0x%llx\n", static_cast<unsigned long long>(at - base));
    else
        std::snprintf(buf, sizeof(buf), "[rt] fault at 0x%llx (outside the title image)\n",
                      static_cast<unsigned long long>(at));
    out += buf;
    if (!info || !info->ContextRecord) return out;
    const CONTEXT& c = *info->ContextRecord;
    if (known) {
        std::string calls;
        int found = 0;
        for (int i = 0; i < max_slots && found < max_found; ++i) {
            std::uintptr_t v = 0;
            if (!read_qword(static_cast<std::uintptr_t>(c.Rsp) + std::uintptr_t(i) * 8, v)) break;
            if (inside(v) && v != base) {
                std::snprintf(buf, sizeof(buf), " +0x%llx", static_cast<unsigned long long>(v - base));
                calls += buf;
                ++found;
            }
        }
        out += "[rt] title addresses on the stack (innermost first):" + (calls.empty() ? std::string(" none") : calls) +
               "\n";
    }
    std::snprintf(buf, sizeof(buf),
                  "[rt] registers: rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx rbp=%llx rsp=%llx r8=%llx "
                  "r9=%llx r12=%llx r13=%llx r14=%llx r15=%llx\n",
                  static_cast<unsigned long long>(c.Rax), static_cast<unsigned long long>(c.Rbx),
                  static_cast<unsigned long long>(c.Rcx), static_cast<unsigned long long>(c.Rdx),
                  static_cast<unsigned long long>(c.Rsi), static_cast<unsigned long long>(c.Rdi),
                  static_cast<unsigned long long>(c.Rbp), static_cast<unsigned long long>(c.Rsp),
                  static_cast<unsigned long long>(c.R8), static_cast<unsigned long long>(c.R9),
                  static_cast<unsigned long long>(c.R12), static_cast<unsigned long long>(c.R13),
                  static_cast<unsigned long long>(c.R14), static_cast<unsigned long long>(c.R15));
    out += buf;
    std::uintptr_t vtable = 0;
    if (known && read_qword(static_cast<std::uintptr_t>(c.Rbx), vtable) && inside(vtable)) {
        std::snprintf(buf, sizeof(buf), "[rt] rbx points at an object with vtable title +0x%llx\n",
                      static_cast<unsigned long long>(vtable - base));
        out += buf;
    }
    return out;
}

}  // namespace rt::fatal_report
