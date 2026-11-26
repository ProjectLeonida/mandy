// The parts of a fatal-fault report (rt::print_fault_stacks, RT-CRASH-LOG) that are only text: the faulting thread's
// registers, what an exception's parameters say, the host line and the header of one report in the crash log. Pure
// and portable, so runtime/test/crash_report_test.cpp checks them on every platform; the Windows side
// (thread_stacks.cpp) fills them from the EXCEPTION_RECORD and CONTEXT and writes the result.
//
// The idea of keeping each report in a file that outlives the console, with the host's CPU and Windows build and the
// thread's registers, is from Marvisak/VIctory (crates/victory-loader/src/crash.rs, GPL-3.0); no code of it is used.
#pragma once

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>
#include <string_view>

namespace rt::crash_report {

// The general registers of the faulting context, in CONTEXT's names.
struct Registers {
    std::uint64_t rax = 0, rbx = 0, rcx = 0, rdx = 0, rsi = 0, rdi = 0, rbp = 0, rsp = 0;
    std::uint64_t r8 = 0, r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0;
    std::uint64_t rip = 0;
    std::uint32_t eflags = 0;
};

// Four lines, four registers each and then rip and eflags, every line starting with `prefix`.
inline std::string registers_lines(const Registers& r, std::string_view prefix) {
    const struct {
        const char* name;
        std::uint64_t value;
    } all[] = {{"rax", r.rax}, {"rbx", r.rbx}, {"rcx", r.rcx}, {"rdx", r.rdx}, {"rsi", r.rsi}, {"rdi", r.rdi},
               {"rbp", r.rbp}, {"rsp", r.rsp}, {"r8", r.r8},   {"r9", r.r9},   {"r10", r.r10}, {"r11", r.r11},
               {"r12", r.r12}, {"r13", r.r13}, {"r14", r.r14}, {"r15", r.r15}};
    std::string out;
    char field[32];
    for (std::size_t i = 0; i < std::size(all); ++i) {
        if (i % 4 == 0) out += prefix;
        std::snprintf(field, sizeof field, "%-3s=%016" PRIx64, all[i].name, all[i].value);
        out += field;
        out += i % 4 == 3 ? "\n" : " ";
    }
    std::snprintf(field, sizeof field, "rip=%016" PRIx64, r.rip);
    out += prefix;
    out += field;
    std::snprintf(field, sizeof field, " eflags=%08" PRIx32 "\n", r.eflags);
    out += field;
    return out;
}

// Exception codes a report names (winnt.h / ntstatus.h values, and MSVC's C++ throw).
inline constexpr std::uint32_t kAccessViolation = 0xC0000005u;
inline constexpr std::uint32_t kInPageError = 0xC0000006u;
inline constexpr std::uint32_t kIllegalInstruction = 0xC000001Du;
inline constexpr std::uint32_t kIntDivideByZero = 0xC0000094u;
inline constexpr std::uint32_t kPrivilegedInstruction = 0xC0000096u;
inline constexpr std::uint32_t kStackOverflow = 0xC00000FDu;
inline constexpr std::uint32_t kHeapCorruption = 0xC0000374u;
inline constexpr std::uint32_t kBreakpoint = 0x80000003u;
inline constexpr std::uint32_t kCppException = 0xE06D7363u;  // 'msc' | 0xE0000000: a C++ throw nothing caught

inline const char* exception_name(std::uint32_t code) {
    switch (code) {
    case kAccessViolation: return "access violation";
    case kInPageError: return "in-page error";
    case kIllegalInstruction: return "illegal instruction";
    case kIntDivideByZero: return "integer divide by zero";
    case kPrivilegedInstruction: return "privileged instruction";
    case kStackOverflow: return "stack overflow";
    case kHeapCorruption: return "heap corruption";
    case kBreakpoint: return "breakpoint";
    case kCppException: return "C++ exception";
    default: return nullptr;
    }
}

// What an access to memory was: ExceptionInformation[0] of an access violation or an in-page error.
inline const char* access_kind(std::uint64_t kind) {
    return kind == 0 ? "read" : kind == 1 ? "write" : kind == 8 ? "execute" : "access";
}

// One line (ending in '\n', starting with `prefix`) that says what the exception's parameters mean, or its raw
// parameters for a code it does not know. `type_name` is the mangled name of a C++ exception's thrown type when the
// caller could read it (".?AVruntime_error@std@@"), else empty.
inline std::string exception_detail(std::uint32_t code, std::uint32_t count, const std::uint64_t* params,
                                    std::string_view type_name, std::string_view prefix) {
    char line[256];
    const char* name = exception_name(code);
    std::string out(prefix);
    if ((code == kAccessViolation || code == kInPageError) && count >= 2) {
        std::snprintf(line, sizeof line, "%s: %s of 0x%" PRIx64, name, access_kind(params[0]), params[1]);
        out += line;
        if (code == kInPageError && count >= 3) {
            std::snprintf(line, sizeof line, ", status 0x%08" PRIx32, static_cast<std::uint32_t>(params[2]));
            out += line;
        }
    } else if (code == kCppException && count >= 3) {
        std::snprintf(line, sizeof line, "%s: object 0x%" PRIx64 ", throw info 0x%" PRIx64, name, params[1],
                      params[2]);
        out += line;
        if (!type_name.empty()) {
            out += ", type ";
            out += type_name;
        }
    } else {
        std::snprintf(line, sizeof line, "%s 0x%08" PRIx32 ", %" PRIu32 " parameter%s", name ? name : "exception",
                      code, count, count == 1 ? "" : "s");
        out += line;
        for (std::uint32_t i = 0; i < count && i < 15; ++i) {
            std::snprintf(line, sizeof line, " 0x%" PRIx64, params[i]);
            out += line;
        }
    }
    out += '\n';
    return out;
}

// The host line: the CPU decides which of the console's Zen 2 instructions are emulated (cpu_emulation.h), and the
// Windows build which loader and driver stack the title runs on.
inline std::string host_line(std::string_view cpu, std::uint32_t major, std::uint32_t minor, std::uint32_t build,
                             std::string_view prefix) {
    char line[96];
    std::snprintf(line, sizeof line, "host: Windows %" PRIu32 ".%" PRIu32 ".%" PRIu32 ", CPU ", major, minor, build);
    std::string out(prefix);
    out += line;
    // CPUID's brand string is padded with spaces at either end.
    while (!cpu.empty() && (cpu.front() == ' ' || cpu.front() == '\0')) cpu.remove_prefix(1);
    while (!cpu.empty() && (cpu.back() == ' ' || cpu.back() == '\0')) cpu.remove_suffix(1);
    out += cpu.empty() ? std::string_view("unknown") : cpu;
    out += '\n';
    return out;
}

// The first line of one report in the crash log, so reports appended one after another can be told apart.
inline std::string log_header(int year, int month, int day, int hour, int minute, int second, std::uint32_t pid) {
    char line[96];
    std::snprintf(line, sizeof line, "==== %04d-%02d-%02d %02d:%02d:%02d UTC, process %" PRIu32 " ====\n", year, month,
                  day, hour, minute, second, pid);
    return line;
}

// How LEONIDA_CRASH_LOG reads: unset or empty for the default file, "0" for no file, anything else a path.
enum class LogChoice { default_file, off, path };
inline LogChoice log_choice(std::string_view value) {
    if (value.empty()) return LogChoice::default_file;
    if (value == "0") return LogChoice::off;
    return LogChoice::path;
}

}  // namespace rt::crash_report
