// src/crash_report.h: the text of a fatal-fault report (registers, exception parameters, host line, log header and
// the LEONIDA_CRASH_LOG choice). Synthetic values only; no game data, no Windows.
#include "../src/crash_report.h"

#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

using namespace rt::crash_report;

void registers() {
    Registers r;
    r.rax = 0x1;
    r.rsp = 0x000000a1b2c3d4e0;
    r.r15 = 0xffffffffffffffff;
    r.rip = 0x00007ff612340abc;
    r.eflags = 0x10246;
    const std::string text = registers_lines(r, "[rt] fault: ");
    check(has(text, "[rt] fault: rax=0000000000000001 rbx=0000000000000000"), "rax and rbx lead the first line");
    check(has(text, "rsp=000000a1b2c3d4e0\n"), "rsp ends the second line");
    check(has(text, "r15=ffffffffffffffff\n"), "r15 ends the fourth line");
    check(has(text, "[rt] fault: rip=00007ff612340abc eflags=00010246\n"), "rip and eflags are the last line");
    std::size_t lines = 0;
    for (char c : text) lines += c == '\n';
    check(lines == 5, "five lines, each with the prefix");
}

void exceptions() {
    const std::uint64_t av[2] = {1, 0xc};
    check(exception_detail(kAccessViolation, 2, av, {}, "> ") == "> access violation: write of 0xc\n",
          "an access violation names the access and the address");
    const std::uint64_t exec[2] = {8, 0x1234};
    check(has(exception_detail(kAccessViolation, 2, exec, {}, ""), "execute of 0x1234"), "a DEP fault is an execute");
    const std::uint64_t page[3] = {0, 0x5000, 0xC000009Cu};
    check(exception_detail(kInPageError, 3, page, {}, "") == "in-page error: read of 0x5000, status 0xc000009c\n",
          "an in-page error adds the status that failed the read");
    const std::uint64_t cpp[4] = {0x19930520, 0x00000012345678a0, 0x00007ff6000f0000, 0x00007ff600000000};
    const std::string thrown = exception_detail(kCppException, 4, cpp, ".?AVruntime_error@std@@", "");
    check(has(thrown, "C++ exception: object 0x12345678a0, throw info 0x7ff6000f0000"), "a C++ throw names its object");
    check(has(thrown, ", type .?AVruntime_error@std@@\n"), "and its type when it could be read");
    check(!has(exception_detail(kCppException, 4, cpp, {}, ""), "type"), "no type when it could not");
    const std::uint64_t raw[3] = {0xa, 0xb, 0xc};
    check(exception_detail(0xC0000420u, 3, raw, {}, "") == "exception 0xc0000420, 3 parameters 0xa 0xb 0xc\n",
          "an unknown code prints its raw parameters");
    check(exception_detail(kIllegalInstruction, 0, nullptr, {}, "") == "illegal instruction 0xc000001d, 0 parameters\n",
          "a known code without parameters is named");
    check(exception_detail(kAccessViolation, 1, av, {}, "").starts_with("access violation 0xc0000005, 1 parameter "),
          "an access violation short of its parameters falls back to the raw form");
}

void host() {
    check(host_line("  AMD Ryzen 9 7900X 12-Core Processor          ", 10, 0, 26100, "[rt] fault: ") ==
              "[rt] fault: host: Windows 10.0.26100, CPU AMD Ryzen 9 7900X 12-Core Processor\n",
          "the host line trims CPUID's padding");
    check(has(host_line(std::string_view("\0\0", 2), 10, 0, 19045, ""), "CPU unknown"), "an empty brand is unknown");
}

void header_and_choice() {
    check(log_header(2026, 11, 26, 9, 5, 3, 4242) == "==== 2026-11-26 09:05:03 UTC, process 4242 ====\n",
          "the log header dates the report");
    check(log_choice("") == LogChoice::default_file, "LEONIDA_CRASH_LOG unset or empty: the default file");
    check(log_choice("0") == LogChoice::off, "LEONIDA_CRASH_LOG=0: no file");
    check(log_choice("D:\\logs\\crash.log") == LogChoice::path, "anything else is a path");
    check(log_choice("00") == LogChoice::path, "only a lone 0 turns it off");
}

}  // namespace

int main() {
    registers();
    exceptions();
    host();
    header_and_choice();
    std::printf(g_failures ? "crash_report_test: %d failure(s)\n" : "crash_report_test: all checks passed\n",
                g_failures);
    return g_failures ? 1 : 0;
}
