// rt::fatal_report::describe (runtime/src/fatal_report.h): the fatal-exception lines in the title's terms, from a
// made-up exception record, context and stack over a stand-in "image" buffer. No game data, no GPU.
//   1. a fault inside the image is printed as an RVA;
//   2. the stack's qwords that point into the image are listed innermost first, others skipped, up to the limit;
//   3. the registers line carries rbx and rdi;
//   4. rbx at an object whose first qword points into the image gives that vtable's RVA;
//   5. a fault outside the image, an unknown image and an unreadable stack pointer give a report without faulting.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <string>

#include "../src/fatal_report.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }

}  // namespace

int main() {
    static std::uint8_t image[0x1000];
    const auto base = reinterpret_cast<std::uintptr_t>(image);
    const std::size_t size = sizeof(image);
    std::uintptr_t object[2] = {base + 0x800, 0};  // "vtable" at +0x800
    std::uintptr_t stack[8] = {0x1234, base + 0x10, 7, base + 0x20, base + 0x30, 0, base + 0x40, base + 0x50};

    EXCEPTION_RECORD rec{};
    rec.ExceptionCode = EXCEPTION_BREAKPOINT;
    rec.ExceptionAddress = image + 0x650;
    CONTEXT ctx{};
    ctx.Rsp = reinterpret_cast<DWORD64>(stack);
    ctx.Rbx = reinterpret_cast<DWORD64>(object);
    ctx.Rdi = 0x21c000;
    EXCEPTION_POINTERS info{&rec, &ctx};

    const std::string r = rt::fatal_report::describe(&info, base, size, 8, 4);
    std::printf("%s", r.c_str());
    check(has(r, "fault at title +0x650"), "1. fault as an RVA");
    check(has(r, "stack (innermost first): +0x10 +0x20 +0x30 +0x40\n"), "2. image addresses in order, limit 4");
    check(has(r, "rbx=") && has(r, "rdi=21c000"), "3. registers");
    check(has(r, "vtable title +0x800"), "4. rbx's vtable");

    rec.ExceptionAddress = reinterpret_cast<void*>(0x1000);
    ctx.Rsp = 0x10;  // unreadable
    ctx.Rbx = 0;
    const std::string o = rt::fatal_report::describe(&info, base, size);
    check(has(o, "outside the title image") && has(o, "stack (innermost first): none"), "5. outside, bad stack");
    const std::string u = rt::fatal_report::describe(&info, 0, 0);
    check(has(u, "outside the title image") && !has(u, "stack (innermost first)") && has(u, "registers"),
          "5. image not known");
    check(has(rt::fatal_report::describe(nullptr, base, size), "fault at 0x0"), "5. no exception pointers");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
