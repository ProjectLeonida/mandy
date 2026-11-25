// src/working_set.h: the memory probe's estimate of how much of a range is resident in RAM. Synthetic memory only.
#include "../src/working_set.h"

#include <cstdio>
#include <cstring>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    using leonida::working_set::resident_bytes;
    constexpr SIZE_T size = 8u << 20;
    auto* touched = static_cast<unsigned char*>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    auto* untouched = static_cast<unsigned char*>(VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    auto* reserved = static_cast<unsigned char*>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS));
    check(touched && untouched && reserved, "allocations");
    if (failures) return 1;
    std::memset(touched, 1, size);
    const auto base = [](const void* p) { return reinterpret_cast<std::uint64_t>(p); };

    check(resident_bytes(base(touched), size) == size, "written pages are resident");
    check(resident_bytes(base(untouched), size) == 0, "committed pages never touched are not resident");
    check(resident_bytes(base(reserved), size) == 0, "reserved pages are not resident");
    check(resident_bytes(base(touched), 0) == 0, "an empty range");
    // Half of a range written: the estimate is half of its size (every 16th page sampled, 2,048 pages).
    std::memset(untouched, 1, size / 2);
    check(resident_bytes(base(untouched), size) == size / 2, "half written, half resident");
    // VirtualUnlock on a range that is not locked removes it from the working set (Win32 VirtualUnlock remarks).
    VirtualUnlock(touched, size);
    check(resident_bytes(base(touched), size) < size / 4, "pages dropped from the working set are not resident");

    VirtualFree(touched, 0, MEM_RELEASE);
    VirtualFree(untouched, 0, MEM_RELEASE);
    VirtualFree(reserved, 0, MEM_RELEASE);
    if (failures) return 1;
    std::printf("working_set: written, untouched, reserved, half-written and trimmed ranges\n");
    return 0;
}
