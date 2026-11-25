// leonida::kernel_writes::read_file (runtime/src/title_page_watch.h) into memory that is reserved until its first
// touch, as xmem maps the title's physical pages (SEC_RESERVE section views committed by a fault handler): the kernel's
// write does not fault such a page in, so the read fails with ERROR_NOACCESS unless the committer runs first. Then
// kernel_writes::hold, the pin a host import keeps over its pages: recorded with no opener set (no tracker yet), seen
// by for_each_pin until release(), untouched by a file read's pin ending over the same pages. No game data.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/title_page_watch.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

constexpr DWORD kViewBytes = 1 << 20;
constexpr DWORD kFileBytes = 256 * 1024;

std::uint8_t* g_view = nullptr;
int g_commits = 0;
int g_opens = 0;
std::uintptr_t g_opened_begin = 0, g_opened_end = 0;

void count_open(std::uintptr_t begin, std::uintptr_t end) {
    ++g_opens;
    g_opened_begin = begin;
    g_opened_end = end;
}

// The bytes of [begin, end) the pins cover.
std::size_t pinned_bytes(std::uintptr_t begin, std::uintptr_t end) {
    std::size_t bytes = 0;
    leonida::kernel_writes::for_each_pin(begin, end, [&](std::uintptr_t from, std::uintptr_t to) { bytes += to - from; });
    return bytes;
}

void check_holds() {
    namespace kw = leonida::kernel_writes;
    static std::uint8_t pages[3 * 4096];
    const auto begin = reinterpret_cast<std::uintptr_t>(pages);
    kw::set_opener(nullptr);
    check(pinned_bytes(begin, begin + sizeof(pages)) == 0, "no pin before the hold");
    check(kw::hold(pages, 0) == 0 && kw::hold(nullptr, 4096) == 0, "an empty hold pins nothing");
    // No opener: begin_write records nothing, a hold does (a tracker made later must still pass over the pages).
    check(kw::begin_write(pages, 4096, nullptr) == 0, "begin_write records nothing without an opener");
    const std::uint64_t first = kw::hold(pages, 2 * 4096);
    check(first != 0 && pinned_bytes(begin, begin + sizeof(pages)) == 2 * 4096,
          "a hold without an opener pins its range");
    check(pinned_bytes(begin + 2 * 4096, begin + sizeof(pages)) == 0, "and nothing past it");
    // With an opener: the range is opened once the pin is in.
    kw::set_opener(&count_open);
    const std::uint64_t second = kw::hold(pages + 4096, 2 * 4096);
    check(second != 0 && second != first && g_opens == 1 && g_opened_begin == begin + 4096 &&
              g_opened_end == begin + 3 * 4096,
          "a hold with an opener opens exactly its range");
    // A file read's pin over the same pages ends without ending the holds.
    const std::uint64_t read = kw::begin_write(pages, sizeof(pages), nullptr);
    check(read != 0 && pinned_bytes(begin, begin + sizeof(pages)) == 2 * 4096 + 2 * 4096 + sizeof(pages),
          "a file read's pin adds to the holds");
    kw::end_write(read, false);
    check(pinned_bytes(begin, begin + sizeof(pages)) == 4 * 4096, "its end leaves the holds");
    kw::end_write(first, true);  // as a pending overlapped read would: a hold has no OVERLAPPED, so it ends too
    check(pinned_bytes(begin, begin + 4096) == 0 && pinned_bytes(begin + 4096, begin + sizeof(pages)) == 2 * 4096,
          "ending one hold leaves the other");
    kw::release(second);
    kw::release(0);
    check(pinned_bytes(begin, begin + sizeof(pages)) == 0, "released, no pin is left");
    kw::set_opener(nullptr);
}

// Stands in for rt::xmem::commit_physical_range: commits the part of the test view the range covers.
bool commit_view(const void* address, std::size_t bytes) {
    ++g_commits;
    const auto* begin = static_cast<const std::uint8_t*>(address);
    if (begin < g_view || begin + bytes > g_view + kViewBytes) return true;  // not ours
    return VirtualAlloc(const_cast<std::uint8_t*>(begin), bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

}  // namespace

int main() {
    wchar_t dir[MAX_PATH], path[MAX_PATH];
    GetTempPathW(MAX_PATH, dir);
    GetTempFileNameW(dir, L"kwt", 0, path);
    std::vector<std::uint8_t> bytes(kFileBytes);
    for (DWORD i = 0; i < kFileBytes; ++i) bytes[i] = static_cast<std::uint8_t>(i * 7 + (i >> 9));
    HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    DWORD written = 0;
    check(file != INVALID_HANDLE_VALUE && WriteFile(file, bytes.data(), kFileBytes, &written, nullptr) &&
              written == kFileBytes, "temporary file written");

    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, kViewBytes,
                                        nullptr);
    g_view = section ? static_cast<std::uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kViewBytes))
                     : nullptr;
    MEMORY_BASIC_INFORMATION info{};
    check(g_view && VirtualQuery(g_view, &info, sizeof(info)) && info.State == MEM_RESERVE,
          "the destination view is reserved, not committed");
    std::uint8_t* destination = g_view + 0x10000;

    // Without a committer: what the title saw (Windows answers ERROR_NOACCESS; another host may fault the page in).
    OVERLAPPED at{};
    DWORD got = 0;
    const BOOL plain = leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at);
    std::printf("   without a committer: %s, error %lu\n", plain ? "read" : "failed", plain ? 0ul : GetLastError());

    leonida::kernel_writes::set_committer(&commit_view);
    at = {};
    got = 0;
    const BOOL ok = leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at);
    check(ok && got == kFileBytes, "with the committer the read completes");
    check(ok && std::memcmp(destination, bytes.data(), kFileBytes) == 0, "and the destination holds the file's bytes");
    check(plain || g_commits == 1, "the committer ran once, for the failed read's retry");

    // A read that succeeds first time does not call the committer.
    const int before = g_commits;
    at = {};
    check(leonida::kernel_writes::read_file(file, destination, kFileBytes, &got, &at) && g_commits == before,
          "a read into committed memory does not call the committer");

    leonida::kernel_writes::set_committer(nullptr);
    check_holds();
    UnmapViewOfFile(g_view);
    CloseHandle(section);
    CloseHandle(file);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
