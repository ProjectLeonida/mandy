// shims/io_trace.h: LEONIDA_IO_TRACE's lines for reads, written through at once and readable while open. A
// temporary folder; no game files, no GPU.
#include "io_trace.h"

#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

static std::string read_all(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);  // works only because the trace shares its file for reading
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("leonida_io_trace_test_" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    const fs::path trace = dir / "io.log";
    const fs::path data = dir / "data.bin";
    { std::ofstream(data, std::ios::binary) << std::string(8192, 'x'); }

    // out() opens the file once, at its first call, so the variable is set before anything else.
    SetEnvironmentVariableW(L"LEONIDA_IO_TRACE", trace.c_str());
    CHECK(rt::io_trace::enabled());

    HANDLE file = CreateFileW(data.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    CHECK(file != INVALID_HANDLE_VALUE);
    // A positioned read: the handle's current offset.
    LARGE_INTEGER at{};
    at.QuadPart = 0x1000;
    CHECK(SetFilePointerEx(file, at, nullptr, FILE_BEGIN));
    rt::io_trace::note_read_file(file, 512, nullptr);
    // An overlapped read: the OVERLAPPED's offset, whatever the handle's position.
    OVERLAPPED overlapped{};
    overlapped.Offset = 0x1800;
    rt::io_trace::note_read_file(file, 256, &overlapped);
    CloseHandle(file);
    // A handle value used again for another file names the new file.
    const fs::path other = dir / "other.bin";
    { std::ofstream(other, std::ios::binary) << "y"; }
    HANDLE second = CreateFileW(other.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    CHECK(second != INVALID_HANDLE_VALUE);
    rt::io_trace::note_read_file(second, 1, nullptr);
    CloseHandle(second);
    // A DirectStorage-style line, with a name pointer the title owns and one that is not readable.
    rt::io_trace::line("ds %s off=0x%llx name=%s", "<memory>", 0ull, rt::io_trace::safe_name("probe").c_str());
    rt::io_trace::line("ds name=%s", rt::io_trace::safe_name(reinterpret_cast<const char*>(16)).c_str());
    rt::io_trace::line("ds name=%s", rt::io_trace::safe_name(nullptr).c_str());

    // Read back while the trace is still open: every line is already on disk.
    const std::string text = read_all(trace);
    std::fprintf(stderr, "%s", text.c_str());
    CHECK(text.find("rf ") != std::string::npos && text.find("data.bin off=0x1000 size=512") != std::string::npos);
    CHECK(text.find("data.bin off=0x1800 size=256") != std::string::npos);
    CHECK(text.find("other.bin off=0x0 size=1") != std::string::npos);
    CHECK(text.find("ds <memory> off=0x0 name=probe") != std::string::npos);
    CHECK(text.find("name=<unreadable>") != std::string::npos);
    CHECK(text.find("name=-") != std::string::npos);
    // Each line starts with hh:mm:ss.mmm.
    CHECK(text.size() > 13 && text[2] == ':' && text[5] == ':' && text[8] == '.');
    // A pipe is not a disk file: no line.
    HANDLE read_end = nullptr, write_end = nullptr;
    CHECK(CreatePipe(&read_end, &write_end, nullptr, 0));
    const std::size_t before = read_all(trace).size();
    rt::io_trace::note_read_file(read_end, 4, nullptr);
    CHECK(read_all(trace).size() == before);
    CloseHandle(read_end);
    CloseHandle(write_end);

    std::printf("io_trace: positioned, overlapped and DirectStorage lines, written through and shared for reading\n");
    return 0;
}
