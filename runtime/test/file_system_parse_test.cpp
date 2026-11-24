// The console-path parse of shims/file_system.cpp (console_drive, lies_under, parse_console) against the version it
// replaced, which lowered copies of the path and both roots and grew each component a character at a time.
//
// The reference below is that earlier code, kept verbatim but for reading the roots from its arguments. Both run on
// hand-picked edge cases and on 200 000 generated paths (mixed case, both separators, ".", "..", ':' and non-ASCII
// letters, the roots' own spellings), and must give the same Parsed result, the same parts and the same prefix. The
// T: scratch folder is checked as a side effect: made when missing, left alone when present, the last error kept.
//
// The .cpp is included so the test reaches its internal functions.
#include "file_system.cpp"

#include <cstdio>
#include <random>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

namespace ref {

using rt::fs::Drive;
using rt::fs::Parsed;

std::wstring lower(std::wstring s) {
    for (auto& c : s) c = static_cast<wchar_t>(std::towlower(c));
    return s;
}

bool lies_under(const std::wstring& path, const std::filesystem::path& root) {
    std::wstring base = lower(root.native());
    for (auto& c : base) if (c == L'/') c = L'\\';
    while (!base.empty() && base.back() == L'\\') base.pop_back();
    if (base.empty()) return false;
    std::wstring p = lower(path);
    for (auto& c : p) if (c == L'/') c = L'\\';
    return p.compare(0, base.size(), base) == 0 && (p.size() == base.size() || p[base.size()] == L'\\');
}

Drive console_drive(const std::wstring& path, size_t& prefix_length, const std::filesystem::path& game,
                    const std::filesystem::path& overlay) {
    if (path.size() >= 5 && lower(path.substr(0, 5)) == L"game:") {
        prefix_length = 5;
        return Drive::Game;
    }
    if (path.size() < 2 || path[1] != L':' || (path.size() > 2 && !rt::fs::is_separator(path[2]))) return Drive::None;
    const wchar_t letter = static_cast<wchar_t>(std::towlower(path[0]));
    if (letter != L'g' && letter != L't') return Drive::None;
    if (game.empty() && overlay.empty()) return Drive::None;
    if (lies_under(path, game) || lies_under(path, overlay)) return Drive::None;
    prefix_length = 2;
    return letter == L'g' ? Drive::Game : Drive::Scratch;
}

Parsed parse_console(const std::wstring& path, std::vector<std::wstring>& parts, const std::filesystem::path& game,
                     const std::filesystem::path& overlay) {
    size_t prefix_length = 0;
    const Drive drive = console_drive(path, prefix_length, game, overlay);
    if (drive == Drive::None) return Parsed::Host;
    parts.clear();
    if (drive == Drive::Scratch) parts.push_back(rt::fs::kScratchDir);
    const size_t floor = parts.size();
    std::wstring part;
    auto flush = [&]() -> bool {
        if (part.empty() || part == L".") {
            part.clear();
            return true;
        }
        if (part == L"..") {
            if (parts.size() <= floor) return false;
            parts.pop_back();
        } else if (part.find(L':') != std::wstring::npos) {
            return false;
        } else {
            parts.push_back(part);
        }
        part.clear();
        return true;
    };
    for (size_t i = prefix_length; i < path.size(); ++i) {
        const wchar_t c = path[i];
        if (rt::fs::is_separator(c)) {
            if (!flush()) return Parsed::Refused;
        } else {
            part.push_back(c);
        }
    }
    if (!flush()) return Parsed::Refused;
    if (drive == Drive::Game && !parts.empty() && rt::fs::internal_dir(parts.front().c_str())) return Parsed::Refused;
    return Parsed::Console;
}

}  // namespace ref

// Both parses of one path agree, and so do console_drive and lies_under. The reference has no scratch-folder side
// effect; that is checked on its own at the end.
bool same(const std::wstring& path, const std::filesystem::path& game, const std::filesystem::path& overlay) {
    std::vector<std::wstring> a{L"stale"}, b{L"stale"};
    const rt::fs::Parsed pa = ref::parse_console(path, a, game, overlay);
    const rt::fs::Parsed pb = rt::fs::parse_console(path, b);
    size_t la = 99, lb = 99;
    const rt::fs::Drive da = ref::console_drive(path, la, game, overlay);
    const rt::fs::Drive db = rt::fs::console_drive(path, lb);
    // On Host neither parse touches parts; on Refused the contents are unspecified to callers, but still compared.
    return pa == pb && a == b && da == db && la == lb && ref::lies_under(path, game) == rt::fs::lies_under(path, game) &&
           ref::lies_under(path, overlay) == rt::fs::lies_under(path, overlay);
}

std::wstring random_path(std::mt19937& rng, const std::vector<std::wstring>& seeds) {
    static const std::vector<std::wstring> pieces = {
        L"game:", L"GAME:", L"GaMe:", L"gam", L"G:", L"g:", L"T:", L"t:", L"C:", L"x:", L"\\", L"/", L"\\\\", L".",
        L"..", L"...", L"a", L"Data", L"x64", L"levels", L"file.rpf", L"a:b", L":", L"\u00e9", L"\u00c9", L"\u212a",
        L".gta6rt-deleted", L".GTA6RT-TEMP", L".gta6rt-copy", L"long_component_name_beyond_small_buffer", L" "};
    std::wstring s;
    if (rng() % 3 == 0) s = seeds[rng() % seeds.size()];
    const int n = static_cast<int>(rng() % 9);
    for (int i = 0; i < n; ++i) s += pieces[rng() % pieces.size()];
    if (rng() % 7 == 0) s.push_back(L'\0');
    return s;
}

}  // namespace

int main() {
    wchar_t tmp[MAX_PATH];
    ::GetTempPathW(MAX_PATH, tmp);
    const std::filesystem::path root = std::filesystem::path(tmp) / L"leonida_file_system_parse_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    const std::filesystem::path game = root / L"Game";
    const std::filesystem::path overlay = root / L"Overlay";
    std::filesystem::create_directories(game, ec);
    std::filesystem::create_directories(overlay, ec);
    const std::filesystem::path scratch = overlay / rt::fs::kScratchDir;

    // No roots: G:/T: are host paths, GAME: still parses.
    rt::fs::configure({}, {});
    for (const wchar_t* p : {L"G:\\x", L"t:/y", L"GAME:\\a\\b", L"game:", L"", L"G", L"G:", L"T:x"})
        check(same(p, {}, {}), "no roots: same as the reference");

    rt::fs::configure(game, overlay);
    const std::wstring g = game.native(), o = overlay.native();
    std::wstring g_slash = g;
    for (auto& c : g_slash) if (c == L'\\') c = L'/';
    const std::vector<std::wstring> seeds = {g,  o,  g + L"\\", o + L"\\\\", ref::lower(g), g_slash,
                                             g + L"x", o.substr(0, o.size() - 1), L"G:\\", L"T:\\", L"game:\\"};

    const std::vector<std::wstring> edges = {
        L"game:", L"GAME:\\", L"game:/a//b/./c/../d", L"game:\\..", L"game:\\a\\..\\..", L"game:\\a:b",
        L"game:\\.gta6rt-deleted\\x", L"game:\\.GTA6RT-COPY", L"game:\\x\\.gta6rt-deleted", L"G:\\a\\b.c",
        L"g:/a", L"G:", L"G:x", L"T:\\", L"t:", L"T:\\..", L"T:\\a\\..\\..", L"T:\\a\\..", L"C:\\Windows",
        L"\\\\server\\share", L"gam", L"gamE", L"GAME:\u00e9\\\u00c9", L"G\u212a:", g, g + L"\\data", o,
        o + L"\\x", g_slash + L"/data", ref::lower(g) + L"\\x", g + L"x\\y", L"game:\\a\\.", L"game:\\.\\.\\a",
        L"game:\\...\\a", std::wstring(L"game:\\a\0b", 9), std::wstring(L"game:\\a\0:b", 10)};
    for (const auto& p : edges) check(same(p, game, overlay), "edge case: same as the reference");

    std::mt19937 rng(0x5eed);
    int mismatches = 0;
    for (int i = 0; i < 200000; ++i) {
        const std::wstring p = random_path(rng, seeds);
        if (!same(p, game, overlay) && ++mismatches <= 5) std::printf("  mismatch on generated path #%d\n", i);
    }
    check(mismatches == 0, "200000 generated paths: same as the reference");

    // lies_under with roots whose own spelling varies: trailing separators, '/' and case.
    for (const auto& r : {std::filesystem::path(g + L"\\\\"), std::filesystem::path(g_slash + L"/"),
                                          std::filesystem::path(L"\\"), std::filesystem::path(L"/"),
                                          std::filesystem::path(L""), std::filesystem::path(ref::lower(o))}) {
        for (const auto& p : edges) check(ref::lies_under(p, r) == rt::fs::lies_under(p, r), "lies_under on a root");
    }

    // The T: scratch folder: made by the first T: parse, kept by the next, and the last error survives either way.
    std::filesystem::remove_all(scratch, ec);
    std::vector<std::wstring> parts;
    ::SetLastError(1234);
    check(rt::fs::parse_console(L"T:\\a", parts) == rt::fs::Parsed::Console, "T: parses");
    check(::GetLastError() == 1234, "T: parse keeps the last error when it makes the folder");
    check(std::filesystem::is_directory(scratch, ec), "T: parse makes the scratch folder");
    ::SetLastError(4321);
    check(rt::fs::parse_console(L"T:\\b", parts) == rt::fs::Parsed::Console, "T: parses again");
    check(::GetLastError() == 4321, "T: parse keeps the last error when the folder is there");
    check(std::filesystem::is_directory(scratch, ec), "the scratch folder is still there");
    // A file in the folder's place: create_directories is still asked (and fails, as before), the file is untouched.
    std::filesystem::remove_all(scratch, ec);
    { std::FILE* f = _wfopen(scratch.c_str(), L"wb"); if (f) std::fclose(f); }
    ::SetLastError(77);
    check(rt::fs::parse_console(L"T:\\c", parts) == rt::fs::Parsed::Console, "T: parses over a file");
    check(::GetLastError() == 77, "T: parse over a file keeps the last error");
    check(std::filesystem::is_regular_file(scratch, ec), "a file in the scratch folder's place is left alone");
    std::filesystem::remove(scratch, ec);
    // No overlay: nothing is made.
    rt::fs::configure(game, {});
    check(rt::fs::parse_console(L"T:\\d", parts) == rt::fs::Parsed::Console, "T: parses without an overlay");
    check(!std::filesystem::exists(scratch, ec), "no overlay: no scratch folder");

    rt::fs::configure({}, {});
    std::filesystem::remove_all(root, ec);
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
