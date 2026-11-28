// library_loader's module names (shims/library_loader.cpp). kevincommits' finding in #1608 (item 2):
//
// - An empty name, or a path that ends in a slash, names no module but came out as ".dll", which then went to the
//   bundled loader's lookups as if it were a module's name.
// - GetProcAddress on a host module looks its name up once and caches it by HMODULE. FreeLibrary kept the entry, so a
//   different DLL loaded later at the same base was served the first one's shims.
//
// The bundled loader here is a fake that records the names it is asked for. The second case uses fixed_base_probe.dll
// (no relocations, one fixed base) copied under two names: wsock32.dll, for which this test registers a shim (the
// shims do not own wsock32.dll, so it stays a host module), and a plain name. No game files.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "shims.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

int probe_export() { return 42; }

const rt::shims::Entry kProbe[] = {{"wsock32.dll", "leonida_loader_probe", reinterpret_cast<void*>(&probe_export)}};
const rt::shims::Table kProbeTable(kProbe, 1);

std::vector<std::string> g_asked;
std::vector<std::wstring> g_asked_w;

void* fake_find_loaded(void*, const std::string& dll) {
    g_asked.push_back(dll);
    return nullptr;
}
void* fake_load(void*, const std::string& dll, uint32_t, bool& handled) {
    g_asked.push_back(dll);
    handled = false;
    return nullptr;
}
void* fake_get_proc(void*, void*, const std::string&, bool& owns) {
    owns = false;
    return nullptr;
}
bool fake_owns(void*, void*) { return false; }
void* fake_find_loaded_w(void*, const std::wstring& dll) {
    g_asked_w.push_back(dll);
    return nullptr;
}
void* fake_load_w(void*, const std::wstring& dll, uint32_t, bool& handled) {
    g_asked_w.push_back(dll);
    handled = false;
    return nullptr;
}
const rt::shims::BundledModuleOps kFake{fake_find_loaded, fake_load, fake_get_proc, fake_owns, fake_find_loaded_w,
                                        fake_load_w};

template <class F>
F shim(const char* name) {
    return reinterpret_cast<F>(rt::shims::find("kernel32.dll", name));
}

bool asked_for_dot_dll() {
    for (const auto& n : g_asked)
        if (n == ".dll") return true;
    for (const auto& n : g_asked_w)
        if (n == L".dll") return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: library_loader_names_test <fixed_base_probe.dll>\n");
        return 2;
    }
    using LoadA = HMODULE(WINAPI*)(LPCSTR);
    using LoadW = HMODULE(WINAPI*)(LPCWSTR);
    using GetProc = FARPROC(WINAPI*)(HMODULE, LPCSTR);
    using Free = BOOL(WINAPI*)(HMODULE);
    const auto load_a = shim<LoadA>("LoadLibraryA");
    const auto load_w = shim<LoadW>("LoadLibraryW");
    const auto get_mod_a = shim<LoadA>("GetModuleHandleA");
    const auto get_mod_w = shim<LoadW>("GetModuleHandleW");
    const auto get_proc = shim<GetProc>("GetProcAddress");
    const auto free_lib = shim<Free>("FreeLibrary");
    if (!load_a || !load_w || !get_mod_a || !get_mod_w || !get_proc || !free_lib) {
        std::fprintf(stderr, "FAIL the loader shims are not registered\n");
        return 1;
    }

    // 1. Names that name no module never reach the bundled loader as ".dll".
    int ctx = 0;
    rt::shims::add_bundled_loader(&ctx, &kFake);
    load_a("");
    load_a("C:\\leonida_no_such_dir\\");
    load_a("C:/leonida_no_such_dir/");
    load_w(L"");
    load_w(L"C:\\leonida_no_such_dir\\");
    load_w(L"C:\\leonida_\u00e9t\u00e9\\");
    get_mod_a("");
    get_mod_a("C:\\leonida_no_such_dir\\");
    get_mod_w(L"C:\\leonida_no_such_dir\\");
    get_mod_w(L"C:\\leonida_\u00e9t\u00e9\\");
    check(!asked_for_dot_dll(), "an empty name or a path ending in a slash is not looked up as \".dll\"");
    g_asked.clear();
    load_a("C:\\leonida_no_such_dir\\some_module");
    check(g_asked.size() == 1 && g_asked[0] == "some_module.dll", "a name without an extension still gets \".dll\"");
    rt::shims::remove_bundled_loader(&ctx);

    // 2. FreeLibrary drops the cached name: the next DLL at the same base is not served the old one's shims.
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("leonida_loader_names_" + std::to_string(GetCurrentProcessId()));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto shimmed = dir / "wsock32.dll";
    const auto plain = dir / "leonida_plain_probe.dll";
    std::filesystem::copy_file(argv[1], shimmed, std::filesystem::copy_options::overwrite_existing, ec);
    std::filesystem::copy_file(argv[1], plain, std::filesystem::copy_options::overwrite_existing, ec);

    HMODULE first = load_w(shimmed.c_str());
    check(first != nullptr, "the fixture loads as wsock32.dll");
    if (first) {
        check(reinterpret_cast<void*>(get_proc(first, "leonida_loader_probe")) ==
                  reinterpret_cast<void*>(&probe_export),
              "GetProcAddress on it serves wsock32.dll's shim");
        check(free_lib(first) == TRUE, "FreeLibrary unloads it");
        HMODULE second = load_w(plain.c_str());
        check(second == first, "the plain copy loads at the same base (fixed, no relocations)");
        if (second == first)
            check(get_proc(second, "leonida_loader_probe") == nullptr,
                  "GetProcAddress on the new module does not serve the freed module's shim");
        if (second) free_lib(second);
    }
    std::filesystem::remove_all(dir, ec);

    std::printf("%d failure(s)\n", g_failures);
    return g_failures ? 1 : 0;
}
