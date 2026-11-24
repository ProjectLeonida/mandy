// pso_warmup <dir of .dxbc compute programs> <pipeline_cache.bin to write>
//
// Builds every compute program in a folder through the runtime's own Vulkan-backed ID3D12Device
// (create_vulkan_device_standalone, the device gta6rt --run uses with LEONIDA_D3D12=vulkan), then writes that device's
// pipeline cache. Driver caches are keyed on the device's enabled extensions and features; vk_pipeline_check enables a
// different set, so its cache misses for the runtime's pipelines (measured on an RTX 3070: a 28 s compile stays a 28 s
// compile with that seed loaded). A cache written here is the runtime's own, and placed beside gta6rt.exe as
// pipeline_cache.bin it seeds the game's device.
//
// The programs are the user's carved files (for example <work>/shaders/bvh/*.dxbc); LEONIDA_SHADERS must point at the
// translated SPIR-V the runtime looks them up in. Nothing of the game is in this file. Exit 0 when every program built,
// 1 when some did not (the cache is still written), 2 for usage or setup errors.
#include <windows.h>
#include <d3d12.h>

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../shims/d3d12_com.h"

int wmain(int argc, wchar_t** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: pso_warmup <dir of .dxbc compute programs> <pipeline_cache.bin to write>\n");
        return 2;
    }
    wchar_t* shaders = nullptr;
    std::size_t shaders_length = 0;
    const bool have_shaders = _wdupenv_s(&shaders, &shaders_length, L"LEONIDA_SHADERS") == 0 && shaders && *shaders;
    std::free(shaders);
    if (!have_shaders) {
        std::fprintf(stderr, "pso_warmup: set LEONIDA_SHADERS to the translated SPIR-V (<work>/shaders/spirv)\n");
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    std::error_code error;
    if (!std::filesystem::is_directory(dir, error)) {
        std::fprintf(stderr, "pso_warmup: %ls is not a folder\n", dir.c_str());
        return 2;
    }

    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device))) || !device) {
        std::fprintf(stderr, "pso_warmup: no Vulkan-backed device\n");
        return 2;
    }

    int built = 0, failed = 0;
    for (const auto& item : std::filesystem::directory_iterator(dir, error)) {
        if (!item.is_regular_file() || item.path().extension() != L".dxbc") continue;
        std::ifstream file(item.path(), std::ios::binary);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (bytes.empty()) continue;
        D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
        desc.CS = {bytes.data(), bytes.size()};
        const auto start = std::chrono::steady_clock::now();
        ID3D12PipelineState* pso = nullptr;
        const HRESULT hr = device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (SUCCEEDED(hr) && pso) {
            ++built;
            pso->Release();
            if (seconds >= 1.0) std::printf("%ls: %.1f s\n", item.path().filename().c_str(), seconds);
        } else {
            ++failed;
            std::printf("%ls: failed (hr=0x%08lx)\n", item.path().filename().c_str(), static_cast<unsigned long>(hr));
        }
    }
    const bool saved = rt::d3d12::save_pipeline_cache(device, argv[2]);
    std::printf("%d built, %d failed; cache %s\n", built, failed, saved ? "written" : "NOT written");
    device->Release();
    if (!saved) return 2;
    return failed ? 1 : 0;
}
