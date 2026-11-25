#include "ffx_vk_backend.h"

#include <windows.h>

#include <cstdlib>

namespace leonida::ffx_vk {

namespace {
constexpr const char* loader_name = "amd_fidelityfx_loader_vk.dll";
constexpr const char* provider_name = "amd_fidelityfx_upscaler_vk.dll";
}

Upscaler::~Upscaler() {
    if (context_ && destroy_) destroy_(&context_, nullptr);
    if (module_) FreeLibrary(static_cast<HMODULE>(module_));
    if (provider_module_) FreeLibrary(static_cast<HMODULE>(provider_module_));
}

bool Upscaler::load(std::string& error) {
    if (module_) return true;
    std::string path = loader_name;
    if (const char* dir = std::getenv("LEONIDA_FFX_DIR"); dir && *dir) {
        // The loader asks for the provider by name; loading it first by full path lets that resolve without changing
        // the process-wide DLL search path.
        if (!provider_module_) provider_module_ = LoadLibraryA((std::string(dir) + "\\" + provider_name).c_str());
        path = std::string(dir) + "\\" + loader_name;
    }
    HMODULE module = LoadLibraryA(path.c_str());
    if (!module) {
        error = path + " not found (error " + std::to_string(GetLastError()) +
                "); set LEONIDA_FFX_DIR to the FidelityFX SDK's signedbin folder";
        return false;
    }
    create_ = reinterpret_cast<ffx::PfnCreateContext>(GetProcAddress(module, "ffxCreateContext"));
    destroy_ = reinterpret_cast<ffx::PfnDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
    query_ = reinterpret_cast<ffx::PfnQuery>(GetProcAddress(module, "ffxQuery"));
    dispatch_ = reinterpret_cast<ffx::PfnDispatch>(GetProcAddress(module, "ffxDispatch"));
    if (!create_ || !destroy_ || !query_ || !dispatch_) {
        FreeLibrary(module);
        error = path + " lacks one of ffxCreateContext, ffxDestroyContext, ffxQuery, ffxDispatch";
        return false;
    }
    module_ = module;
    return true;
}

ffx::ReturnCode Upscaler::create(const Device& device, const ffx::CreateContextDescUpscale& desc) {
    if (!create_ || context_) return ffx::RETURN_ERROR;
    ffx::CreateBackendVKDesc backend{{ffx::CREATE_CONTEXT_DESC_TYPE_BACKEND_VK, nullptr},
                                     {device.device, device.physical,
                                      reinterpret_cast<void*>(device.get_device_proc_addr)}};
    ffx::CreateContextDescUpscale upscale = desc;
    upscale.header.type = ffx::CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    upscale.header.pNext = &backend.header;
    const ffx::ReturnCode rc = create_(&context_, &upscale.header, nullptr);
    if (rc != ffx::RETURN_OK) {
        context_ = nullptr;
        return rc;
    }
    ffx::QueryGetProviderVersion version{{ffx::QUERY_DESC_TYPE_GET_PROVIDER_VERSION, nullptr}, 0, nullptr};
    if (query_(&context_, &version.header) == ffx::RETURN_OK && version.versionName) provider_ = version.versionName;
    return rc;
}

ffx::ReturnCode Upscaler::dispatch(const ffx::DispatchDescUpscale& desc) {
    if (!context_) return ffx::RETURN_ERROR;
    return dispatch_(&context_, &desc.header);
}

}  // namespace leonida::ffx_vk
