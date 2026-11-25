// RT-344: AMD's PC FidelityFX upscaler on a Vulkan device, the AMD counterpart of #4801's DLSS backend. The game calls
// the five FFX functions of the package's amd_fidelityfx_upscaler_xbox.dll (fsr_dispatch_test, docs/d3d12_vulkan.md
// A.3); this backend answers the same calls with AMD's signed amd_fidelityfx_loader_vk.dll from the public
// FidelityFX SDK (v2.4.0 and later), on the layer's own VkDevice, so no D3D12 device is involved. The loader picks
// the newest upscaler the GPU and driver support (measured: FSR 4.1.2 on RDNA4 under RADV and on RDNA3 under AMD's
// Windows driver 26.11.2, FSR 3.1.6 on an RDNA2 iGPU and on NVIDIA; docs/d3d12_vulkan.md A.3). The DLLs are the user's: they are looked up in LEONIDA_FFX_DIR, else next to the program, and nothing of
// AMD's is in the repository (its licence allows redistribution in binary form and forbids reverse engineering).
//
// The shim that routes the game's calls here, and maps its resources to the layer's VkImages, is a later step.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <string>

#include "../test/ffx_api_min.h"

namespace leonida::ffx_vk {

// Device extensions the upscaler's shaders use, as AMD's own Vulkan device setup enables them where the GPU has them
// (FidelityFX SDK 2.4.0, Kits/Cauldron2/framework/render/vk/device_vk.cpp): compute-shader derivatives (FSR 3.1, and
// FSR 4's INT8 prepass), cooperative matrices and 8-bit floats (FSR 4's network on RDNA4). The device the context is
// created on must enable those it has, with all their features: drivers run the shaders without them, but the
// validation layer then reports VUID-VkShaderModuleCreateInfo-pCode-08740 and -08742 (measured on RADV and on AMD's
// Windows driver, #4999).
inline constexpr const char* wanted_device_extensions[] = {
    "VK_KHR_compute_shader_derivatives",
    "VK_KHR_cooperative_matrix",
    "VK_EXT_shader_float8",
};

struct Device {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;
};

class Upscaler {
public:
    Upscaler() = default;
    Upscaler(const Upscaler&) = delete;
    Upscaler& operator=(const Upscaler&) = delete;
    ~Upscaler();

    // Loads amd_fidelityfx_loader_vk.dll. False, with `error` saying why, when it is not found.
    bool load(std::string& error);
    // One upscale context on `device`. `desc` is the game's create description: its flags and sizes are used, its
    // chain (the console backend descriptor) is not, since this context gets the Vulkan backend instead.
    ffx::ReturnCode create(const Device& device, const ffx::CreateContextDescUpscale& desc);
    // The provider the loader chose for the context, e.g. "4.1.2"; empty before create.
    const std::string& provider() const { return provider_; }
    // Records one upscale into `desc.commandList` (a VkCommandBuffer). Each resource is a VkImage in the layout its
    // `state` names: COMPUTE_READ is SHADER_READ_ONLY_OPTIMAL, UNORDERED_ACCESS is GENERAL.
    ffx::ReturnCode dispatch(const ffx::DispatchDescUpscale& desc);

private:
    void* module_ = nullptr;
    void* provider_module_ = nullptr;
    ffx::Context context_ = nullptr;
    ffx::PfnCreateContext create_ = nullptr;
    ffx::PfnDestroyContext destroy_ = nullptr;
    ffx::PfnQuery query_ = nullptr;
    ffx::PfnDispatch dispatch_ = nullptr;
    std::string provider_;
};

}  // namespace leonida::ffx_vk
