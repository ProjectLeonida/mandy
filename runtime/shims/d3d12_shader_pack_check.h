#pragma once

#include <filesystem>

namespace rt::d3d12 {

// Uses the same Vulkan adapter selection as --run, without creating a logical device. An empty directory uses
// LEONIDA_SHADERS or shader_path.txt. Returns 0 for no known mismatch, 1 for blockers, 2 for an incomplete check.
int check_shader_pack(const std::filesystem::path& directory);

}  // namespace rt::d3d12
