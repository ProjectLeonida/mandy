#pragma once

// Metadata-only check for mesh and barycentric requirements. It deliberately does not test shader modules,
// VS/PS interfaces, other Vulkan features, or which entries the title will actually use.
#include "shader_library.h"

namespace leonida {

enum class NativeShaderPath { mesh, barycentric };

namespace detail {

struct ShaderDriverRule {
    NativeShaderPath path;
    std::uint32_t driver, vendor, device;
};

inline constexpr ShaderDriverRule shader_driver_rules[] = {
#define LEONIDA_SHADER_DRIVER_RULE(path, driver, vendor, device) {NativeShaderPath::path, driver, vendor, device},
#include "shader_driver_rules.inc"
#undef LEONIDA_SHADER_DRIVER_RULE
};

}  // namespace detail

inline constexpr bool native_shader_driver_issue(NativeShaderPath path, std::uint32_t driver, std::uint32_t vendor,
                                                 std::uint32_t device) {
    return std::any_of(std::begin(detail::shader_driver_rules), std::end(detail::shader_driver_rules),
                       [&](const detail::ShaderDriverRule& rule) {
                           return rule.path == path && rule.driver == driver &&
                                  (!rule.vendor || rule.vendor == vendor) && (!rule.device || rule.device == device);
                       });
}

struct ShaderPackDevice {
    bool native_mesh = false;
    bool native_barycentric = false;
    bool native_mesh_driver_issue = false;
    bool native_barycentric_driver_issue = false;
};

inline constexpr ShaderPackDevice shader_pack_device(bool native_mesh, bool native_barycentric, bool mesh_subgroups,
                                                     std::uint32_t driver, std::uint32_t vendor, std::uint32_t device) {
    return {native_mesh, native_barycentric,
            mesh_subgroups && native_shader_driver_issue(NativeShaderPath::mesh, driver, vendor, device),
            native_shader_driver_issue(NativeShaderPath::barycentric, driver, vendor, device)};
}

struct ShaderPackCheck {
    std::size_t indexed = 0;
    std::size_t native_mesh_missing = 0;
    std::size_t native_mesh_driver_blocked = 0;
    std::size_t mesh_compute_unimplemented = 0;
    std::size_t native_barycentric_missing = 0;
    std::size_t native_barycentric_driver_blocked = 0;
    std::size_t barycentric_fallback_unimplemented = 0;
    std::size_t unknown = 0;
    std::vector<std::string> optional_index_errors;

    bool blocked() const {
        return native_mesh_missing || native_mesh_driver_blocked || mesh_compute_unimplemented ||
               native_barycentric_missing || native_barycentric_driver_blocked || barycentric_fallback_unimplemented;
    }
    int exit_code() const { return blocked() ? 1 : (!indexed || unknown || !optional_index_errors.empty() ? 2 : 0); }
};

namespace detail {

inline bool shader_pack_string_array(const Json* field) {
    return field && field->kind == Json::Kind::array &&
           std::all_of(field->array.begin(), field->array.end(), [](const Json& item) {
               return item.kind == Json::Kind::string;
           });
}

inline bool shader_pack_contains(const Json* array, std::string_view value) {
    return array && array->kind == Json::Kind::array &&
           std::any_of(array->array.begin(), array->array.end(), [&](const Json& item) {
               return item.kind == Json::Kind::string && item.string == value;
           });
}

inline void check_shader_index(const Json& index, ShaderPackDevice device, ShaderPackCheck& report) {
    for (const auto& [name, entry] : index.object) {
        const auto string = [&](std::string_view key) {
            const Json* field = entry.find(key);
            return field && field->kind == Json::Kind::string;
        };
        // As in ShaderLibrary: no program identity, no runtime lookup. Do not inspect or print shader names.
        if (!string("fssm_sha256") && !string("fspx_code_sha256")) continue;
        const Json* stage_field = entry.find("stage");
        const std::string_view stage = stage_field && stage_field->kind == Json::Kind::string
                                           ? std::string_view(stage_field->string) : std::string_view{};
        if (stage != "ms" && stage != "ps" && stage != "vs" && stage != "gs" && stage != "cs") {
            ++report.unknown;
            continue;
        }
        ++report.indexed;
        if (stage == "ms") {
            if (const Json* compute = entry.find("mesh_compute")) {
                if (compute->kind == Json::Kind::object) ++report.mesh_compute_unimplemented;
                else ++report.unknown;
            } else if (!device.native_mesh) {
                ++report.native_mesh_missing;
            } else if (device.native_mesh_driver_issue) {
                ++report.native_mesh_driver_blocked;
            }
        } else if (stage == "ps") {
            const Json* features = entry.find("features");
            const Json* capabilities = entry.find("capabilities");
            const bool needs_native = shader_pack_contains(features, "fragmentShaderBarycentric") ||
                                      shader_pack_contains(capabilities, "FragmentBarycentricKHR");
            if (needs_native) {
                if (!device.native_barycentric) ++report.native_barycentric_missing;
                else if (device.native_barycentric_driver_issue) ++report.native_barycentric_driver_blocked;
            }
            if (const Json* fallback = entry.find("bary_fallback_input")) {
                const Json* loc = fallback->find("location");
                if (loc && loc->kind == Json::Kind::number && loc->number >= 0 && loc->number <= UINT32_MAX &&
                    static_cast<std::uint32_t>(loc->number) == loc->number)
                    ++report.barycentric_fallback_unimplemented;
                else
                    ++report.unknown;
            } else if (!needs_native && !shader_pack_string_array(features) && !shader_pack_string_array(capabilities)) {
                ++report.unknown;  // old or incomplete PS metadata cannot establish barycentric requirements
            }
        }
    }
}

}  // namespace detail

// A missing or unreadable main index is fatal. Optional failures are reported alongside findings from readable indexes.
inline bool check_shader_pack(const std::filesystem::path& directory, ShaderPackDevice device, ShaderPackCheck& report,
                              std::string& error) {
    ShaderPackCheck checked;
    const auto scan = [&](const std::filesystem::path& index, std::string& why) {
        try {
            Json root;
            if (!read_shader_index(index, root, why)) return false;
            detail::check_shader_index(root, device, checked);
            return true;
        } catch (...) {
            why = "shader index could not be read";
            return false;
        }
    };
    if (!scan(directory / "index.json", error)) return false;
    for (const char* sub : shader_index_subdirectories) {
        const auto index = directory / sub / "index.json";
        std::error_code ec;
        const bool present = std::filesystem::exists(index, ec);
        std::string why;
        if (ec || (present && !scan(index, why))) {
            if (ec) why = ec.message();
            checked.optional_index_errors.push_back(std::string(sub) + "/index.json: " + why);
        }
    }
    report = std::move(checked);
    return true;
}

}  // namespace leonida
