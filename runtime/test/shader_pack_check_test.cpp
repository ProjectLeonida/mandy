// Metadata-only shader-pack checks. The fixtures are handwritten indexes, without any shader modules or game data.
#include "../src/shader_pack_check.h"
#include "test_temp_dir.h"

#include <cstdio>

#define CHECK(condition) do {                                      \
    if (!(condition)) {                                            \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                  \
    }                                                              \
} while (0)

int main() {
    using namespace leonida;
    const testfiles::ScratchDir temp("leonida_shader_pack_check");
    const auto& dir = temp.path();
    const auto write = [&](const std::filesystem::path& path, std::string_view text) {
        std::filesystem::create_directories((dir / path).parent_path());
        std::ofstream file(dir / path, std::ios::binary);
        return static_cast<bool>(file.write(text.data(), static_cast<std::streamsize>(text.size())));
    };
    ShaderPackCheck report;
    std::string error;
    CHECK(!check_shader_pack(dir, {}, report, error) && error.find("cannot open") != std::string::npos);
    CHECK(write("index.json", "{}"));
    CHECK(check_shader_pack(dir, {}, report, error) && report.indexed == 0 && report.exit_code() == 2);

    // Feature and capability metadata both count; an entry the runtime cannot look up does not.
    CHECK(write("index.json", "\xef\xbb\xbf" R"({
        "mesh": {"stage": "ms", "fssm_sha256": "x"},
        "pixel": {"stage": "ps", "fspx_code_sha256": "y", "features": ["fragmentShaderBarycentric"]},
        "pixel2": {"stage": "ps", "fssm_sha256": "z", "capabilities": ["FragmentBarycentricKHR"]},
        "ignored": {"stage": "ms"}
    })"));
    CHECK(check_shader_pack(dir, {true, true}, report, error));
    CHECK(report.indexed == 3 && !report.unknown && report.exit_code() == 0);
    CHECK(check_shader_pack(dir, {}, report, error));
    CHECK(report.indexed == 3 && report.native_mesh_missing == 1 && report.native_barycentric_missing == 2);
    CHECK(report.mesh_compute_unimplemented == 0 && report.barycentric_fallback_unimplemented == 0);
    CHECK(report.exit_code() == 1);

    // Match the builder's driver workarounds even when both Vulkan features are exposed. Intel's mesh rule
    // is limited to its Windows driver with mesh subgroup support; the barycentric rule is for Mesa ANV instead.
    CHECK(check_shader_pack(dir, shader_pack_device(true, true, true, 5, 0x8086, 0x56A0), report, error));
    CHECK(report.native_mesh_driver_blocked == 1 && report.native_barycentric_driver_blocked == 0);
    CHECK(report.native_mesh_missing == 0 && report.exit_code() == 1);
    CHECK(check_shader_pack(dir, shader_pack_device(true, true, false, 5, 0x8086, 0x56A0), report, error));
    CHECK(report.exit_code() == 0);
    CHECK(check_shader_pack(dir, shader_pack_device(true, true, true, 6, 0x8086, 0xE20B), report, error));
    CHECK(report.native_mesh_driver_blocked == 0 && report.native_barycentric_driver_blocked == 2);
    CHECK(report.native_barycentric_missing == 0 && report.exit_code() == 1);
    CHECK(check_shader_pack(dir, shader_pack_device(true, true, true, 1, 0x1002, 0x15E7), report, error));
    CHECK(report.native_barycentric_driver_blocked == 2 && report.exit_code() == 1);
    CHECK(check_shader_pack(dir, shader_pack_device(true, true, true, 3, 0x1002, 0x15E7), report, error));
    CHECK(report.exit_code() == 0);  // RADV on the same GPU is not affected.
    CHECK(check_shader_pack(dir, shader_pack_device(true, true, true, 1, 0x1002, 0x164E), report, error));
    CHECK(report.exit_code() == 0);  // AMD's proprietary driver on a different GPU is not affected.

    // Fallback markers are not proof of runtime support, even on a GPU that supports the native path.
    CHECK(write("index.json", R"({"basic": {"stage": "ps", "fssm_sha256": "a", "features": []}})"));
    CHECK(write("exe/index.json", R"({
        "mesh": {"stage": "ms", "fssm_sha256": "b", "mesh_compute": {"slot_bytes": 256}},
        "pixel": {"stage": "ps", "fssm_sha256": "c", "bary_fallback_input": {"location": 0}}
    })"));
    for (const ShaderPackDevice gpu : {ShaderPackDevice{}, ShaderPackDevice{true, true}}) {
        CHECK(check_shader_pack(dir, gpu, report, error));
        CHECK(report.indexed == 3 && !report.unknown && report.exit_code() == 1);
        CHECK(report.mesh_compute_unimplemented == 1 && report.barycentric_fallback_unimplemented == 1);
        CHECK(report.native_mesh_missing == 0 && report.native_barycentric_missing == 0);
    }

    CHECK(write("exe/index.json", R"({
        "old": {"stage": "ps", "fssm_sha256": "d"},
        "bad_mesh": {"stage": "ms", "fssm_sha256": "e", "mesh_compute": null},
        "bad_pixel": {"stage": "ps", "fssm_sha256": "f", "bary_fallback_input": {"location": 1.5}}
    })"));
    CHECK(check_shader_pack(dir, {}, report, error));
    CHECK(report.indexed == 4 && report.unknown == 3 && report.exit_code() == 2);
    CHECK(write("exe/index.json", R"({"wrong_stage": {"stage": "MS", "fssm_sha256": "b"}})"));
    CHECK(check_shader_pack(dir, {}, report, error));
    CHECK(report.indexed == 1 && report.unknown == 1 && report.exit_code() == 2);

    CHECK(write("fspx_gfx/index.json", "[]"));
    CHECK(check_shader_pack(dir, {}, report, error));
    CHECK(report.optional_index_errors.size() == 1 && report.exit_code() == 2);
    CHECK(report.optional_index_errors.front().find("fspx_gfx/index.json") != std::string::npos);
    CHECK(write("index.json", R"({"first_mesh": {"stage": "ms", "fssm_sha256": "a"}})"));
    CHECK(write("exe/index.json", R"({"later_mesh": {"stage": "ms", "fssm_sha256": "b"}})"));
    CHECK(check_shader_pack(dir, {}, report, error));
    CHECK(report.indexed == 2 && report.native_mesh_missing == 2 && report.exit_code() == 1);
    CHECK(report.optional_index_errors.size() == 1);

    CHECK(write("index.json", "{"));
    CHECK(!check_shader_pack(dir, {}, report, error) && error.find("not a JSON object") != std::string::npos);
    return 0;
}
