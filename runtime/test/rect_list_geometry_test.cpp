// Portable CPU-only unit tests for runtime/src/rect_list_geometry.h (RT-RECT-LIST-GEOMETRY-PREAMBLE-AND-TESTS):
// SPIR-V preamble parsing, validation/refusal rules, and generated geometry shader framing/decorations.
// Hand-built SPIR-V words only; no Vulkan, no GPU, no game files.
#include "../src/rect_list_geometry.h"

#include <cstdio>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);   \
        return 1;                                                               \
    }                                                                           \
} while (0)

namespace {

using Words = std::vector<std::uint32_t>;

constexpr std::uint32_t kOpEntryPoint = 15, kOpExecutionMode = 16, kOpCapability = 17, kOpMemoryModel = 14;
constexpr std::uint32_t kOpTypeFloat = 22, kOpTypeVector = 23, kOpTypeInt = 21, kOpTypePointer = 32;
constexpr std::uint32_t kOpFunction = 54, kOpFunctionEnd = 56, kOpVariable = 59, kOpDecorate = 71;
constexpr std::uint32_t kOpEmitVertex = 218, kOpEndPrimitive = 219, kOpLabel = 248, kOpReturn = 253, kOpNop = 0;
constexpr std::uint32_t kDecoBuiltIn = 11, kDecoLocation = 30;
constexpr std::uint32_t kBuiltInPosition = 0, kBuiltInPointSize = 1, kBuiltInClipDistance = 3, kBuiltInLayer = 9;
constexpr std::uint32_t kStorageInput = 1, kStorageOutput = 3;

void emit(Words& w, std::uint32_t op, std::initializer_list<std::uint32_t> args) {
    w.push_back((std::uint32_t(args.size() + 1) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

// Hand-built vertex shader SPIR-V with optional Position, PointSize, extra builtin, and Location outputs.
Words make_vs(bool with_pos, bool with_point_size, std::initializer_list<std::uint32_t> locs,
              std::uint32_t extra_builtin = 0xffffffffu, bool bad_vec_count = false, bool int_scalar = false,
              std::size_t body_nops = 8) {
    Words w{0x07230203u, 0x00010000u, 0, 100, 0};
    if (with_pos) emit(w, kOpDecorate, {10, kDecoBuiltIn, kBuiltInPosition});
    if (with_point_size) emit(w, kOpDecorate, {11, kDecoBuiltIn, kBuiltInPointSize});
    if (extra_builtin != 0xffffffffu) emit(w, kOpDecorate, {12, kDecoBuiltIn, extra_builtin});
    std::uint32_t next_var = 20;
    for (std::uint32_t loc : locs) {
        emit(w, kOpDecorate, {next_var++, kDecoLocation, loc});
    }
    // An Input variable at Location 0 (must be ignored by output expansion).
    emit(w, kOpDecorate, {15, kDecoLocation, 0});

    emit(w, kOpTypeFloat, {2, 32});
    emit(w, kOpTypeInt, {3, 32, 0});
    emit(w, kOpTypeVector, {4, 2, 4});                             // vec4 float
    emit(w, kOpTypeVector, {5, int_scalar ? 3u : 2u, bad_vec_count ? 3u : 4u});  // attr vector type
    emit(w, kOpTypePointer, {6, kStorageOutput, 4});               // ptr Output vec4 float
    emit(w, kOpTypePointer, {7, kStorageOutput, 5});               // ptr Output attr
    emit(w, kOpTypePointer, {8, kStorageOutput, 2});               // ptr Output float (PointSize/Clip)
    emit(w, kOpTypePointer, {9, kStorageInput, 4});                // ptr Input vec4 float

    if (with_pos) emit(w, kOpVariable, {6, 10, kStorageOutput});
    if (with_point_size) emit(w, kOpVariable, {8, 11, kStorageOutput});
    if (extra_builtin != 0xffffffffu) emit(w, kOpVariable, {8, 12, kStorageOutput});
    emit(w, kOpVariable, {9, 15, kStorageInput});
    next_var = 20;
    for (std::size_t i = 0; i < locs.size(); ++i) {
        emit(w, kOpVariable, {7, next_var++, kStorageOutput});
    }

    emit(w, kOpFunction, {1, 50, 0, 1});
    emit(w, kOpLabel, {51});
    for (std::size_t i = 0; i < body_nops; ++i) emit(w, kOpNop, {});
    emit(w, kOpReturn, {});
    emit(w, kOpFunctionEnd, {});
    return w;
}

// Structural SPIR-V verification of the generated geometry module.
bool verify_geometry_spirv(const Words& gs, const std::vector<std::uint32_t>& expected_locs) {
    if (gs.size() < 5 || gs[0] != 0x07230203u) return false;
    const std::uint32_t bound = gs[3];
    if (bound < 10) return false;

    bool saw_geom_cap = false, saw_mem_model = false, saw_entry = false;
    bool saw_triangles = false, saw_invocations = false, saw_strip = false, saw_out_verts = false;
    std::uint32_t emit_vertices = 0, end_primitives = 0;
    std::uint32_t pos_decos = 0;
    std::vector<std::uint32_t> loc_decos;
    std::size_t io_var_count = 0;

    for (std::size_t i = 5; i < gs.size();) {
        const std::uint32_t n = gs[i] >> 16, op = gs[i] & 0xffffu;
        if (!n || n > gs.size() - i) return false;
        const std::uint32_t* w = &gs[i];
        if (op == kOpCapability && n == 2 && w[1] == 2) saw_geom_cap = true;
        else if (op == kOpMemoryModel && n == 3 && w[1] == 0 && w[2] == 1) saw_mem_model = true;
        else if (op == kOpEntryPoint && n >= 5 && w[1] == 3) {
            saw_entry = true;
            io_var_count = n - 5;  // Geometry, main_fn, "main", 0, then io_vars...
            for (std::size_t k = 5; k < n; ++k) {
                if (w[k] == 0 || w[k] >= bound) return false;
            }
        } else if (op == kOpExecutionMode && n >= 3) {
            if (w[2] == 22 && n == 3) saw_triangles = true;
            else if (w[2] == 0 && n == 4 && w[3] == 1) saw_invocations = true;
            else if (w[2] == 29 && n == 3) saw_strip = true;
            else if (w[2] == 26 && n == 4 && w[3] == 6) saw_out_verts = true;
        } else if (op == kOpDecorate && n == 4) {
            if (w[1] == 0 || w[1] >= bound) return false;
            if (w[2] == kDecoBuiltIn && w[3] == kBuiltInPosition) ++pos_decos;
            if (w[2] == kDecoLocation) loc_decos.push_back(w[3]);
        } else if (op == kOpEmitVertex && n == 1) {
            ++emit_vertices;
        } else if (op == kOpEndPrimitive && n == 1) {
            ++end_primitives;
        }
        i += n;
    }

    if (!saw_geom_cap || !saw_mem_model || !saw_entry) return false;
    if (!saw_triangles || !saw_invocations || !saw_strip || !saw_out_verts) return false;
    if (emit_vertices != 6 || end_primitives != 2) return false;
    if (pos_decos != 2) return false;  // input Position[3] + output Position
    if (io_var_count != (expected_locs.size() + 1) * 2) return false;
    if (loc_decos.size() != expected_locs.size() * 2) return false;
    for (std::size_t k = 0; k < expected_locs.size(); ++k) {
        if (loc_decos[2 * k] != expected_locs[k] || loc_decos[2 * k + 1] != expected_locs[k]) return false;
    }
    return true;
}

}  // namespace

int main() {
    std::string why;

    // 1. Basic Position + unsorted/duplicate Locations + PointSize allowed -> sorted deduplicated geometry stage.
    {
        const Words vs = make_vs(true, true, {3, 0, 2, 0});
        const Words gs = leonida::rect_list_geometry(vs, why);
        CHECK(!gs.empty() && why.empty());
        CHECK(verify_geometry_spirv(gs, {0, 2, 3}));
    }

    // 2. Position-only vertex shader (depth-only rect pass) succeeds with 0 Location params.
    {
        const Words vs = make_vs(true, false, {});
        const Words gs = leonida::rect_list_geometry(vs, why);
        CHECK(!gs.empty() && why.empty());
        CHECK(verify_geometry_spirv(gs, {}));
    }

    // 3. Large function body after OpFunction is skipped without affecting output.
    {
        const Words small_vs = make_vs(true, false, {0, 1}, 0xffffffffu, false, false, 4);
        const Words large_vs = make_vs(true, false, {0, 1}, 0xffffffffu, false, false, 4096);
        const Words gs_small = leonida::rect_list_geometry(small_vs, why);
        CHECK(!gs_small.empty() && why.empty());
        const Words gs_large = leonida::rect_list_geometry(large_vs, why);
        CHECK(gs_large == gs_small && why.empty());
    }

    // 4. Reject non-SPIR-V, short module, zero-length instruction, and truncated preamble instruction.
    {
        CHECK(leonida::rect_list_geometry({}, why).empty() && !why.empty());
        CHECK(leonida::rect_list_geometry({0x07230203u, 0x00010000u, 0, 10, 0}, why).empty() && !why.empty());
        CHECK(leonida::rect_list_geometry({1, 2, 3, 4, 5, 6}, why).empty() && !why.empty());

        Words trunc = make_vs(true, false, {0});
        trunc[5] = (9999u << 16) | kOpDecorate;
        CHECK(leonida::rect_list_geometry(trunc, why).empty() && !why.empty());

        Words zero = make_vs(true, false, {0});
        zero[5] = kOpDecorate;
        CHECK(leonida::rect_list_geometry(zero, why).empty() && !why.empty());
    }

    // 5. Reject missing Position.
    {
        const Words vs = make_vs(false, false, {0, 1});
        CHECK(leonida::rect_list_geometry(vs, why).empty() && !why.empty());
    }

    // 6. Reject unsupported builtins (ClipDistance, Layer).
    {
        const Words vs_clip = make_vs(true, false, {0}, kBuiltInClipDistance);
        CHECK(leonida::rect_list_geometry(vs_clip, why).empty() && !why.empty());
        const Words vs_layer = make_vs(true, false, {0}, kBuiltInLayer);
        CHECK(leonida::rect_list_geometry(vs_layer, why).empty() && !why.empty());
    }

    // 7. Reject non-vec4 float Location outputs (vec3 float, uvec4, or missing type declaration).
    {
        const Words vs_vec3 = make_vs(true, false, {0}, 0xffffffffu, true, false);
        CHECK(leonida::rect_list_geometry(vs_vec3, why).empty() && !why.empty());
        const Words vs_uvec4 = make_vs(true, false, {0}, 0xffffffffu, false, true);
        CHECK(leonida::rect_list_geometry(vs_uvec4, why).empty() && !why.empty());

        // Missing pointer type declaration (dangling type id 999) refuses cleanly without silent map insertion.
        Words vs_dangling{0x07230203u, 0x00010000u, 0, 100, 0};
        emit(vs_dangling, kOpDecorate, {10, kDecoBuiltIn, kBuiltInPosition});
        emit(vs_dangling, kOpDecorate, {20, kDecoLocation, 0});
        emit(vs_dangling, kOpTypeFloat, {2, 32});
        emit(vs_dangling, kOpTypeVector, {4, 2, 4});
        emit(vs_dangling, kOpTypePointer, {6, kStorageOutput, 4});
        emit(vs_dangling, kOpVariable, {6, 10, kStorageOutput});
        emit(vs_dangling, kOpVariable, {999, 20, kStorageOutput});
        emit(vs_dangling, kOpFunction, {1, 50, 0, 1});
        CHECK(leonida::rect_list_geometry(vs_dangling, why).empty() && !why.empty());
    }

    // 8. Constants sanity check.
    CHECK(leonida::topology_rect_list == 17 && leonida::topology_quad_list == 19);
    CHECK(leonida::topology_type_rect_list == 19);
    CHECK((leonida::quad_triangle_indices == std::array<std::uint32_t, 6>{0, 1, 2, 0, 2, 3}));

    std::printf("rect_list_geometry_test: all checks passed\n");
    return 0;
}
