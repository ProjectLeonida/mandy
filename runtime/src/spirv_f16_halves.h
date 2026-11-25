#pragma once

// Per-pipeline `--f16-interp-fallback` half placement and SPIR-V specialization (SHADER-378).
//
// Under `--f16-interp-fallback` (MoltenVK / devices without PerVertexKHR), a pixel shader reads decoded f16 halves
// as ordinary smooth vec4 inputs recorded in `index.json` under `f16_halves: {param: [lo_loc, hi_loc]}`. When a
// producer (VS) exports many params (up to 16) or a PS reads high attributes alongside f16 attributes, no fixed
// static location map fits all pairs within MoltenVK's 30 locations (0..29, 124 output components counting
// gl_Position). At graphics PSO creation, once both VS and PS are known, `link_f16_halves`:
//   1. Fast-paths unchanged when every PS `f16_halves` entry is already exported at the exact same locations by VS.
//   2. Chooses mutually non-colliding `[lo_loc, hi_loc]` slots in `0..top_location` (keeping PS locations unchanged
//      whenever they do not collide with any raw VS output; otherwise relocating PS `OpDecorate Location` in place).
//   3. Specializes the VS SPIR-V binary for the pair: reuses matching static `f16_halves` outputs (updating their
//      `Location` decorations), demotes unneeded static `f16_halves` outputs to `Private` (freeing their locations),
//      and synthesizes `UnpackHalf2x16` half exports right before every `OpReturn` in the Vertex entry function for
//      any param the VS did not emit statically.
//
// Pure C++20 stdlib, no Vulkan or Windows headers required.

#include "shader_library.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace leonida {

namespace spirv_detail {

constexpr std::uint32_t kSpirvMagic = 0x07230203u;
constexpr std::uint32_t kSpirvVersion14 = 0x00010400u;

constexpr std::uint16_t kOpSourceContinued = 2;
constexpr std::uint16_t kOpSource = 3;
constexpr std::uint16_t kOpSourceExtension = 4;
constexpr std::uint16_t kOpName = 5;
constexpr std::uint16_t kOpMemberName = 6;
constexpr std::uint16_t kOpString = 7;
constexpr std::uint16_t kOpExtension = 10;
constexpr std::uint16_t kOpExtInstImport = 11;
constexpr std::uint16_t kOpExtInst = 12;
constexpr std::uint16_t kOpMemoryModel = 14;
constexpr std::uint16_t kOpEntryPoint = 15;
constexpr std::uint16_t kOpExecutionMode = 16;
constexpr std::uint16_t kOpCapability = 17;
constexpr std::uint16_t kOpTypeInt = 21;
constexpr std::uint16_t kOpTypeFloat = 22;
constexpr std::uint16_t kOpTypeVector = 23;
constexpr std::uint16_t kOpTypePointer = 32;
constexpr std::uint16_t kOpConstantNull = 46;
constexpr std::uint16_t kOpFunction = 54;
constexpr std::uint16_t kOpFunctionEnd = 56;
constexpr std::uint16_t kOpVariable = 59;
constexpr std::uint16_t kOpLoad = 61;
constexpr std::uint16_t kOpStore = 62;
constexpr std::uint16_t kOpAccessChain = 65;
constexpr std::uint16_t kOpInBoundsAccessChain = 66;
constexpr std::uint16_t kOpDecorate = 71;
constexpr std::uint16_t kOpMemberDecorate = 72;
constexpr std::uint16_t kOpDecorationGroup = 73;
constexpr std::uint16_t kOpGroupDecorate = 74;
constexpr std::uint16_t kOpGroupMemberDecorate = 75;
constexpr std::uint16_t kOpCompositeConstruct = 80;
constexpr std::uint16_t kOpCompositeExtract = 81;
constexpr std::uint16_t kOpBitcast = 124;
constexpr std::uint16_t kOpReturn = 253;
constexpr std::uint16_t kOpModuleProcessed = 330;
constexpr std::uint16_t kOpExecutionModeId = 331;
constexpr std::uint16_t kOpDecorateId = 332;
constexpr std::uint16_t kOpDecorateString = 5632;
constexpr std::uint16_t kOpMemberDecorateString = 5633;

constexpr std::uint32_t kExecModelVertex = 0;
constexpr std::uint32_t kExecModelFragment = 4;

constexpr std::uint32_t kStorageInput = 1;
constexpr std::uint32_t kStorageOutput = 3;
constexpr std::uint32_t kStoragePrivate = 6;

constexpr std::uint32_t kDecorationLocation = 30;
constexpr std::uint32_t kGlslUnpackHalf2x16 = 62;

inline constexpr bool is_interface_only_decoration(std::uint32_t dec) {
    switch (dec) {
    case 13:  // NoPerspective
    case 14:  // Flat
    case 15:  // Patch
    case 16:  // Centroid
    case 17:  // Sample
    case 18:  // Invariant
    case kDecorationLocation:
    case 31:  // Component
        return true;
    default:
        return false;
    }
}

struct Inst {
    std::uint16_t opcode = 0;
    std::vector<std::uint32_t> words;
};

inline Inst make_inst(std::uint16_t opcode, std::initializer_list<std::uint32_t> operands) {
    Inst inst;
    inst.opcode = opcode;
    inst.words.reserve(1 + operands.size());
    inst.words.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
    for (std::uint32_t w : operands) inst.words.push_back(w);
    return inst;
}

inline Inst make_inst_vec(std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
    Inst inst;
    inst.opcode = opcode;
    inst.words.reserve(1 + operands.size());
    inst.words.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
    inst.words.insert(inst.words.end(), operands.begin(), operands.end());
    return inst;
}

struct Module {
    std::array<std::uint32_t, 5> header{};
    std::vector<Inst> insts;

    bool parse(std::span<const std::uint32_t> words) {
        if (words.size() < 5 || words[0] != kSpirvMagic || words[3] == 0) return false;
        for (std::size_t i = 0; i < 5; ++i) header[i] = words[i];
        insts.clear();
        std::size_t pos = 5;
        while (pos < words.size()) {
            const std::uint32_t head = words[pos];
            const std::uint32_t wc = head >> 16;
            const std::uint16_t op = std::uint16_t(head & 0xffffu);
            if (wc == 0 || wc > words.size() - pos) return false;
            Inst inst;
            inst.opcode = op;
            inst.words.assign(words.begin() + std::ptrdiff_t(pos), words.begin() + std::ptrdiff_t(pos + wc));
            insts.push_back(std::move(inst));
            pos += wc;
        }
        return true;
    }

    std::vector<std::uint32_t> serialize() const {
        std::size_t total = 5;
        for (const Inst& inst : insts) total += inst.words.size();
        std::vector<std::uint32_t> out;
        out.reserve(total);
        out.insert(out.end(), header.begin(), header.end());
        for (const Inst& inst : insts) out.insert(out.end(), inst.words.begin(), inst.words.end());
        return out;
    }

    std::uint32_t alloc_id() { return header[3]++; }
};

inline bool is_pre_type_opcode(std::uint16_t op) {
    switch (op) {
    case kOpCapability:
    case kOpExtension:
    case kOpExtInstImport:
    case kOpMemoryModel:
    case kOpEntryPoint:
    case kOpExecutionMode:
    case kOpExecutionModeId:
    case kOpString:
    case kOpSourceContinued:
    case kOpSource:
    case kOpSourceExtension:
    case kOpName:
    case kOpMemberName:
    case kOpModuleProcessed:
    case kOpDecorate:
    case kOpMemberDecorate:
    case kOpDecorationGroup:
    case kOpGroupDecorate:
    case kOpGroupMemberDecorate:
    case kOpDecorateId:
    case kOpDecorateString:
    case kOpMemberDecorateString:
        return true;
    default:
        return false;
    }
}

inline std::string decode_literal_string(std::span<const std::uint32_t> words) {
    std::string s;
    for (std::uint32_t w : words) {
        for (int b = 0; b < 4; ++b) {
            const char c = char((w >> (8 * b)) & 0xffu);
            if (c == '\0') return s;
            s.push_back(c);
        }
    }
    return s;
}

inline std::size_t entry_point_interface_start(std::span<const std::uint32_t> words) {
    // OpEntryPoint <ExecModel> <EntryPointId> <Name...> <InterfaceIds...>
    for (std::size_t i = 3; i < words.size(); ++i) {
        const std::uint32_t w = words[i];
        if ((w & 0x000000ffu) == 0 || (w & 0x0000ff00u) == 0 || (w & 0x00ff0000u) == 0 || (w & 0xff000000u) == 0)
            return i + 1;
    }
    return words.size();
}

inline std::size_t entry_point_interface_start(const Inst& ep) {
    return entry_point_interface_start(std::span<const std::uint32_t>(ep.words));
}

inline void emit_words(std::vector<std::uint32_t>& out, std::uint16_t opcode,
                       std::initializer_list<std::uint32_t> operands) {
    out.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
    out.insert(out.end(), operands.begin(), operands.end());
}

// Validates SPIR-V instruction framing in one flat pass and collects global OpVariables of `storage_class` that carry
// `OpDecorate <id> Location <loc>` (first variable found at that location). Also reports the offset of the first
// OpFunction (or `words.size()` if none) and whether any OpDecorate appeared after it.
inline bool validate_and_collect_located_vars(std::span<const std::uint32_t> words, std::uint32_t storage_class,
                                              std::map<std::uint32_t, std::uint32_t>& by_loc,
                                              std::size_t* first_fn_pos = nullptr,
                                              bool* has_post_fn_decorate = nullptr) {
    if (words.size() < 5 || words[0] != kSpirvMagic || words[3] == 0) return false;
    by_loc.clear();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> loc_decos;
    std::vector<std::uint32_t> matching_vars;
    std::size_t fn_pos = words.size();
    bool post_fn_dec = false;
    std::size_t pos = 5;
    while (pos < words.size()) {
        const std::uint32_t head = words[pos];
        const std::uint32_t wc = head >> 16;
        if (wc == 0 || wc > words.size() - pos) return false;
        const std::uint16_t op = std::uint16_t(head & 0xffffu);
        if (fn_pos == words.size()) {
            if (op == kOpFunction) {
                fn_pos = pos;
            } else if (op == kOpDecorate && wc >= 4 && words[pos + 2] == kDecorationLocation) {
                loc_decos.emplace_back(words[pos + 1], words[pos + 3]);
            } else if (op == kOpVariable && wc >= 4 && words[pos + 3] == storage_class) {
                matching_vars.push_back(words[pos + 2]);
            }
        } else if (op == kOpDecorate) {
            post_fn_dec = true;
            if (wc >= 4 && words[pos + 2] == kDecorationLocation)
                loc_decos.emplace_back(words[pos + 1], words[pos + 3]);
        }
        pos += wc;
    }
    if (first_fn_pos) *first_fn_pos = fn_pos;
    if (has_post_fn_decorate) *has_post_fn_decorate = post_fn_dec;
    std::sort(matching_vars.begin(), matching_vars.end());
    for (const auto& [target, loc] : loc_decos) {
        if (std::binary_search(matching_vars.begin(), matching_vars.end(), target))
            by_loc.emplace(loc, target);
    }
    return true;
}

// Collects global OpVariables of `storage_class` that carry `OpDecorate <id> Location <loc>`.
// Returns map: location -> variable_id (first variable found at that location).
inline std::map<std::uint32_t, std::uint32_t> collect_located_vars(const Module& mod, std::uint32_t storage_class) {
    std::map<std::uint32_t, std::uint32_t> var_storage;
    for (const Inst& inst : mod.insts) {
        if (inst.opcode == kOpFunction) break;
        if (inst.opcode == kOpVariable && inst.words.size() >= 4) var_storage[inst.words[2]] = inst.words[3];
    }
    std::map<std::uint32_t, std::uint32_t> by_loc;
    for (const Inst& inst : mod.insts) {
        if (inst.opcode == kOpDecorate && inst.words.size() >= 4 && inst.words[2] == kDecorationLocation) {
            const std::uint32_t target = inst.words[1];
            const auto it = var_storage.find(target);
            if (it != var_storage.end() && it->second == storage_class) by_loc.emplace(inst.words[3], target);
        }
    }
    return by_loc;
}

}  // namespace spirv_detail

// Highest usable vec4 Location (capped at 29, MoltenVK's 124-component limit) for a device's stage interface limits.
// Reserves 4 components for BuiltIn Position (so 128 -> 29, 124 -> 29, 64 -> 14).
inline constexpr std::uint32_t f16_half_top_location(std::uint32_t max_vertex_output_components,
                                                     std::uint32_t max_fragment_input_components) {
    const std::uint32_t limit = std::min(max_vertex_output_components, max_fragment_input_components);
    if (limit < 8) return 0;
    return std::min<std::uint32_t>(29u, limit / 4u - 2u);
}

// Specializes/patches `vs_words` and (if needed) `ps_words` so that every entry in `ps_entry.f16_halves` is
// exported by the VS at the exact smooth location the PS reads, within `0..top_location` (default 29, MoltenVK's
// 124-component vertex output limit counting gl_Position).
//
// Returns true on success (including the no-op fast path when no patching is needed). On failure returns false and
// writes a concise reason to `*error` if non-null; the caller's word vectors are only replaced on success.
inline bool link_f16_halves(std::vector<std::uint32_t>& vs_words, const ShaderEntry& vs_entry,
                            std::vector<std::uint32_t>& ps_words, const ShaderEntry& ps_entry,
                            std::string* error = nullptr, std::uint32_t top_location = 29) {
    const auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };
    if (ps_entry.f16_halves.empty() && vs_entry.f16_halves.empty()) return true;

    using namespace spirv_detail;
    std::map<std::uint32_t, std::uint32_t> vs_outputs, ps_inputs;
    std::size_t vs_first_fn = vs_words.size();
    bool vs_has_post_fn_decorate = false;
    if (!validate_and_collect_located_vars(vs_words, kStorageOutput, vs_outputs, &vs_first_fn,
                                           &vs_has_post_fn_decorate))
        return fail("f16_halves: malformed VS SPIR-V");
    std::size_t ps_first_fn = ps_words.size();
    bool ps_has_post_fn_decorate = false;
    if (!validate_and_collect_located_vars(ps_words, kStorageInput, ps_inputs, &ps_first_fn,
                                           &ps_has_post_fn_decorate))
        return fail("f16_halves: malformed PS SPIR-V");

    std::set<std::uint32_t> vs_static_half_locs;
    for (const auto& [param, pair] : vs_entry.f16_halves) {
        vs_static_half_locs.insert(pair[0]);
        vs_static_half_locs.insert(pair[1]);
    }
    std::set<std::uint32_t> ps_static_half_locs;
    for (const auto& [param, pair] : ps_entry.f16_halves) {
        ps_static_half_locs.insert(pair[0]);
        ps_static_half_locs.insert(pair[1]);
    }

    // Raw outputs of VS (all located Output variables excluding VS's own static f16_halves).
    std::map<std::uint32_t, std::uint32_t> vs_raw_outputs;
    std::set<std::uint32_t> vs_raw_locs;
    for (const auto& [loc, var_id] : vs_outputs) {
        if (!vs_static_half_locs.count(loc)) {
            vs_raw_outputs[loc] = var_id;
            vs_raw_locs.insert(loc);
        }
    }
    std::set<std::uint32_t> ps_raw_locs;
    for (const auto& [loc, var_id] : ps_inputs) {
        if (!ps_static_half_locs.count(loc)) ps_raw_locs.insert(loc);
    }

    // Fast path: every PS half pair is already exported at the exact same locations by VS, all static halves fit
    // within 0..top_location, and no unused VS static half collides with a raw PS input.
    bool already_linked = true;
    for (const auto& [param, ps_pair] : ps_entry.f16_halves) {
        const auto it = vs_entry.f16_halves.find(param);
        if (it == vs_entry.f16_halves.end() || it->second != ps_pair || ps_pair[0] > top_location ||
            ps_pair[1] > top_location) {
            already_linked = false;
            break;
        }
    }
    if (already_linked) {
        for (const auto& [param, vs_pair] : vs_entry.f16_halves) {
            const bool used_by_ps = ps_entry.f16_halves.count(param) != 0;
            for (std::uint32_t loc : vs_pair) {
                if (loc > top_location || (!used_by_ps && ps_raw_locs.count(loc))) {
                    already_linked = false;
                    break;
                }
            }
            if (!already_linked) break;
        }
    }
    if (already_linked) return true;

    // Choose paired [lo, hi] locations for every param in ps_entry.f16_halves.
    // Keep PS's existing locations unchanged whenever none collides with VS/PS raw outputs/inputs or exceeds top_location.
    bool keep_ps_locs = true;
    std::set<std::uint32_t> seen_ps_halves;
    for (const auto& [param, pair] : ps_entry.f16_halves) {
        for (std::uint32_t loc : pair) {
            if (loc > top_location || vs_raw_locs.count(loc) || ps_raw_locs.count(loc) ||
                !seen_ps_halves.insert(loc).second) {
                keep_ps_locs = false;
                break;
            }
        }
        if (!keep_ps_locs) break;
    }

    std::map<std::uint32_t, std::array<std::uint32_t, 2>> pair_halves;
    if (keep_ps_locs) {
        pair_halves = ps_entry.f16_halves;
    } else {
        std::set<std::uint32_t> reserved = vs_raw_locs;
        reserved.insert(ps_raw_locs.begin(), ps_raw_locs.end());
        std::vector<std::uint32_t> free_locs;
        for (std::int64_t loc = top_location; loc >= 0; --loc)
            if (!reserved.count(std::uint32_t(loc))) free_locs.push_back(std::uint32_t(loc));
        const std::size_t need = 2 * ps_entry.f16_halves.size();
        if (free_locs.size() < need)
            return fail("f16_halves: need " + std::to_string(need) + " free locations in 0.." +
                        std::to_string(top_location) + ", only " + std::to_string(free_locs.size()) + " available");
        std::size_t k = 0;
        for (const auto& [param, old_pair] : ps_entry.f16_halves) {
            const std::uint32_t hi = free_locs[2 * k];
            const std::uint32_t lo = free_locs[2 * k + 1];
            pair_halves[param] = {lo, hi};
            ++k;
        }
    }

    // Collect in-place PS OpDecorate Location patches (committed only after VS succeeds).
    std::vector<std::pair<std::size_t, std::uint32_t>> ps_patches;
    if (!keep_ps_locs) {
        std::map<std::uint32_t, std::uint32_t> ps_var_new_loc;
        for (const auto& [param, old_pair] : ps_entry.f16_halves) {
            const auto& new_pair = pair_halves[param];
            for (std::size_t h = 0; h < 2; ++h) {
                const auto it = ps_inputs.find(old_pair[h]);
                if (it != ps_inputs.end()) ps_var_new_loc[it->second] = new_pair[h];
            }
        }
        const std::size_t ps_scan_end = ps_has_post_fn_decorate ? ps_words.size() : ps_first_fn;
        for (std::size_t pos = 5; pos < ps_scan_end;) {
            const std::uint32_t head = ps_words[pos];
            const std::uint32_t wc = head >> 16;
            const std::uint16_t op = std::uint16_t(head & 0xffffu);
            if (op == kOpDecorate && wc >= 4 && ps_words[pos + 2] == kDecorationLocation) {
                const auto it = ps_var_new_loc.find(ps_words[pos + 1]);
                if (it != ps_var_new_loc.end() && ps_words[pos + 3] != it->second)
                    ps_patches.emplace_back(pos + 3, it->second);
            }
            pos += wc;
        }
    }

    // Inspect VS preamble (types, globals, entry point) directly over the flat word stream.
    std::uint32_t float_ty = 0, uint_ty = 0, v2float_ty = 0, v4float_ty = 0;
    std::uint32_t ptr_out_v4_ty = 0, ptr_priv_v4_ty = 0, null_v4_id = 0, glsl_id = 0;
    std::map<std::uint32_t, std::uint32_t> ptr_pointee;     // ptr_type_id -> pointee_type_id
    std::map<std::uint32_t, std::uint32_t> var_ptr_type;    // var_id -> ptr_type_id
    std::uint32_t vs_entry_fn = 0;
    std::size_t vs_ep_pos = vs_words.size();

    for (std::size_t pos = 5; pos < vs_first_fn;) {
        const std::uint32_t head = vs_words[pos];
        const std::uint32_t wc = head >> 16;
        const std::uint16_t op = std::uint16_t(head & 0xffffu);
        const std::uint32_t* w = vs_words.data() + pos;
        if (op == kOpExtInstImport && wc >= 3) {
            if (decode_literal_string(std::span<const std::uint32_t>(w + 2, wc - 2)) == "GLSL.std.450")
                glsl_id = w[1];
        } else if (op == kOpEntryPoint && wc >= 4 && w[1] == kExecModelVertex) {
            if (vs_ep_pos == vs_words.size()) {
                vs_ep_pos = pos;
                vs_entry_fn = w[2];
            }
        } else if (op == kOpTypeFloat && wc == 3 && w[2] == 32) {
            float_ty = w[1];
        } else if (op == kOpTypeInt && wc == 4 && w[2] == 32 && w[3] == 0) {
            uint_ty = w[1];
        } else if (op == kOpTypeVector && wc == 4) {
            if (float_ty && w[2] == float_ty && w[3] == 2) v2float_ty = w[1];
            if (float_ty && w[2] == float_ty && w[3] == 4) v4float_ty = w[1];
        } else if (op == kOpTypePointer && wc == 4) {
            ptr_pointee[w[1]] = w[3];
            if (v4float_ty && w[3] == v4float_ty) {
                if (w[2] == kStorageOutput) ptr_out_v4_ty = w[1];
                if (w[2] == kStoragePrivate) ptr_priv_v4_ty = w[1];
            }
        } else if (op == kOpConstantNull && wc == 3 && v4float_ty && w[1] == v4float_ty) {
            null_v4_id = w[2];
        } else if (op == kOpVariable && wc >= 4) {
            var_ptr_type[w[2]] = w[1];
        }
        pos += wc;
    }
    if (vs_ep_pos == vs_words.size() || !vs_entry_fn) return fail("f16_halves: VS has no Vertex entry point");

    const auto is_v4float_var = [&](std::uint32_t vid) {
        if (!v4float_ty) return false;
        const auto pty = var_ptr_type.find(vid);
        if (pty == var_ptr_type.end()) return false;
        const auto pt = ptr_pointee.find(pty->second);
        return pt != ptr_pointee.end() && pt->second == v4float_ty;
    };

    // Decide per VS static f16_halves entry: reuse (update Location) or demote (Output -> Private, drop Location).
    std::map<std::uint32_t, std::uint32_t> vs_var_new_loc;
    std::set<std::uint32_t> demote_vars;
    std::set<std::uint32_t> satisfied_params;
    for (const auto& [param, old_pair] : vs_entry.f16_halves) {
        const auto lo_it = vs_outputs.find(old_pair[0]);
        const auto hi_it = vs_outputs.find(old_pair[1]);
        const auto want_it = pair_halves.find(param);
        if (want_it != pair_halves.end() && lo_it != vs_outputs.end() && hi_it != vs_outputs.end()) {
            vs_var_new_loc[lo_it->second] = want_it->second[0];
            vs_var_new_loc[hi_it->second] = want_it->second[1];
            satisfied_params.insert(param);
        } else {
            for (const auto& it : {lo_it, hi_it}) {
                if (it == vs_outputs.end()) continue;
                const std::uint32_t vid = it->second;
                if (!is_v4float_var(vid))
                    return fail("f16_halves: cannot demote non-vec4 or access-chained VS static half output");
                demote_vars.insert(vid);
            }
        }
    }

    // Only scan function bodies for OpAccessChain when there is actually a static half variable to demote.
    if (!demote_vars.empty()) {
        std::vector<std::uint32_t> demote_sorted(demote_vars.begin(), demote_vars.end());
        for (std::size_t pos = vs_first_fn; pos < vs_words.size();) {
            const std::uint32_t head = vs_words[pos];
            const std::uint32_t wc = head >> 16;
            const std::uint16_t op = std::uint16_t(head & 0xffffu);
            if ((op == kOpAccessChain || op == kOpInBoundsAccessChain) && wc >= 4 &&
                std::binary_search(demote_sorted.begin(), demote_sorted.end(), vs_words[pos + 3])) {
                return fail("f16_halves: cannot demote non-vec4 or access-chained VS static half output");
            }
            pos += wc;
        }
    }

    // Params that must be synthesized in the VS epilogue.
    struct SynthParam {
        std::uint32_t param = 0;
        std::uint32_t src_var = 0;  // 0 if VS does not export this param (emit null vec4)
        std::uint32_t lo_loc = 0;
        std::uint32_t hi_loc = 0;
        std::uint32_t lo_var = 0;
        std::uint32_t hi_var = 0;
    };
    std::vector<SynthParam> synth;
    bool need_null_v4 = false;
    for (const auto& [param, target_pair] : pair_halves) {
        if (satisfied_params.count(param)) continue;
        SynthParam sp;
        sp.param = param;
        sp.lo_loc = target_pair[0];
        sp.hi_loc = target_pair[1];
        const auto raw_it = vs_raw_outputs.find(param);
        if (raw_it != vs_raw_outputs.end()) {
            if (!is_v4float_var(raw_it->second))
                return fail("f16_halves: VS param " + std::to_string(param) + " output is not vec4 float");
            sp.src_var = raw_it->second;
        } else {
            need_null_v4 = true;
        }
        synth.push_back(sp);
    }

    std::uint32_t next_id = vs_words[3];
    bool id_overflow = false;
    const auto alloc_id = [&]() -> std::uint32_t {
        if (next_id == 0xffffffffu) {
            id_overflow = true;
            return 0;
        }
        return next_id++;
    };

    // Create any missing types / GLSL.std.450 import needed for demotion or synthesis.
    std::vector<std::uint32_t> new_types;
    const bool need_glsl = std::any_of(synth.begin(), synth.end(), [](const SynthParam& s) { return s.src_var != 0; });
    bool insert_glsl = false;
    if (need_glsl && !glsl_id) {
        glsl_id = alloc_id();
        insert_glsl = true;
    }
    if ((!synth.empty() || !demote_vars.empty()) && !float_ty) {
        float_ty = alloc_id();
        emit_words(new_types, kOpTypeFloat, {float_ty, 32});
    }
    if (need_glsl && !uint_ty) {
        uint_ty = alloc_id();
        emit_words(new_types, kOpTypeInt, {uint_ty, 32, 0});
    }
    if (need_glsl && !v2float_ty) {
        v2float_ty = alloc_id();
        emit_words(new_types, kOpTypeVector, {v2float_ty, float_ty, 2});
    }
    if ((!synth.empty() || !demote_vars.empty()) && !v4float_ty) {
        v4float_ty = alloc_id();
        emit_words(new_types, kOpTypeVector, {v4float_ty, float_ty, 4});
    }
    if (!synth.empty() && !ptr_out_v4_ty) {
        ptr_out_v4_ty = alloc_id();
        emit_words(new_types, kOpTypePointer, {ptr_out_v4_ty, kStorageOutput, v4float_ty});
    }
    if (!demote_vars.empty() && !ptr_priv_v4_ty) {
        ptr_priv_v4_ty = alloc_id();
        emit_words(new_types, kOpTypePointer, {ptr_priv_v4_ty, kStoragePrivate, v4float_ty});
    }
    if (need_null_v4 && !null_v4_id) {
        null_v4_id = alloc_id();
        emit_words(new_types, kOpConstantNull, {v4float_ty, null_v4_id});
    }

    // Allocate Output OpVariables for synthesized params.
    std::vector<std::uint32_t> new_decorates;
    std::vector<std::uint32_t> new_vars;
    new_decorates.reserve(synth.size() * 8);
    new_vars.reserve(synth.size() * 8);
    for (SynthParam& sp : synth) {
        sp.lo_var = alloc_id();
        sp.hi_var = alloc_id();
        emit_words(new_decorates, kOpDecorate, {sp.lo_var, kDecorationLocation, sp.lo_loc});
        emit_words(new_decorates, kOpDecorate, {sp.hi_var, kDecorationLocation, sp.hi_loc});
        emit_words(new_vars, kOpVariable, {ptr_out_v4_ty, sp.lo_var, kStorageOutput});
        emit_words(new_vars, kOpVariable, {ptr_out_v4_ty, sp.hi_var, kStorageOutput});
    }

    // Rebuild VS directly into one flat word vector in exact SPIR-V logical layout order.
    // Demoted OpVariables are emitted right after new_types (before the first OpFunction) so ptr_priv_v4_ty is
    // always defined before every OpVariable that uses it.
    const bool drop_private_from_ep = vs_words[1] < kSpirvVersion14;
    std::vector<std::uint32_t> demoted_var_words;
    std::vector<std::uint32_t> out_words;
    out_words.reserve(vs_words.size() + new_types.size() + new_decorates.size() + new_vars.size() +
                      synth.size() * 144 + 16);
    out_words.insert(out_words.end(), vs_words.begin(), vs_words.begin() + 5);

    bool glsl_inserted = !insert_glsl;
    bool decorates_inserted = new_decorates.empty();
    bool globals_inserted = new_types.empty() && new_vars.empty() && demote_vars.empty();
    bool in_entry_fn = false;
    bool found_entry_return = false;

    for (std::size_t pos = 5; pos < vs_words.size();) {
        const std::uint32_t head = vs_words[pos];
        const std::uint32_t wc = head >> 16;
        const std::uint16_t op = std::uint16_t(head & 0xffffu);
        const std::uint32_t* w = vs_words.data() + pos;

        // Insert OpExtInstImport "GLSL.std.450" right before OpMemoryModel if the module didn't already have it.
        if (!glsl_inserted && op == kOpMemoryModel) {
            // "GLSL.std.450\0" -> 4 little-endian words: 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000
            emit_words(out_words, kOpExtInstImport, {glsl_id, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u});
            glsl_inserted = true;
        }

        // Insert new OpDecorate instructions at the end of the Annotations section (before the first type/global).
        if (!decorates_inserted && !is_pre_type_opcode(op)) {
            out_words.insert(out_words.end(), new_decorates.begin(), new_decorates.end());
            decorates_inserted = true;
        }

        // Insert new Types / Constants, demoted Private OpVariables, and new Output OpVariables before OpFunction.
        if (!globals_inserted && op == kOpFunction) {
            out_words.insert(out_words.end(), new_types.begin(), new_types.end());
            out_words.insert(out_words.end(), demoted_var_words.begin(), demoted_var_words.end());
            out_words.insert(out_words.end(), new_vars.begin(), new_vars.end());
            globals_inserted = true;
        }

        // When no epilogue synthesis is needed and all header/preamble edits are done, bulk-copy the function section.
        if (op == kOpFunction && synth.empty() && !vs_has_post_fn_decorate && glsl_inserted && decorates_inserted &&
            globals_inserted) {
            out_words.insert(out_words.end(), vs_words.begin() + std::ptrdiff_t(pos), vs_words.end());
            break;
        }

        // Update Vertex OpEntryPoint interface list.
        if (pos == vs_ep_pos) {
            const std::span<const std::uint32_t> ep_span(w, wc);
            const std::size_t if_start = entry_point_interface_start(ep_span);
            const std::size_t head_idx = out_words.size();
            out_words.push_back(0);  // patched below once final word count is known
            out_words.insert(out_words.end(), w + 1, w + if_start);
            for (std::size_t i = if_start; i < wc; ++i) {
                const std::uint32_t id = w[i];
                if (drop_private_from_ep && demote_vars.count(id)) continue;
                out_words.push_back(id);
            }
            for (const SynthParam& sp : synth) {
                out_words.push_back(sp.lo_var);
                out_words.push_back(sp.hi_var);
            }
            const std::size_t ep_wc = out_words.size() - head_idx;
            if (ep_wc > 0xffffu) return fail("f16_halves: Vertex OpEntryPoint exceeds SPIR-V word count limit");
            out_words[head_idx] = (std::uint32_t(ep_wc) << 16) | kOpEntryPoint;
            pos += wc;
            continue;
        }

        // Update Location or strip interface-only decorations on static VS f16_halves outputs.
        if (op == kOpDecorate && wc >= 3) {
            const std::uint32_t target = w[1];
            const std::uint32_t dec = w[2];
            if (demote_vars.count(target) && is_interface_only_decoration(dec)) {
                pos += wc;
                continue;
            }
            const std::size_t base = out_words.size();
            out_words.insert(out_words.end(), w, w + wc);
            if (dec == kDecorationLocation && wc >= 4) {
                const auto it = vs_var_new_loc.find(target);
                if (it != vs_var_new_loc.end()) out_words[base + 3] = it->second;
            }
            pos += wc;
            continue;
        }

        // Demote unneeded static VS f16_halves OpVariable from Output to Private (emitted after new_types).
        if (!globals_inserted && op == kOpVariable && wc >= 4 && demote_vars.count(w[2])) {
            const std::size_t base = demoted_var_words.size();
            demoted_var_words.insert(demoted_var_words.end(), w, w + wc);
            demoted_var_words[base + 1] = ptr_priv_v4_ty;
            demoted_var_words[base + 3] = kStoragePrivate;
            pos += wc;
            continue;
        }

        // Track Vertex entry function and inject UnpackHalf2x16 epilogue before every OpReturn inside it.
        if (op == kOpFunction && wc >= 3 && w[2] == vs_entry_fn) {
            in_entry_fn = true;
        } else if (op == kOpFunctionEnd && in_entry_fn) {
            in_entry_fn = false;
        } else if (in_entry_fn && op == kOpReturn && !synth.empty()) {
            found_entry_return = true;
            for (const SynthParam& sp : synth) {
                if (sp.src_var == 0) {
                    emit_words(out_words, kOpStore, {sp.lo_var, null_v4_id});
                    emit_words(out_words, kOpStore, {sp.hi_var, null_v4_id});
                    continue;
                }
                const std::uint32_t raw_id = alloc_id();
                emit_words(out_words, kOpLoad, {v4float_ty, raw_id, sp.src_var});
                std::array<std::uint32_t, 4> lo_c{}, hi_c{};
                for (std::uint32_t c = 0; c < 4; ++c) {
                    const std::uint32_t e_id = alloc_id();
                    const std::uint32_t u_id = alloc_id();
                    const std::uint32_t h2_id = alloc_id();
                    const std::uint32_t a_id = alloc_id();
                    const std::uint32_t b_id = alloc_id();
                    emit_words(out_words, kOpCompositeExtract, {float_ty, e_id, raw_id, c});
                    emit_words(out_words, kOpBitcast, {uint_ty, u_id, e_id});
                    emit_words(out_words, kOpExtInst, {v2float_ty, h2_id, glsl_id, kGlslUnpackHalf2x16, u_id});
                    emit_words(out_words, kOpCompositeExtract, {float_ty, a_id, h2_id, 0});
                    emit_words(out_words, kOpCompositeExtract, {float_ty, b_id, h2_id, 1});
                    lo_c[c] = a_id;
                    hi_c[c] = b_id;
                }
                const std::uint32_t lo_vec = alloc_id();
                const std::uint32_t hi_vec = alloc_id();
                emit_words(out_words, kOpCompositeConstruct, {v4float_ty, lo_vec, lo_c[0], lo_c[1], lo_c[2], lo_c[3]});
                emit_words(out_words, kOpCompositeConstruct, {v4float_ty, hi_vec, hi_c[0], hi_c[1], hi_c[2], hi_c[3]});
                emit_words(out_words, kOpStore, {sp.lo_var, lo_vec});
                emit_words(out_words, kOpStore, {sp.hi_var, hi_vec});
            }
        }

        out_words.insert(out_words.end(), w, w + wc);
        pos += wc;
    }

    if (id_overflow) return fail("f16_halves: SPIR-V ID bound overflow");
    if (!synth.empty() && !found_entry_return)
        return fail("f16_halves: Vertex entry point has no OpReturn to emit decoded halves");
    if (!glsl_inserted || !decorates_inserted || !globals_inserted)
        return fail("f16_halves: unexpected VS SPIR-V layout");

    out_words[3] = next_id;
    vs_words = std::move(out_words);
    for (const auto& [word_idx, new_loc] : ps_patches) ps_words[word_idx] = new_loc;
    return true;
}

}  // namespace leonida
