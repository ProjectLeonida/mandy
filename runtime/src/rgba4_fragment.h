#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace leonida::rgba4 {

using Inst = std::vector<std::uint32_t>;

inline Inst instruction(std::uint32_t op, std::initializer_list<std::uint32_t> args) {
    Inst w;
    w.reserve(args.size() + 1);
    w.push_back((std::uint32_t(args.size() + 1) << 16) | op);
    w.insert(w.end(), args);
    return w;
}

// Quantize only written color outputs at the requested locations. For a wider fallback attachment
// this reproduces four-bit UNORM writes for pipelines whose affected attachments use no blending.
inline bool quantize(Inst& words, unsigned mask) {
    if (!mask) return true;
    if (words.size() < 5 || words[0] != 0x07230203u) return false;

    constexpr std::uint32_t kOpExtInstImport = 11, kOpExtInst = 12, kOpMemoryModel = 14;
    constexpr std::uint32_t kOpTypeFloat = 22, kOpTypeVector = 23, kOpTypePointer = 32;
    constexpr std::uint32_t kOpConstant = 43, kOpConstantComposite = 44;
    constexpr std::uint32_t kOpFunction = 54, kOpVariable = 59, kOpStore = 62, kOpCopyMemory = 63;
    constexpr std::uint32_t kOpAccessChain = 65, kOpInBoundsAccessChain = 66, kOpPtrAccessChain = 83;
    constexpr std::uint32_t kOpDecorate = 71, kOpFMul = 133;
    constexpr std::uint32_t kDecoLocation = 30, kStorageOutput = 3;
    constexpr std::uint32_t kGlslRoundEven = 2, kGlslFClamp = 43;

    struct Kv { std::uint32_t key, value; };
    struct TypeInfo { std::uint32_t id, width, scalar; };

    std::vector<Kv> locations, pointer_type, variables;
    std::vector<TypeInfo> types;
    std::uint32_t import_id = 0, float_type = 0;
    std::size_t mem_model_at = 0;
    std::size_t first_func_at = words.size();

    // Pass 1: flat preamble scan up to the first OpFunction (no per-instruction vector allocations).
    for (std::size_t at = 5; at < words.size();) {
        const std::uint32_t n = words[at] >> 16, op = words[at] & 0xffffu;
        if (!n || n > words.size() - at) return false;
        if (op == kOpFunction) {
            first_func_at = at;
            break;
        }
        const std::uint32_t* w = &words[at];
        if (op == kOpMemoryModel && !mem_model_at) {
            mem_model_at = at;
        } else if (op == kOpExtInstImport && n >= 3) {
            const auto bytes = std::size_t(n - 2) * 4;
            if (bytes >= 13 && std::memcmp(w + 2, "GLSL.std.450", 13) == 0) import_id = w[1];
        } else if (op == kOpDecorate && n == 4 && w[2] == kDecoLocation) {
            locations.push_back({w[1], w[3]});
        } else if (op == kOpTypeFloat && n == 3 && w[2] == 32u) {
            float_type = w[1];
            types.push_back({w[1], 1u, w[1]});
        } else if (op == kOpTypeVector && n == 4) {
            types.push_back({w[1], w[3], w[2]});
        } else if (op == kOpTypePointer && n == 4 && w[2] == kStorageOutput) {
            pointer_type.push_back({w[1], w[3]});
        } else if (op == kOpVariable && n >= 4 && w[3] == kStorageOutput) {
            variables.push_back({w[2], w[1]});
        }
        at += n;
    }

    if (!float_type) return false;

    const auto sort_kv = [](std::vector<Kv>& v) {
        std::stable_sort(v.begin(), v.end(), [](const Kv& a, const Kv& b) { return a.key < b.key; });
    };
    const auto find_kv = [](const std::vector<Kv>& v, std::uint32_t key) -> const Kv* {
        const auto it = std::upper_bound(v.begin(), v.end(), key,
                                         [](std::uint32_t k, const Kv& e) { return k < e.key; });
        return (it != v.begin() && (it - 1)->key == key) ? &*(it - 1) : nullptr;
    };
    sort_kv(locations);
    sort_kv(pointer_type);
    sort_kv(variables);
    std::stable_sort(types.begin(), types.end(), [](const TypeInfo& a, const TypeInfo& b) { return a.id < b.id; });
    const auto find_type = [&](std::uint32_t id) -> const TypeInfo* {
        const auto it = std::upper_bound(types.begin(), types.end(), id,
                                         [](std::uint32_t k, const TypeInfo& e) { return k < e.id; });
        return (it != types.begin() && (it - 1)->id == id) ? &*(it - 1) : nullptr;
    };

    // Outputs is small (<= 8 color targets plus any derived access-chain pointers): a flat list fits in one cache line.
    std::vector<Kv> outputs;
    const auto find_output = [&](std::uint32_t id) -> const Kv* {
        for (std::size_t i = outputs.size(); i-- > 0;)
            if (outputs[i].key == id) return &outputs[i];
        return nullptr;
    };
    for (std::size_t idx = 0; idx < variables.size(); ++idx) {
        if (idx + 1 < variables.size() && variables[idx + 1].key == variables[idx].key) continue;
        const std::uint32_t id = variables[idx].key;
        const std::uint32_t ptr = variables[idx].value;
        const Kv* loc = find_kv(locations, id);
        if (!loc || loc->value >= 8u || !(mask & (1u << loc->value))) continue;
        const Kv* pointee = find_kv(pointer_type, ptr);
        if (pointee) outputs.push_back({id, pointee->value});
    }
    if (outputs.empty()) return true;
    if (first_func_at == words.size()) return false;

    const bool add_import = !import_id;
    if (add_import && !mem_model_at) return false;

    // Pass 2: validate body framing, track derived Output pointers, and record OpStore sites without mutating `words`.
    struct StorePatch {
        std::size_t at;
        std::uint32_t type;
    };
    std::vector<StorePatch> patches;
    for (std::size_t at = first_func_at; at < words.size();) {
        const std::uint32_t n = words[at] >> 16, op = words[at] & 0xffffu;
        if (!n || n > words.size() - at) return false;
        const std::uint32_t* w = &words[at];
        if ((op == kOpAccessChain || op == kOpInBoundsAccessChain || op == kOpPtrAccessChain) && n >= 4) {
            if (find_output(w[3])) {
                if (const Kv* pointee = find_kv(pointer_type, w[1])) outputs.push_back({w[2], pointee->value});
            }
        } else if (op == kOpCopyMemory && n >= 3) {
            if (find_output(w[1])) return false;
        } else if (op == kOpStore && n >= 3) {
            if (const Kv* out_var = find_output(w[1])) {
                const TypeInfo* ti = find_type(out_var->value);
                if (!ti || ti->width < 1u || ti->width > 4u || ti->scalar != float_type) return false;
                patches.push_back({at, out_var->value});
            }
        }
        at += n;
    }
    if (patches.empty()) return false;

    // Build shared scalar constants (emitted once) and per-vector-type composite constants.
    std::uint32_t next = words[3];
    if (add_import) import_id = next++;

    std::vector<std::uint32_t> declarations;
    declarations.reserve(32);
    const auto emit_words = [](std::vector<std::uint32_t>& dst, std::uint32_t op,
                               std::initializer_list<std::uint32_t> args) {
        dst.push_back((std::uint32_t(args.size() + 1) << 16) | op);
        dst.insert(dst.end(), args.begin(), args.end());
    };

    constexpr std::array<float, 4> kVals{0.f, 1.f, 15.f, 1.f / 15.f};
    std::array<std::uint32_t, 4> scalar_consts{};
    for (unsigned j = 0; j < 4; ++j) {
        scalar_consts[j] = next++;
        emit_words(declarations, kOpConstant, {float_type, scalar_consts[j], std::bit_cast<std::uint32_t>(kVals[j])});
    }

    struct ConstSet {
        std::uint32_t type;
        std::array<std::uint32_t, 4> ids;
    };
    std::vector<ConstSet> type_consts;
    type_consts.push_back({float_type, scalar_consts});
    const auto constants_for = [&](std::uint32_t type) -> const std::array<std::uint32_t, 4>& {
        for (const ConstSet& cs : type_consts)
            if (cs.type == type) return cs.ids;
        const TypeInfo* ti = find_type(type);
        const std::uint32_t width = ti ? ti->width : 1u;
        std::array<std::uint32_t, 4> ids{};
        for (unsigned j = 0; j < 4; ++j) {
            ids[j] = next++;
            declarations.push_back(((3u + width) << 16) | kOpConstantComposite);
            declarations.push_back(type);
            declarations.push_back(ids[j]);
            for (std::uint32_t k = 0; k < width; ++k) declarations.push_back(scalar_consts[j]);
        }
        type_consts.push_back({type, ids});
        return type_consts.back().ids;
    };
    for (const StorePatch& p : patches) (void)constants_for(p.type);

    // Single-pass assembly into a pre-reserved buffer.
    Inst result;
    result.reserve(words.size() + (add_import ? 6u : 0u) + declarations.size() + patches.size() * 24u);
    result.insert(result.end(), words.begin(), words.begin() + 5);
    if (add_import) {
        result.insert(result.end(), words.begin() + 5, words.begin() + static_cast<std::ptrdiff_t>(mem_model_at));
        const std::size_t imp_at = result.size();
        result.resize(imp_at + 6, 0u);
        result[imp_at] = (6u << 16) | kOpExtInstImport;
        result[imp_at + 1] = import_id;
        std::memcpy(&result[imp_at + 2], "GLSL.std.450", 13);
        result.insert(result.end(), words.begin() + static_cast<std::ptrdiff_t>(mem_model_at),
                      words.begin() + static_cast<std::ptrdiff_t>(first_func_at));
    } else {
        result.insert(result.end(), words.begin() + 5, words.begin() + static_cast<std::ptrdiff_t>(first_func_at));
    }
    result.insert(result.end(), declarations.begin(), declarations.end());

    std::size_t cursor = first_func_at;
    for (const StorePatch& p : patches) {
        result.insert(result.end(), words.begin() + static_cast<std::ptrdiff_t>(cursor),
                      words.begin() + static_cast<std::ptrdiff_t>(p.at));
        const auto& c = constants_for(p.type);
        const std::uint32_t clamped = next++, scaled = next++, rounded = next++, value = next++;
        const std::uint32_t orig_val = words[p.at + 2];
        emit_words(result, kOpExtInst, {p.type, clamped, import_id, kGlslFClamp, orig_val, c[0], c[1]});
        emit_words(result, kOpFMul, {p.type, scaled, clamped, c[2]});
        emit_words(result, kOpExtInst, {p.type, rounded, import_id, kGlslRoundEven, scaled});
        emit_words(result, kOpFMul, {p.type, value, rounded, c[3]});

        const std::uint32_t store_n = words[p.at] >> 16;
        result.push_back(words[p.at]);
        result.push_back(words[p.at + 1]);
        result.push_back(value);
        if (store_n > 3) {
            result.insert(result.end(), words.begin() + static_cast<std::ptrdiff_t>(p.at + 3),
                          words.begin() + static_cast<std::ptrdiff_t>(p.at + store_n));
        }
        cursor = p.at + store_n;
    }
    result.insert(result.end(), words.begin() + static_cast<std::ptrdiff_t>(cursor), words.end());
    result[3] = next;
    words = std::move(result);
    return true;
}

}  // namespace leonida::rgba4
