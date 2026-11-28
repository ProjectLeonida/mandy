#pragma once

// Does a SPIR-V module declare a storage image, storage texel buffer or storage buffer it may write?
//
// Vulkan lets a shader stage other than compute write such a variable only when the device enabled a feature for
// it: fragmentStoresAndAtomics for the fragment stage (VUID-RuntimeSpirv-NonWritable-06340), and
// vertexPipelineStoresAndAtomics for the vertex, tessellation and geometry stages (-06341). Without the feature,
// every such variable of the stage must carry the NonWritable decoration. The translator declares the storage
// images of a shader that stores or does an image atomic without it, so the device has to know which modules need
// the feature before it builds a pipeline from them.
//
// The scan is conservative the way the rule is: it looks at declarations, not at whether an instruction writes.
// No Vulkan headers; the caller has already checked the binary's framing (spirv_words.h).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace leonida {

inline bool spirv_declares_writable_storage(std::span<const std::uint32_t> words) {
    constexpr std::uint32_t magic = 0x07230203;
    constexpr std::uint16_t op_type_image = 25, op_type_array = 28, op_type_runtime_array = 29, op_type_struct = 30,
                            op_type_pointer = 32, op_function = 54, op_variable = 59, op_decorate = 71,
                            op_member_decorate = 72, op_group_decorate = 74, op_group_member_decorate = 75;
    constexpr std::uint32_t decoration_buffer_block = 3, decoration_non_writable = 24;
    constexpr std::uint32_t dim_subpass_data = 6;  // an input attachment: Sampled 2 as well, but read-only
    constexpr std::uint32_t class_uniform_constant = 0, class_uniform = 2, class_storage_buffer = 12;
    if (words.size() < 5 || words[0] != magic) return false;

    using Pair = std::pair<std::uint32_t, std::uint32_t>;
    struct Variable {
        std::uint32_t pointer_type, id, storage_class;
    };
    std::vector<std::uint32_t> storage_images;                  // OpTypeImage with Sampled == 2
    std::vector<Pair> element_of;                               // array type -> element type
    std::vector<Pair> member_count;                             // struct type -> members
    std::vector<Pair> pointers;                                 // pointer type -> pointee type
    std::vector<std::uint32_t> non_writable, buffer_blocks;     // decorated ids
    std::vector<Pair> non_writable_members;                     // (struct type, member)
    std::vector<Variable> variables;
    // OpGroupDecorate / OpGroupMemberDecorate: a decoration group's decorations reach these targets. Resolved after
    // the scan, once every OpDecorate of the group has been seen.
    std::vector<Pair> group_targets;                                         // (group, target)
    std::vector<std::pair<std::uint32_t, Pair>> group_members;               // (group, (type, member))

    bool has_image_var = false;
    bool has_uniform_var = false;
    bool has_storage_buffer_var = false;

    // In SPIR-V's logical module layout, every annotation, type and global variable (UniformConstant, Uniform,
    // StorageBuffer) precedes the first OpFunction; function-body OpVariables have StorageClass Function.
    for (std::size_t at = 5; at < words.size();) {
        const std::uint32_t count = words[at] >> 16;
        const std::uint16_t op = static_cast<std::uint16_t>(words[at] & 0xffff);
        if (count == 0 || count > words.size() - at) return false;  // not a module the driver would take either
        if (op == op_function) break;
        const std::uint32_t* w = &words[at];
        switch (op) {
        case op_type_image:  // result, sampled type, dim, depth, arrayed, ms, sampled, format
            if (count >= 9 && w[7] == 2 && w[3] != dim_subpass_data) storage_images.push_back(w[1]);
            break;
        case op_type_array:
        case op_type_runtime_array:
            if (count >= 3) element_of.emplace_back(w[1], w[2]);
            break;
        case op_type_struct:
            if (count >= 2) member_count.emplace_back(w[1], count - 2);
            break;
        case op_type_pointer:
            if (count >= 4) pointers.emplace_back(w[1], w[3]);
            break;
        case op_variable:
            if (count >= 4) {
                const std::uint32_t sc = w[3];
                if (sc == class_uniform_constant) {
                    has_image_var = true;
                    variables.push_back({w[1], w[2], sc});
                } else if (sc == class_uniform) {
                    has_uniform_var = true;
                    variables.push_back({w[1], w[2], sc});
                } else if (sc == class_storage_buffer) {
                    has_storage_buffer_var = true;
                    variables.push_back({w[1], w[2], sc});
                }
            }
            break;
        case op_decorate:
            if (count >= 3 && w[2] == decoration_non_writable) non_writable.push_back(w[1]);
            if (count >= 3 && w[2] == decoration_buffer_block) buffer_blocks.push_back(w[1]);
            break;
        case op_member_decorate:
            if (count >= 4 && w[3] == decoration_non_writable) non_writable_members.emplace_back(w[1], w[2]);
            break;
        case op_group_decorate:  // group, targets...
            for (std::uint32_t i = 2; i < count; ++i) group_targets.emplace_back(w[1], w[i]);
            break;
        case op_group_member_decorate:  // group, (target, member) pairs
            for (std::uint32_t i = 2; i + 1 < count; i += 2) group_members.emplace_back(w[1], Pair{w[i], w[i + 1]});
            break;
        default:
            break;
        }
        at += count;
    }

    if (variables.empty()) return false;
    const bool maybe_image = has_image_var && !storage_images.empty();
    const bool maybe_uniform_ssbo = has_uniform_var && (!buffer_blocks.empty() || !group_targets.empty());
    if (!maybe_image && !has_storage_buffer_var && !maybe_uniform_ssbo) return false;

    const auto sort_unique = [](std::vector<std::uint32_t>& v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    const auto has_id = [](const std::vector<std::uint32_t>& v, std::uint32_t id) {
        return std::binary_search(v.begin(), v.end(), id);
    };
    const auto sort_map = [](std::vector<Pair>& v) {
        std::stable_sort(v.begin(), v.end(), [](const Pair& a, const Pair& b) { return a.first < b.first; });
    };
    const auto find_map = [](const std::vector<Pair>& v, std::uint32_t key, std::uint32_t& out) -> bool {
        const auto it =
            std::upper_bound(v.begin(), v.end(), key, [](std::uint32_t k, const Pair& p) { return k < p.first; });
        if (it == v.begin() || std::prev(it)->first != key) return false;
        out = std::prev(it)->second;
        return true;
    };

    sort_unique(non_writable);
    sort_unique(buffer_blocks);
    if (!group_targets.empty() || !group_members.empty()) {
        std::vector<std::uint32_t> extra_non_writable, extra_buffer_blocks;
        for (const auto& [group, target] : group_targets) {
            if (has_id(non_writable, group)) extra_non_writable.push_back(target);
            if (has_id(buffer_blocks, group)) extra_buffer_blocks.push_back(target);
        }
        for (const auto& [group, member] : group_members)
            if (has_id(non_writable, group)) non_writable_members.push_back(member);
        if (!extra_non_writable.empty()) {
            non_writable.insert(non_writable.end(), extra_non_writable.begin(), extra_non_writable.end());
            sort_unique(non_writable);
        }
        if (!extra_buffer_blocks.empty()) {
            buffer_blocks.insert(buffer_blocks.end(), extra_buffer_blocks.begin(), extra_buffer_blocks.end());
            sort_unique(buffer_blocks);
        }
    }
    if (!maybe_image && !has_storage_buffer_var && buffer_blocks.empty()) return false;

    sort_unique(storage_images);
    sort_map(pointers);
    sort_map(element_of);
    sort_map(member_count);
    std::sort(non_writable_members.begin(), non_writable_members.end());
    non_writable_members.erase(std::unique(non_writable_members.begin(), non_writable_members.end()),
                               non_writable_members.end());

    for (const Variable& variable : variables) {
        std::uint32_t type = 0;
        if (!find_map(pointers, variable.pointer_type, type)) continue;
        // An array of images or of buffer blocks: the element decides. Arrays nest a few levels at most; the bound
        // is for a malformed module whose array type names itself, which would otherwise never end.
        constexpr int max_array_depth = 16;
        int depth = 0;
        for (std::uint32_t next = 0; depth < max_array_depth && find_map(element_of, type, next); ++depth) type = next;
        if (depth == max_array_depth) continue;  // not a type a driver would accept either
        if (has_id(non_writable, variable.id)) continue;
        if (variable.storage_class == class_uniform_constant) {
            if (has_id(storage_images, type)) return true;  // a storage image or a storage texel buffer
            continue;
        }
        const bool storage_buffer = variable.storage_class == class_storage_buffer ||
                                    (variable.storage_class == class_uniform && has_id(buffer_blocks, type));
        if (!storage_buffer) continue;
        // A block is read-only only when every declared member [0, total) is NonWritable; because the deduplicated
        // slice for `type` is strictly ascending non-negative integers, [0, total) is covered iff at least `total`
        // entries exist and the (total - 1)-th entry equals (total - 1).
        std::uint32_t total = 0;
        if (!find_map(member_count, type, total) || total == 0) return true;
        const auto lo = std::lower_bound(non_writable_members.begin(), non_writable_members.end(), Pair{type, 0u});
        const auto hi =
            std::upper_bound(lo, non_writable_members.end(), type, [](std::uint32_t k, const Pair& p) { return k < p.first; });
        if (static_cast<std::size_t>(hi - lo) < total || (lo + static_cast<std::ptrdiff_t>(total - 1))->second != total - 1)
            return true;
    }
    return false;
}

}  // namespace leonida
