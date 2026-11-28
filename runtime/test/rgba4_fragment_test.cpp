// Portable CPU-only unit tests for runtime/src/rgba4_fragment.h (RT-RGBA4-FRAGMENT-FLAT-SCAN-AND-TESTS):
// 4-bit UNORM output quantization rewrite, GLSL.std.450 import injection/reuse, access-chain component stores,
// shared scalar constants across scalar+vector targets, and refusal rules. Hand-built SPIR-V words only; no GPU.
#include "../src/rgba4_fragment.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

#define CHECK(cond) do {                                                        \
    if (!(cond)) {                                                              \
        std::fprintf(stderr, "line %d: CHECK failed: %s\n", __LINE__, #cond);   \
        return 1;                                                               \
    }                                                                           \
} while (0)

namespace {

using Words = leonida::rgba4::Inst;

constexpr std::uint32_t kOpExtInstImport = 11, kOpExtInst = 12, kOpMemoryModel = 14, kOpCapability = 17;
constexpr std::uint32_t kOpTypeInt = 21, kOpTypeFloat = 22, kOpTypeVector = 23, kOpTypePointer = 32;
constexpr std::uint32_t kOpConstant = 43, kOpConstantComposite = 44;
constexpr std::uint32_t kOpFunction = 54, kOpFunctionEnd = 56, kOpVariable = 59, kOpStore = 62, kOpCopyMemory = 63;
constexpr std::uint32_t kOpAccessChain = 65, kOpDecorate = 71, kOpFMul = 133, kOpLabel = 248, kOpReturn = 253;
constexpr std::uint32_t kDecoLocation = 30, kStorageOutput = 3;

void emit(Words& w, std::uint32_t op, std::initializer_list<std::uint32_t> args) {
    w.push_back((std::uint32_t(args.size() + 1) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

void emit_glsl_import(Words& w, std::uint32_t id) {
    const std::size_t at = w.size();
    w.resize(at + 6, 0u);
    w[at] = (6u << 16) | kOpExtInstImport;
    w[at + 1] = id;
    std::memcpy(&w[at + 2], "GLSL.std.450", 13);
}

struct ModuleStats {
    bool valid_framing = true;
    std::uint32_t glsl_imports = 0;
    std::uint32_t scalar_consts = 0;
    std::uint32_t composite_consts = 0;
    std::uint32_t fclamp_ops = 0;
    std::uint32_t round_ops = 0;
    std::uint32_t fmul_ops = 0;
    std::uint32_t stores = 0;
    bool import_before_mem_model = true;
    bool consts_before_function = true;
};

ModuleStats inspect(const Words& w) {
    ModuleStats s{};
    if (w.size() < 5 || w[0] != 0x07230203u) {
        s.valid_framing = false;
        return s;
    }
    bool saw_mem_model = false, saw_func = false;
    for (std::size_t at = 5; at < w.size();) {
        const std::uint32_t n = w[at] >> 16, op = w[at] & 0xffffu;
        if (!n || n > w.size() - at) {
            s.valid_framing = false;
            return s;
        }
        const std::uint32_t* p = &w[at];
        if (op == kOpMemoryModel) saw_mem_model = true;
        else if (op == kOpFunction) saw_func = true;
        else if (op == kOpExtInstImport && n >= 3 && (n - 2) * 4 >= 13 && std::memcmp(p + 2, "GLSL.std.450", 13) == 0) {
            ++s.glsl_imports;
            if (saw_mem_model) s.import_before_mem_model = false;
        } else if (op == kOpConstant) {
            ++s.scalar_consts;
            if (saw_func) s.consts_before_function = false;
        } else if (op == kOpConstantComposite) {
            ++s.composite_consts;
            if (saw_func) s.consts_before_function = false;
        } else if (op == kOpExtInst && n >= 5) {
            if (p[4] == 43u) ++s.fclamp_ops;
            else if (p[4] == 2u) ++s.round_ops;
        } else if (op == kOpFMul) {
            ++s.fmul_ops;
        } else if (op == kOpStore) {
            ++s.stores;
        }
        at += n;
    }
    return s;
}

Words make_ps(bool with_glsl_import, bool with_loc1_store, bool with_access_chain, bool with_copy_mem,
              bool int_output = false, bool with_extra_store_operands = false) {
    Words w{0x07230203u, 0x00010600u, 0, 40u, 0};
    emit(w, kOpCapability, {1});
    if (with_glsl_import) emit_glsl_import(w, 1u);
    emit(w, kOpMemoryModel, {0, 1});
    emit(w, kOpDecorate, {10, kDecoLocation, 0});
    emit(w, kOpDecorate, {11, kDecoLocation, 1});
    emit(w, kOpTypeFloat, {2, 32});
    emit(w, kOpTypeInt, {3, 32, 0});
    emit(w, kOpTypeVector, {4, int_output ? 3u : 2u, 4});
    emit(w, kOpTypePointer, {5, kStorageOutput, 4});  // ptr Output vec4
    emit(w, kOpTypePointer, {6, kStorageOutput, 2});  // ptr Output float
    emit(w, kOpVariable, {5, 10, kStorageOutput});    // Location 0
    emit(w, kOpVariable, {5, 11, kStorageOutput});    // Location 1
    emit(w, kOpFunction, {1, 20, 0, 1});
    emit(w, kOpLabel, {21});
    if (with_copy_mem) {
        emit(w, kOpCopyMemory, {10, 11});
    } else if (with_access_chain) {
        emit(w, kOpAccessChain, {6, 25, 10, 22});
        emit(w, kOpStore, {25, 26});  // scalar store via derived access-chain pointer
        emit(w, kOpStore, {11, 27});  // vec4 store at Location 1
    } else {
        if (with_extra_store_operands) {
            emit(w, kOpStore, {10, 27, 2, 16});  // Aligned 16 memory operands
        } else {
            emit(w, kOpStore, {10, 27});
        }
        if (with_loc1_store) emit(w, kOpStore, {11, 28});
    }
    emit(w, kOpReturn, {});
    emit(w, kOpFunctionEnd, {});
    return w;
}

}  // namespace

int main() {
    // 1. Zero mask returns true without modifying words.
    {
        Words w = make_ps(false, true, false, false);
        const Words orig = w;
        CHECK(leonida::rgba4::quantize(w, 0u) && w == orig);
    }

    // 2. Mask targeting an unused Location returns true unchanged (fast preamble-only path).
    {
        Words w = make_ps(false, true, false, false);
        const Words orig = w;
        CHECK(leonida::rgba4::quantize(w, 1u << 3) && w == orig);
    }

    // 3. Rewrite Location 0 only (injects GLSL.std.450 before OpMemoryModel, preserves extra OpStore operands).
    {
        Words w = make_ps(false, true, false, false, false, true);
        CHECK(leonida::rgba4::quantize(w, 1u << 0));
        const ModuleStats s = inspect(w);
        CHECK(s.valid_framing && s.glsl_imports == 1 && s.import_before_mem_model);
        CHECK(s.consts_before_function && s.scalar_consts == 4 && s.composite_consts == 4);
        CHECK(s.fclamp_ops == 1 && s.round_ops == 1 && s.fmul_ops == 2 && s.stores == 2);
        // Verify extra OpStore operands (Aligned 16) were preserved on the patched Location 0 store.
        bool found_aligned_store = false;
        for (std::size_t at = 5; at < w.size(); at += w[at] >> 16) {
            if ((w[at] & 0xffffu) == kOpStore && (w[at] >> 16) == 5 && w[at + 1] == 10 &&
                w[at + 3] == 2 && w[at + 4] == 16 && w[at + 2] != 27) {
                found_aligned_store = true;
            }
        }
        CHECK(found_aligned_store);
    }

    // 4. Existing GLSL.std.450 import is reused, and scalar + vec4 stores share a single set of 4 scalar OpConstants.
    {
        Words w = make_ps(true, true, true, false);
        CHECK(leonida::rgba4::quantize(w, (1u << 0) | (1u << 1)));
        const ModuleStats s = inspect(w);
        CHECK(s.valid_framing && s.glsl_imports == 1 && s.import_before_mem_model);
        CHECK(s.consts_before_function && s.scalar_consts == 4 && s.composite_consts == 4);
        CHECK(s.fclamp_ops == 2 && s.round_ops == 2 && s.fmul_ops == 4 && s.stores == 2);
    }

    // 5. Refusal cases leave `words` untouched: OpCopyMemory, non-float output, no matching store, bad header,
    //    truncated preamble/body, zero-word instruction, missing OpMemoryModel when import is needed.
    {
        Words copy_mem = make_ps(true, false, false, true);
        const Words cm_orig = copy_mem;
        CHECK(!leonida::rgba4::quantize(copy_mem, 1u << 0) && copy_mem == cm_orig);

        Words int_out = make_ps(true, false, false, false, true);
        const Words io_orig = int_out;
        CHECK(!leonida::rgba4::quantize(int_out, 1u << 0) && int_out == io_orig);

        Words no_store = make_ps(true, false, false, false);
        const Words ns_orig = no_store;
        CHECK(!leonida::rgba4::quantize(no_store, 1u << 1) && no_store == ns_orig);

        Words bad_magic{1, 2, 3, 4, 5, 6};
        CHECK(!leonida::rgba4::quantize(bad_magic, 1u));

        Words short_hdr{0x07230203u, 0x00010600u, 0, 10};
        CHECK(!leonida::rgba4::quantize(short_hdr, 1u));

        Words trunc_pre = make_ps(true, true, false, false);
        trunc_pre[5] = (999u << 16) | kOpCapability;
        CHECK(!leonida::rgba4::quantize(trunc_pre, 1u));

        Words zero_op = make_ps(true, true, false, false);
        zero_op[5] = kOpCapability;
        CHECK(!leonida::rgba4::quantize(zero_op, 1u));

        Words trunc_body = make_ps(true, true, false, false);
        trunc_body.push_back((50u << 16) | kOpReturn);
        const Words tb_orig = trunc_body;
        CHECK(!leonida::rgba4::quantize(trunc_body, 1u) && trunc_body == tb_orig);

        // Missing OpMemoryModel when GLSL.std.450 must be injected refuses cleanly.
        Words no_mm{0x07230203u, 0x00010600u, 0, 40u, 0};
        emit(no_mm, kOpDecorate, {10, kDecoLocation, 0});
        emit(no_mm, kOpTypeFloat, {2, 32});
        emit(no_mm, kOpTypeVector, {4, 2, 4});
        emit(no_mm, kOpTypePointer, {5, kStorageOutput, 4});
        emit(no_mm, kOpVariable, {5, 10, kStorageOutput});
        emit(no_mm, kOpFunction, {1, 20, 0, 1});
        emit(no_mm, kOpLabel, {21});
        emit(no_mm, kOpStore, {10, 27});
        emit(no_mm, kOpReturn, {});
        emit(no_mm, kOpFunctionEnd, {});
        const Words nm_orig = no_mm;
        CHECK(!leonida::rgba4::quantize(no_mm, 1u) && no_mm == nm_orig);
    }

    std::printf("rgba4_fragment_test: all checks passed\n");
    return 0;
}
