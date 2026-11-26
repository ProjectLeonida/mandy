// leonida::rgba4::quantize (src/rgba4_fragment.h) over hand-built SPIR-V words: 4-bit UNORM output
// quantization for B4G4R4A4 fallback render targets, location masks, decoration groups, pointer alias
// tracking, and malformed-module guards. No Vulkan, no GPU, no shader files.
#include "../src/rgba4_fragment.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "rgba4_fragment_test: FAILED: %s\n", what);
        ++failures;
    }
}

using Words = leonida::rgba4::Inst;

// SPIR-V opcodes and enumerants.
constexpr std::uint32_t OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14, OpTypeVoid = 19,
                        OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypePointer = 32,
                        OpTypeFunction = 33, OpConstant = 43, OpConstantComposite = 44, OpFunction = 54,
                        OpFunctionEnd = 56, OpFunctionCall = 57, OpVariable = 59, OpStore = 62,
                        OpCopyMemory = 63, OpCopyMemorySized = 64, OpAccessChain = 65,
                        OpPtrAccessChain = 67, OpDecorate = 71, OpDecorationGroup = 73,
                        OpGroupDecorate = 74, OpCopyObject = 83, OpBitcast = 124, OpFMul = 133,
                        OpSelect = 169, OpLabel = 248, OpPhi = 245, OpReturn = 253;
constexpr std::uint32_t StorageOutput = 3, DecorationLocation = 30;
constexpr std::uint32_t GlslRoundEven = 2, GlslFClamp = 43;

Words header(std::uint32_t bound = 100) { return {0x07230203u, 0x00010600u, 0u, bound, 0u}; }

void append(Words& w, std::uint32_t op, std::initializer_list<std::uint32_t> args) {
    const Words i = leonida::rgba4::instruction(op, args);
    w.insert(w.end(), i.begin(), i.end());
}

void append_glsl_import(Words& w, std::uint32_t id) {
    Words imp = leonida::rgba4::instruction(OpExtInstImport, {id, 0, 0, 0, 0});
    std::memcpy(imp.data() + 2, "GLSL.std.450", 13);
    w.insert(w.end(), imp.begin(), imp.end());
}

struct Scan {
    unsigned imports = 0;
    unsigned import_id = 0;
    unsigned fclamps = 0;
    unsigned roundevens = 0;
    unsigned fmuls = 0;
    unsigned stores = 0;
    bool import_before_mem = true;
    bool consts_before_fn = true;
    bool valid_framing = true;
    bool extinst_import_matches = true;
    std::uint32_t last_store_val = 0;
};

Scan inspect(const Words& w) {
    Scan s;
    if (w.size() < 5 || w[0] != 0x07230203u) {
        s.valid_framing = false;
        return s;
    }
    bool seen_mem = false, seen_fn = false;
    for (std::size_t at = 5; at < w.size();) {
        const std::uint32_t n = w[at] >> 16, op = w[at] & 0xffffu;
        if (!n || n > w.size() - at) {
            s.valid_framing = false;
            return s;
        }
        const std::uint32_t* i = &w[at];
        if (op == OpExtInstImport) {
            ++s.imports;
            s.import_id = i[1];
            if (seen_mem) s.import_before_mem = false;
        } else if (op == OpMemoryModel) {
            seen_mem = true;
        } else if (op == OpConstant || op == OpConstantComposite) {
            if (seen_fn) s.consts_before_fn = false;
        } else if (op == OpFunction) {
            seen_fn = true;
        } else if (op == OpExtInst && n >= 5) {
            if (i[3] != s.import_id || s.import_id == 0) s.extinst_import_matches = false;
            if (i[4] == GlslFClamp) ++s.fclamps;
            if (i[4] == GlslRoundEven) ++s.roundevens;
        } else if (op == OpFMul) {
            ++s.fmuls;
        } else if (op == OpStore && n >= 3) {
            ++s.stores;
            s.last_store_val = i[2];
        }
        at += n;
    }
    return s;
}

// Minimal valid fragment-like module writing %val (id 20) to %out0 (id 10, Location 0).
// Optional: existing GLSL import, second output at Location 1, OpMemoryModel inclusion.
Words basic_vec4_module(bool with_import = false, bool with_mem_model = true, std::uint32_t bound = 100) {
    Words w = header(bound);
    if (with_import) append_glsl_import(w, 1);
    if (with_mem_model) append(w, OpMemoryModel, {0, 1});
    append(w, OpDecorate, {10, DecorationLocation, 0});
    append(w, OpTypeVoid, {2});
    append(w, OpTypeFunction, {3, 2});
    append(w, OpTypeFloat, {4, 32});
    append(w, OpTypeVector, {5, 4, 4});
    append(w, OpTypePointer, {6, StorageOutput, 5});
    append(w, OpTypePointer, {7, StorageOutput, 4});
    append(w, OpVariable, {6, 10, StorageOutput});
    append(w, OpFunction, {2, 30, 0, 3});
    append(w, OpLabel, {31});
    append(w, OpStore, {10, 20});
    append(w, OpReturn, {});
    append(w, OpFunctionEnd, {});
    return w;
}

}  // namespace

int main() {
    using leonida::rgba4::quantize;

    // 1. Mask 0 is a no-op.
    {
        Words w = basic_vec4_module();
        const Words orig = w;
        check(quantize(w, 0u) && w == orig, "mask 0 leaves module untouched");
    }

    // 2. Basic vec4 output quantization without pre-existing GLSL import.
    {
        Words w = basic_vec4_module(false, true);
        check(quantize(w, 1u), "vec4 location 0 quantizes without pre-existing GLSL import");
        const Scan s = inspect(w);
        check(s.valid_framing && s.imports == 1 && s.import_before_mem && s.consts_before_fn,
              "inserted import precedes OpMemoryModel and constants precede OpFunction");
        check(s.fclamps == 1 && s.roundevens == 1 && s.fmuls == 2 && s.stores == 1 && s.extinst_import_matches,
              "one FClamp, one RoundEven, two FMuls emitted against imported GLSL.std.450");
        check(s.last_store_val != 20 && s.last_store_val < w[3], "OpStore writes quantized value below bound");
    }

    // 3. Reuses existing GLSL.std.450 import instead of inserting a second one.
    {
        Words w = basic_vec4_module(true, true);
        check(quantize(w, 1u), "vec4 location 0 quantizes with existing GLSL import");
        const Scan s = inspect(w);
        check(s.imports == 1 && s.import_id == 1 && s.extinst_import_matches && s.fclamps == 1,
              "existing GLSL.std.450 import id is reused");
    }

    // 4. Fragment shader with no OpTypeFloat 32 and no masked outputs succeeds untouched.
    {
        Words w = header();
        append(w, OpMemoryModel, {0, 1});
        append(w, OpDecorate, {10, DecorationLocation, 1});
        append(w, OpTypeVoid, {2});
        append(w, OpTypeFunction, {3, 2});
        append(w, OpTypeInt, {4, 32, 0});
        append(w, OpTypePointer, {6, StorageOutput, 4});
        append(w, OpVariable, {6, 10, StorageOutput});
        append(w, OpFunction, {2, 30, 0, 3});
        append(w, OpLabel, {31});
        append(w, OpStore, {10, 20});
        append(w, OpReturn, {});
        append(w, OpFunctionEnd, {});
        const Words orig = w;
        check(quantize(w, 1u) && w == orig, "integer-only shader with no masked outputs passes untouched");
    }

    // 5. Masked output variable missing from OpTypePointer Output is rejected, not silently skipped.
    {
        Words w = header();
        append(w, OpMemoryModel, {0, 1});
        append(w, OpDecorate, {10, DecorationLocation, 0});
        append(w, OpTypeVoid, {2});
        append(w, OpTypeFunction, {3, 2});
        append(w, OpTypeFloat, {4, 32});
        append(w, OpVariable, {99, 10, StorageOutput});  // pointer type 99 undeclared
        append(w, OpFunction, {2, 30, 0, 3});
        append(w, OpLabel, {31});
        append(w, OpStore, {10, 20});
        append(w, OpReturn, {});
        append(w, OpFunctionEnd, {});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "masked output with missing pointer type is rejected");
    }

    // 6. OpGroupDecorate providing Location is recognized and quantized.
    {
        Words w = header();
        append(w, OpMemoryModel, {0, 1});
        append(w, OpDecorate, {40, DecorationLocation, 0});
        append(w, OpDecorationGroup, {40});
        append(w, OpGroupDecorate, {40, 10});
        append(w, OpTypeVoid, {2});
        append(w, OpTypeFunction, {3, 2});
        append(w, OpTypeFloat, {4, 32});
        append(w, OpTypeVector, {5, 4, 4});
        append(w, OpTypePointer, {6, StorageOutput, 5});
        append(w, OpVariable, {6, 10, StorageOutput});
        append(w, OpFunction, {2, 30, 0, 3});
        append(w, OpLabel, {31});
        append(w, OpStore, {10, 20});
        append(w, OpReturn, {});
        append(w, OpFunctionEnd, {});
        check(quantize(w, 1u), "OpGroupDecorate Location output is quantized");
        const Scan s = inspect(w);
        check(s.fclamps == 1 && s.roundevens == 1 && s.fmuls == 2 && s.last_store_val != 20,
              "OpGroupDecorate store rewritten");
    }

    // 7. Missing OpMemoryModel when GLSL.std.450 import must be inserted fails without mutating words.
    {
        Words w = basic_vec4_module(false, false);
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "missing OpMemoryModel rejects when GLSL import must be added");
    }

    // 8. OpMemoryModel placed after OpFunction fails without mutating words.
    {
        Words w = header();
        append(w, OpDecorate, {10, DecorationLocation, 0});
        append(w, OpTypeVoid, {2});
        append(w, OpTypeFunction, {3, 2});
        append(w, OpTypeFloat, {4, 32});
        append(w, OpTypeVector, {5, 4, 4});
        append(w, OpTypePointer, {6, StorageOutput, 5});
        append(w, OpVariable, {6, 10, StorageOutput});
        append(w, OpFunction, {2, 30, 0, 3});
        append(w, OpLabel, {31});
        append(w, OpStore, {10, 20});
        append(w, OpReturn, {});
        append(w, OpFunctionEnd, {});
        append(w, OpMemoryModel, {0, 1});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpMemoryModel after OpFunction is rejected");
    }

    // 9. OpPtrAccessChain on a masked output is tracked and quantized.
    {
        Words w = header();
        append(w, OpMemoryModel, {0, 1});
        append(w, OpDecorate, {10, DecorationLocation, 0});
        append(w, OpTypeVoid, {2});
        append(w, OpTypeFunction, {3, 2});
        append(w, OpTypeFloat, {4, 32});
        append(w, OpTypeVector, {5, 4, 4});
        append(w, OpTypePointer, {6, StorageOutput, 5});
        append(w, OpTypePointer, {7, StorageOutput, 4});
        append(w, OpVariable, {6, 10, StorageOutput});
        append(w, OpFunction, {2, 30, 0, 3});
        append(w, OpLabel, {31});
        append(w, OpPtrAccessChain, {7, 11, 10, 8, 9});
        append(w, OpStore, {11, 20});
        append(w, OpReturn, {});
        append(w, OpFunctionEnd, {});
        check(quantize(w, 1u), "OpPtrAccessChain store into masked output is quantized");
        const Scan s = inspect(w);
        check(s.fclamps == 1 && s.roundevens == 1 && s.last_store_val != 20, "OpPtrAccessChain store rewritten");
    }

    // 10. OpCopyMemorySized, untracked OpAccessChain pointer type, OpBitcast, OpSelect, OpPhi, OpFunctionCall
    // on a masked output are rejected conservatively (even if another direct store also exists).
    const auto with_extra_op = [](std::uint32_t op, std::initializer_list<std::uint32_t> args) {
        Words w = header();
        append(w, OpMemoryModel, {0, 1});
        append(w, OpDecorate, {10, DecorationLocation, 0});
        append(w, OpTypeVoid, {2});
        append(w, OpTypeFunction, {3, 2});
        append(w, OpTypeFloat, {4, 32});
        append(w, OpTypeVector, {5, 4, 4});
        append(w, OpTypePointer, {6, StorageOutput, 5});
        append(w, OpVariable, {6, 10, StorageOutput});
        append(w, OpFunction, {2, 30, 0, 3});
        append(w, OpLabel, {31});
        append(w, OpStore, {10, 20});
        append(w, op, args);
        append(w, OpReturn, {});
        append(w, OpFunctionEnd, {});
        return w;
    };
    {
        Words w = with_extra_op(OpCopyMemory, {10, 25});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpCopyMemory to masked output is rejected");
    }
    {
        Words w = with_extra_op(OpCopyMemorySized, {10, 25, 26});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpCopyMemorySized to masked output is rejected");
    }
    {
        Words w = with_extra_op(OpAccessChain, {99, 11, 10, 8});  // unknown pointer type 99
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpAccessChain with unknown result pointer type is rejected");
    }
    {
        Words w = with_extra_op(OpBitcast, {99, 11, 10});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpBitcast of masked output pointer is rejected");
    }
    {
        Words w = with_extra_op(OpSelect, {6, 11, 27, 10, 10});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpSelect of masked output pointer is rejected");
    }
    {
        Words w = with_extra_op(OpPhi, {6, 11, 10, 31});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpPhi of masked output pointer is rejected");
    }
    {
        Words w = with_extra_op(OpFunctionCall, {2, 11, 30, 10});
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "OpFunctionCall passing masked output pointer is rejected");
    }

    // 11. Zero ID bound and ID bound overflow are rejected without mutating words.
    {
        Words w = basic_vec4_module(false, true, 0u);
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "zero SPIR-V ID bound is rejected");
    }
    {
        Words w = basic_vec4_module(false, true, 0xfffffffcu);
        const Words orig = w;
        check(!quantize(w, 1u) && w == orig, "ID bound overflow is rejected without wrapping");
    }

    // 12. OpCopyObject preserves tracking and quantizes.
    {
        Words w = header();
        append(w, OpMemoryModel, {0, 1});
        append(w, OpDecorate, {10, DecorationLocation, 0});
        append(w, OpTypeVoid, {2});
        append(w, OpTypeFunction, {3, 2});
        append(w, OpTypeFloat, {4, 32});
        append(w, OpTypeVector, {5, 4, 4});
        append(w, OpTypePointer, {6, StorageOutput, 5});
        append(w, OpVariable, {6, 10, StorageOutput});
        append(w, OpFunction, {2, 30, 0, 3});
        append(w, OpLabel, {31});
        append(w, OpCopyObject, {6, 11, 10});
        append(w, OpStore, {11, 20});
        append(w, OpReturn, {});
        append(w, OpFunctionEnd, {});
        check(quantize(w, 1u), "OpCopyObject of masked output is quantized");
        const Scan s = inspect(w);
        check(s.fclamps == 1 && s.roundevens == 1 && s.last_store_val != 20, "OpCopyObject store rewritten");
    }

    if (failures) return 1;
    std::printf("rgba4_fragment_test: OK (quantization, import reuse, decoration groups, pointer aliases, bounds)\n");
    return 0;
}
