#pragma once

#include "gfx10_descriptors.h"

#include <cstdint>

namespace leonida::title_vsharp_guard {

// Redirects a decoded RDNA2 buffer V# whose base is an unrebased title-memory virtual address (or GPU-only title view)
// to a zeroed arena scratch address with num_records = 0 (w[2] = 0). Translated shaders bounds-check buffer loads and
// stores against num_records, so an OOB read returns 0 and an OOB write is dropped instead of reaching the GPU with an
// unmapped raw title VA (READ_INVALID / WRITE_INVALID -> VK_ERROR_DEVICE_LOST).
//
// Leaves untouched: null descriptors, image/sampler descriptors (w[3] >> 30 != 0), arena addresses (base < arena_size),
// any known GPU VA (host imports, geometry views, upload/readback buffers), non-V# data that fails gfx10::decode, and
// addresses outside title memory. Returns true iff `w` was redirected.
template <class ToCpu, class IsKnownGpuVa, class IsTitleMemory, class GetScratchVa>
bool redirect_untranslated_vsharp(std::uint32_t w[4], std::uint64_t arena_size, ToCpu&& to_cpu,
                                  IsKnownGpuVa&& is_known_gpu_va, IsTitleMemory&& is_title_memory,
                                  GetScratchVa&& get_scratch_va, std::uint64_t* raw_va_out = nullptr,
                                  std::uint64_t* span_out = nullptr) {
    const std::uint32_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    if (!(w0 | w1 | w2 | w3) || (w3 >> 30)) return false;
    const std::uint64_t base = std::uint64_t(w0) | (std::uint64_t(w1 & 0xFFFFu) << 32);
    if (base < arena_size || is_known_gpu_va(base)) return false;
    const leonida::gfx10::BufferWords words{w0, w1, w2, w3};
    leonida::gfx10::Buffer desc{};
    if (!leonida::gfx10::decode(words, desc)) return false;
    const std::uint32_t stride = (w1 >> 16) & 0x3FFFu;
    const std::uint64_t span = std::uint64_t(w2) * (stride ? stride : 1);
    const std::uint64_t cpu_base = to_cpu(base, span);
    if (!is_title_memory(base) && (cpu_base == base || !is_title_memory(cpu_base))) return false;
    const std::uint64_t scratch = get_scratch_va();
    if (scratch == UINT64_MAX || (scratch >> 48)) return false;
    w[0] = std::uint32_t(scratch);
    w[1] = (w1 & 0xFFFF0000u) | std::uint32_t((scratch >> 32) & 0xFFFFu);
    w[2] = 0;
    if (raw_va_out) *raw_va_out = base;
    if (span_out) *span_out = span;
    return true;
}

}  // namespace leonida::title_vsharp_guard
