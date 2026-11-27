// GPU detile & linear mirror direct-copy bit-for-bit parity test (Stage 3b, src/vulkan_title_detile.h).
//
// Verifies bit-for-bit parity between leonida::TitleDetilePipeline (GPU compute detile & direct linear copy into a
// real optimal-tiling VkImage, read back via vkCmdCopyImageToBuffer) and the CPU console texture detiler
// (leonida::console_texture::Layout, matching rt::xg_xs::copy_elements<false> bit-for-bit) across:
//   - BC1, BC3, BC4, BC5, BC7, R8, RG8, RGBA8, R16F, RG16F, RGBA16F, R32F, RGBA32F
//   - Swizzle modes: 4KB_S, 64KB_S, 64KB_D, and Linear direct copy
//   - Full mip chains (exercising both macro-tiled mips and mip-tail mips), array slices, odd/NPOT extents, and
//     non-zero TitleMirror chunk offsets.
// Skips with exit code 77 when no Vulkan 1.2/1.3 device with textureCompressionBC is available. No game data.

#include "vk_test_device.h"
#include "../src/console_texture_layout.h"
#include "../src/vulkan_title_detile.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(cond)                                                                         \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);             \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

namespace {

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyBuffer destroy = nullptr;
    PFN_vkFreeMemory free_mem = nullptr;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::uint8_t* mapped = nullptr;
    VkDeviceSize size = 0;

    ~Buffer() {
        if (handle && destroy) destroy(device, handle, nullptr);
        if (memory && free_mem) free_mem(device, memory, nullptr);
    }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Image {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkDestroyImage destroy = nullptr;
    PFN_vkFreeMemory free_mem = nullptr;
    VkImage handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

    ~Image() {
        if (handle && destroy) destroy(device, handle, nullptr);
        if (memory && free_mem) free_mem(device, memory, nullptr);
    }
    Image() = default;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
};

VkDeviceMemory alloc_memory(const vktest::Device& d, const VkMemoryRequirements& req,
                            VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
    VkPhysicalDeviceMemoryProperties mp{};
    d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical, &mp);
    std::uint32_t chosen = UINT32_MAX;
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(req.memoryTypeBits & (1u << i))) continue;
        const auto flags = mp.memoryTypes[i].propertyFlags;
        if ((flags & required) != required) continue;
        if (chosen == UINT32_MAX) chosen = i;
        if ((flags & preferred) == preferred) {
            chosen = i;
            break;
        }
    }
    if (chosen == UINT32_MAX) return VK_NULL_HANDLE;
    VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    info.allocationSize = req.size;
    info.memoryTypeIndex = chosen;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &info, nullptr, &mem) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    return mem;
}

bool make_host_buffer(const vktest::Device& d, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
    out.device = d.device;
    out.destroy = d.fn<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    out.free_mem = d.fn<PFN_vkFreeMemory>("vkFreeMemory");
    out.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    if (d.fn<PFN_vkCreateBuffer>("vkCreateBuffer")(d.device, &info, nullptr, &out.handle) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    d.fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(d.device, out.handle, &req);
    out.memory = alloc_memory(d, req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    if (!out.memory) return false;
    if (d.fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(d.device, out.handle, out.memory, 0) != VK_SUCCESS)
        return false;
    void* ptr = nullptr;
    if (d.fn<PFN_vkMapMemory>("vkMapMemory")(d.device, out.memory, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS)
        return false;
    out.mapped = static_cast<std::uint8_t*>(ptr);
    return true;
}

bool make_image(const vktest::Device& d, VkFormat format, std::uint32_t width, std::uint32_t height,
                std::uint32_t mips, std::uint32_t layers, Image& out) {
    out.device = d.device;
    out.destroy = d.fn<PFN_vkDestroyImage>("vkDestroyImage");
    out.free_mem = d.fn<PFN_vkFreeMemory>("vkFreeMemory");
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {width, height, 1};
    info.mipLevels = mips;
    info.arrayLayers = layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (d.fn<PFN_vkCreateImage>("vkCreateImage")(d.device, &info, nullptr, &out.handle) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    d.fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(d.device, out.handle, &req);
    out.memory = alloc_memory(d, req, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (!out.memory) return false;
    return d.fn<PFN_vkBindImageMemory>("vkBindImageMemory")(d.device, out.handle, out.memory, 0) == VK_SUCCESS;
}

leonida::TitleDetileDesc to_detile_desc(const leonida::console_texture::Layout& layout) {
    leonida::TitleDetileDesc out{};
    switch (layout.desc.swizzle) {
    case leonida::console_texture::kLinear: out.mode = leonida::TitleDetileMode::linear; break;
    case leonida::console_texture::kSw4kbS: out.mode = leonida::TitleDetileMode::sw_4kb_s; break;
    case leonida::console_texture::kSw64kbS: out.mode = leonida::TitleDetileMode::sw_64kb_s; break;
    case leonida::console_texture::kSw64kbD: out.mode = leonida::TitleDetileMode::sw_64kb_d; break;
    default: break;
    }
    out.dimension = 3;  // 2D
    out.element_bytes = layout.desc.element_bytes;
    out.block_side = layout.desc.block;
    out.tile_width = layout.block[0];
    out.tile_height = layout.block[1];
    out.array_layers = layout.slices;
    out.total_size = layout.size;
    out.slice_bytes = layout.slice_size;
    out.mips.resize(layout.mips.size());
    for (std::size_t i = 0; i < layout.mips.size(); ++i) {
        const auto& m = layout.mips[i];
        auto& dst = out.mips[i];
        dst.offset = m.offset;
        dst.width_elements = m.width;
        dst.height_elements = m.height;
        dst.width_texels = std::max(1u, layout.desc.width >> i);
        dst.height_texels = std::max(1u, layout.desc.height >> i);
        dst.depth = m.depth;
        dst.padded_width = m.pitch;
        dst.padded_height = m.rows;
        dst.pitch_bytes = m.pitch * layout.desc.element_bytes;
        dst.pitch_texels = m.pitch * layout.desc.block;
        dst.in_tail = m.tail.has_value();
        if (m.tail) {
            dst.tail_x = m.tail->x;
            dst.tail_y = m.tail->y;
        }
    }
    return out;
}

struct FormatCase {
    const char* name;
    VkFormat vk_format;
    std::uint32_t element_bytes;
    std::uint32_t block_side;
};

struct ExtentCase {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t mips;
    std::uint32_t layers;
    std::uint32_t swizzle;
    VkDeviceSize src_offset;
};

int run_case(vktest::Device& d, leonida::TitleDetilePipeline& pipe, const FormatCase& fmt, const ExtentCase& ec,
             std::uint64_t& serial) {
    namespace ct = leonida::console_texture;
    ct::Desc cd{};
    cd.dimension = ct::Dimension::tex2d;
    cd.width = ec.width;
    cd.height = ec.height;
    cd.depth = ec.layers;
    cd.mips = ec.mips;
    cd.element_bytes = fmt.element_bytes;
    cd.block = fmt.block_side;
    cd.samples = 1;
    cd.swizzle = ec.swizzle;
    const auto layout_opt = ct::Layout::make(cd);
    EXPECT(layout_opt.has_value());
    const ct::Layout& layout = *layout_opt;
    const auto desc = to_detile_desc(layout);

    // 1. Fill simulated TitleMirror chunk buffer with deterministic non-repeating bytes at ec.src_offset.
    const VkDeviceSize chunk_size = ec.src_offset + layout.size + 4096;
    Buffer mirror_chunk;
    EXPECT(make_host_buffer(d, chunk_size,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, mirror_chunk));
    std::memset(mirror_chunk.mapped, 0xCC, std::size_t(chunk_size));
    std::uint8_t* const tiled_ptr = mirror_chunk.mapped + ec.src_offset;
    std::uint64_t seed = 0x9e3779b97f4a7c15ull ^ (serial * 0xbf58476d1ce4e5b9ull) ^ (std::uint64_t(ec.width) << 16) ^
                         ec.height ^ (std::uint64_t(ec.swizzle) << 32) ^ (std::uint64_t(fmt.element_bytes) << 40);
    for (std::size_t i = 0; i < layout.size; ++i) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        tiled_ptr[i] = static_cast<std::uint8_t>(seed >> 56);
    }

    // 2. Compute CPU ground truth for every (layer, mip) via console_texture::Layout::detile().
    std::vector<std::vector<std::uint8_t>> cpu_truth;
    cpu_truth.reserve(std::size_t(ec.layers) * ec.mips);
    std::vector<VkBufferImageCopy> readback_regions;
    VkDeviceSize readback_bytes = 0;
    const VkDeviceSize rb_align = std::max<VkDeviceSize>(4u, fmt.element_bytes) * 4u;
    for (std::uint32_t layer = 0; layer < ec.layers; ++layer) {
        for (std::uint32_t mip = 0; mip < ec.mips; ++mip) {
            const auto& m = layout.mips[mip];
            const std::size_t row_bytes = std::size_t(m.width) * fmt.element_bytes;
            const std::size_t slice_bytes = row_bytes * m.height;
            std::vector<std::uint8_t> linear(slice_bytes, 0xA5);
            EXPECT(layout.detile(tiled_ptr, std::size_t(layout.size), mip, layer, linear.data(), linear.size(),
                                 row_bytes, slice_bytes) == ct::Error::ok);
            cpu_truth.push_back(std::move(linear));

            VkBufferImageCopy rb{};
            rb.bufferOffset = leonida::detail::align_up_size(readback_bytes, rb_align);
            rb.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
            rb.imageExtent = {std::max(1u, ec.width >> mip), std::max(1u, ec.height >> mip), 1u};
            readback_regions.push_back(rb);
            readback_bytes = rb.bufferOffset + slice_bytes;
        }
    }

    // 3. Create target optimal-tiling VkImage and host-visible readback buffer.
    Image image;
    EXPECT(make_image(d, fmt.vk_format, ec.width, ec.height, ec.mips, ec.layers, image));
    Buffer readback;
    EXPECT(make_host_buffer(d, std::max<VkDeviceSize>(readback_bytes, 64), VK_BUFFER_USAGE_TRANSFER_DST_BIT, readback));
    std::memset(readback.mapped, 0x5A, std::size_t(readback.size));

    // 4. Execute Stage 3b GPU path (direct linear copy OR compute detile -> scratch -> image) + readback.
    const bool is_linear = ec.swizzle == ct::kLinear;
    std::vector<VkBufferImageCopy> copy_regions;
    std::vector<leonida::TitleDetileDispatch> dispatches;
    leonida::TitleDetilePipeline::ScratchSlot* scratch = nullptr;
    VkBuffer copy_src = VK_NULL_HANDLE;

    if (is_linear) {
        EXPECT(leonida::TitleDetilePipeline::can_direct_linear(desc, ec.src_offset, chunk_size, ec.width, ec.height, 1,
                                                               ec.mips, ec.layers, false));
        EXPECT(leonida::TitleDetilePipeline::build_linear_regions(desc, ec.src_offset, ec.width, ec.height, 1, false,
                                                                  copy_regions));
        copy_src = mirror_chunk.handle;
    } else {
        EXPECT(pipe.can_gpu_detile(desc, ec.src_offset, chunk_size, ec.width, ec.height, ec.mips, ec.layers, false));
        VkDeviceSize scratch_end = 0;
        EXPECT(leonida::TitleDetilePipeline::build_scratch_regions(desc, 0, ec.width, ec.height, copy_regions,
                                                                   scratch_end));
        // Pass serial as `reached` because vktest::Device::run() waits for queue idle after each test case.
        scratch = pipe.acquire_scratch(serial, scratch_end, 1);
        EXPECT(scratch != nullptr);
        EXPECT(pipe.prepare_texture(*scratch, mirror_chunk.handle, ec.src_offset, desc, 0, scratch_end, copy_regions,
                                    dispatches));
        copy_src = scratch->buffer;
    }

    const bool ran = d.run([&](VkCommandBuffer cb) {
        const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        const auto copy_b2i = d.fn<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
        const auto copy_i2b = d.fn<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");

        // Mirror visibility barrier (HOST_WRITE | TRANSFER_WRITE -> SHADER_READ | TRANSFER_READ).
        VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mem.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mem.dstAccessMask =
            VkAccessFlags(is_linear ? 0u : VK_ACCESS_SHADER_READ_BIT) | VK_ACCESS_TRANSFER_READ_BIT;
        const VkPipelineStageFlags dst_stage =
            VkPipelineStageFlags(is_linear ? 0u : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) |
            VK_PIPELINE_STAGE_TRANSFER_BIT;
        barrier(cb, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, dst_stage, 0, 1, &mem, 0, nullptr, 0,
                nullptr);

        VkImageMemoryBarrier to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.srcQueueFamilyIndex = to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_dst.image = image.handle;
        to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, ec.mips, 0, ec.layers};
        barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &to_dst);

        if (!is_linear) {
            pipe.emit_dispatches(cb, dispatches);
            VkMemoryBarrier cs_to_xfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            cs_to_xfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            cs_to_xfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &cs_to_xfer, 0,
                    nullptr, 0, nullptr);
        }

        copy_b2i(cb, copy_src, image.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 static_cast<std::uint32_t>(copy_regions.size()), copy_regions.data());

        VkImageMemoryBarrier to_src{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_src.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_src.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_src.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        to_src.srcQueueFamilyIndex = to_src.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_src.image = image.handle;
        to_src.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, ec.mips, 0, ec.layers};
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &to_src);

        copy_i2b(cb, image.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle,
                 static_cast<std::uint32_t>(readback_regions.size()), readback_regions.data());

        VkMemoryBarrier host_vis{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_vis.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host_vis.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_vis, 0, nullptr, 0,
                nullptr);
        return true;
    });
    EXPECT(ran);
    if (scratch) scratch->value = serial;
    ++serial;

    // 5. Bit-for-bit comparison of every single byte across every layer and mip!
    for (std::size_t idx = 0; idx < cpu_truth.size(); ++idx) {
        const auto& expected = cpu_truth[idx];
        const std::uint8_t* actual = readback.mapped + readback_regions[idx].bufferOffset;
        if (std::memcmp(actual, expected.data(), expected.size()) != 0) {
            for (std::size_t b = 0; b < expected.size(); ++b) {
                if (actual[b] != expected[b]) {
                    std::fprintf(stderr,
                                 "MISMATCH fmt=%s swizzle=%u %ux%u mips=%u layers=%u region=%zu byte=%zu: "
                                 "gpu=0x%02x cpu=0x%02x (in_tail=%d)\n",
                                 fmt.name, ec.swizzle, ec.width, ec.height, ec.mips, ec.layers, idx, b,
                                 unsigned(actual[b]), unsigned(expected[b]),
                                 int(layout.mips[readback_regions[idx].imageSubresource.mipLevel].tail.has_value()));
                    return 1;
                }
            }
        }
    }
    return 0;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::puts("vk_title_detile_test: no Vulkan 1.2/1.3 device available (skipping)");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    if (!d.bc) {
        std::puts("vk_title_detile_test: textureCompressionBC unavailable (skipping)");
        return 77;
    }

    VkPhysicalDeviceProperties props{};
    d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &props);

    leonida::TitleDetilePipeline pipe(
        d.device_proc, d.device, props.limits,
        [&d](const VkMemoryRequirements& req, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
            return alloc_memory(d, req, required, preferred);
        });

    const FormatCase formats[] = {
        {"BC1", VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, 4},
        {"BC3", VK_FORMAT_BC3_UNORM_BLOCK, 16, 4},
        {"BC4", VK_FORMAT_BC4_UNORM_BLOCK, 8, 4},
        {"BC5", VK_FORMAT_BC5_UNORM_BLOCK, 16, 4},
        {"BC7", VK_FORMAT_BC7_UNORM_BLOCK, 16, 4},
        {"R8", VK_FORMAT_R8_UNORM, 1, 1},
        {"RG8", VK_FORMAT_R8G8_UNORM, 2, 1},
        {"RGBA8", VK_FORMAT_R8G8B8A8_UNORM, 4, 1},
        {"R16F", VK_FORMAT_R16_SFLOAT, 2, 1},
        {"RG16F", VK_FORMAT_R16G16_SFLOAT, 4, 1},
        {"RGBA16F", VK_FORMAT_R16G16B16A16_SFLOAT, 8, 1},
        {"R32F", VK_FORMAT_R32_SFLOAT, 4, 1},
        {"RGBA32F", VK_FORMAT_R32G32B32A32_SFLOAT, 16, 1},
    };

    namespace ct = leonida::console_texture;
    const ExtentCase extents[] = {
        // Multi-mip chains with both macro-tiled mips and mip tails across all 3 swizzle modes:
        {256, 256, 9, 1, ct::kSw64kbS, 65536},
        {256, 256, 9, 1, ct::kSw64kbD, 65536},
        {256, 128, 9, 1, ct::kSw4kbS, 4096},
        // Odd / non-power-of-two extents with multi-mip chains & mip tails:
        {127, 95, 7, 1, ct::kSw64kbS, 65536},
        {73, 51, 6, 1, ct::kSw64kbD, 65536},
        {37, 29, 5, 1, ct::kSw4kbS, 4096},
        // Tiny odd extents (tests 1B/2B word packing boundary & partial tail bytes):
        {13, 7, 1, 1, ct::kSw4kbS, 4096},
        {3, 5, 1, 1, ct::kSw64kbD, 65536},
        {1, 1, 1, 1, ct::kSw64kbS, 65536},
        // Array textures (multiple slices + mips + mip tails):
        {64, 64, 6, 3, ct::kSw64kbD, 65536},
        {33, 27, 4, 2, ct::kSw4kbS, 4096},
        // Linear direct copy (multi-mip, array slices, odd extents, non-zero offset):
        {128, 64, 5, 2, ct::kLinear, 4096},
        {37, 29, 4, 1, ct::kLinear, 256},
        {13, 7, 1, 1, ct::kLinear, 256},
    };

    std::uint64_t serial = 1;
    std::uint32_t passed = 0;
    for (const auto& fmt : formats) {
        for (const auto& ec : extents) {
            // Skip odd extents smaller than a 4x4 block for BC formats when mip count would be valid,
            // Note: console_texture::Layout handles odd BC extents (rounds up to blocks); test all of them!
            if (run_case(d, pipe, fmt, ec, serial) != 0) return 1;
            ++passed;
        }
    }

    EXPECT(d.validation_errors() == 0);
    std::printf("vk_title_detile_test: %u cases passed bit-for-bit (13 formats x 14 layout/mode/mip/array/odd/linear "
                "cases, 0 validation errors)\n",
                passed);
    return 0;
}
