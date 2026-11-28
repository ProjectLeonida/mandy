// Also the repack of 96-bit R32G32B32 texels, held in R32G32B32A32 images (Depth24Job::rgb96, D3D-62): the same
// scratch, parts and lifetime, with texels of three words in the footprint and four in the scratch.
//
// The compute pass behind D24_UNORM_S8_UINT on a GPU without D24 (depth24.h, depth24_convert.comp). A copy between a
// footprint and the depth aspect of such a texture goes through a scratch buffer of floats: an upload converts the
// footprint's words into the scratch and a copy command moves the scratch into the image; a readback copies the image
// into the scratch and the pass converts it into the footprint. Single-threaded: the caller holds the context's lock.
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include "vulkan_device_fault.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "depth24.h"
#include "depth24_convert_spv.h"
#include "vulkan_aftermath.h"

namespace leonida {

class Depth24Converter {
public:
    // A device-local allocation for the scratch buffers; null on failure.
    using Allocate = std::function<VkDeviceMemory(const VkMemoryRequirements&)>;

    // `max_range`: the device's maxStorageBufferRange; a copy is split into dispatches that stay inside it.
    Depth24Converter(PFN_vkGetDeviceProcAddr get, VkDevice device, Allocate allocate,
                     std::uint64_t max_range = 1u << 27, DeviceFaultDiagnostics* diagnostics = nullptr)
        : device_(device), diagnostics_(diagnostics), allocate_(std::move(allocate)), max_range_(max_range) {
        loaded_ = vk_.load(get, device);
    }

    ~Depth24Converter() {
        if (!loaded_) return;
        DeviceFaultDiagnostics::observe(diagnostics_, vk_.vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
        release(current_);
        for (auto& batch : retired_) release(batch);
        for (auto& chunk : spare_) destroy(chunk);
        if (pipeline_) vk_.vkDestroyPipeline(device_, pipeline_, nullptr);
        if (pipeline_layout_) vk_.vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    }

    Depth24Converter(const Depth24Converter&) = delete;
    Depth24Converter& operator=(const Depth24Converter&) = delete;

    bool loaded() const { return loaded_; }

    // Everything the next submission needs for a copy: the pipeline (once), a tightly packed scratch region for the
    // whole copy (unless `whole` brings its own scratch buffer, as a test does; its scratch_offset is then the region's
    // start) and the dispatches, `parts`, each with a descriptor set over windows of the application's buffer and the
    // scratch. A part covers as many slices as fit one storage buffer range, or rows of one slice when a slice alone is
    // larger, so no copy is too big for the device. `whole.scratch` and `whole.scratch_offset` tell the copy where
    // the floats are.
    bool prepare(Depth24Job& whole, std::vector<Depth24Job>& parts) {
        // A row of the footprint is `aw` words wide and a row of the scratch `sw` (a texel is one word of each for
        // D24, three and four for 96-bit texels).
        const std::uint64_t aw = std::uint64_t(whole.width) * whole.app_texel_words();
        const std::uint64_t sw = std::uint64_t(whole.width) * whole.scratch_texel_words();
        if (!loaded_ || !make_pipeline() || !whole.app || !whole.width || !whole.height || !whole.slices ||
            whole.app_offset % 4 || (whole.scratch && whole.scratch_offset % 4) || whole.app_row_words < aw)
            return false;
        const std::uint64_t h = whole.height, s = whole.slices;
        const std::uint64_t row_span = std::uint64_t(whole.app_row_words) * (h - 1);
        if (row_span > UINT64_MAX - aw) return false;
        const std::uint64_t one_slice_words = row_span + aw;
        if (whole.app_slice_words < one_slice_words) return false;
        const std::uint64_t slice_span = std::uint64_t(whole.app_slice_words) * (s - 1);
        if (slice_span > UINT64_MAX - one_slice_words) return false;
        const std::uint64_t span_words = slice_span + one_slice_words;
        if (span_words > UINT64_MAX / 4) return false;
        const std::uint64_t span_bytes = span_words * 4;
        if (span_bytes > whole.app_size || whole.app_offset > whole.app_size - span_bytes) return false;
        if (h > UINT64_MAX / sw) return false;
        const std::uint64_t slice_scratch_words = sw * h;
        if (s > UINT64_MAX / slice_scratch_words) return false;
        const std::uint64_t total_scratch_words = slice_scratch_words * s;
        if (total_scratch_words > UINT64_MAX / 4) return false;
        const std::uint64_t total_scratch_bytes = total_scratch_words * 4;
        if (whole.scratch && total_scratch_bytes > UINT64_MAX - whole.scratch_offset) return false;

        // A window starts at a multiple of 256 (the largest minStorageBufferOffsetAlignment), which costs up to 252 bytes.
        constexpr std::uint64_t slack = 256;
        if (max_range_ <= slack) return false;
        const std::uint64_t usable_words = (max_range_ - slack) / 4;
        std::uint64_t slices_per_part = 0;  // 0: a slice alone is too big
        if (one_slice_words <= usable_words && slice_scratch_words <= usable_words) {
            const std::uint64_t app_cap = 1 + (usable_words - one_slice_words) / whole.app_slice_words;
            const std::uint64_t scratch_cap = usable_words / slice_scratch_words;
            slices_per_part = std::min({s, app_cap, scratch_cap});
        }
        std::uint64_t rows_per_part = h;
        if (!slices_per_part) {
            if (aw > usable_words || sw > usable_words) return false;
            const std::uint64_t app_cap = 1 + (usable_words - aw) / whole.app_row_words;
            const std::uint64_t scratch_cap = usable_words / sw;
            rows_per_part = std::min({h, app_cap, scratch_cap});
            if (!rows_per_part) return false;
        }
        const VkBuffer saved_scratch = whole.scratch;
        const VkDeviceSize saved_scratch_offset = whole.scratch_offset;
        if (!whole.scratch && !scratch(total_scratch_bytes, whole.scratch, whole.scratch_offset)) return false;

        parts.clear();
        for (std::uint64_t z = 0; z < s;) {
            const std::uint64_t n = slices_per_part ? std::min(slices_per_part, s - z) : 1;
            for (std::uint64_t y = 0; y < h; y += rows_per_part) {
                const std::uint64_t rows = std::min(rows_per_part, h - y);
                Depth24Job part = whole;
                part.slices = std::uint32_t(n);
                part.height = std::uint32_t(rows);
                part.app_offset = whole.app_offset + (z * whole.app_slice_words + y * whole.app_row_words) * 4;
                part.scratch_offset = whole.scratch_offset + ((z * h + y) * sw) * 4;
                if (!make_part(part, whole.app_size)) {
                    parts.clear();
                    whole.scratch = saved_scratch;
                    whole.scratch_offset = saved_scratch_offset;
                    return false;
                }
                parts.push_back(part);
            }
            z += n;
        }
        return true;
    }


    // Records the dispatch. The caller records the barriers around it.
    void run(VkCommandBuffer command_buffer, const Depth24Job& job) const {
        if (!loaded_ || !pipeline_ || !job.width || !job.height || !job.slices) return;
        struct Push {
            std::uint32_t width, height, slices, to_float, app_base, app_row, app_slice, scratch_base;
        } push{job.width, job.height, job.slices, (job.to_float ? 1u : 0u) | (job.rgb96 ? 2u : 0u), job.app_base_words,
               job.app_row_words, job.app_slice_words, job.scratch_base_words};
        vk_.vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vk_.vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &job.set, 0,
                                    nullptr);
        vk_.vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vk_.vkCmdDispatch(command_buffer, (job.width - 1u) / 8u + 1u, (job.height - 1u) / 8u + 1u, job.slices);
    }

    // The jobs prepared since the last call belong to a submission that `timeline` reaches `ticket` for.
    void retire(VkSemaphore timeline, std::uint64_t ticket) {
        if (current_.chunks.empty() && current_.pools.empty()) return;
        current_.timeline = timeline;
        current_.ticket = ticket;
        retired_.push_back(std::move(current_));
        current_ = {};
    }

    // The submissions on `timeline` are over and the semaphore is about to go (its queue is idle).
    void forget(VkSemaphore timeline) {
        std::size_t keep = 0;
        for (std::size_t i = 0; i < retired_.size(); ++i) {
            if (retired_[i].timeline == timeline) {
                release(retired_[i]);
            } else {
                if (keep != i) retired_[keep] = std::move(retired_[i]);
                ++keep;
            }
        }
        retired_.resize(keep);
    }

    // Frees (or keeps for reuse) what finished submissions used.
    void collect() {
        struct SemaphoreCacheEntry {
            VkSemaphore semaphore = VK_NULL_HANDLE;
            std::uint64_t value = 0;
            bool ok = false;
        };
        SemaphoreCacheEntry cache[8];
        std::size_t cache_count = 0;
        const auto query_semaphore = [&](VkSemaphore timeline, std::uint64_t& done) -> bool {
            for (std::size_t c = 0; c < cache_count; ++c) {
                if (cache[c].semaphore == timeline) {
                    done = cache[c].value;
                    return cache[c].ok;
                }
            }
            done = 0;
            const bool ok =
                DeviceFaultDiagnostics::observe(diagnostics_,
                    vk_.vkGetSemaphoreCounterValue(device_, timeline, &done),
                    "vkGetSemaphoreCounterValue (scratch)") == VK_SUCCESS;
            if (cache_count < 8) cache[cache_count++] = {timeline, done, ok};
            return ok;
        };

        std::size_t keep = 0;
        for (std::size_t i = 0; i < retired_.size(); ++i) {
            if (retired_[i].ticket != 0) {
                std::uint64_t done = 0;
                if (!retired_[i].timeline || !query_semaphore(retired_[i].timeline, done) ||
                    done < retired_[i].ticket) {
                    if (keep != i) retired_[keep] = std::move(retired_[i]);
                    ++keep;
                    continue;
                }
            }
            for (auto& chunk : retired_[i].chunks) {
                chunk.used = 0;
                if (spare_.size() < 4) spare_.push_back(chunk);
                else destroy(chunk);
            }
            retired_[i].chunks.clear();
            release(retired_[i]);
        }
        retired_.resize(keep);
    }

private:
    struct Chunk {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0, used = 0;
    };
    struct Batch {
        std::vector<Chunk> chunks;
        std::vector<VkDescriptorPool> pools;
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::uint64_t ticket = 0;
    };

    struct Functions {
#define LEONIDA_DEPTH24_FUNCTIONS(X)                                                                           \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout)                            \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)                       \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool)      \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateBuffer) X(vkDestroyBuffer)                 \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkFreeMemory) X(vkCmdBindPipeline)                \
    X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkGetSemaphoreCounterValue)            \
    X(vkDeviceWaitIdle)
#define LEONIDA_DEPTH24_MEMBER(name) PFN_##name name = nullptr;
        LEONIDA_DEPTH24_FUNCTIONS(LEONIDA_DEPTH24_MEMBER)
#undef LEONIDA_DEPTH24_MEMBER

        bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
            if (!get || !device) return false;
            bool all = true;
#define LEONIDA_DEPTH24_LOAD(name)                              \
    name = reinterpret_cast<PFN_##name>(get(device, #name));    \
    all = all && name != nullptr;
            LEONIDA_DEPTH24_FUNCTIONS(LEONIDA_DEPTH24_LOAD)
#undef LEONIDA_DEPTH24_LOAD
#undef LEONIDA_DEPTH24_FUNCTIONS
            return all;
        }
    };

    bool make_pipeline() {
        if (pipeline_) return true;
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 2;
        set_info.pBindings = bindings;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = sizeof(depth24_convert_spirv);
        module_info.pCode = depth24_convert_spirv;
        VkShaderModule module = VK_NULL_HANDLE;
        bool ok = vk_.vkCreateDescriptorSetLayout(device_, &set_info, nullptr, &set_layout_) == VK_SUCCESS;
        layout_info.pSetLayouts = &set_layout_;
        ok = ok && vk_.vkCreatePipelineLayout(device_, &layout_info, nullptr, &pipeline_layout_) == VK_SUCCESS &&
             (aftermath::ShaderName("(runtime) depth24_convert.comp"),
              vk_.vkCreateShaderModule(device_, &module_info, nullptr, &module) == VK_SUCCESS);
        if (ok) {
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT,
                          module, "main", nullptr};
            info.layout = pipeline_layout_;
            ok = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline_) == VK_SUCCESS;
        }
        if (module) vk_.vkDestroyShaderModule(device_, module, nullptr);
        if (!ok) {  // leave nothing half made; the next call tries again
            if (pipeline_layout_) vk_.vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
            if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
            pipeline_layout_ = VK_NULL_HANDLE;
            set_layout_ = VK_NULL_HANDLE;
        }
        return ok;
    }

    // The descriptor set of one dispatch: windows of the application's buffer and of the scratch, each starting at
    // a multiple of 256 bytes, with the part's first word given to the shader as a base.
    bool make_part(Depth24Job& part, std::uint64_t app_size) {
        constexpr std::uint64_t alignment = 256;
        const std::uint64_t span = ((std::uint64_t(part.slices) - 1) * part.app_slice_words +
                                    (std::uint64_t(part.height) - 1) * part.app_row_words +
                                    std::uint64_t(part.width) * part.app_texel_words()) * 4;
        const std::uint64_t window = part.app_offset & ~(alignment - 1);
        const std::uint64_t range = part.app_offset - window + span;
        const std::uint64_t tight = std::uint64_t(part.width) * part.scratch_texel_words() * part.height * part.slices * 4;
        const std::uint64_t scratch_window = part.scratch_offset & ~(alignment - 1);
        const std::uint64_t scratch_range = part.scratch_offset - scratch_window + tight;
        if (range > app_size || window > app_size - range) return false;
        part.app_base_words = std::uint32_t((part.app_offset - window) / 4);
        part.scratch_base_words = std::uint32_t((part.scratch_offset - scratch_window) / 4);
        part.set = allocate_set();
        if (!part.set) return false;
        const VkDescriptorBufferInfo buffers[2] = {{part.app, window, range}, {part.scratch, scratch_window, scratch_range}};
        VkWriteDescriptorSet writes[2]{};
        for (std::uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = part.set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffers[i];
        }
        vk_.vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
        return true;
    }

    VkDescriptorSet allocate_set() {
        for (int attempt = 0; attempt < 2; ++attempt) {
            if (!current_.pools.empty()) {
                VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                info.descriptorPool = current_.pools.back();
                info.descriptorSetCount = 1;
                info.pSetLayouts = &set_layout_;
                VkDescriptorSet set = VK_NULL_HANDLE;
                if (vk_.vkAllocateDescriptorSets(device_, &info, &set) == VK_SUCCESS) return set;
            }
            constexpr std::uint32_t sets = 64;
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * 2};
            VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            info.maxSets = sets;
            info.poolSizeCount = 1;
            info.pPoolSizes = &size;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (vk_.vkCreateDescriptorPool(device_, &info, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
            current_.pools.push_back(pool);
        }
        return VK_NULL_HANDLE;
    }

    // `bytes` of scratch, aligned for a descriptor offset, from this submission's chunks.
    bool scratch(std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset) {
        constexpr VkDeviceSize alignment = 256, minimum = 4u << 20;
        if (!bytes || bytes > UINT64_MAX - (alignment - 1)) return false;
        for (auto& chunk : current_.chunks) {
            if (chunk.used > UINT64_MAX - (alignment - 1)) continue;
            const VkDeviceSize start = (chunk.used + alignment - 1) & ~(alignment - 1);
            if (start <= chunk.size && bytes <= chunk.size - start) {
                chunk.used = start + bytes;
                buffer = chunk.buffer;
                offset = start;
                return true;
            }
        }
        Chunk chunk;
        for (std::size_t i = 0; i < spare_.size(); ++i)
            if (spare_[i].size >= bytes) {
                chunk = spare_[i];
                spare_.erase(spare_.begin() + std::ptrdiff_t(i));
                break;
            }
        if (!chunk.buffer && !make_chunk(std::max<std::uint64_t>(bytes, minimum), chunk)) return false;
        chunk.used = bytes;
        buffer = chunk.buffer;
        offset = 0;
        current_.chunks.push_back(chunk);
        return true;
    }

    bool make_chunk(std::uint64_t size, Chunk& out) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (vk_.vkCreateBuffer(device_, &info, nullptr, &out.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, out.buffer, &requirements);
        out.memory = allocate_ ? allocate_(requirements) : VK_NULL_HANDLE;
        if (!out.memory || vk_.vkBindBufferMemory(device_, out.buffer, out.memory, 0) != VK_SUCCESS) {
            destroy(out);
            return false;
        }
        out.size = size;
        return true;
    }

    void destroy(Chunk& chunk) {
        if (chunk.buffer) vk_.vkDestroyBuffer(device_, chunk.buffer, nullptr);
        if (chunk.memory) vk_.vkFreeMemory(device_, chunk.memory, nullptr);
        chunk = {};
    }

    void release(Batch& batch) {
        for (auto& chunk : batch.chunks) destroy(chunk);
        for (const auto pool : batch.pools) vk_.vkDestroyDescriptorPool(device_, pool, nullptr);
        batch.chunks.clear();
        batch.pools.clear();
    }

    VkDevice device_;
    DeviceFaultDiagnostics* diagnostics_;
    Allocate allocate_;
    std::uint64_t max_range_;
    Functions vk_;
    bool loaded_ = false;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    Batch current_;                // prepared, not yet submitted
    std::vector<Batch> retired_;   // submitted, waiting for the GPU
    std::vector<Chunk> spare_;     // finished scratch buffers kept for reuse
};

}  // namespace leonida
