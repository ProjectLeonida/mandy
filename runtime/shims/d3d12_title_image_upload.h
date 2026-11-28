#pragma once

#include "xg_xs_layout.h"
#include "../src/vulkan_backend.h"
#include "../src/vulkan_title_detile.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <vector>

namespace rt::d3d12 {

// Upload buffers, GPU detile pipeline, and batched command buffers per device, used under title_upload_mutex.
//
// Supports three upload paths, freely mixed inside a single batched command buffer and a single vkQueueSubmit per
// refresh pass:
//   1. cpu_slot: CPU detiled into a host-visible staging slot -> vkCmdCopyBufferToImage (fallback for unmirrored,
//      chunk-straddling, 3D-swizzled, or LEONIDA_GPU_DETILE=0 textures).
//   2. mirror_linear: direct vkCmdCopyBufferToImage from TitleMirror's sparse chunk buffer (zero CPU read/detile/stage).
//   3. mirror_detile: GPU compute detile from TitleMirror's sparse chunk buffer into a device-local scratch buffer ->
//      vkCmdCopyBufferToImage (zero CPU read/detile/stage).
//
// Nothing here waits for the GPU on the CPU (except if all kMaxSlots / kMaxBatchCmds are in flight). A single
// timeline semaphore (`ready_`) orders all transfers behind prior GPU work (retire tickets + TitleMirror sparse bind
// timeline) and orders all consumer queues behind the batch in one wait-submit per consumer queue.
class TitleImageUpload {
public:
    VkDeviceSize capacity() const {
        VkDeviceSize total = 0;
        for (const auto& slot : slots_) total += slot->capacity;
        return total;
    }

    template<class Device>
    void release(Device& d) {
        auto& vk = d.context->vk;
        wait_all(d);
        if (detile_) {
            detile_->release();
            detile_.reset();
        }
        for (auto& slot : slots_) release_buffer(d, *slot);
        batch_cmds_.clear();
        if (pool_) vk.vkDestroyCommandPool(d.info.device, pool_, nullptr);
        if (ready_) {
            {
                std::lock_guard lock(d.context->submit_mutex());
                d.context->clear_title_upload_locked(ready_);
                d.context->retire_queue_idle_locked(ready_);
            }
            vk.vkDestroySemaphore(d.info.device, ready_, nullptr);
        }
        pool_ = VK_NULL_HANDLE;
        for (auto& slot : slots_) slot->command = VK_NULL_HANDLE;
        ready_ = VK_NULL_HANDLE;
        detile_init_attempted_ = false;
    }

    struct Slot;

    // Converts rt::xg_xs::Texture into the portable leonida::TitleDetileDesc consumed by TitleDetilePipeline.
    static bool make_detile_desc(const xg_xs::Texture& layout, leonida::TitleDetileDesc& out) {
        out = {};
        switch (layout.mode) {
        case xg_xs::kLinear: out.mode = leonida::TitleDetileMode::linear; break;
        case xg_xs::k4kbS: out.mode = leonida::TitleDetileMode::sw_4kb_s; break;
        case xg_xs::k64kbS: out.mode = leonida::TitleDetileMode::sw_64kb_s; break;
        case xg_xs::k64kbD: out.mode = leonida::TitleDetileMode::sw_64kb_d; break;
        default: return false;
        }
        if (layout.slab_bytes || layout.tile_depth != 1) return false;
        const auto mips = layout.layout.mips;
        if (!mips || mips > xg_xs::kMaxMips) return false;
        out.dimension = layout.layout.dimension;
        out.element_bytes = layout.format.bytes;
        out.block_side = layout.format.side;
        out.tile_width = layout.tile_width;
        out.tile_height = layout.tile_height;
        out.array_layers = layout.array;
        out.total_size = layout.layout.size;
        out.slice_bytes = layout.layout.plane[0].slice_bytes;
        out.mips.resize(mips);
        const xg_xs::ElementAddress addr(layout);
        for (unsigned level = 0; level < mips; ++level) {
            const auto& src = layout.layout.plane[0].mips[level];
            auto& dst = out.mips[level];
            dst.offset = src.offset;
            dst.width_elements = src.width_elements;
            dst.height_elements = src.height_elements;
            dst.width_texels = src.width_texels;
            dst.height_texels = src.height_texels;
            dst.depth = src.depth;
            dst.padded_width = src.padded_width;
            dst.padded_height = src.padded_height;
            dst.pitch_bytes = src.pitch_bytes;
            dst.pitch_texels = src.pitch_texels;
            dst.in_tail = src.in_tail != 0;
            if (dst.in_tail) {
                const auto tail = addr.tail_origin(src);
                dst.tail_x = tail[0];
                dst.tail_y = tail[1];
            }
        }
        return true;
    }

    enum class UploadKind : std::uint8_t { cpu_slot, mirror_linear, mirror_detile };

    // Determines whether a mirrored texture in a single TitleMirror chunk can take GPU direct copy or GPU
    // compute detile. Returns true and sets `out_kind` when supported; false means use CPU fallback.
    template<class Device>
    bool can_gpu_upload(Device& d, const leonida::ResourceBinding& image, const xg_xs::Texture& layout,
                        VkDeviceSize mirror_offset, VkDeviceSize mirror_chunk_size, UploadKind& out_kind) {
        if (!leonida::title_gpu_detile_enabled()) return false;
        const auto block = leonida::copy_block_format(image, VK_IMAGE_ASPECT_COLOR_BIT);
        if (!block.bytes || block.bytes != layout.format.bytes || block.block != layout.format.side) return false;
        leonida::TitleDetileDesc desc;
        if (!make_detile_desc(layout, desc)) return false;
        if (desc.mode == leonida::TitleDetileMode::linear) {
            if (leonida::TitleDetilePipeline::can_direct_linear(
                    desc, mirror_offset, mirror_chunk_size, image.width, image.height, image.depth, image.mip_levels,
                    image.array_layers, image.volume)) {
                out_kind = UploadKind::mirror_linear;
                return true;
            }
            return false;
        }
        auto* pipe = ensure_detile(d);
        if (pipe && pipe->can_gpu_detile(desc, mirror_offset, mirror_chunk_size, image.width, image.height,
                                         image.mip_levels, image.array_layers, image.volume)) {
            out_kind = UploadKind::mirror_detile;
            return true;
        }
        return false;
    }

    struct BatchItem {
        UploadKind kind = UploadKind::cpu_slot;
        leonida::ResourceId id{};
        leonida::ResourceBinding image{};
        const xg_xs::Texture* layout = nullptr;
        unsigned from = 0;
        // cpu_slot:
        Slot* slot = nullptr;
        // mirror_linear / mirror_detile:
        VkBuffer mirror_buffer = VK_NULL_HANDLE;
        VkDeviceSize mirror_offset = 0;
        VkDeviceSize mirror_chunk_size = 0;
        VkSemaphore mirror_timeline = VK_NULL_HANDLE;
        std::uint64_t mirror_value = 0;
        // Set by commit_batch():
        bool ok = false;
    };

    // One CPU-staged upload in three steps (uploads the mips from `from` on; `tiled` then needs only
    // mip_chain_bytes(layout, from) bytes):
    //   reserve (serial): the copy regions, a free slot and its buffer;
    //   fill (any thread, one slot each): the detile into the slot's mapped buffer;
    //   commit / commit_batch (serial): the transfer recorded and submitted; or unreserve when unchanged.
    template<class Device>
    Slot* reserve(Device& d, const leonida::ResourceBinding& image, const xg_xs::Texture& layout, unsigned from = 0) {
        const auto block = leonida::copy_block_format(image, VK_IMAGE_ASPECT_COLOR_BIT);
        const auto mips = layout.layout.mips;
        const auto layers = layout.array;
        if (!block.bytes || block.bytes != layout.format.bytes || from >= mips || mips > image.mip_levels ||
            !layers || layers > image.array_layers) return nullptr;
        regions_.clear();
        VkDeviceSize bytes = 0;
        for (unsigned layer = 0; layer < layers; ++layer) {
            for (unsigned level = from; level < mips; ++level) {
                const auto& mip = layout.layout.plane[0].mips[level];
                VkBufferImageCopy copy{};
                copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
                copy.imageExtent = {std::max(1u, image.width >> level), std::max(1u, image.height >> level),
                                    image.volume ? std::max(1u, image.depth >> level) : 1u};
                if (mip.width_elements != (copy.imageExtent.width + block.block - 1) / block.block ||
                    mip.height_elements != (copy.imageExtent.height + block.block - 1) / block.block ||
                    (image.volume && mip.depth != copy.imageExtent.depth)) return nullptr;
                // Vulkan requires offsets aligned to four bytes and to the texel block size.
                const auto align = 4u * block.bytes;
                copy.bufferOffset = bytes + (align - bytes % align) % align;
                const VkDeviceSize row = VkDeviceSize(mip.width_elements) * block.bytes;
                const VkDeviceSize slice = row * mip.height_elements;
                if (slice > UINT32_MAX || (UINT64_MAX - copy.bufferOffset) / copy.imageExtent.depth < slice)
                    return nullptr;
                bytes = copy.bufferOffset + slice * copy.imageExtent.depth;
                regions_.push_back(copy);
            }
        }
        Slot* slot = take_slot(d, bytes);
        if (!slot || !prepare(d, *slot, bytes)) return nullptr;
        slot->regions = regions_;
        slot->reserved = true;
        return slot;
    }

    // Thread-safe for distinct slots: reads only `tiled` and writes only the slot's buffer and `columns`.
    static void fill(Slot& slot, const leonida::ResourceBinding& image, const xg_xs::Texture& layout,
                     const std::uint8_t* tiled, std::vector<std::uint64_t>& columns, bool inspect, bool& nonzero) {
        leonida::submit_phases::note_change(leonida::submit_phases::change_textures);  // not a "submit floor" one
        const auto block = leonida::copy_block_format(image, VK_IMAGE_ASPECT_COLOR_BIT);
        for (const auto& copy : slot.regions) {
            const auto level = copy.imageSubresource.mipLevel;
            const auto& mip = layout.layout.plane[0].mips[level];
            const auto row = mip.width_elements * block.bytes, slice = row * mip.height_elements;
            auto* dst = slot.mapped + copy.bufferOffset;
            xg_xs::copy_elements<false>(layout, level, copy.imageSubresource.baseArrayLayer, tiled, dst, row, slice, columns);
            if (inspect && !nonzero)
                nonzero = std::any_of(dst, dst + VkDeviceSize(slice) * copy.imageExtent.depth,
                                      [](std::uint8_t c) { return c != 0; });
        }
    }

    static void unreserve(Slot& slot) { slot.reserved = false; }

    // Commits a single CPU-staged slot (used by fill_title_texture / fill_unlocked).
    template<class Device>
    bool commit(Device& d, leonida::ResourceId id, const leonida::ResourceBinding& image,
                const xg_xs::Texture& layout, Slot& slot, unsigned from = 0) {
        slot.reserved = false;
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, from, layout.layout.mips - from, 0, layout.array};
        const auto result = d.context->write_staged_title_image(
            id, range,
            [&](const leonida::ReplayFunctions& replay, const leonida::LayoutPieces& pieces) -> VkResult {
                auto& vk = d.context->vk;
                VkResult r = vk.vkResetCommandBuffer(slot.command, 0);
                if (r != VK_SUCCESS) return r;
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if ((r = vk.vkBeginCommandBuffer(slot.command, &begin)) != VK_SUCCESS) return r;
                leonida::detail::image_barrier(replay, slot.command, image, range, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
                vk.vkCmdCopyBufferToImage(slot.command, slot.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          static_cast<std::uint32_t>(slot.regions.size()), slot.regions.data());
                for (const auto& piece : pieces)
                    leonida::detail::image_barrier(replay, slot.command, image, piece.range, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        piece.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : piece.layout,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                return vk.vkEndCommandBuffer(slot.command);
            },
            [&](const auto& waits) -> VkResult {
                const VkResult sr = submit_cmd(d, slot.command, waits, VK_NULL_HANDLE, 0);
                if (sr == VK_SUCCESS) slot.value = value_;
                return sr;
            });
        return result == VK_SUCCESS;
    }

    // Commits a batch of CPU-staged, linear-mirror, and GPU-detiled uploads into a single command buffer and single
    // vkQueueSubmit (+ one wait-submit per consumer queue), with coalesced image and memory barriers.
    template<class Device>
    void commit_batch(Device& d, std::vector<BatchItem>& items) {
        if (items.empty()) return;
        for (auto& item : items) {
            item.ok = false;
            if (item.kind == UploadKind::cpu_slot && item.slot) item.slot->reserved = false;
        }
        if (!ensure_shared(d)) return;

        // Partition into safe sub-batches: split only if the same VkImage appears twice so all transitions & copies
        // within a sub-batch operate on disjoint images and commit atomically under write_staged_title_images().
        std::size_t start = 0;
        while (start < items.size()) {
            std::size_t end = start;
            batch_seen_images_.clear();
            while (end < items.size()) {
                const VkImage img = items[end].image.image;
                if (end > start &&
                    std::find(batch_seen_images_.begin(), batch_seen_images_.end(), img) != batch_seen_images_.end())
                    break;
                batch_seen_images_.push_back(img);
                ++end;
            }
            commit_sub_batch(d, items.data() + start, end - start);
            start = end;
        }
    }

    // Uploads the mips from `from` on; `tiled` then needs only mip_chain_bytes(layout, from) bytes.
    template<class Device>
    bool write(Device& d, leonida::ResourceId id, const leonida::ResourceBinding& image,
               const xg_xs::Texture& layout, const std::uint8_t* tiled, std::vector<std::uint64_t>& columns,
               bool inspect, bool& nonzero, unsigned from = 0) {
        Slot* slot = reserve(d, image, layout, from);
        if (!slot) return false;
        fill(*slot, image, layout, tiled, columns, inspect, nonzero);
        return commit(d, id, image, layout, *slot, from);
    }

    static constexpr std::size_t kMaxSlots = 64;
    static constexpr std::size_t kMaxBatchCmds = 64;
    struct Slot {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        std::uint8_t* mapped = nullptr;
        VkDeviceSize capacity = 0;
        VkCommandBuffer command{};
        std::uint64_t value = 0;  // the ready_ value its last transfer signals; 0: none submitted
        bool reserved = false;    // between reserve() and commit()/unreserve()
        std::vector<VkBufferImageCopy> regions;
    };

private:
    struct BatchCmd {
        VkCommandBuffer command = VK_NULL_HANDLE;
        std::uint64_t value = 0;
    };

    struct PreparedItem {
        std::size_t index = 0;
        VkImageSubresourceRange range{};
        VkBuffer src_buffer = VK_NULL_HANDLE;
        std::vector<VkBufferImageCopy> owned_regions;  // non-empty for mirror_linear / mirror_detile
        const std::vector<VkBufferImageCopy>* external_regions = nullptr;  // non-null for cpu_slot (slot->regions)
        const std::vector<VkBufferImageCopy>& copy_regions() const {
            return external_regions ? *external_regions : owned_regions;
        }
    };

    template<class Device>
    leonida::TitleDetilePipeline* ensure_detile(Device& d) {
        if (detile_) return detile_.get();
        if (detile_init_attempted_) return nullptr;
        detile_init_attempted_ = true;
        if (!d.info.get_instance_proc || !d.info.instance || !d.info.physical || !d.info.device) return nullptr;
        const auto get_dev_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            d.info.get_instance_proc(d.info.instance, "vkGetDeviceProcAddr"));
        const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            d.info.get_instance_proc(d.info.instance, "vkGetPhysicalDeviceProperties"));
        if (!get_dev_proc || !get_props) return nullptr;
        VkPhysicalDeviceProperties props{};
        get_props(d.info.physical, &props);
        detile_ = std::make_unique<leonida::TitleDetilePipeline>(
            get_dev_proc, d.info.device, props.limits,
            [&d](const VkMemoryRequirements& req, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
                return d.allocate(req, required, preferred);
            });
        return detile_.get();
    }

    template<class Device>
    bool ensure_shared(Device& d) {
        auto& vk = d.context->vk;
        const auto dev = d.info.device;
        if (!pool_) {
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.queueFamilyIndex = d.info.queue_family;
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            if (vk.vkCreateCommandPool(dev, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        }
        if (!ready_) {
            VkSemaphoreTypeCreateInfo timeline{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            timeline.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timeline};
            if (vk.vkCreateSemaphore(dev, &info, nullptr, &ready_) != VK_SUCCESS) return false;
        }
        return true;
    }

    template<class Device>
    std::uint64_t poll_ready(Device& d) const {
        if (!ready_) return 0;
        auto& vk = d.context->vk;
        std::uint64_t reached = 0;
        if (vk.vkGetSemaphoreCounterValue &&
            vk.vkGetSemaphoreCounterValue(d.info.device, ready_, &reached) == VK_SUCCESS)
            return reached;
        return 0;
    }

    template<class Device>
    bool wait_value(Device& d, std::uint64_t target) {
        if (!target || !ready_) return true;
        if (poll_ready(d) >= target) return true;
        VkSemaphoreWaitInfo done{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        done.semaphoreCount = 1;
        done.pSemaphores = &ready_;
        done.pValues = &target;
        return d.context->vk.vkWaitSemaphores(d.info.device, &done, UINT64_MAX) == VK_SUCCESS;
    }

    template<class Device>
    BatchCmd* take_batch_cmd(Device& d, std::uint64_t& reached) {
        reached = poll_ready(d);
        BatchCmd* oldest = nullptr;
        for (auto& b : batch_cmds_) {
            if (!b.value || b.value <= reached) {
                b.value = 0;
                return &b;
            }
            if (!oldest || b.value < oldest->value) oldest = &b;
        }
        if (batch_cmds_.size() < kMaxBatchCmds) {
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            alloc.commandPool = pool_;
            alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc.commandBufferCount = 1;
            if (d.context->vk.vkAllocateCommandBuffers(d.info.device, &alloc, &cmd) != VK_SUCCESS) return nullptr;
            batch_cmds_.push_back({cmd, 0});
            return &batch_cmds_.back();
        }
        if (!oldest || !wait_value(d, oldest->value)) return nullptr;
        reached = std::max(reached, oldest->value);
        oldest->value = 0;
        return oldest;
    }

    template<class Device>
    void commit_sub_batch(Device& d, BatchItem* items, std::size_t count) {
        if (!count) return;
        std::uint64_t reached = 0;
        BatchCmd* bcmd = take_batch_cmd(d, reached);
        if (!bcmd) return;

        // 1. Prepare GPU detile scratch allocations & compute dispatches (if any mirror_detile items).
        std::uint32_t detile_sets = 0;
        VkDeviceSize total_scratch = 0;
        bool has_detile_items = false;
        for (std::size_t i = 0; i < count; ++i) {
            if (items[i].kind == UploadKind::mirror_detile) {
                has_detile_items = true;
                break;
            }
        }
        leonida::TitleDetilePipeline* pipe = nullptr;
        if (has_detile_items) {
            if (batch_descs_.size() < count) batch_descs_.resize(count);
            batch_dst_bases_.assign(count, 0);
            batch_dst_ends_.assign(count, 0);
            if (batch_detile_regions_.size() < count) batch_detile_regions_.resize(count);
            for (std::size_t i = 0; i < count; ++i) batch_detile_regions_[i].clear();

            for (std::size_t i = 0; i < count; ++i) {
                if (items[i].kind != UploadKind::mirror_detile) continue;
                pipe = ensure_detile(d);
                if (!pipe || !items[i].layout || !make_detile_desc(*items[i].layout, batch_descs_[i])) continue;
                const VkDeviceSize item_align =
                    (std::max)(pipe->min_storage_alignment(), VkDeviceSize(4u) * batch_descs_[i].element_bytes);
                batch_dst_bases_[i] = leonida::detail::align_up_size(total_scratch, item_align);
                if (!leonida::TitleDetilePipeline::build_scratch_regions(batch_descs_[i], batch_dst_bases_[i],
                                                                         items[i].image.width, items[i].image.height,
                                                                         batch_detile_regions_[i],
                                                                         batch_dst_ends_[i])) {
                    batch_detile_regions_[i].clear();
                    continue;
                }
                total_scratch = batch_dst_ends_[i];
                ++detile_sets;
            }
        }

        leonida::TitleDetilePipeline::ScratchSlot* scratch_slot = nullptr;
        batch_detile_sets_.clear();
        std::size_t next_detile_set = 0;
        if (detile_sets > 0 && pipe) {
            scratch_slot = pipe->acquire_scratch(reached, total_scratch, detile_sets);
            if (!scratch_slot) {
                // All scratch slots in flight: wait ONLY for the oldest in-flight scratch slot instead of draining the
                // entire upload pipeline (`value_`).
                const std::uint64_t oldest_scratch = pipe->oldest_in_flight_scratch_value();
                if (oldest_scratch > reached && wait_value(d, oldest_scratch)) {
                    reached = (std::max)(reached, oldest_scratch);
                    scratch_slot = pipe->acquire_scratch(reached, total_scratch, detile_sets);
                }
            }
            if (scratch_slot && !pipe->allocate_descriptor_sets(*scratch_slot, detile_sets, batch_detile_sets_)) {
                scratch_slot = nullptr;
            }
        }

        // 2. Build prepared list of executable items (zero self-referential pointers: copy_regions() resolves safely
        //    after vector moves/reallocations).
        batch_prepared_.clear();
        batch_prepared_.reserve(count);
        batch_targets_.clear();
        batch_targets_.reserve(count);
        batch_dispatches_.clear();
        batch_descriptor_updates_.clear();
        batch_descriptor_updates_.reserve(detile_sets);
        bool any_mirror = false;
        bool any_detile = false;
        VkSemaphore mirror_sem = VK_NULL_HANDLE;
        std::uint64_t mirror_val = 0;

        for (std::size_t i = 0; i < count; ++i) {
            BatchItem& item = items[i];
            if (!item.layout) continue;
            PreparedItem prep{};
            prep.index = i;
            prep.range = {VK_IMAGE_ASPECT_COLOR_BIT, item.from, item.layout->layout.mips - item.from, 0, item.layout->array};
            if (item.kind == UploadKind::cpu_slot) {
                if (!item.slot) continue;
                prep.src_buffer = item.slot->buffer;
                prep.external_regions = &item.slot->regions;
            } else if (item.kind == UploadKind::mirror_linear) {
                leonida::TitleDetileDesc lin_desc;
                if (!make_detile_desc(*item.layout, lin_desc) ||
                    !leonida::TitleDetilePipeline::build_linear_regions(lin_desc, item.mirror_offset, item.image.width,
                                                                        item.image.height, item.image.depth,
                                                                        item.image.volume, prep.owned_regions))
                    continue;
                prep.src_buffer = item.mirror_buffer;
                any_mirror = true;
                if (item.mirror_timeline && item.mirror_value > mirror_val) {
                    mirror_sem = item.mirror_timeline;
                    mirror_val = item.mirror_value;
                }
            } else {  // mirror_detile
                if (!scratch_slot || !pipe || batch_detile_regions_[i].empty() ||
                    next_detile_set >= batch_detile_sets_.size())
                    continue;
                const VkDescriptorSet pre_set = batch_detile_sets_[next_detile_set++];
                const std::size_t disp_start = batch_dispatches_.size();
                if (!pipe->prepare_texture(*scratch_slot, item.mirror_buffer, item.mirror_offset, batch_descs_[i],
                                           batch_dst_bases_[i], batch_dst_ends_[i], batch_detile_regions_[i],
                                           batch_dispatches_, &batch_descriptor_updates_, pre_set)) {
                    batch_dispatches_.resize(disp_start);
                    continue;
                }
                prep.src_buffer = scratch_slot->buffer;
                prep.owned_regions = std::move(batch_detile_regions_[i]);
                any_mirror = true;
                any_detile = true;
                if (item.mirror_timeline && item.mirror_value > mirror_val) {
                    mirror_sem = item.mirror_timeline;
                    mirror_val = item.mirror_value;
                }
            }
            batch_targets_.push_back({item.id, prep.range});
            batch_prepared_.push_back(std::move(prep));
        }

        if (batch_prepared_.empty()) return;
        if (pipe && !batch_descriptor_updates_.empty()) pipe->flush_descriptor_updates(batch_descriptor_updates_);
        if (any_mirror) leonida::submit_phases::note_change(leonida::submit_phases::change_textures);

        // 3. Atomically capture all layouts and record under tracker_mutex_ alone (submit_mutex_ is acquired only
        //    around the single vkQueueSubmit, and LayoutTracker updates ONLY on VK_SUCCESS).
        const std::size_t n = batch_prepared_.size();
        const auto r = d.context->write_staged_title_images(
            std::span<const leonida::VulkanContext::StagedImageTarget>(batch_targets_.data(), batch_targets_.size()),
            [&](const leonida::ReplayFunctions&, const std::vector<leonida::LayoutPieces>& all_pieces) -> VkResult {
                auto& vk = d.context->vk;
                VkResult res = vk.vkResetCommandBuffer(bcmd->command, 0);
                if (res != VK_SUCCESS) return res;
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                if ((res = vk.vkBeginCommandBuffer(bcmd->command, &begin)) != VK_SUCCESS) return res;

                if (any_mirror) {
                    VkMemoryBarrier mem{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    mem.srcAccessMask =
                        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                    mem.dstAccessMask =
                        VkAccessFlags(any_detile ? VK_ACCESS_SHADER_READ_BIT : 0u) | VK_ACCESS_TRANSFER_READ_BIT;
                    const VkPipelineStageFlags dst_stage =
                        VkPipelineStageFlags(any_detile ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0u) |
                        VK_PIPELINE_STAGE_TRANSFER_BIT;
                    vk.vkCmdPipelineBarrier(bcmd->command,
                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                            dst_stage, 0, 1, &mem, 0, nullptr, 0, nullptr);
                }

                pre_barriers_.clear();
                pre_barriers_.reserve(n);
                for (const auto& p : batch_prepared_) {
                    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
                    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    b.image = items[p.index].image.image;
                    b.subresourceRange = p.range;
                    pre_barriers_.push_back(b);
                }
                vk.vkCmdPipelineBarrier(bcmd->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                                        static_cast<std::uint32_t>(pre_barriers_.size()), pre_barriers_.data());

                if (any_detile && !batch_dispatches_.empty()) {
                    detile_->emit_dispatches(bcmd->command, batch_dispatches_);
                    VkMemoryBarrier cs_to_xfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    cs_to_xfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    cs_to_xfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vk.vkCmdPipelineBarrier(bcmd->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &cs_to_xfer, 0, nullptr, 0, nullptr);
                }

                for (const auto& p : batch_prepared_) {
                    const auto& regs = p.copy_regions();
                    vk.vkCmdCopyBufferToImage(bcmd->command, p.src_buffer, items[p.index].image.image,
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                              static_cast<std::uint32_t>(regs.size()), regs.data());
                }

                post_barriers_.clear();
                post_barriers_.reserve(n);
                for (std::size_t k = 0; k < n; ++k) {
                    const auto& p = batch_prepared_[k];
                    for (const auto& piece : all_pieces[k]) {
                        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                        b.newLayout =
                            piece.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : piece.layout;
                        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        b.image = items[p.index].image.image;
                        b.subresourceRange = piece.range;
                        post_barriers_.push_back(b);
                    }
                }
                vk.vkCmdPipelineBarrier(bcmd->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr,
                                        static_cast<std::uint32_t>(post_barriers_.size()), post_barriers_.data());

                return vk.vkEndCommandBuffer(bcmd->command);
            },
            [&](const auto& waits) -> VkResult {
                const VkResult sr = submit_cmd(d, bcmd->command, waits, mirror_sem, mirror_val);
                if (sr == VK_SUCCESS) {
                    bcmd->value = value_;
                    if (any_detile && scratch_slot) scratch_slot->value = value_;
                    for (const auto& p : batch_prepared_) {
                        auto& it = items[p.index];
                        if (it.kind == UploadKind::cpu_slot && it.slot) it.slot->value = value_;
                    }
                }
                return sr;
            });
        if (r == VK_SUCCESS) {
            for (const auto& p : batch_prepared_) items[p.index].ok = true;
        }
    }

    // Waits (outside every lock but title_upload_mutex) until the slot's last transfer has completed.
    template<class Device>
    bool wait_slot(Device& d, Slot& slot) {
        if (!slot.value || !ready_) return true;
        if (!wait_value(d, slot.value)) return false;
        slot.value = 0;
        return true;
    }

    template<class Device>
    void wait_all(Device& d) {
        for (auto& slot : slots_) wait_slot(d, *slot);
        for (auto& b : batch_cmds_) {
            if (b.value && wait_value(d, b.value)) b.value = 0;
        }
        if (value_ && ready_) wait_value(d, value_);
    }

    // Best-fit idle slot with capacity >= bytes (smallest fit to avoid burning large slots on small uploads);
    // otherwise the largest idle slot (grown in place so smaller slots stay reusable), a new slot up to kMaxSlots,
    // or the oldest in-flight slot waited for.
    template<class Device>
    Slot* take_slot(Device& d, VkDeviceSize bytes) {
        const std::uint64_t reached = poll_ready(d);
        Slot* best_fit = nullptr;
        Slot* best_grow = nullptr;
        Slot* oldest = nullptr;
        for (auto& slot : slots_) {
            if (slot->reserved) continue;
            if (!slot->value || slot->value <= reached) {
                slot->value = 0;
                if (slot->capacity >= bytes) {
                    if (!best_fit || slot->capacity < best_fit->capacity) best_fit = slot.get();
                } else if (!best_grow || slot->capacity > best_grow->capacity) {
                    best_grow = slot.get();
                }
                continue;
            }
            if (!oldest || slot->value < oldest->value) oldest = slot.get();
        }
        if (best_fit) return best_fit;
        if (best_grow) return best_grow;
        if (slots_.size() < kMaxSlots) return slots_.emplace_back(std::make_unique<Slot>()).get();
        return oldest && wait_slot(d, *oldest) ? oldest : nullptr;
    }

    template<class Device>
    void release_buffer(Device& d, Slot& slot) {
        auto& vk = d.context->vk;
        if (slot.buffer) vk.vkDestroyBuffer(d.info.device, slot.buffer, nullptr);
        if (slot.memory) vk.vkFreeMemory(d.info.device, slot.memory, nullptr);
        slot.buffer = VK_NULL_HANDLE;
        slot.memory = VK_NULL_HANDLE;
        slot.mapped = nullptr;
        slot.capacity = 0;
    }

    template<class Device>
    bool prepare(Device& d, Slot& slot, VkDeviceSize size) {
        auto& vk = d.context->vk;
        const auto dev = d.info.device;
        if (!ensure_shared(d)) return false;
        if (!slot.command) {
            VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            alloc.commandPool = pool_;
            alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            alloc.commandBufferCount = 1;
            if (vk.vkAllocateCommandBuffers(dev, &alloc, &slot.command) != VK_SUCCESS) return false;
        }
        // The slot's buffer and command buffer are rewritten below and in write(): its last transfer must be done.
        if (!wait_slot(d, slot)) return false;
        if (size <= slot.capacity) return true;
        release_buffer(d, slot);
        // Round up to power of two (min 256 KiB) to avoid frequent VkBuffer/VkDeviceMemory reallocation churn.
        VkDeviceSize alloc_size = std::max<VkDeviceSize>(256ull << 10, size);
        if (alloc_size < (UINT64_MAX >> 1)) alloc_size = std::bit_ceil(alloc_size);
        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = alloc_size;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (vk.vkCreateBuffer(dev, &buffer, nullptr, &slot.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req{};
        vk.vkGetBufferMemoryRequirements(dev, slot.buffer, &req);
        slot.memory = d.allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        void* map = nullptr;
        if (!slot.memory || vk.vkBindBufferMemory(dev, slot.buffer, slot.memory, 0) != VK_SUCCESS ||
            vk.vkMapMemory(dev, slot.memory, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
            release_buffer(d, slot);
            return false;
        }
        slot.mapped = static_cast<std::uint8_t*>(map);
        slot.capacity = alloc_size;
        return true;
    }

    // Under tracker_mutex_ and submit_mutex_ (write_staged_title_image): submits `command`, waiting on the GPU for
    // `waits` (every queue's submitted work) and optionally `mirror_sem` at `mirror_val` (pending TitleMirror sparse
    // binds), signals `ready_`, and queues one GPU-side wait per consumer queue without waiting on the CPU.
    template<class Device, class Waits>
    VkResult submit_cmd(Device& d, VkCommandBuffer command, const Waits& waits, VkSemaphore mirror_sem,
                        std::uint64_t mirror_val) {
        ++value_;
        wait_semaphores_.clear();
        wait_values_.clear();
        for (const auto& [semaphore, value] : waits) {
            wait_semaphores_.push_back(semaphore);
            wait_values_.push_back(value);
        }
        if (mirror_sem) {
            // Re-check under submit_mutex_ so any mirror bind published after commit_batch() snapshotted mirror_val
            // is waited on before this upload reads from TitleMirror chunks.
            VkSemaphore latest_sem = VK_NULL_HANDLE;
            std::uint64_t latest_val = 0;
            d.context->mirror_bind_locked(latest_sem, latest_val);
            if (latest_sem == mirror_sem && latest_val > mirror_val) mirror_val = latest_val;
        }
        if (mirror_sem && mirror_val) {
            bool merged = false;
            for (std::size_t i = 0; i < wait_semaphores_.size(); ++i) {
                if (wait_semaphores_[i] == mirror_sem) {
                    if (mirror_val > wait_values_[i]) wait_values_[i] = mirror_val;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                wait_semaphores_.push_back(mirror_sem);
                wait_values_.push_back(mirror_val);
            }
        }
        wait_stages_.assign(wait_semaphores_.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues = &value_;
        timeline.waitSemaphoreValueCount = static_cast<std::uint32_t>(wait_values_.size());
        timeline.pWaitSemaphoreValues = wait_values_.data();
        VkSubmitInfo transfer{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
        transfer.waitSemaphoreCount = static_cast<std::uint32_t>(wait_semaphores_.size());
        transfer.pWaitSemaphores = wait_semaphores_.data();
        transfer.pWaitDstStageMask = wait_stages_.data();
        transfer.commandBufferCount = 1;
        transfer.pCommandBuffers = &command;
        transfer.signalSemaphoreCount = 1;
        transfer.pSignalSemaphores = &ready_;
        auto r = d.context->queue_submit(d.info.queue, 1, &transfer, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) {
            --value_;
            return r;
        }
        // Record this upload in retire_tickets_ under submit_mutex_ so note_mirror_bind() covers in-flight GPU mirror
        // reads and freed TitleMirror homes cannot be reused while this upload is executing, and publish (ready_, value_)
        // lazily so each consumer queue waits on it on its next command-buffer submit_multi() (zero empty submits).
        d.context->record_submit_locked(ready_, value_);
        d.context->note_title_upload_locked(ready_, value_);
        return VK_SUCCESS;
    }

    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<BatchCmd> batch_cmds_;
    std::unique_ptr<leonida::TitleDetilePipeline> detile_;
    bool detile_init_attempted_ = false;
    std::vector<VkSemaphore> wait_semaphores_;
    std::vector<std::uint64_t> wait_values_;
    std::vector<VkPipelineStageFlags> wait_stages_;
    std::vector<VkImage> batch_seen_images_;
    std::vector<PreparedItem> batch_prepared_;
    std::vector<leonida::VulkanContext::StagedImageTarget> batch_targets_;
    std::vector<leonida::TitleDetileDispatch> batch_dispatches_;
    std::vector<VkDescriptorSet> batch_detile_sets_;
    std::vector<leonida::TitleDetilePipeline::PendingDescriptorUpdate> batch_descriptor_updates_;
    std::vector<leonida::TitleDetileDesc> batch_descs_;
    std::vector<VkDeviceSize> batch_dst_bases_;
    std::vector<VkDeviceSize> batch_dst_ends_;
    std::vector<std::vector<VkBufferImageCopy>> batch_detile_regions_;
    std::vector<VkImageMemoryBarrier> pre_barriers_;
    std::vector<VkImageMemoryBarrier> post_barriers_;
    VkCommandPool pool_{};
    VkSemaphore ready_{};
    std::uint64_t value_ = 0;
    std::vector<VkBufferImageCopy> regions_;
};

}  // namespace rt::d3d12
