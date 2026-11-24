// BindlessSet's CPU bookkeeping, with a fake Vulkan device: no GPU. Checks that the pending-write queue (one write
// per binding and index, in first-queued order, the last info winning) and the free-index search give what the
// former std::unordered_map queue and byte loop gave, and times both.
//
//   bindless_queue_test          equivalence checks (what ctest runs), then a short benchmark
//   bindless_queue_test --bench  a longer benchmark
//
// Each run prints a digest of every descriptor write the fake device received and every index the set returned.
// Building this file against another copy of the header (-DLEONIDA_BINDLESS_HEADER="\"path\"") and comparing the
// digests checks the whole class, not only the copies of the old algorithms below.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#ifndef LEONIDA_BINDLESS_HEADER
#define LEONIDA_BINDLESS_HEADER "../src/vulkan_bindless.h"
#endif
#include LEONIDA_BINDLESS_HEADER

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Every heap allocation in the process, to count what a hot loop allocates.
static std::atomic<std::uint64_t> g_allocations{0};
void* operator new(std::size_t size) {
    g_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

#define EXPECT(condition)                                                          \
    do {                                                                           \
        if (!(condition)) {                                                        \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);           \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

using namespace leonida;

namespace {

// ---- the fake device -------------------------------------------------------------------------------------------

struct Write {
    std::uint32_t binding, element;
    VkDescriptorType type;
    VkDescriptorImageInfo info;
};

struct Fake {
    std::uint64_t next_handle = 0x1000;
    std::map<std::uint64_t, std::vector<std::uint8_t>> memory;  // what vkMapMemory hands out
    std::vector<std::vector<Write>> updates;                    // one entry per vkUpdateDescriptorSets
    bool record = true;
    std::uint64_t digest = 1469598103934665603ull;  // FNV-1a over every write and returned index
    void mix(std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            digest ^= (v >> (8 * i)) & 0xff;
            digest *= 1099511628211ull;
        }
    }
};
Fake* g_fake = nullptr;

template <typename T>
T handle() {
    return reinterpret_cast<T>(static_cast<std::uintptr_t>(g_fake->next_handle++));
}
std::uint64_t raw(const void* h) { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(h)); }

VKAPI_ATTR VkResult VKAPI_CALL fake_create_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo*,
                                                      const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
    *out = handle<VkDescriptorSetLayout>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_set_layout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL fake_create_pool(VkDevice, const VkDescriptorPoolCreateInfo*,
                                                const VkAllocationCallbacks*, VkDescriptorPool* out) {
    *out = handle<VkDescriptorPool>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL fake_allocate_sets(VkDevice, const VkDescriptorSetAllocateInfo* info,
                                                  VkDescriptorSet* out) {
    for (std::uint32_t i = 0; i < info->descriptorSetCount; ++i) out[i] = handle<VkDescriptorSet>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_update_sets(VkDevice, std::uint32_t count, const VkWriteDescriptorSet* writes,
                                            std::uint32_t, const VkCopyDescriptorSet*) {
    Fake& f = *g_fake;
    if (!f.record) return;  // benchmarks: the fake costs nothing
    f.mix(0xfeedull);
    std::vector<Write> update;
    for (std::uint32_t w = 0; w < count; ++w) {
        const VkWriteDescriptorSet& write = writes[w];
        for (std::uint32_t e = 0; e < write.descriptorCount; ++e) {
            const VkDescriptorImageInfo info = write.pImageInfo ? write.pImageInfo[e] : VkDescriptorImageInfo{};
            f.mix(write.dstBinding);
            f.mix(write.dstArrayElement + e);
            f.mix(write.descriptorType);
            f.mix(raw(info.sampler));
            f.mix(raw(info.imageView));
            f.mix(info.imageLayout);
            update.push_back({write.dstBinding, write.dstArrayElement + e, write.descriptorType, info});
        }
    }
    f.updates.push_back(std::move(update));
}
VKAPI_ATTR VkResult VKAPI_CALL fake_create_buffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*,
                                                  VkBuffer* out) {
    *out = handle<VkBuffer>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL fake_buffer_requirements(VkDevice, VkBuffer, VkMemoryRequirements* out) {
    *out = {1 << 20, 256, 1};
}
VKAPI_ATTR VkResult VKAPI_CALL fake_bind_buffer(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_allocate_memory(VkDevice, const VkMemoryAllocateInfo* info,
                                                    const VkAllocationCallbacks*, VkDeviceMemory* out) {
    *out = handle<VkDeviceMemory>();
    g_fake->memory[raw(*out)].assign(std::size_t(info->allocationSize), 0);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    if (g_fake) g_fake->memory.erase(raw(memory));
}
VKAPI_ATTR VkResult VKAPI_CALL fake_map_memory(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize,
                                               VkMemoryMapFlags, void** out) {
    *out = g_fake->memory[raw(memory)].data();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*,
                                                 VkImage* out) {
    *out = handle<VkImage>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL fake_image_requirements(VkDevice, VkImage, VkMemoryRequirements* out) {
    *out = {256, 256, 1};
}
VKAPI_ATTR VkResult VKAPI_CALL fake_bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL fake_create_view(VkDevice, const VkImageViewCreateInfo*, const VkAllocationCallbacks*,
                                                VkImageView* out) {
    *out = handle<VkImageView>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_view(VkDevice, VkImageView, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL fake_create_sampler(VkDevice, const VkSamplerCreateInfo*, const VkAllocationCallbacks*,
                                                   VkSampler* out) {
    *out = handle<VkSampler>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_sampler(VkDevice, VkSampler, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                            VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                            const VkBufferMemoryBarrier*, std::uint32_t,
                                            const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_copy(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_update(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {}
VKAPI_ATTR void VKAPI_CALL fake_cmd_clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*,
                                          std::uint32_t, const VkImageSubresourceRange*) {}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL fake_get(VkDevice, const char* name) {
    struct Entry {
        const char* name;
        PFN_vkVoidFunction fn;
    };
#define F(n, f) {#n, reinterpret_cast<PFN_vkVoidFunction>(&f)}
    static const Entry entries[] = {
        F(vkCreateDescriptorSetLayout, fake_create_set_layout), F(vkDestroyDescriptorSetLayout, fake_destroy_set_layout),
        F(vkCreateDescriptorPool, fake_create_pool),           F(vkDestroyDescriptorPool, fake_destroy_pool),
        F(vkAllocateDescriptorSets, fake_allocate_sets),       F(vkUpdateDescriptorSets, fake_update_sets),
        F(vkCreateBuffer, fake_create_buffer),                 F(vkDestroyBuffer, fake_destroy_buffer),
        F(vkGetBufferMemoryRequirements, fake_buffer_requirements), F(vkBindBufferMemory, fake_bind_buffer),
        F(vkAllocateMemory, fake_allocate_memory),             F(vkFreeMemory, fake_free_memory),
        F(vkMapMemory, fake_map_memory),                       F(vkCreateImage, fake_create_image),
        F(vkDestroyImage, fake_destroy_image),                 F(vkGetImageMemoryRequirements, fake_image_requirements),
        F(vkBindImageMemory, fake_bind_image),                 F(vkCreateImageView, fake_create_view),
        F(vkDestroyImageView, fake_destroy_view),              F(vkCreateSampler, fake_create_sampler),
        F(vkDestroySampler, fake_destroy_sampler),             F(vkCmdPipelineBarrier, fake_cmd_barrier),
        F(vkCmdCopyBuffer, fake_cmd_copy),                     F(vkCmdUpdateBuffer, fake_cmd_update),
        F(vkCmdClearColorImage, fake_cmd_clear),
    };
#undef F
    for (const Entry& e : entries)
        if (std::strcmp(e.name, name) == 0) return e.fn;
    return nullptr;
}

VkPhysicalDeviceMemoryProperties fake_memory() {
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memory.memoryHeapCount = 1;
    memory.memoryHeaps[0].size = 1ull << 30;
    return memory;
}

VkDevice fake_device() { return reinterpret_cast<VkDevice>(static_cast<std::uintptr_t>(0x10)); }
VkCommandBuffer fake_command_buffer() { return reinterpret_cast<VkCommandBuffer>(static_cast<std::uintptr_t>(0x20)); }

BindlessFeatures features(bool ms, std::uint32_t images_2d) {
    BindlessFeatures f;
    f.partially_bound = true;
    f.update_after_bind = true;
    if (ms) f.ms_samples = {VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_4_BIT, VK_SAMPLE_COUNT_1_BIT};
    if (images_2d) f.counts[bindless::sampled_binding(ImageDimension::dim_2d)] = images_2d;
    return f;
}

// ---- the former algorithms, as they were in vulkan_bindless.h --------------------------------------------------

struct Pending {
    std::uint32_t binding, index;
    VkDescriptorImageInfo info;
};

struct OldQueue {
    std::vector<Pending> pending;
    std::unordered_map<std::uint64_t, std::size_t> slots;
    void queue(std::uint32_t binding, std::uint32_t index, const VkDescriptorImageInfo& info) {
        const std::uint64_t slot = (std::uint64_t(binding) << 32) | index;
        const auto [at, added] = slots.try_emplace(slot, pending.size());
        if (added)
            pending.push_back({binding, index, info});
        else
            pending[at->second].info = info;
    }
    void clear() {
        pending.clear();
        slots.clear();
    }
};

struct NewQueue {  // the same steps as BindlessSet::queue() and the end of flush()
    std::vector<Pending> pending;
    std::array<std::vector<std::uint32_t>, bindless::binding_count> at;
    explicit NewQueue(const BindlessFeatures& f) {
        for (std::uint32_t b = 0; b < bindless::binding_count; ++b) at[b].assign(f.counts[b], 0);
    }
    void queue(std::uint32_t binding, std::uint32_t index, const VkDescriptorImageInfo& info) {
        auto& a = at[binding];
        if (index >= a.size()) a.resize(std::size_t(index) + 1, 0);
        if (a[index]) {
            pending[a[index] - 1].info = info;
            return;
        }
        pending.push_back({binding, index, info});
        a[index] = std::uint32_t(pending.size());
    }
    void clear() {
        for (const Pending& p : pending) at[p.binding][p.index] = 0;
        pending.clear();
    }
};

std::uint32_t old_allocate(std::vector<std::uint8_t>& used, std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t i = begin; i < end; ++i) {
        if (used[i]) continue;
        used[i] = 1;
        return i;
    }
    return 0;
}

std::uint32_t new_allocate(std::vector<std::uint8_t>& used, std::uint32_t begin, std::uint32_t end) {
    if (begin >= end) return 0;
    const void* found = std::memchr(used.data() + begin, 0, end - begin);
    if (!found) return 0;
    const auto i = std::uint32_t(static_cast<const std::uint8_t*>(found) - used.data());
    used[i] = 1;
    return i;
}

bool same(const VkDescriptorImageInfo& a, const VkDescriptorImageInfo& b) {
    return a.sampler == b.sampler && a.imageView == b.imageView && a.imageLayout == b.imageLayout;
}

VkImageView view(std::uint64_t n) {
    return reinterpret_cast<VkImageView>(static_cast<std::uintptr_t>((1ull << 40) + n));
}
VkSampler sampler_handle(std::uint64_t n) {
    return reinterpret_cast<VkSampler>(static_cast<std::uintptr_t>((1ull << 41) + n));
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// ---- equivalence of the extracted algorithms ------------------------------------------------------------------

void check_queue_equivalence() {
    const BindlessFeatures f = features(true, 0);
    // Fixed: repeats of a slot keep its first place and take the last info; clear() forgets everything.
    {
        OldQueue o;
        NewQueue n(f);
        const VkDescriptorImageInfo a{VK_NULL_HANDLE, view(1), VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorImageInfo b{VK_NULL_HANDLE, view(2), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        o.queue(0, 5, a), n.queue(0, 5, a);
        o.queue(3, 5, a), n.queue(3, 5, a);
        o.queue(0, 5, b), n.queue(0, 5, b);
        o.queue(0, 6, b), n.queue(0, 6, b);
        EXPECT(o.pending.size() == 3 && n.pending.size() == 3);
        EXPECT(n.pending[0].binding == 0 && n.pending[0].index == 5 && same(n.pending[0].info, b));
        EXPECT(n.pending[1].binding == 3 && same(n.pending[1].info, a));
        o.clear(), n.clear();
        o.queue(0, 5, a), n.queue(0, 5, a);
        EXPECT(n.pending.size() == 1 && same(n.pending[0].info, a));
        // An index past the binding's count (not reached by BindlessSet) still gets one place.
        o.queue(1, 100000, a), n.queue(1, 100000, a);
        o.queue(1, 100000, b), n.queue(1, 100000, b);
        EXPECT(o.pending.size() == n.pending.size());
        EXPECT(same(n.pending.back().info, b));
    }
    // Randomized: 200 rounds of up to 4000 queues each over random bindings and indices, with many repeats.
    std::mt19937_64 rng(0x5eed);
    for (int round = 0; round < 200; ++round) {
        OldQueue o;
        NewQueue n(f);
        for (int flush = 0; flush < 4; ++flush) {
            const int queues = int(rng() % 4000);
            const std::uint32_t spread = 1 + std::uint32_t(rng() % 256);
            for (int i = 0; i < queues; ++i) {
                std::uint32_t binding = std::uint32_t(rng() % bindless::binding_count);
                if (binding == bindless::table_binding) binding = 0;
                const std::uint32_t index = std::uint32_t(rng() % std::min(spread, f.counts[binding]));
                const VkDescriptorImageInfo info{sampler_handle(rng() % 7), view(rng() % 1000),
                                                 VkImageLayout(rng() % 8)};
                o.queue(binding, index, info);
                n.queue(binding, index, info);
            }
            EXPECT(o.pending.size() == n.pending.size());
            for (std::size_t i = 0; i < o.pending.size(); ++i) {
                EXPECT(o.pending[i].binding == n.pending[i].binding);
                EXPECT(o.pending[i].index == n.pending[i].index);
                EXPECT(same(o.pending[i].info, n.pending[i].info));
            }
            o.clear();
            n.clear();
        }
        for (const auto& a : n.at)
            for (const std::uint32_t v : a) EXPECT(v == 0);  // clear() left nothing behind
    }
}

void check_allocate_equivalence() {
    // Fixed: empty range, full range, first free, range bounds.
    {
        std::vector<std::uint8_t> a(16, 1), b(16, 1);
        EXPECT(old_allocate(a, 1, 16) == 0 && new_allocate(b, 1, 16) == 0);
        EXPECT(old_allocate(a, 5, 5) == 0 && new_allocate(b, 5, 5) == 0);
        EXPECT(old_allocate(a, 9, 3) == 0 && new_allocate(b, 9, 3) == 0);
        a[7] = b[7] = 0;
        a[12] = b[12] = 0;
        EXPECT(old_allocate(a, 8, 16) == 12 && new_allocate(b, 8, 16) == 12);
        EXPECT(old_allocate(a, 1, 16) == 7 && new_allocate(b, 1, 16) == 7);
        EXPECT(a == b);
    }
    // Randomized: sizes up to 70000, random fill, random ranges, alternating allocations and frees.
    std::mt19937_64 rng(0xa110c);
    for (int round = 0; round < 300; ++round) {
        const std::uint32_t size = 1 + std::uint32_t(rng() % 70000);
        std::vector<std::uint8_t> a(size);
        const unsigned density = unsigned(rng() % 101);
        for (auto& v : a) v = std::uint8_t(rng() % 100 < density);
        std::vector<std::uint8_t> b = a;
        for (int op = 0; op < 400; ++op) {
            if (rng() % 3 == 0) {
                const std::uint32_t i = std::uint32_t(rng() % size);
                a[i] = b[i] = 0;
                continue;
            }
            std::uint32_t begin = std::uint32_t(rng() % (size + 1)), end = std::uint32_t(rng() % (size + 1));
            if (rng() % 2) begin = std::min<std::uint32_t>(1, size);
            if (rng() % 2) end = size;
            EXPECT(old_allocate(a, begin, end) == new_allocate(b, begin, end));
        }
        EXPECT(a == b);
    }
}

// ---- the whole class, through the fake device -------------------------------------------------------------------

struct Model {  // what BindlessSet promises about indices: the lowest free one, storage-capable ones first
    std::array<std::vector<std::uint8_t>, 6> used;
    std::vector<std::uint8_t> samplers;
};

// Random add/update/remove of images and samplers with a flush every few operations. Each in-place flush must hold
// one write per binding and index; the indices must be what the former byte loop would have returned.
std::uint64_t drive(bool ms, std::uint32_t images_2d, std::uint64_t seed, int frames, int ops_per_frame,
                    bool insert_keys, bool check) {
    Fake fake;
    g_fake = &fake;
    fake.record = true;
    const BindlessFeatures f = features(ms, images_2d);
    std::uint64_t digest = 0;
    {
        BindlessSet set(&fake_get, fake_device(), fake_memory(), f);
        EXPECT(set.ok());
        Model model;
        for (std::uint32_t d = 0; d < 6; ++d) {
            model.used[d].assign(set.image_capacity(ImageDimension(d)), 0);
            model.used[d][0] = 1;
        }
        model.samplers.assign(set.sampler_capacity(), 0);
        model.samplers[0] = 1;
        std::array<std::vector<std::uint32_t>, 6> live;
        std::vector<std::uint32_t> live_samplers;
        std::mt19937_64 rng(seed);
        std::uint64_t next_view = 1;
        std::uint32_t key_counter = 0;
        std::vector<DescriptorKey> keys;
        for (int frame = 0; frame < frames; ++frame) {
            fake.updates.clear();
            for (int op = 0; op < ops_per_frame; ++op) {
                const auto dim = ImageDimension(rng() % (ms ? 6 : 4));
                const std::uint32_t d = std::uint32_t(dim);
                const unsigned kind = unsigned(rng() % 10);
                if (kind < 5) {
                    StorageViews storage;
                    if (rng() % 2) storage.as_float = view(next_view++);
                    if (rng() % 3 == 0) storage.as_uint = view(next_view++);
                    if (rng() % 4 == 0) storage.r32ui = view(next_view++);
                    const std::uint32_t cls = std::uint32_t(rng() % 3);
                    const std::uint32_t index =
                        set.add_image(dim, rng() % 8 ? view(next_view++) : VK_NULL_HANDLE,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, storage, cls);
                    // The former allocation steps, on the model.
                    std::uint32_t expect = 0;
                    const bool stores = storage.as_float || storage.as_uint || storage.as_sint || storage.r32ui;
                    if (!(bindless::multisampled(dim) && (stores || !set.supports_ms_class(cls)))) {
                        auto& used = model.used[d];
                        const std::uint32_t low_end =
                            std::min(std::uint32_t(used.size()), set.storage_capacity(dim));
                        expect = stores ? 0 : old_allocate(used, low_end, std::uint32_t(used.size()));
                        if (!expect) expect = old_allocate(used, 1, low_end);
                    }
                    EXPECT(index == expect);
                    fake.mix(index);
                    if (index) live[d].push_back(index);
                    if (index && insert_keys && rng() % 4 == 0) {
                        DescriptorKey key{};
                        key[0] = ++key_counter;
                        key[1] = 0xabc00000u | d;
                        set.insert_key(key, index);
                        keys.push_back(key);
                    }
                } else if (kind < 7 && !live[d].empty()) {
                    const std::size_t at = std::size_t(rng() % live[d].size());
                    set.update_image(dim, live[d][at], view(next_view++), VK_IMAGE_LAYOUT_GENERAL);
                } else if (kind < 9 && !live[d].empty()) {
                    const std::size_t at = std::size_t(rng() % live[d].size());
                    set.remove_image(dim, live[d][at]);
                    model.used[d][live[d][at]] = 0;
                    live[d][at] = live[d].back();
                    live[d].pop_back();
                } else if (rng() % 2 || live_samplers.empty()) {
                    const std::uint32_t index = set.add_sampler(sampler_handle(next_view++));
                    EXPECT(index == old_allocate(model.samplers, 1, std::uint32_t(model.samplers.size())));
                    fake.mix(index);
                    if (index) live_samplers.push_back(index);
                } else {
                    const std::size_t at = std::size_t(rng() % live_samplers.size());
                    set.remove_sampler(live_samplers[at]);
                    model.samplers[live_samplers[at]] = 0;
                    live_samplers[at] = live_samplers.back();
                    live_samplers.pop_back();
                }
            }
            if (insert_keys && !keys.empty() && rng() % 3 == 0) {
                set.erase_key(keys.back());
                keys.pop_back();
            }
            const bool replaces = set.flush_replaces_table();
            EXPECT(set.flush(fake_command_buffer(), std::uint64_t(frame) + 1));
            set.table_upload_result(true);
            set.release_retired(std::uint64_t(frame));
            if (check && !replaces) {
                // In place: no binding and index written twice in one flush.
                std::vector<std::uint64_t> seen;
                for (const auto& update : fake.updates)
                    for (const Write& w : update) seen.push_back((std::uint64_t(w.binding) << 32) | w.element);
                std::sort(seen.begin(), seen.end());
                EXPECT(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
            }
            for (std::uint32_t dd = 0; dd < 6; ++dd)
                EXPECT(set.free_images(ImageDimension(dd)) ==
                       std::uint32_t(std::count(model.used[dd].begin(), model.used[dd].end(), std::uint8_t(0))));
        }
        set.release_retired();
    }
    digest = fake.digest;
    g_fake = nullptr;
    return digest;
}

// A frame of texture streaming: each frame adds `per_frame` images with storage views and removes as many, then
// flushes. Returns seconds per frame; *allocations gets the heap allocations per frame.
double bench_set(std::uint32_t images_2d, int frames, int per_frame, double* allocations) {
    Fake fake;
    g_fake = &fake;
    fake.record = false;
    double per = 0;
    {
        BindlessSet set(&fake_get, fake_device(), fake_memory(), features(false, images_2d));
        EXPECT(set.ok());
        // Fill most of the 2D array so the free-index search has to walk.
        std::vector<std::uint32_t> resident;
        const std::uint32_t fill = set.image_capacity(ImageDimension::dim_2d) * 7 / 8;
        for (std::uint32_t i = 0; i < fill; ++i)
            resident.push_back(set.add_image(ImageDimension::dim_2d, view(i + 1), VK_IMAGE_LAYOUT_GENERAL));
        set.flush(fake_command_buffer(), 1);
        set.table_upload_result(true);
        std::mt19937_64 rng(77);
        std::uint64_t next_view = 1u << 24;
        std::vector<std::uint32_t> streamed;
        const auto frame = [&]() {
            for (int i = 0; i < per_frame; ++i) {
                StorageViews storage;
                storage.as_float = view(next_view++);
                storage.as_uint = view(next_view++);
                streamed.push_back(set.add_image(ImageDimension::dim_2d, view(next_view++),
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, storage));
                set.update_image(ImageDimension::dim_2d, resident[rng() % resident.size()], view(next_view++),
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
            set.flush(fake_command_buffer(), 2);
            for (const std::uint32_t index : streamed) set.remove_image(ImageDimension::dim_2d, index);
            streamed.clear();
            set.flush(fake_command_buffer(), 2);
        };
        frame();  // warm-up: the reused vectors grow here
        const std::uint64_t before = g_allocations.load();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < frames; ++i) frame();
        per = seconds_since(start) / frames;
        *allocations = double(g_allocations.load() - before) / frames;
    }
    g_fake = nullptr;
    return per;
}

template <typename Queue>
double bench_queue(Queue& q, int frames, int per_frame, double* allocations) {
    std::mt19937_64 rng(9);
    std::vector<std::uint32_t> indices(static_cast<std::size_t>(per_frame));
    for (auto& i : indices) i = 1 + std::uint32_t(rng() % 4000);
    const VkDescriptorImageInfo info{VK_NULL_HANDLE, view(3), VK_IMAGE_LAYOUT_GENERAL};
    const auto frame = [&]() {
        for (const std::uint32_t index : indices) {  // write_image on a 2D index: sampled, 3 storage, r32ui
            q.queue(0, index, info);
            q.queue(4, index % 1024, info);
            q.queue(13, index % 1024, info);
            q.queue(14, index % 1024, info);
            q.queue(16, index % 1024, info);
        }
        q.clear();
    };
    frame();
    const std::uint64_t before = g_allocations.load();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) frame();
    const double per = seconds_since(start) / frames;
    *allocations = double(g_allocations.load() - before) / frames;
    return per;
}

template <typename Allocate>
double bench_allocate(Allocate allocate, std::uint32_t size, int rounds, std::uint32_t* checksum) {
    std::vector<std::uint8_t> used(size, 1);
    std::mt19937_64 rng(11);
    std::vector<std::uint32_t> frees(64);
    std::uint32_t sum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < rounds; ++r) {
        for (auto& f : frees) f = std::uint32_t(size / 2 + rng() % (size / 2));  // free indices high in the array
        for (const std::uint32_t f : frees) used[f] = 0;
        for (std::size_t i = 0; i < frees.size(); ++i) sum += allocate(used, 1, size);
    }
    *checksum = sum;
    return seconds_since(start) / (double(rounds) * double(frees.size()));
}

} // namespace

int main(int argc, char** argv) {
    const bool long_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;

    check_queue_equivalence();
    check_allocate_equivalence();
    std::printf("extracted queue and allocate: old == new on fixed and randomized inputs\n");

    std::uint64_t digest = 0;
    struct Case {
        bool ms;
        std::uint32_t images_2d;
        bool keys;
    };
    const Case cases[] = {{false, 0, false}, {true, 0, false}, {false, 0, true}, {true, 0, true}, {false, 512, true}};
    for (std::uint64_t seed = 1; seed <= 6; ++seed)
        for (const Case& c : cases) digest ^= drive(c.ms, c.images_2d, seed * 7919, 60, 120, c.keys, true) + seed;
    std::printf("BindlessSet through the fake device: indices and one write per slot hold; digest %016llx\n",
                static_cast<unsigned long long>(digest));

    const int scale = long_bench ? 10 : 1;
    double old_allocs = 0, new_allocs = 0;
    OldQueue oq;
    NewQueue nq(features(false, 0));
    const double old_q = bench_queue(oq, 200 * scale, 512, &old_allocs);
    const double new_q = bench_queue(nq, 200 * scale, 512, &new_allocs);
    std::printf("queue, 512 images x 5 writes + clear: old %.1f us (%.0f allocs), new %.1f us (%.0f allocs): %.2fx\n",
                old_q * 1e6, old_allocs, new_q * 1e6, new_allocs, old_q / new_q);

    std::uint32_t old_sum = 0, new_sum = 0;
    const double old_a = bench_allocate(old_allocate, 65536, 50 * scale, &old_sum);
    const double new_a = bench_allocate(new_allocate, 65536, 50 * scale, &new_sum);
    EXPECT(old_sum == new_sum);
    std::printf("allocate, 65536 indices, free ones in the upper half: old %.1f ns, new %.1f ns: %.2fx\n",
                old_a * 1e9, new_a * 1e9, old_a / new_a);

    double set_allocs = 0;
    const double set_frame = bench_set(65536, 100 * scale, 256, &set_allocs);
    std::printf("BindlessSet frame, 65536 2D images 7/8 full, 256 streamed + 256 updated: %.1f us, %.1f allocs\n",
                set_frame * 1e6, set_allocs);
    double set_allocs_default = 0;
    const double set_frame_default = bench_set(0, 100 * scale, 256, &set_allocs_default);
    std::printf("BindlessSet frame, default 4096 2D images 7/8 full, same frame: %.1f us, %.1f allocs\n",
                set_frame_default * 1e6, set_allocs_default);
    return 0;
}
