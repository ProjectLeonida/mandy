// Portable pure-CPU unit tests for runtime/shims/d3d12_title_buffer_windows.h (TitleBufferWindows):
//   - zero-byte refusal and cap-underflow guard when reserved() > cap;
//   - rejection of device addresses below arena.address (preventing 64-bit bias wrap into [0, 2^47));
//   - O(log B) lookup() over out-of-order block device addresses, boundaries, and misses;
//   - most-recent-first tail reuse across earlier slabs (including after hitting cap) and pruning of full slabs;
//   - clean release() and reuse of the same TitleBufferWindows instance.
// No GPU, Vulkan loader, Windows headers, or game files required.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../shims/d3d12_title_buffer_windows.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

struct FakeBuf {
    VkDeviceSize size = 0;
    VkDeviceAddress bda = 0;
    std::uint8_t* storage = nullptr;
};

struct MockDevice {
    std::vector<VkDeviceAddress> next_bdas;
    std::size_t bda_cursor = 0;
    VkDeviceAddress bda_step_base = 0x200000000ull;
    int create_calls = 0;
    int destroy_calls = 0;
    int unmap_calls = 0;
    int free_calls = 0;
    VkDeviceSize req_extra = 0;

    static MockDevice*& active() {
        static MockDevice* d = nullptr;
        return d;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL create_buf(VkDevice, const VkBufferCreateInfo* info,
                                                     const VkAllocationCallbacks*, VkBuffer* out) {
        ++active()->create_calls;
        auto* b = new FakeBuf{info->size, 0, nullptr};
        *out = reinterpret_cast<VkBuffer>(b);
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL mem_req(VkDevice, VkBuffer buffer, VkMemoryRequirements* out) {
        *out = {};
        out->size = reinterpret_cast<FakeBuf*>(buffer)->size + active()->req_extra;
        out->alignment = 65536;
        out->memoryTypeBits = 1;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL bind_buf(VkDevice, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize) {
        reinterpret_cast<FakeBuf*>(buffer)->storage = reinterpret_cast<std::uint8_t*>(memory);
        return VK_SUCCESS;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL map_mem(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize,
                                                  VkMemoryMapFlags, void** out) {
        *out = reinterpret_cast<void*>(memory);
        return VK_SUCCESS;
    }
    static VKAPI_ATTR void VKAPI_CALL unmap_mem(VkDevice, VkDeviceMemory) { ++active()->unmap_calls; }
    static VKAPI_ATTR VkDeviceAddress VKAPI_CALL get_bda(VkDevice, const VkBufferDeviceAddressInfo* info) {
        auto* b = reinterpret_cast<FakeBuf*>(info->buffer);
        auto* d = active();
        if (d->bda_cursor < d->next_bdas.size()) {
            b->bda = d->next_bdas[d->bda_cursor++];
        } else {
            b->bda = d->bda_step_base;
            d->bda_step_base += 0x10000000ull;
        }
        return b->bda;
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_buf(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
        ++active()->destroy_calls;
        delete reinterpret_cast<FakeBuf*>(buffer);
    }
    static VKAPI_ATTR void VKAPI_CALL free_mem(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
        ++active()->free_calls;
        std::free(reinterpret_cast<void*>(memory));
    }

    struct VkTable {
        PFN_vkCreateBuffer vkCreateBuffer = create_buf;
        PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = mem_req;
        PFN_vkBindBufferMemory vkBindBufferMemory = bind_buf;
        PFN_vkMapMemory vkMapMemory = map_mem;
        PFN_vkUnmapMemory vkUnmapMemory = unmap_mem;
        PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = get_bda;
        PFN_vkDestroyBuffer vkDestroyBuffer = destroy_buf;
        PFN_vkFreeMemory vkFreeMemory = free_mem;
    } vk_;
    struct Ctx {
        VkTable& vk;
    } ctx_{vk_};
    Ctx* context = &ctx_;
    struct {
        VkDevice device = VK_NULL_HANDLE;
    } info;
    struct {
        VkDeviceAddress address = 0x100000000ull;
    } arena;

    void share(VkBufferCreateInfo&) {}
    VkDeviceMemory allocate(const VkMemoryRequirements& r, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags* chosen, bool) {
        if (chosen) *chosen = required | preferred;
        // Allocate small backing only when size is small, else allocate exact requested bytes (all tests use <= 8 MiB).
        return reinterpret_cast<VkDeviceMemory>(std::malloc(static_cast<std::size_t>(r.size)));
    }
};

}  // namespace

int main() {
    using rt::d3d12::TitleBufferWindows;

    // 1. Zero-byte request is refused without touching Vulkan.
    {
        MockDevice dev;
        MockDevice::active() = &dev;
        TitleBufferWindows win;
        VkDeviceSize off = 0;
        std::uint8_t* ptr = nullptr;
        CHECK(!win.allocate(dev, 0, off, ptr, true));
        CHECK(dev.create_calls == 0 && win.reserved() == 0);
    }

    // 2. Cap underflow guard: when reserved() > cap, new slab must be refused with *at_cap = true.
    {
        MockDevice dev;
        MockDevice::active() = &dev;
        TitleBufferWindows win;
        VkDeviceSize off = 0;
        std::uint8_t* ptr = nullptr;
        CHECK(win.allocate(dev, 3ull << 20, off, ptr, true, 8ull << 20));
        CHECK(win.reserved() == (4ull << 20));
        // Lower cap below reserved_ (e.g. cap = 2 MiB < 4 MiB reserved) and request 2 MiB (doesn't fit 1 MiB tail).
        bool at_cap = false;
        CHECK(!win.allocate(dev, 2ull << 20, off, ptr, true, 2ull << 20, &at_cap));
        CHECK(at_cap);
        CHECK(dev.create_calls == 1);
        win.release(dev);
        CHECK(dev.destroy_calls == 1 && dev.unmap_calls == 1 && dev.free_calls == 1 && win.reserved() == 0);
    }

    // 3. Reject BDA below arena.address even when (bda - arena.address) would wrap modulo 2^64 into [0, 2^47).
    {
        MockDevice dev;
        MockDevice::active() = &dev;
        dev.arena.address = 0xffffffff00000000ull;  // high VA: 0x1000 - arena wraps to 0x100001000 < 2^47!
        dev.next_bdas = {0x1000ull};
        TitleBufferWindows win;
        VkDeviceSize off = 0;
        std::uint8_t* ptr = nullptr;
        CHECK(!win.allocate(dev, 4096, off, ptr, true));
        CHECK(win.reserved() == 0 && dev.destroy_calls == 1 && dev.unmap_calls == 1 && dev.free_calls == 1);
    }

    // 4. Out-of-order block BDAs: O(log B) lookup() finds exact host pointer and remaining bytes, rejects misses.
    {
        MockDevice dev;
        MockDevice::active() = &dev;
        dev.arena.address = 0x100000000ull;
        dev.next_bdas = {0x500000000ull, 0x200000000ull, 0x800000000ull, 0x300000000ull};
        TitleBufferWindows win;
        struct Rec {
            VkDeviceAddress bda;
            std::uint8_t* ptr;
            VkDeviceSize len;
        };
        std::vector<Rec> recs;
        for (int i = 0; i < 4; ++i) {
            VkDeviceSize off = 0;
            std::uint8_t* ptr = nullptr;
            const VkDeviceSize sz = (3ull << 20) + VkDeviceSize(i) * 4096;
            CHECK(win.allocate(dev, sz, off, ptr, false));
            recs.push_back({dev.arena.address + off, ptr, sz});
        }
        std::uint64_t host = 0, rem = 0;
        for (const auto& r : recs) {
            CHECK(win.lookup(r.bda, host, rem));
            CHECK(host == reinterpret_cast<std::uintptr_t>(r.ptr) && rem == r.len);
            CHECK(win.lookup(r.bda + r.len - 1, host, rem));
            CHECK(host == reinterpret_cast<std::uintptr_t>(r.ptr + r.len - 1) && rem == 1);
            CHECK(!win.lookup(r.bda + r.len, host, rem));
        }
        CHECK(!win.lookup(0x100000000ull, host, rem));
        CHECK(!win.lookup(0x900000000ull, host, rem));
        win.release(dev);
        CHECK(dev.destroy_calls == 4 && dev.free_calls == 4);
    }

    // 5. Tail reuse across earlier slabs in most-recent-first order, even after budget cap is reached,
    //    plus pruning of exhausted tails.
    {
        MockDevice dev;
        MockDevice::active() = &dev;
        dev.arena.address = 0x100000000ull;
        dev.next_bdas = {0x200000000ull, 0x300000000ull};
        TitleBufferWindows win;
        VkDeviceSize off0 = 0, off1 = 0, off2 = 0, off3 = 0;
        std::uint8_t* ptr = nullptr;
        // Slab 0: take 3.5 MiB -> 512 KiB tail left.
        CHECK(win.allocate(dev, (3ull << 20) + (512ull << 10), off0, ptr, true, 8ull << 20));
        // Slab 1: request 3.75 MiB (doesn't fit slab 0's 512 KiB tail) -> new slab 1, 256 KiB tail left.
        CHECK(win.allocate(dev, (3ull << 20) + (768ull << 10), off1, ptr, true, 8ull << 20));
        CHECK(dev.create_calls == 2 && win.reserved() == (8ull << 20));
        // Now at cap (8 MiB): request 384 KiB -> skips slab 1 (only 256 KiB free), reuses slab 0's 512 KiB tail!
        bool at_cap = false;
        CHECK(win.allocate(dev, 384ull << 10, off2, ptr, true, 8ull << 20, &at_cap));
        CHECK(!at_cap && dev.create_calls == 2);
        CHECK(off2 == (0x200000000ull - dev.arena.address) + (3ull << 20) + (512ull << 10));
        // Fill remaining 128 KiB of slab 0 so its tail is completely exhausted and pruned.
        CHECK(win.allocate(dev, 128ull << 10, off3, ptr, true, 8ull << 20, &at_cap));
        // Slab 1 fits 128 KiB first (blocks_.back() is checked before open_tails_) -> fill slab 1's 256 KiB first.
        CHECK(off3 == (0x300000000ull - dev.arena.address) + (3ull << 20) + (768ull << 10));
        CHECK(win.allocate(dev, 128ull << 10, off3, ptr, true, 8ull << 20, &at_cap));  // exhausts slab 1
        CHECK(win.allocate(dev, 128ull << 10, off3, ptr, true, 8ull << 20, &at_cap));  // exhausts slab 0 tail!
        CHECK(off3 == (0x200000000ull - dev.arena.address) + (3ull << 20) + (896ull << 10));
        // Now both slabs are 100% full and cap is reached: even 16 bytes must report at_cap = true.
        CHECK(!win.allocate(dev, 16, off3, ptr, true, 8ull << 20, &at_cap));
        CHECK(at_cap);
        win.release(dev);
        CHECK(dev.destroy_calls == 2 && dev.free_calls == 2 && win.reserved() == 0);
        // Instance is reusable after release().
        dev.bda_cursor = 0;
        CHECK(win.allocate(dev, 64, off0, ptr, true, 8ull << 20));
        win.release(dev);
    }

    std::puts("title_buffer_windows_test: all checks passed");
    return 0;
}
