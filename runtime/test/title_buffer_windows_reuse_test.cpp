// rt::d3d12::TitleBufferWindows (runtime/shims/d3d12_title_buffer_windows.h): reuse of earlier blocks and of the
// lock-free readers' slots. A fake Vulkan device gives the blocks host memory. No game data, no GPU.
//   1. a request that is not device_local (the device's title windows) reuses the spare block free() keeps when it is
//      not the last block, and a freed hole in an earlier block, instead of making a new block;
//   2. blocks made after 4,096 destroyed ones over the pool's lifetime still reach for_each_block and find_mapped.
#include <windows.h>

#include <cstdint>
#include <cstdio>

#include "../shims/d3d12_title_buffer_windows.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

// The Vulkan calls TitleBufferWindows::allocate makes, over host memory (as in geometry_views_test).
struct FakeBuffer {
    VkDeviceSize size = 0;
    void* memory = nullptr;
};
VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo* info, const VkAllocationCallbacks*,
                                             VkBuffer* out) {
    *out = reinterpret_cast<VkBuffer>(new FakeBuffer{info->size});
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL requirements(VkDevice, VkBuffer buffer, VkMemoryRequirements* out) {
    *out = {};
    out->size = reinterpret_cast<FakeBuffer*>(buffer)->size;
    out->alignment = 65536;
    out->memoryTypeBits = 0xf;
}
VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize) {
    reinterpret_cast<FakeBuffer*>(buffer)->memory = memory;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice, VkDeviceMemory memory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                   void** out) {
    *out = memory;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL unmap(VkDevice, VkDeviceMemory) {}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL address(VkDevice, const VkBufferDeviceAddressInfo* info) {
    return VkDeviceAddress(reinterpret_cast<std::uintptr_t>(reinterpret_cast<FakeBuffer*>(info->buffer)->memory));
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    delete reinterpret_cast<FakeBuffer*>(buffer);
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    VirtualFree(memory, 0, MEM_RELEASE);
}

struct FakeDevice {
    struct {
        PFN_vkCreateBuffer vkCreateBuffer = create_buffer;
        PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = requirements;
        PFN_vkBindBufferMemory vkBindBufferMemory = bind;
        PFN_vkMapMemory vkMapMemory = map;
        PFN_vkUnmapMemory vkUnmapMemory = unmap;
        PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress = address;
        PFN_vkDestroyBuffer vkDestroyBuffer = destroy_buffer;
        PFN_vkFreeMemory vkFreeMemory = free_memory;
    } vk_;
    struct {
        decltype(vk_)& vk;
    } context_{vk_};
    decltype(context_)* context = &context_;
    struct {
        VkDevice device = VK_NULL_HANDLE;
    } info;
    struct {
        VkDeviceAddress address = 0;
    } arena;
    void share(VkBufferCreateInfo&) {}
    VkDeviceMemory allocate(const VkMemoryRequirements& r, VkMemoryPropertyFlags required,
                            VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags* chosen, bool) {
        if (chosen) *chosen = required | preferred;
        return reinterpret_cast<VkDeviceMemory>(VirtualAlloc(nullptr, r.size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    }
};

constexpr VkDeviceSize kMiB = 1ull << 20;

std::size_t published_blocks(const rt::d3d12::TitleBufferWindows& pool) {
    std::size_t n = 0;
    pool.for_each_block([&](VkDeviceAddress, VkDeviceSize) { ++n; });
    return n;
}

void test_earlier_blocks_are_reused_by_title_windows() {
    FakeDevice d;
    rt::d3d12::TitleBufferWindows pool;
    VkDeviceSize a = 0, b = 0, c = 0, e = 0;
    std::uint8_t *ma = nullptr, *mb = nullptr, *mc = nullptr, *me = nullptr;
    // The device's title windows: device_local is false (DeviceState's external_windows.allocate).
    check(pool.allocate(d, 3 * kMiB, a, ma) && pool.block_count() == 1, "3 MiB in block 0");
    check(pool.allocate(d, 3 * kMiB, b, mb) && pool.block_count() == 2, "3 MiB more does not fit: block 1");
    check(pool.free(d, a, 3 * kMiB) && pool.block_count() == 2 && pool.empty_blocks() == 1,
          "freeing block 0's range keeps it as the spare");
    check(pool.allocate(d, 2 * kMiB, c, mc), "a 2 MiB title window is allocated");
    check(pool.block_count() == 2, "it reuses the spare block 0, no third block is made");
    check(mc == ma, "at the start of block 0");

    // A freed hole in an earlier, still live block.
    check(pool.allocate(d, 2 * kMiB, e, me) && pool.block_count() == 2 && me == ma + 2 * kMiB,
          "2 MiB more does not fit block 1's tail and goes after it in block 0");
    check(pool.free(d, c, 2 * kMiB) && pool.empty_blocks() == 0,
          "the first 2 MiB of block 0 are freed; its other window keeps it live");
    VkDeviceSize f = 0;
    std::uint8_t* mf = nullptr;
    check(pool.allocate(d, 2 * kMiB, f, mf) && pool.block_count() == 2, "a 2 MiB window takes the freed hole");
    check(mf == ma, "the hole at the start of block 0");
    pool.release(d);
}

void test_blocks_after_4096_destroyed_ones_are_published() {
    FakeDevice d;
    rt::d3d12::TitleBufferWindows pool;
    // Under a 1 MiB cap a block is 1 MiB, not the standard 4 MiB, so free() destroys it rather than keeping a spare.
    constexpr VkDeviceSize cap = kMiB;
    VkDeviceSize offset = 0;
    std::uint8_t* mapped = nullptr;
    bool made_all = true, freed_all = true;
    for (int i = 0; i < 4100; ++i) {
        made_all = made_all && pool.allocate(d, 64 * 1024, offset, mapped, false, cap) && pool.block_count() == 1;
        freed_all = freed_all && pool.free(d, offset, 64 * 1024) && pool.block_count() == 0;
    }
    check(made_all && freed_all, "4,100 blocks are made and destroyed one after another");
    check(pool.allocate(d, 64 * 1024, offset, mapped, false, cap) && pool.block_count() == 1, "one more block is live");
    check(published_blocks(pool) == 1, "for_each_block sees it");
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize at = 0;
    check(pool.find_mapped(mapped + 16, 16, buffer, at) && buffer != VK_NULL_HANDLE && at == 16,
          "find_mapped finds it");
    pool.release(d);
    check(published_blocks(pool) == 0, "release empties the published blocks");
}

}  // namespace

int main() {
    test_earlier_blocks_are_reused_by_title_windows();
    test_blocks_after_4096_destroyed_ones_are_published();
    std::printf("title_buffer_windows_reuse_test: %s (%d checks, %d failures)\n", g_failures ? "FAILED" : "ok",
                g_checks, g_failures);
    return g_failures ? 1 : 0;
}
