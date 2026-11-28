#include "../src/vulkan_memory_budget.h"

#include <cstdint>
#include <cstdio>
#include <limits>

#define EXPECT(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

int main() {
    using namespace leonida;
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

    // Basic multi-heap sum.
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryHeapCount = 3;
    memory.memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[1].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    memory.memoryHeaps[2].flags = 0;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.heapUsage[0] = 100;
    budget.heapBudget[0] = 500;
    budget.heapUsage[1] = 200;
    budget.heapBudget[1] = 600;
    budget.heapUsage[2] = 50;
    const VramBudget normal = sum_vram_budget(memory, budget);
    EXPECT(normal.available && normal.local_usage == 300 && normal.local_budget == 1100 && normal.shared_usage == 50);

    // Regression: two DEVICE_LOCAL heaps whose budgets wrap uint64_t to 0 must saturate to UINT64_MAX
    // and remain available rather than reporting "VRAM n/a".
    VkPhysicalDeviceMemoryBudgetPropertiesEXT wrap{};
    wrap.heapBudget[0] = kMax;
    wrap.heapBudget[1] = 1;
    wrap.heapUsage[0] = kMax - 10;
    wrap.heapUsage[1] = 50;
    memory.memoryHeapCount = 4;
    memory.memoryHeaps[3].flags = 0;
    wrap.heapUsage[2] = kMax - 5;
    wrap.heapUsage[3] = 20;

    const VramBudget saturated = sum_vram_budget(memory, wrap);
    EXPECT(saturated.available);
    EXPECT(saturated.local_budget == kMax);
    EXPECT(saturated.local_usage == kMax);
    EXPECT(saturated.shared_usage == kMax);

    // Null guard on query_vram_budget.
    EXPECT(!query_vram_budget(nullptr, VK_NULL_HANDLE).available);

    std::puts("vulkan_memory_budget_test: ok");
    return 0;
}
