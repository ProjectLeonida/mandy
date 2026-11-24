// LEONIDA_MEM_PROBE (src/mem_probe.h), off and on, on a stand-in function table: no GPU, no game data.
//
//   mem_probe_test         the variable unset: the table is left alone, no thread starts, nothing is written
//   mem_probe_test --on    LEONIDA_MEM_PROBE=1: the nine entries are wrapped and still reach the driver, the counts
//                          follow what was made and destroyed, and the lines say so
//
// Both forms make the same calls through the table: three allocations (one imported), one freed, one mapped; two
// shader modules, one destroyed; three graphics pipelines and one compute pipeline, one graphics pipeline destroyed.
// stderr is sent to a file while they run.
#include "../src/mem_probe.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <iterator>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::printf("line %d: %s\n", __LINE__, #condition); std::fflush(stdout); return 1; } \
} while (0)

namespace {

namespace probe = leonida::mem_probe;

// The stand-in driver: every call is counted and hands out the next handle.
int g_calls = 0;
std::uint64_t g_next = 0x1000;
template<class Handle>
Handle next_handle() { return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(g_next += 0x10)); }

VKAPI_ATTR VkResult VKAPI_CALL fake_allocate(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                                             VkDeviceMemory* out) {
    ++g_calls;
    *out = next_handle<VkDeviceMemory>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_free(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++g_calls; }
char g_mapped[4096];
VKAPI_ATTR VkResult VKAPI_CALL fake_map(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags,
                                        void** data) {
    ++g_calls;
    *data = g_mapped;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_unmap(VkDevice, VkDeviceMemory) { ++g_calls; }
VKAPI_ATTR VkResult VKAPI_CALL fake_create_module(VkDevice, const VkShaderModuleCreateInfo*,
                                                  const VkAllocationCallbacks*, VkShaderModule* out) {
    ++g_calls;
    *out = next_handle<VkShaderModule>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) { ++g_calls; }
VKAPI_ATTR VkResult VKAPI_CALL fake_create_graphics(VkDevice, VkPipelineCache, std::uint32_t count,
                                                    const VkGraphicsPipelineCreateInfo*, const VkAllocationCallbacks*,
                                                    VkPipeline* out) {
    ++g_calls;
    for (std::uint32_t i = 0; i < count; ++i) out[i] = next_handle<VkPipeline>();
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_create_compute(VkDevice, VkPipelineCache, std::uint32_t count,
                                                   const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                   VkPipeline* out) {
    ++g_calls;
    for (std::uint32_t i = 0; i < count; ++i) out[i] = next_handle<VkPipeline>();
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) { ++g_calls; }

// The nine entries install() wraps, named as in leonida::VulkanDeviceFunctions.
struct Table {
    PFN_vkAllocateMemory vkAllocateMemory = fake_allocate;
    PFN_vkFreeMemory vkFreeMemory = fake_free;
    PFN_vkMapMemory vkMapMemory = fake_map;
    PFN_vkUnmapMemory vkUnmapMemory = fake_unmap;
    PFN_vkCreateShaderModule vkCreateShaderModule = fake_create_module;
    PFN_vkDestroyShaderModule vkDestroyShaderModule = fake_destroy_module;
    PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines = fake_create_graphics;
    PFN_vkCreateComputePipelines vkCreateComputePipelines = fake_create_compute;
    PFN_vkDestroyPipeline vkDestroyPipeline = fake_destroy_pipeline;
};

bool untouched(const Table& t) {
    return t.vkAllocateMemory == fake_allocate && t.vkFreeMemory == fake_free && t.vkMapMemory == fake_map &&
           t.vkUnmapMemory == fake_unmap && t.vkCreateShaderModule == fake_create_module &&
           t.vkDestroyShaderModule == fake_destroy_module && t.vkCreateGraphicsPipelines == fake_create_graphics &&
           t.vkCreateComputePipelines == fake_create_compute && t.vkDestroyPipeline == fake_destroy_pipeline;
}

// Sends stderr to `path` while it lives; finish() restores it and returns what was written.
struct StderrCapture {
    std::filesystem::path path;
    int saved = -1;
    bool redirected = false;
    explicit StderrCapture(const std::filesystem::path& file) : path(file) {
        std::fflush(stderr);
        std::FILE* out = nullptr;
        if (_wfopen_s(&out, path.c_str(), L"wb") != 0) out = nullptr;
        saved = _dup(2);
        if (out && saved >= 0) redirected = _dup2(_fileno(out), 2) == 0;
        if (out) std::fclose(out);
    }
    std::string finish() {
        std::fflush(stderr);
        if (saved >= 0) {
            _dup2(saved, 2);
            _close(saved);
            saved = -1;
        }
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }
    ~StderrCapture() { finish(); }
};

int count_of(const std::string& text, const std::string& needle) {
    int count = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++count;
    return count;
}

// The calls both forms make. The handles the stand-in driver gave must come back unchanged through the table.
int use_table(Table& vk) {
    const VkDevice device = nullptr;
    VkMemoryAllocateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    VkDeviceMemory first = VK_NULL_HANDLE, second = VK_NULL_HANDLE, imported = VK_NULL_HANDLE;
    info.allocationSize = 1u << 20;
    info.memoryTypeIndex = 0;
    EXPECT(vk.vkAllocateMemory(device, &info, nullptr, &first) == VK_SUCCESS && first);
    info.allocationSize = 3u << 20;
    EXPECT(vk.vkAllocateMemory(device, &info, nullptr, &second) == VK_SUCCESS && second && second != first);
    VkImportMemoryHostPointerInfoEXT host{};
    host.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    info.pNext = &host;
    info.allocationSize = 2u << 20;
    info.memoryTypeIndex = 1;
    EXPECT(vk.vkAllocateMemory(device, &info, nullptr, &imported) == VK_SUCCESS && imported);
    void* mapped = nullptr;
    EXPECT(vk.vkMapMemory(device, second, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS && mapped == g_mapped);
    vk.vkFreeMemory(device, first, nullptr);

    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    VkShaderModule kept = VK_NULL_HANDLE, dropped = VK_NULL_HANDLE;
    module_info.codeSize = 1u << 20;
    EXPECT(vk.vkCreateShaderModule(device, &module_info, nullptr, &kept) == VK_SUCCESS && kept);
    module_info.codeSize = 2u << 20;
    EXPECT(vk.vkCreateShaderModule(device, &module_info, nullptr, &dropped) == VK_SUCCESS && dropped);
    vk.vkDestroyShaderModule(device, dropped, nullptr);

    VkPipeline graphics[3] = {}, compute = VK_NULL_HANDLE;
    const VkGraphicsPipelineCreateInfo graphics_info[3] = {};
    const VkComputePipelineCreateInfo compute_info{};
    EXPECT(vk.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 3, graphics_info, nullptr, graphics) == VK_SUCCESS);
    EXPECT(graphics[0] && graphics[1] && graphics[2]);
    EXPECT(vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compute_info, nullptr, &compute) == VK_SUCCESS);
    vk.vkDestroyPipeline(device, graphics[1], nullptr);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool on = argc == 2 && std::strcmp(argv[1], "--on") == 0;
    if (argc > 1 && !on) {
        std::printf("usage: mem_probe_test [--on]\n");
        return 2;
    }
    // Read at the first interval_seconds(). An empty value removes the variable, so the plain form does not depend
    // on the environment it is started in.
    EXPECT(_putenv_s("LEONIDA_MEM_PROBE", on ? "1" : "") == 0);
    std::error_code ec;
    const std::filesystem::path log_file =
        std::filesystem::temp_directory_path(ec) / ("leonida_mem_probe_test_" + std::to_string(GetCurrentProcessId()) + ".txt");

    probe::DeviceFacts facts;
    facts.memory.memoryTypeCount = 2;
    facts.memory.memoryTypes[0] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    facts.memory.memoryTypes[1] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    facts.describe = [] { return std::string("arena 64 MiB"); };

    Table table, second_table;
    std::string log, later;
    bool redirected = false, installed = false, installed_again = false, sampling = false;
    int used = 1;
    {
        StderrCapture capture(log_file);
        redirected = capture.redirected;
        installed = probe::install(table);
        installed_again = probe::install(second_table);  // a second device of the process
        used = use_table(table);
        if (on && used == 0) probe::sample(facts);
        {
            const auto sampler = probe::start(facts);
            sampling = sampler != nullptr;
            Sleep(2500);  // on: the sampler writes at 1 s and 2 s, and once more when it is stopped
        }
        log = capture.finish();
    }
    {
        StderrCapture capture(log_file);
        Sleep(1500);  // a stopped sampler writes nothing more
        later = capture.finish();
    }
    std::filesystem::remove(log_file, ec);
    if (used != 0) return used;
    EXPECT(redirected);
    EXPECT(g_calls == 11);  // every call reached the stand-in driver, wrapped or not
    EXPECT(untouched(second_table) && !installed_again);
    EXPECT(count_of(later, "mem-probe") == 0);

    probe::State& s = probe::state();
    if (!on) {
        EXPECT(probe::interval_seconds() == 0);
        EXPECT(!installed && untouched(table) && !sampling);
        EXPECT(log.empty());
        EXPECT(s.allocations.empty() && s.modules.empty() && s.pipelines.empty() && !s.modules_made);
        std::printf("probe off: 11 calls reached the driver through an untouched table, no thread, nothing written\n");
        return 0;
    }

    EXPECT(probe::interval_seconds() == 1);
    EXPECT(installed && sampling);
    EXPECT(table.vkAllocateMemory == probe::allocate_memory && table.vkFreeMemory == probe::free_memory &&
           table.vkMapMemory == probe::map_memory && table.vkUnmapMemory == probe::unmap_memory &&
           table.vkCreateShaderModule == probe::create_shader_module &&
           table.vkDestroyShaderModule == probe::destroy_shader_module &&
           table.vkCreateGraphicsPipelines == probe::create_graphics_pipelines &&
           table.vkCreateComputePipelines == probe::create_compute_pipelines &&
           table.vkDestroyPipeline == probe::destroy_pipeline);
    EXPECT(s.allocations.size() == 2 && s.modules.size() == 1 && s.pipelines.size() == 3);
    EXPECT(s.graphics_live == 2 && s.compute_live == 1 && s.graphics_made == 3 && s.compute_made == 1);
    EXPECT(s.module_bytes == (1u << 20) && s.modules_made == 2 && s.module_bytes_made == (3u << 20));
    EXPECT(count_of(log, "[rt] mem-probe: installed, every 1 s\n") == 1);
    // One line from the direct sample(), the sampler's timed ones and its last: all say the same.
    const int samples = count_of(log, "[rt] mem-probe t=");
    EXPECT(samples >= 3 && samples <= 5);
    EXPECT(count_of(log, "pipelines live graphics 2 compute 1 (made 3 / 1,") == samples);
    EXPECT(count_of(log, "shader modules live 1, 1 MiB SPIR-V (made 2, 3 MiB)") == samples);
    EXPECT(count_of(log, "type 0 DL heap 0: 1 x 3 MiB; type 1 HV+HC heap 1: 0 x 0 MiB "
                         "(+1 imported, 2 MiB); arena 64 MiB") == samples);
    std::printf("probe on: 9 entries wrapped, 11 calls reached the driver, %d samples, thread stopped\n", samples);
    return 0;
}
