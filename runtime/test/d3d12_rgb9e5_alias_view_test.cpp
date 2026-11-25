// D3D-RGB9E5-UINT-ALIAS-VIEW: the console title packs light into R32_UINT textures as R9G9B9E5_SHAREDEXP bits from
// compute shaders, which cannot store that format, and samples them through R9G9B9E5_SHAREDEXP shader-resource views
// of the same texture (sky, sky and light probes, bounced light, cloud and sky volumes). Desktop D3D12 refuses such a
// view and so did the layer; the console allows it. On the title's device, the Vulkan device behind the Xbox-ABI proxy
// (rt::d3d12::set_device_outer), a texture of that pair that allows unordered access is a mutable image listing both
// formats, and takes a shader-resource view in the other one (docs/d3d12_vulkan.md, "R32_UINT and R9G9B9E5 views").
//
// Exercises, on the title's device:
//   1. R32_UINT UAV textures from device slot 76 (committed, the title's call), CreatePlacedResource and a 3D
//      committed one: an R9G9B9E5_SHAREDEXP SRV of each is a 5_9_9_9_FLOAT T# (132) that set 0 registered. An
//      R9G9B9E5 UAV of the same texture, and an R9G9B9E5 SRV of an R32_UINT texture without unordered access, stay
//      null descriptors.
//   2. The reverse: an R32_UINT SRV of an R9G9B9E5_SHAREDEXP UAV texture is a 32_UINT T# (20), registered; an
//      R32_UINT UAV of it stays null.
//   3. A castable-format list as well (CreateCommittedResource3 with {R32_FLOAT}): R9G9B9E5 and R32_FLOAT SRVs.
//   4. On the GPU: one compute dispatch stores 16 packed R9G9B9E5 words through the R32_UINT UAV (set-0 storage
//      bindings 9 and 4), a second samples them through the R9G9B9E5 SRV (binding 0), and every float equals
//      uav_clear::decode of its word; CopyTextureRegion then reads the 16 words back unchanged.
// And on a device without the proxy (the PC path, which the WARP parity tests use): the same views stay null.
//
// Synthetic data only; no game files. Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"
#include "../src/uav_clear.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace g = leonida::gfx10;

// Device slot 76 as the title calls it: the PC CreateCommittedResource's arguments with the IID by pointer.
using XboxCreateResourceFn = HRESULT(WINAPI*)(void* device, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                              const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                              const D3D12_CLEAR_VALUE*, const IID*, void**);

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT kSize = 4;  // every 2D texture here is 4x4: 16 texels, one dispatch of 16 lanes

D3D12_RESOURCE_DESC texture(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                            D3D12_RESOURCE_DIMENSION dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            UINT16 depth = 1) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dimension;
    d.Width = kSize;
    d.Height = kSize;
    d.DepthOrArraySize = depth;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

D3D12_SHADER_RESOURCE_VIEW_DESC srv(DXGI_FORMAT format, bool volume = false) {
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = format;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (volume) {
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
        v.Texture3D.MipLevels = 1;
    } else {
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Texture2D.MipLevels = 1;
    }
    return v;
}

D3D12_UNORDERED_ACCESS_VIEW_DESC uav(DXGI_FORMAT format) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
    v.Format = format;
    v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    return v;
}

// CPU descriptor slots in a heap of the PC device behind the proxy; the proxy's view calls write them unchanged
// (the handles are not in a title-backed heap, so translate_cpu_descriptor_handle keeps them).
struct Slots {
    Ref<ID3D12DescriptorHeap> heap;
    std::byte* base = nullptr;
    UINT step = 0;

    bool create(ID3D12Device* device, UINT count) {
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, count};
        if (FAILED(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap.p)))) return false;
        base = reinterpret_cast<std::byte*>(heap->GetCPUDescriptorHandleForHeapStart().ptr);
        step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        return base && step >= sizeof(g::ImageWords);
    }
    D3D12_CPU_DESCRIPTOR_HANDLE handle(UINT i) const {
        return {reinterpret_cast<SIZE_T>(base + std::size_t(i) * step)};
    }
    // The slot for a view call, filled with a pattern first so that a call which writes nothing does not pass for a
    // null view.
    D3D12_CPU_DESCRIPTOR_HANDLE at(UINT i) const {
        std::memset(base + std::size_t(i) * step, 0xaa, step);
        return handle(i);
    }
    g::ImageWords words(UINT i) const {
        g::ImageWords w{};
        std::memcpy(w.data(), base + std::size_t(i) * step, sizeof(w));
        return w;
    }
    bool null(UINT i) const {
        const auto* p = reinterpret_cast<const std::uint8_t*>(base + std::size_t(i) * step);
        return std::all_of(p, p + step, [](std::uint8_t b) { return b == 0; });
    }
};

// The T# in slot `i` decodes with GFX10 format `format` and is in set 0's hash table (not the fallback, index 0).
bool written_and_registered(ID3D12Device* device, const Slots& slots, UINT i, std::uint32_t format) {
    const g::ImageWords words = slots.words(i);
    g::Image image{};
    if (!g::decode(words, image) || image.format != format) {
        std::fprintf(stderr, "slot %u: no T# of format %u (decoded %u)\n", i, format, unsigned(image.format));
        return false;
    }
    std::uint32_t key[8];
    std::memcpy(key, words.data(), sizeof(key));
    if (!rt::d3d12::bindless_hash_lookup(device, key)) {
        std::fprintf(stderr, "slot %u: the T# is not in set 0's hash table\n", i);
        return false;
    }
    return true;
}

#ifdef LEONIDA_TEST_CS_RGB9E5_ALIAS_SPV
bool read_spirv(const char* path, std::vector<std::uint32_t>& out) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto size = file.tellg();
    if (size <= 0 || size % 4 != 0) return false;
    out.resize(std::size_t(size) / 4);
    file.seekg(0);
    return bool(file.read(reinterpret_cast<char*>(out.data()), size));
}

// The push constants of test/glsl/cs_r9g9b9e5_uav.comp.
struct ShaderParams {
    std::uint32_t tsharp[8];
    std::uint32_t ssharp[4];
    std::uint32_t mode;  // 0: store packed words through the T#'s storage views; 1: sample through its sampled view
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pad;
};

// Stores `words` through the UAV T# (an R32_UINT storage view), then samples the texture through the SRV T# (an
// E5B9G9R9 sampled view of the same image) into `out`, four floats a texel. Both against the layer's set 0.
bool store_then_sample(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                       const g::ImageWords& uav_words, const g::ImageWords& srv_words, const g::SamplerWords& sampler,
                       const std::array<std::uint32_t, 16>& words, std::array<float, 64>& out) {
    std::vector<std::uint32_t> code;
    if (!read_spirv(LEONIDA_TEST_CS_RGB9E5_ALIAS_SPV, code)) return false;
    void* in_ptr = nullptr;
    void* out_ptr = nullptr;
    const VkBuffer in_buf = d.buffer(sizeof(words), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &in_ptr);
    const VkBuffer out_buf = d.buffer(sizeof(out), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &out_ptr);
    if (!in_buf || !out_buf) return false;
    std::memcpy(in_ptr, words.data(), sizeof(words));

    const auto create_dsl = d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout");
    const auto destroy_dsl = d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout");
    const auto create_pool = d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
    const auto destroy_pool = d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
    const auto alloc_sets = d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    const auto update_sets = d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    const auto create_layout = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
    const auto destroy_layout = d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
    const auto create_module = d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule");
    const auto destroy_module = d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
    const auto create_pipelines = d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
    const auto destroy_pipeline = d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline");

    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto done = [&](bool ok) {
        if (pipeline) destroy_pipeline(d.device, pipeline, nullptr);
        if (module) destroy_module(d.device, module, nullptr);
        if (layout) destroy_layout(d.device, layout, nullptr);
        if (pool) destroy_pool(d.device, pool, nullptr);
        if (set1_layout) destroy_dsl(d.device, set1_layout, nullptr);
        return ok;
    };

    // Set 1: the packed words in, the sampled floats out.
    const VkDescriptorSetLayoutBinding bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set1_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set1_info.bindingCount = 2;
    set1_info.pBindings = bindings;
    if (create_dsl(d.device, &set1_info, nullptr, &set1_layout) != VK_SUCCESS) return done(false);
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (create_pool(d.device, &pool_info, nullptr, &pool) != VK_SUCCESS) return done(false);
    VkDescriptorSetAllocateInfo alloc_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &set1_layout;
    VkDescriptorSet set1 = VK_NULL_HANDLE;
    if (alloc_sets(d.device, &alloc_info, &set1) != VK_SUCCESS) return done(false);
    const VkDescriptorBufferInfo buffers[2] = {{in_buf, 0, sizeof(words)}, {out_buf, 0, sizeof(out)}};
    VkWriteDescriptorSet writes[2]{};
    for (UINT i = 0; i < 2; ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet = set1;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    update_sets(d.device, 2, writes, 0, nullptr);

    const VkDescriptorSetLayout set_layouts[2] = {set0_layout, set1_layout};
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderParams)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 2;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    if (create_layout(d.device, &layout_info, nullptr, &layout) != VK_SUCCESS) return done(false);
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size() * sizeof(std::uint32_t);
    module_info.pCode = code.data();
    if (create_module(d.device, &module_info, nullptr, &module) != VK_SUCCESS) return done(false);
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = layout;
    if (create_pipelines(d.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS)
        return done(false);

    const bool ok = d.run([&](VkCommandBuffer cb) {
        const auto bind_pipeline = d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
        const auto bind_sets = d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        const auto barrier = d.fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        const VkDescriptorSet sets[2] = {set0, set1};
        bind_pipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, nullptr);

        ShaderParams store{};
        std::memcpy(store.tsharp, uav_words.data(), sizeof(store.tsharp));
        std::memcpy(store.ssharp, sampler.data(), sizeof(store.ssharp));
        store.mode = 0;
        store.width = store.height = kSize;
        push(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(store), &store);
        dispatch(cb, 1, 1, 1);
        VkMemoryBarrier written{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        written.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        written.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &written, 0,
                nullptr, 0, nullptr);

        ShaderParams sample = store;
        std::memcpy(sample.tsharp, srv_words.data(), sizeof(sample.tsharp));
        sample.mode = 1;
        push(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sample), &sample);
        dispatch(cb, 1, 1, 1);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                nullptr);
        return true;
    });
    if (ok) std::memcpy(out.data(), out_ptr, sizeof(out));
    return done(ok);
}
#endif

// Runs `record` on a fresh direct list of `device` and waits for it.
template<class Record>
bool execute(ID3D12Device* device, Record&& record) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    record(list.p);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (rt::d3d12::last_execute_result(queue.p) != leonida::ReplayResult::success) return false;
    return SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr));
}

rt::d3d12::VulkanDeviceInfo device_info(const vktest::Device& d) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = d.buffer_device_address;
    info.bindless = d.bindless;
    info.bindless_partially_bound = d.partially_bound;
    info.image_view_min_lod = d.image_view_min_lod;
    info.host_image_copy = d.host_image_copy;
    return info;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;  // heaps, so placed textures, need the GPU address arena
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device has no bindless set 0, which the views are registered in\n");
        return 77;
    }
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    constexpr auto uint32 = DXGI_FORMAT_R32_UINT;
    constexpr auto rgb9e5 = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    constexpr auto unordered = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    constexpr std::uint32_t kFormat32Uint = 20, kFormat5999Float = 132;  // GFX10_FORMAT_32_UINT, _5_9_9_9_FLOAT
    D3D12_RESOURCE_ALLOCATION_INFO title_allocation{}, pc_allocation{};

    {
        // The title's device: the Vulkan device behind the Xbox-ABI proxy, as D3D12XboxCreateDevice makes it.
        const rt::d3d12::VulkanDeviceInfo info = device_info(d);
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`
        Slots slots;
        EXPECT(slots.create(inner, 16));

        // 1. R32_UINT UAV textures: slot 76 (committed, the title's own call shape), placed, and a volume.
        const IID resource_iid = __uuidof(ID3D12Resource);
        D3D12_RESOURCE_DESC light_desc = texture(uint32, unordered);
        light_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);  // the Xbox tiled layout word slot 76 replaces
        Ref<ID3D12Resource> light;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxCreateResourceFn>(slot(device.p, 76))(
            device.p, &default_heap, D3D12_HEAP_FLAG_NONE, &light_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, &resource_iid, reinterpret_cast<void**>(&light.p))));
        device->CreateUnorderedAccessView(light.p, nullptr, nullptr, slots.at(0));
        device->CreateShaderResourceView(light.p, nullptr, slots.at(1));
        const D3D12_SHADER_RESOURCE_VIEW_DESC packed_view = srv(rgb9e5);
        device->CreateShaderResourceView(light.p, &packed_view, slots.at(2));
        EXPECT(written_and_registered(device.p, slots, 0, kFormat32Uint));
        EXPECT(written_and_registered(device.p, slots, 1, kFormat32Uint));
        EXPECT(written_and_registered(device.p, slots, 2, kFormat5999Float));
        g::Image own{}, packed{};  // one texture, two formats: the same address, RGB1 for the packed view
        EXPECT(g::decode(slots.words(1), own) && g::decode(slots.words(2), packed));
        EXPECT(packed.address == own.address && packed.swizzle == g::channels_rgb);
        const D3D12_UNORDERED_ACCESS_VIEW_DESC packed_store = uav(rgb9e5);
        device->CreateUnorderedAccessView(light.p, nullptr, &packed_store, slots.at(3));
        EXPECT(slots.null(3));  // shaders store the words through R32_UINT; no typed R9G9B9E5 store is made up

        const D3D12_RESOURCE_DESC placed_desc = texture(uint32, unordered);
        if (d.buffer_device_address) {  // heaps live in the GPU address arena
            D3D12_HEAP_DESC heap_desc{};
            heap_desc.SizeInBytes = 1 << 20;
            heap_desc.Properties = default_heap;
            heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
            Ref<ID3D12Heap> heap;
            EXPECT(SUCCEEDED(device->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
            Ref<ID3D12Resource> placed;
            EXPECT(SUCCEEDED(device->CreatePlacedResource(heap.p, 0, &placed_desc,
                                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                          IID_PPV_ARGS(&placed.p))));
            device->CreateShaderResourceView(placed.p, &packed_view, slots.at(4));
            EXPECT(written_and_registered(device.p, slots, 4, kFormat5999Float));
        } else {
            std::printf("no bufferDeviceAddress, so no heaps: the placed texture was not checked\n");
        }

        const D3D12_RESOURCE_DESC volume_desc = texture(uint32, unordered, D3D12_RESOURCE_DIMENSION_TEXTURE3D, 4);
        Ref<ID3D12Resource> volume;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &volume_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&volume.p))));
        const D3D12_SHADER_RESOURCE_VIEW_DESC packed_volume = srv(rgb9e5, true);
        device->CreateShaderResourceView(volume.p, &packed_volume, slots.at(5));
        EXPECT(written_and_registered(device.p, slots, 5, kFormat5999Float));

        // Only textures a shader can write are listed with both formats; a plain R32_UINT texture keeps the PC rule.
        const D3D12_RESOURCE_DESC plain_desc = texture(uint32, D3D12_RESOURCE_FLAG_NONE);
        Ref<ID3D12Resource> plain;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &plain_desc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&plain.p))));
        device->CreateShaderResourceView(plain.p, &packed_view, slots.at(6));
        EXPECT(slots.null(6));

        // 2. The reverse: an R32_UINT SRV of an R9G9B9E5 UAV texture reads its raw bits.
        D3D12_RESOURCE_DESC packed_desc = texture(rgb9e5, unordered);
        Ref<ID3D12Resource> packed_texture;
        EXPECT(SUCCEEDED(reinterpret_cast<XboxCreateResourceFn>(slot(device.p, 76))(
            device.p, &default_heap, D3D12_HEAP_FLAG_NONE, &packed_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr, &resource_iid, reinterpret_cast<void**>(&packed_texture.p))));
        const D3D12_SHADER_RESOURCE_VIEW_DESC bits_view = srv(uint32);
        device->CreateShaderResourceView(packed_texture.p, &bits_view, slots.at(7));
        EXPECT(written_and_registered(device.p, slots, 7, kFormat32Uint));
        const D3D12_UNORDERED_ACCESS_VIEW_DESC bits_store = uav(uint32);
        device->CreateUnorderedAccessView(packed_texture.p, nullptr, &bits_store, slots.at(8));
        EXPECT(slots.null(8));

        // 3. A castable-format list as well: both its format and the packed one are views.
        Ref<ID3D12Device10> device10;
        EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&device10.p))));
        D3D12_RESOURCE_DESC1 listed_desc{};
        std::memcpy(&listed_desc, &placed_desc, sizeof(placed_desc));
        const DXGI_FORMAT castable[] = {DXGI_FORMAT_R32_FLOAT};
        Ref<ID3D12Resource> listed;
        EXPECT(SUCCEEDED(device10->CreateCommittedResource3(&default_heap, D3D12_HEAP_FLAG_NONE, &listed_desc,
                                                            D3D12_BARRIER_LAYOUT_COMMON, nullptr, nullptr, 1,
                                                            castable, IID_PPV_ARGS(&listed.p))));
        device->CreateShaderResourceView(listed.p, &packed_view, slots.at(9));
        EXPECT(written_and_registered(device.p, slots, 9, kFormat5999Float));
        const D3D12_SHADER_RESOURCE_VIEW_DESC float_view = srv(DXGI_FORMAT_R32_FLOAT);
        device->CreateShaderResourceView(listed.p, &float_view, slots.at(10));
        EXPECT(written_and_registered(device.p, slots, 10, 22));  // GFX10_FORMAT_32_FLOAT
        title_allocation = inner->GetResourceAllocationInfo(0, 1, &placed_desc);

#ifdef LEONIDA_TEST_CS_RGB9E5_ALIAS_SPV
        // 4. On the GPU. The list's clear puts the texture in its unordered-access layout and brings set 0 up to date.
        const UINT zero[4] = {};
        const g::ImageWords store_words = slots.words(0), sample_words = slots.words(2);
        device->CreateUnorderedAccessView(light.p, nullptr, nullptr, slots.at(11));
        EXPECT(execute(inner, [&](ID3D12GraphicsCommandList* list) {
            list->ClearUnorderedAccessViewUint({0}, slots.handle(11), light.p, zero, 0, nullptr);
        }));
        Ref<ID3D12DescriptorHeap> sampler_heap;
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1};
        EXPECT(SUCCEEDED(inner->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        D3D12_SAMPLER_DESC point{};
        point.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        point.AddressU = point.AddressV = point.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        const D3D12_CPU_DESCRIPTOR_HANDLE sampler_slot = sampler_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateSampler(&point, sampler_slot);
        g::SamplerWords sampler{};
        std::memcpy(sampler.data(), reinterpret_cast<const void*>(sampler_slot.ptr), sizeof(sampler));
        // A second execute brings the sampler into set 0 too.
        EXPECT(execute(inner, [](ID3D12GraphicsCommandList*) {}));

        // Shared exponents from 2^-24 to the largest finite value, and mantissas that tell the channels apart.
        const std::array<std::uint32_t, 16> words = {
            0x94010040u, 0x79010100u, 0x80000100u, 0x80020000u, 0x84000000u, 0xb0080190u, 0xf80001ffu, 0xffffffffu,
            0x08000100u, 0x00000100u, 0x00000001u, 0x7556ab55u, 0x840e0501u, 0x98040100u, 0x00000000u, 0x88040280u,
        };
        std::array<float, 64> sampled{};
        EXPECT(store_then_sample(d, rt::d3d12::bindless_set(device.p), rt::d3d12::bindless_set_layout(device.p),
                                 store_words, sample_words, sampler, words, sampled));
        for (UINT i = 0; i < 16; ++i) {
            leonida::uav_clear::Element element{};
            element.size = 4;
            std::memcpy(element.bytes.data(), &words[i], 4);
            VkClearColorValue want{};
            EXPECT(leonida::uav_clear::decode(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, element, want));  // RGB, alpha 1
            for (UINT c = 0; c < 4; ++c) {  // exact: every value is a 9-bit mantissa times a power of two
                const float got = sampled[i * 4 + c];
                if (got != want.float32[c]) {
                    std::fprintf(stderr, "texel %u channel %u: sampled %g, want %g (word 0x%08x)\n", i, c, got,
                                 want.float32[c], words[i]);
                    return 1;
                }
            }
        }

        // The same memory: the words the shader stored are what the R32_UINT texture holds.
        Ref<ID3D12Resource> readback;
        D3D12_RESOURCE_DESC readback_desc{};
        readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        readback_desc.Width = 256 * kSize;
        readback_desc.Height = 1;
        readback_desc.DepthOrArraySize = 1;
        readback_desc.MipLevels = 1;
        readback_desc.SampleDesc.Count = 1;
        readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        EXPECT(SUCCEEDED(inner->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&readback.p))));
        EXPECT(execute(inner, [&](ID3D12GraphicsCommandList* list) {
            D3D12_RESOURCE_BARRIER to_source{};
            to_source.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            to_source.Transition = {light.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
            list->ResourceBarrier(1, &to_source);
            D3D12_TEXTURE_COPY_LOCATION to{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            to.PlacedFootprint = {0, {uint32, kSize, kSize, 1, 256}};
            D3D12_TEXTURE_COPY_LOCATION from{light.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        }));
        void* mapped = nullptr;
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
        for (UINT i = 0; i < 16; ++i) {
            std::uint32_t word = 0;
            std::memcpy(&word, static_cast<const std::uint8_t*>(mapped) + (i / kSize) * 256 + (i % kSize) * 4, 4);
            if (word != words[i]) {
                std::fprintf(stderr, "texel %u: the texture holds 0x%08x, the shader stored 0x%08x\n", i, word,
                             words[i]);
                return 1;
            }
        }
        readback->Unmap(0, nullptr);
        std::printf("GPU: 16/16 packed words stored through the R32_UINT UAV sampled back through the R9G9B9E5 SRV "
                    "as their RGB9E5 floats, and read back unchanged\n");
#else
        std::printf("glslangValidator was not found at configure time: the GPU store-and-sample part did not run\n");
#endif
    }

    {
        // 5. The PC path: a device without the proxy keeps the PC rule, so the WARP parity tests see what they did.
        const rt::d3d12::VulkanDeviceInfo info = device_info(d);
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Slots slots;
        EXPECT(slots.create(device.p, 4));
        const D3D12_RESOURCE_DESC light_desc = texture(uint32, unordered);
        Ref<ID3D12Resource> light;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &light_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&light.p))));
        const D3D12_SHADER_RESOURCE_VIEW_DESC packed_view = srv(rgb9e5);
        device->CreateShaderResourceView(light.p, &packed_view, slots.at(0));
        EXPECT(slots.null(0));
        device->CreateShaderResourceView(light.p, nullptr, slots.at(1));
        EXPECT(written_and_registered(device.p, slots, 1, kFormat32Uint));  // its own format still is a view
        const D3D12_RESOURCE_DESC packed_desc = texture(rgb9e5, unordered);
        Ref<ID3D12Resource> packed_texture;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &packed_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&packed_texture.p))));
        const D3D12_SHADER_RESOURCE_VIEW_DESC bits_view = srv(uint32);
        device->CreateShaderResourceView(packed_texture.p, &bits_view, slots.at(2));
        EXPECT(slots.null(2));
        pc_allocation = device->GetResourceAllocationInfo(0, 1, &light_desc);
    }
    std::printf("4x4 R32_UINT UAV texture: %llu bytes on the title's device (both formats listed), %llu on a PC one\n",
                static_cast<unsigned long long>(title_allocation.SizeInBytes),
                static_cast<unsigned long long>(pc_allocation.SizeInBytes));

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_rgb9e5_alias_view_test: OK (0 validation errors)\n");
    return 0;
}
