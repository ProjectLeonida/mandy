// A typeless multisampled colour target through the Xbox-ABI proxy over the Vulkan device.
//   - A 4x4, 4x R8G8B8A8_TYPELESS render target is cleared to blue through a R8G8B8A8_UNORM RTV, a red triangle
//     is drawn over the half above the diagonal, and a green quad over the bottom-right quadrant with PSO
//     SampleMask 0x1 (one sample of four).
//   - Its shader-resource view names R8G8B8A8_UNORM, a concrete member of the texture's typeless family, as D3D12
//     requires of a typeless resource. The T# it writes must decode to the same image as the default SRV of a
//     typed R8G8B8A8_UNORM 4x target of the same shape (its format, type, width, height and sample count), and
//     the bindless set must sample it: a compute shader looks the T# up in set 0's hash table and reads all four
//     samples of the 4x4 image. Red where the triangle covers every sample, blue where the clear shows through,
//     green in sample 0 of the masked quad, and red or blue (never green) on the diagonal's partially covered
//     pixels. No game data.
// Views that must stay null descriptors: a typeless format, UNKNOWN on a typeless texture, a format from another
// family, a single-sample view of the 4x texture, and an MS view of an R32G8X24_TYPELESS depth texture (the
// retail prototype's logged format; multisampled depth/stencil SRVs are not sampled).
// Shaders: test/glsl/vs_draw.vert, ps_draw.frag and cs_sample.comp (TEST_MS_FLOAT).
// Set LEONIDA_VK_VALIDATION=1 to run under VK_LAYER_KHRONOS_validation.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/gfx10_descriptors.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_MS_FLOAT_SPV)
#error "LEONIDA_TEST_VS_SPV, LEONIDA_TEST_PS_SPV and LEONIDA_TEST_MS_FLOAT_SPV must name the compiled test shaders"
#endif

namespace g = leonida::gfx10;

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

struct Vertex {
    float x, y;
    std::uint8_t r, g, b, a;
};

D3D12_CPU_DESCRIPTOR_HANDLE heap_start(ID3D12DescriptorHeap* heap) {
    return {SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(heap, 9))(heap))};
}

#ifdef LEONIDA_TEST_MS_FLOAT_SPV
// What cs_sample.comp is pushed: the T# to look up, and where in the output buffer its 64 samples go.
struct SampleKeys {
    std::uint32_t tsharp[8];
    std::uint32_t ssharp[4];
    std::uint32_t first;
    std::uint32_t mode;  // 2: multisampled
    std::uint32_t layer;
};

VkShaderModule load_shader(vktest::Device& d, const char* path) {
    std::ifstream file(vktest::shader_fixture_path(path), std::ios::binary | std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> code(std::size_t(file.tellg()) / 4);
    file.seekg(0);
    if (code.empty() || !file.read(reinterpret_cast<char*>(code.data()), std::streamsize(code.size() * 4)))
        return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * 4;
    info.pCode = code.data();
    VkShaderModule module{};
    return d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(d.device, &info, nullptr, &module) == VK_SUCCESS
        ? module : VK_NULL_HANDLE;
}

// Runs test/glsl/cs_sample.comp once for `key` on set 0, the device's bindless set, and copies the 4x4 texels it
// read to `results`: 64 samples as RGBA8, with the sample index fastest. Set 1 is the test's output buffer.
bool sample_through_set(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                        const SampleKeys& key, std::uint8_t* results) {
    const VkDevice dev = d.device;
    constexpr VkDeviceSize texels = 64;
    void* mapped = nullptr;
    const VkBuffer output = d.buffer(texels * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    if (!output) return false;

    const VkShaderModule module = load_shader(d, LEONIDA_TEST_MS_FLOAT_SPV);
    VkDescriptorSetLayout set1_layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorSet set1 = VK_NULL_HANDLE;

    const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
                                               nullptr};
    VkDescriptorSetLayoutCreateInfo set1_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set1_info.bindingCount = 1;
    set1_info.pBindings = &binding;
    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    bool ok = module && d.fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(
                            dev, &set1_info, nullptr, &set1_layout) == VK_SUCCESS &&
              d.fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(dev, &pool_info, nullptr, &pool) == VK_SUCCESS;
    if (ok) {
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &set1_layout;
        ok = d.fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(dev, &allocate, &set1) == VK_SUCCESS;
    }
    if (ok) {
        const VkDescriptorBufferInfo buffer_info{output, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set1;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_info;
        d.fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(dev, 1, &write, 0, nullptr);

        const VkDescriptorSetLayout layouts[2] = {set0_layout, set1_layout};
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SampleKeys)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName = "main";
        ok = d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(dev, &layout_info, nullptr, &layout) ==
             VK_SUCCESS;
        pipeline_info.layout = layout;
        ok = ok && d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
                        dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) == VK_SUCCESS;
    }
    if (ok) {
        const auto bind_sets = d.fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        const auto push_constants = d.fn<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        const auto dispatch = d.fn<PFN_vkCmdDispatch>("vkCmdDispatch");
        ok = d.run([&](VkCommandBuffer cb) {
            d.fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            const VkDescriptorSet sets[2] = {set0, set1};
            bind_sets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, nullptr);
            push_constants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(key), &key);
            dispatch(cb, 1, 1, 1);
            return true;
        });
    }
    if (ok) std::memcpy(results, mapped, texels * 4);
    if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(dev, pipeline, nullptr);
    if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(dev, layout, nullptr);
    if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(dev, pool, nullptr);
    if (set1_layout) d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(dev, set1_layout, nullptr);
    if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(dev, module, nullptr);
    return ok;
}
#endif

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU (hosted CI): ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto vs_program = fssm_program({0x56530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const testfiles::ScratchDir scratch("leonida_msaa_typeless_srv_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_SPV), library / "vs_draw.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_draw.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_draw\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": ["
                 "{\"semantic\": \"COLOR\", \"semantic_index\": 0, \"register\": 0, \"mask\": 15, \"rw_mask\": 15, "
                 "\"system_value\": 0, \"component_type\": \"float\"}, "
                 "{\"semantic\": \"POSITION\", \"semantic_index\": 0, \"register\": 1, \"mask\": 3, \"rw_mask\": 3, "
                 "\"system_value\": 0, \"component_type\": \"float\"}]},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    constexpr UINT size = 4;
    constexpr UINT samples = 4;
    constexpr std::uint8_t red[4] = {255, 0, 0, 255};
    constexpr std::uint8_t green[4] = {0, 255, 0, 255};
    constexpr std::uint8_t blue[4] = {0, 0, 255, 255};
    std::array<std::uint8_t, 64 * 4> pixels{};
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.bindless = d.bindless;
        info.bindless_partially_bound = d.partially_bound;
        info.shader_dir = library.string();
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);
        if (rt::d3d12::device_push_constants_need_fallback(inner)) {
            std::printf("d3d12_msaa_typeless_srv_test: skipped, the PSOs need 144 push-constant bytes and this GPU "
                        "has %u\n", rt::d3d12::device_max_push_constants_size(inner));
            return 77;
        }
        if (!d.bindless) {
            std::printf("d3d12_msaa_typeless_srv_test: skipped, this device has no bindless set 0\n");
            return 77;
        }

        // The concrete format the SRV names must be one the GPU samples at 4x; the format-support query is what a
        // title checks (MULTISAMPLE_LOAD).
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{DXGI_FORMAT_R8G8B8A8_UNORM};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))));
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{DXGI_FORMAT_R8G8B8A8_UNORM, samples};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality,
                                                     sizeof(quality))));
        if (!(support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD) || !quality.NumQualityLevels) {
            std::printf("d3d12_msaa_typeless_srv_test: skipped, the device does not advertise 4x R8G8B8A8_UNORM "
                        "multisample loads\n");
            return 77;
        }

        D3D12_ROOT_PARAMETER params[2] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[1].Descriptor = {1, 0};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));

        const D3D12_INPUT_ELEMENT_DESC elements[2] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_APPEND_ALIGNED_ELEMENT,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0}};
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.VS = {vs_container.data(), vs_container.size()};
        pso_desc.PS = {ps_container.data(), ps_container.size()};
        pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        pso_desc.SampleMask = 0xffffffff;
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.DSVFormat = DXGI_FORMAT_UNKNOWN;
        pso_desc.SampleDesc = {samples, 0};
        Ref<ID3D12PipelineState> pso, masked;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));
        pso_desc.SampleMask = 0x1;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&masked.p))));

        // Vertices 0-2: the triangle above the diagonal; 3-6: the bottom-right quadrant.
        const Vertex vertices[7] = {
            {-1, 1, 255, 0, 0, 255}, {1, 1, 255, 0, 0, 255}, {-1, -1, 255, 0, 0, 255},
            {0, 0, 255, 0, 0, 255}, {1, 0, 255, 0, 0, 255}, {0, -1, 255, 0, 0, 255}, {1, -1, 255, 0, 0, 255}};
        const std::uint16_t indices[6] = {3, 4, 5, 5, 4, 6};
        const float tints[2][3] = {{1, 0, 1}, {0, 1, 0}};  // red stays red; then green

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        Ref<ID3D12Resource> vertex_buffer, upload, target, typed, depth;
        const auto vb_desc = buffer_desc(sizeof(vertices));
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&vertex_buffer.p))));
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = size;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
        rt_desc.SampleDesc = {samples, 0};
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&target.p))));
        D3D12_RESOURCE_DESC typed_desc = rt_desc;
        typed_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &typed_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&typed.p))));
        D3D12_RESOURCE_DESC depth_desc = rt_desc;
        depth_desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
        depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        const bool depth_created = SUCCEEDED(device->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
            IID_PPV_ARGS(&depth.p)));

        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 256, indices, sizeof(indices));
        std::memcpy(up + 512, tints[0], sizeof(tints[0]));
        std::memcpy(up + 768, tints[1], sizeof(tints[1]));
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 8,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap, rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const UINT srv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const D3D12_CPU_DESCRIPTOR_HANDLE srv = heap_start(srv_heap.p);
        const auto srv_slot = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{srv.ptr + SIZE_T(i) * srv_step}; };
        const auto srv_bytes = [&](UINT i) {
            return reinterpret_cast<const std::uint8_t*>(srv.ptr + SIZE_T(i) * srv_step);
        };
        const auto null_at = [&](UINT i) {
            const std::uint8_t* bytes = srv_bytes(i);
            return std::all_of(bytes, bytes + srv_step, [](std::uint8_t byte) { return byte == 0; });
        };
        const auto image_at = [&](UINT i, leonida::gfx10::Image& image) {
            leonida::gfx10::ImageWords words{};
            std::memcpy(words.data(), srv_bytes(i), sizeof(words));
            return leonida::gfx10::is_image(words) && leonida::gfx10::decode(words, image);
        };

        constexpr UINT typeless_view = 0, typed_view = 1, foreign_view = 2, typeless_format_view = 3,
                       unknown_view = 4, typed_foreign_view = 5, single_sample_view = 6, depth_view = 7;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        device->CreateShaderResourceView(target.p, &srv_desc, srv_slot(typeless_view));
        device->CreateShaderResourceView(typed.p, nullptr, srv_slot(typed_view));

        leonida::gfx10::Image image{}, typed_image{};
        EXPECT(image_at(typeless_view, image));
        EXPECT(image_at(typed_view, typed_image));
        EXPECT(image.type == leonida::gfx10::ImageType::image_2d_msaa && image.samples == samples &&
               image.width == size && image.height == size && image.first_mip == 0 && image.mip_count == 1 &&
               image.first_layer == 0 && image.layer_count == 1);
        EXPECT(image.format == g::format_from_vulkan(VK_FORMAT_R8G8B8A8_UNORM));
        // The address is the one thing a typeless SRV cannot share with the typed twin's default SRV.
        image.address = typed_image.address = 0;
        EXPECT(image == typed_image);

        srv_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        device->CreateShaderResourceView(target.p, &srv_desc, srv_slot(foreign_view));
        EXPECT(null_at(foreign_view));  // another typeless family
        srv_desc.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
        device->CreateShaderResourceView(target.p, &srv_desc, srv_slot(typeless_format_view));
        EXPECT(null_at(typeless_format_view));  // a typeless format names no concrete one
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        device->CreateShaderResourceView(target.p, &srv_desc, srv_slot(unknown_view));
        EXPECT(null_at(unknown_view));  // nor does UNKNOWN inherit a typeless texture's format
        srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(target.p, &srv_desc, srv_slot(single_sample_view));
        EXPECT(null_at(single_sample_view));  // a single-sample view of the 4x texture
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        srv_desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        device->CreateShaderResourceView(typed.p, &srv_desc, srv_slot(typed_foreign_view));
        EXPECT(null_at(typed_foreign_view));  // a typed MS texture still needs its own format
        srv_desc.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        if (depth_created) {
            device->CreateShaderResourceView(depth.p, &srv_desc, srv_slot(depth_view));
            EXPECT(null_at(depth_view));  // multisampled depth/stencil SRVs are not sampled
        }

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        const D3D12_CPU_DESCRIPTOR_HANDLE rtv = heap_start(rtv_heap.p);
        D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
        rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
        device->CreateRenderTargetView(target.p, &rtv_desc, rtv);

        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_INDEX_BUFFER_VIEW ibv{upload_va + 256, sizeof(indices), DXGI_FORMAT_R16_UINT};
        list->IASetIndexBuffer(&ibv);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        const float origin[2] = {0, 0};
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);

        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        D3D12_RESOURCE_BARRIER to_vertex = transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        list->ResourceBarrier(1, &to_vertex);
        const FLOAT clear[4] = {0, 0, 1, 1};
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);  // red
        list->DrawInstanced(3, 1, 0, 0);
        list->SetPipelineState(masked.p);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 768);  // green, sample 0 only
        list->DrawIndexedInstanced(6, 1, 0, 0, 0);
        D3D12_RESOURCE_BARRIER to_sample = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET,
                                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &to_sample);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

#ifdef LEONIDA_TEST_MS_FLOAT_SPV
        // Sample the typeless SRV through set 0, as a translated shader would from its T#.
        const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
        const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
        EXPECT(set0 && set0_layout);
        SampleKeys key{};
        std::memcpy(key.tsharp, srv_bytes(typeless_view), sizeof(key.tsharp));
        key.mode = 2;
        EXPECT(sample_through_set(d, set0, set0_layout, key, pixels.data()));
#endif

        for (auto* r : {&vertex_buffer, &upload, &target, &typed, &depth}) r->reset();
        pso.reset();
        masked.reset();
        root.reset();
        srv_heap.reset();
        rtv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }
#ifdef LEONIDA_TEST_MS_FLOAT_SPV
    // Sample i of texel (x, y) is pixels[4 * (16 y + 4 x + i)]. The diagonal runs through texel centres on
    // x + y == 2..4 at this size, where the triangle covers some samples and not others: those are red or blue,
    // never green. The masked quad's sample 0 is green everywhere in the bottom-right quadrant, and its other
    // samples are what the triangle or the clear left there.
    unsigned mismatches = 0;
    for (UINT y = 0; y < 4; ++y) {
        for (UINT x = 0; x < 4; ++x) {
            for (UINT s = 0; s < 4; ++s) {
                const std::uint8_t* got = pixels.data() + 4 * (16 * y + 4 * x + s);
                const bool quadrant = x >= 2 && y >= 2;
                bool ok = true;
                if (quadrant && s == 0) ok = std::memcmp(got, green, 4) == 0;
                else if (x + y <= 1) ok = std::memcmp(got, red, 4) == 0;
                else if (x + y >= 5) ok = std::memcmp(got, blue, 4) == 0;
                else ok = std::memcmp(got, red, 4) == 0 || std::memcmp(got, blue, 4) == 0;
                if (!ok && mismatches++ < 8)
                    std::fprintf(stderr, "texel (%u, %u) sample %u: got %u,%u,%u,%u\n", x, y, s, got[0], got[1],
                                 got[2], got[3]);
            }
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("MSAA typeless SRV: R8G8B8A8_TYPELESS 4x target cleared and drawn, 64 samples matched through its "
                "R8G8B8A8_UNORM T#; typeless, foreign-family, UNKNOWN, single-sample and depth views stayed null\n");
#endif
    return 0;
}
