// A multisampled depth-stencil target's shader-resource view through the Xbox-ABI proxy over the Vulkan device:
// the retail case, DXGI_FORMAT_R32G8X24_TYPELESS (a D32_SFLOAT_S8_UINT image here).
//   - A 4x4, 4x R32G8X24_TYPELESS depth target is cleared to depth 0.5 (stencil 0x5a), and a red triangle over the
//     half above the diagonal is drawn with PSO SampleMask 0x1, so its sample 0 gets depth 0 and samples 1-3 keep
//     the clear's 0.5.
//   - Its SRV names R32_FLOAT_X8X24_TYPELESS, the depth-plane format of the texture's family. The T# must decode to
//     the X24_8_32_FLOAT format, a multisampled 2D image, four samples, and set 0 must sample it: a compute shader
//     looks the T# up in the bindless set's hash table and reads all four samples of the 4x4 image, comparing the
//     raw float bits (0.0 where the triangle's sample 0 wrote, 0.5 from the clear elsewhere).
// Negatives that must stay null descriptors: a single-sampled view of the 4x texture, a multisampled view of a
// single-sampled texture, another depth family's plane format, R32_FLOAT on this family, a colour format, UNKNOWN on
// a typeless depth texture, and the depth-plane format on a texture without a stencil. A sample count the GPU does
// not support is refused when the resource is created. D3D12's TEXTURE2DMS SRV has no plane field, so the stencil
// plane is not an SRV shape.
// Shaders: test/glsl/vs_draw.vert, ps_draw.frag and cs_sample.comp (TEST_MS_DEPTH). No game data.
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

#if !defined(LEONIDA_TEST_VS_SPV) || !defined(LEONIDA_TEST_PS_SPV) || !defined(LEONIDA_TEST_MS_DEPTH_SPV)
#error "LEONIDA_TEST_VS_SPV, LEONIDA_TEST_PS_SPV and LEONIDA_TEST_MS_DEPTH_SPV must name the compiled test shaders"
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

// Runs test/glsl/cs_sample.comp (TEST_MS_DEPTH) once for `key` on set 0, the device's bindless set, and copies the
// 4x4 texels it read to `results`: 64 samples as uint depth bits, with the sample index fastest.
bool sample_through_set(vktest::Device& d, VkDescriptorSet set0, VkDescriptorSetLayout set0_layout,
                        const SampleKeys& key, std::uint32_t* results) {
    const VkDevice dev = d.device;
    constexpr VkDeviceSize samples = 64;
    void* mapped = nullptr;
    const VkBuffer output = d.buffer(samples * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &mapped);
    if (!output) return false;

    const VkShaderModule module = load_shader(d, LEONIDA_TEST_MS_DEPTH_SPV);
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
    if (ok) std::memcpy(results, mapped, samples * 4);
    if (pipeline) d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(dev, pipeline, nullptr);
    if (layout) d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(dev, layout, nullptr);
    if (pool) d.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(dev, pool, nullptr);
    if (set1_layout) d.fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(dev, set1_layout, nullptr);
    if (module) d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(dev, module, nullptr);
    return ok;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU (hosted CI): ctest skips
    EXPECT(device_status == vktest::DeviceStatus::ready);

    const auto vs_program = fssm_program({0x56530001, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    const testfiles::ScratchDir scratch("leonida_msaa_depth_srv_test");
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
    constexpr std::uint32_t zero_bits = 0, half_bits = 0x3f000000;  // 0.0f and 0.5f
    std::array<std::uint32_t, 64> depth_samples{};
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
            std::printf("d3d12_msaa_depth_srv_test: skipped, the PSOs need 144 push-constant bytes and this GPU has "
                        "%u\n", rt::d3d12::device_max_push_constants_size(inner));
            return 77;
        }
        if (!d.bindless) {
            std::printf("d3d12_msaa_depth_srv_test: skipped, this device has no bindless set 0\n");
            return 77;
        }

        // The depth target is a 4x D32_SFLOAT_S8_UINT image; a depth SRV needs the GPU to sample it at 4x. The
        // quality-levels query is the same Vulkan image-format query the layer's sampler uses.
        D3D12_FEATURE_DATA_FORMAT_SUPPORT plane_support{DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &plane_support,
                                                     sizeof(plane_support))));
        EXPECT((plane_support.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD) != 0);  // as the host reports it
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{DXGI_FORMAT_D32_FLOAT_S8X24_UINT, samples};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality,
                                                     sizeof(quality))));
        if (!quality.NumQualityLevels) {
            std::printf("d3d12_msaa_depth_srv_test: skipped, the device does not support 4x D32_FLOAT_S8X24_UINT\n");
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
        pso_desc.SampleMask = 0x1;  // one sample of four writes depth
        pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pso_desc.RasterizerState.DepthClipEnable = TRUE;
        pso_desc.DepthStencilState.DepthEnable = TRUE;
        pso_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        pso_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        pso_desc.InputLayout = {elements, 2};
        pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pso_desc.NumRenderTargets = 1;
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        pso_desc.SampleDesc = {samples, 0};
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        // Vertices 0-2: the triangle above the diagonal. It is the only draw: the point is which samples of the
        // depth target it writes (sample 0 through the PSO's mask), not its colour.
        const Vertex vertices[3] = {{-1, 1, 255, 0, 0, 255}, {1, 1, 255, 0, 0, 255}, {-1, -1, 255, 0, 0, 255}};
        const float tint[3] = {1, 0, 1};

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        Ref<ID3D12Resource> vertex_buffer, upload, colour, depth, depth_single, depth_typed;
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
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt_desc.SampleDesc = {samples, 0};
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&colour.p))));
        D3D12_RESOURCE_DESC ds_desc = rt_desc;
        ds_desc.Format = DXGI_FORMAT_R32G8X24_TYPELESS;
        ds_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ds_desc,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                         IID_PPV_ARGS(&depth.p))));
        D3D12_RESOURCE_DESC ds_single = ds_desc;
        ds_single.SampleDesc = {1, 0};
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ds_single,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                         IID_PPV_ARGS(&depth_single.p))));
        D3D12_RESOURCE_DESC ds_typed = ds_desc;
        ds_typed.Format = DXGI_FORMAT_D32_FLOAT;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ds_typed,
                                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                         IID_PPV_ARGS(&depth_typed.p))));

        // A sample count the GPU does not support is refused at creation: 16x on most parts. The query above says
        // which counts exist, so only an unsupported one is asserted.
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS sixteen{DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 16};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &sixteen,
                                                     sizeof(sixteen))));
        D3D12_RESOURCE_DESC ds_16 = ds_desc;
        ds_16.SampleDesc = {16, 0};
        Ref<ID3D12Resource> unusable;
        const HRESULT sixteen_result = device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &ds_16,
                                                                       D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                                       IID_PPV_ARGS(&unusable.p));
        EXPECT(sixteen.NumQualityLevels ? SUCCEEDED(sixteen_result) : (sixteen_result == E_INVALIDARG && !unusable.p));
        unusable.reset();

        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        std::memcpy(up, vertices, sizeof(vertices));
        std::memcpy(up + 512, tint, sizeof(tint));
        const D3D12_GPU_VIRTUAL_ADDRESS upload_va = upload->GetGPUVirtualAddress();

        const D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 10,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1,
                                                       D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> srv_heap, rtv_heap, dsv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap.p))));
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
        const auto image_at = [&](UINT i, g::Image& image) {
            g::ImageWords words{};
            std::memcpy(words.data(), srv_bytes(i), sizeof(words));
            return g::is_image(words) && g::decode(words, image);
        };

        constexpr UINT depth_view = 0, single_view = 1, non_msaa_view = 2, ms_single_view = 3,
                       other_family_view = 4, colour_format_view = 5, colour_on_depth_view = 6,
                       unknown_view = 7, no_stencil_view = 8;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc{};
        srv_desc.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        device->CreateShaderResourceView(depth.p, &srv_desc, srv_slot(depth_view));
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(depth_single.p, &srv_desc, srv_slot(single_view));

        g::Image image{};
        EXPECT(image_at(depth_view, image));
        EXPECT(image.type == g::ImageType::image_2d_msaa && image.samples == samples && image.width == size &&
               image.height == size && image.first_mip == 0 && image.mip_count == 1 && image.first_layer == 0 &&
               image.layer_count == 1);
        EXPECT(image.format == g::format_from_vulkan(VK_FORMAT_D32_SFLOAT_S8_UINT));
        g::Image single{};
        EXPECT(image_at(single_view, single) && single.type == g::ImageType::image_2d && single.samples == 1 &&
               single.format == image.format);

        srv_desc.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        device->CreateShaderResourceView(depth.p, &srv_desc, srv_slot(non_msaa_view));
        EXPECT(null_at(non_msaa_view));  // a single-sampled view of the 4x depth texture
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
        device->CreateShaderResourceView(depth_single.p, &srv_desc, srv_slot(ms_single_view));
        EXPECT(null_at(ms_single_view));  // a multisampled view of a single-sampled texture
        srv_desc.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        device->CreateShaderResourceView(depth.p, &srv_desc, srv_slot(other_family_view));
        EXPECT(null_at(other_family_view));  // another depth family's plane format
        srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
        device->CreateShaderResourceView(depth.p, &srv_desc, srv_slot(colour_format_view));
        EXPECT(null_at(colour_format_view));  // the depth-plane format of a family without a stencil
        srv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        device->CreateShaderResourceView(depth.p, &srv_desc, srv_slot(colour_on_depth_view));
        EXPECT(null_at(colour_on_depth_view));  // a colour format on a depth texture
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        device->CreateShaderResourceView(depth.p, &srv_desc, srv_slot(unknown_view));
        EXPECT(null_at(unknown_view));  // a typeless depth texture needs its plane format
        srv_desc.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        device->CreateShaderResourceView(depth_typed.p, &srv_desc, srv_slot(no_stencil_view));
        EXPECT(null_at(no_stencil_view));  // a stencil-bearing plane format on a depth texture without a stencil

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
        const D3D12_CPU_DESCRIPTOR_HANDLE dsv = heap_start(dsv_heap.p);
        device->CreateRenderTargetView(colour.p, nullptr, rtv);
        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc{};
        dsv_desc.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMS;
        device->CreateDepthStencilView(depth.p, &dsv_desc, dsv);

        list->SetGraphicsRootSignature(root.p);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        const D3D12_VERTEX_BUFFER_VIEW vbv{vertex_buffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(Vertex)};
        list->IASetVertexBuffers(0, 1, &vbv);
        const D3D12_VIEWPORT viewport{0, 0, float(size), float(size), 0, 1};
        const D3D12_RECT scissor{0, 0, LONG(size), LONG(size)};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
        const float origin[2] = {0, 0};
        list->SetGraphicsRoot32BitConstants(0, 2, origin, 0);

        list->CopyBufferRegion(vertex_buffer.p, 0, upload.p, 0, sizeof(vertices));
        D3D12_RESOURCE_BARRIER to_vertex = transition(vertex_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        list->ResourceBarrier(1, &to_vertex);
        const FLOAT clear_colour[4] = {0, 0, 0, 1};
        list->ClearRenderTargetView(rtv, clear_colour, 0, nullptr);
        list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5f, 0x5a, 0, nullptr);
        list->SetGraphicsRootConstantBufferView(1, upload_va + 512);
        list->DrawInstanced(3, 1, 0, 0);
        D3D12_RESOURCE_BARRIER to_sample = transition(depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1, &to_sample);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

        // Sample the depth SRV through set 0, as a translated shader would from its T#.
        const VkDescriptorSet set0 = rt::d3d12::bindless_set(device.p);
        const VkDescriptorSetLayout set0_layout = rt::d3d12::bindless_set_layout(device.p);
        EXPECT(set0 && set0_layout);
        SampleKeys key{};
        std::memcpy(key.tsharp, srv_bytes(depth_view), sizeof(key.tsharp));
        key.mode = 2;
        EXPECT(sample_through_set(d, set0, set0_layout, key, depth_samples.data()));

        for (auto* r : {&vertex_buffer, &upload, &colour, &depth, &depth_single, &depth_typed}) r->reset();
        pso.reset();
        root.reset();
        srv_heap.reset();
        rtv_heap.reset();
        dsv_heap.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }

    // Sample s of texel (x, y) is depth_samples[16 y + 4 x + s]. The triangle covers the samples above the
    // diagonal: fully covered texels were written by sample 0 (depth 0) and keep 0.5 on the other three, fully
    // uncovered ones read the clear on all four, and the diagonal's partially covered samples are one or the other.
    unsigned mismatches = 0;
    for (UINT y = 0; y < 4; ++y) {
        for (UINT x = 0; x < 4; ++x) {
            for (UINT s = 0; s < 4; ++s) {
                const std::uint32_t got = depth_samples[16 * y + 4 * x + s];
                bool ok = true;
                if (x + y <= 1) ok = got == (s == 0 ? zero_bits : half_bits);
                else if (x + y >= 5) ok = got == half_bits;
                else ok = got == zero_bits || got == half_bits;
                if (!ok && mismatches++ < 8)
                    std::fprintf(stderr, "texel (%u, %u) sample %u: depth bits 0x%08x\n", x, y, s, got);
            }
        }
    }
    EXPECT(!mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("MSAA depth SRV: R32G8X24_TYPELESS 4x target cleared and sample-masked drawn, 64 depth samples "
                "matched through its R32_FLOAT_X8X24_TYPELESS T#; single-sample, other-family, colour, UNKNOWN and "
                "stencil-less plane views and an unsupported sample count were refused\n");
    return 0;
}
