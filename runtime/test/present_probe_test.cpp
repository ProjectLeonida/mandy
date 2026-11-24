// LEONIDA_PRESENT_PROBE and LEONIDA_CAPTURE_SOURCE (src/present_probe.h, src/vulkan_present.h), off and on:
//
//   present_probe_test         both unset: no probe line, no source capture
//   present_probe_test --on    both set: every probe line, and the presented texture next to each captured frame
//
// Both forms make the same calls through the Xbox-ABI proxy on the Vulkan device and expect the same results from
// them: a target cleared to blue, one draw without a viewport (not recorded) and one full-screen draw (yellow), then
// PresentX with one plane (the blit), with no plane (blank), with an overlay plane (the composition), and two calls
// that are refused (three planes, and a token that was never handed out). LEONIDA_CAPTURE is on in both, so the
// frames the presenter showed are compared as well: the probe must not change them.
//
// stderr is sent to a file while the calls run. Off, it must hold no `present-probe` line and the capture folder no
// source file. On, it must hold the lines for the clear, both draws, the three presents and the two refusals, and
// the folder the yellow source of frame 1 with its index line.
//
// The shaders are the GLSL of d3d12_draw_test (test/glsl/vs_fullscreen.vert, ps_draw.frag). No game data.
// Needs a window and a Vulkan GPU with presentation; exits 77 without one.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/xbox_present.h"
#include "../src/shader_library.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <io.h>
#include <iterator>
#include <string>
#include <vector>

#if !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_FULLSCREEN_SPV and LEONIDA_TEST_PS_SPV must name the compiled test shaders"
#endif

// To stdout: stderr is in a file while the calls run.
#define EXPECT(condition) do { \
    if (!(condition)) { std::printf("line %d: %s\n", __LINE__, #condition); std::fflush(stdout); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

constexpr UINT width = 320, height = 240;

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

// The line of `text` that starts with `start`, or nothing.
std::string line_with(const std::string& text, const std::string& start) {
    std::size_t at = text.find(start);
    while (at != std::string::npos && at != 0 && text[at - 1] != '\n') at = text.find(start, at + 1);
    if (at == std::string::npos) return {};
    return text.substr(at, text.find('\n', at) - at);
}

std::string hex(std::uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof text, "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// True when `path` is a binary PPM (P6, 8 bits) whose every pixel is r, g, b. `exact`: it must be `width` x `height`,
// as a presented texture is; a swapchain frame has the extent of the window, which a window manager may change.
bool ppm_is(const std::filesystem::path& path, int r, int g, int b, bool exact) {
    const std::string file = read_file(path);
    const std::string name = path.filename().string();
    unsigned w = 0, h = 0, maximum = 0;
    int header = 0;
    if (std::sscanf(file.c_str(), "P6\n%u %u\n%u%n", &w, &h, &maximum, &header) != 3 || maximum != 255 || !w || !h ||
        file.size() != std::size_t(header) + 1 + std::size_t(w) * h * 3 || (exact && (w != width || h != height))) {
        std::printf("%s: not the expected P6 file (%zu bytes, %ux%u)\n", name.c_str(), file.size(), w, h);
        return false;
    }
    const auto* p = reinterpret_cast<const std::uint8_t*>(file.data()) + header + 1;
    for (std::size_t i = 0; i < std::size_t(w) * h; ++i)
        if (p[i * 3] != r || p[i * 3 + 1] != g || p[i * 3 + 2] != b) {
            std::printf("%s: pixel %zu is %d,%d,%d, expected %d,%d,%d\n", name.c_str(), i, p[i * 3], p[i * 3 + 1],
                        p[i * 3 + 2], r, g, b);
            return false;
        }
    return true;
}

int files_named(const std::filesystem::path& dir, const char* prefix) {
    int count = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        if (entry.path().filename().string().rfind(prefix, 0) == 0) ++count;
    return count;
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

struct Ids {
    std::uint64_t scene = 0, overlay = 0;  // the resource ids of the two presented textures
    std::uint64_t first_token = 0, refused_token = 0;
};

// The calls both forms make. 0 when every call answered as expected; stderr is the caller's to redirect.
int present_frames(vktest::Device& d, HWND window, const std::filesystem::path& library, Ids& ids) {
    const auto vs_program = fssm_program({0x56530002, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    const auto vs_container = shader_container(vs_program), ps_container = shader_container(ps_program);
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV), library / "vs_fullscreen.spv"));
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), library / "ps_draw.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
                 " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
              << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    }

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.presentation = true;
    info.swapchain_maintenance1 = d.swapchain_maintenance1;
    info.buffer_device_address = true;
    info.subgroup = d.subgroup;
    info.shader_dir = library.string();
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);
    auto presenter = rt::d3d12::create_window_presenter(inner, window);
    EXPECT(presenter);
    leonida::ManualFrameClock clock;
    auto xbox = std::make_unique<rt::d3d12::XboxPresentation>(std::move(presenter), &clock);

    // Root signature: [0] two constants (s0, s1), [1] a root CBV (s2:s3: the pixel shader's tint).
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
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.VS = {vs_container.data(), vs_container.size()};
    pso_desc.PS = {ps_container.data(), ps_container.size()};
    pso_desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso_desc.SampleMask = 0xffffffff;
    pso_desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso_desc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    pso_desc.RasterizerState.DepthClipEnable = TRUE;
    pso_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso_desc.NumRenderTargets = 1;
    pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso_desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> pso;
    EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    // The tint the pixel shader reads through the root CBV: (1, 1, 1) leaves the vertex shader's yellow.
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    Ref<ID3D12Resource> upload;
    const auto upload_desc = buffer_desc(4096);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));
    std::uint8_t* up = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    const float tint[3] = {1, 1, 1};
    std::memcpy(up, tint, sizeof tint);

    // Two display-capable back buffers in the PRESENT state, each with a render-target view.
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const SIZE_T rtv_start = SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p));
    const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = width;
    texture.Height = height;
    texture.DepthOrArraySize = 1;
    texture.MipLevels = 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture.SampleDesc.Count = 1;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> scene, overlay;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &texture,
                                                     D3D12_RESOURCE_STATE_PRESENT, nullptr, IID_PPV_ARGS(&scene.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &texture,
                                                     D3D12_RESOURCE_STATE_PRESENT, nullptr, IID_PPV_ARGS(&overlay.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE scene_rtv{rtv_start}, overlay_rtv{rtv_start + rtv_step};
    device->CreateRenderTargetView(scene.p, nullptr, scene_rtv);
    device->CreateRenderTargetView(overlay.p, nullptr, overlay_rtv);
    ids.scene = rt::d3d12::resource_id(scene.p);
    ids.overlay = rt::d3d12::resource_id(overlay.p);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p, IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    // One list: the scene cleared to blue, a draw before any viewport is set (D3D12 rasterizes nothing: the layer does
    // not record it), the same draw with a viewport (yellow over the whole target), and the overlay cleared to
    // transparent black.
    D3D12_RESOURCE_BARRIER barriers[2] = {
        transition(scene.p, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET),
        transition(overlay.p, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)};
    list->ResourceBarrier(2, barriers);
    const FLOAT blue[4] = {0, 0, 1, 1}, transparent[4] = {0, 0, 0, 0};
    list->ClearRenderTargetView(scene_rtv, blue, 0, nullptr);
    list->SetGraphicsRootSignature(root.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->OMSetRenderTargets(1, &scene_rtv, FALSE, nullptr);
    list->SetGraphicsRootConstantBufferView(1, upload->GetGPUVirtualAddress());
    list->DrawInstanced(3, 1, 0, 0);
    const D3D12_VIEWPORT viewport{0, 0, float(width), float(height), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(width), LONG(height)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->DrawInstanced(3, 1, 0, 0);
    list->ClearRenderTargetView(overlay_rtv, transparent, 0, nullptr);
    barriers[0] = transition(scene.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    barriers[1] = transition(overlay.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    list->ResourceBarrier(2, barriers);
    EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    // The title's frame loop: wait for the origin, then PresentX with that token.
    const UINT period = rt::d3d12::xbox_frame_interval_60hz_us;
    EXPECT(SUCCEEDED(xbox->SetFrameIntervalX(nullptr, period, 1, 0)));
    EXPECT(SUCCEEDED(xbox->ScheduleFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0)));
    const auto next_token = [&](std::uint64_t& token) {
        clock.advance_us(period);
        return xbox->WaitFrameEventX(rt::d3d12::xbox_frame_event_origin, 0, 0, &token) == S_OK;
    };
    std::uint64_t token = 0;
    EXPECT(next_token(token));
    ids.first_token = token;
    rt::d3d12::XboxPresentPlaneParameters planes[3] = {};
    planes[0] = {token, 1, &scene.p, nullptr, nullptr};
    EXPECT(xbox->PresentX(1, planes, nullptr) == S_OK);  // frame 1: the blit
    EXPECT(xbox->PresentX(0, nullptr, nullptr) == S_OK);  // frame 2: blank
    clock.advance_us(period);
    EXPECT(next_token(token));
    planes[0] = {token, 1, &scene.p, nullptr, nullptr};
    planes[1] = {token, 1, &overlay.p, nullptr, nullptr};
    planes[1].Flags = rt::d3d12::xbox_plane_flag_overlay;
    EXPECT(xbox->PresentX(2, planes, nullptr) == S_OK);  // frame 3: the composition
    EXPECT(xbox->presenter().composed_presents() == 1);
    planes[2] = planes[1];
    EXPECT(xbox->PresentX(3, planes, nullptr) == E_INVALIDARG);  // more than two planes
    ids.refused_token = xbox->pacer().tokens_issued() + 1;
    planes[0] = {ids.refused_token, 1, &scene.p, nullptr, nullptr};
    EXPECT(xbox->PresentX(1, planes, nullptr) == E_INVALIDARG);  // a token that was never handed out
    EXPECT(xbox->presents() == 3);
    // The presenter writes a captured frame when the next one is submitted or when it is destroyed.
    xbox.reset();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool on = argc == 2 && std::strcmp(argv[1], "--on") == 0;
    if (argc > 1 && !on) {
        std::printf("usage: present_probe_test [--on]\n");
        return 2;
    }
    const testfiles::ScratchDir scratch("leonida_present_probe_test");
    const std::filesystem::path library = scratch.path() / "shaders", captures = scratch.path() / "captures";
    std::error_code ec;
    std::filesystem::create_directories(library, ec);
    // The switches are read when the presenter is made and at the first probed call. An empty value removes the
    // variable, so the plain form does not depend on the environment it is started in.
    EXPECT(_putenv_s("LEONIDA_LOCAL_DIAGNOSTICS", "") == 0);
    EXPECT(_putenv_s("LEONIDA_PRESENT_PROBE", on ? "1" : "") == 0);
    EXPECT(_putenv_s("LEONIDA_CAPTURE_SOURCE", on ? "1" : "") == 0);
    EXPECT(_putenv_s("LEONIDA_CAPTURE", captures.string().c_str()) == 0);
    EXPECT(_putenv_s("LEONIDA_CAPTURE_EVERY", "1") == 0);
    EXPECT(_putenv_s("LEONIDA_CAPTURE_LIMIT", "8") == 0);

    vktest::Device d;
    d.request_buffer_device_address = true;  // the draw's pixel shader reads its tint through a buffer reference
    const auto device_status = d.initialize(true);
    if (device_status == vktest::DeviceStatus::unavailable) {
        std::printf("SKIP: no Vulkan device with presentation\n");
        return 77;
    }
    EXPECT(device_status == vktest::DeviceStatus::ready);
    if (!d.buffer_device_address) {
        std::printf("SKIP: the device has no bufferDeviceAddress\n");
        return 77;
    }

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"LeonidaPresentProbeTest";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    EXPECT(RegisterClassW(&window_class));
    RECT frame_rect{0, 0, int(width), int(height)};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida: present probe", WS_OVERLAPPEDWINDOW, 100,
                                  100, frame_rect.right - frame_rect.left, frame_rect.bottom - frame_rect.top, nullptr,
                                  nullptr, window_class.hInstance, nullptr);
    EXPECT(window);
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    pump();

    Ids ids;
    std::string log;
    int result = 1;
    bool redirected = false;
    {
        StderrCapture capture(scratch.path() / "stderr.txt");
        redirected = capture.redirected;
        result = present_frames(d, window, library, ids);
        log = capture.finish();
    }
    DestroyWindow(window);
    if (result != 0) {
        std::printf("%s", log.c_str());  // whatever the layer said on the way to the failed call
        return result;
    }
    EXPECT(redirected);

    // What both forms must leave: three shown frames. The blit and the composition show the yellow scene (the overlay
    // is transparent), the blank present shows black.
    EXPECT(files_named(captures, "frame_") == 3);
    EXPECT(ppm_is(captures / "frame_00001.ppm", 255, 255, 0, false));
    EXPECT(ppm_is(captures / "frame_00002.ppm", 0, 0, 0, false));
    EXPECT(ppm_is(captures / "frame_00003.ppm", 255, 255, 0, false));

    const int probe_lines = count_of(log, "present-probe");
    const int source_files = files_named(captures, "source_");
    if (!on) {
        EXPECT(probe_lines == 0);
        EXPECT(count_of(log, "source capture") == 0);
        EXPECT(source_files == 0);
        std::printf("probe off: 3 frames presented and 2 refused, 0 present-probe lines, 0 source files\n");
        return 0;
    }

    const std::string scene = hex(ids.scene), overlay = hex(ids.overlay);
    // The clears: the scene to blue, the overlay to transparent black.
    EXPECT(count_of(log, "[rt] present-probe: ClearRenderTargetView #") == 2);
    EXPECT(line_with(log, "[rt] present-probe: ClearRenderTargetView #1 ") ==
           "[rt] present-probe: ClearRenderTargetView #1 resource " + scene +
               " mip 0 layers 0+1 colour 0.000 0.000 1.000 1.000 rects 0");
    EXPECT(line_with(log, "[rt] present-probe: ClearRenderTargetView #2 ") ==
           "[rt] present-probe: ClearRenderTargetView #2 resource " + overlay +
               " mip 0 layers 0+1 colour 0.000 0.000 0.000 0.000 rects 0");
    // The draws: the first has no viewport and is not recorded, the second is.
    EXPECT(count_of(log, "[rt] present-probe: draw #") == 2);
    const std::string dropped = line_with(log, "[rt] present-probe: draw #1 ");
    EXPECT(dropped.find("SKIPPED: no viewport") != std::string::npos);
    EXPECT(dropped.find(" rt0 " + scene + " mip 0 viewport 0x0 at 0,0 ") != std::string::npos);
    const std::string recorded = line_with(log, "[rt] present-probe: draw #2 ");
    EXPECT(recorded.find("recorded; 1 skipped") != std::string::npos);
    EXPECT(recorded.find(" rt0 " + scene + " mip 0 viewport 320x240 at 0,0 scissor 0,0-320,240 ") != std::string::npos);
    EXPECT(count_of(log, "[rt] PresentX diag: planes ") == 5);
    EXPECT(count_of(log, "[rt] PresentX diag: refused at line ") == 2);
    EXPECT(count_of(log, "hr 0x80070057") == 2);
    EXPECT(log.find("token " + hex(ids.first_token)) != std::string::npos);
    EXPECT(log.find("token " + hex(ids.refused_token)) != std::string::npos);
    EXPECT(count_of(log, "[rt] present-probe: submit #") == 3);
    const std::string blit = line_with(log, "[rt] present-probe: submit #1 ");
    EXPECT(blit.find("path present(blit) source " + scene + " resolved 1 ") != std::string::npos);
    const std::string blank = line_with(log, "[rt] present-probe: submit #2 ");
    EXPECT(blank.find("path present_blank source 0x0 resolved 0 ") != std::string::npos);
    const std::string compose = line_with(log, "[rt] present-probe: submit #3 ");
    EXPECT(compose.find("path present_planes(compose) source " + scene + " resolved 1 ") != std::string::npos);
    // The source capture: the texture handed to frames 1 and 3 (the blank frame has none), yellow as drawn.
    EXPECT(count_of(log, "[rt] present-probe: source capture on") == 1);
    EXPECT(source_files == 3);  // two images and the index
    EXPECT(ppm_is(captures / "source_00001.ppm", 255, 255, 0, true));
    EXPECT(ppm_is(captures / "source_00003.ppm", 255, 255, 0, true));
    const std::string index = read_file(captures / "source_index.txt");
    EXPECT(count_of(index, "\n") == 2);
    const std::string first = line_with(index, "frame 1 "), third = line_with(index, "frame 3 ");
    EXPECT(first.find("path present(blit) format 37 extent 320x240 layout ") != std::string::npos);
    EXPECT(first.find(" mean rgb 255.00 255.00 0.00 nonzero texels 76800") != std::string::npos);
    EXPECT(third.find("path present_planes(compose) format 37 extent 320x240 layout ") != std::string::npos);
    EXPECT(third.find(" mean rgb 255.00 255.00 0.00 nonzero texels 76800") != std::string::npos);
    std::printf("probe on: %d present-probe lines (2 clears, 2 draws of which 1 not recorded, 5 PresentX of which 2 "
                "refused, 3 submits: blit, blank, compose), 2 source images and their index match the yellow scene\n",
                probe_lines);
    return 0;
}
