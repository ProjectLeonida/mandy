// Present a small solid texture through the window blit and the PresentX plane composer. Only the composed
// image has a black lower half. Decode the PNGs independently with the conformance tool's WIC reader.
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <objbase.h>

#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "../src/vulkan_present.h"
#include "../../tools/conformance/native/src/png_io.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string_view>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

int main(int argc, char** argv) {
    using namespace leonida;
    namespace fs = std::filesystem;
    vktest::Device d;
    const auto status = d.initialize(true);
    if (status == vktest::DeviceStatus::unavailable) return 77;
    EXPECT(status == vktest::DeviceStatus::ready);
    EXPECT(SUCCEEDED(CoInitializeEx(nullptr, COINIT_MULTITHREADED)));
    testfiles::ScratchDir scratch("leonida_present_dump");
    const fs::path dir = scratch.path() / "frames";
    const std::string_view mode = argc == 2 ? argv[1] : "";
    EXPECT(mode.empty() || mode == "--default" || mode == "--invalid" || mode == "--splash");
    const int interval = mode == "--default" ? 60 : 2;
    const auto setting = dir.wstring() + (mode == "--default" ? L"" : mode == "--invalid" ? L",0" : L",2");
    EXPECT(_wputenv_s(L"LEONIDA_PRESENT_DUMP", setting.c_str()) == 0);
    // An earlier dump might be a hard link outside the requested directory. Replacing it must leave that file intact.
    const auto sentinel = scratch.path() / "outside";
    if (mode == "--default") {
        EXPECT(fs::create_directory(dir));
        EXPECT(std::ofstream(sentinel).put('X'));
        std::error_code ec;
        fs::create_hard_link(sentinel, dir / "present_000001.png", ec);  // optional on filesystems without links
    }

    constexpr int width = 192, height = 100;
    const HINSTANCE instance = GetModuleHandleW(nullptr);
    HWND window = CreateWindowExW(0, L"STATIC", L"Leonida present dump test", WS_POPUP, 0, 0, width, height,
                                  nullptr, nullptr, instance, nullptr);
    EXPECT(window);
    ShowWindow(window, SW_SHOWNOACTIVATE);
    VulkanContext context(d.device_proc, d.device, true);
    VulkanQueue queue(context, d.queue, d.family);
    EXPECT(context.loaded() && queue.ok());
    ResourceBinding back{};
    back.format = VK_FORMAT_R8G8B8A8_UNORM;
    back.width = back.height = 2;
    back.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    back.sampled = back.color_attachment = true;
    back.image = d.image(back.format, 2, 2, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    EXPECT(back.image);
    context.add_resource(1, back, VK_IMAGE_LAYOUT_UNDEFINED);
    ResourceBinding splash = back;
    splash.image = d.image(splash.format, 2, 2, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    EXPECT(splash.image);
    context.add_resource(2, splash, VK_IMAGE_LAYOUT_UNDEFINED);
    CommandList commands;
    EXPECT(commands.transition({1, 0, d3d12_state::common, d3d12_state::render_target}));
    EXPECT(commands.clear_render_target({1, 0, 0, 1}, {1.0f, 0.4f, 0.0f, 1.0f}));
    EXPECT(commands.transition({1, 0, d3d12_state::render_target, d3d12_state::common}));
    EXPECT(commands.transition({2, 0, d3d12_state::common, d3d12_state::render_target}));
    EXPECT(commands.clear_render_target({2, 0, 0, 1}, {0.0f, 0.0f, 1.0f, 1.0f}));
    EXPECT(commands.transition({2, 0, d3d12_state::render_target, d3d12_state::common}));
    EXPECT(commands.close());
    const CommandList* list = &commands;
    EXPECT(queue.execute({&list, 1}) == ReplayResult::success);

    VkWin32SurfaceCreateInfoKHR info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    info.hinstance = instance;
    info.hwnd = window;
    VkSurfaceKHR surface{};
    EXPECT(d.instance_fn<PFN_vkCreateWin32SurfaceKHR>("vkCreateWin32SurfaceKHR")(
        d.instance, &info, nullptr, &surface) == VK_SUCCESS);
    {
        VulkanPresenter presenter(context, d.instance_proc, d.instance, d.physical, d.queue, d.family, surface,
                                  false, [] { return VkExtent2D{width, height}; });
        EXPECT(presenter.ok());
        if (mode == "--splash") {
            // Launch splash enables capture only for splash handoff; dismissing it on a non-black title frame
            // must keep the capture buffer alive while LEONIDA_PRESENT_DUMP is active.
            EXPECT(presenter.present_splash(2, std::make_shared<int>(1)));  // frame 1 (due: blue splash)
            EXPECT(presenter.present_blank(true) == VK_SUCCESS);            // frame 2 (splash stays up)
            EXPECT(presenter.present(1, true) == VK_SUCCESS);               // frame 3 (handoff + due: orange)
            EXPECT(presenter.present_blank(true) == VK_SUCCESS);            // frame 4
            PresentPlanes planes;
            planes.plane[0].source = 1;
            planes.plane[0].dest[3] = 0.5f;
            EXPECT(presenter.present_planes(planes, true) == VK_SUCCESS);   // frame 5 (post-handoff due: composed)
            EXPECT(presenter.presented() == 5 && presenter.composed_presents() == 1);
        } else {
            EXPECT(presenter.present(1, true) == VK_SUCCESS);
            for (int i = 1; i < interval; ++i) EXPECT(presenter.present_blank(true) == VK_SUCCESS);
            PresentPlanes planes;
            planes.plane[0].source = 1;
            planes.plane[0].dest[3] = 0.5f;
            EXPECT(presenter.present_planes(planes, true) == VK_SUCCESS);
            EXPECT(presenter.presented() == unsigned(interval + 1) && presenter.composed_presents() == 1);
        }
    }
    DestroyWindow(window);
    EXPECT(d.validation_errors() == 0);
    if (mode == "--invalid") {
        EXPECT(!fs::exists(dir));
        CoUninitialize();
        return 0;
    }
    if (mode == "--default") EXPECT(fs::file_size(sentinel) == 1 && std::ifstream(sentinel).get() == 'X');
    const auto verify_png = [&](const char* name, std::array<std::uint8_t, 4> top, std::array<std::uint8_t, 4> bottom) {
        PixelSurface png;
        EXPECT(ReadPngFile(dir / name, png));
        EXPECT(png.width == width && png.height == height);
        for (int y = 0; y < height; ++y) {
            const auto expected = y >= height / 2 ? bottom : top;
            for (int x = 0; x < width; ++x)
                EXPECT(std::equal(expected.begin(), expected.end(), png.At(x, y)));
        }
        return 0;
    };
    constexpr std::array<std::uint8_t, 4> orange{255, 102, 0, 255};
    constexpr std::array<std::uint8_t, 4> blue{0, 0, 255, 255};
    constexpr std::array<std::uint8_t, 4> black{0, 0, 0, 255};
    if (mode == "--splash") {
        if (verify_png("present_000001.png", blue, blue) != 0) return 1;
        if (verify_png("present_000003.png", orange, orange) != 0) return 1;
        if (verify_png("present_000005.png", orange, black) != 0) return 1;
        EXPECT(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 3);
    } else {
        if (verify_png("present_000001.png", orange, orange) != 0) return 1;
        const auto* composed_name = interval == 2 ? "present_000003.png" : "present_000061.png";
        if (verify_png(composed_name, orange, black) != 0) return 1;
        EXPECT(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 2);
    }
    CoUninitialize();
    return 0;
}
