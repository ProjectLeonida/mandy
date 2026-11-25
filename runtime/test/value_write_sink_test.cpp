// Portable unit tests for leonida::value_sink (src/value_write_sink.h).
// Verifies arena dispatch, title_pages dispatch, null/invalid size guards,
// device-matched clearing, and thread safety.
// No Windows headers, no Vulkan, no game data.
#include "../src/value_write_sink.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1; \
    } \
} while (0)

namespace {

using namespace leonida::value_sink;

int test_initial_and_reset_state() {
    reset_for_test();
    const std::uint64_t val = 0x1122334455667788ull;
    EXPECT(arena(0x1000, &val, 4) == Landing::elsewhere);
    EXPECT(arena(0x2000, &val, 8) == Landing::elsewhere);
    EXPECT(!title_pages(reinterpret_cast<const void*>(0x3000), &val, 4));
    EXPECT(!title_pages(reinterpret_cast<const void*>(0x4000), &val, 8));
    return 0;
}

int test_null_and_invalid_size_guards() {
    reset_for_test();
    const std::uint32_t val = 0xabcdef01;
    // arena() guards
    EXPECT(arena(0x1000, nullptr, 4) == Landing::refused);
    EXPECT(arena(0x1000, &val, 0) == Landing::refused);
    EXPECT(arena(0x1000, &val, 1) == Landing::refused);
    EXPECT(arena(0x1000, &val, 2) == Landing::refused);
    EXPECT(arena(0x1000, &val, 3) == Landing::refused);
    EXPECT(arena(0x1000, &val, 5) == Landing::refused);
    EXPECT(arena(0x1000, &val, 7) == Landing::refused);
    EXPECT(arena(0x1000, &val, 9) == Landing::refused);
    EXPECT(arena(0x1000, &val, 16) == Landing::refused);

    // title_pages() guards
    EXPECT(!title_pages(nullptr, &val, 4));
    EXPECT(!title_pages(reinterpret_cast<const void*>(0x1000), nullptr, 4));
    EXPECT(!title_pages(reinterpret_cast<const void*>(0x1000), &val, 0));
    return 0;
}

struct MockDevice {
    std::uint64_t last_address = 0;
    std::uint64_t last_value = 0;
    unsigned last_size = 0;
    Landing answer = Landing::landed;
};

Landing mock_arena_write(void* device, std::uint64_t address, const void* bytes, unsigned size) {
    auto* dev = static_cast<MockDevice*>(device);
    dev->last_address = address;
    dev->last_size = size;
    if (size == 4) {
        dev->last_value = *static_cast<const std::uint32_t*>(bytes);
    } else if (size == 8) {
        dev->last_value = *static_cast<const std::uint64_t*>(bytes);
    }
    return dev->answer;
}

int test_arena_registration_dispatch_and_clear() {
    reset_for_test();
    MockDevice dev1;
    MockDevice dev2;

    set_arena(&mock_arena_write, &dev1);

    const std::uint32_t v32 = 0x12345678;
    dev1.answer = Landing::landed;
    EXPECT(arena(0x10000, &v32, 4) == Landing::landed);
    EXPECT(dev1.last_address == 0x10000);
    EXPECT(dev1.last_value == 0x12345678);
    EXPECT(dev1.last_size == 4);

    const std::uint64_t v64 = 0xdeadbeefcafebabeull;
    dev1.answer = Landing::refused;
    EXPECT(arena(0x20000, &v64, 8) == Landing::refused);
    EXPECT(dev1.last_address == 0x20000);
    EXPECT(dev1.last_value == 0xdeadbeefcafebabeull);
    EXPECT(dev1.last_size == 8);

    dev1.answer = Landing::elsewhere;
    EXPECT(arena(0x30000, &v32, 4) == Landing::elsewhere);

    // Mismatched device clear: clear_arena(&dev2) does not clear dev1
    clear_arena(&dev2);
    dev1.answer = Landing::landed;
    EXPECT(arena(0x40000, &v32, 4) == Landing::landed);

    // Null device clear does not clear
    clear_arena(nullptr);
    EXPECT(arena(0x40000, &v32, 4) == Landing::landed);

    // Matching device clear: clears
    clear_arena(&dev1);
    EXPECT(arena(0x40000, &v32, 4) == Landing::elsewhere);

    return 0;
}

struct MockTitlePages {
    const void* last_address = nullptr;
    std::uint32_t last_value = 0;
    std::size_t last_size = 0;
    bool answer = true;
};

static MockTitlePages g_mock_title_pages;

bool mock_title_pages_write(const void* address, const void* bytes, std::size_t size) {
    g_mock_title_pages.last_address = address;
    g_mock_title_pages.last_size = size;
    if (size == 4) {
        g_mock_title_pages.last_value = *static_cast<const std::uint32_t*>(bytes);
    }
    return g_mock_title_pages.answer;
}

int test_title_pages_registration_dispatch_and_clear() {
    reset_for_test();
    g_mock_title_pages = MockTitlePages{};

    set_title_pages(&mock_title_pages_write);

    const std::uint32_t v32 = 0x87654321;
    const void* target = reinterpret_cast<const void*>(0x50000);
    g_mock_title_pages.answer = true;
    EXPECT(title_pages(target, &v32, 4));
    EXPECT(g_mock_title_pages.last_address == target);
    EXPECT(g_mock_title_pages.last_value == 0x87654321);
    EXPECT(g_mock_title_pages.last_size == 4);

    g_mock_title_pages.answer = false;
    EXPECT(!title_pages(target, &v32, 4));

    clear_title_pages();
    EXPECT(!title_pages(target, &v32, 4));

    return 0;
}

int test_concurrency() {
    reset_for_test();
    MockDevice dev;
    dev.answer = Landing::landed;
    set_arena(&mock_arena_write, &dev);
    set_title_pages(&mock_title_pages_write);

    std::atomic<bool> running{true};
    std::atomic<std::uint32_t> error_count{0};

    auto reader = [&] {
        const std::uint32_t v = 0x55aa55aa;
        while (running.load(std::memory_order_relaxed)) {
            const Landing l = arena(0x1000, &v, 4);
            if (l != Landing::landed && l != Landing::elsewhere) {
                error_count.fetch_add(1, std::memory_order_relaxed);
            }
            title_pages(reinterpret_cast<const void*>(0x2000), &v, 4);
        }
    };

    auto arena_toggler = [&] {
        for (int i = 0; i < 2000; ++i) {
            set_arena(&mock_arena_write, &dev);
            clear_arena(&dev);
        }
    };

    auto title_pages_toggler = [&] {
        for (int i = 0; i < 2000; ++i) {
            set_title_pages(&mock_title_pages_write);
            clear_title_pages();
        }
    };

    std::vector<std::thread> threads;
    threads.emplace_back(reader);
    threads.emplace_back(reader);
    threads.emplace_back(arena_toggler);
    threads.emplace_back(title_pages_toggler);

    for (std::size_t i = 2; i < threads.size(); ++i) {
        threads[i].join();
    }
    running.store(false, std::memory_order_relaxed);
    threads[0].join();
    threads[1].join();

    EXPECT(error_count.load() == 0);
    reset_for_test();
    return 0;
}

}  // namespace

int main() {
    if (test_initial_and_reset_state() != 0) return 1;
    if (test_null_and_invalid_size_guards() != 0) return 2;
    if (test_arena_registration_dispatch_and_clear() != 0) return 3;
    if (test_title_pages_registration_dispatch_and_clear() != 0) return 4;
    if (test_concurrency() != 0) return 5;
    std::printf("OK: value_write_sink_test passed all checks\n");
    return 0;
}
