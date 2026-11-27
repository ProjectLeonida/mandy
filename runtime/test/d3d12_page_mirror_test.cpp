// The page mirror in the game's device (src/title_mirror.h, DeviceState::refresh_page_mirror) over xmem's pages: the
// stream pool's shape (a CPU view with the title's physical pages, a GPU-only alias that queue slot 31 gives pages by
// table), placed by xmem in the mirror's range. What a shader reads at heap_base + a title address (a V# or pointer the
// title wrote) is read back with the GPU after the mirror's last bind (page_mirror_gpu_read):
//   1. the GPU-only page after a slot-31 copy, its CPU twin and the title's own view hold the title's bytes;
//   2. a write by the title's CPU through its own view reaches all three at the next refresh (the geometry watch's cell
//      over the view faults, the page's home is filled again once);
//   3. a remap of the GPU-only page (another physical page) is a rebind: the next read sees the new page, and no host
//      remap was made;
//   4. a page of the reservation nothing is mapped at reads zero, with no fault;
//   5. ordinary memory (committed XMemVirtualAlloc pages) is mirrored when a reader names it: a constant-buffer view
//      keeps the title's address as its base, the GPU reads the bytes and a later title write there, and a binding of
//      it is served by the mirror's chunk.
// Exit 77 when the device makes no page mirror (no sparse aliased strict buffers or capture-replay addresses: llvmpipe).
// No shaders, no game data.
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../shims/xmem.h"
#include "../src/title_mirror_space.h"
#include "../src/title_page_watch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "page mirror line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, std::uint64_t*, std::uint64_t*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, const std::uint64_t*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, std::uint64_t, const std::uint64_t*);
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
};

std::uint32_t gpu_word(ID3D12Device* device, std::uint64_t va) {
    std::uint32_t word = 0xdeadbeef;
    if (!rt::d3d12::page_mirror_gpu_read(device, va, &word, sizeof(word))) return 0xdeadbeef;
    return word;
}

int test() {
    // Signed V# offsets (the sparse arena needs them): as a pack of sign-extending modules gives.
    _putenv_s("LEONIDA_IMPORT_BELOW_ARENA", "1");
    auto reserve = reinterpret_cast<XMemVirtualAllocFn>(rt::shims::find("xmem.dll", "XMemVirtualAlloc"));
    auto allocate = reinterpret_cast<XMemAllocatePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemAllocatePhysicalPages"));
    auto map = reinterpret_cast<XMemMapPhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemMapPhysicalPages"));
    auto free_pages = reinterpret_cast<XMemFreePhysicalPagesFn>(rt::shims::find("xmem.dll", "XMemFreePhysicalPages"));
    EXPECT(reserve && allocate && map && free_pages);
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) {
        std::puts("SKIP: no standalone Vulkan device");
        return 77;
    }
    const auto state = rt::d3d12::page_mirror_state(device.p);
    if (!state.on) {
        std::puts("SKIP: the device made no page mirror (see its line above)");
        return 77;
    }
    constexpr unsigned n = 8;
    auto window = [&](DWORD protect) {
        return static_cast<std::uint8_t*>(reserve(nullptr, n * kPage64k, MEM_RESERVE | kXmem64kPages, 0, protect));
    };
    auto* pool_cpu = window(PAGE_READWRITE);
    auto* pool = window(0x801);
    auto* cpu = window(PAGE_READWRITE);
    auto* dst = window(0x801);
    EXPECT(pool_cpu && pool && cpu && dst);
    EXPECT(leonida::title_mirror_space::contains(std::uintptr_t(dst)) && leonida::title_gpu_views::to_cpu(std::uintptr_t(dst)) == std::uintptr_t(cpu));
    std::uint64_t pages[2]{}, count = 2;
    EXPECT(allocate(0x280, &count, pages) && count == 2);
    EXPECT(map(pool_cpu, 2, pages) == pool_cpu);
    auto* words = reinterpret_cast<volatile std::uint32_t*>(pool_cpu);
    for (unsigned i = 0; i < 2 * kPage64k / 4; i += 1024) words[i] = 0x51000000u + i;
    words[kPage64k / 4] = 0x52000000u;  // page 1's first word

    // 1. A slot-31 copy onto the GPU-only range (pages found through the pool's CPU view).
    rt::xmem::Slot31Counts before{};
    rt::xmem::slot31_mapping_counts(before);
    EXPECT(rt::xmem::copy_page_mappings_fn && rt::xmem::copy_page_mappings_fn(dst, pool, 2, false));
    EXPECT(gpu_word(device.p, std::uintptr_t(dst) + 4096) == 0x51000400u);
    EXPECT(gpu_word(device.p, std::uintptr_t(cpu) + 4096) == 0x51000400u);
    EXPECT(gpu_word(device.p, std::uintptr_t(pool_cpu) + 4096) == 0x51000400u);
    EXPECT(gpu_word(device.p, std::uintptr_t(dst) + kPage64k) == 0x52000000u);

    // 2. The title's CPU writes through its own view: one fault, one upload, seen at every address of the page.
    const auto uploads = rt::d3d12::page_mirror_state(device.p).uploads;
    std::thread([&] { words[1024] = 0x5eed0001u; }).join();
    EXPECT(gpu_word(device.p, std::uintptr_t(dst) + 4096) == 0x5eed0001u);
    EXPECT(gpu_word(device.p, std::uintptr_t(cpu) + 4096) == 0x5eed0001u);
    EXPECT(rt::d3d12::page_mirror_state(device.p).uploads == uploads + 1);

    // 3. A remap of the GPU-only page to the pool's second page: a rebind, no host remap.
    EXPECT(rt::xmem::copy_page_mappings_fn(dst, pool + kPage64k, 1, false));
    EXPECT(gpu_word(device.p, std::uintptr_t(dst)) == 0x52000000u);
    EXPECT(gpu_word(device.p, std::uintptr_t(cpu)) == 0x52000000u);
    rt::xmem::Slot31Counts after{};
    rt::xmem::slot31_mapping_counts(after);
    EXPECT(after.host() == before.host());

    // 4. A page of the reservation with nothing mapped.
    EXPECT(gpu_word(device.p, std::uintptr_t(dst) + 5 * kPage64k) == 0);

    // 5. Ordinary memory (an XMemVirtualAlloc region of committed pages) is mirrored when a reader names it: a
    //    constant-buffer view over it keeps the title's address as its V# base (heap_base + it is the mirror, no copy),
    //    the GPU reads the bytes there, and a title write reaches it at the next refresh. A draw's binding of it is
    //    served by the mirror's chunk.
    auto* plain = static_cast<std::uint8_t*>(reserve(nullptr, 2 * kPage64k, MEM_RESERVE | MEM_COMMIT, 0, PAGE_READWRITE));
    EXPECT(plain && leonida::title_mirror_space::contains(std::uintptr_t(plain)));
    auto* plain_words = reinterpret_cast<volatile std::uint32_t*>(plain);
    plain_words[0x40] = 0x61000040u;
    std::uint64_t unnamed = 64;
    EXPECT(!rt::d3d12::page_mirror_buffer_at(device.p, std::uintptr_t(plain), unnamed));  // not named yet
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(device.p->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE handle = heap.p->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CONSTANT_BUFFER_VIEW_DESC view{std::uintptr_t(plain), 0x200};
    device.p->CreateConstantBufferView(&view, handle);
    std::uint32_t vsharp[4]{};
    std::memcpy(vsharp, reinterpret_cast<const void*>(handle.ptr), sizeof(vsharp));
    const std::uint64_t base = std::uint64_t(vsharp[0]) | (std::uint64_t(vsharp[1] & 0xffffu) << 32);
    EXPECT(base == (std::uintptr_t(plain) & 0xFFFFFFFFFFFFull));
    EXPECT(gpu_word(device.p, std::uintptr_t(plain) + 0x100) == 0x61000040u);
    std::thread([&] { plain_words[0x40] = 0x61000041u; }).join();
    EXPECT(gpu_word(device.p, std::uintptr_t(plain) + 0x100) == 0x61000041u);
    std::uint64_t binding = 0x200;
    EXPECT(rt::d3d12::page_mirror_buffer_at(device.p, std::uintptr_t(plain), binding) && binding == 0x200);
    // The second page, named on its own (a vertex view's range), and a GPU-only page (physical: mirrored as mapped).
    plain_words[kPage64k / 4] = 0x62000000u;
    EXPECT(rt::d3d12::page_mirror_name(device.p, std::uintptr_t(plain) + kPage64k, 64));
    EXPECT(gpu_word(device.p, std::uintptr_t(plain) + kPage64k) == 0x62000000u);
    std::uint64_t pool_binding = 256;
    EXPECT(rt::d3d12::page_mirror_buffer_at(device.p, std::uintptr_t(dst), pool_binding));

    const auto end = rt::d3d12::page_mirror_state(device.p);
    std::printf("page mirror: heap_base 0x%llx, %llu pages in %llu homes over %llu chunk(s); slot-31 copies, a title "
                "write through its own view and a remap reach heap_base + the GPU-only address; an unmapped page reads "
                "zero\n",
                static_cast<unsigned long long>(end.heap_base), static_cast<unsigned long long>(end.pages),
                static_cast<unsigned long long>(end.homes), static_cast<unsigned long long>(end.chunks));
    EXPECT(rt::xmem::copy_page_mappings_fn(dst, nullptr, 2, false));
    EXPECT(map(pool_cpu, 2, nullptr) == pool_cpu && free_pages(2, pages));
    return 0;
}

}  // namespace

int main() { return test(); }
