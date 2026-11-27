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
//      it is served by the mirror's chunk;
//   6. a slot-133 buffer over mirrored memory is the mirror's chunk (no shadow); the GPU's writes into it go back to
//      the title's page, and a 256-byte piece written on both sides keeps the title's bytes;
//   7. one page, the GPU writes piece A and the title piece B in the same submission window (the title's write is
//      uploaded before the GPU's goes back): both survive, on both sides;
//   8. XMemAlloc blocks lie in the mirror's range and are mirrored as the heap commits them: the GPU reads a small and
//      a large block's bytes through a pointer into them that nothing named;
//   9. the write-back over 64 MiB (one submission's most) returns the one piece the GPU wrote, in well under a second.
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
#include <chrono>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "page mirror line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocFn = void*(__stdcall*)(size_t, std::uint64_t);
using XMemFreeFn = void(__stdcall*)(void*, std::uint64_t);
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

    // 6. A slot-133 buffer over mirrored memory is the mirror's chunk there: its GPU VA is the title's address, no
    //    shadow. What the GPU writes into its home goes back to the title's page (the pieces that differ), and a page
    //    the title wrote too keeps the title's bytes.
    auto* placed = static_cast<std::uint8_t*>(reserve(nullptr, kPage64k, MEM_RESERVE | MEM_COMMIT, 0, PAGE_READWRITE));
    EXPECT(placed);
    auto* placed_words = reinterpret_cast<volatile std::uint32_t*>(placed);
    for (unsigned i = 0; i < kPage64k / 4; i += 64) placed_words[i] = 0x71000000u + i;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kPage64k;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Ref<ID3D12Resource> buffer;
    EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, placed, desc, 0xFFFFFFFFu, IID_PPV_ARGS(&buffer.p))));
    EXPECT(buffer.p->GetGPUVirtualAddress() == std::uintptr_t(placed) && !rt::d3d12::title_shadow_buffer(buffer.p));
    EXPECT(gpu_word(device.p, std::uintptr_t(placed) + 0x100) == 0x71000040u);  // and the page is guarded now
    const std::uint32_t result[4] = {0xC0DE0001u, 0xC0DE0002u, 0xC0DE0003u, 0xC0DE0004u};
    EXPECT(rt::d3d12::page_mirror_gpu_write(device.p, std::uintptr_t(placed) + 0x200, result, sizeof(result)));
    EXPECT(placed_words[0x80] == 0x71000080u);  // not yet: after the submission retires
    EXPECT(rt::d3d12::page_mirror_write_back(device.p, std::uintptr_t(placed), kPage64k) == 256);
    EXPECT(placed_words[0x80] == 0xC0DE0001u && placed_words[0x83] == 0xC0DE0004u && placed_words[0x40] == 0x71000040u);
    EXPECT(gpu_word(device.p, std::uintptr_t(placed) + 0x200) == 0xC0DE0001u);  // the home agrees after the refill
    // Written on both sides: the title's write wins.
    const std::uint32_t other[4] = {0xBAD00001u, 0xBAD00002u, 0xBAD00003u, 0xBAD00004u};
    EXPECT(rt::d3d12::page_mirror_gpu_write(device.p, std::uintptr_t(placed) + 0x400, other, sizeof(other)));
    std::thread([&] { placed_words[0x100] = 0x7171u; }).join();
    EXPECT(rt::d3d12::page_mirror_write_back(device.p, std::uintptr_t(placed), kPage64k) == 0);
    EXPECT(placed_words[0x100] == 0x7171u);
    EXPECT(gpu_word(device.p, std::uintptr_t(placed) + 0x400) == 0x7171u);

    // 7. Pieces A (0x800) and B (0xc00) of one page: the GPU writes A, the title B; a GPU read refreshes the mirror
    //    first (B is uploaded, A's home bytes stay), then A goes back to the title's page.
    const std::uint32_t gpu_a[4] = {0xA0A00001u, 0xA0A00002u, 0xA0A00003u, 0xA0A00004u};
    EXPECT(rt::d3d12::page_mirror_gpu_write(device.p, std::uintptr_t(placed) + 0x800, gpu_a, sizeof(gpu_a)));
    std::thread([&] { placed_words[0x300] = 0xB0B00001u; }).join();
    EXPECT(gpu_word(device.p, std::uintptr_t(placed) + 0xc00) == 0xB0B00001u);
    EXPECT(gpu_word(device.p, std::uintptr_t(placed) + 0x800) == 0xA0A00001u);
    EXPECT(rt::d3d12::page_mirror_write_back(device.p, std::uintptr_t(placed), kPage64k) == 256);
    EXPECT(placed_words[0x200] == 0xA0A00001u && placed_words[0x203] == 0xA0A00004u);
    EXPECT(placed_words[0x300] == 0xB0B00001u);
    EXPECT(gpu_word(device.p, std::uintptr_t(placed) + 0x800) == 0xA0A00001u);
    EXPECT(gpu_word(device.p, std::uintptr_t(placed) + 0xc00) == 0xB0B00001u);

    // 8. XMemAlloc blocks.
    auto xmem_alloc = reinterpret_cast<XMemAllocFn>(rt::shims::find("xmem.dll", "XMemAlloc"));
    auto xmem_free = reinterpret_cast<XMemFreeFn>(rt::shims::find("xmem.dll", "XMemFree"));
    EXPECT(xmem_alloc && xmem_free);
    auto* heap_small = static_cast<volatile std::uint32_t*>(xmem_alloc(3000, 0x5e50000));
    auto* heap_large = static_cast<volatile std::uint32_t*>(xmem_alloc(200000, 0));
    EXPECT(heap_small && heap_large && std::uintptr_t(heap_small) % 64 == 0);
    EXPECT(leonida::title_mirror_space::contains(std::uintptr_t(heap_small)) &&
           leonida::title_mirror_space::contains(std::uintptr_t(heap_large)));
    heap_small[700] = 0x81000700u;
    heap_large[40000] = 0x82040000u;
    EXPECT(gpu_word(device.p, std::uintptr_t(heap_small) + 2800) == 0x81000700u);
    EXPECT(gpu_word(device.p, std::uintptr_t(heap_large) + 160000) == 0x82040000u);
    std::thread([&] { heap_small[700] = 0x81000701u; }).join();
    EXPECT(gpu_word(device.p, std::uintptr_t(heap_small) + 2800) == 0x81000701u);
    xmem_free(const_cast<std::uint32_t*>(heap_small), 0);
    xmem_free(const_cast<std::uint32_t*>(heap_large), 0);
    EXPECT(gpu_word(device.p, std::uintptr_t(heap_large) + 160000) == 0);  // decommitted: unmapped, reads zero

    // 9. The write-back over 64 MiB of mirrored memory (the most one submission's write-back covers), the GPU having
    //    written one piece of it: that piece goes back, and the call takes well under a second. The homes are copied
    //    out by the GPU; read by the CPU from device-local memory it took 1.6 s for 16 MiB here, 14.5 s in run 64w.
    {
        constexpr SIZE_T kBig = 64u << 20;
        auto* big = static_cast<std::uint8_t*>(reserve(nullptr, kBig, MEM_RESERVE | MEM_COMMIT, 0, PAGE_READWRITE));
        EXPECT(big && rt::d3d12::page_mirror_name(device.p, std::uintptr_t(big), kBig));
        const std::uint32_t mark[4] = {0x9A9A0001u, 0x9A9A0002u, 0x9A9A0003u, 0x9A9A0004u};
        EXPECT(rt::d3d12::page_mirror_gpu_write(device.p, std::uintptr_t(big) + kBig - 0x1000, mark, sizeof(mark)));
        const auto t0 = std::chrono::steady_clock::now();
        const auto back = rt::d3d12::page_mirror_write_back(device.p, std::uintptr_t(big), kBig);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("page mirror: write-back over 64 MiB in %.1f ms\n", ms);
        EXPECT(back == 256 && reinterpret_cast<volatile std::uint32_t*>(big + kBig - 0x1000)[3] == 0x9A9A0004u);
        EXPECT(ms < 1000.0);
    }

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
