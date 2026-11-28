// A console descriptor heap read in place (DeviceState::add_heap_in_place): its pages are the title's, in a reservation
// xmem placed in the page mirror's range; shaders read them at heap_base + an offset below 4 GiB (the pages bound a
// second time into the arena), through the mirror's homes, which the filter fills (filter_heap_page). Checked, with the
// GPU reading where a shader would (arena_gpu_read) and the CPU reading the title's own memory:
//   1. the title's bytes are never changed: what the title wrote stays, byte for byte, while the GPU reads the filtered
//      slots: a V# whose base is nobody's (past the mirror's range) has no base and no records there, a data word in a
//      V#'s place reads null, a V# into the arena and one into the mirror read exactly as written, a zero slot zero;
//   2. a slot the title rewrites (through its own view: a guard fault) reaches the GPU at the next refresh, filtered;
//      the title's bytes are again as written;
//   3. the null slots after the heap read zero, and the heap's address is below 4 GiB;
//   4. no copy: the device names it "read in place";
//   5. a T# in the heap (the words CreateShaderResourceView wrote for a texture of ours) reads with the bindless index
//      in its dword 6 (gfx10_descriptors.h, tsharp_index_word: the tag and the texture's index) once the filter and its
//      finishing have run, while the title's bytes keep the dword 6 the title wrote;
//   6. a texture the title places in its own memory (device slot 133, in a reservation at 16 TiB and up) where a slot-70
//      view had made a texture of the layer's first, named by a T# the title writes itself (its dword 6 the title's, so
//      the T# is found by its address), binds: tag and index (the slot-70 texture's retirement used to take the
//      address from the placed one, and the T# found no texture); when the texture goes the slot reads the fallback
//      (0: vulkan_bindless.h's 1x1 zeros), and a texture placed again at the address binds it again;
//   7. data the title keeps inline in a heap page (four words a compute program reads as sizes, which look like a V# with
//      a base nothing holds) and a V# of the title pointing at them: after a slot of the same page is rewritten (the
//      filter runs), the V#'s address (the title's address, through the mirror) reads the title's four words exactly,
//      the table address reads the filtered entries (that "V#" null whole, the rewritten slot filtered), and the
//      title's bytes are unchanged. With one filtered home bound at both addresses, the V# read [0 0 0 720].
// Exit 77 when the device makes no page mirror (llvmpipe). No shaders, no game data.
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../shims/xmem.h"
#include "../src/gfx10_descriptors.h"
#include "../src/title_mirror_space.h"
#include "../src/title_va_alias.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "heap in place line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using XMemVirtualAllocFn = void*(WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
using XMemAllocatePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, std::uint64_t*, std::uint64_t*);
using XMemFreePhysicalPagesFn = BOOL(WINAPI*)(std::uint64_t, const std::uint64_t*);
using XMemMapPhysicalPagesFn = void*(WINAPI*)(void*, std::uint64_t, const std::uint64_t*);
constexpr DWORD kXmem64kPages = 0x20400000;
constexpr SIZE_T kPage64k = 0x10000;
constexpr unsigned kPages = 4;
constexpr UINT kDescriptor = 32;

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

leonida::gfx10::BufferWords raw_vsharp(std::uint64_t address, std::uint32_t bytes) {
    leonida::gfx10::Buffer b{};
    b.address = address;
    b.num_records = bytes;
    b.format = leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT);
    b.oob_select = leonida::gfx10::OobSelect::raw;
    return leonida::gfx10::encode(b);
}
// Device slot 133 (CreatePlacedResourceX) as the title calls it: (va, desc, layout, clear, riid, out).
using SlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t, std::uint64_t);
HRESULT place(void* xbox, void* va, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** out) {
    void* const* vtable = *static_cast<void* const* const*>(xbox);
    const GUID iid = __uuidof(ID3D12Resource);
    const std::uint64_t r = reinterpret_cast<SlotFn>(vtable[133])(
        xbox, reinterpret_cast<std::uint64_t>(va), reinterpret_cast<std::uint64_t>(&desc), 0xFFFFFFFFull, 0,
        reinterpret_cast<std::uint64_t>(&iid), reinterpret_cast<std::uint64_t>(out));
    return static_cast<HRESULT>(static_cast<std::uint32_t>(r));
}

std::uint64_t base_of(const std::uint32_t* w) { return std::uint64_t(w[0]) | (std::uint64_t(w[1] & 0xFFFFu) << 32); }

int test() {
    _putenv_s("LEONIDA_IMPORT_BELOW_ARENA", "1");  // signed V# offsets, as the sparse arena needs
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
    if (!rt::d3d12::page_mirror_state(device.p).on) {
        std::puts("SKIP: the device made no page mirror (see its line above)");
        return 77;
    }
    // The title's heap memory: physical pages in a reservation in the mirror's range.
    auto* title = static_cast<std::uint8_t*>(reserve(nullptr, kPages * kPage64k, MEM_RESERVE | kXmem64kPages, 0,
                                                     PAGE_READWRITE));
    EXPECT(title && leonida::title_mirror_space::contains(std::uintptr_t(title)));
    std::uint64_t pages[kPages]{}, count = kPages;
    EXPECT(allocate(0x280, &count, pages) && count == kPages);
    EXPECT(map(title, kPages, pages) == title);
    std::memset(title, 0, kPages * kPage64k);

    // A buffer of ours in the arena.
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 64 * 1024;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> buffer;
    EXPECT(SUCCEEDED(device.p->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                       D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&buffer.p))));
    const std::uint64_t arena_va = buffer.p->GetGPUVirtualAddress();
    const std::uint64_t nobody = 0x500000000000ull;  // 80 TiB: past the mirror's range, nothing of ours
    const std::uint64_t mirrored = std::uintptr_t(title) + 3 * kPage64k;

    const auto put = [&](unsigned slot, const leonida::gfx10::BufferWords& w) {
        std::memcpy(title + slot * kDescriptor, w.data(), sizeof(w));
    };
    // A texture of ours and its T#, as CreateShaderResourceView writes it into a CPU heap (before the index word is on:
    // dword 6 as encode() writes it). The index word is then on, as a library of modules that read it turns it on.
    D3D12_RESOURCE_DESC tex_desc{};
    tex_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex_desc.Width = 64;
    tex_desc.Height = 64;
    tex_desc.DepthOrArraySize = 1;
    tex_desc.MipLevels = 1;
    tex_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex_desc.SampleDesc.Count = 1;
    Ref<ID3D12Resource> texture;
    EXPECT(SUCCEEDED(device.p->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc,
                                                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
                                                       IID_PPV_ARGS(&texture.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC cpu_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                                              0};
    Ref<ID3D12DescriptorHeap> cpu_heap;
    EXPECT(SUCCEEDED(device.p->CreateDescriptorHeap(&cpu_desc, IID_PPV_ARGS(&cpu_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE cpu = cpu_heap.p->GetCPUDescriptorHandleForHeapStart();
    device.p->CreateShaderResourceView(texture.p, nullptr, cpu);
    std::uint32_t tsharp[8];
    std::memcpy(tsharp, reinterpret_cast<const void*>(cpu.ptr), sizeof(tsharp));
    leonida::gfx10::ImageWords tsharp_words{};
    std::memcpy(tsharp_words.data(), tsharp, sizeof(tsharp_words));
    EXPECT(leonida::gfx10::is_image(tsharp_words) && !leonida::gfx10::has_tsharp_index(tsharp_words));
    tsharp[leonida::gfx10::kTsharpIndexDword] = 0x00C0FFEEu;  // the title's own dword 6 (compression metadata)
    rt::d3d12::set_tsharp_index_words_for_test(device.p, true);

    put(0, raw_vsharp(arena_va, 256));
    put(1, raw_vsharp(nobody, 256));
    put(2, raw_vsharp(mirrored, 256));
    const std::uint32_t data[8] = {0x12345000u, 0x00007777u, 0x00000400u, 0x40000000u, 1, 2, 3, 4};
    std::memcpy(title + 3 * kDescriptor, data, sizeof(data));
    std::memcpy(title + 6 * kDescriptor, tsharp, sizeof(tsharp));
    std::uint8_t written[5 * kDescriptor];
    std::memcpy(written, title, sizeof(written));

    const UINT slots = UINT(kPages * kPage64k / kDescriptor);
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots,
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(rt::d3d12::create_console_descriptor_heap(device.p, &heap_desc, std::uintptr_t(title),
                                                               kPages * kPage64k, IID_PPV_ARGS(&heap.p))));
    D3D12_DESCRIPTOR_HEAP_DESC got{};
    D3D12_GPU_VIRTUAL_ADDRESS shader_va = 0;
    EXPECT(rt::d3d12::descriptor_heap_info(heap.p, got, shader_va));
    EXPECT(shader_va && shader_va + kPages * kPage64k <= (1ull << 32));  // 3: 32-bit table pointers reach it

    // 1. The GPU reads the filtered slots; the title's bytes are as written.
    std::uint32_t gpu[5][8]{};
    EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va, gpu, sizeof(gpu)));
    EXPECT(!std::memcmp(gpu[0], written, kDescriptor));                    // arena V#: as written
    EXPECT(base_of(gpu[1]) == 0 && gpu[1][2] == 0);                        // nobody's: no base, no records
    EXPECT(!std::memcmp(gpu[2], written + 2 * kDescriptor, kDescriptor));  // mirror V#: as written
    EXPECT(!(gpu[3][0] | gpu[3][1] | gpu[3][2] | gpu[3][3]));              // data in a V#'s place: null
    EXPECT(!(gpu[4][0] | gpu[4][1] | gpu[4][2] | gpu[4][3]));              // zero
    EXPECT(!std::memcmp(title, written, sizeof(written)));                 // the title's bytes untouched
    std::uint32_t nobody_words[8];
    std::memcpy(nobody_words, title + kDescriptor, sizeof(nobody_words));
    EXPECT(base_of(nobody_words) == nobody);

    // 2. The title rewrites slot 1 through its own view, from another thread (its write faults on the mirror's
    // guard): the next refresh filters it; the title keeps what it wrote.
    const auto rewritten = raw_vsharp(arena_va + 512, 128);
    std::thread([&] { put(1, rewritten); }).join();
    EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + kDescriptor, gpu[1], kDescriptor));
    EXPECT(!std::memcmp(gpu[1], rewritten.data(), sizeof(rewritten)));
    EXPECT(!std::memcmp(title + kDescriptor, rewritten.data(), sizeof(rewritten)));
    // And back to nobody's: filtered again, the title's bytes as written.
    std::thread([&] { put(1, raw_vsharp(nobody, 256)); }).join();
    EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + kDescriptor, gpu[1], kDescriptor));
    EXPECT(base_of(gpu[1]) == 0 && gpu[1][2] == 0);
    std::memcpy(nobody_words, title + kDescriptor, sizeof(nobody_words));
    EXPECT(base_of(nobody_words) == nobody);

    // A store the GPU makes into the heap's pages through the title's address (a V# of the title into its heap) goes
    // back to the title's bytes as any page's does: the home there is the title's bytes. The table address, a filtered
    // copy of its own, takes it as a rewrite: these words read as a T# naming no texture, left as written there
    // apart from dword 6, which is the fallback (0) while nothing binds it.
    const std::uint32_t stored[8] = {0xdead0001u, 0xdead0002u, 0xdead0003u, 0xdead0004u, 5, 6, 7, 8};
    EXPECT(rt::d3d12::page_mirror_gpu_write(device.p, std::uintptr_t(title) + 5 * kDescriptor, stored, sizeof(stored)));
    EXPECT(rt::d3d12::page_mirror_write_back(device.p, std::uintptr_t(title), kPages * kPage64k) > 0);
    EXPECT(!std::memcmp(title + 5 * kDescriptor, stored, sizeof(stored)));
    std::uint32_t stored_entry[8]{};
    EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 5 * kDescriptor, stored_entry, sizeof(stored_entry)));
    for (unsigned i = 0; i < 8; ++i)
        EXPECT(stored_entry[i] == (i == leonida::gfx10::kTsharpIndexDword ? 0u : stored[i]));
    std::thread([&] { std::memset(title + 5 * kDescriptor, 0, kDescriptor); }).join();  // as before, for the cases below

    // 5. The T#: dword 6 carries the tag and the texture's index, the other words as written; the title's dword 6 is
    // its own.
    std::uint32_t gpu_tsharp[8]{};
    EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 6 * kDescriptor, gpu_tsharp, sizeof(gpu_tsharp)));
    const std::uint32_t word = gpu_tsharp[leonida::gfx10::kTsharpIndexDword];
    EXPECT((word >> 28) == leonida::gfx10::kTsharpIndexTag && (word & 0xFFFFu) != 0);
    for (unsigned i = 0; i < 8; ++i)
        if (i != leonida::gfx10::kTsharpIndexDword) EXPECT(gpu_tsharp[i] == tsharp[i]);
    EXPECT(!std::memcmp(title + 6 * kDescriptor, tsharp, sizeof(tsharp)));
    EXPECT(reinterpret_cast<const std::uint32_t*>(title + 6 * kDescriptor)[leonida::gfx10::kTsharpIndexDword] ==
           0x00C0FFEEu);

    // 6. A texture the title places in its own memory and names by a T# it writes.
    {
        auto* pixels = static_cast<std::uint8_t*>(reserve(nullptr, kPage64k, MEM_RESERVE | kXmem64kPages, 0,
                                                          PAGE_READWRITE));
        std::uint64_t pixel_page = 0, one = 1;
        EXPECT(pixels && leonida::title_mirror_space::contains(std::uintptr_t(pixels)));
        EXPECT(allocate(0x280, &one, &pixel_page) && one == 1 && map(pixels, 1, &pixel_page) == pixels);
        std::memset(pixels, 0x40, kPage64k);
        device.p->AddRef();  // the proxy takes one reference
        Ref<ID3D12Device> xbox;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(device.p, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&xbox.p))));
        rt::d3d12::set_device_outer(device.p, xbox.p);
        D3D12_RESOURCE_DESC placed_desc = tex_desc;
        placed_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x100);
        // First a slot-70 view of the memory, before any texture is placed there: the layer makes a texture of its
        // own for it (the title's order for memory it streams into and then places a texture on).
        std::array<std::uint32_t, 14> resource_words{};
        std::memcpy(resource_words.data(), &placed_desc, sizeof(placed_desc));
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = placed_desc.Format;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture2D.MipLevels = 1;
        std::array<std::uint32_t, 16> view_words{};
        std::memcpy(view_words.data(), &view, sizeof(view));
        leonida::gfx10::ImageWords streamed{};
        EXPECT(rt::d3d12::write_placed_texture_view(device.p, false, resource_words.data(), view_words.data(),
                                                    std::uintptr_t(pixels),
                                                    {reinterpret_cast<SIZE_T>(streamed.data())}));
        // Then the texture placed there (slot 133): it takes the address over.
        Ref<ID3D12Resource> placed;
        EXPECT(place(xbox.p, pixels, placed_desc, &placed.p) == S_OK && placed.p);
        EXPECT(placed.p->GetGPUVirtualAddress() == std::uintptr_t(pixels));
        // The T# the title writes: the words a view of the texture has, with the title's own dword 6.
        device.p->CreateShaderResourceView(placed.p, nullptr, cpu);
        std::uint32_t named[8];
        std::memcpy(named, reinterpret_cast<const void*>(cpu.ptr), sizeof(named));
        leonida::gfx10::ImageWords named_words{};
        std::memcpy(named_words.data(), named, sizeof(named_words));
        leonida::gfx10::Image named_image{};
        EXPECT(leonida::gfx10::decode(named_words, named_image) && named_image.address == std::uintptr_t(pixels));
        named[leonida::gfx10::kTsharpIndexDword] = 0x00BEEF00u;
        std::thread([&] { std::memcpy(title + 7 * kDescriptor, named, sizeof(named)); }).join();
        std::uint32_t got[8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 7 * kDescriptor, got, sizeof(got)));
        const std::uint32_t placed_word = got[leonida::gfx10::kTsharpIndexDword];
        std::printf("placed texture at 0x%llx: the T# reads index word 0x%08x\n",
                    static_cast<unsigned long long>(std::uintptr_t(pixels)), placed_word);
        EXPECT((placed_word >> 28) == leonida::gfx10::kTsharpIndexTag && (placed_word & 0xFFFFu) != 0);
        EXPECT(reinterpret_cast<const std::uint32_t*>(title + 7 * kDescriptor)[leonida::gfx10::kTsharpIndexDword] ==
               0x00BEEF00u);
        // The texture goes: the slot waits and reads the fallback. Another placed at the address: bound again.
        placed.reset();
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 7 * kDescriptor, got, sizeof(got)));
        EXPECT(got[leonida::gfx10::kTsharpIndexDword] == 0);
        Ref<ID3D12Resource> again;
        EXPECT(place(xbox.p, pixels, placed_desc, &again.p) == S_OK && again.p);
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 7 * kDescriptor, got, sizeof(got)));
        const std::uint32_t again_word = got[leonida::gfx10::kTsharpIndexDword];
        EXPECT((again_word >> 28) == leonida::gfx10::kTsharpIndexTag && (again_word & 0xFFFFu) != 0);
        again.reset();
        rt::d3d12::set_device_outer(device.p, nullptr);
        xbox.reset();
        EXPECT(map(pixels, 1, nullptr) == pixels && free_pages(1, &pixel_page));
    }

    // 7. Inline data in the heap's page, read through a V# of the title.
    {
        const std::uint32_t sizes[4] = {512, 256, 1280, 720};  // a "V#" of base 0x100_00000200, nothing there
        const std::uint64_t data_va = std::uintptr_t(title) + 10 * kDescriptor;
        const auto pointer = raw_vsharp(data_va, sizeof(sizes));
        std::thread([&] {
            std::memcpy(title + 10 * kDescriptor, sizes, sizeof(sizes));
            put(11, pointer);
        }).join();
        std::uint32_t table[2][8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 10 * kDescriptor, table, sizeof(table)));
        // A slot of the same page rewritten: the filter runs over the page again.
        const auto other = raw_vsharp(arena_va + 1024, 64);
        std::thread([&] { put(1, other); }).join();
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 10 * kDescriptor, table, sizeof(table)));
        std::uint32_t through_vsharp[4]{};
        EXPECT(rt::d3d12::page_mirror_gpu_read(device.p, data_va, through_vsharp, sizeof(through_vsharp)));
        std::printf("inline data: through the V# [%u %u %u %u], through the table [%u %u %u %u]\n", through_vsharp[0],
                    through_vsharp[1], through_vsharp[2], through_vsharp[3], table[0][0], table[0][1], table[0][2],
                    table[0][3]);
        EXPECT(!std::memcmp(through_vsharp, sizes, sizeof(sizes)));                   // the title's words, exactly
        EXPECT(!(table[0][0] | table[0][1] | table[0][2] | table[0][3]));             // null whole in the table
        EXPECT(!std::memcmp(table[1], pointer.data(), sizeof(pointer)));              // the V#: mirrored, as written
        std::uint32_t rewritten_entry[8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + kDescriptor, rewritten_entry, kDescriptor));
        EXPECT(!std::memcmp(rewritten_entry, other.data(), sizeof(other)));           // filtered: an arena V#
        EXPECT(!std::memcmp(title + 10 * kDescriptor, sizes, sizeof(sizes)));         // the title's bytes unchanged
        EXPECT(!std::memcmp(title + 11 * kDescriptor, pointer.data(), sizeof(pointer)));
    }

    // 8. Oversized & out-of-range V# regression: an in-place heap V# naming > 256 MiB (past kMirrorNameMax) of a committed plain
    //    region inside title_mirror_space must NEVER create an orphan TitleWindow, must null the filtered copy while
    //    keeping the title's bytes intact, and repeated V# halves must answer identically without creating windows.
    //    Also verify O(1)-rejected bit-47 and out-of-arena-span V# halves null cleanly.
    {
        constexpr SIZE_T kBigReserve = 300ull << 20;
        constexpr std::uint32_t kOverMirrorMax = 260u << 20;  // > kMirrorNameMax (256 MiB)
        auto* big_plain = static_cast<std::uint8_t*>(reserve(nullptr, kBigReserve, MEM_RESERVE, 0, PAGE_READWRITE));
        EXPECT(big_plain && leonida::title_mirror_space::contains(std::uintptr_t(big_plain)));
        EXPECT(VirtualAlloc(big_plain, kPage64k, MEM_COMMIT, PAGE_READWRITE) == big_plain);
        std::memset(big_plain, 0x5A, kPage64k);
        const std::size_t aliases_before = leonida::title_va_aliases().size();

        const auto wide_vsharp = raw_vsharp(std::uintptr_t(big_plain), kOverMirrorMax);
        const auto bit47_vsharp = raw_vsharp((1ull << 47) | 0x200000ull, 256);
        const std::uint64_t arena_sz = rt::d3d12::device_arena_size(device.p);
        // base < arena_sz with span > arena_sz - base (fails reachable_as_written; O(1)-rejected in filter).
        leonida::gfx10::Buffer over_arena{};
        over_arena.address = arena_sz > 0x10000 ? arena_sz - 0x10000 : 0x10000;
        over_arena.stride = 16;
        over_arena.num_records = 0x4000;  // 256 KiB > 64 KiB remaining in arena
        over_arena.format = leonida::gfx10::format_from_vulkan(VK_FORMAT_R32_UINT);
        const auto over_arena_vsharp = leonida::gfx10::encode(over_arena);

        std::thread([&] {
            put(12, wide_vsharp);
            put(13, wide_vsharp);  // repeated V# half in same batch
            put(14, bit47_vsharp);
            put(15, over_arena_vsharp);
        }).join();
        std::uint32_t out[4][8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 12 * kDescriptor, out, sizeof(out)));
        for (int i = 0; i < 4; ++i) EXPECT(!(out[i][0] | out[i][1] | out[i][2] | out[i][3]));
        // Write another repeated wide V# on a subsequent submission (hits heap.records.answer()).
        std::thread([&] { put(16, wide_vsharp); }).join();
        std::uint32_t repeat_out[8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 16 * kDescriptor, repeat_out, sizeof(repeat_out)));
        EXPECT(!(repeat_out[0] | repeat_out[1] | repeat_out[2] | repeat_out[3]));

        // Zero orphan TitleWindows created in title_va_aliases.
        EXPECT(leonida::title_va_aliases().size() == aliases_before);
        EXPECT(leonida::title_va_aliases().shader_va(std::uintptr_t(big_plain)) == std::uintptr_t(big_plain));
        // Title's own bytes remain untouched.
        EXPECT(!std::memcmp(title + 12 * kDescriptor, wide_vsharp.data(), sizeof(wide_vsharp)));
        EXPECT(!std::memcmp(title + 14 * kDescriptor, bit47_vsharp.data(), sizeof(bit47_vsharp)));
        EXPECT(VirtualFree(big_plain, 0, MEM_RELEASE));
    }

    // 9. Deferred-slot and pending-alias regressions:
    //    (a) A slot queued in heap.deferred by apply_in_place_changes (texture_gone) and overwritten in the same
    //        refresh's upload() with non-T# words (w[3] >> 30 != 0, !is_image) must zero w[0..3] in finish_in_place
    //        instead of writing raw unfiltered words into home[slot];
    //    (b) A V# naming base >= arena.size not yet in title_memory or aliases is nulled and recorded as waiting, so
    //        adding an alias for that base later wakes and rebases the slot automatically.
    {
        // (a) Trigger apply_in_place_changes (texture_gone -> queues slot 17 in heap.deferred) + concurrent page write
        //     of raw non-T# words into slot 17 before refresh, forcing finish_in_place() to see not_tsharp.
        Ref<ID3D12Resource> tmp_tex;
        EXPECT(SUCCEEDED(device.p->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex_desc,
                                                           D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr,
                                                           IID_PPV_ARGS(&tmp_tex.p))));
        device.p->CreateShaderResourceView(tmp_tex.p, nullptr, cpu);
        std::uint32_t tmp_tsharp[8];
        std::memcpy(tmp_tsharp, reinterpret_cast<const void*>(cpu.ptr), sizeof(tmp_tsharp));
        tmp_tsharp[leonida::gfx10::kTsharpIndexDword] = 0x00C0FFEEu;
        std::thread([&] { std::memcpy(title + 17 * kDescriptor, tmp_tsharp, sizeof(tmp_tsharp)); }).join();
        std::uint32_t bound_tsharp[8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 17 * kDescriptor, bound_tsharp, sizeof(bound_tsharp)));
        EXPECT((bound_tsharp[leonida::gfx10::kTsharpIndexDword] >> 28) == leonida::gfx10::kTsharpIndexTag);

        const std::uint32_t bad_non_tsharp[8] = {0xdead0000u, 0x0000ffffu, 0x00000400u, 0x40000000u, 1, 2, 3, 4};
        tmp_tex.reset();  // notes texture_gone -> next refresh's apply_in_place_changes defers slot 17
        std::thread([&] { std::memcpy(title + 17 * kDescriptor, bad_non_tsharp, sizeof(bad_non_tsharp)); }).join();
        std::uint32_t cleared[8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 17 * kDescriptor, cleared, sizeof(cleared)));
        EXPECT(!(cleared[0] | cleared[1] | cleared[2] | cleared[3]));
        EXPECT(!std::memcmp(title + 17 * kDescriptor, bad_non_tsharp, sizeof(bad_non_tsharp)));

        // (b) V# at unbacked base >= arena.size: nulled initially, then woken and rebased when an alias arrives.
        constexpr std::uint64_t kLateVa = 0x510000000000ull;
        const auto late_vsharp = raw_vsharp(kLateVa, 256);
        std::thread([&] { put(18, late_vsharp); }).join();
        std::uint32_t late_out[8]{};
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 18 * kDescriptor, late_out, sizeof(late_out)));
        EXPECT(!(late_out[0] | late_out[1] | late_out[2] | late_out[3]));

        EXPECT(leonida::title_va_aliases().add(kLateVa, 4096, arena_va));
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 18 * kDescriptor, late_out, sizeof(late_out)));
        EXPECT(base_of(late_out) == arena_va && late_out[2] == 256);

        leonida::title_va_aliases().remove(kLateVa, arena_va);
        EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + 18 * kDescriptor, late_out, sizeof(late_out)));
        EXPECT(!(late_out[0] | late_out[1] | late_out[2] | late_out[3]));
    }

    // 10. With a live console resource heap (console_heap_count > 0), unpromoted slot-133 BC placements sharing
    //     zero_sampled_image deduplicate onto a single bindless slot, and promoting one gives it a private slot
    //     while leaving the other intact on the shared zero slot.
    {
        constexpr UINT width = 32, height = 32, pitch = width / 4 * 8;
        void* bc_pages = VirtualAlloc(nullptr, 2 * kPage64k, MEM_RESERVE, PAGE_NOACCESS);
        EXPECT(bc_pages);
        D3D12_RESOURCE_DESC bc_desc{};
        bc_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        bc_desc.Width = width;
        bc_desc.Height = height;
        bc_desc.DepthOrArraySize = bc_desc.MipLevels = bc_desc.SampleDesc.Count = 1;
        bc_desc.Format = DXGI_FORMAT_BC1_UNORM;
        Ref<ID3D12Resource> bc_a, bc_b;
        EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, bc_pages, bc_desc, 0,
                                                                  IID_PPV_ARGS(&bc_a.p))));
        EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(
            device.p, static_cast<char*>(bc_pages) + kPage64k, bc_desc, 0, IID_PPV_ARGS(&bc_b.p))));
        std::uint32_t av[8]{}, bv[8]{};
        device.p->CreateShaderResourceView(bc_a.p, nullptr, {reinterpret_cast<SIZE_T>(av)});
        device.p->CreateShaderResourceView(bc_b.p, nullptr, {reinterpret_cast<SIZE_T>(bv)});
        const std::uint32_t first = rt::d3d12::bindless_hash_lookup(device.p, av);
        const std::uint32_t second = rt::d3d12::bindless_hash_lookup(device.p, bv);
        const bool shared = !rt::d3d12::vulkan_memory_flags(bc_a.p);
        EXPECT(first && second);
        if (shared) EXPECT(first == second);
        std::array<std::uint8_t, pitch * height / 4> pattern{};
        for (std::size_t i = 0; i < pattern.size(); ++i) pattern[i] = std::uint8_t(i * 11 + 37);
        EXPECT(SUCCEEDED(bc_a.p->WriteToSubresource(0, nullptr, pattern.data(), pitch, UINT(pattern.size()))));
        EXPECT(rt::d3d12::vulkan_memory_flags(bc_a.p));
        const std::uint32_t private_a = rt::d3d12::bindless_hash_lookup(device.p, av);
        EXPECT(private_a && (!shared || private_a != first));
        EXPECT(rt::d3d12::bindless_hash_lookup(device.p, bv) == second);
        bc_a.reset();
        bc_b.reset();
        EXPECT(VirtualFree(bc_pages, 0, MEM_RELEASE));
    }

    // 3. The null slots after the heap.
    std::uint32_t tail[64]{};
    std::memset(tail, 0xff, sizeof(tail));
    EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + kPages * kPage64k, tail, sizeof(tail)));
    for (const std::uint32_t word : tail) EXPECT(word == 0);

    std::printf("heap in place: at heap_base + 0x%llx; the GPU reads a nobody's V# and a data word null and the others "
                "as written, a rewrite at the next refresh, zeros after the heap, a T# with its index word (0x%08x); the title's bytes never change\n",
                static_cast<unsigned long long>(shader_va), word);
    heap.reset();
    EXPECT(map(title, kPages, nullptr) == title && free_pages(kPages, pages));
    return 0;
}

}  // namespace

int main() { return test(); }
