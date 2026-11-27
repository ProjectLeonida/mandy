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
//   4. no copy: the device names it "read in place".
// Exit 77 when the device makes no page mirror (llvmpipe). No shaders, no game data.
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/shims.h"
#include "../shims/xmem.h"
#include "../src/gfx10_descriptors.h"
#include "../src/title_mirror_space.h"

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
    put(0, raw_vsharp(arena_va, 256));
    put(1, raw_vsharp(nobody, 256));
    put(2, raw_vsharp(mirrored, 256));
    const std::uint32_t data[8] = {0x12345000u, 0x00007777u, 0x00000400u, 0x40000000u, 1, 2, 3, 4};
    std::memcpy(title + 3 * kDescriptor, data, sizeof(data));
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

    // No write-back from the heap: a store the GPU makes into the heap's pages (through the mirror at the title's
    // address) is never copied over the title's bytes, which are not the home's.
    const std::uint32_t stored[8] = {0xdead0001u, 0xdead0002u, 0xdead0003u, 0xdead0004u, 5, 6, 7, 8};
    EXPECT(rt::d3d12::page_mirror_gpu_write(device.p, std::uintptr_t(title) + 5 * kDescriptor, stored, sizeof(stored)));
    EXPECT(rt::d3d12::page_mirror_write_back(device.p, std::uintptr_t(title), kPages * kPage64k) == 0);
    for (unsigned i = 0; i < kDescriptor; ++i) EXPECT(title[5 * kDescriptor + i] == 0);

    // 3. The null slots after the heap.
    std::uint32_t tail[64]{};
    std::memset(tail, 0xff, sizeof(tail));
    EXPECT(rt::d3d12::arena_gpu_read(device.p, shader_va + kPages * kPage64k, tail, sizeof(tail)));
    for (const std::uint32_t word : tail) EXPECT(word == 0);

    std::printf("heap in place: at heap_base + 0x%llx; the GPU reads a nobody's V# and a data word null and the others "
                "as written, a rewrite at the next refresh, zeros after the heap; the title's bytes never change\n",
                static_cast<unsigned long long>(shader_va));
    heap.reset();
    EXPECT(map(title, kPages, nullptr) == title && free_pages(kPages, pages));
    return 0;
}

}  // namespace

int main() { return test(); }
