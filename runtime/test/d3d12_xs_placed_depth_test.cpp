// Unit tests for PlacedDepthTextures and PlacedDepthTexture (runtime/shims/d3d12_xs_placed_depth.h)
// Verifies null out pointer guards, argument validation, backing failure propagation,
// COM identity wrapper isolation, aliasing matching and refusal rules, weak pointer cleanup,
// and container clearing.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>

#include "../shims/d3d12_xs_placed_depth.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s\n", message);
    } else {
        std::printf("PASS: %s\n", message);
    }
}

D3D12_RESOURCE_DESC make_desc(DXGI_FORMAT format = DXGI_FORMAT_R32G8X24_TYPELESS, UINT64 width = 256, UINT height = 128) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    return desc;
}

// Minimal synthetic ID3D12Resource mock for testing PlacedDepthTextures
class FakeResource final : public ID3D12Resource {
    std::atomic<ULONG> ref_count_{1};
    UINT64 va_{0x10000};

public:
    explicit FakeResource(UINT64 va = 0x10000) : va_(va) {}

    ULONG STDMETHODCALLTYPE AddRef() override {
        return ref_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG const prev = ref_count_.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) {
            delete this;
            return 0;
        }
        return prev - 1;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override {
        if (!ppvObject) return E_POINTER;
        *ppvObject = nullptr;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) ||
            riid == __uuidof(ID3D12DeviceChild) || riid == __uuidof(ID3D12Pageable) ||
            riid == __uuidof(ID3D12Resource)) {
            *ppvObject = static_cast<ID3D12Resource*>(this);
            AddRef();
            return S_OK;
        }
        if (riid == rt::d3d12_xs::kVulkanResourceTag) {
            *ppvObject = this;
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void** ppvDevice) override {
        if (ppvDevice) *ppvDevice = nullptr;
        return S_OK;
    }
#if defined(__MINGW32__)
    D3D12_RESOURCE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_RESOURCE_DESC* out) override {
        if (out) *out = make_desc();
        return out;
    }
#else
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override {
        return make_desc();
    }
#endif
    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override { return va_; }
    HRESULT STDMETHODCALLTYPE GetHeapProperties(D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE Map(UINT, const D3D12_RANGE*, void** ppData) override {
        if (ppData) *ppData = nullptr;
        return S_OK;
    }
    void STDMETHODCALLTYPE Unmap(UINT, const D3D12_RANGE*) override {}
    HRESULT STDMETHODCALLTYPE WriteToSubresource(UINT, const D3D12_BOX*, const void*, UINT, UINT) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE ReadFromSubresource(void*, UINT, UINT, UINT, const D3D12_BOX*) override { return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return S_OK; }
};

} // namespace

int main() {
    rt::d3d12_xs::PlacedDepthTextures pool;
    const auto make_backing = [](ID3D12Resource** out) -> HRESULT {
        *out = new FakeResource();
        return S_OK;
    };
    const auto fail_backing = [](ID3D12Resource** out) -> HRESULT {
        *out = nullptr;
        return E_OUTOFMEMORY;
    };

    D3D12_RESOURCE_DESC desc = make_desc();
    D3D12_RESOURCE_ALLOCATION_INFO alloc{0x10000, 0x1000};
    const UINT64 address = 0x20000;

    // 1. Null out pointer guard returns E_POINTER
    {
        HRESULT hr = pool.create(address, desc, alloc, nullptr, true, __uuidof(ID3D12Resource), nullptr, make_backing);
        check(hr == E_POINTER, "create returns E_POINTER when out is nullptr");
        check(pool.empty(), "pool remains empty on null out");
    }

    // 2. Argument validation and setting *out = nullptr on failure
    {
        void* out = reinterpret_cast<void*>(0xDEADBEEF);
        // address == 0
        HRESULT hr = pool.create(0, desc, alloc, nullptr, true, __uuidof(ID3D12Resource), &out, make_backing);
        check(hr == E_INVALIDARG, "create returns E_INVALIDARG when address is 0");
        check(out == nullptr, "out is cleared to nullptr when address is 0");

        // alignment == 0
        out = reinterpret_cast<void*>(0xDEADBEEF);
        D3D12_RESOURCE_ALLOCATION_INFO zero_align{0x10000, 0};
        hr = pool.create(address, desc, zero_align, nullptr, true, __uuidof(ID3D12Resource), &out, make_backing);
        check(hr == E_INVALIDARG, "create returns E_INVALIDARG when alignment is 0");
        check(out == nullptr, "out is cleared to nullptr when alignment is 0");

        // unaligned address
        out = reinterpret_cast<void*>(0xDEADBEEF);
        hr = pool.create(address + 1, desc, alloc, nullptr, true, __uuidof(ID3D12Resource), &out, make_backing);
        check(hr == E_INVALIDARG, "create returns E_INVALIDARG when address is unaligned");
        check(out == nullptr, "out is cleared to nullptr when address is unaligned");

        // size == 0
        out = reinterpret_cast<void*>(0xDEADBEEF);
        D3D12_RESOURCE_ALLOCATION_INFO zero_size{0, 0x1000};
        hr = pool.create(address, desc, zero_size, nullptr, true, __uuidof(ID3D12Resource), &out, make_backing);
        check(hr == E_INVALIDARG, "create returns E_INVALIDARG when size is 0");
        check(out == nullptr, "out is cleared to nullptr when size is 0");

        // address + size overflow
        out = reinterpret_cast<void*>(0xDEADBEEF);
        D3D12_RESOURCE_ALLOCATION_INFO overflow_size{0x2000, 0x1000};
        hr = pool.create(UINT64_MAX - 0x1000, desc, overflow_size, nullptr, true, __uuidof(ID3D12Resource), &out, make_backing);
        check(hr == E_INVALIDARG, "create returns E_INVALIDARG on address + size overflow");
        check(out == nullptr, "out is cleared to nullptr on address + size overflow");
    }

    // 3. create_backing failure propagation
    {
        void* out = reinterpret_cast<void*>(0xDEADBEEF);
        HRESULT hr = pool.create(address, desc, alloc, nullptr, true, __uuidof(ID3D12Resource), &out, fail_backing);
        check(hr == E_OUTOFMEMORY, "create propagates create_backing HRESULT failure");
        check(out == nullptr, "out is cleared to nullptr on create_backing failure");
        check(pool.empty(), "pool remains empty on backing creation failure");
    }

    // 4. Successful creation and wrapping
    ID3D12Resource* res1 = nullptr;
    {
        HRESULT hr = pool.create(address, desc, alloc, nullptr, true, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&res1), make_backing);
        check(hr == S_OK, "create returns S_OK on valid arguments");
        check(res1 != nullptr, "res1 is non-null on success");
        check(pool.size() == 1, "pool has size 1 after initial create");
        check(!pool.empty(), "pool is not empty after create");

        if (res1) {
            check(res1->GetGPUVirtualAddress() == address, "PlacedDepthTexture returns configured GPU address");
            const ID3D12Resource* backing = rt::d3d12_xs::texture_backing(res1);
            check(backing != nullptr, "texture_backing resolves backing resource");
            check(backing != res1, "texture_backing returns inner resource rather than wrapper");

            // QueryInterface forwarding
            void* tag_out = nullptr;
            HRESULT qi_tag = res1->QueryInterface(rt::d3d12_xs::kVulkanResourceTag, &tag_out);
            check(qi_tag == S_OK && tag_out != nullptr, "QueryInterface with kVulkanResourceTag succeeds");
            if (tag_out) reinterpret_cast<IUnknown*>(tag_out)->Release();

            void* d3d_out = nullptr;
            HRESULT qi_res = res1->QueryInterface(__uuidof(ID3D12Resource), &d3d_out);
            check(qi_res == S_OK && d3d_out == res1, "QueryInterface with ID3D12Resource returns wrapper");
            if (d3d_out) reinterpret_cast<IUnknown*>(d3d_out)->Release();
        }
    }

    // 5. Aliasing and backing reuse for identical shape and clear
    ID3D12Resource* res2 = nullptr;
    {
        HRESULT hr = pool.create(address, desc, alloc, nullptr, true, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&res2), make_backing);
        check(hr == S_OK, "create succeeds for identical placement alias");
        check(res2 != nullptr, "res2 is non-null");
        check(res1 != res2, "distinct COM wrappers created for alias");
        check(pool.size() == 1, "pool still has size 1 (reused backing)");

        if (res1 && res2) {
            const ID3D12Resource* b1 = rt::d3d12_xs::texture_backing(res1);
            const ID3D12Resource* b2 = rt::d3d12_xs::texture_backing(res2);
            check(b1 == b2, "both aliases share identical backing resource");
            check(res2->GetGPUVirtualAddress() == address, "alias has identical GPU address");
        }
    }

    // 6. Incompatible shape at same address with non-opaque layout returns E_NOTIMPL
    {
        void* out = reinterpret_cast<void*>(0xDEADBEEF);
        D3D12_RESOURCE_DESC diff_desc = make_desc(DXGI_FORMAT_R32G8X24_TYPELESS, 512, 128);
        HRESULT hr = pool.create(address, diff_desc, alloc, nullptr, false, __uuidof(ID3D12Resource), &out, make_backing);
        check(hr == E_NOTIMPL, "incompatible shape with non-opaque layout returns E_NOTIMPL");
        check(out == nullptr, "out is cleared to nullptr on E_NOTIMPL");
    }

    // 7. Non-opaque layout overlapping an existing range returns E_NOTIMPL
    {
        void* out = reinterpret_cast<void*>(0xDEADBEEF);
        HRESULT hr = pool.create(address + 0x1000, desc, alloc, nullptr, false, __uuidof(ID3D12Resource), &out, make_backing);
        check(hr == E_NOTIMPL, "non-opaque layout overlapping existing range returns E_NOTIMPL");
        check(out == nullptr, "out is cleared to nullptr on overlap refusal");
    }

    // 8. Differently shaped opaque depth targets alternate at same offset and get separate backings
    ID3D12Resource* res_diff = nullptr;
    {
        D3D12_RESOURCE_DESC diff_desc = make_desc(DXGI_FORMAT_R32G8X24_TYPELESS, 512, 128);
        HRESULT hr = pool.create(address, diff_desc, alloc, nullptr, true, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&res_diff), make_backing);
        check(hr == S_OK, "differently shaped opaque layout at same address gets separate backing");
        check(res_diff != nullptr, "res_diff is non-null");
        check(pool.size() == 2, "pool tracks 2 distinct shapes at same address");
        check(rt::d3d12_xs::texture_backing(res_diff) != rt::d3d12_xs::texture_backing(res1), "separate backings allocated");
    }

    // 9. Weak backing expiration and cleanup
    if (res1) res1->Release();
    if (res2) res2->Release();
    if (res_diff) res_diff->Release();
    res1 = nullptr;
    res2 = nullptr;
    res_diff = nullptr;

    {
        // Now all aliases at 0x20000 are released, backings are expired.
        // Creating at a new address should trigger erase_if on expired entries.
        const UINT64 address2 = 0x40000;
        ID3D12Resource* res3 = nullptr;
        HRESULT hr = pool.create(address2, desc, alloc, nullptr, true, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&res3), make_backing);
        check(hr == S_OK, "create at address2 succeeds");
        check(res3 != nullptr, "res3 is non-null");
        check(pool.size() == 1, "expired entries at address were erased, leaving pool size 1");
        if (res3) res3->Release();
    }

    // 9. clear() and empty()
    {
        check(!pool.empty(), "pool not empty before clear()");
        pool.clear();
        check(pool.empty(), "pool is empty after clear()");
        check(pool.size() == 0, "pool size is 0 after clear()");
    }

    // 10. OptimizedClear comparison (D16 vs D32 and stencil plane)
    {
        D3D12_CLEAR_VALUE c1{};
        c1.Format = DXGI_FORMAT_D32_FLOAT;
        c1.DepthStencil.Depth = 1.0f;
        c1.DepthStencil.Stencil = 0;

        D3D12_CLEAR_VALUE c2 = c1;
        c2.DepthStencil.Depth = 0.0f;

        ID3D12Resource* r_c1 = nullptr;
        HRESULT hr1 = pool.create(address, desc, alloc, &c1, true, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&r_c1), make_backing);
        check(hr1 == S_OK, "create with clear value c1 succeeds");

        void* out_c2 = nullptr;
        HRESULT hr2 = pool.create(address, desc, alloc, &c2, true, __uuidof(ID3D12Resource), &out_c2, make_backing);
        check(hr2 == E_NOTIMPL, "different clear value depth returns E_NOTIMPL");

        if (r_c1) r_c1->Release();
        pool.clear();
    }

    std::printf("\nSummary: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
