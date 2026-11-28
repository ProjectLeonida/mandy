// The keys ID3D12Device2::CreatePipelineState gives a stream's pipeline (shims/d3d12_vulkan_device.cpp): the cache key
// (GetCachedBlob) and the pipeline_order.log key. A stream carries state the plain graphics description does not:
// DEPTH_STENCIL1/2's depth-bounds flag and DEPTH_STENCIL2's back-face stencil masks. Two streams that differ only there
// build different pipelines, so they must not share a key (RT-STATE-PIPELINE-KEY-DS2, item 5 of kevincommits'
// RT-STATE-AUDIT-PIPELINE finding). A stream without that state keeps the key it had before. Parses only: no device,
// no GPU, no game data.
#include "../shims/d3d12_com.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
#define EXPECT(cond)                                                              \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "%s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                           \
        }                                                                         \
    } while (0)

// A pipeline state stream, laid out as the D3D12 helpers do: each subobject's type, its payload at the payload's
// alignment, the next subobject at pointer alignment.
class Stream {
public:
    template <class T>
    Stream& add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const T& value) {
        place(&type, sizeof(type), alignof(void*));
        place(&value, sizeof(T), alignof(T));
        bytes_.resize(round_up(bytes_.size(), alignof(void*)));
        return *this;
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc() { return {bytes_.size(), bytes_.data()}; }

private:
    static std::size_t round_up(std::size_t v, std::size_t a) { return (v + a - 1) / a * a; }
    void place(const void* p, std::size_t n, std::size_t align) {
        const std::size_t at = round_up(bytes_.size(), align);
        bytes_.resize(at + n);
        std::memcpy(bytes_.data() + at, p, n);
    }
    std::vector<std::uint8_t> bytes_;  // new[]'s alignment covers every payload's
};

const std::uint8_t kVs[16] = {'n', 'o', 't', ' ', 'a', ' ', 'r', 'e', 'a', 'l', ' ', 'v', 's', 0, 1, 2};

struct Keys {
    std::string cache, order;
    bool operator==(const Keys& o) const { return cache == o.cache && order == o.order; }
};

// The cache and order keys of a stream holding a VS, one render target, a depth-stencil format and the given
// depth-stencil subobject.
template <class T>
Keys keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const T& depth_stencil) {
    Stream s;
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE{kVs, sizeof(kVs)});
    D3D12_RT_FORMAT_ARRAY formats{};
    formats.NumRenderTargets = 1;
    formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, formats);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
    s.add(type, depth_stencil);
    Keys k;
    const HRESULT hr = rt::d3d12::pipeline_stream_keys_for_test(s.desc(), k.cache, k.order);
    EXPECT(SUCCEEDED(hr));
    EXPECT(!k.cache.empty() && !k.order.empty());
    return k;
}

D3D12_DEPTH_STENCILOP_DESC face_op() {
    return {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_INCR, D3D12_STENCIL_OP_REPLACE, D3D12_COMPARISON_FUNC_EQUAL};
}
D3D12_DEPTH_STENCILOP_DESC1 face_op1(UINT8 read, UINT8 write) {
    const D3D12_DEPTH_STENCILOP_DESC f = face_op();
    return {f.StencilFailOp, f.StencilDepthFailOp, f.StencilPassOp, f.StencilFunc, read, write};
}
D3D12_DEPTH_STENCIL_DESC plain() {
    return {TRUE, D3D12_DEPTH_WRITE_MASK_ALL, D3D12_COMPARISON_FUNC_LESS, TRUE, 0x0f, 0xf0, face_op(), face_op()};
}
D3D12_DEPTH_STENCIL_DESC1 with_bounds(BOOL bounds) {
    const D3D12_DEPTH_STENCIL_DESC d = plain();
    return {d.DepthEnable, d.DepthWriteMask, d.DepthFunc, d.StencilEnable, d.StencilReadMask, d.StencilWriteMask,
            d.FrontFace, d.BackFace, bounds};
}
D3D12_DEPTH_STENCIL_DESC2 with_back(UINT8 back_read, UINT8 back_write, BOOL bounds = FALSE) {
    const D3D12_DEPTH_STENCIL_DESC d = plain();
    return {d.DepthEnable, d.DepthWriteMask, d.DepthFunc, d.StencilEnable, face_op1(d.StencilReadMask, d.StencilWriteMask),
            face_op1(back_read, back_write), bounds};
}

}  // namespace

int main() {
    const Keys base = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, plain());

    // Depth bounds: DEPTH_STENCIL1 with the test off is the plain description, key included; on, another pipeline.
    const Keys bounds_off = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, with_bounds(FALSE));
    const Keys bounds_on = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, with_bounds(TRUE));
    EXPECT(bounds_off == base);
    EXPECT(bounds_on.cache != bounds_off.cache);
    EXPECT(bounds_on.order != bounds_off.order);
    std::printf("depth bounds: on and off get different cache and order keys; off keeps the plain key\n");

    // Back-face stencil masks: DEPTH_STENCIL2 with the back face's masks equal to the front's builds the plain
    // pipeline (without DEPTH_STENCIL2 the back face takes the front's), so it keeps the plain key; other masks, and
    // two different sets of them, get keys of their own.
    const Keys back_same = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, with_back(0x0f, 0xf0));
    const Keys back_read = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, with_back(0x3c, 0xf0));
    const Keys back_write = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, with_back(0x0f, 0x01));
    EXPECT(back_same == base);
    EXPECT(back_read.cache != base.cache && back_read.order != base.order);
    EXPECT(back_write.cache != base.cache && back_write.order != base.order);
    EXPECT(back_read.cache != back_write.cache && back_read.order != back_write.order);
    std::printf("back-face stencil masks: masks unlike the front's get keys of their own; equal ones keep the plain key\n");

    // Both at once differ from either alone.
    const Keys both = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, with_back(0x3c, 0xf0, TRUE));
    const Keys bounds_only2 = keys(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2, with_back(0x0f, 0xf0, TRUE));
    EXPECT(bounds_only2 == bounds_on);
    EXPECT(both.cache != bounds_on.cache && both.cache != back_read.cache);
    EXPECT(both.order != bounds_on.order && both.order != back_read.order);
    std::printf("depth bounds with back-face masks: a key unlike either alone\n");

    if (failures) {
        std::fprintf(stderr, "d3d12_pipeline_stream_key_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("d3d12_pipeline_stream_key_test: all passed\n");
    return 0;
}
