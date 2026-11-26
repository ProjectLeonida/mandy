#include "../shims/d3d12_title_image_census.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>

namespace {

struct FakeSampleDesc {
    unsigned Count = 1;
};

struct FakeDesc {
    std::uint64_t Width = 1920;
    unsigned Height = 1080;
    unsigned Format = 28;
    unsigned Flags = 0;
    FakeSampleDesc SampleDesc{};
};

struct FakeResource {
    FakeDesc d{};
    bool replayed = false;
    bool views = false;
    const FakeDesc& desc() const { return d; }
    bool replay_seen() const { return replayed; }
    bool shader_views() const { return views; }
};

struct FakeEntry {
    bool owned = false;
    std::uint64_t unbound_bytes = 0;
    std::uint64_t backing_bytes = 0;
    std::uint64_t cpu = 0;
    bool mapped = false;
    unsigned fills = 0;
    bool attachment_4k = false;
    bool gpu_written = false;
    std::unique_ptr<FakeResource> resource;
};

}  // namespace

int main() {
    std::map<std::uint64_t, std::unique_ptr<FakeEntry>> textures;

    // Valid placed entry.
    auto placed = std::make_unique<FakeEntry>();
    placed->cpu = 0x1000;
    placed->backing_bytes = 2ull << 20;
    placed->resource = std::make_unique<FakeResource>();
    textures.emplace(1, std::move(placed));

    // Regression: null entry pointer and placed entry with null resource must not crash.
    textures.emplace(2, nullptr);
    auto no_res = std::make_unique<FakeEntry>();
    no_res->cpu = 0x2000;
    no_res->backing_bytes = 1ull << 20;
    textures.emplace(3, std::move(no_res));

    // Owned entry aliasing placed CPU address.
    auto owned = std::make_unique<FakeEntry>();
    owned->owned = true;
    owned->cpu = 0x1000;
    owned->backing_bytes = 2ull << 20;
    textures.emplace(4, std::move(owned));

    rt::d3d12::log_title_image_uses(textures, 1, 1ull << 20, 0);
    std::puts("d3d12_title_image_census_test: ok");
    return 0;
}
