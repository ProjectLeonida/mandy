#pragma once

#include "xg_xs_layout.h"
#include "../src/gfx10_descriptors.h"
#include <d3d12.h>
#include <unordered_map>
#include <vector>

namespace rt::d3d12 {

// GPU resources can have overlapping lifetimes at one placement VA. Resource references are
// held by the D3D12 caller. The device serializes this index with descriptor registration.
template<class Resource>
class TitleGpuPlacements {
public:
    void add(std::uint64_t address, Resource* resource) {
        if (!resource) return;
        Bucket& bucket = resources_[address];
        if (!bucket.first)
            bucket.first = resource;
        else
            bucket.extra.push_back(resource);
    }

    void remove(std::uint64_t address, const void* owner) {
        const auto it = resources_.find(address);
        if (it == resources_.end()) return;
        Bucket& bucket = it->second;
        if (bucket.first == owner) {
            if (bucket.extra.empty()) {
                resources_.erase(it);
            } else {
                bucket.first = bucket.extra.front();
                bucket.extra.erase(bucket.extra.begin());
            }
            return;
        }
        for (auto eit = bucket.extra.begin(); eit != bucket.extra.end(); ++eit) {
            if (*eit == owner) {
                bucket.extra.erase(eit);
                return;
            }
        }
    }

    template<class Matches>
    Resource* find(std::uint64_t va, std::uint64_t alias, Matches matches) const {
        const auto search = [&](std::uint64_t address) -> Resource* {
            if (!address) return nullptr;
            const auto it = resources_.find(address);
            if (it == resources_.end()) return nullptr;
            const Bucket& bucket = it->second;
            for (std::size_t i = bucket.extra.size(); i-- > 0;)
                if (matches(*bucket.extra[i])) return bucket.extra[i];
            if (bucket.first && matches(*bucket.first)) return bucket.first;
            return nullptr;
        };
        if (Resource* hit = search(va)) return hit;
        if (alias != va) return search(alias);
        return nullptr;
    }

    static bool matches(const rt::xg_xs::ResourceDesc& d, const Resource& resource) {
        const auto& r = resource.desc();
        return UINT(r.Dimension) == d.dimension && r.Width == d.width && r.Height == d.height &&
               r.DepthOrArraySize == d.depth_or_array && (!d.mips || r.MipLevels == d.mips) &&
               r.SampleDesc.Count == d.samples &&
               (UINT(r.Format) == d.format || resource.lists_castable_format(DXGI_FORMAT(d.format)));
    }

    static bool matches(const leonida::gfx10::Image& image, const D3D12_RESOURCE_DESC& d) {
        namespace g = leonida::gfx10;
        if (d.Width != image.width || d.Height != image.height) return false;
        if (g::multisampled(image.type) != (d.SampleDesc.Count > 1)) return false;
        if (d.SampleDesc.Count > 1 && d.SampleDesc.Count != image.samples) return false;
        if (image.first_mip > d.MipLevels || image.mip_count > d.MipLevels - image.first_mip) return false;
        if (image.type == g::ImageType::image_3d)
            return d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D && image.depth == d.DepthOrArraySize;
        const auto dimension = image.type == g::ImageType::image_1d
                                   ? D3D12_RESOURCE_DIMENSION_TEXTURE1D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        return d.Dimension == dimension && image.first_layer <= d.DepthOrArraySize &&
               image.layer_count <= d.DepthOrArraySize - image.first_layer;
    }

private:
    struct Bucket {
        Resource* first = nullptr;
        std::vector<Resource*> extra;
    };
    std::unordered_map<std::uint64_t, Bucket> resources_;
};

}  // namespace rt::d3d12
