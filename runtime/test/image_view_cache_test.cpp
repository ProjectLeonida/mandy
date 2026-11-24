// ImageViewCache (runtime/src/image_view_cache.h): VulkanContext::view's attachment-view cache. CPU only.
//
// Checks fixed cases, then random insert / lookup / destroy sequences against the std::map keyed by a 6-tuple that
// the cache replaces (with the linear destroy scan it used), and prints how long a lookup and an image destroy take
// with each, and how many heap allocations they make. `--bench` times larger tables.

#include "../src/image_view_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <new>
#include <random>
#include <tuple>
#include <vector>

namespace {

std::uint64_t allocations = 0;
std::uint64_t allocated_bytes = 0;  // as requested, without the allocator's own headers

} // namespace

void* operator new(std::size_t n) {
    ++allocations;
    allocated_bytes += n;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

// Opaque handles, as VkImage and VkImageView are on 64-bit hosts.
struct ImageT;
struct ViewT;
using Image = ImageT*;
using View = ViewT*;
using Cache = leonida::ImageViewCache<Image, View>;

Image image(std::uintptr_t i) { return reinterpret_cast<Image>(i * 0x40 + 0x10000); }
View view(std::uintptr_t i) { return reinterpret_cast<View>(i * 0x40 + 0x8000000); }

// The cache before image_view_cache.h, as vulkan_backend.h had it.
struct Reference {
    using Key = std::tuple<Image, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;
    std::map<Key, View> views;

    static Key key(const Cache::Key& k) { return {k.image, k.aspect, k.mip, k.layer, k.layers, k.format}; }
    View find(const Cache::Key& k) const {
        const auto it = views.find(key(k));
        return it == views.end() ? nullptr : it->second;
    }
    bool insert(const Cache::Key& k, View v) { return views.emplace(key(k), v).second; }
    template <class Release>
    void erase_image(Image img, Release&& release) {
        for (auto it = views.begin(); it != views.end();) {
            if (std::get<0>(it->first) == img) {
                release(it->second);
                it = views.erase(it);
            } else {
                ++it;
            }
        }
    }
};

int failures = 0;

void check(bool ok, const char* what) {
    if (ok) return;
    std::printf("FAIL: %s\n", what);
    ++failures;
}

void fixed_cases() {
    Cache cache;
    const Cache::Key a{image(1), 1, 0, 0, 1, 37};
    check(cache.find(a) == nullptr, "an empty cache holds nothing");
    check(cache.insert(a, view(1)), "a new key is inserted");
    check(cache.find(a) == view(1), "the inserted view is found");
    check(!cache.insert(a, view(2)), "a cached key is not replaced");
    check(cache.find(a) == view(1), "a cached key keeps its first view");

    // Every field is part of the key.
    const Cache::Key variants[] = {
        {image(2), 1, 0, 0, 1, 37}, {image(1), 2, 0, 0, 1, 37}, {image(1), 1, 1, 0, 1, 37},
        {image(1), 1, 0, 1, 1, 37}, {image(1), 1, 0, 0, 2, 37}, {image(1), 1, 0, 0, 1, 38},
    };
    for (std::uintptr_t i = 0; i < std::size(variants); ++i) {
        check(cache.find(variants[i]) == nullptr, "a key differing in one field is a different key");
        check(cache.insert(variants[i], view(10 + i)), "a key differing in one field is inserted");
    }
    check(cache.size() == 7, "seven distinct keys");

    std::vector<View> released;
    cache.erase_image(image(1), [&](View v) { released.push_back(v); });
    check(released.size() == 6, "destroying image 1 releases its six views");
    check(cache.size() == 1 && cache.find(variants[0]) == view(10), "image 2's view survives");
    check(cache.find(a) == nullptr, "image 1's views are gone");
    released.clear();
    cache.erase_image(image(1), [&](View v) { released.push_back(v); });
    check(released.empty(), "destroying an image twice releases nothing the second time");
    cache.erase_image(image(99), [&](View v) { released.push_back(v); });
    check(released.empty(), "destroying an image with no views releases nothing");

    // A destroyed image's handle can be reused by a new image: its views start empty.
    check(cache.insert(a, view(50)), "a reused image handle takes new views");
    check(cache.find(a) == view(50), "the reused handle's view is the new one");
    std::size_t visited = 0;
    cache.for_each([&](View) { ++visited; });
    check(visited == 2, "for_each visits every cached view");
    cache.clear();
    check(cache.size() == 0 && cache.find(a) == nullptr, "clear empties the cache");
}

// Random operations on both caches; every result must agree.
void random_cases() {
    std::mt19937_64 rng(0x1e0d1da);
    for (int round = 0; round < 200; ++round) {
        Cache cache;
        Reference ref;
        const std::uint32_t images = 1 + rng() % 24;
        std::uintptr_t next_view = 1;
        for (int op = 0; op < 400; ++op) {
            const Cache::Key k{image(rng() % images), std::uint32_t(1u << (rng() % 3)), std::uint32_t(rng() % 4),
                               std::uint32_t(rng() % 6), std::uint32_t(1 + rng() % 2), std::uint32_t(rng() % 3)};
            switch (rng() % 8) {
            case 0: {
                std::vector<View> got, want;
                cache.erase_image(k.image, [&](View v) { got.push_back(v); });
                ref.erase_image(k.image, [&](View v) { want.push_back(v); });
                std::sort(got.begin(), got.end());
                std::sort(want.begin(), want.end());
                check(got == want, "erase_image releases the views the linear scan released");
                break;
            }
            case 1:
            case 2:
            case 3: {
                const View v = view(next_view++);
                check(cache.insert(k, v) == ref.insert(k, v), "insert agrees with std::map::emplace");
                break;
            }
            default:
                check(cache.find(k) == ref.find(k), "find agrees with std::map::find");
            }
            check(cache.size() == ref.views.size(), "sizes agree");
        }
    }
}

using Clock = std::chrono::steady_clock;

double ns_since(Clock::time_point start, std::uint64_t n) {
    return std::chrono::duration<double, std::nano>(Clock::now() - start).count() / double(n);
}

// `images` images with `per_image` views each (aspect, mip and layer vary, as a render target's mips and array
// layers do). Times view() hits in random order and the destroy of one image, then counts allocations of both.
template <class C>
void fill(C& c, std::uint32_t images, std::uint32_t per_image, std::vector<Cache::Key>& keys) {
    keys.clear();
    std::uintptr_t n = 1;
    for (std::uint32_t i = 0; i < images; ++i)
        for (std::uint32_t j = 0; j < per_image; ++j) {
            const Cache::Key k{image(i), j % 2 ? 2u : 1u, j / 2 % 4, j / 8, 1, 37};
            c.insert(k, view(n++));
            keys.push_back(k);
        }
}

template <class C>
void bench_one(const char* name, std::uint32_t images, std::uint32_t per_image, double& lookup_ns, double& destroy_ns,
               std::uint64_t& lookup_allocs, std::uint64_t& destroy_allocs, double& bytes_per_view) {
    std::vector<Cache::Key> keys;
    keys.reserve(std::size_t(images) * per_image);
    const std::uint64_t b0 = allocated_bytes;
    C c;
    fill(c, images, per_image, keys);
    bytes_per_view = double(allocated_bytes - b0) / double(keys.size());
    std::vector<std::uint32_t> order(1u << 20);
    std::mt19937 rng(7);
    for (auto& o : order) o = rng() % keys.size();

    std::uintptr_t sink = 0;
    const std::uint64_t a0 = allocations;
    auto t = Clock::now();
    for (const auto o : order) sink += reinterpret_cast<std::uintptr_t>(c.find(keys[o]));
    lookup_ns = ns_since(t, order.size());
    lookup_allocs = allocations - a0;

    // Destroy and recreate one image at a time: each round is one ~Resource of a render target plus its views'
    // re-creation by the next draw that uses its replacement.
    const std::uint32_t rounds = std::min<std::uint32_t>(images, 2000);
    double erase_total = 0;
    std::uint64_t erase_allocs = 0;
    for (std::uint32_t r = 0; r < rounds; ++r) {
        const Image img = image(rng() % images);
        const std::uint64_t e0 = allocations;
        t = Clock::now();
        c.erase_image(img, [&](View v) { sink += reinterpret_cast<std::uintptr_t>(v); });
        erase_total += std::chrono::duration<double, std::nano>(Clock::now() - t).count();
        erase_allocs += allocations - e0;
        for (std::uint32_t j = 0; j < per_image; ++j) c.insert({img, j % 2 ? 2u : 1u, j / 2 % 4, j / 8, 1, 37}, view(j + 1));
    }
    destroy_ns = erase_total / rounds;
    destroy_allocs = erase_allocs;
    if (sink == 42) std::printf("%s\n", name);  // keeps the loops
}

void bench(std::uint32_t images, std::uint32_t per_image) {
    double ref_lookup, ref_destroy, new_lookup, new_destroy, ref_bytes, new_bytes;
    std::uint64_t ref_la, ref_da, new_la, new_da;
    bench_one<Reference>("map", images, per_image, ref_lookup, ref_destroy, ref_la, ref_da, ref_bytes);
    bench_one<Cache>("cache", images, per_image, new_lookup, new_destroy, new_la, new_da, new_bytes);
    std::printf("%6u images x %2u views: lookup %6.1f -> %5.1f ns (%.1fx), destroy %9.0f -> %6.0f ns (%.0fx), "
                "allocations in lookups %llu -> %llu, in destroys %llu -> %llu, heap %.0f -> %.0f bytes/view\n",
                images, per_image, ref_lookup, new_lookup, ref_lookup / new_lookup, ref_destroy, new_destroy,
                ref_destroy / new_destroy, (unsigned long long)ref_la, (unsigned long long)new_la,
                (unsigned long long)ref_da, (unsigned long long)new_da, ref_bytes, new_bytes);
}

} // namespace

int main(int argc, char** argv) {
    fixed_cases();
    random_cases();
    const bool big = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    bench(256, 2);
    bench(1024, 4);
    if (big) {
        bench(4096, 4);
        bench(16384, 2);
        bench(2048, 32);
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("image_view_cache: all checks passed\n");
    return 0;
}
