// Portable C++20 tests for runtime/shims/d3d12_feature_compat.h
#include "../shims/d3d12_feature_compat.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

int main() {
    using namespace rt::d3d12::compat;

    // 1. ABI constant values.
    CHECK(static_cast<UINT>(kFeatureOptions20) == 49u);
    CHECK(static_cast<UINT>(kFeaturePredication) == 50u);
    CHECK(static_cast<UINT>(kFeatureOptions21) == 53u);
    CHECK(static_cast<UINT>(kFeatureApplicationSpecificDriverState) == 56u);
    CHECK(static_cast<UINT>(kFeatureBytecodeBypassHashSupported) == 57u);

    CHECK(kRecreateAtTierNotSupported == 0u);
    CHECK(kWorkGraphsTierNotSupported == 0u);
    CHECK(kExecuteIndirectTier1_0 == 10u);
    CHECK(static_cast<UINT>(kIndirectArgumentIncrementingConstant) ==
          static_cast<UINT>(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH + 1));

    // 2. Struct sizes and member layout.
    CHECK(sizeof(FeatureSupportFlag) == sizeof(BOOL));
    CHECK(sizeof(FeatureDataOptions20) == 8);
    CHECK(offsetof(FeatureDataOptions20, ComputeOnlyWriteWatchSupported) == 0);
    CHECK(offsetof(FeatureDataOptions20, RecreateAtTier) == 4);

    CHECK(sizeof(FeatureDataOptions21) == 16);
    CHECK(offsetof(FeatureDataOptions21, WorkGraphsTier) == 0);
    CHECK(offsetof(FeatureDataOptions21, ExecuteIndirectTier) == 4);
    CHECK(offsetof(FeatureDataOptions21, SampleCmpGradientAndBiasSupported) == 8);
    CHECK(offsetof(FeatureDataOptions21, ExtendedCommandInfoSupported) == 12);

    // 3. Identification predicate: is_compat_feature.
    CHECK(is_compat_feature(kFeatureOptions20));
    CHECK(is_compat_feature(kFeaturePredication));
    CHECK(is_compat_feature(kFeatureOptions21));
    CHECK(is_compat_feature(kFeatureApplicationSpecificDriverState));
    CHECK(is_compat_feature(kFeatureBytecodeBypassHashSupported));

    CHECK(!is_compat_feature(D3D12_FEATURE_D3D12_OPTIONS));
    CHECK(!is_compat_feature(D3D12_FEATURE_ARCHITECTURE));
    CHECK(!is_compat_feature(D3D12_FEATURE_FEATURE_LEVELS));
    CHECK(!is_compat_feature(static_cast<D3D12_FEATURE>(48)));
    CHECK(!is_compat_feature(static_cast<D3D12_FEATURE>(51)));
    CHECK(!is_compat_feature(static_cast<D3D12_FEATURE>(52)));
    CHECK(!is_compat_feature(static_cast<D3D12_FEATURE>(54)));
    CHECK(!is_compat_feature(static_cast<D3D12_FEATURE>(55)));
    CHECK(!is_compat_feature(static_cast<D3D12_FEATURE>(58)));
    CHECK(!is_compat_feature(static_cast<D3D12_FEATURE>(9999)));

    // 4. Expected data payload sizes: compat_feature_data_size.
    CHECK(compat_feature_data_size(kFeatureOptions20) == sizeof(FeatureDataOptions20));
    CHECK(compat_feature_data_size(kFeatureOptions21) == sizeof(FeatureDataOptions21));
    CHECK(compat_feature_data_size(kFeaturePredication) == sizeof(FeatureSupportFlag));
    CHECK(compat_feature_data_size(kFeatureApplicationSpecificDriverState) == sizeof(FeatureSupportFlag));
    CHECK(compat_feature_data_size(kFeatureBytecodeBypassHashSupported) == sizeof(FeatureSupportFlag));

    CHECK(compat_feature_data_size(D3D12_FEATURE_D3D12_OPTIONS) == 0);
    CHECK(compat_feature_data_size(D3D12_FEATURE_ARCHITECTURE) == 0);
    CHECK(compat_feature_data_size(static_cast<D3D12_FEATURE>(9999)) == 0);

    // 5. Size validation helper: is_valid_compat_feature_size.
    CHECK(is_valid_compat_feature_size(kFeatureOptions20, sizeof(FeatureDataOptions20)));
    CHECK(!is_valid_compat_feature_size(kFeatureOptions20, 0));
    CHECK(!is_valid_compat_feature_size(kFeatureOptions20, sizeof(FeatureDataOptions20) - 1));
    CHECK(!is_valid_compat_feature_size(kFeatureOptions20, sizeof(FeatureDataOptions20) + 1));
    CHECK(!is_valid_compat_feature_size(kFeatureOptions20, 16));

    CHECK(is_valid_compat_feature_size(kFeatureOptions21, sizeof(FeatureDataOptions21)));
    CHECK(!is_valid_compat_feature_size(kFeatureOptions21, 0));
    CHECK(!is_valid_compat_feature_size(kFeatureOptions21, 8));
    CHECK(!is_valid_compat_feature_size(kFeatureOptions21, 24));

    CHECK(is_valid_compat_feature_size(kFeaturePredication, sizeof(FeatureSupportFlag)));
    CHECK(!is_valid_compat_feature_size(kFeaturePredication, 0));
    CHECK(!is_valid_compat_feature_size(kFeaturePredication, 8));

    CHECK(is_valid_compat_feature_size(kFeatureApplicationSpecificDriverState, sizeof(FeatureSupportFlag)));
    CHECK(!is_valid_compat_feature_size(kFeatureApplicationSpecificDriverState, 0));

    CHECK(is_valid_compat_feature_size(kFeatureBytecodeBypassHashSupported, sizeof(FeatureSupportFlag)));
    CHECK(!is_valid_compat_feature_size(kFeatureBytecodeBypassHashSupported, 0));

    // Non-compat features return false even if size happens to match 4 or 8.
    CHECK(!is_valid_compat_feature_size(D3D12_FEATURE_D3D12_OPTIONS, 4));
    CHECK(!is_valid_compat_feature_size(D3D12_FEATURE_D3D12_OPTIONS, 8));

    // 6. Identifier naming: compat_feature_name.
    CHECK(std::strcmp(compat_feature_name(kFeatureOptions20), "D3D12_FEATURE_D3D12_OPTIONS20") == 0);
    CHECK(std::strcmp(compat_feature_name(kFeaturePredication), "D3D12_FEATURE_PREDICATION") == 0);
    CHECK(std::strcmp(compat_feature_name(kFeatureOptions21), "D3D12_FEATURE_D3D12_OPTIONS21") == 0);
    CHECK(std::strcmp(compat_feature_name(kFeatureApplicationSpecificDriverState),
                      "D3D12_FEATURE_APPLICATION_SPECIFIC_DRIVER_STATE") == 0);
    CHECK(std::strcmp(compat_feature_name(kFeatureBytecodeBypassHashSupported),
                      "D3D12_FEATURE_BYTECODE_BYPASS_HASH_SUPPORTED") == 0);
    CHECK(compat_feature_name(D3D12_FEATURE_D3D12_OPTIONS) == nullptr);
    CHECK(compat_feature_name(static_cast<D3D12_FEATURE>(9999)) == nullptr);

    // 7. Value and default initialization behaviors.
    FeatureDataOptions20 o20{};
    CHECK(o20.ComputeOnlyWriteWatchSupported == FALSE);
    CHECK(o20.RecreateAtTier == kRecreateAtTierNotSupported);

    FeatureDataOptions21 o21{};
    CHECK(o21.WorkGraphsTier == kWorkGraphsTierNotSupported);
    CHECK(o21.ExecuteIndirectTier == 0);
    CHECK(o21.SampleCmpGradientAndBiasSupported == FALSE);
    CHECK(o21.ExtendedCommandInfoSupported == FALSE);

    o21.ExecuteIndirectTier = kExecuteIndirectTier1_0;
    CHECK(o21.ExecuteIndirectTier == 10u);

    FeatureSupportFlag flag{TRUE};
    CHECK(flag.Supported == TRUE);

    std::printf("d3d12_feature_compat: all ABI constants, struct geometry, validation and naming tests passed\n");
    return 0;
}
