#pragma once

#include <d3d12.h>

#include <cstddef>

// D3D12 values newer than some Windows SDK 10.0.26100 releases. The README asks for "Windows SDK 10.0.26100 or newer",
// but the SDK's d3d12.h moved within 10.0.26100 (D3D12_SDK_VERSION 612 in 10.0.26100.6584, 616 later), so anything the
// runtime uses from a newer header is declared here with its ABI value and payload, and checked against the header
// where the header has it. That keeps the runtime and its tests building on every 10.0.26100 SDK.
namespace rt::d3d12::compat {

// Feature values introduced after Windows SDK 10.0.26100.0.
inline constexpr D3D12_FEATURE kFeatureApplicationSpecificDriverState = static_cast<D3D12_FEATURE>(56);
inline constexpr D3D12_FEATURE kFeatureBytecodeBypassHashSupported = static_cast<D3D12_FEATURE>(57);

struct FeatureSupportFlag {
    BOOL Supported;
};

static_assert(sizeof(FeatureSupportFlag) == sizeof(BOOL));

// D3D12_FEATURE_D3D12_OPTIONS20 and D3D12_FEATURE_PREDICATION: D3D12_SDK_VERSION 611 (DirectX-Headers v1.611.0); not
// in 610. PREDICATION's payload is one BOOL, FeatureSupportFlag above.
inline constexpr D3D12_FEATURE kFeatureOptions20 = static_cast<D3D12_FEATURE>(49);
inline constexpr D3D12_FEATURE kFeaturePredication = static_cast<D3D12_FEATURE>(50);
inline constexpr UINT kRecreateAtTierNotSupported = 0;  // D3D12_RECREATE_AT_TIER_NOT_SUPPORTED

struct FeatureDataOptions20 {
    BOOL ComputeOnlyWriteWatchSupported;
    UINT RecreateAtTier;  // D3D12_RECREATE_AT_TIER
};

static_assert(sizeof(FeatureDataOptions20) == 8);

#if defined(D3D12_SDK_VERSION) && D3D12_SDK_VERSION >= 611
static_assert(kFeatureOptions20 == D3D12_FEATURE_D3D12_OPTIONS20);
static_assert(kFeaturePredication == D3D12_FEATURE_PREDICATION);
static_assert(kRecreateAtTierNotSupported == UINT(D3D12_RECREATE_AT_TIER_NOT_SUPPORTED));
static_assert(sizeof(FeatureDataOptions20) == sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS20));
static_assert(sizeof(D3D12_RECREATE_AT_TIER) == sizeof(UINT));
static_assert(offsetof(FeatureDataOptions20, RecreateAtTier) ==
              offsetof(D3D12_FEATURE_DATA_D3D12_OPTIONS20, RecreateAtTier));
static_assert(sizeof(FeatureSupportFlag) == sizeof(D3D12_FEATURE_DATA_PREDICATION));
#endif

// D3D12_FEATURE_D3D12_OPTIONS21 (work graphs and the ExecuteIndirect tier): D3D12_SDK_VERSION 613 (DirectX-Headers
// v1.613.0); not in 611. The enums are UINT-sized in d3d12.h, as here.
inline constexpr D3D12_FEATURE kFeatureOptions21 = static_cast<D3D12_FEATURE>(53);
inline constexpr UINT kWorkGraphsTierNotSupported = 0;    // D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED
inline constexpr UINT kExecuteIndirectTier1_0 = 10;       // D3D12_EXECUTE_INDIRECT_TIER_1_0

struct FeatureDataOptions21 {
    UINT WorkGraphsTier;       // D3D12_WORK_GRAPHS_TIER
    UINT ExecuteIndirectTier;  // D3D12_EXECUTE_INDIRECT_TIER
    BOOL SampleCmpGradientAndBiasSupported;
    BOOL ExtendedCommandInfoSupported;
};

static_assert(sizeof(FeatureDataOptions21) == 16);

// D3D12_INDIRECT_ARGUMENT_TYPE_INCREMENTING_CONSTANT (ExecuteIndirect tier 1.1): the value after DISPATCH_MESH, from
// the same header version as OPTIONS21.
inline constexpr D3D12_INDIRECT_ARGUMENT_TYPE kIndirectArgumentIncrementingConstant =
    static_cast<D3D12_INDIRECT_ARGUMENT_TYPE>(D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH + 1);

#if defined(D3D12_SDK_VERSION) && D3D12_SDK_VERSION >= 613
// A header that declares these must agree with the values above.
static_assert(kFeatureOptions21 == D3D12_FEATURE_D3D12_OPTIONS21);
static_assert(kWorkGraphsTierNotSupported == UINT(D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED));
static_assert(kExecuteIndirectTier1_0 == UINT(D3D12_EXECUTE_INDIRECT_TIER_1_0));
static_assert(kIndirectArgumentIncrementingConstant == D3D12_INDIRECT_ARGUMENT_TYPE_INCREMENTING_CONSTANT);
static_assert(sizeof(FeatureDataOptions21) == sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS21));
static_assert(sizeof(D3D12_WORK_GRAPHS_TIER) == sizeof(UINT) && sizeof(D3D12_EXECUTE_INDIRECT_TIER) == sizeof(UINT));
static_assert(offsetof(FeatureDataOptions21, ExecuteIndirectTier) ==
              offsetof(D3D12_FEATURE_DATA_D3D12_OPTIONS21, ExecuteIndirectTier));
static_assert(offsetof(FeatureDataOptions21, SampleCmpGradientAndBiasSupported) ==
              offsetof(D3D12_FEATURE_DATA_D3D12_OPTIONS21, SampleCmpGradientAndBiasSupported));
static_assert(offsetof(FeatureDataOptions21, ExtendedCommandInfoSupported) ==
              offsetof(D3D12_FEATURE_DATA_D3D12_OPTIONS21, ExtendedCommandInfoSupported));
#endif

// Returns true if the feature is one of the compatibility-defined features in this header.
constexpr bool is_compat_feature(D3D12_FEATURE feature) noexcept {
    return feature == kFeatureOptions20 ||
           feature == kFeaturePredication ||
           feature == kFeatureOptions21 ||
           feature == kFeatureApplicationSpecificDriverState ||
           feature == kFeatureBytecodeBypassHashSupported;
}

// Returns the expected byte size of the payload for a compat feature, or 0 if not a recognized compat feature.
constexpr std::size_t compat_feature_data_size(D3D12_FEATURE feature) noexcept {
    switch (feature) {
        case kFeatureOptions20:
            return sizeof(FeatureDataOptions20);
        case kFeatureOptions21:
            return sizeof(FeatureDataOptions21);
        case kFeaturePredication:
        case kFeatureApplicationSpecificDriverState:
        case kFeatureBytecodeBypassHashSupported:
            return sizeof(FeatureSupportFlag);
        default:
            return 0;
    }
}

// Returns true if the size matches the expected size for the compat feature.
constexpr bool is_valid_compat_feature_size(D3D12_FEATURE feature, std::size_t size) noexcept {
    const std::size_t expected = compat_feature_data_size(feature);
    return expected != 0 && expected == size;
}

// Returns a human-readable identifier for known compat features, or nullptr for unrecognized features.
constexpr const char* compat_feature_name(D3D12_FEATURE feature) noexcept {
    switch (feature) {
        case kFeatureOptions20:
            return "D3D12_FEATURE_D3D12_OPTIONS20";
        case kFeaturePredication:
            return "D3D12_FEATURE_PREDICATION";
        case kFeatureOptions21:
            return "D3D12_FEATURE_D3D12_OPTIONS21";
        case kFeatureApplicationSpecificDriverState:
            return "D3D12_FEATURE_APPLICATION_SPECIFIC_DRIVER_STATE";
        case kFeatureBytecodeBypassHashSupported:
            return "D3D12_FEATURE_BYTECODE_BYPASS_HASH_SUPPORTED";
        default:
            return nullptr;
    }
}

}  // namespace rt::d3d12::compat
