// The PC form of the resource descriptions the retail title builds for the console, for the Xbox-ABI device proxy.
//
// The console lets a depth-stencil texture be a TEXTURE3D. The measured call is the retail boot's stop after #517:
// device slot 127 (CreateCommittedResource3) asks for a 768x768x6 R16_TYPELESS volume with one mip, one sample and
// ALLOW_DEPTH_STENCIL, and the PC device refuses it. PC D3D12 has no volume depth-stencil: the host's
// GetResourceAllocationInfo answers UINT64_MAX and CreateCommittedResource3 fails, which this layer reports as the
// same refusal (d3d12_vulkan_device.cpp: allocation()'s "no volume depth" check and texture_format()'s depth-aspect
// check). The proxy describes that resource as a 2D depth array instead: Dimension TEXTURE2D with DepthOrArraySize
// unchanged, so each console depth slice is one array slice. The format (R16_TYPELESS, still in the depth-stencil
// family), ALLOW_DEPTH_STENCIL, the clear value the caller passes separately, the mip count, the sample description,
// width and height are the title's words and are left alone.
//
// This is a translation at the console boundary, not a relaxation of the PC rule: a PC caller that asks the layer
// itself for a volume depth-stencil resource is still refused. The caller's description is never written. Slots 25
// (GetResourceAllocationInfo, which sizes the resource before creation), 76 (CreateCommittedResource) and 127
// (CreateCommittedResource3) and the slot 133 (CreatePlacedResourceX) handler each work on their own copy. When the
// console's slot 108 (GetResourceAllocationInfo2) is routed, its descriptions must take the same path.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <cstdint>

namespace rt::d3d12_xs {

// True for a console description D3D12 can only make as a 2D depth array: a volume depth-stencil texture. A 3D
// colour texture (no ALLOW_DEPTH_STENCIL) keeps its dimension.
template <class Desc>
inline bool is_volume_depth_stencil(const Desc& desc) {
    return desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
           (static_cast<std::uint32_t>(desc.Flags) & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
}

// A copy of `xbox` with a volume depth-stencil described as a 2D depth array, or an unchanged copy. The array size is
// the volume's slice count, which is already DepthOrArraySize.
template <class Desc>
inline Desc pc_depth_desc(const Desc& xbox) {
    Desc pc = xbox;
    if (is_volume_depth_stencil(pc)) pc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    return pc;
}

}  // namespace rt::d3d12_xs
