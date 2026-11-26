#pragma once

#include <d3d12.h>
#include <wrl/client.h>

namespace sparse_test {

struct FormatCase {
    DXGI_FORMAT format;
    UINT bytes;
    UINT tile_width;
    UINT tile_height;
    UINT width;
    UINT height;
};

bool sparse_image(ID3D12Resource* resource);
D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT format, UINT width, UINT height);
bool linear_buffer(ID3D12Device& device, UINT64 bytes, D3D12_HEAP_TYPE heap_type, D3D12_RESOURCE_STATES state,
                   Microsoft::WRL::ComPtr<ID3D12Resource>& resource);
void transition(ID3D12GraphicsCommandList& list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after);
// Closes and submits list, checks shim replay when layer is true, then waits for completion.
bool submit_and_wait(ID3D12Device& device, ID3D12CommandQueue& queue, ID3D12GraphicsCommandList& list, bool layer);

bool test_transfers(ID3D12Device& device, ID3D12Device* warp, const FormatCase& c);
bool test_mappings(ID3D12Device& device, bool native);

}  // namespace sparse_test
