# Placed buffers in `OpenExistingHeapFromAddress` heaps (`RT-EXISTING-HEAP-PLACED`)

This note records the measured Windows D3D12 contract, Vulkan host-memory import rules, and Intel WDDM residency findings for placing buffers inside `ID3D12Device3::OpenExistingHeapFromAddress` heaps (`runtime/shims/d3d12_vulkan_device.cpp`).

## 1. Host D3D12 cross-adapter and creation rules

An `OpenExistingHeapFromAddress` heap opens a `VirtualAlloc` region whose entire committed span is `PAGE_READWRITE` or `PAGE_READWRITE | PAGE_WRITECOMBINE` and reports `D3D12_HEAP_TYPE_CUSTOM`, `D3D12_MEMORY_POOL_L0`, `D3D12_CPU_PAGE_PROPERTY_WRITE_BACK` (or `WRITE_COMBINE`), alignment `64 KiB`, and flags `SHARED | SHARED_CROSS_ADAPTER | ALLOW_SHADER_ATOMICS` (`0x421`).

Measured across live Windows 11 D3D12 adapters (NVIDIA GeForce RTX 5060 Laptop GPU, Intel UHD Graphics / RaptorLake-S, and WARP):

| Case on an existing heap | Windows host D3D12 | Vulkan layer (`create_placed`) | Notes |
|---|---|---|---|
| Buffer with `D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER` (`0x10`) at 64 KiB-aligned offset | `S_OK` | `S_OK` | Same `GetHeapProperties` (`CUSTOM / WRITE_BACK or WRITE_COMBINE / L0`, node masks `1 / 1`, flags `0x421`). |
| Buffer without `ALLOW_CROSS_ADAPTER` (e.g. `NONE` or `ALLOW_UNORDERED_ACCESS` alone) | `E_INVALIDARG` | `E_INVALIDARG` | Applies to `CreatePlacedResource`, `CreatePlacedResource1`, and `CreatePlacedResource2`. |
| Unaligned offset (`4096`), out-of-bounds range, zero width, clear value on buffer | `E_INVALIDARG` | `E_INVALIDARG` | Checked before `E_NOTIMPL` or `out == nullptr`. |
| Valid buffer args with `ppvResource == nullptr` | `E_UNEXPECTED` | `E_UNEXPECTED` | Nothing created. |
| `CreatePlacedResource2` with `D3D12_BARRIER_LAYOUT_COMMON` or castable format list on buffer | `E_INVALIDARG` | `E_INVALIDARG` | `D3D12_BARRIER_LAYOUT_UNDEFINED` succeeds. |
| Placed texture (`TEXTURE2D`) with `ALLOW_CROSS_ADAPTER` | driver-dependent | `E_NOTIMPL` | Only placed buffers in existing heaps are implemented; placed textures stay unimplemented cleanly. |

On heaps without `D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER` (and in `CreateCommittedResource`), `D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER` remains refused (`E_INVALIDARG`), matching Windows host D3D12.

Through the Xbox-ABI proxy (`runtime/shims/d3d12_xs_device.cpp`), vtable slot `48` is shadowed by Xbox `CreateCommandQueueX`, whereas slot `29` (`CreatePlacedResource`) forwards directly to the backing PC device and accepts heaps opened on that device.

## 2. Vulkan backing, `Map()`, and GPU virtual addresses

- **Memory type selection:** `OpenExistingHeapFromAddress` probes a temporary `VkBuffer` configured with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT` (`DeviceState::host_buffer_info`) and intersects `VkMemoryHostPointerPropertiesEXT::memoryTypeBits & probe_req.memoryTypeBits` before `vkAllocateMemory`. `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` and `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` are attached only when `DeviceState::need_buffer_device_address()` is true (explicit `buffer_device_address` or live GPU address arena).
- **Zero-copy buffer placement (`Resource::create_existing_heap_buffer`):** `CreatePlacedResource`, `CreatePlacedResource1`, and `CreatePlacedResource2` create a `VkBuffer` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT` and bind it directly to the heap's imported `VkDeviceMemory` at `heap_offset`.
- **Preservation and visibility:** Pre-filled host pages are never zeroed on heap creation or buffer placement. `Map(0, nullptr, &ptr)` returns the caller's live address (`heap_host_address + heap_offset`). `ResourceBinding::host_readable` is set `true` so GPU writes into placed buffers emit a `TRANSFER -> HOST_READ` barrier (`runtime/src/vulkan_backend.h` and `runtime/src/vulkan_replay.h`) before completion, making GPU writes immediately visible to direct CPU reads on `VirtualAlloc` pages.
- **Address tracking and aliasing:** When the host address sits above `arena.size`, `va_` takes the host virtual address (`buffer_va_ - arena.address` otherwise), keeping `GetGPUVirtualAddress()` monotonically offset by `heap_offset` without colliding with arena offsets. Overlapping placed buffers in the same existing heap are registered in `BufferVaIndex` (`runtime/src/buffer_va_index.h`), which resolves replay lookups to a binding covering the requested byte range.

## 3. Teardown safety and the Intel WDDM unpin bug

`VK_EXT_external_memory_host` requires caller pages to outlive the imported `VkDeviceMemory`, and `ID3D12Resource` holds a `std::shared_ptr<const leonida::TileHeapAllocation>` reference so releasing `ID3D12Heap` before releasing placed buffers keeps the imported `VkDeviceMemory` alive until the last placed buffer dies. On destruction:

1. `Resource::~Resource()` waits on `context->wait_retire_ticket(context->submitted_retire_ticket())`, calls `vkDestroyBuffer` **before** `heap_range_.reset()` so buffers always die before the heap's `VkDeviceMemory`.
2. When the last reference to the existing heap's `TileHeapAllocation` drops, its deleter waits for the latest submitted retire ticket and calls `vkFreeMemory`.

**Intel WDDM finding (RaptorLake-S, vendor `0x8086`):** Direct Vulkan probes showed that calling `vkFreeMemory` on a `VK_EXT_external_memory_host` import and immediately returning the pages with `VirtualFree` without a real GPU submission in between leaves Intel's Windows WDDM driver asynchronously unpinning `ExistingSysMem` against freed pages, driving subsequent `vkQueueSubmit` calls into `VK_ERROR_DEVICE_LOST` (`-4`). Calling `vkDeviceWaitIdle`, `vkQueueWaitIdle`, or submitting an empty command buffer did not clear the unpin, whereas submitting a single barrier-only command buffer (`vkCmdPipelineBarrier` via `DeviceState::flush_host_import_unpin()` on `staging_queue`) and idling the queue resolves it deterministically across both `PAGE_READWRITE` and `PAGE_READWRITE | PAGE_WRITECOMBINE`.

## 4. Verification

All checks run with `runtime/test/d3d12_existing_heap_test.cpp` and `runtime/test/d3d12_existing_heap_placed_test.cpp` (`0` compiler warnings under MSVC `/W4 /permissive-`, `0` Vulkan errors):

- **NVIDIA GeForce RTX 5060 Laptop GPU (driver `621.24`):** `8/8` `OpenExistingHeapFromAddress` parity cases + `15/15` placed existing-heap parity cases (`CreatePlacedResource`/`1`/`2` + `GetHeapProperties`), `no-bda` and `bda+arena` bidirectional CPU↔GPU coherence, overlapping aliases, heap-before-buffer release, `64`-iteration `VirtualAlloc`→`VirtualFree` stress loop, and Xbox device proxy (`slot 29`) all pass.
- **Intel(R) RaptorLake-S Mobile Graphics Controller:** Same `8/8` and `15/15` parity cases plus all coherence, aliasing, lifetime, `64`-iteration stress, and Xbox proxy checks pass; pre-existing failure in `d3d12_existing_heap` after `open_cases` is eliminated by `flush_host_import_unpin()`.
- **Microsoft Basic Render Driver (WARP):** `8/8` `OpenExistingHeapFromAddress` parity cases and `15/15` existing-heap `CreatePlacedResource`/`1`/`2` + `GetHeapProperties` parity cases match identically.
