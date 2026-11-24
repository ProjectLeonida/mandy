# The runtime ctest suite on two hosts: Wine and native Windows

The same runtime test suite, the same GPU model, two hosts: Wine on Linux ([#1150](https://github.com/ubiquitarianism/leonida/issues/1150)) and native Windows 10. The point is not to compare percentages but to separate the failures that belong to Wine's host DLLs from the ones that are ours. A test that fails only under Wine is a limit of the host; a test that fails on both is a bug in the layer, and one that fails only natively is a bug Wine was hiding.

No game files were used by either run. The suite builds its own fixtures.

## The two runs

| | Wine run (#1150) | Native run (this page) |
|---|---|---|
| Host | Parrot Linux (Debian 13), Wine 10.0 | Windows 10 Pro 19045, no Wine |
| GPU | NVIDIA GPU (Pascal class, 4 GB). Model and PCI id withheld | the same card |
| Driver | NVIDIA 555.58 | NVIDIA 582.28 |
| Compiler | clang-cl 21 / lld-link, xwin SDK 10.0.26100 | MSVC 19.44 x64, installed SDK |
| Commit | `3aa97542` | `58be9cdf` |
| Tests run | 304 of 304 | 307 of 308 (one disabled) |
| Result | 239 passed, 19 failed, 5 timeouts, 40 skipped — 92% | 278 passed, 11 failed (9 failed, 1 SegFault, 1 timeout), 18 skipped — 96%, 452 s |
| ctest valve | `--timeout 120` | `--timeout 120` |

The commits differ by a day of `main`. Nothing in the 24 Wine failures is a test added between them except `d3d12_existing_heap`, `d3d12_xmem_placed` and `root_sig`, which all exist in both.

## 19 Wine failures pass natively

These are host limits. Wine's `ntdll`/`bcrypt`/`dxgi`/vkd3d answer differently from the Windows DLLs the tests compare against:

| Test | What Wine does that heals here |
|---|---|
| `xmem` | protection windows, page charging and read-only faults: Windows virtual-memory semantics Wine does not give |
| `file_system` | `ReadFile` into uncommitted pages does not fail with `ERROR_NO_ACCESS` |
| `bcrypt_rng`, `bcrypt_rng_seeded` | the unknown-algorithm probe: the shim delegates to Wine's `bcrypt`, which answers another status |
| `d3d12_clear`, `d3d12_clear_formats` | no WARP under Wine, so the differential test cannot run |
| `dxgi_swapchain` | `CreateSwapChainForHwnd` returned `0x887a0001` against the host's `0x00000000` |
| `dstorage` | the timeout with no output after `OpenFile: null or empty path` is Wine's, not ours |
| `run_startup_cmdline` | the Wine init-message issue |
| `vk_device_filter_tools` | the script ran the `.exe` directly on Linux (`Exec format error`) instead of through Wine |
| `root_sig` | the all-zero-checksum Xbox blob: Wine's chained serialization, as the issue suspected |
| `d3d12_residency`, `d3d12_fence_timing`, `d3d12_indirect_dispatch_count`, `d3d12_xmem_placed` | the documented host-comparison family |
| `d3d12_buffer_tiles`, `d3d12_texture_tiles` | the sparse-residency deadlock: the first `UpdateTileMappings` is refused under Wine and teardown blocks on the queue |
| `dxgi_host_parity`, `dxgi_waitable_latency` | the page fault on write to `0x0` only turned into a hang because winedbg attached |

The four hangs are the clearest case: natively they are ordinary sub-second passes.

## 5 Wine failures reproduce natively, so they are ours

| Test | Measured |
|---|---|
| `d3d12_vulkan_device` | fails as under Wine |
| `d3d12_subresource_copy` | `CPU copies (host image copy): the steps differ from the host's`, line 391 |
| `d3d12_heap_properties` | `heap properties committed texture, CUSTOM write-back L0: ours 0x8007000e host 0x0` — our `E_OUTOFMEMORY` where the host creates the resource |
| `d3d12_existing_heap` | `a title-placed buffer (device slot 133) could not import its pages…`, line 196 `GetGPUVirtualAddress() == (D3D12_GPU_VIRTUAL_ADDRESS)pages` (`HRESULT 0x80004001`) |
| `d3d12_r9g9b9e5_uav` | line 669 `SUCCEEDED(device->CreateCommittedResource(&custom_heap, ..., &cpu_uav.p))` — CPU-heap `HOST_TRANSFER` texture gets `E_OUTOFMEMORY` on Pascal (same root cause as `d3d12_subresource_copy` and `d3d12_heap_properties`, after the expected `GetResourceAllocationInfo format not supported yet: 67` refusal log) |

## 6 that only fail natively, deterministic 3 of 3

Each was rerun three times alone with `ctest -R '^name$' --repeat until-fail:3`; every repetition failed the same way on this build.

| Test | Symptom |
|---|---|
| `d3d12_subgroup` | **SegFault** at 1.4 s, right after `device slot 16 (CreateRootSignature) first call`. Stack-layout-dependent: device slot 14 (`CreateDescriptorHeap`) accepted an unaligned caller stack word (`0x65626f72705f7370`, `"ps_probe"`) as a console heap address whenever the next word happened to equal `32` ([#1328](https://github.com/ubiquitarianism/leonida/pull/1328)); an independent Pascal 4 GB build with a different stack frame passed 3/3 |
| `d3d12_set_descriptor_heaps` | `CHECK failed: !blob.empty()` (line 341), after two `upload heaps in system memory (no Resizable BAR)` lines |
| `root_signature_oracle` | `FAIL 1.1 with every root-signature flag: Microsoft serializer accepts it` (`Unsupported bit-flag set (root signature flags c7f)` — Windows 10's `d3d12.dll` refuses the SM 6.6 heap-direct bits `0xC00`) |
| `d3d12_close_parity` | the compared steps diverge from the host from the third case on |
| `d3d12_pso_validation` | line 436 `check_gfx_parity("gfx: Flags=DYNAMIC_DEPTH_BIAS", ...)` (`layer 0x00000000 != host 0x80070057`: Windows 10's runtime refuses `DYNAMIC_DEPTH_BIAS`; the `depth-stencil format not supported: 9999, ...` lines are negative-case logs) |
| `xdxr_build_check` | a **Timeout at 120 s** that is not a hang: with `--timeout 400` it passes in **212 s** |

`d3d12_subgroup` is the most serious: a crash in device slot 14's console-layout heuristic ([#1328](https://github.com/ubiquitarianism/leonida/pull/1328)) that depended on adjacent stack words, which is why neither the `clang-cl` Wine build nor a second MSVC build on the same Pascal 4 GB class hit it. `xdxr_build_check` costs a full-suite run two minutes of wall clock and a false fail on this card; 212 s on a 4 GB Pascal for a DXR build is a plausible ceiling for any low-end machine.

## Reading it

- A native pass does not mean the Wine report was wrong: it means the missing piece was Wine. Both runs stay useful.
- Skipped here are 18 (7 retail-package skips plus 11 validation, DXR, conservative-tier-3 and mesh-shader skips); the Wine run skipped 40 for its own reasons. A skip is neither a pass nor a fail and the percentages do not compare one for one.
- This run was from an interactive desktop session, where `xgameruntime_offline` and `xgameruntime_offline_hdr` pass (`0.11 s` and `0.03 s`); over a non-interactive SSH session they fail `XDisplayAcquireTimeoutDeferral`.
- The implicit Vulkan layers were disabled (`VK_LOADER_LAYERS_DISABLE=~implicit~`): this machine has OBS and RenderDoc layers installed.

## What follows

The 5 reproduced and the 6 native-only are the actionable list (5 + 6 = 11 failures). Since this run: [#1364](https://github.com/ubiquitarianism/leonida/pull/1364) fixed the Pascal `HOST_TRANSFER` CPU-heap image failure behind `d3d12_subresource_copy`, `d3d12_r9g9b9e5_uav` and the `0x8007000e` rows of `d3d12_heap_properties` and `d3d12_vulkan_device`; open pull requests cover the remaining host-tier-1 heap comparisons ([#1389](https://github.com/ubiquitarianism/leonida/pull/1389)), sub-page host buffer imports on 256-byte UBO GPUs (`d3d12_existing_heap`, [#1458](https://github.com/ubiquitarianism/leonida/pull/1458)), `d3d12_subgroup` ([#1328](https://github.com/ubiquitarianism/leonida/pull/1328)), `d3d12_set_descriptor_heaps` ([#1353](https://github.com/ubiquitarianism/leonida/pull/1353)), `root_signature_oracle` ([#988](https://github.com/ubiquitarianism/leonida/pull/988)), `d3d12_close_parity` ([#989](https://github.com/ubiquitarianism/leonida/pull/989)), `d3d12_pso_validation` ([#967](https://github.com/ubiquitarianism/leonida/pull/967), [#1377](https://github.com/ubiquitarianism/leonida/pull/1377)), and `xdxr_build_check`'s timeout ([#1424](https://github.com/ubiquitarianism/leonida/pull/1424)).
