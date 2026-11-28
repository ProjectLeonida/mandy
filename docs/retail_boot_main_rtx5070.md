# The retail boot on `main`, RTX 5070: where `gta6rt --run` stops, with and without #601

**Status: one run per tree on one machine, 2026-11-24. A measurement, not a claim on any code item.** It continues
[retail_boot_chain.md](retail_boot_chain.md) for a tree with none of the pull requests that page merged locally, and
then with #601 and #417 merged on top. Nothing here quotes the executable: no addresses, offsets or bytes; only slot
numbers, call shapes, counts and the runtime's own log classes.

Setup: `main` `21625bca8`. AMD Ryzen 7 7800X3D, 32 GB, Windows 11 build 26200, NVIDIA GeForce RTX 5070 (Windows
driver version 32.0.16.1714) and an AMD integrated GPU. `gta6rt` built with Visual Studio 2026 (v18), CMake 4.4,
`RelWithDebInfo`, `VULKAN_SDK` set so that the Vulkan device is built in; `llvm-mc` 23.1.6 from conda-forge. Own
1.0.2.1 copy, `LEONIDA_D3D12=vulkan`, `--overlay` outside the package, no other switches. Every run logs
`[rt] d3d12: Vulkan device: NVIDIA GeForce RTX 5070 (maxPushConstantsSize: 256 bytes)`.

Without Vulkan headers CMake only warns ("Vulkan headers not found ... `LEONIDA_D3D12=vulkan` is unavailable") and the
build still succeeds; `LEONIDA_D3D12=vulkan` is then ignored and the title runs on the host's D3D12 device. A first
version of this page was written from runs like that and was wrong; the figures below are from the rebuilt binary.

## 1. What holds

| Tree | `--run` result |
|---|---|
| `main` `21625bca8` | 13 D3D12 calls, 12 ok; call #13, `ID3D12Device::CreateCommandList`, failed; then the title's `int3` (`0x80000003`) |
| `main` + #601 (`6fbb08cb8`) | 1,639 calls, 1,619 ok, 20 `called` without an outcome; the last is `CreateUnorderedAccessView` (#1639); then the title's `int3`, at another place than above |
| `main` + #601 + #417 | the same 1,639 calls and the same last call as with #601 alone |

Passes on `main`: `make_game.py --stages check,shaders,runtime` (the log ends with `exit=0`, 43 minutes; the merged
`shaders/spirv/index.json` of the main corpus has 26,503 entries), `--dry-run` and `--init` (exit 0, no exception, 4
`DllMain` ok, no unimplemented import). The three runs above were taken before the `shaders` stage had written its
`index.json`, so they ran without translated shaders; section 4 is the run with them. The 4 minute limit was not
reached in any run (the shell reported 3 for the runs above).

The startup trace of the `main` run has 25 events and no bad line: 5 `map`, 2 `tls_callback`, 4 `dllmain`, 5
`xmem_attributes`, 1 `xmem_physical_pages` (attributes `0x280`, 11,360 pages), 6 `debug_string` (text not shown) and 1
`exception`. With #601 the trace has 2 more `xmem_attributes` events and the same exception count.

## 2. Stop 9 on `main`

The runtime logs the first call of device slots 26 (`GetCustomHeapProperties`), 8 (`CreateCommandQueue`), 36
(`CreateFence`), 48 (routed to `CreateCommandQueueX`), 9 (`CreateCommandAllocator`) and 73 (routed to
`CreateCommandListX`, node mask 0, type 0). The D3D12 trace then holds `ID3D12Device::CreateCommandList` as failed.
This is stop 9 of [retail_boot_chain.md](retail_boot_chain.md).

`proxy_create_command_list_x` in `runtime/shims/d3d12_xs_device.cpp` hands the title's `riid` to the PC
`ID3D12Device::CreateCommandList`, clears the output pointer first and returns the failure when the device does not
know the interface. #601 and #359 describe the title's `riid` as a console command-list IID the PC device refuses, so
the title's output pointer stays null and the `int3` is its check on it. The trace shows the failed call, and with
#601 the log says that slot 73 was asked for a console command-list IID, that the list is built as
`ID3D12GraphicsCommandList` and that it is answered with the console list proxy. The `allocators of this device are not
wrapped` line printed before slot 73 comes from slot 9 and is not the cause.

## 3. Past it, with #601

With #601 the boot goes on through the console descriptor heap (`CreateDescriptorHeap`, then heap slots 9 and 10),
slot 70 (a shader-resource view from raw words, answered with a null view), slot 127 (`CreateCommittedResource3`) and
a run of descriptor-view calls (`CreateUnorderedAccessView` from call #1192 on). The 185 MiB `XMemVirtualAlloc`
reserve with attribute `0x7` of stop 10 is in the trace and the boot does not stop at it. The last three calls are
`CreateShaderResourceView` (#1637), `CreateRenderTargetView` (#1638) and `CreateUnorderedAccessView` (#1639); the
last has no outcome in the trace, and the title's `int3` is in a different place than at stop 9. Which numbered stop
this is was not worked out here.

For comparison only: [retail_boot_intel_host.md](retail_boot_intel_host.md) records stop 18 after 1,640 D3D12 calls on
its integration branch (a C++ throw in `XboxBvhBuilder.dll`). This run has one exception in its trace, the `int3`, so it
is not shown to be that stop, and the trees and CPUs differ.

## 4. With the translated shaders and `LEONIDA_SKIP_INT3=1`

`main` + #601 + #417, `LEONIDA_SHADERS` pointing at the work folder's `shaders/spirv`, `LEONIDA_SKIP_INT3=1`. Two runs,
the second after the build had finished; the shader folder differed between them.

| | run A, folder still being completed | run B, finished folder |
|---|---|---|
| `translated shaders` in the log | 26,576 | 29,104 |
| D3D12 calls traced | 1,639, the last `CreateUnorderedAccessView` | 1,655, the last two `CreateShaderResourceView` |
| `int3_skipped` events | 1 | 2 |
| end | write access violation in the title at 32.0 s | read access violation in the title at 9.0 s |

The shell reported 139 for both. Both are an access violation in the title, not a C++ throw.

- The same run without `LEONIDA_SHADERS` printed `no translated shaders`, and the title's first compute pipeline then
  had nothing to load.
- In both runs the log holds the first call of device slot 16 (`CreateRootSignature`) and slot 62 (`CreateComputePipelineState`),
  and right after it `root signature usable for graphics only: root parameter 8 does not fit in the user SGPRs from s4
  (spill array not modelled)`. This is the refusal that the `finding:` in the coordination issue ties to the 52 compute
  shaders embedded in `XboxBvhBuilder.dll`; it is the same message, and these runs do not establish more than that.
- Run B does not stop at that refusal. After it the title creates four committed resources, a command allocator and four
  command lists (each closed at once), a descriptor heap and two shader-resource views, and the log shows the first
  calls of device slots 39, 27, 41 and 61 (the last two forwarded). What the title does with the refused signature is not
  shown.

So with the shaders in place the boot is past the missing-shader stop and the signature refusal does not end the run
by itself. Why run A ended earlier than run B was not isolated: the shader folder differs, and neither run was repeated.

## 5. What this does not show

No repeat of any run, no debugger. Which instruction the last `int3` guards is not known, and the two runs of section 4
disagree with each other. #417 did not change the stop in the runs of section 1; whether it matters further on is not
shown. The `translated shaders` count of section 4 depends on how far the build's last step had got, and was not tied
to the shader groups of the index.

## 6. Reproduce

The commands of [boot.md](boot.md) section 2, with `--overlay` and an empty `--trace` name, after checking that the
log carries the `Vulkan device:` line. Keep the traces out of the repository: they hold addresses of the executable.
