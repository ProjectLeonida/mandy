# Descriptor lookup cost on NVIDIA RTX 3070 (Ampere)

Divergent lookups in host-visible system RAM showed a steep cliff once the table reached 100,000 entries on this PCIe 3.0 x16 Ampere card with Resizable BAR off. On the GPU-only memory type, one million logical lane lookups took 270.3 µs at 1,000 live descriptors, 773.3 µs at 10,000 and 2,502.6 µs at 100,000 for divergent misses, and 73.0, 94.3 and 231.9 µs for divergent texture hits. The 214 MiB BAR1 memory type (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, exposed even with Resizable BAR disabled) stayed within 0.5% of GPU-only in every cell across all three sizes. By contrast, host-visible memory outside VRAM stayed within 10% of GPU-only at 1,000 entries and within 16% at 10,000, but at 100,000 entries it was 5.0× slower for uniform texture hits, 14.2× slower for divergent texture hits, 21.9× slower for divergent T#+S# pairs and 22.6× slower for divergent misses. These are synthetic compute results from one GeForce RTX 3070 on an idle Windows 11 desktop with NVIDIA driver 621.24. The benchmark ran unchanged at `5d74c27` (with the current default `--warmup-ms 300`).

## Measurements

Each cell is gross GPU time in **µs per million logical lane lookups**, including key reads and result accumulation. The benchmark also prints a figure with a key-read baseline subtracted (summarised below). The table uses its gross column so it compares directly with the [RTX 4070 SUPER (Ada) run](descriptor_lookup_ada.md).

| Live descriptors | Memory type | Uniform T# hit | Uniform S# hit | Divergent T# hit | Divergent T#+S# pair | Divergent miss |
|---:|:---|---:|---:|---:|---:|---:|
| 1,000 | GPU-only | 15.0 | 15.1 | 73.0 | 68.2 | 270.3 |
| 1,000 | Host-visible | 16.5 | 16.4 | 71.3 | 69.9 | 267.6 |
| 1,000 | BAR1 (`host_device_local`) | 15.0 | 15.1 | 73.0 | 68.2 | 270.3 |
| 10,000 | GPU-only | 15.5 | 15.0 | 94.3 | 115.9 | 773.3 |
| 10,000 | Host-visible | 16.8 | 17.0 | 104.4 | 134.4 | 880.7* |
| 10,000 | BAR1 (`host_device_local`) | 15.5 | 15.0 | 94.4 | 115.4 | 771.7 |
| 100,000 | GPU-only | 21.7 | 15.7 | 231.9 | 216.6 | 2,502.6 |
| 100,000 | Host-visible | 108.0 | 18.4 | 3,290.5 | 4,734.5 | 56,519.6* |
| 100,000 | BAR1 (`host_device_local`) | 21.7 | 15.8 | 231.8 | 216.7 | 2,499.3 |

\* Shortened by the benchmark's Windows timeout guard in all 12 rounds (see Method).

Net timings (after subtracting the matching key-read baseline) at 100,000 entries were: GPU-only `14.6`, `8.6`, `207.6`, `204.5`, `2,478.3`; BAR1 `14.6`, `8.6`, `207.5`, `204.6`, `2,475.1`; Host-visible `101.0`, `11.5`, `3,265.5`, `4,722.4`, `56,495.1*` µs per million.

At 100,000 entries on type 1 (GPU-only), a divergent T# dispatch took 3.89 ms, a T#+S# dispatch took 7.27 ms and a divergent miss dispatch took 41.99 ms. Each paired iteration resolves two descriptors.

GPU-only is type 1, `DEVICE_LOCAL`, on heap 0 (8,018 MiB). Host-visible is type 3, `HOST_VISIBLE | HOST_COHERENT` without `HOST_CACHED`, on heap 1 (16,345 MiB, not device-local). BAR1 (`host_device_local`, selected by `--memory rebar`) is type 5, `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, on heap 2 (214 MiB). Resizable BAR was **off**: `nvidia-smi` reported a 256 MiB BAR1 aperture (227 MiB used, 29 MiB free at idle), and every allocation on type 5 still succeeded up to the 20.0 MiB table. On this card, keeping the descriptor table out of non-device-local host-visible RAM matters far more at 100,000 entries than on the [Ada PCIe 4.0 + ReBAR card](descriptor_lookup_ada.md) (14.2×–22.6× penalty on divergent traffic here vs 1.45×–2.5× there), while the 214 MiB BAR1 heap remained within 0.5% of GPU-only VRAM in every cell whenever the table fit inside it.

At 100,000 entries on GPU-only and BAR1, the 12 round medians split deterministically by which table size ran first in the process: the four rounds that started with 1,000 entries (rounds 1, 5, 8 and 12) measured 167.4–167.9 µs per million for divergent T# hits and 1,791.5–1,791.9 for divergent misses; the four that started with 10,000 entries (rounds 3, 6, 7 and 10) measured 231.8–232.1 and 2,498.4–2,505.4; and the four that started with 100,000 entries (rounds 2, 4, 9 and 11) measured 232.2–232.9 and 2,581.9–2,591.8. The table medians follow the eight rounds that did not start at 1,000 entries. The cause was not measured.

The tables contain 90% T# texture keys and 10% S# sampler keys. Their allocations were 163,844, 2,621,444 and 20,971,524 bytes (0.156, 2.50 and 20.0 MiB). The capacities were 4,096, 65,536 and 524,288 slots, at 24.4%, 15.3% and 19.1% load. At 100,000 entries the ordinary uniform hit runs covered all 90,000 textures or all 10,000 samplers. Each workgroup shared a key for each iteration. The separate hot-key control (`uniform_hot_tsharp`) reused one texture and measured 12.8–13.1 µs per million gross (5.7–6.8 µs net) across all three memory types and sizes. Divergent misses visited 1,024, 16,384 and 131,072 distinct absent keys across the three sizes. Cache residency and memory traffic were unmeasured.

## Method and replay

Each dispatch used 4,096 groups of 64 lanes and 64 iterations. The RTX 3070 reported 32-lane subgroups (`minSubgroupSize = maxSubgroupSize = 32`). The denominator was 16,777,216 logical lookups per single-key dispatch and 33,554,432 per paired dispatch, counting the texture and sampler separately. Vulkan timestamps bracketed one dispatch. The compute queue advertised 64 valid bits and a 1 ns tick period.

Each cell is the median of 12 process medians, each from 25 timed dispatches after five warmups plus the default `--warmup-ms 300` clock-warmup window (TEST-277). The 11 scenarios ran in the benchmark's fixed order. There were 36 processes: three memory types in each of 12 rounds. Across them there were 29,124 timed dispatches. Every process exited 0, wrote nothing to stderr and reported `timing_status` `ok` on all 1,188 rows. The run used the default Windows Vulkan loader with validation layers off and `DISABLE_RTSS_LAYER=1` and `DISABLE_VULKAN_OBS_CAPTURE=1` set to keep overlay layers out of the run; desktop GPU utilization at idle was 2% (993 MiB VRAM used, P5).

As in the [Ada run](descriptor_lookup_ada.md), this run kept the default `--max-dispatch-ms 120` guard to stay clear of the Windows 2 s GPU timeout (TDR). Only host-visible cells were shortened, all 12 of 12 rounds: at 10,000 entries, `divergent_miss` ran with 13–26 iterations (median dispatch 4.38 ms, range 1.60–6.37 ms); at 100,000 entries, `divergent_miss` ran with 19 iterations in all 12 rounds (median dispatch 281.51 ms, range 276.11–285.21 ms) and `latency_divergent` (outside the summary table; gross median 8,135.0 µs per million) ran with 26 iterations in all 12 rounds (median dispatch 55.45 ms, range 47.57–56.15 ms).

From a Leonida checkout on Windows, with Visual Studio C++ tools, CMake + Ninja, Windows SDK 10.0.26100+ and Vulkan headers (`VULKAN_SDK` or `-DLEONIDA_VULKAN_HEADERS_FETCH=ON`):

```powershell
runtime\build\vk_descriptor_bench.exe --list-devices
runtime\build\vk_descriptor_bench.exe --self-test
$env:DISABLE_RTSS_LAYER = "1"
$env:DISABLE_VULKAN_OBS_CAPTURE = "1"
foreach ($memory in 'device', 'host', 'rebar') {
  runtime\build\vk_descriptor_bench.exe --device 0 --memory $memory --sizes 1000,10000,100000 `
    --groups 4096 --iters 64 --warmup 5 --reps 25 --warmup-ms 300 --csv
}
```

Device 0 was the RTX 3070, the only Vulkan device. For the full 12-round series, let A, B and C denote 1k, 10k and 100k. The rounds used the same size orders and starting memory types as the [M3 Max](descriptor_lookup_apple.md) and [Ada](descriptor_lookup_ada.md) runs: the first six size orders were ABC, CBA, BAC, CAB, ACB and BCA, starting with `device`, `host`, `device`, `device`, `host` and `host` respectively; the next six reversed the sequence of rounds and swapped the starting memory type in each, with `rebar` third in every round.

## System

| | |
|:---|:---|
| GPU | NVIDIA GeForce RTX 3070 (Ampere), 8,192 MiB (discrete) |
| PCI id | `0x10de:0x2488`, subsystem `0x404e1458` |
| Link | PCIe 3.0 x16 (current and maximum reported by `nvidia-smi`; B450 AORUS M) |
| Resizable BAR | Off (BAR1 aperture 256 MiB, `nvidia-smi -q -d MEMORY`; Vulkan heap 2 reports 214 MiB `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT`) |
| Subgroup size | 32 |
| Driver | NVIDIA 621.24 (`nvidia-smi`), Vulkan API 1.4.357 |
| OS | Windows 11 Pro 26H2, build 26300.9974 |
| Vulkan headers | 1.4.363.0 (`-DLEONIDA_VULKAN_HEADERS_FETCH=ON`) |
| Compiler | Visual Studio 18 Professional (MSVC 19.51.36260.0), `RelWithDebInfo` via CMake + Ninja |
