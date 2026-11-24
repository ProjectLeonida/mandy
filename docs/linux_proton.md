# Linux (Proton/Wine) and second-machine test report

This page describes how the runtime is built and booted on Linux through Proton's Wine, what a healthy boot looks
like there, and what was observed on the machines that tested this branch (one Linux, two native Windows). It holds
counts, timings and settings only: no game data, no traces.

Status in one line: on Linux/Proton and on native Windows the retail 1.0.2.1 package launches, shows its launch
movie, first-run pages and landing prompt, runs the story load to its end with the loading slideshow drawing its
artwork, and enters the world phase. A first world frame has not been verified yet on any of the three machines; what
stops it is listed under [Where it stops](#where-it-stops).

## Test machines

| | Linux | Windows A | Windows B |
|---|---|---|---|
| OS | CachyOS (Arch), kernel 7.3 | Windows 11 | Windows 11 |
| Run through | Proton Experimental (Wine 11) | native | native |
| CPU | Ryzen 5 5600X | Core i9-13900K | Ryzen 7 9800X3D |
| GPU, driver | RTX 4070 12 GB, NVIDIA 620.73 | RTX 4070 Ti 12 GB, NVIDIA 630.20 | RTX 5070 |
| RAM | 16 GB (+32 GB swap file) | 64 GB | not reported yet |
| Toolchain | clang-cl 23 + xwin SDK, Ninja, ccache | MSVC 2022, SDK 26100 | - |
| Results here | yes, ~80 boots on 2026-11-24 | yes | joining; none yet |

Windows A reaches every stage three to five times faster than the Linux machine (initialisation ~70 s against
~180 s, story load from ~110 s against ~240-300 s), so it is the better machine for iterating on the world phase;
the Linux machine is the one that shows what 16 GB of RAM allows.

## Building from a checkout

The runtime is a Windows program. On Linux it is cross-built with clang-cl against an
[xwin](https://github.com/Jake-Shadle/xwin) copy of the Windows SDK ([linux.md](linux.md) has the packages and the
one-time `xwin` command):

```sh
scripts/linux-boot/configure-worktree.sh <checkout>      # XWIN_DIR=<sdk>; SDL3_SRC=<an SDL3 source tree> configures offline
cmake --build <checkout>/runtime/build/boot --target gta6rt -j6
```

`configure-worktree.sh` configures a Ninja Release build in `<checkout>/runtime/build/boot` with
`runtime/cmake/clang-cl-xwin.cmake`. `gta6rt_tls_host.dll` is built with the `gta6rt` target and must stay next to the
executable: without it the title gets no module TLS index and exits at once, so `run-boot.sh` refuses to start.

## Booting

`boot.sh` is the plain launcher. `run-boot.sh` wraps it as a `systemd --user` unit.

```sh
export LEONIDA_GAME_DIR=<your copy of the package>     # the folder with game_scarlett_master.exe
export LEONIDA_SHADERS=<work>/shaders/spirv            # the translated library make_game.py built
scripts/linux-boot/boot.sh dry-run                     # imports only
scripts/linux-boot/boot.sh init                        # TLS callbacks and DLL initialization
scripts/linux-boot/boot.sh run 150                     # start the title, stop it after 150 s
scripts/linux-boot/run-boot.sh first runtime/build/boot/gta6rt.exe 150 LEONIDA_ARENA_MB=2048
```

| Variable | Default | Meaning |
|---|---|---|
| `LEONIDA_GAME_DIR` | none, required | The folder of your copy of the package. It is read in place. |
| `LEONIDA_SHADERS` | none, required for `run` | The translated shader library. |
| `LEONIDA_BOOT_DIR` | `~/.local/share/leonida-boot` | Where traces, the overlay (the title's writes), the driver's shader cache, the locks and the runs go. |
| `GTA6RT` | `runtime/build/boot/gta6rt.exe` of the checkout | The loader `boot.sh` starts. `run-boot.sh` takes it as its second argument. |
| `WINE` | Proton Experimental's `files/bin/wine` when Steam has it, else `wine` | The Wine to use. **One Wine per prefix, always.** A Proton prefix links to files inside the Proton installation, and another Wine version that starts in it updates the prefix by writing through those links. On the machine below that overwrote about 1,200 of Proton's own files, and every Proton start failed afterwards; the tool's files then have to be verified or reinstalled through Steam. |
| `WINEPREFIX` | `~/.local/share/leonida-boot-prefix` | A prefix for boots alone. |
| `LEONIDA_BOOT_MIN_FREE_MB` | `6500` | `run-boot.sh` waits up to ten minutes for this much memory before it boots. |

What the scripts set, and why:

- **No network.** The loader runs inside `unshare -rn`, a new user and network namespace that needs no root. The
  title has no network at all, which is the project's rule (nothing online) enforced by the kernel.
- **Wine.** `WINEDLLOVERRIDES="mscoree,mshtml=;winedbg.exe=d"`: a new prefix does not ask to install Mono and Gecko,
  and a process that crashes ends instead of starting `winedbg`. With the Wine installation broken as the table
  above describes, `winedbg` started for a crashing Wine service crashed as well and was started again for itself,
  about ten thousand times, until memory ran out; the loader's own trace already records the title's exceptions.
  Keep the
  boot prefix apart from the one unit tests use. On the machine below, a test's `wineserver` that was still alive
  (started without a display) was joined by the next boot, which then had no window.
- **Backend.** `LEONIDA_D3D12=vulkan` and `LEONIDA_SHADERS` for `run`.
- **NVIDIA's shader disk cache** is moved under `LEONIDA_BOOT_DIR` and its size limit raised
  (`__GL_SHADER_DISK_CACHE_SIZE`, 20 GiB unless you set it), with `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`, so that
  the pipelines one boot compiled are still there for the next.
- **Traces.** `--trace` always and `--trace-d3d12` for `run`; `tools/boot_trace.py` summarises them next to the logs.

Other switches pass through: any `KEY=VALUE` after the seconds of `run-boot.sh` is set in the unit, for example
`LEONIDA_KEYBOARD_PAD=always` ([gameinput.md](gameinput.md)) or `LEONIDA_SKIP_INT3=1` ([boot.md](boot.md)).

## What `run-boot.sh` adds

- The boot runs outside your shell's process tree, with `OOMScoreAdjust=900`: when memory runs out the kernel takes
  the game, not the session that started it.
- It waits for memory headroom and for swap below 70 %, then holds a GPU lock (one boot at a time) and a build lock
  for the whole boot. A build that takes the same lock never competes with the game for memory:
  `flock "$LEONIDA_BOOT_DIR/.build.lock" cmake --build ...`.
- RAM, swap, the unit's own memory and VRAM (`nvidia-smi`, when present) are sampled every 2 s into `resources.tsv`.
- A frame capture is on (`LEONIDA_CAPTURE`, every 4800th presented frame) and so is Wine's error channel.
- A second small unit waits for the boot to end, also when `systemd-oomd` or the kernel killed it, and writes
  `digest.txt`: the verdict (`OK`, `EXITED WITH ERROR`, `KILLED BY OOM`, `KILLED BY SIGNAL`), how long it queued and
  ran, counts from the D3D12 trace, captured frames and how many are not black, memory peaks, kernel events (OOM,
  NVIDIA Xid, segfaults), every distinct error line with its count, and the last lines of the trace. One line per
  boot is appended to `boots.tsv`.

Everything is in `$LEONIDA_BOOT_DIR/runs/<name>/` and `$LEONIDA_BOOT_DIR/traces/`. `run-boot.sh` prints the digest
and exits 0 only for `OK`. **Traces, logs and digests can quote the title's own text. Keep them on the machine that
produced them**: no trace, log or capture goes into a commit, an issue or a pull request.

## What a healthy boot looks like (Linux machine)

| Time | Stage |
|---|---|
| 0-10 s | package splash |
| ~60-90 s | launch movie, then the landing prompt appears while initialisation finishes |
| ~180 s | initialisation done; A is accepted |
| ~240-300 s | story load starts: loading slideshow with its artwork |
| ~540-630 s | story load ends (about 38,000 streamed images registered), world phase begins |

On Windows A the same stages are at ~70 s, ~110 s and ~190 s.

## What this branch changes to get there

Each item was a stop on at least one of the machines. Tests named in the pull request cover them.

- **Presentation:** frames go to the title's own window (they went to a window that was never shown).
- **DirectStorage:** reads pin write-watched destination pages, page-list scatter/gather, BCPack and unshuffle, a
  locally generated predictor file, bounded staging.
- **Story-load allocation stall:** one device allocation per streamed texture exhausted the driver's CPU-visible
  mappings near 16,000 images; small streamed images now share pooled pages.
- **Images on plain device-local memory:** host-copy usage excluded it on NVIDIA and 4.8 GB of images spilled into
  host memory; streamed images are now filled through staging.
- **Deferred backing:** placed and streamed images get private memory at their first write or GPU copy. VRAM at the
  end of the load went from 11.8 GB to ~5 GB on Linux; on Windows A host private memory went from 22-24 GB to 11.9 GB.
- **Enhanced barrier into `LAYOUT_UNDEFINED`:** accepted; it failed `Close`, which the title treats as fatal.
- **Bindless table:** a table rebuild that cannot allocate no longer refuses the whole submission; larger image
  arrays; a hash finalizer for the descriptor table (runtime and translator halves, off until a library carries it).
- **Translator:** loop bounds on every emitted loop (5,128 compute modules have loops), the eight FidelityFX programs
  that used an instruction form the disassembler skips (all 29 translate), buffer bounds modes per view kind.
- **Directly indexed descriptor heaps:** the heap offset is supplied in the user-data dword after the explicit root
  arguments for compute and for VS/PS/GS draws. Mesh stages are not covered yet.
- **Title buffers at raw addresses:** a window outside the arena when title reservations fragment it; opt-in
  persistent device-local copies of streamed buffer views.
- **Native Windows:** write faults on shared pages retried under the tracker lock, one tracker per run of shared
  pages, write-watched title memory, a fault-lock that never waits for the process heap.
- **Diagnostics:** device-loss report with the last dispatches and NVIDIA checkpoints, a probe for the descriptors a
  compute dispatch receives, memory probe, press log. All off by default.

## Where it stops

- **Streamed memory that reaches its destination late or not at all.** The title calls queue slot 31
  (`CopyPageMappingsBatchX`) about 300 times per boot. With #968 this branch maps each batch's pool pages at the
  destination and at the paired CPU view the layer reads (`docs/d3d12_vulkan.md`, section 4.9), so most streamed
  views find readable memory; before it, tens of thousands of them waited for memory that never became readable. Two gaps
  remain: the mappings change when the call is made, not in queue order, and a run with the CPU-view fix still had
  views waiting after 420 s, apparently for ranges no mapping call had reached yet.
- **Culling compute at world start** reads buffers named by descriptors in the directly indexed heap; where the
  memory behind them is not mapped, those reads leave mapped memory and the device is lost (NVIDIA Xid 13/109 on
  Linux).
- **Mesh stages** do not get the heap offset yet.
- **16 GB of RAM:** at the end of the load the process holds ~14.6 GB (RAM plus swap) on the Linux machine after the
  cuts above. zram made this worse (compressed pages still occupy RAM); a swap file on disk is the better choice.
- **Initialisation takes ~180 s under Proton** against ~70 s natively. Not investigated.
- **Under Wine** a thread that takes an access violation while holding the process heap lock can still deadlock
  with xmem's fault handler; the test for it is skipped there.

## Reporting a result

Say the machine (OS, GPU and driver, RAM), the commit, the environment switches that differ from the defaults, and
counts: presents, the last stage reached with its time, the first refusal or device-loss line. Keep logs and traces
on the machine that produced them.
