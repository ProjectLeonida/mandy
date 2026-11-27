# Selective reverse engineering with REA

How to use an AI-driven decompiler on the local package to answer the interface questions that block the retail
boot, without turning this into a full decompilation project. Everything here runs against the player's own copy;
nothing measured is committed except distilled contracts, counts and constants, like the rest of `docs/`.

The rule is the same as [boot_leads.md](boot_leads.md): **isolate one question, answer it, write the contract down,
then implement and test it.** An unverified hypothesis is labelled as one.

Status: the first target, the `XboxBvhBuilder.dll` contract, is measured in [xdxr.md](xdxr.md) §6 and §7.

## Why selective

The executable is a static recompilation target, not a source base: its own code already runs natively on the host.
What fails is the boundary — console system software (XMem, D3D12Xbox, XDXR/BVH, DirectStorage, audio DSP, GameInput)
and the few invariants the title assumes about it. So the useful questions are almost always "what does this call
site expect back", not "what does the game do". That keeps each investigation to one function, one export or one
import, and keeps the corpus out of the loop.

## Installed toolchain

Measured on 2026-11-25, Windows 11 Pro 26300, Ryzen 7 9700X, 15.1 GB RAM, RTX 3080 Ti (no Ghidra involvement in
the GPU path).

| Piece | Version | Location |
|---|---|---|
| REA (`rea-agents`, MCP server + CLI) | 6.10.1 | `C:\tools\rea\node_modules\rea-agents` |
| Node for REA only (system Node stays 24.14) | 26.13.1 | `C:\tools\node26\node.exe` |
| Ghidra (Windows x64 provider) | 12.2 (20261016) | `C:\tools\ghidra_12.2_PUBLIC` |
| JDK for Ghidra only (system JDK 21 untouched) | Temurin 25.0.5.1+2 | `C:\tools\jdk-25.0.5.1+2` |
| MCP registration | `mcp.rea` in `~/.config/opencode/opencode.jsonc` | 147 tools, families `direct,enhanced,native,artifact,session` |
| Skill | `~/.agents/skills/reverse-engineer-anything` | auto-loaded by opencode |

Registration environment (already in the opencode config): `REA_ANALYSIS_PROVIDER=ghidra`,
`GHIDRA_INSTALL_DIR`, `JAVA_HOME`, `GHIDRA_HEADLESS_MAXMEM=8G`, `REA_GHIDRA_STARTUP_TIMEOUT_MS=900000`,
`REA_GHIDRA_DISABLED_ANALYZERS=WindowsResourceReference`, and `experimental.mcp_timeout=900000` so a cold import
does not time out. Changes to that file need an opencode restart.

CLI equivalents:

```powershell
$env:GHIDRA_INSTALL_DIR='C:\tools\ghidra_12.2_PUBLIC'
$env:JAVA_HOME='C:\tools\jdk-25.0.5.1+2'
$env:REA_ANALYSIS_PROVIDER='ghidra'
$rea='C:\tools\rea\node_modules\rea-agents\scripts\rea.mjs'
& 'C:\tools\node26\node.exe' $rea doctor --json
& 'C:\tools\node26\node.exe' $rea decompile <path> --procedure '<mangled export>' --format json
```

Measured cold costs on this host (ephemeral project, one JVM per process):

| Target | Size | Operation | Time |
|---|---:|---|---:|
| `amd_fidelityfx_upscaler_xbox.dll` | 0.75 MB | `inspect` (Ghidra import + inventory: 401 procedures, 399 strings) | 100.6 s |
| `XboxBvhBuilder.dll` | 0.84 MB | `decompile --procedure ?Set64BitMortonCodes@...` | 19.5 s |

Two constraints to design around:

- **Ephemeral projects.** Windows P0 cannot open an existing Ghidra project, so every new MCP session (i.e. every
  opencode restart) re-imports. Batch all questions for one binary into one session; do not restart between them.
- **64 MiB default read budget.** Static PE tools refuse bigger files until `--max-file-bytes` is raised; the MCP
  tools take the same limit per call. The retail executable is 114.6 MiB.

## Decision ladder: cheapest evidence first

1. **Docs and corpus first.** `docs/design.md`, `runtime_imports.md`, `xmem_exe_calls.md`,
   `audio_hal_exe_calls.md`, `d3d12_xs_console_abi.md`, `xdxr.md`, `bvh.md`, `fspx.md`. Several "questions" are
   already answered with offsets; re-measuring wastes a claim.
2. **Static, no Ghidra.** REA built-ins (`inspect-pe-imports`, `inspect-pe-load-config`, `inspect-binary-hardening`,
   `inspect-pe-resources`, `inspect-gpu-code-objects`) and the in-repo tools `tools/import_call_sites.py`,
   `tools/pe_inspect.py`, `tools/recomp_census.py`. Seconds, and the executable is admitted without a decompiler.
3. **Ghidra on a shipped DLL.** The package's small DLLs are the console software itself and carry their exports:
   `XboxBvhBuilder.dll`, `amd_fidelityfx_upscaler_xbox.dll`, `amd_fidelityfx_loader_xbox.dll`,
   `Bink2Scarlett.dll`, `libHttpClient.GDK.dll`, `XCurl.dll`, `xapu.dll`, `AcpHal.dll`, `chad.dll`, `xg_xs.dll`,
   `xmem.dll`. Tens of seconds each, exact symbols.
4. **Ghidra on the executable.** Last resort, batched: 120,181,760 bytes, `.text` 99.9 MB raw, `.pdata` 3.59 MB
   (about 299k `RUNTIME_FUNCTION` entries), image base `0x140000000`, entry `0x56A1EC8`, SHA-256
   `e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993`. Full analysis is **not** in scope. Either
   decompile by file offset after a bounded import, or fall back to the custom Ghidra lane below.

### The executable lane (only when a blocker needs game-side code)

- First measure one import with analyzers restricted and a hard wall clock (30 min). If the Ghidra import cannot
  finish inside 8 GB of heap, stop and use the fallback; do not raise the heap past about 10 GB on a 15 GB host
  with the game running.
- `.pdata` is the function census. A helper that reads the exception directory and decompiles *only* the requested
  entry is the cheap path: Ghidra headless `-noanalysis` plus a script that creates one function at the requested
  address, disassembles it, decompiles it and prints C. Keep that script under `tools/` if it is written; it needs
  no game data to test (synthetic PE with a `.pdata` entry).
- REA's own read-only P0 boundary is fine for the DLLs; the custom script exists because REA was not built for a
  299k-function image.

## Targets, in playability order

Each target names the question, the cheapest sufficient method, what "answered" means, and where the result is
recorded. Claim status must be re-checked in the coordination issue and against open PRs before starting; the
`main` tree moves.

### T-1: the world-start device loss (current stop)

- **Question:** what does the title expect from the compute/host pair that loses the device at world start? The
  stop is `lib_cs_03426/03424/03418/03422` (slot 133, `D16S8`), `VK_ERROR_DEVICE_LOST` out of `vkDeviceWaitIdle`;
  the title also stops at `+0x23cf683` after the same run.
- **Method:** boot trace + `--trace-d3d12` first (`scripts/dev.py boot-trace`); the missing contract is on the shim
  side, so read the call sites the trace names with `tools/import_call_sites.py` before any decompilation. Only if
  a *game-side* value is still unexplained, decompile that one function by offset.
- **Answered when:** the resource/barrier/queue invariant the title assumes is written as a contract and a
  synthetic test reproduces the failure without the game.
- **Record in:** `docs/boot.md` and `docs/d3d12_vulkan.md`. **Needs:** game files; a GPU.

### T-2: `XboxBvhBuilder.dll` semantics (XDXR) — contract done

- **Question:** exact behaviour of `BuildRaytracingAccelerationStructureBatched2X`,
  `RehydrateRaytracingAccelerationStructureBatched2X`, `GetRaytracingAccelerationStructurePrebuildInfo`,
  `CopyRaytracingAccelerationStructure`, `Set64BitMortonCodes` — what the title gets back and what the stream
  descriptors mean.
- **Method:** ladder step 3 (0.84 MB, exports decompile in about 20 s) plus the executable's call sites.
- **Answered:** the measured argument table and the DLL's rehydrate validation are in [xdxr.md](xdxr.md) §6 and §7;
  the open rehydrate implementation is #1759, and serialization/deserialization is #626.
- **Record in:** `docs/xdxr.md`.

### T-3: FidelityFX upscaler contract

- **Question:** the upscaler crashed at `amd_fidelityfx_upscaler_xbox.dll+0x65ff` before #1862 translated the two
  missing FSPX shaders; the remaining question is the `ffxCreateContext`/`ffxQuery`/`ffxDispatch` descriptor and the
  FSPX kinds the DLL expects, so the failure is classified instead of merely gone.
- **Method:** ladder step 3 on the loader and upscaler DLLs (0.01 MB / 0.75 MB), plus `docs/fspx.md`. Map the crash
  offset to the export and record the contract, not the body.
- **Answered when:** the descriptor layout, the queried capabilities and the dispatch kind/slot mapping are written
  down and a synthetic dispatch test covers them.
- **Record in:** `docs/fspx.md`. **Needs:** the shipped DLLs; a GPU for the end-to-end run.

### T-4: D3D12Xbox ABI gaps found by crashes

- **Question:** the console ABI is documented in `docs/d3d12_xs_console_abi.md`; each new stop (slot 133 `D16S8`,
  the `Close` failure after #1759, the PSO `E_INVALIDARG` class) names a slot whose contract is still thin.
- **Method:** trace first; for each slot, one decompile of the calling wrapper in a shipped DLL if the slot is
  driven from there, otherwise a file-offset decompile in the executable. Add only the missing slots.
- **Answered when:** the slot's console-only fields are named from observed arguments, with a mock test.
- **Record in:** `docs/d3d12_xs_console_abi.md`. **Needs:** game files; GPU for validation.

### T-5: audio bring-up contracts (menu and in-game sound)

- **Question:** `xapu.dll` (`XApuConnect/EnqueueCommand/DequeueResult`), `AcpHal.dll` (`Apu*`, `AcpHal*`) and
  `chad.dll` (`XDsp*`) define the console audio DSP boundary; the shim surface exists but the buffer/command
  contract is not recorded as measured.
- **Method:** ladder step 3 on the three small DLLs; `docs/audio_hal_exe_calls.md` already has the executable's call
  constants, so start there and only decompile what it leaves open.
- **Answered when:** command ring, result semantics and timing expectations are contracts with synthetic tests.
- **Record in:** `docs/audio.md`. **Needs:** the shipped DLLs; audio hardware for verification.

### T-6: XMem page mapping (RT-5)

- **Question:** seven of the twelve `xmem.dll` imports have no shim (`XMemMapPhysicalPages`,
  `XMemAllocatePhysicalPages`, `XMemFreePhysicalPages`, `XMemMapPhysicalPagesScatter`, ...).
  `docs/xmem_exe_calls.md` records the call sites and constant arguments; the open part is the page/owner contract
  behind them.
- **Method:** `tools/import_call_sites.py` for callers, then ladder step 2/3 on `xmem.dll` for the twelve entry
  contracts. Check the coordination issue first: these are tracked work items, so this is a contribution to the
  existing claim, not a new PR.
- **Answered when:** the shim exists with the measured refusal/acceptance rules and a synthetic allocator test.
- **Record in:** `docs/xmem_exe_calls.md`. **Needs:** the shipped DLL; no GPU.

### T-7: the fail-fast stop at `+0x23cf683`

- **Question:** why the title takes its own fail-fast path after the world-start loss.
- **Method:** only after T-1. One file-offset decompile of that function (executable lane), plus a search of its
  string/caller references. If T-1 removes the stop, close this as stale.
- **Answered when:** the condition is named and mapped to a shim contract, or the stop is shown to be downstream.
- **Record in:** `docs/exe_fail_fast.md`. **Needs:** game files.

## Isolation and evidence hygiene

- Work directory outside the repo: `F:\leonida-work\re\<yyyy-mm-dd>-<target>\` for REA evidence JSON, decompiled
  text, traces and screenshots. Nothing from there is committed.
- Commit only distilled facts: a contract table, constants, counts, target SHA-256. Never pseudocode bodies,
  decompiled source, strings dumps or anything `scripts/check_no_game_data.py` rejects. Existing pages
  (`xmem_exe_calls.md`) are the template: call counts and the constants a site passes, no code.
- One target per Ghidra session; if the session must restart, the import cost is paid again.
- Reproduce before decompiling: a boot run with `--trace-d3d12`, `LEONIDA_PEEK`, `LEONIDA_DEVICE_FAULT` (see
  `scripts/windows-boot/boot.ps1`) is cheaper than reading code and tells you which function matters.
- An answered question becomes a shim change **with a synthetic test** and a boot A/B delta, or it stays a finding
  posted in the coordination issue as `finding: <one line>`.

## Non-goals

- Decompiling the executable wholesale, renaming/annotating a 299k-function database, or building a symbol map.
- Re-translating shaders (the FSPX gap is #1862's job, already translated 29/29 locally).
- Comparing against the `leonida-ultimate` fork; the decision is to treat this repository as the reference.
- Anything that touches keys, decryption or protection.

## Next targets

1. T-3 classification of the upscaler entry points (0.75 MB) and a note in `fspx.md`.
2. T-1 trace-first pass on the world-start loss, then decide whether the executable lane is needed at all.
3. Re-check the rehydrate finding against #1759 once it changes.
