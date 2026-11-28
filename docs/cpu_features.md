# CPU instructions the executable needs

The Xbox Series CPU is an AMD Zen 2, and the runtime runs the game's code natively
([design.md](design.md), "Why a loader before a recompiler"). Code built for one known CPU can use every instruction
that CPU has without asking CPUID first. An instruction the host lacks raises `#UD`
(`STATUS_ILLEGAL_INSTRUCTION`) where it is reached, and a privileged one faults in user mode. This page records
which such instructions `game_scarlett_master.exe` contains, measured with `tools/cpu_feature_scan.py`. Counts only;
nothing from the executable is quoted.

## Reproduce

```powershell
python -I tools\cpu_feature_scan.py <game>\game_scarlett_master.exe            # summary
python -I tools\cpu_feature_scan.py <game>\game_scarlett_master.exe --sites    # plus the RVA of each flagged site
```

It streams MSVC `dumpbin /disasm:nobytes` (a few minutes for this executable, with no listing written to disk; a saved
listing works too, with `--image <exe>` for `.pdata`). Each instruction is mapped to the CPUID feature or privilege class it
needs, and each count is split by whether the site lies inside a `.pdata` function range. A linear disassembly also
decodes data in `.text` (jump tables, padding), so a hit outside every `.pdata` range is often not code; a leaf
function that does not touch the stack has no `.pdata` entry either, so those sites need a look by hand.

## Measured result (1.0.2.1)

26,674,075 instructions decoded in `.text`, 22,435,554 of them inside the 299,257 `.pdata` ranges.

| Feature | Class | Sites | Inside `.pdata` | What a host without it does |
|---|---|---:|---:|---|
| SSE4a (`insertq`) | AMD only | 19 | 17 | `#UD` on every Intel CPU |
| CLZERO | AMD only | 10 | 0 | `#UD` on every Intel CPU |
| MONITORX (`monitorx`, `mwaitx`) | AMD only | 4 | 4 | `#UD` on every Intel CPU |
| CLWB | newer Intel only | 8 | 6 | `#UD` on Intel client CPUs before Ice Lake / Tiger Lake (Skylake to Comet Lake) |
| Port I/O, `cli`/`hlt` | privileged | 8 | 2 | fault in user mode |
| `int n` (not 3/29h/2Ch/2Dh) | kernel call | 1 | 1 | a software interrupt Windows does not serve |
| CPUID / XGETBV | host query | 9 / 2 | 9 / 2 | run everywhere; answer with the host's values |

Each flagged site was read by hand in a private copy (not quoted here):

- All 41 SSE4a, CLZERO, MONITORX and CLWB sites are real code: each sits among instructions of the same kind of
  code (vector arithmetic around the `insertq` sites, cache-line loops around CLZERO and CLWB, a wait on a memory
  address around the two `monitorx`/`mwaitx` pairs). The 10 CLZERO sites are one leaf routine, a cache-line zeroing
  loop with no `.pdata` entry, which is why none of them is inside a `.pdata` range.
- The 8 privileged and 1 `int n` hits are data: tables of 32-bit offsets placed in `.text` (some inside a `.pdata`
  range) that the linear sweep decodes as instructions. They are listed for completeness, not as code to replace.

No AVX-512, VAES, VPCLMULQDQ, SHA, RDPID, RDPRU, XOP, FMA4, TBM, SYSCALL or SYSENTER instruction was found. No RDTSC,
RDTSCP, RDRAND, RDSEED, FSGSBASE or CLFLUSHOPT instruction is in the image either. The remaining features counted by
`tools/cpu_feature_scan.py` (at `0b11d86b`) are in its `baseline` class — present on every Intel CPU since
Haswell/Broadwell and every AMD CPU since Zen:

| Feature | Sites | Inside `.pdata` | Outside `.pdata` | Notes |
|---|---:|---:|---:|---|
| `AVX` | 2,981,637 | 2,871,070 | 110,567 | Includes 31,970 `vbroadcastss` and 131 `vbroadcastsd` sites in `.pdata` with an `xmm` register source (`mod == 3`), which are AVX2 rather than AVX1 |
| `FMA` (FMA3) | 204,279 | 200,924 | 3,355 | 45 mnemonics (`132`/`213`/`231`, `ps`/`pd`/`ss`/`sd`, including 3 `vfmaddsub*`/`vfmsubadd*` sites) |
| `AVX2` | 86,018 | 81,500 | 4,518 | Zero `vgather*` or `vpgather*` sites; 284 of the `.pdata` hits are AVX1 mnemonics that share the `vp` prefix (`234` `vptest`, `42` `vperm2f128`, `8` `vpermilps`) |
| `BMI1` | 60,850 | 58,398 | 2,452 | `bextr` (21,546), `andn` (17,585), `tzcnt` (11,827), `blsr` (7,301), `blsi` (139); zero `blsmsk` |
| `BMI2` | 14,959 | 12,403 | 2,556 | `shlx` (7,581), `mulx` (1,612), `shrx` (1,309), `rorx` (1,064), `bzhi` (808), `sarx` (29); zero `pdep` or `pext` in `.pdata` |
| `AES` | 3,390 | 13 | 3,377 | `vaesimc` in `.pdata`; the hits outside `.pdata` are data |
| `F16C` | 3,303 | 2,966 | 337 | `vcvtph2ps` (1,526) and `vcvtps2ph` (1,440) |
| `ADX` | 2,678 | 0 | 2,678 | Data outside `.pdata`; zero sites inside `.pdata` |
| `PCLMULQDQ` | 1,128 | 0 | 1,128 | Data outside `.pdata`; zero sites inside `.pdata` |
| `POPCNT` | 458 | 442 | 16 | — |
| `MOVBE` | 422 | 195 | 227 | — |
| `LZCNT` | 321 | 294 | 27 | 149 4-byte and 145 5-to-9-byte encodings in `.pdata` |
| `CX16` | 56 | 56 | 0 | `cmpxchg16b` |
| `PREFETCHW` | 5 | 5 | 0 | `0F 0D /1`: architectural NOP on Intel 64 CPUs without `PREFETCHW` |

## What it means for the runtime

- On an AMD Zen 2 or later host every flagged user-mode instruction runs as on the console.
- On an Intel host the SSE4a, CLZERO and MONITORX sites, and on Intel client CPUs before Ice Lake the CLWB sites,
  raise `STATUS_ILLEGAL_INSTRUCTION` when reached. `gta6rt` emulates them (`runtime/src/cpu_emulation.h`, see
  [runtime/README.md](../runtime/README.md), "PE loader"): CLZERO zeroes the 64-byte line, CLWB is a write-back hint
  with nothing to do on coherent memory, MWAITX returns at once (an early return is one the manual allows), and
  `insertq`/`extrq` are bit-field operations on the low quadword. `cpu_emulation_test` checks each form against the
  CPU on a host that has it and against the handler on one that does not (an i7-10700K: all four lacking, all checks
  pass).
- The CLZERO routine is too hot for the #UD path: one exception and two `VirtualQuery` calls per 64-byte line. On an
  i9-13900K the routine held one thread at ~99% for the whole boot (sampled: `VirtualQuery` under
  `cpu_emulation::apply`) and the title presented ~3 frames a second. On a CPU without CLZERO the loader therefore
  rewrites the routine's loops before any image code runs (`runtime/src/clzero_patch.h`): each of the two loop shapes
  (6 + 4 sites) jumps to a stub within 2 GiB of the image that zeroes the line with eight 8-byte stores and keeps the
  loop's registers and flags. Same boot afterwards: 164,946 PresentX in 300 s against 869 in 420 s, no CLZERO left on
  the #UD path. `LEONIDA_CLZERO_PATCH=0` keeps the emulation; `clzero_patch_test` runs both shapes on any CPU.
- The CPUID sites see the host's vendor, family and features, not the console's. What the game does with the
  answers is not measured here.

Not measured: which of these sites a boot reaches, and in which order.

### Hosts with AVX1 and SSE4.2 (Sandy Bridge and Ivy Bridge)

An Intel Sandy Bridge or Ivy Bridge host (for example an Intel Core i7-2600, `CPUID.1:ECX = 0x1FBAE3FF`, `XCR0 = 0x7`)
has 256-bit `YMM0`–`YMM15`, `XSAVE`/`OSXSAVE`, `AVX`, `SSE4.2`, `POPCNT`, `CX16`, `AES-NI` and `PCLMULQDQ`, so it
passes the `host_cpu` AVX preflight, executes the 2,839,253 genuine AVX1 `.pdata` sites (`2,871,070` minus the 32,101
register-source `vbroadcastss`/`vbroadcastsd` sites, plus the 284 `vp`-prefixed AVX1 sites) natively, and treats the 5
`PREFETCHW` sites (`0F 0D /1`) as hint NOPs (Intel SDM Vol. 2B). What it lacks inside `.pdata` is the Haswell/ABM
delta: `FMA` (200,924), `AVX2` (81,216 plus 32,101 register-source `vbroadcastss`/`vbroadcastsd`), `BMI1` (58,398),
`BMI2` (12,403), `F16C` (2,966 on Sandy Bridge; native on Ivy Bridge), `LZCNT` (294) and `MOVBE` (195).

Those instructions split into two execution classes on a pre-Haswell CPU:

- **Fault with `#UD` (`STATUS_ILLEGAL_INSTRUCTION`):** every VEX `0F38`/`0F3A` instruction in that set (`FMA3`, `F16C`,
  `bextr`/`andn`/`blsr`/`blsi`, `BMI2`, `AVX2` broadcasts/permutes/blends/variable shifts, and register-source
  `vbroadcastss`/`vbroadcastsd`), 256-bit `0F`-map integer `vp* ymm` operations, and `MOVBE`.
- **Do not fault (`LZCNT` and `TZCNT`):** legacy `LZCNT` (`F3 0F BD`) and `TZCNT` (`F3 0F BC`) have no VEX form in
  x86-64, and a CPU without `LZCNT`/`BMI1` ignores the `F3` prefix and executes `BSR` or `BSF` instead (Intel SDM
  Vol. 2A/2B). `BSR` returns the bit index (`width - 1 - lzcnt`) rather than the leading-zero count and leaves the
  destination undefined when the source is zero. `BSF` writes the same bit index as `TZCNT` for non-zero inputs, but
  leaves `CF` undefined and leaves the destination unmodified when the source is zero, whereas 988 of the 11,827
  `.pdata` `tzcnt` sites are immediately followed by a `CF`-reading instruction (`793` `cmovb`, `174` `cmovae`, `20`
  `jae`, `1` `jb`). Supporting an `AVX1` host therefore requires rewriting `LZCNT` and `TZCNT` sites before code runs
  (149 of the 294 `.pdata` `lzcnt` sites are 4-byte `F3 0F BD /r` encodings without REX, shorter than a 5-byte
  `jmp rel32`) rather than relying on `#UD`.

### Hosts without AVX

Measured on one host: Intel Xeon X5675 (Westmere, SSE4.2; no AVX, AVX2, FMA, BMI2 or XSAVE), Fedora 44,
`wine-11.21 (Staging)`, Intel SDE 10.13.1, and a local `gta6rt.exe` build from 2026-11-24.

- **What works without AVX:** building the native Linux runtime and Vulkan test tools (`ctest`), running the Vulkan GPU
  suite (`scripts/run_gpu_tests.py`), shader translation and asset export (`make_game.py` stages).
- **Running `gta6rt` directly:** `--init` and `--run` exit 4 at the AVX preflight. With `LEONIDA_SKIP_CPU_CHECK=1`,
  the preflight is skipped and the guest faults on its first AVX instruction.
- **Under Intel SDE (`sde64 -hsw`):** without `-chip_check_disable 1`, SDE aborts on `clzero` instead of raising
  `SIGILL` for `gta6rt`'s handler. With `-chip_check_disable 1`, the guest passes its first AVX instruction and reaches
  `InitializeApiImplEx2` (offline) after ~110 s, then stops before any window or D3D12 call: SDE traps an `int 3`
  without `-raise-int3`, and with `-raise-int3` the process terminates with `SIGSEGV` (`LEONIDA_SKIP_INT3=1` with
  `--trace` did not change this, recording `0` `int3_skipped` events). Adding `-ptr-raise` causes Pin to abort with an
  internal instrumentation error under Wine's exception handling.
- **Outcome on this host:** no useful path to boot `gta6rt --run` was found under SDE + Wine.

Not measured: other emulators (such as QEMU TCG with AVX2) or Rosetta 2.
