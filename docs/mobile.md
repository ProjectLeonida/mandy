# Mobile / arm64 port notes

The PC port works because the Xbox Series CPU is x86-64, the same ISA a PC runs; `runtime/` maps the
PE image and executes it natively. On Apple silicon that stops working for exactly one reason: the
guest is x86-64 and the host is arm64. Everything else (PE loading, import shims, D3D12→graphics,
assets) is a port of layers that already exist. This note records the design and the current state
of the x86-64 → arm64 groundwork.

## Route

- **Static/AOT recompilation** is the only clean iOS path: lift the exe's `.text` to arm64 ahead of
  time and ship a plain arm64 binary — no JIT entitlement, no sandbox fights. `.pdata` gives exact
  function boundaries (the same table `RtlAddFunctionTable` uses for SEH on the PC side), so the
  recompiler can enumerate code without guessing; `.xdata` unwinding has to be re-emitted as arm64
  unwind info or handled in the translated prologue/epilogue, which is the genuinely hard part.
- **Dynamic translation (FEX/Box64-style)** is the fallback on Android, where JIT and Vulkan are
  unrestricted; on iOS it needs `com.apple.developer.cs.jit` or workarounds and is not the plan.
- **arm64 macOS first, iOS second.** Same ISA, same Metal, same Mach kernel, but no entitlement
  fights and debuggable. If "mobile" matters more than "iOS", Android is dramatically easier:
  native Vulkan (the whole GPU layer ports) plus unrestricted JIT.
- Graphics on Apple: **MoltenVK** over the existing Vulkan backend for a first pixel (no
  conservative raster / geometry shaders in Metal — needs fallbacks), or **SPIRV-Cross → MSL**
  for a native Metal backend longer term.

## What exists today

| Piece | Where | State |
|---|---|---|
| x86-64 decoder | `tools/x64_recomp.py` `disasm`/`decode_one` | prefixes/REX, ModRM+SIB, RIP-relative and moffs addressing, hi8 regs (`ah`/`ch`/`dh`/`bh`), `Iv`/`Iz`/`J` immediates incl. imm64, groups 1–5/8/15/1A/11, VEX stubs |
| arm64 assembler | `tools/x64_recomp.py` `Asm` | integer/logical/bitfield/shift/multiply/load-store/branch/system encoders, label fixups, `li()` with `pad4` for self-referential constant materialization |
| Translator | `tools/x64_recomp.py` `FunctionTranslator` | linear-sweep over decoded insns; lazy flag analysis picks arm-NZCV when one producer covers a consumer, else materializes canonical x86 EFLAGS in the ctx block; a shift or rotate by `cl` may leave the flags alone (x86 changes none for a count of 0), so a consumer behind one reads the ctx word and the writer before it stores its word too; unsupported insns emit `brk` and land in `stats["unhandled"]` |
| Dispatch runtime | `tools/x64_recomp.py` `emit_rt` | `rt_dispatch` (guest pc → host pc through a generated table), `rt_mkflags` (NZCV+operands → canonical x86 EFLAGS; preserves NZCV), `rt_print` (semihosting result output) |
| Verification | `tools/x64_recomp.py` `build_test_kernel`/`run_kernel` | flat kernel image → `qemu-system-aarch64 -M virt -cpu cortex-a53 -kernel -semihosting`; guest `ret` off the sentinel stack exits and prints `R=xxxxxxxx` (guest edi) |
| PE census | `tools/recomp_census.py` | walks every `.pdata` function, decodes + translates it, reports coverage %, hazard mnemonics, indirect control flow, unwind-info flags |
| Tests | `tests/test_x64_recomp.py`, `tests/test_recomp_census.py` | hand-built byte fixtures only; exec tests skip when qemu is absent |

### Translator contract (as implemented)

- Guest regs `rax`–`r15` → host `x19`–`x27`, `x9`–`x15` (`GMAP`); `x28` = guest ctx pointer,
  `x16` = dispatch target, `x17`/`x30` scratch/link, `x0`–`x3` temps.
- Ctx block: `+0x00` canonical x86 EFLAGS dword, `+0x08` last result, `+0x10`/`+0x18` last
  operands (for AF/PF reconstruction), `+0x38` the semihosting exit block in test images.
- Flag model: arithmetic runs on arm NZCV directly; a conditional reads arm NZCV only when one
  flag-writing producer covers every flag it needs (`_flag_analysis`, `_producer_ok`); otherwise
  producers materialize x86 EFLAGS through `rt_mkflags` and consumers read the ctx word.
  Sub-kind producers carry inverted C (arm borrow ≠ x86 borrow); `popf`/`sahf` reconstruct
  NZCV from the x86 word; `rt_mkflags` restores NZCV before returning so calls can't clobber it.
- Control flow: direct branches/calls become arm `b`; `ret`, `jmp/call reg` and unknown targets
  load the guest pc into `x16` and go through `rt_dispatch`; guest pc 0 is the exit sentinel.

## Status and honest limits

- Hand-written fixtures run end-to-end under qemu: arithmetic, loops, taken/not-taken signed and
  unsigned branches, `setcc`/`cmov`, `adc`/`sbb` carry chains, shifts with CF, `movsx`, high-byte
  regs, `call`/`ret`/`leave`/`pushf`/`popf`, `rep movsb`, `repne scasb`, moffs/LEA memory
  addressing, and indirect jumps through the pc table — all hand-assembled, no game bytes.
- Not covered yet (each is an `unhandled` entry at translate time, not silent wrongness): SIMD/SSE
  beyond decode, x87, `rcl`/`rcr`, `std` (DF=1), system/privileged insns, `int`/`syscall`, far
  transfers. Real binaries are x87/SIMD-heavy, so coverage on the engine image will be far below
  the fixture suite until those lands — `recomp_census.py` is the tool that will measure it.
- The engine image itself is still not in the repository, but `recomp_census.py` has now been run
  against the local `1.0.2.1` retail executable (`game_scarlett_master.exe`, 120,181,760 bytes) —
  see [Retail executable census (1.0.2.1)](#retail-executable-census-1021) below. Next milestone is a
  same-behaviour run of a real function under qemu against a native x86-64 run.

## Next steps

1. ~~`recomp_census.py` on the real image: instruction histogram, coverage %, hazard list.~~ Done on
   `1.0.2.1` (84.77% instruction coverage across 299,257 `.pdata` functions; VEX/SSE/AVX ops dominate
   the remaining 15.23%).
2. SIMD/SSE/AVX emitters (the largest hazard class on the retail binary), then `bt`, `cmpxchg`, and
   `std`/RCL-RCR.
3. SEH story: translate `.xdata` prologue ops or run translated code inside a frame the host
   unwinder can walk — needed before C++ `throw` works.
4. arm64 userland runner (Mach-O + `mmap` JIT-free) to replace the qemu harness on Apple silicon.
5. Vulkan backend under MoltenVK on macOS as the first graphics target; Metal backend later.

## Retail executable census (1.0.2.1)

Measured on `game_scarlett_master.exe` (Xbox Series `1.0.2.1`, 120,181,760 bytes) with
`python -I tools/recomp_census.py <exe> --out <dir>` at `main` `bcaa13dad` with PR
[#1121](https://github.com/ubiquitarianism/leonida/pull/1121) (`old-pr/9855`, which fixes the truncated
VEX/EVEX `IndexError` and `vzeroupper` desync that otherwise abort the census on `main`). Aggregate
counts only; no game bytes or code.

| Metric | Measured |
|---|---|
| `.pdata` functions | 299,257 (92,353,193 code bytes) |
| Decoded instructions | 21,518,571 |
| Translated instructions | 84.77% (151,107 functions fully covered; 148,150 with gaps) |
| Decode errors | 41,091 (41,006 `decode:op_90`) |
| Branch instructions | 2,446,825 |
| Indirect control flow | `call` 105,179; `jmp` 8,018; `ret` 232,851 |
| Unwind info flags (`main` `.xdata` reader) | `none` 298,146; `FHANDLER` 515; `EHANDLER,UHANDLER` 496; `UHANDLER` 97; `EHANDLER` 3 |

Top unhandled / hazard mnemonics (VEX-prefixed SIMD/AVX `v*` opcodes dominate):

- **VEX/SIMD (`v*`):** `v28` (367,994), `v29` (317,513), `v10` (268,958), `v59` (232,529),
  `v11` (220,760), `v18` (175,501), `v57` (138,025), `vc6` (97,178), `v7c` (79,604),
  `v2e` (75,814), `v0c` (74,738), `v58` (71,159), `vc2` (70,707), `v5c` (59,246),
  `v77` (48,224), `va8` (46,138), `vb8` (44,516), `v4a` (41,775).
- **Non-VEX:** `jcc` (41,099), `decode:op_90` (41,006), `bt` (18,009), `cmpxchg` (17,878).

Note on `decode:op_90`: `0x90` (bare `nop`) is missing from `PRIMARY` on `main` (the `xchg` loop starts
at `0x91`), and `recomp_census.py` stops decoding a function at its first decode error, so decoded
instruction counts and coverage undercount functions that hit `0x90`.
