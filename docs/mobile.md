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

### Apple Silicon userland runner (Mach-O + `mmap`, JIT-free)

`tools/x64_recomp.py` also builds and executes native arm64 Mach-O binaries on Apple Silicon macOS
(`build_test_macho`, `run_macho`, `run_macho_result`, and `pack_macho_arm64`), replacing the QEMU kernel
harness on arm64 Darwin with no JIT entitlement, no external compiler, and no `codesign` dependency:

- **Signed `__TEXT` without JIT:** `pack_macho_arm64` wraps the PIE arm64 image in a 64-bit `MH_EXECUTE`
  (`MH_PIE`, 4 GiB `__PAGEZERO`, 16 KiB-aligned `__TEXT`/`__LINKEDIT`, `/usr/lib/dyld`,
  `/usr/lib/libSystem.B.dylib`, `LC_BUILD_VERSION`, `LC_MAIN`) with a pure-Python ad-hoc SHA-256
  `LC_CODE_SIGNATURE` (`CodeDirectory` v`0x20400`, `CS_ADHOC | CS_LINKER_SIGNED`, 4 KiB code-signing pages).
- **PIE-safe dispatch:** under `emit_rt(..., darwin=True)`, `rt_lookup` locates `pc_table` with `adr` and
  adds each entry's signed 64-bit relative offset, so `call`/`ret` and indirect jumps work under any ASLR slide.
- **Guest memory via `mmap`:** the startup stub maps `DARWIN_RW` (`0x10047000000`, 2 MiB; above macOS arm64's
  `__TEXT`/stack/dyld-cache/guard/malloc regions) with `MAP_PRIVATE | MAP_ANON` (no `MAP_FIXED`, failing with
  exit code 77 if the address is unavailable), zeroes guest GPRs and `NZCV`, and optionally maps the input
  bytes read-only at `guest_va` (`map_code=True` / `--map-code`) so RIP-relative reads execute natively.
- **CLI:** `python -I tools/x64_recomp.py --emit-macho IN OUT [--base VA] [--rw-base VA] [--map-code]`
  writes a runnable Mach-O on any OS; `--run FILE [--base VA] [--rw-base VA] [--map-code]` translates and
  executes it natively on Apple Silicon (or via QEMU when `QEMU_AARCH64` is set or on other hosts).

## Status and honest limits

- Hand-written fixtures run end-to-end under qemu and natively on Apple Silicon macOS (measured on Apple M5,
  macOS 27.2.1: 46/46 tests in `tests/test_x64_recomp.py` pass without QEMU or JIT, including RIP-relative
  constant reads, multi-page signed images, and PIE indirect dispatch across ASLR runs): arithmetic, loops,
  taken/not-taken signed and unsigned branches, `setcc`/`cmov`, `adc`/`sbb` carry chains, shifts with CF,
  `movsx`, high-byte regs, `call`/`ret`/`leave`/`pushf`/`popf`, `rep movsb`, `repne scasb`, moffs/LEA memory
  addressing, and indirect jumps through the pc table — all hand-assembled, no game bytes.
- Not covered yet (each is an `unhandled` entry at translate time, not silent wrongness): SIMD/SSE
  beyond decode, x87, `rcl`/`rcr`, `std` (DF=1), system/privileged insns, `int`/`syscall`, far
  transfers. Real binaries are x87/SIMD-heavy, so coverage on the engine image will be far below
  the fixture suite until those lands — `recomp_census.py` is the tool that will measure it.
- The engine image itself is still not in the repository (the exe obtained so far is a packaging
  probe — `docs/boot.md`), so nothing here has run against real code. First real milestone is the
  census over a genuine PE, then a same-behaviour run of a real function under qemu or the native Mach-O
  runner against a native x86-64 run.

## Next steps

1. `recomp_census.py` on the real image: instruction histogram, coverage %, hazard list.
2. SIMD/SSE emitters (the largest hazard class on any modern binary), then `std`/RCL-RCR.
3. SEH story: translate `.xdata` prologue ops or run translated code inside a frame the host
   unwinder can walk — needed before C++ `throw` works.
4. Extend the JIT-free Mach-O + `mmap` userland runner from single functions (`build_test_macho`/`run_macho`)
   to multi-function PE images with import-shim thunks on Apple silicon.
5. Vulkan backend under MoltenVK on macOS as the first graphics target; Metal backend later.
