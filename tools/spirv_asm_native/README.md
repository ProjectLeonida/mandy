# Native SPIR-V assembler

An optional Rust build of `assemble()` in `tools/spirv_asm.py`. `src/lib.rs` ports its tokenizer, ID numbering,
operand grammar walk, literal rules (integers of every width and signedness, decimal and hex floats, OpSwitch case
widths, bit-mask parameters in ascending bit order) and header, so the `.spv` bytes are identical. The grammar is
not duplicated: `spirv_asm.py` serialises the tables of `tools/spirv_asm_tables.py` and passes them in once. The
crate has no dependencies.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/spirv_asm_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/spirv_asm_native/Cargo.toml
python -I tests/test_spirv_asm_native.py
```

`spirv_asm.py` loads `target/release/spirv_asm_native.dll` (`libspirv_asm_native.so`, `libspirv_asm_native.dylib`)
when it is there and assembles in Python otherwise. `LEONIDA_SPIRV_ASM_NATIVE=0` turns the native path off.

The native path never reports an error itself. It declines, and `assemble()` runs the Python code with its usual
result or `SpirvAsmError`, on any error and on what it does not reproduce exactly: text that is not ASCII, control
characters other than tab and line ends, integer spellings `int(tok, 0)` reads but the port does not (`1_000`),
decimal floats spelled with letters (`inf`, `nan`), hex floats that go through `float.fromhex` (subnormal results or
more mantissa digits than the type holds) and integer types other than 8, 16, 32 and 64 bits.
`tests/test_spirv_asm_native.py` compares the two assemblers on hand-written modules, the translated
`tests/semantic` fixtures and random literals of every type, and checks the fallbacks; its comparisons skip when the
library is not built.

On the 221 synthetic audit shaders (14.6 MB of text from `tools/rdna2spirv.py`), `assemble()` takes about 0.9 s in
Python and 55 ms native (about 16 times faster), with all 221 modules byte-identical and none declined.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
