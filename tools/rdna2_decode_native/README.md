# Native RDNA2 disassembler

An optional Rust build of `disasm_bytes` in `tools/rdna2_decode.py`, the decoder that disassembles the shaders
without LLVM. `src/lib.rs` is a branch-by-branch port of `_decode_one`, `_format_operand` and `format_insn`, so the
listing and the invalid-word count are byte-identical. The opcode tables (`tools/rdna2_isa_tables.py`) and the name
maps of `rdna2_decode.py` (hwreg, sendmsg, SDWA selectors, MTBUF formats, ...) are not duplicated:
`tools/rdna2_native.py` serialises the Python ones when the library loads. The crate has no dependencies.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/rdna2_decode_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/rdna2_decode_native/Cargo.toml
python -I tests/test_rdna2_decode_native.py
```

`disasm_bytes` loads `target/release/rdna2_decode_native.dll` (`librdna2_decode_native.so`,
`librdna2_decode_native.dylib`) on its first call and uses Python when:

- the library is not built, or `LEONIDA_RDNA2_DECODE_NATIVE=0`;
- the library was built from another version of the Python decoder. `FINGERPRINT` in `src/lib.rs` is the SHA-256
  of the ported functions (`rdna2_native.PORTED`); a change to any of them turns the native path off until it is
  ported here and the fingerprint updated (`python -I -c "import sys; sys.path.insert(0, 'tools'); import
  rdna2_decode, rdna2_native; print(rdna2_native.fingerprint(vars(rdna2_decode)))"`). A change to a table or a
  name map alone needs nothing;
- `_decode_one`, `format_insn` or `_iter_insns` has been replaced (a test wrapping them), the buffer is not bytes,
  or the library refuses a case Python would raise on.

`decode_bytes` (the instruction dictionaries) stays in Python.

`tests/test_rdna2_decode_native.py` compares the two on synthetic words only: every opcode of every table at
random operand and modifier bits, sweeps of every DPP control, swizzle, waitcnt, hwreg and sendmsg immediate,
every source operand code in each source slot, every MTBUF format, MIMG dimension with a16/d16 and NSA, export
target and FLAT `saddr`, in wave32 and wave64 for gfx1013 and gfx1030, truncated instructions and random bytes. Its
comparisons skip when the library is not built or is stale.

On a 1 MiB synthetic shader-like stream (230,832 instructions), Python takes about 1.6 s and the native path 44 ms
(about 36 times faster); split into 336 shaders of 0.5 to 8 KiB it is 1.6 s and 46 ms (36 times), and on random
bytes 1.7 s and 43 ms (40 times). Loading the library and its tables costs about 40 ms once per process.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
