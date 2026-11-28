# Native BC7 decoder

An optional Rust build of the BC7 decoder in `tools/dds_png.py`. `src/lib.rs` is a line-by-line port of
`_decode_bc7` and `_decode_bc7_block`: the same mode table, field order, endpoint unquantisation, weights,
rounding, rotation and transparent-black reserved mode, so the RGBA8 output is byte-identical. The partition masks
and anchors are not duplicated; `dds_png.py` passes the ones in `tools/bcpack_tables_bptc.py`. The crate has no
dependencies.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/bc7_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/bc7_native/Cargo.toml
python -I tests/test_bc7_native.py
```

`dds_png.py` loads `target/release/bc7_native.dll` (`libbc7_native.so`, `libbc7_native.dylib`) when it is there and
decodes in Python otherwise, or when the library refuses a call. `LEONIDA_BC7_NATIVE=0` turns the native path off.
`tests/test_bc7_native.py` compares the two decoders on random synthetic blocks of every mode, the reserved mode
and unconstrained bytes, at edge-cropped sizes, and checks the fallbacks; its comparisons skip when the library is
not built.

On random synthetic blocks, a 512×512 surface takes about 370 ms in Python and 2.7 ms native (about 130 times
faster); 2048×2048 takes 5.9 s and 43 ms.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
