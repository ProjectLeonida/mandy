# Native BVH rehydration

This optional Rust library rehydrates the leaves, parent entries, BOX16 bounds, and BOX32 bounds for
[`tools/bvh_rehydrate.py`](../bvh_rehydrate.py), using the layout in [BVH format notes](../../docs/bvh.md#4a-rehydration-and-source-geometry-measured).
It has zero crate dependencies. Python prepares the leaf frame and supplies it to Rust so both paths
use the same rotation, translation and scale. Rust matches the float32 rounding points in the Python
transforms.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/bvh_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/bvh_native/Cargo.toml
python -I tests/test_bvh_native.py
```

The Python tool automatically looks for the appropriate 64-bit library in `target/release`
(`bvh_native.dll`, `libbvh_native.so`, or `libbvh_native.dylib`). Set `LEONIDA_BVH_NATIVE=0` to run
the Python implementation. Python also handles unavailable libraries, rejected input, custom geometry,
and edge cases such as input float32 NaN payloads, half NaN bounds (CPython 3.14 keeps their payload bits where 3.12 quiets them) or finite BOX16 bounds past the half range on the
inward side. A native refusal discards the partial output before Python runs.

The standard-library tests generate leveled trees, paired leaves, both box types, and transforms in
memory. They verify that the native path ran, compare every output byte, and exercise fallback
behavior and error messages. On Windows 11 with a Ryzen 9 5900X and Python 3.12.10, synthetic trees with
16,384 leaves took 225–656 ms in Python and 2.07–2.95 ms with Rust, depending on the transforms.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
