# Optional Rust animation decoder

Build from the repository root with Cargo:

    cargo build --release --locked --offline --manifest-path tools/anim_clips_native/Cargo.toml
    cargo test --release --locked --offline --manifest-path tools/anim_clips_native/Cargo.toml
    python -I tests/test_anim_clips_native.py

The crate has no external dependencies. On Windows, Linux and macOS the Python tool loads the release
library directly from this crate's `target/release` folder. Set `LEONIDA_ANIM_NATIVE=0` for Python-only decoding.

The Rust decoder accepts the plain sequence bytes described in [Keyframe decoding](../../docs/animation.md#keyframe-decoding).
It decodes frame fields and type-10 deltas, then assembles tracks and quaternions. It returns packed binary64
rows and sequence counters to Python. Python constructs the same lists and tuples and joins consecutive sequences
with the existing boundary checks. Each native result buffer is freed by the library that allocated it.

Python handles missing or incompatible libraries, and reruns any sequence the native decoder refuses so the
original exception and partial counters are preserved. The native path defers non-finite float inputs and fields
wider than 64 bits. It also bounds temporary storage for unusually large sequences.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
