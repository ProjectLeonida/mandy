# Native SPIR-V optimizer

An optional Rust build of `optimize()` in `tools/spirv_opt.py`. `src/lib.rs` follows the Python passes line for
line: the header scan, escape and slot analysis, the CFG and dominator tree, the dominator-scoped walk with its undo
logs, the dead-slot store pass and the backward DCE. Its output text is therefore identical. The crate has no
dependencies.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/spirv_opt_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/spirv_opt_native/Cargo.toml
python -I tests/test_spirv_opt_native.py
```

`spirv_opt.py` loads `target/release/spirv_opt_native.dll` (or `libspirv_opt_native.so` / `libspirv_opt_native.dylib`)
when it is there, and optimizes in Python otherwise. `LEONIDA_SPIRV_OPT_NATIVE=0` turns the native path off.

The native path never reports an error itself. In the following cases it declines and `optimize()` runs the Python
code, with its usual result or exception:

- text that is not ASCII, or control characters other than tab and line ends;
- an integer literal longer than 1000 characters, near CPython's int/str digit limit;
- more than one `OpConstantTrue` or `OpConstantFalse`, where Python's `next(iter(set))` depends on string hashing;
- inputs where the Python code raises (an instruction with a result and no operands, or a `del` of a missing key);
- an alias cycle, where Python's `resolve()` never returns;
- an output buffer that cannot be allocated.

`tests/test_spirv_opt_native.py` compares the two optimizers on 1,500 random synthetic modules, random pass counts,
hand-written edge cases and the translator's output for the hand-written `tests/semantic` fixtures. It also checks
threads and the fallbacks. Its comparisons skip when the library is not built.

`cargo run --release --example bench -- <folder of .spvasm>` times the Rust core on its own.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
