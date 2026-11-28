# Optional meshlet page kernels

Build the release library with:

```sh
cargo build --release --locked --offline --manifest-path tools/meshlet_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/meshlet_native/Cargo.toml
python -I tests/test_meshlet_native.py
```

On 64-bit CPython, `tools/meshlet_page.py` automatically loads the DLL, dylib,
or shared object from this crate's `target/release` directory. Set
`LEONIDA_MESHLET_NATIVE=0` to use the Python implementation. The build needs
no crates or game files.

The library checks ordinary meshlet groups, decodes packed positions, normals,
and tangents, and computes packed-block and group geometry counts. The group
pass works directly on the page bytes and computes one cross product per
triangle, sharing it across the face, normal, and cone checks. Python still
selects the block associated with each group and constructs the JSON report.

A missing or incompatible library, mutable or custom input buffers, malformed
groups, wrapped offsets, and non-finite position transforms go through the
Python reference. The centre check also defers radius comparisons close enough
to depend on the final rounding of Python's `math.dist`. Python retains the
original results and exceptions for these inputs. Signed indices supplied
directly to the Python geometry helpers retain regular Python indexing.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
