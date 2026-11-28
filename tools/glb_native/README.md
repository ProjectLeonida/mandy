# Native GLB packing

This optional Rust library speeds up the two loops in [`tools/glb_writer.py`](../glb_writer.py) that
touch every vertex. One is `pack_primitive`, which packs float32 positions, normals and UVs with the
same checks, validates and packs u16/u32 indices, computes area-weighted smooth normals or
normalises the given ones, and works out the POSITION min/max. The other is `compute_bounds`. The
crate has no dependencies. The output is byte-identical to the Python code, and every error comes
from Python.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/glb_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/glb_native/Cargo.toml
python -I tests/test_glb_native.py
```

`glb_writer` looks for the 64-bit library in `target/release` (`glb_native.dll`,
`libglb_native.so` or `libglb_native.dylib`). Set `LEONIDA_GLB_NATIVE=0` to run the Python
implementation only. Python also runs when the library is missing or fails to load, and whenever
the library declines an input (it then writes nothing).

The library declines anything it does not reproduce exactly:

- anything Python would reject: index range, triangle count, non-finite float32 values,
  mismatched lengths;
- values other than exact `float` and `int` (bools, subclasses, strings, other number types);
- rows that are not lists or tuples of exactly 3 (or 2) values;
- position ints above 2\*\*24 in magnitude, because Python works out their cross products exactly.

How it reads the data:

- **Lists in place, GIL held.** The library reads `pack_primitive`'s lists in place through CPython
  API functions that the binding passes from `ctypes.pythonapi`. It is loaded with `ctypes.PyDLL`,
  so the GIL stays held for the whole call.
- **Checked before use.** When the library loads, the binding checks `Py_TYPE` against `id(type)`.
  It also checks the in-place reads of list and tuple items and float values against `len()`,
  `id()` and the values. If those reads do not match, the library falls back to the API functions.
  If `Py_TYPE` does not match (a free-threaded build, for example), the library is not used.
- **The same `hypot`.** Normals use a port of CPython 3.12's 3-argument `math.hypot`, with
  hardware FMA when the CPU has it. A library that disagrees with the running interpreter's
  `math.hypot` on a fixed sample of 4,096 vectors is not used.

The tests generate their meshes in memory from seeded random sources. They compare each result's
bytes and float bit patterns, or the error class and message, with the Python path. They also cover
the fallback cases: opt-out, missing library, refusal codes, a wrong layout and a disagreeing
`hypot`. Benchmark below: Windows 11, CPython 3.12.10, rustc 1.100.0. The figures are the best of
three timed runs.

| Workload | Python | Native | Speedup |
| --- | ---: | ---: | ---: |
| `pack_primitive`, 65,536 vertices, smooth normals | 222 ms | 7.4 ms | 30x |
| `pack_primitive`, 65,536 vertices, smooth normals + UV | 231 ms | 8.3 ms | 28x |
| `pack_primitive`, 65,536 vertices, given normals + UV | 120 ms | 7.7 ms | 16x |
| `build_glb`, 4 × 16,384 vertices, smooth normals | 216 ms | 11.7 ms | 18x |
| `build_glb`, 4 × 16,384 vertices, given normals + UV | 150 ms | 15.0 ms | 10x |

The native time is now spent mostly on reading the Python objects: about 8 ns per index int, the
same as `sum()` over the list, and about 32 ns per position tuple. The JSON chunk and buffer
assembly stay in Python.

## Workspace

This crate is a member of the Cargo workspace in [`tools/native`](../native/README.md), so the commands above build
into `tools/native/target/release/`, the first folder the Python loader looks in (this crate's own `target/release/`
is the second). `LEONIDA_NATIVE=0` turns off every native path at once.
