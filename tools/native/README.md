# Native accelerators workspace

One Cargo workspace for the optional Rust ports under `tools/`. Each crate keeps its own directory, its tests and
its Python loader; this folder only holds the workspace manifest and the shared `Cargo.lock`. No member has an
external dependency, so the build works offline.

| Crate | Python module it speeds up | Per-module switch |
| --- | --- | --- |
| [`anim_clips_native`](../anim_clips_native/README.md) | `anim_clips.py` | `LEONIDA_ANIM_NATIVE` |
| [`bc7_native`](../bc7_native/README.md) | `dds_png.py` | `LEONIDA_BC7_NATIVE` |
| [`bvh_native`](../bvh_native/README.md) | `bvh_rehydrate.py` | `LEONIDA_BVH_NATIVE` |
| [`glb_native`](../glb_native/README.md) | `glb_writer.py` | `LEONIDA_GLB_NATIVE` |
| [`meshlet_native`](../meshlet_native/README.md) | `meshlet_page.py` | `LEONIDA_MESHLET_NATIVE` |
| [`rdna2_decode_native`](../rdna2_decode_native/README.md) | `rdna2_decode.py` | `LEONIDA_RDNA2_DECODE_NATIVE` |
| [`spirv_asm_native`](../spirv_asm_native/README.md) | `spirv_asm.py` | `LEONIDA_SPIRV_ASM_NATIVE` |
| [`spirv_opt_native`](../spirv_opt_native/README.md) | `spirv_opt.py` | `LEONIDA_SPIRV_OPT_NATIVE` |

```sh
cargo build --release --workspace --locked --offline --manifest-path tools/native/Cargo.toml
cargo test --release --workspace --locked --offline --manifest-path tools/native/Cargo.toml
```

The libraries land in `tools/native/target/release/`. Every loader looks there first and then in its crate's own
`target/release/`, and falls back to Python when neither holds the library or the library declines an input. Set
`LEONIDA_NATIVE=0` to run all of them in Python, or the per-module switch to turn off one.
