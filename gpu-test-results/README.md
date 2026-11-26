# GPU test records

One text file per GPU, driver and run of the GPU tests. [docs/testing.md](../docs/testing.md) has how to run them and
what to post; a record is a pull request that adds a file here.

## Format

A record is a list of fields. A field starts in the first column as `Key:` and continues on indented lines. The
files here are the examples; these are the fields `tools/gpu_results.py` reads:

| Field | What goes in it |
|---|---|
| `GPU:` | Vendor and model, then the family in parentheses: `NVIDIA GeForce RTX 4070 (Ada / AD104, PCI 10de:2786)` |
| `Driver:` | Driver and version first, the rest after a comma: `RADV (Mesa 26.3.1), Vulkan 1.4.363` |
| `Subgroup:` | The device's subgroup range: `32..64` |
| `OS:` | System and version |
| `Commit:` | The commit that was built, as a hash |
| `Tools:` | llvm-mc, SPIRV-Tools, compiler, CMake and Python versions |
| `run_gpu_tests.py:` | One line per run with what it printed: `native: 197/248 passed, 49 skipped, 2 failed` and `LEONIDA_W64_EMULATE=1: 229/248 passed, 17 skipped, 2 failed` |
| `Failed:` | `none`, or the failed fixtures first and then what was read: `cs_a, cs_b (both runs): got 2071, want 71`. A further fixture starts a line of its own: `ps_c: ...` |
| `Skipped:` | Why the skipped ones were skipped |
| `Notes:` | Anything else, as `- ` lines; say here if it was AI-assisted |

Other fields (`build.py:`, `check.py:`, `ctest:`, `CPU:`, `RAM:` and so on) are kept as written. A run of a few
selected tests says so, as docs/testing.md asks.

## Overview of all records

```sh
python -I tools/gpu_results.py              # every record, each failing fixture and where it fails
python -I tools/gpu_results.py --check gpu-test-results/MyGPU.txt   # does a new record have the fields?
python -I tools/gpu_results.py --markdown   # the record table for a document
```

The overview lists each record's native and wave64-emulated counts, each failing fixture with the records and
vendors it fails on, and how many records taken at newer commits no longer list it: many records name failures
that `main` has fixed since.
