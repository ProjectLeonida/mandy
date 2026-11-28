# Synthetic Test Inventory & Subsystem Map (DX-10)

This document provides the authoritative subsystem test map and prerequisite catalog for Leonida, implemented under roadmap task **DX-10**.

## Overview

`tools/test_inventory.py` dynamically discovers source test modules across `tests/` and `runtime/test/` (skipping bytecode caches like `__pycache__` and matching `scripts/run_semantic_tests.py` package discovery rules inside `tests/semantic/`). Over 99% of test files are synthetic and run without retail game packages, executables, or proprietary assets.

```bash
# Generate the full subsystem breakdown and metrics
python -I tools/test_inventory.py --summary

# Output machine-readable JSON inventory
python -I tools/test_inventory.py --json

# Filter tests by subsystem (runtime, shader, tools, viewer)
python -I tools/test_inventory.py --subsystem shader

# Display tests mentioning retail package environment variables (tagged optional vs required)
python -I tools/test_inventory.py --gaps
```

## Subsystems

- **`shader`**: RDNA2 -> SPIR-V translator, float ALU rounding, vector compares, atomics, mesh/wave64 dispatch (`python_unittest`).
- **`tools`**: Asset exporters, PE inspection, manifest checkers, BCPack/texture swizzlers, developer scripts (`python_unittest`, `pytest`).
- **`runtime`**: D3D12/Vulkan shims, PE loader, root signature parser, XMA2 audio shims, memory and descriptor heaps (`ctest_cpp`).
- **`viewer`**: Map viewer HTTP streaming, LOD residency handoffs, camera positioning, world grid chunking (`python_unittest`, `node_test`).

## Runners

- **`python_unittest`**: Pure standard-library Python test suites runnable with `python -I`.
- **`ctest_cpp`**: Native C++20 tests built with CMake/Ninja under `runtime/build/`.
- **`pytest`**: Tests requiring the virtual environment dependencies pinned in `tests/requirements.txt`.
- **`node_test`**: Viewer streaming protocol tests running under Node.js (`node --test`).

## Retail-Gated & Optional Retail Checks (`--gaps`)

`--gaps` surfaces files mentioning `LEONIDA_GAME_DIR`, `LEONIDA_MOVIE`, or `requires_retail_package`, labelling them with `retail_requirement`:
- **`optional`**: Tests whose primary suite runs synthetically and only execute an optional retail leg when `LEONIDA_GAME_DIR`/`LEONIDA_MOVIE` is set (printing `SKIP` otherwise, such as `pe_loader_test.cpp`). These still count toward `synthetic` coverage.
- **`required`**: Tests strictly gated on retail package prerequisites.
