# Standalone BCPack decoding

This local change replaces the production BCPack reconstruction in both Mandy
runtimes and the Python assets exporter with fixed BC1-BC7 decoding. It fixes
incorrect BC7 indices in the Rust table reader, valid BC1/BC4 streams rejected by
its lane scheduler, and missing or fitted initial dictionaries in the older
C++/Python production paths. BC2 is now implemented in all three paths.

Submission base: `4938673668d807bf365b61284bf22b1ff8d02cc4` (`main`).
The original decoder audit used `6dfca642807631bbe41d6925f152529f99c71ce2`;
the decoder files were unchanged when rebasing onto the submission base.
Branch: `bcpack/standalone-correctness`.
Implementation and review were AI-assisted with Codex. Coordination claim:
[BCPACK-STANDALONE-BC1-BC7](https://github.com/ProjectLeonida/mandy/issues/1608#issuecomment-7236694053).

## Implementation and provenance

The C++ decoder is vendored from
[Xbox2Windows commit d049001ab9a906588f47868277b77e9a00e27053](https://github.com/R0Y00/Xbox2Windows/commit/d049001ab9a906588f47868277b77e9a00e27053),
introduced in [Xbox2Windows #3293](https://github.com/R0Y00/Xbox2Windows/pull/3293).
The local portability adaptation uses C++20 `std::bit_width` and guards MSVC
intrinsics. MSVC x64 dispatches AVX2 only after CPU feature and OS state checks;
other supported C++20 builds use the scalar path. License and upstream notices
are preserved under `third_party/x2w-bcpack`.

Python `tools/bcpack_standalone.py` reconstructs blocks from the same public
source constants, generated into `bcpack_standalone_tables.json` by
`tools/gen_bcpack_standalone_tables.py`. The generator also emits Rust's BC2
companion tables; `--check` verifies both generated files. The existing Rust
format tables remain in `bcpack-tables.bin`.

The Rust BC7 dictionary record is exactly 116 bytes: a u16 control word,
24 u16 endpoint values, two u8 bit fields and 32 i16 index values. It has no
padding between records. The old reader skipped one extra i16 for entries after
the first, shifting their index values. The fixed test checks all signed indices
at independent binary offsets and exercises them through packed-stream goldens.

Rust's lane scheduler now tracks the output budget separately, applies output
backpressure and handles omitted flush zero bytes. BC2 adds 15 lanes and separate
explicit-alpha and colour predictors. The three implementations rebuild BC
blocks byte for byte; texture detiling and GPU sampling remain separate stages.

Research included developer implementation inspection and synthetic API queries;
this is not described as a strict clean-room implementation. No SDK binary,
private header, disassembly, title input or title-derived output is included.
The GDK was used only as a local test oracle; users do not need it to decode.

## Production use and offline research

`runtime/src/bcpack.cpp`, the DirectStorage shim, the D3D12 Rust HLE decoder and
`tools/bcpack.py` use fixed format dictionaries. The standard assets/runtime
build no longer trains `history.bcph`, reads a calibration profile or uses
fitted/provisional predictors. Staged DirectStorage output is published only
after all transforms succeed.

`tools/bcpack.py` accepts the historical quality/history keyword arguments for
caller compatibility and ignores them. The matching CLI flags explain this.
The optional Python native library is used only for `unpack_model` entropy
decoding; its old fitted block reconstruction is not called. Pure Python remains
available without the compiled library or third-party Python dependencies.

Existing explicit offline calibration tools are retained. Their old decoder is
isolated in `tools/bcpack_research.py`, and their tests select it explicitly.
`ttpf_mips.py --calibration` is labelled as an offline research export. Normal
`make_game.py` assets/runtime stages do not import or select this path. This
change removes fitting from production; it does not delete the repository's
separate historical research tools.

The assets pass hashes every standalone decoder input, so a decoder update
revisits previous partial or calibrated exports. Changing a legacy calibration
file does not change the output or pass stamp. DDS and manifest replacement
preserve the existing publication and recovery rules.

## Independent verification

The initial audit compared 15,402 synthetic packed streams with local GDK
`BCPackDecompress`. Each implementation received the same packed bytes and target
size. C++ and Python previously agreed on their incomplete dictionary coverage;
Rust accepted more formats but returned incorrect BC7 bytes. Counts below are
exact-byte matches / rejected / different; no game data was used.

| Format | Original C++ / Python | Original Rust | Replacement, each of C++ / Rust / Python |
| --- | --- | --- | --- |
| BC1 | 255 / 519 / 0 | 5 / 769 / 0 | 774 / 0 / 0 |
| BC2 | 0 / 774 / 0 | 0 / 774 / 0 | 774 / 0 / 0 |
| BC3 | 2 / 772 / 0 | 774 / 0 / 0 | 774 / 0 / 0 |
| BC4 | 4 / 770 / 0 | 5 / 769 / 0 | 774 / 0 / 0 |
| BC5 | 7 / 767 / 0 | 774 / 0 / 0 | 774 / 0 / 0 |
| BC6H | 0 / 10,758 / 0 | 10,758 / 0 / 0 | 10,758 / 0 / 0 |
| BC7 | 2 / 772 / 0 | 260 / 0 / 514 | 774 / 0 / 0 |

An additional 168 cases used GDK `BCPackCompress` followed by GDK decompression
on synthetic BC blocks. All roundtrips reproduced the original blocks, and all
three replacement decoders matched. These cases cover each format at 32, 64,
255, 256, 257, 1,024, 4,096 and 8,192 blocks, with repeated and mixed blocks.

The committed text fixture contains 1,076 cases: 950 independently encoded
streams with GDK expected bytes and 126 GDK compressor roundtrips up to 1,024
blocks. It includes all 14 BC6H modes and 285 valid BC7 controls, distinct
endpoint/index dictionary references, history wrap and representative requests.
Its expected bytes are not produced by the replacement decoders. The 4,096- and
8,192-block roundtrips remain a local audit to limit fixture size.

Some additional hand-generated rANS experiments decoded different lane symbols
in the reference than their encoder intended. Those streams were not accepted
as valid regression cases merely because the reference returned success. They
are excluded from the fixture; no output fitting was introduced for them.

Final checks on Windows x64 with MSVC 19.43 and the Rust toolchain:

- C++ portable CMake/CTest: 1,076 goldens on every supported reference, scalar
  and AVX2 path, all non-reserved slots of 128 entropy models, output canaries,
  truncated input, nonzero trailing bytes and invalid block sizing.
- Rust `cargo test -p leonida-formats`: 42 passed, five pre-existing ignored
  tests, no failures; includes all committed goldens and malformed-input checks.
- Python BCPack suite with its optional native library built: 133 tests, one
  platform skip, no failures. Production goldens run through pure Python and
  native entropy with Python fallback. Without the library the suite also
  passes, with its existing native tests skipped.
- TTPF mip/export suite: 93 passed. Assets suite: 28 tests, two existing
  platform/tool-dependent skips, no failures. Offline calibrated exporter:
  32 passed, preserving the explicit research tool. Offline sidecar research:
  six passed. The assets build entry point: 13 passed.
- DirectStorage shim, compiled directly with MSVC: 546 checks, zero failures.
  Synthetic BC1-BC7 requests cover plain/zlib input, memory/file sources,
  malformed streams, pages and publication/error handling.
- Combined related Python run on the submission base: 308 tests, three
  platform/tool-dependent skips, no failures; includes the suites listed above
  and line-ending checks.
- Full translator regression on the submission base: 505 files, 6,397 tests,
  zero files failed (`scripts/run_semantic_tests.py tests/semantic -j 12`).
- Shared-table generation check, Python compilation, `make_game.py --help`,
  text line-ending checks, game-data screening and Git whitespace checks.

The portable CMake configuration used `LEONIDA_BUILD_WINDOWS_RUNTIME=OFF` and
`LEONIDA_VULKAN_HEADERS_FETCH=ON`. The full Windows runtime configuration needs
Windows SDK 10.0.26100; this machine currently selects 10.0.22621. Direct MSVC
compilation verifies the modified DirectStorage shim despite that full-build
constraint. No game launch, GPU image comparison or frame-rate benchmark was
performed for this decoder change. These finite synthetic tests do not prove
correctness for every possible bit sequence.

## Existing pull request overlap

This proposal was prepared after reviewing the existing claims and pull
requests. It is based directly on `main` and does not apply the following
unmerged pull requests. They overlap or conflict with the replacement and need
maintainer coordination; they are not implementation prerequisites. Their
performance work is not claimed as applied by this branch.

| Existing work | Overlap with this local branch |
| --- | --- |
| [#1044](https://github.com/ProjectLeonida/mandy/pull/1044) | BC6H/BC7 calibrated runtime/request history; this branch replaces that production need with fixed dictionaries. Reviewed head: `f7a18eecddb6db8a050d61fb21c89ca376f775fe`. |
| [#1103](https://github.com/ProjectLeonida/mandy/pull/1103) | BC1/3/4/5 calibrated sidecars built on #1044; the same production files are affected. |
| [#1935](https://github.com/ProjectLeonida/mandy/pull/1935) | rANS advance-table performance/guards; this branch replaces the old C++ entropy path. Reviewed head: `1dc85d99fd6d3857cfe8f2b5dda6307f99039e8e`. |
| [#1974](https://github.com/ProjectLeonida/mandy/pull/1974) | C++ BitWriter performance; the old rebuilder is replaced here. |
| [#1715](https://github.com/ProjectLeonida/mandy/pull/1715) | Lenient lost-block handling; this branch retains decompression errors for malformed input. |
| [#2512](https://github.com/ProjectLeonida/mandy/pull/2512) | Assets local-history reuse; the default assets pass no longer uses local history. |
| [#2502](https://github.com/ProjectLeonida/mandy/pull/2502) | Predictor documentation; the production predictor description changes here. |
| [#2236](https://github.com/ProjectLeonida/mandy/pull/2236), [#2148](https://github.com/ProjectLeonida/mandy/pull/2148), [#1538](https://github.com/ProjectLeonida/mandy/pull/1538) | History fitting/calibration tools; retained as explicit research, isolated from production. |
| [#1422](https://github.com/ProjectLeonida/mandy/pull/1422), [#1811](https://github.com/ProjectLeonida/mandy/pull/1811) | CLI/table-generator hardening; this branch changes CLI history behavior and isolates the legacy generator import. |
| [#1628](https://github.com/ProjectLeonida/mandy/pull/1628) | Shared generated runtime tables; production table generation is removed here. |
| [#2427](https://github.com/ProjectLeonida/mandy/pull/2427), [#2434](https://github.com/ProjectLeonida/mandy/pull/2434) | Legacy format/table documentation and BC7 control research; current production behavior is documented separately here. |

No corresponding open Rust table-parser or BC2 decoder fix was identified in the
reviewed pull requests. Coordination was read in
[issue #1608](https://github.com/ProjectLeonida/mandy/issues/1608), with the
submission claim linked above. The relevant open heads were checked again before
submission; #1044, #1103, #1935, #1974 and #2512 remained open.
