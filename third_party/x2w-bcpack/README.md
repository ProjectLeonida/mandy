# Standalone BCPack decoder

Imported from Xbox2Windows commit `d049001ab9a906588f47868277b77e9a00e27053`
(PR https://github.com/R0Y00/Xbox2Windows/pull/3293).

BC1-BC7 packed-block reconstruction, exact format dictionaries and rank tables.
No asset fitting, external predictor profile or developer SDK is used at runtime.
The only local adaptation is portable C++20 bit-width detection; MSVC x64 keeps
runtime-dispatched AVX2 and other compilers use the scalar path.

Preserve `THIRD_PARTY_NOTICES.md` and `LICENSE-MIT`. The decoder incorporates
MIT/Apache-2.0 bcpack-rs and MIT silicon-rans material. New BC1/BC2/BC6H routines
are MIT; BC6H wire descriptors adapt MIT DirectXTex. Research included developer
implementation inspection and synthetic API queries; no strict clean-room claim
is made. Neither SDK binaries nor title payloads are redistributed.
