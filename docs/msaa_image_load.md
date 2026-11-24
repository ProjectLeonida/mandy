# Reading one sample from a multisampled texture

Pass `--msaa-image-load` to the standalone `tools/rdna2spirv.py` translator when the target's set 0 provides the float, unsigned-integer and signed-integer multisample binding pairs (20–25). The driver can support multisampling while the runtime's descriptor budget omits a pair. The translator therefore requires explicit opt-in.

| Instruction | Runtime contract |
|---|---|
| Ordinary `image_load` on 2D MSAA textures and arrays | The address supplies x, y, optional array slice and the sample index. A16 packs pairs of unsigned address halves. `OpImageFetch Sample` reads one sample through the numeric class in the texture descriptor. The channel mask selects output components. D16 packs the selected components into VGPRs. |

The synthetic checks in `tests/semantic/test_image_load_multisample.py` distinguish pixel, slice, sample, channel and numeric class. The separate `image_msaa_load` instruction reads multiple samples of one component. [AMD RDNA2 ISA, table 42](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf#page=72) · [AMD machine-readable ISA](https://gpuopen.com/machine-readable-isa/)
