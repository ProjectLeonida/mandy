// Native shader paths that need a translation fallback on a known driver. Shared with tools/vk_subgroup_info.py.
// Columns: path, VkDriverId, PCI vendor ID, PCI device ID. Zero in a PCI column matches any ID.
// BUILD-2202: Intel Windows mesh draws crash on the corpus's live push-constant loads.
LEONIDA_SHADER_DRIVER_RULE(mesh, 5, 0, 0)
// SHADER-925: Mesa ANV returns a zero BaryCoordKHR.i on Intel Arc.
LEONIDA_SHADER_DRIVER_RULE(barycentric, 6, 0, 0)
// AMD's proprietary Windows driver returns a wrong BaryCoordKHR component on gfx90c (#5933).
// These are amdgpu's CHIP_RENOIR PCI IDs (Renoir, Lucienne, Cezanne and Barcelo).
LEONIDA_SHADER_DRIVER_RULE(barycentric, 1, 0x1002, 0x15E7)
LEONIDA_SHADER_DRIVER_RULE(barycentric, 1, 0x1002, 0x1636)
LEONIDA_SHADER_DRIVER_RULE(barycentric, 1, 0x1002, 0x1638)
LEONIDA_SHADER_DRIVER_RULE(barycentric, 1, 0x1002, 0x164C)
