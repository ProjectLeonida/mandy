"""Measure whether a Vulkan device applies VkSamplerCreateInfo::mipLodBias, with no built test programs.

tools/vk_subgroup_info.py --lod-bias calls measure() for the device it picks and reports the answer as
"explicit_lod_bias". The same MoltenVK 1.4.2 drops the sampler's bias on an Apple M3 and applies it on an Apple M5
Pro (#1774), so the answer is measured, not looked up: a 4x4 R32_SFLOAT image gets 0.0 in level 0 and 1.0 in level 1
(vkCmdClearColorImage), and one compute invocation samples it at explicit LOD 0 through a sampler with mipLodBias 0
and one with +1 (nearest filters and mips, clamp to edge), plus once at explicit LOD 1 with no bias, writing the three
texels to a host-visible buffer. The LOD 1 read is the control: it must return 1.0, or level 1 was never cleared (or
mip selection is broken) and a biased 0.0 would not mean the bias was dropped.

* plain 0.0, biased 1.0, level 1 1.0   -> False: the sampler applies the bias (--explicit-lod-bias would count it twice)
* plain 0.0, biased 0.0, level 1 1.0   -> True: the sampler drops it (translate with rdna2spirv.py --explicit-lod-bias)
* anything else, or any Vulkan error   -> None, with the reason

The shader is SHADER_WORDS, compiled from this GLSL with glslangValidator -V -g0 --target-env vulkan1.0:

    #version 450
    layout(local_size_x = 1) in;
    layout(set = 0, binding = 0) uniform sampler2D plain;
    layout(set = 0, binding = 1) uniform sampler2D biased;
    layout(set = 0, binding = 2) buffer Out { float r[3]; };
    void main() {
        r[0] = textureLod(plain, vec2(0.5), 0.0).r;
        r[1] = textureLod(biased, vec2(0.5), 0.0).r;
        r[2] = textureLod(plain, vec2(0.5), 1.0).r;
    }

Standard library only (ctypes); 64-bit struct layouts, as vk_subgroup_info.py. No game files.
"""
import ctypes
import struct

SHADER_WORDS = (
    0x07230203, 0x00010000, 0x0008000b, 0x0000002d, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001,
    0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0005000f, 0x00000005,
    0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011, 0x00000001, 0x00000001, 0x00000001,
    0x00040047, 0x00000009, 0x00000006, 0x00000004, 0x00030047, 0x0000000a, 0x00000003, 0x00050048, 0x0000000a,
    0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x0000000c, 0x00000021, 0x00000002, 0x00040047, 0x0000000c,
    0x00000022, 0x00000000, 0x00040047, 0x00000012, 0x00000021, 0x00000000, 0x00040047, 0x00000012, 0x00000022,
    0x00000000, 0x00040047, 0x0000001f, 0x00000021, 0x00000001, 0x00040047, 0x0000001f, 0x00000022, 0x00000000,
    0x00040047, 0x0000002c, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003, 0x00000002,
    0x00030016, 0x00000006, 0x00000020, 0x00040015, 0x00000007, 0x00000020, 0x00000000, 0x0004002b, 0x00000007,
    0x00000008, 0x00000003, 0x0004001c, 0x00000009, 0x00000006, 0x00000008, 0x0003001e, 0x0000000a, 0x00000009,
    0x00040020, 0x0000000b, 0x00000002, 0x0000000a, 0x0004003b, 0x0000000b, 0x0000000c, 0x00000002, 0x00040015,
    0x0000000d, 0x00000020, 0x00000001, 0x0004002b, 0x0000000d, 0x0000000e, 0x00000000, 0x00090019, 0x0000000f,
    0x00000006, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x0003001b, 0x00000010,
    0x0000000f, 0x00040020, 0x00000011, 0x00000000, 0x00000010, 0x0004003b, 0x00000011, 0x00000012, 0x00000000,
    0x00040017, 0x00000014, 0x00000006, 0x00000002, 0x0004002b, 0x00000006, 0x00000015, 0x3f000000, 0x0005002c,
    0x00000014, 0x00000016, 0x00000015, 0x00000015, 0x0004002b, 0x00000006, 0x00000017, 0x00000000, 0x00040017,
    0x00000018, 0x00000006, 0x00000004, 0x0004002b, 0x00000007, 0x0000001a, 0x00000000, 0x00040020, 0x0000001c,
    0x00000002, 0x00000006, 0x0004002b, 0x0000000d, 0x0000001e, 0x00000001, 0x0004003b, 0x00000011, 0x0000001f,
    0x00000000, 0x0004002b, 0x0000000d, 0x00000024, 0x00000002, 0x0004002b, 0x00000006, 0x00000026, 0x3f800000,
    0x00040017, 0x0000002a, 0x00000007, 0x00000003, 0x0004002b, 0x00000007, 0x0000002b, 0x00000001, 0x0006002c,
    0x0000002a, 0x0000002c, 0x0000002b, 0x0000002b, 0x0000002b, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003d, 0x00000010, 0x00000013, 0x00000012, 0x00070058, 0x00000018,
    0x00000019, 0x00000013, 0x00000016, 0x00000002, 0x00000017, 0x00050051, 0x00000006, 0x0000001b, 0x00000019,
    0x00000000, 0x00060041, 0x0000001c, 0x0000001d, 0x0000000c, 0x0000000e, 0x0000000e, 0x0003003e, 0x0000001d,
    0x0000001b, 0x0004003d, 0x00000010, 0x00000020, 0x0000001f, 0x00070058, 0x00000018, 0x00000021, 0x00000020,
    0x00000016, 0x00000002, 0x00000017, 0x00050051, 0x00000006, 0x00000022, 0x00000021, 0x00000000, 0x00060041,
    0x0000001c, 0x00000023, 0x0000000c, 0x0000000e, 0x0000001e, 0x0003003e, 0x00000023, 0x00000022, 0x0004003d,
    0x00000010, 0x00000025, 0x00000012, 0x00070058, 0x00000018, 0x00000027, 0x00000025, 0x00000016, 0x00000002,
    0x00000026, 0x00050051, 0x00000006, 0x00000028, 0x00000027, 0x00000000, 0x00060041, 0x0000001c, 0x00000029,
    0x0000000c, 0x0000000e, 0x00000024, 0x0003003e, 0x00000029, 0x00000028, 0x000100fd, 0x00010038
)

u32, u64, f32, vp = ctypes.c_uint32, ctypes.c_uint64, ctypes.c_float, ctypes.c_void_p


def _s(name, fields):
    return type(name, (ctypes.Structure,), {"_fields_": fields})


QueueCreateInfo = _s("QueueCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("queueFamilyIndex", u32),
                                          ("queueCount", u32), ("pQueuePriorities", ctypes.POINTER(f32))])
DeviceCreateInfo = _s("DeviceCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("queueCreateInfoCount", u32),
                                            ("pQueueCreateInfos", ctypes.POINTER(QueueCreateInfo)),
                                            ("enabledLayerCount", u32), ("ppEnabledLayerNames", vp),
                                            ("enabledExtensionCount", u32),
                                            ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
                                            ("pEnabledFeatures", vp)])
QueueFamily = _s("QueueFamily", [("queueFlags", u32), ("queueCount", u32), ("timestampValidBits", u32),
                                  ("granularity", u32 * 3)])
Extent3D = _s("Extent3D", [("width", u32), ("height", u32), ("depth", u32)])
ImageCreateInfo = _s("ImageCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("imageType", u32),
                                          ("format", u32), ("extent", Extent3D), ("mipLevels", u32),
                                          ("arrayLayers", u32), ("samples", u32), ("tiling", u32), ("usage", u32),
                                          ("sharingMode", u32), ("queueFamilyIndexCount", u32),
                                          ("pQueueFamilyIndices", vp), ("initialLayout", u32)])
MemoryRequirements = _s("MemoryRequirements", [("size", u64), ("alignment", u64), ("memoryTypeBits", u32)])
MemoryType = _s("MemoryType", [("propertyFlags", u32), ("heapIndex", u32)])
MemoryHeap = _s("MemoryHeap", [("size", u64), ("flags", u32)])
MemoryProperties = _s("MemoryProperties", [("memoryTypeCount", u32), ("memoryTypes", MemoryType * 32),
                                            ("memoryHeapCount", u32), ("memoryHeaps", MemoryHeap * 16)])
MemoryAllocateInfo = _s("MemoryAllocateInfo", [("sType", u32), ("pNext", vp), ("allocationSize", u64),
                                                ("memoryTypeIndex", u32)])
SubresourceRange = _s("SubresourceRange", [("aspectMask", u32), ("baseMipLevel", u32), ("levelCount", u32),
                                            ("baseArrayLayer", u32), ("layerCount", u32)])
ImageViewCreateInfo = _s("ImageViewCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("image", u64),
                                                  ("viewType", u32), ("format", u32), ("components", u32 * 4),
                                                  ("subresourceRange", SubresourceRange)])
SamplerCreateInfo = _s("SamplerCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("magFilter", u32),
                                              ("minFilter", u32), ("mipmapMode", u32), ("addressModeU", u32),
                                              ("addressModeV", u32), ("addressModeW", u32), ("mipLodBias", f32),
                                              ("anisotropyEnable", u32), ("maxAnisotropy", f32),
                                              ("compareEnable", u32), ("compareOp", u32), ("minLod", f32),
                                              ("maxLod", f32), ("borderColor", u32),
                                              ("unnormalizedCoordinates", u32)])
BufferCreateInfo = _s("BufferCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("size", u64),
                                            ("usage", u32), ("sharingMode", u32), ("queueFamilyIndexCount", u32),
                                            ("pQueueFamilyIndices", vp)])
LayoutBinding = _s("LayoutBinding", [("binding", u32), ("descriptorType", u32), ("descriptorCount", u32),
                                      ("stageFlags", u32), ("pImmutableSamplers", vp)])
SetLayoutCreateInfo = _s("SetLayoutCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("bindingCount", u32),
                                                  ("pBindings", ctypes.POINTER(LayoutBinding))])
PoolSize = _s("PoolSize", [("type", u32), ("descriptorCount", u32)])
PoolCreateInfo = _s("PoolCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("maxSets", u32),
                                        ("poolSizeCount", u32), ("pPoolSizes", ctypes.POINTER(PoolSize))])
SetAllocateInfo = _s("SetAllocateInfo", [("sType", u32), ("pNext", vp), ("descriptorPool", u64),
                                          ("descriptorSetCount", u32), ("pSetLayouts", ctypes.POINTER(u64))])
DescriptorImageInfo = _s("DescriptorImageInfo", [("sampler", u64), ("imageView", u64), ("imageLayout", u32)])
DescriptorBufferInfo = _s("DescriptorBufferInfo", [("buffer", u64), ("offset", u64), ("range", u64)])
WriteDescriptorSet = _s("WriteDescriptorSet", [("sType", u32), ("pNext", vp), ("dstSet", u64), ("dstBinding", u32),
                                                ("dstArrayElement", u32), ("descriptorCount", u32),
                                                ("descriptorType", u32),
                                                ("pImageInfo", ctypes.POINTER(DescriptorImageInfo)),
                                                ("pBufferInfo", ctypes.POINTER(DescriptorBufferInfo)),
                                                ("pTexelBufferView", vp)])
ShaderModuleCreateInfo = _s("ShaderModuleCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32),
                                                        ("codeSize", ctypes.c_size_t), ("pCode", ctypes.POINTER(u32))])
PipelineLayoutCreateInfo = _s("PipelineLayoutCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32),
                                                            ("setLayoutCount", u32),
                                                            ("pSetLayouts", ctypes.POINTER(u64)),
                                                            ("pushConstantRangeCount", u32),
                                                            ("pPushConstantRanges", vp)])
StageCreateInfo = _s("StageCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32), ("stage", u32),
                                          ("module", u64), ("pName", ctypes.c_char_p), ("pSpecializationInfo", vp)])
ComputePipelineCreateInfo = _s("ComputePipelineCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32),
                                                              ("stage", StageCreateInfo), ("layout", u64),
                                                              ("basePipelineHandle", u64),
                                                              ("basePipelineIndex", ctypes.c_int32)])
CommandPoolCreateInfo = _s("CommandPoolCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32),
                                                      ("queueFamilyIndex", u32)])
CommandBufferAllocateInfo = _s("CommandBufferAllocateInfo", [("sType", u32), ("pNext", vp), ("commandPool", u64),
                                                              ("level", u32), ("commandBufferCount", u32)])
CommandBufferBeginInfo = _s("CommandBufferBeginInfo", [("sType", u32), ("pNext", vp), ("flags", u32),
                                                        ("pInheritanceInfo", vp)])
ImageMemoryBarrier = _s("ImageMemoryBarrier", [("sType", u32), ("pNext", vp), ("srcAccessMask", u32),
                                                ("dstAccessMask", u32), ("oldLayout", u32), ("newLayout", u32),
                                                ("srcQueueFamilyIndex", u32), ("dstQueueFamilyIndex", u32),
                                                ("image", u64), ("subresourceRange", SubresourceRange)])
BufferMemoryBarrier = _s("BufferMemoryBarrier", [("sType", u32), ("pNext", vp), ("srcAccessMask", u32),
                                                  ("dstAccessMask", u32), ("srcQueueFamilyIndex", u32),
                                                  ("dstQueueFamilyIndex", u32), ("buffer", u64), ("offset", u64),
                                                  ("size", u64)])
SubmitInfo = _s("SubmitInfo", [("sType", u32), ("pNext", vp), ("waitSemaphoreCount", u32), ("pWaitSemaphores", vp),
                                ("pWaitDstStageMask", vp), ("commandBufferCount", u32),
                                ("pCommandBuffers", ctypes.POINTER(vp)), ("signalSemaphoreCount", u32),
                                ("pSignalSemaphores", vp)])
FenceCreateInfo = _s("FenceCreateInfo", [("sType", u32), ("pNext", vp), ("flags", u32)])

R32_SFLOAT, LEVELS, SIZE = 100, 2, 4
QUEUE_COMPUTE = 0x2
IGNORED = 0xFFFFFFFF
WHOLE = 0xFFFFFFFFFFFFFFFF


class VkFailure(RuntimeError):
    pass


def _check(name, result):
    if result != 0:
        raise VkFailure(f"{name} failed ({result})")


def _proto(vk):
    """argtypes for every call measure() makes (all pointers and 64-bit handles)."""
    names = {
        "vkGetPhysicalDeviceQueueFamilyProperties": [vp, vp, vp],
        "vkGetPhysicalDeviceMemoryProperties": [vp, vp],
        "vkCreateDevice": [vp, vp, vp, vp], "vkDestroyDevice": [vp, vp], "vkGetDeviceQueue": [vp, u32, u32, vp],
        "vkCreateImage": [vp, vp, vp, vp], "vkDestroyImage": [vp, u64, vp],
        "vkGetImageMemoryRequirements": [vp, u64, vp], "vkGetBufferMemoryRequirements": [vp, u64, vp],
        "vkAllocateMemory": [vp, vp, vp, vp], "vkFreeMemory": [vp, u64, vp],
        "vkBindImageMemory": [vp, u64, u64, u64], "vkBindBufferMemory": [vp, u64, u64, u64],
        "vkMapMemory": [vp, u64, u64, u64, u32, vp], "vkUnmapMemory": [vp, u64],
        "vkCreateImageView": [vp, vp, vp, vp], "vkDestroyImageView": [vp, u64, vp],
        "vkCreateSampler": [vp, vp, vp, vp], "vkDestroySampler": [vp, u64, vp],
        "vkCreateBuffer": [vp, vp, vp, vp], "vkDestroyBuffer": [vp, u64, vp],
        "vkCreateDescriptorSetLayout": [vp, vp, vp, vp], "vkDestroyDescriptorSetLayout": [vp, u64, vp],
        "vkCreateDescriptorPool": [vp, vp, vp, vp], "vkDestroyDescriptorPool": [vp, u64, vp],
        "vkAllocateDescriptorSets": [vp, vp, vp], "vkUpdateDescriptorSets": [vp, u32, vp, u32, vp],
        "vkCreateShaderModule": [vp, vp, vp, vp], "vkDestroyShaderModule": [vp, u64, vp],
        "vkCreatePipelineLayout": [vp, vp, vp, vp], "vkDestroyPipelineLayout": [vp, u64, vp],
        "vkCreateComputePipelines": [vp, u64, u32, vp, vp, vp], "vkDestroyPipeline": [vp, u64, vp],
        "vkCreateCommandPool": [vp, vp, vp, vp], "vkDestroyCommandPool": [vp, u64, vp],
        "vkAllocateCommandBuffers": [vp, vp, vp], "vkBeginCommandBuffer": [vp, vp], "vkEndCommandBuffer": [vp],
        "vkCmdPipelineBarrier": [vp, u32, u32, u32, u32, vp, u32, vp, u32, vp],
        "vkCmdClearColorImage": [vp, u64, u32, vp, u32, vp],
        "vkCmdBindPipeline": [vp, u32, u64], "vkCmdBindDescriptorSets": [vp, u32, u64, u32, u32, vp, u32, vp],
        "vkCmdDispatch": [vp, u32, u32, u32],
        "vkCreateFence": [vp, vp, vp, vp], "vkDestroyFence": [vp, u64, vp],
        "vkQueueSubmit": [vp, u32, vp, u64], "vkWaitForFences": [vp, u32, vp, u32, u64],
    }
    for name, args in names.items():
        getattr(vk, name).argtypes = args


def _memory_type(props, bits, wanted):
    for i in range(props.memoryTypeCount):
        if bits >> i & 1 and props.memoryTypes[i].propertyFlags & wanted == wanted:
            return i
    raise VkFailure(f"no memory type with flags {wanted:#x}")


def measure(vk, physical_device, extensions=()):
    """(True | False | None, reason) for one physical device; `extensions` are its extension names (bytes)."""
    try:
        _proto(vk)
    except AttributeError as e:
        return None, f"the Vulkan library lacks a call: {e}"
    pd = ctypes.c_void_p(physical_device) if not isinstance(physical_device, ctypes.c_void_p) else physical_device
    count = u32()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, ctypes.byref(count), None)
    families = (QueueFamily * count.value)()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, ctypes.byref(count), families)
    family = next((i for i, f in enumerate(families) if f.queueFlags & QUEUE_COMPUTE), None)
    if family is None:
        return None, "no compute queue"
    destroy = []   # (call, handle) in creation order, destroyed backwards
    dev = ctypes.c_void_p()
    try:
        prio = f32(1.0)
        qinfo = QueueCreateInfo(2, None, 0, family, 1, ctypes.pointer(prio))
        ext = [b"VK_KHR_portability_subset"] if b"VK_KHR_portability_subset" in extensions else []
        ext_arr = (ctypes.c_char_p * max(len(ext), 1))(*ext)
        dinfo = DeviceCreateInfo(3, None, 0, 1, ctypes.pointer(qinfo), 0, None, len(ext), ext_arr, None)
        _check("vkCreateDevice", vk.vkCreateDevice(pd, ctypes.byref(dinfo), None, ctypes.byref(dev)))
        queue = ctypes.c_void_p()
        vk.vkGetDeviceQueue(dev, family, 0, ctypes.byref(queue))
        mem_props = MemoryProperties()
        vk.vkGetPhysicalDeviceMemoryProperties(pd, ctypes.byref(mem_props))

        def make(call, info, kind):
            handle = u64()
            _check(call, getattr(vk, call)(dev, ctypes.byref(info), None, ctypes.byref(handle)))
            destroy.append((kind, handle.value))
            return handle.value

        image = make("vkCreateImage", ImageCreateInfo(14, None, 0, 1, R32_SFLOAT, Extent3D(SIZE, SIZE, 1), LEVELS, 1, 1,
                                                       0, 0x2 | 0x4, 0, 0, None, 0), "vkDestroyImage")
        req = MemoryRequirements()
        vk.vkGetImageMemoryRequirements(dev, image, ctypes.byref(req))
        image_mem = make("vkAllocateMemory", MemoryAllocateInfo(5, None, req.size,
                                                                 _memory_type(mem_props, req.memoryTypeBits, 0x1)),
                         "vkFreeMemory")
        _check("vkBindImageMemory", vk.vkBindImageMemory(dev, image, image_mem, 0))
        whole = SubresourceRange(1, 0, LEVELS, 0, 1)
        view = make("vkCreateImageView", ImageViewCreateInfo(15, None, 0, image, 1, R32_SFLOAT, (u32 * 4)(0, 0, 0, 0),
                                                              whole), "vkDestroyImageView")
        samplers = [make("vkCreateSampler", SamplerCreateInfo(31, None, 0, 0, 0, 0, 2, 2, 2, bias, 0, 1.0, 0, 0, 0.0,
                                                              1000.0, 0, 0), "vkDestroySampler") for bias in (0.0, 1.0)]
        buffer = make("vkCreateBuffer", BufferCreateInfo(12, None, 0, 12, 0x20, 0, 0, None), "vkDestroyBuffer")
        breq = MemoryRequirements()
        vk.vkGetBufferMemoryRequirements(dev, buffer, ctypes.byref(breq))
        buffer_mem = make("vkAllocateMemory", MemoryAllocateInfo(5, None, breq.size,
                                                                  _memory_type(mem_props, breq.memoryTypeBits, 0x6)),
                          "vkFreeMemory")
        _check("vkBindBufferMemory", vk.vkBindBufferMemory(dev, buffer, buffer_mem, 0))
        bindings = (LayoutBinding * 3)(LayoutBinding(0, 1, 1, 0x20, None), LayoutBinding(1, 1, 1, 0x20, None),
                                       LayoutBinding(2, 7, 1, 0x20, None))
        set_layout = make("vkCreateDescriptorSetLayout", SetLayoutCreateInfo(32, None, 0, 3, bindings),
                          "vkDestroyDescriptorSetLayout")
        sizes = (PoolSize * 2)(PoolSize(1, 2), PoolSize(7, 1))
        pool = make("vkCreateDescriptorPool", PoolCreateInfo(33, None, 0, 1, 2, sizes), "vkDestroyDescriptorPool")
        layouts = (u64 * 1)(set_layout)
        dset = u64()
        _check("vkAllocateDescriptorSets", vk.vkAllocateDescriptorSets(dev, ctypes.byref(
            SetAllocateInfo(34, None, pool, 1, layouts)), ctypes.byref(dset)))
        images = [DescriptorImageInfo(s, view, 5) for s in samplers]
        binfo = DescriptorBufferInfo(buffer, 0, WHOLE)
        writes = (WriteDescriptorSet * 3)(
            WriteDescriptorSet(35, None, dset.value, 0, 0, 1, 1, ctypes.pointer(images[0]), None, None),
            WriteDescriptorSet(35, None, dset.value, 1, 0, 1, 1, ctypes.pointer(images[1]), None, None),
            WriteDescriptorSet(35, None, dset.value, 2, 0, 1, 7, None, ctypes.pointer(binfo), None))
        vk.vkUpdateDescriptorSets(dev, 3, writes, 0, None)
        code = (u32 * len(SHADER_WORDS))(*SHADER_WORDS)
        module = make("vkCreateShaderModule", ShaderModuleCreateInfo(16, None, 0, 4 * len(SHADER_WORDS), code),
                      "vkDestroyShaderModule")
        playout = make("vkCreatePipelineLayout", PipelineLayoutCreateInfo(30, None, 0, 1, layouts, 0, None),
                       "vkDestroyPipelineLayout")
        cinfo = ComputePipelineCreateInfo(29, None, 0, StageCreateInfo(18, None, 0, 0x20, module, b"main", None),
                                          playout, 0, -1)
        pipeline = u64()
        _check("vkCreateComputePipelines", vk.vkCreateComputePipelines(dev, 0, 1, ctypes.byref(cinfo), None,
                                                                       ctypes.byref(pipeline)))
        destroy.append(("vkDestroyPipeline", pipeline.value))
        cpool = make("vkCreateCommandPool", CommandPoolCreateInfo(39, None, 0, family), "vkDestroyCommandPool")
        cmd = ctypes.c_void_p()
        _check("vkAllocateCommandBuffers", vk.vkAllocateCommandBuffers(dev, ctypes.byref(
            CommandBufferAllocateInfo(40, None, cpool, 0, 1)), ctypes.byref(cmd)))
        _check("vkBeginCommandBuffer", vk.vkBeginCommandBuffer(cmd, ctypes.byref(CommandBufferBeginInfo(42, None, 1,
                                                                                                         None))))
        to_dst = ImageMemoryBarrier(45, None, 0, 0x1000, 0, 7, IGNORED, IGNORED, image, whole)
        vk.vkCmdPipelineBarrier(cmd, 0x1, 0x1000, 0, 0, None, 0, None, 1, ctypes.byref(to_dst))
        for level, value in ((0, 0.0), (1, 1.0)):
            color = (f32 * 4)(value, value, value, value)
            rng = SubresourceRange(1, level, 1, 0, 1)
            vk.vkCmdClearColorImage(cmd, image, 7, ctypes.byref(color), 1, ctypes.byref(rng))
        to_read = ImageMemoryBarrier(45, None, 0x1000, 0x20, 7, 5, IGNORED, IGNORED, image, whole)
        vk.vkCmdPipelineBarrier(cmd, 0x1000, 0x800, 0, 0, None, 0, None, 1, ctypes.byref(to_read))
        vk.vkCmdBindPipeline(cmd, 1, pipeline.value)
        sets = (u64 * 1)(dset.value)
        vk.vkCmdBindDescriptorSets(cmd, 1, playout, 0, 1, sets, 0, None)
        vk.vkCmdDispatch(cmd, 1, 1, 1)
        to_host = BufferMemoryBarrier(44, None, 0x40, 0x2000, IGNORED, IGNORED, buffer, 0, WHOLE)
        vk.vkCmdPipelineBarrier(cmd, 0x800, 0x4000, 0, 0, None, 1, ctypes.byref(to_host), 0, None)
        _check("vkEndCommandBuffer", vk.vkEndCommandBuffer(cmd))
        fence = make("vkCreateFence", FenceCreateInfo(8, None, 0), "vkDestroyFence")
        cmds = (ctypes.c_void_p * 1)(cmd.value)
        submit = SubmitInfo(4, None, 0, None, None, 1, cmds, 0, None)
        _check("vkQueueSubmit", vk.vkQueueSubmit(queue, 1, ctypes.byref(submit), fence))
        fences = (u64 * 1)(fence)
        _check("vkWaitForFences", vk.vkWaitForFences(dev, 1, fences, 1, 10_000_000_000))
        mapped = ctypes.c_void_p()
        _check("vkMapMemory", vk.vkMapMemory(dev, buffer_mem, 0, 12, 0, ctypes.byref(mapped)))
        plain, biased, level1 = struct.unpack("<3f", ctypes.string_at(mapped, 12))
        vk.vkUnmapMemory(dev, buffer_mem)
    except VkFailure as e:
        return None, str(e)
    finally:
        if dev.value:
            for call, handle in reversed(destroy):
                getattr(vk, call)(dev, handle, None)
            vk.vkDestroyDevice(dev, None)
    return verdict(plain, biased, level1)


def verdict(plain: float, biased: float, level1: float):
    """The answer from the three texels: level 0 holds 0.0 and level 1 holds 1.0.

    `level1` is the control read at explicit LOD 1: unless it is 1.0, a biased 0.0 could come from a level 1 that was
    never cleared rather than from a dropped bias, so there is no answer.
    """
    if level1 != 1.0:
        return None, f"control read at LOD 1 gave {level1}, not 1.0 (level 1 not cleared or not selected)"
    if plain == 0.0 and biased == 1.0:
        return False, "the sampler applies mipLodBias (+1 reads level 1)"
    if plain == 0.0 and biased == 0.0:
        return True, "the sampler drops mipLodBias (+1 still reads level 0)"
    return None, f"unexpected texels: plain {plain}, biased {biased}"
