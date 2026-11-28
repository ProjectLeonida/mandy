// Submission retention and rollback with deterministic fake timeline progress. No Vulkan loader or GPU.
#include "../src/vulkan_backend.h"
#include <cstdio>
#include <cstring>

using namespace leonida;
namespace {
int failures = 0, live = 0;
std::uintptr_t next = 100;
std::uint64_t done = 0;
bool fail_submit = false, fail_upload = false;
std::uint32_t upload_attempts = 0, fail_upload_attempt = 0;
std::vector<std::weak_ptr<void>> owners;
template<class T> T handle() { return reinterpret_cast<T>(++next); }
void expect(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}
VkResult VKAPI_CALL pool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* p) {
    *p = handle<VkCommandPool>(); return VK_SUCCESS;
}
VkResult VKAPI_CALL semaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* p) {
    *p = handle<VkSemaphore>(); return VK_SUCCESS;
}
VkResult VKAPI_CALL allocate(VkDevice, const VkCommandBufferAllocateInfo*, VkCommandBuffer* p) {
    *p = handle<VkCommandBuffer>(); return VK_SUCCESS;
}
VkResult VKAPI_CALL begin(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VkResult VKAPI_CALL end(VkCommandBuffer) { return VK_SUCCESS; }
VkResult VKAPI_CALL reset(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VkResult VKAPI_CALL idle(VkQueue) { return VK_SUCCESS; }
VkResult VKAPI_CALL counter(VkDevice, VkSemaphore, std::uint64_t* out) { *out = done; return VK_SUCCESS; }
VkResult VKAPI_CALL submit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) {
    return fail_submit ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}
void VKAPI_CALL destroy_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
void VKAPI_CALL destroy_semaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {}
void VKAPI_CALL bind(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
ShaderUserData last_push{};
void VKAPI_CALL push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t size,
                     const void* data) {
    if (size == sizeof(last_push)) std::memcpy(&last_push, data, sizeof(last_push));
}
void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}
PFN_vkVoidFunction VKAPI_CALL get(VkDevice, const char* name) {
#define PROC(n, f) if (!std::strcmp(name, #n)) return reinterpret_cast<PFN_vkVoidFunction>(f)
    PROC(vkCmdBindPipeline, bind);
    PROC(vkCmdPushConstants, push);
    PROC(vkCmdDispatch, dispatch);
#undef PROC
    return nullptr;
}
}  // namespace

int main() {
    VulkanContext context(get, handle<VkDevice>(), false);
    auto& vk = context.vk;
    vk.vkCreateCommandPool = pool;
    vk.vkDestroyCommandPool = destroy_pool;
    vk.vkCreateSemaphore = semaphore;
    vk.vkDestroySemaphore = destroy_semaphore;
    vk.vkAllocateCommandBuffers = allocate;
    vk.vkBeginCommandBuffer = begin;
    vk.vkEndCommandBuffer = end;
    vk.vkResetCommandBuffer = reset;
    vk.vkGetSemaphoreCounterValue = counter;
    vk.vkQueueWaitIdle = idle;
    vk.vkQueueSubmit = submit;
    context.add_pipeline(1, {handle<VkPipeline>(), handle<VkPipelineLayout>()});
    context.set_root_argument_upload([](std::span<const std::uint32_t>, std::uint64_t& va) -> std::shared_ptr<void> {
        if (++upload_attempts == fail_upload_attempt || fail_upload) return {};
        ++live;
        auto owner = std::shared_ptr<void>(new int(0), [](void* p) { delete static_cast<int*>(p); --live; });
        owners.emplace_back(owner);
        va = owners.size() * 256;
        return owner;
    });
    CommandList list;
    expect(list.dispatch({1, {}, 1, 1, 1, {11, 22}}) && list.close(), "record dispatch");
    const CommandList* lists[] = {&list};
    {
        VulkanQueue a(context, handle<VkQueue>(), 0), b(context, handle<VkQueue>(), 0);
        expect(a.execute(lists) == ReplayResult::success && live == 1, "first owner retained after submission");
        expect(a.execute(lists) == ReplayResult::success && live == 2, "repeat execution has independent owner");
        expect(b.execute(lists) == ReplayResult::success && live == 3, "second queue has independent owner");
        expect(!owners[0].expired() && !owners[1].expired() && !owners[2].expired(), "no premature release");
        fail_submit = true;
        expect(a.execute(lists) == ReplayResult::unsupported && live == 3, "failed submit releases only new owner");
        expect(owners.back().expired(), "failed submission not retained by prepared storage");
        fail_submit = false;
        fail_upload = true;
        expect(a.execute(lists) == ReplayResult::unsupported && live == 3, "failed upload leaves in-flight owners");
        fail_upload = false;
        CommandList invalid;
        expect(invalid.dispatch({1, {}, 1, 1, 1, {33}}) &&
               invalid.dispatch({1, {}, 1, 1, 1, {44}}) && invalid.close(), "record two valid dispatches");
        const CommandList* bad[] = {&invalid};
        // Main may skip an unresolved pipeline, so inject a real second-upload failure instead.
        fail_upload_attempt = upload_attempts + 2;
        expect(a.execute(bad) == ReplayResult::unsupported && live == 3,
               "later upload failure releases earlier unsubmitted allocation");
        expect(owners.back().expired(), "earlier allocation in the refused batch was released");
        fail_upload_attempt = 0;
        done = UINT64_MAX;
        expect(a.execute(lists) == ReplayResult::success && live == 2, "completed first queue owners collected");
        expect(owners[0].expired() && owners[1].expired() && !owners[2].expired(),
               "collection on one queue cannot retire another queue's owner");
    }
    expect(live == 0, "queue destruction releases all remaining owners after idle");
    {
        struct DeferredCtx {
            VkPipeline pipeline = handle<VkPipeline>();
            bool built_flag = false;
            int builds = 0;
            int live_at_build = -1;
        } dctx;
        ComputePipeline deferred{};
        deferred.layout = handle<VkPipelineLayout>();
        deferred.build_context = &dctx;
        deferred.built = [](void* ctx) {
            auto* c = static_cast<DeferredCtx*>(ctx);
            return c->built_flag ? c->pipeline : VK_NULL_HANDLE;
        };
        deferred.build = [](void* ctx) {
            auto* c = static_cast<DeferredCtx*>(ctx);
            ++c->builds;
            c->live_at_build = live;
            c->built_flag = true;
            return c->pipeline;
        };
        context.add_pipeline(2, deferred);
        CommandList deferred_list;
        expect(deferred_list.dispatch({2, {}, 1, 1, 1, {55, 66}}) && deferred_list.close(),
               "record deferred compile dispatch");
        const CommandList* dlists[] = {&deferred_list};
        done = 0;
        last_push = {};
        VulkanQueue q(context, handle<VkQueue>(), 0);
        expect(q.execute(dlists) == ReplayResult::success && dctx.builds == 1,
               "deferred compile succeeds on re-prepare");
        expect(dctx.live_at_build == 0 && live == 1,
               "pass-0 argument block released before compile and only pass-1 block retained");
        expect(owners.size() >= 2 && owners[owners.size() - 2].expired() && !owners.back().expired(),
               "pass-0 owner expired while pass-1 owner stays alive");
        const std::uint64_t expected_va = owners.size() * 256;
        expect(last_push.s[0] == std::uint32_t(expected_va) && last_push.s[1] == std::uint32_t(expected_va >> 32),
               "emitted push constants carry pass-1 argument block VA");
    }
    expect(live == 0, "deferred queue destruction releases pass-1 owner");
    std::printf("queue root arguments: %d failures\n", failures);
    return failures ? 1 : 0;
}
