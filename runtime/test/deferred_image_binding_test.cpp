// Unit tests for DeferredImageBinding (runtime/src/deferred_image_binding.h)
// Verifies null callback protection, failure retry behavior, idempotency,
// owner pointer routing, and concurrent single-execution synchronization across threads.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#include "../src/deferred_image_binding.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s\n", message);
    } else {
        std::printf("PASS: %s\n", message);
    }
}

struct MockTarget {
    std::uint64_t id = 42;
    std::atomic<int> bind_calls{0};
    bool should_succeed = true;
};

bool mock_bind(void* owner) {
    if (!owner) return false;
    auto* target = static_cast<MockTarget*>(owner);
    target->bind_calls.fetch_add(1, std::memory_order_relaxed);
    return target->should_succeed;
}

} // namespace

int main() {
    using leonida::DeferredImageBinding;

    // 1. Initial state and owner accessor
    {
        MockTarget target{100, 0, true};
        DeferredImageBinding binding(&target, mock_bind);
        check(!binding.bound(), "initial bound() is false");
        check(binding.owner() == &target, "owner() matches target pointer");
    }

    // 2. Null callback guard
    {
        MockTarget target{101, 0, true};
        DeferredImageBinding null_binding(&target, nullptr);
        check(!null_binding.ensure(), "ensure() with null callback returns false");
        check(!null_binding.bound(), "bound() remains false after null callback ensure()");
    }

    // 3. Failed bind and retry behavior
    {
        MockTarget target{102, 0, false};  // fail initial attempts
        DeferredImageBinding binding(&target, mock_bind);

        check(!binding.ensure(), "ensure() returns false when callback fails");
        check(!binding.bound(), "bound() remains false on failure");
        check(target.bind_calls.load() == 1, "callback invoked once on first attempt");

        // Retry while still failing
        check(!binding.ensure(), "retry ensure() returns false when callback still fails");
        check(!binding.bound(), "bound() still false");
        check(target.bind_calls.load() == 2, "callback invoked second time on retry");

        // Now permit success
        target.should_succeed = true;
        check(binding.ensure(), "ensure() returns true once callback succeeds");
        check(binding.bound(), "bound() becomes true");
        check(target.bind_calls.load() == 3, "callback invoked third time and succeeded");

        // 4. Idempotency: subsequent calls do not invoke callback again
        check(binding.ensure(), "subsequent ensure() returns true immediately");
        check(target.bind_calls.load() == 3, "callback not invoked again after success");
        check(binding.bound(), "bound() remains true");
    }

    // 5. Null owner pointer handling
    {
        std::atomic<bool> called{false};
        auto null_owner_bind = [](void* owner) -> bool {
            return owner == nullptr;
        };
        DeferredImageBinding binding(nullptr, null_owner_bind);
        check(binding.owner() == nullptr, "owner() is nullptr");
        check(binding.ensure(), "ensure() with nullptr owner passes through to callback");
        check(binding.bound(), "bound() is true");
    }

    // 6. Multithreaded concurrency test (16 threads racing on ensure)
    {
        MockTarget shared_target{999, 0, true};
        DeferredImageBinding shared_binding(&shared_target, mock_bind);

        constexpr int kThreadCount = 16;
        std::vector<std::thread> workers;
        workers.reserve(kThreadCount);
        std::atomic<int> success_count{0};

        for (int i = 0; i < kThreadCount; ++i) {
            workers.emplace_back([&] {
                if (shared_binding.ensure()) {
                    success_count.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }

        for (auto& t : workers) {
            t.join();
        }

        check(success_count.load() == kThreadCount, "all 16 concurrent threads reported ensure() == true");
        check(shared_target.bind_calls.load() == 1, "callback executed exactly once under concurrent races");
        check(shared_binding.bound(), "shared_binding.bound() is true after threads finish");
    }

    // 7. Multiple independent instances
    {
        MockTarget t1{1, 0, true};
        MockTarget t2{2, 0, true};
        DeferredImageBinding b1(&t1, mock_bind);
        DeferredImageBinding b2(&t2, mock_bind);

        check(b1.ensure(), "b1 binds successfully");
        check(b1.bound() && !b2.bound(), "b1 is bound while b2 remains unbound");
        check(t1.bind_calls.load() == 1 && t2.bind_calls.load() == 0, "only t1 was invoked");

        check(b2.ensure(), "b2 binds successfully");
        check(b1.bound() && b2.bound(), "both bindings are now bound");
        check(t1.bind_calls.load() == 1 && t2.bind_calls.load() == 1, "both targets invoked exactly once");
    }

    std::printf("\nSummary: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
