#include "../src/pipeline_wait.h"

#include <cstdint>
#include <cstdio>
#include <limits>

#define EXPECT(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

int main() {
    using namespace leonida::pipeline_wait;

    // Initial state: no submission recorded.
    last_submission().store(0, std::memory_order_relaxed);
    EXPECT(!title_submitting());

    // Recording a submission makes title_submitting() true for a positive window.
    note_submission();
    EXPECT(title_submitting(1000000000));

    // Regression: non-positive window must always return false, even if last_submission is in the future.
    last_submission().store(now_ns() + 1000000000, std::memory_order_relaxed);
    EXPECT(!title_submitting(0));
    EXPECT(!title_submitting(-1));
    EXPECT(title_submitting(1000000000));

    // Regression: negative last_submission (e.g. INT64_MIN) must not trigger signed overflow UB.
    last_submission().store(std::numeric_limits<std::int64_t>::min(), std::memory_order_relaxed);
    EXPECT(!title_submitting(1000000000));

    // Old submission outside the window returns false.
    const std::int64_t now = now_ns();
    if (now > 2000000000) {
        last_submission().store(now - 2000000000, std::memory_order_relaxed);
        EXPECT(!title_submitting(1000000000));
    }

    std::puts("pipeline_wait_test: ok");
    return 0;
}
