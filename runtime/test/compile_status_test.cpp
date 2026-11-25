#include "../src/compile_status.h"

int main() {
    using namespace std::chrono_literals;
    if (leonida::format_compile_status(1, 0s, "sample", 0) !=
        "[rt] compile: 1 in flight (oldest 0 s: sample), 0 finished\n") return 1;
    if (leonida::format_compile_status(3, 130s, "oldest", 312) !=
        "[rt] compile: 3 in flight (oldest 130 s: oldest), 312 finished\n") return 1;

    std::mutex mutex;
    std::condition_variable received;
    bool saw_both = false;
    leonida::CompileStatus status(5ms, [&](const std::string& line) {
        if (!line.starts_with("[rt] compile: 2 in flight (oldest ") ||
            !line.ends_with(" s: first), 1 finished\n")) return;
        const std::lock_guard lock(mutex);
        saw_both = true;
        received.notify_one();
    });
    status.track("done", [] {});
    const bool reported = status.track("first", [&] {
        return status.track("second", [&] {
            std::unique_lock lock(mutex);
            return received.wait_for(lock, 2s, [&] { return saw_both; });
        });
    });
    if (!reported) {
        std::fputs("No heartbeat identified the oldest of two in-flight calls\n", stderr);
        return 1;
    }
    return 0;
}
