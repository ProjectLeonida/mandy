#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <thread>

namespace leonida {

inline std::string format_compile_status(std::size_t active, std::chrono::seconds oldest, std::size_t finished) {
    return "[rt] compile: " + std::to_string(active) + " in flight (oldest " +
           std::to_string(oldest.count()) + " s), " + std::to_string(finished) + " finished\n";
}

// Each device owns its reporter. It can sample while the calling threads are blocked inside the driver.
class CompileStatus {
    using Clock = std::chrono::steady_clock;
    using Starts = std::multiset<Clock::time_point>;

    std::mutex mutex_;
    std::condition_variable_any wake_;
    Starts active_;
    std::size_t finished_ = 0;
    // Destroyed first: request stop and join before the sampled state is destroyed.
    std::jthread reporter_;

    void report(std::stop_token stop) {
        using namespace std::chrono_literals;
        std::unique_lock lock(mutex_);
        while (!stop.stop_requested()) {
            wake_.wait_for(lock, stop, 10s, [] { return false; });
            if (stop.stop_requested()) break;
            if (active_.empty()) continue;
            const auto count = active_.size();
            const auto oldest = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - *active_.begin());
            const auto finished = finished_;
            lock.unlock();
            const auto line = format_compile_status(count, oldest, finished);
            std::fputs(line.c_str(), stderr);
            lock.lock();
        }
    }

    class InFlight {
        CompileStatus& status_;
        Starts::iterator start_;

    public:
        explicit InFlight(CompileStatus& status) : status_(status) {
            const std::lock_guard lock(status_.mutex_);
            start_ = status_.active_.insert(Clock::now());
        }

        ~InFlight() {
            const std::lock_guard lock(status_.mutex_);
            status_.active_.erase(start_);
            ++status_.finished_;
        }

        InFlight(const InFlight&) = delete;
        InFlight& operator=(const InFlight&) = delete;
    };

public:
    CompileStatus() {
        const char* value = std::getenv("LEONIDA_COMPILE_STATUS");
        if (value && std::string_view(value) == "1") {
            reporter_ = std::jthread([this](std::stop_token stop) { report(stop); });
        }
    }

    template<class Call>
    auto track(Call call) {
        if (!reporter_.joinable()) return call();
        const InFlight in_flight(*this);
        return call();
    }
};

}  // namespace leonida
