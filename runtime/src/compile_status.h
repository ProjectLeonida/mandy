#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace leonida {

inline std::string format_compile_status(std::size_t active, std::chrono::seconds oldest,
                                         std::string_view label, std::size_t finished) {
    return "[rt] compile: " + std::to_string(active) + " in flight (oldest " +
           std::to_string(oldest.count()) + " s: " + std::string(label) + "), " +
           std::to_string(finished) + " finished\n";
}

// Each device owns its reporter. It can sample while the calling threads are blocked inside the driver.
class CompileStatus {
    using Clock = std::chrono::steady_clock;
    using Starts = std::multimap<Clock::time_point, std::string_view>;

    std::mutex mutex_;
    std::condition_variable_any wake_;
    Starts active_;
    std::size_t finished_ = 0;
    // Destroyed first: request stop and join before the sampled state is destroyed.
    std::jthread reporter_;

    void report(std::stop_token stop, std::chrono::milliseconds period,
                const std::function<void(const std::string&)>& emit) {
        std::unique_lock lock(mutex_);
        while (!stop.stop_requested()) {
            wake_.wait_for(lock, stop, period, [] { return false; });
            if (stop.stop_requested()) break;
            if (active_.empty()) continue;
            const auto& [started, label] = *active_.begin();
            const auto oldest = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - started);
            const auto line = format_compile_status(active_.size(), oldest, label, finished_);
            lock.unlock();
            emit(line);
            lock.lock();
        }
    }

    class InFlight {
        CompileStatus& status_;
        Starts::iterator start_;

    public:
        InFlight(CompileStatus& status, std::string_view label) : status_(status) {
            const std::lock_guard lock(status_.mutex_);
            start_ = status_.active_.emplace(Clock::now(), label);
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
            start_reporter(std::chrono::seconds(10), [](const std::string& line) {
                std::fputs(line.c_str(), stderr);
            });
        }
    }

    CompileStatus(std::chrono::milliseconds period, std::function<void(const std::string&)> emit) {
        start_reporter(period, std::move(emit));
    }

    bool enabled() const { return reporter_.joinable(); }

    // The label is borrowed until call returns. The reporter formats it while holding mutex_.
    template<class Call>
    auto track(std::string_view label, Call call) {
        if (!enabled()) return call();
        const InFlight in_flight(*this, label);
        return call();
    }

private:
    void start_reporter(std::chrono::milliseconds period, std::function<void(const std::string&)> emit) {
        reporter_ = std::jthread([this, period, emit = std::move(emit)](std::stop_token stop) {
            report(stop, period, emit);
        });
    }
};

}  // namespace leonida
