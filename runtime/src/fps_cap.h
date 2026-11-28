#pragma once
// LEONIDA_FPS_CAP=<fps>: a present-rate ceiling on top of the title's own frame interval, so the GPU is given idle
// time between frames instead of being fed every frame the title can produce (steadier pacing, lower power). It
// composes with the console path's own FramePacer rather than replacing it: SetFrameIntervalX lengthens a requested
// interval shorter than the cap, so the vblank grid, latency, tokens and statistics all stay consistent. The DXGI
// swapchain has no such pacer, so its Present waits at the cap.
//
// 0 or unset is off. The value is clamped to 1..10000 fps. No Vulkan or Windows headers here.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace leonida {

// The period LEONIDA_FPS_CAP names, in microseconds; 0 when the cap is off.
inline std::uint32_t fps_cap_period_us() {
    const char* text = std::getenv("LEONIDA_FPS_CAP");
    if (!text || !*text) return 0;
    long fps = std::strtol(text, nullptr, 10);
    if (fps <= 0) return 0;
    if (fps > 10000) fps = 10000;
    return static_cast<std::uint32_t>(1'000'000L / fps);
}

// The interval to hand the pacer: the title's, unless the cap is longer (so a cap below the title's rate wins).
inline std::uint32_t capped_interval_us(std::uint32_t interval_us) {
    const std::uint32_t period = fps_cap_period_us();
    return period && interval_us < period ? period : interval_us;
}

// One line when the cap is first consulted, so a run says whether it is capped.
inline void log_fps_cap_once() {
    static const bool logged = [] {
        if (const std::uint32_t period = fps_cap_period_us())
            std::fprintf(stderr, "[rt] present: frame cap %.2f fps (LEONIDA_FPS_CAP, %.3f ms)\n",
                         1'000'000.0 / double(period), double(period) / 1000.0);
        return true;
    }();
    (void)logged;
}

// Paces calls to wait() at the cap with the steady clock: sleep most of the wait, spin the last millisecond (Windows
// timers oversleep by a scheduler quantum, and a cap that oversleeps is worse than none). A late call does not
// accumulate a debt. Off (no wait) when the cap is unset.
class PresentRateLimiter {
public:
    PresentRateLimiter() : period_ns_(std::uint64_t(fps_cap_period_us()) * 1000) {}

    bool enabled() const { return period_ns_ != 0; }
    std::uint64_t period_ns() const { return period_ns_; }

    void wait() {
        if (!period_ns_) return;
        const auto now = now_ns();
        if (next_ns_ == 0 || now >= next_ns_ + std::int64_t(period_ns_)) {
            next_ns_ = now + std::int64_t(period_ns_);  // first frame or an overrun: resync, do not burst
            return;
        }
        constexpr std::int64_t spin_ns = 1'000'000;
        if (next_ns_ - now > spin_ns) std::this_thread::sleep_for(std::chrono::nanoseconds(next_ns_ - now - spin_ns));
        while (now_ns() < next_ns_) std::this_thread::yield();
        next_ns_ += std::int64_t(period_ns_);
    }

private:
    static std::int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

    std::uint64_t period_ns_ = 0;
    std::int64_t next_ns_ = 0;
};

}  // namespace leonida
