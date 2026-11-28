// LEONIDA_FPS_CAP (runtime/src/fps_cap.h): the present-rate ceiling's period and how it clamps the title's frame
// interval, plus that the limiter is off when the cap is. Pure CPU, no GPU, no game data.
#include "../src/fps_cap.h"

#include <cstdint>
#include <cstdio>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

void set_cap(const char* value) {
#if defined(_WIN32)
    _putenv_s("LEONIDA_FPS_CAP", value);
#else
    setenv("LEONIDA_FPS_CAP", value, 1);
#endif
}

}  // namespace

int main() {
    set_cap("");
    check(leonida::fps_cap_period_us() == 0, "unset is off");
    check(leonida::capped_interval_us(8333) == 8333, "off leaves a 120 fps interval alone");

    set_cap("60");
    check(leonida::fps_cap_period_us() == 16'666, "60 fps is 16,666 us");
    check(leonida::capped_interval_us(8333) == 16'666, "a 120 fps interval is lengthened to the 60 fps cap");
    check(leonida::capped_interval_us(33'333) == 33'333, "a 30 fps interval is slower than the cap and stays");
    check(leonida::capped_interval_us(16'666) == 16'666, "an interval equal to the cap stays");

    set_cap("30");
    check(leonida::fps_cap_period_us() == 33'333, "30 fps is 33,333 us");
    check(leonida::capped_interval_us(16'666) == 33'333, "a 60 fps interval is lengthened to the 30 fps cap");

    set_cap("120");
    check(leonida::fps_cap_period_us() == 8333, "120 fps is 8,333 us");

    set_cap("99999");
    check(leonida::fps_cap_period_us() == 100, "above the ceiling clamps to 10000 fps (100 us)");

    set_cap("0");
    check(leonida::fps_cap_period_us() == 0, "0 is off");
    set_cap("-5");
    check(leonida::fps_cap_period_us() == 0, "a negative is off");
    set_cap("abc");
    check(leonida::fps_cap_period_us() == 0, "junk is off");

    // The limiter is off (and waits nothing) when the cap is.
    set_cap("");
    {
        leonida::PresentRateLimiter off;
        check(!off.enabled() && off.period_ns() == 0, "the limiter is off with no cap");
        off.wait();
        check(true, "an off limiter's wait returns");
    }
    set_cap("60");
    {
        leonida::PresentRateLimiter limited;
        check(limited.enabled() && limited.period_ns() == 16'666'000, "the limiter's period matches the cap");
    }

    set_cap("");
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
