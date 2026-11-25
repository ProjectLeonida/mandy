// Tests console identity resolution and registry ConsoleMode mapping
// (runtime/shims/console_identity.h).
// Synthetic environment and types only; no game files or GPU required.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <string>

#include "console_identity.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

// RAII helper to save and restore the LEONIDA_DEVICE_TYPE environment variable.
struct EnvGuard {
    std::string original_value;
    bool was_set = false;

    EnvGuard() {
        char buf[64] = {};
        const DWORD n = GetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", buf, sizeof(buf));
        if (n > 0 && n < sizeof(buf)) {
            original_value = buf;
            was_set = true;
        }
    }

    ~EnvGuard() {
        if (was_set) {
            SetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", original_value.c_str());
        } else {
            SetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", nullptr);
        }
    }

    void set(const char* val) {
        SetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", val);
    }

    void clear() {
        SetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", nullptr);
    }
};

void test_console_mode_for() {
    using namespace rt::shims::console_identity;

    DWORD mode = 0;

    // Series S (Lockhart): 0x60
    check(console_mode_for(static_cast<uint32_t>(DeviceType::XboxScarlettLockhart), mode),
          "Lockhart has a console mode");
    check(mode == 0x60, "Lockhart mode is 0x60");

    // Series X (Anaconda): 0x64
    mode = 0;
    check(console_mode_for(static_cast<uint32_t>(DeviceType::XboxScarlettAnaconda), mode),
          "Anaconda has a console mode");
    check(mode == 0x64, "Anaconda mode is 0x64");

    // Devkit: 0x68
    mode = 0;
    check(console_mode_for(static_cast<uint32_t>(DeviceType::XboxScarlettDevkit), mode),
          "Scarlett Devkit has a console mode");
    check(mode == 0x68, "Scarlett Devkit mode is 0x68");

    // Devices without console modes (PC and Xbox One family): return false
    mode = 0xbeef;
    check(!console_mode_for(static_cast<uint32_t>(DeviceType::Unknown), mode),
          "Unknown device type has no console mode");
    check(!console_mode_for(static_cast<uint32_t>(DeviceType::Pc), mode),
          "PC device type has no console mode");
    check(!console_mode_for(static_cast<uint32_t>(DeviceType::XboxOne), mode),
          "XboxOne has no console mode");
    check(!console_mode_for(static_cast<uint32_t>(DeviceType::XboxOneS), mode),
          "XboxOneS has no console mode");
    check(!console_mode_for(static_cast<uint32_t>(DeviceType::XboxOneX), mode),
          "XboxOneX has no console mode");
    check(!console_mode_for(static_cast<uint32_t>(DeviceType::XboxOneXDevkit), mode),
          "XboxOneXDevkit has no console mode");

    // Out-of-range values: return false
    check(!console_mode_for(9, mode), "Type 9 has no console mode");
    check(!console_mode_for(100, mode), "Type 100 has no console mode");
    check(!console_mode_for(UINT32_MAX, mode), "Type UINT32_MAX has no console mode");
}

void test_configured_device_type() {
    using namespace rt::shims::console_identity;

    EnvGuard env;

    // Unset variable defaults to Series X (Anaconda, 7)
    env.clear();
    check(configured_device_type() == static_cast<uint32_t>(DeviceType::XboxScarlettAnaconda),
          "Unset LEONIDA_DEVICE_TYPE defaults to Series X");

    // Valid integer inputs 0..8
    env.set("0");
    check(configured_device_type() == 0, "LEONIDA_DEVICE_TYPE=0 yields Unknown");
    env.set("1");
    check(configured_device_type() == 1, "LEONIDA_DEVICE_TYPE=1 yields Pc");
    env.set("2");
    check(configured_device_type() == 2, "LEONIDA_DEVICE_TYPE=2 yields XboxOne");
    env.set("3");
    check(configured_device_type() == 3, "LEONIDA_DEVICE_TYPE=3 yields XboxOneS");
    env.set("4");
    check(configured_device_type() == 4, "LEONIDA_DEVICE_TYPE=4 yields XboxOneX");
    env.set("5");
    check(configured_device_type() == 5, "LEONIDA_DEVICE_TYPE=5 yields XboxOneXDevkit");
    env.set("6");
    check(configured_device_type() == 6, "LEONIDA_DEVICE_TYPE=6 yields XboxScarlettLockhart");
    env.set("7");
    check(configured_device_type() == 7, "LEONIDA_DEVICE_TYPE=7 yields XboxScarlettAnaconda");
    env.set("8");
    check(configured_device_type() == 8, "LEONIDA_DEVICE_TYPE=8 yields XboxScarlettDevkit");

    // Non-numeric text counts as 0 per contract
    env.set("text_not_number");
    check(configured_device_type() == 0, "Non-numeric text counts as 0");
    env.set("foobar");
    check(configured_device_type() == 0, "foobar counts as 0");

    // Numbers outside 0..8 give Series X (7)
    env.set("9");
    check(configured_device_type() == 7, "Value 9 outside 0..8 gives Series X");
    env.set("10");
    check(configured_device_type() == 7, "Value 10 gives Series X");
    env.set("-1");
    check(configured_device_type() == 7, "Negative value gives Series X");
    env.set("-99");
    check(configured_device_type() == 7, "Negative value -99 gives Series X");

    // Empty string counts as unset -> Series X (7)
    env.set("");
    check(configured_device_type() == 7, "Empty string gives Series X");

    // Overly long string (>16 chars) counts as unset/overflow -> Series X (7)
    env.set("12345678901234567890");
    check(configured_device_type() == 7, "Oversized string gives Series X");
}

void test_end_to_end_agreement() {
    using namespace rt::shims::console_identity;

    EnvGuard env;

    // Lockhart (6)
    env.set("6");
    const uint32_t lockhart_type = configured_device_type();
    DWORD lockhart_mode = 0;
    check(lockhart_type == 6, "Configured type is Lockhart");
    check(console_mode_for(lockhart_type, lockhart_mode) && lockhart_mode == 0x60,
          "Lockhart agrees across XSystem and OEM ConsoleMode");

    // Anaconda (7)
    env.set("7");
    const uint32_t anaconda_type = configured_device_type();
    DWORD anaconda_mode = 0;
    check(anaconda_type == 7, "Configured type is Anaconda");
    check(console_mode_for(anaconda_type, anaconda_mode) && anaconda_mode == 0x64,
          "Anaconda agrees across XSystem and OEM ConsoleMode");

    // Devkit (8)
    env.set("8");
    const uint32_t devkit_type = configured_device_type();
    DWORD devkit_mode = 0;
    check(devkit_type == 8, "Configured type is Devkit");
    check(console_mode_for(devkit_type, devkit_mode) && devkit_mode == 0x68,
          "Devkit agrees across XSystem and OEM ConsoleMode");

    // PC (1)
    env.set("1");
    const uint32_t pc_type = configured_device_type();
    DWORD pc_mode = 0;
    check(pc_type == 1, "Configured type is Pc");
    check(!console_mode_for(pc_type, pc_mode), "PC leaves ConsoleMode to host registry");
}

}  // namespace

int main() {
    test_console_mode_for();
    test_configured_device_type();
    test_end_to_end_agreement();

    std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures ? 1 : 0;
}
