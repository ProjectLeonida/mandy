// The retail title's hang watchdog, as far as the runtime needs to know it (main.cpp).
//
// The retail executable (1.0.2.1) runs a hang watchdog thread (function at RVA 0x2215a60): when the game's main loop
// has not set its "kick" byte (RVA 0x71b6331) for 30 one-second ticks, it calls a crash callback and then writes to
// address 0 (`movl $0, 0` at RVA 0x2215c35). Under the runtime the game thread can legitimately stall for longer than
// that, e.g. while the driver compiles pipelines against a cold cache (first boot after a driver update), and the
// title then kills itself. By default the watchdog stays armed, as on the console. With LEONIDA_TITLE_WATCHDOG=0, and
// only when that exact trap instruction sits where it does in this build, a runtime thread sets the kick byte twice a
// second so the title's watchdog never fires.
//
// When it does fire, the log used to show only an access violation writing address 0 in title code, which reads as a
// fault of the layer or of whatever happened just before. fired() recognises the case so the runtime can say what it
// was. Nothing here needs Windows: the caller supplies the image bytes, the fault and the stack words.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace leonida::title_watchdog {

inline constexpr std::uint32_t kTrapRva = 0x2215c35;
inline constexpr std::uint32_t kKickRva = 0x71b6331;
inline constexpr unsigned char kTrap[] = {0xC7, 0x04, 0x25, 0, 0, 0, 0, 0, 0, 0, 0};  // movl $0, 0x0

// Whether `image` (its first `image_bytes` bytes readable) has the trap instruction where this build has it. Every
// other use of the two RVAs above is gated on this: another build may have anything there.
inline bool trap_present(const std::uint8_t* image, std::size_t image_bytes) {
    return image && image_bytes >= kTrapRva + sizeof(kTrap) &&
           std::memcmp(image + kTrapRva, kTrap, sizeof(kTrap)) == 0;
}

// An access violation, as the unhandled-exception filter sees it.
struct Fault {
    std::uint64_t image_base = 0;   // where the title's image is mapped
    std::uint64_t instruction = 0;  // the faulting instruction's address
    bool write = false;             // the access was a write
    std::uint64_t address = 0;      // the address it touched
    const std::uint64_t* stack = nullptr;  // the faulting thread's stack, from its stack pointer up
    std::size_t stack_words = 0;
};

// How much of the stack the filter hands over: the crash callback's own frames sit between the fault and the
// watchdog's return address (measured: the return address is the second frame).
inline constexpr std::size_t kStackWords = 128;

// True when the fault is the watchdog ending the run: a write to address 0, either by the trap instruction itself or
// below the watchdog's call of its crash callback, whose return address is the trap's address (the call is the
// instruction before it). The caller has checked trap_present().
inline bool fired(const Fault& fault) {
    if (!fault.write || fault.address != 0) return false;
    const std::uint64_t trap = fault.image_base + kTrapRva;
    if (fault.instruction == trap) return true;
    for (std::size_t i = 0; fault.stack && i < fault.stack_words; ++i)
        if (fault.stack[i] == trap) return true;
    return false;
}

inline constexpr const char* kFiredNote =
    "the title's own hang watchdog ended this run: its main loop did not report progress for 30 one-second ticks, so "
    "the title called its crash callback and wrote to address 0 on purpose. The access violation that follows is that "
    "write, not a fault of its own; what to look at is what kept the main loop busy (pipeline builds, a long wait). "
    "LEONIDA_TITLE_WATCHDOG=0 keeps the watchdog kicked";

}  // namespace leonida::title_watchdog
