// title_watchdog.h (RT-WATCHDOG-FIRED-NOTE): recognising the title's hang watchdog ending a run. CPU only, no game
// file: the "image" is a zeroed buffer with the trap bytes written where the header says they are.
#include "../src/title_watchdog.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace wd = leonida::title_watchdog;

static int failures = 0;
static void check(bool condition, const char* what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++failures;
}

int main() {
    // trap_present: the bytes, at the place, inside the image.
    std::vector<std::uint8_t> image(wd::kTrapRva + sizeof(wd::kTrap) + 16, 0xCC);
    check(!wd::trap_present(image.data(), image.size()), "trap_present: other bytes at the RVA are not the trap");
    std::memcpy(image.data() + wd::kTrapRva, wd::kTrap, sizeof(wd::kTrap));
    check(wd::trap_present(image.data(), image.size()), "trap_present: the trap bytes at the RVA");
    check(!wd::trap_present(image.data(), wd::kTrapRva + sizeof(wd::kTrap) - 1),
          "trap_present: an image that ends inside the trap does not have it");
    check(!wd::trap_present(image.data(), 16), "trap_present: an image smaller than the RVA does not have it");
    check(!wd::trap_present(nullptr, image.size()), "trap_present: no image");
    image[wd::kTrapRva + 3] = 1;
    check(!wd::trap_present(image.data(), image.size()), "trap_present: one byte off is another instruction");

    const std::uint64_t base = 0x140000000ull;
    const std::uint64_t trap = base + wd::kTrapRva;

    // The trap instruction itself faults: the crash callback returned.
    wd::Fault at_trap;
    at_trap.image_base = base;
    at_trap.instruction = trap;
    at_trap.write = true;
    at_trap.address = 0;
    check(wd::fired(at_trap), "fired: a write to address 0 by the trap instruction, with no stack given");

    // The usual case: the fault is inside the crash callback, and the watchdog's return address (the trap's own
    // address) is a few words up the stack.
    std::vector<std::uint64_t> stack(wd::kStackWords, 0x1111);
    stack[17] = trap;
    wd::Fault in_callback;
    in_callback.image_base = base;
    in_callback.instruction = base + 0x1000;
    in_callback.write = true;
    in_callback.address = 0;
    in_callback.stack = stack.data();
    in_callback.stack_words = stack.size();
    check(wd::fired(in_callback), "fired: a write to address 0 below the watchdog's call (return address on the stack)");

    wd::Fault f = in_callback;
    f.stack_words = 17;
    check(!wd::fired(f), "fired: the return address past the words handed over is not seen");
    f.stack_words = 18;
    check(wd::fired(f), "fired: the return address as the last word handed over is seen");

    f = in_callback;
    f.write = false;
    check(!wd::fired(f), "fired: a read of address 0 is not the watchdog");
    f = in_callback;
    f.address = 8;
    check(!wd::fired(f), "fired: a write near address 0 is not the watchdog (the trap writes address 0 exactly)");
    f = at_trap;
    f.address = 0x1000;
    check(!wd::fired(f), "fired: the trap address with another target is not the watchdog");

    // A null write somewhere else in the title, with nothing of the watchdog on the stack.
    stack[17] = 0x1111;
    check(!wd::fired(in_callback), "fired: a write to address 0 without the watchdog on the stack is an ordinary fault");
    stack[5] = trap + 1;
    check(!wd::fired(in_callback), "fired: an address next to the trap is not its return address");

    // The image mapped elsewhere: the return address moves with it.
    wd::Fault moved = in_callback;
    moved.image_base = 0x7ff600000000ull;
    stack[40] = moved.image_base + wd::kTrapRva;
    check(wd::fired(moved), "fired: follows the image base");
    check(!wd::fired(in_callback), "fired: and the old base no longer matches");

    wd::Fault no_stack = in_callback;
    no_stack.stack = nullptr;
    check(!wd::fired(no_stack), "fired: a null stack with a word count reads nothing");

    check(std::strstr(wd::kFiredNote, "hang watchdog") && std::strstr(wd::kFiredNote, "LEONIDA_TITLE_WATCHDOG=0"),
          "note: names the watchdog and the variable");

    std::printf("%d failures\n", failures);
    return failures ? 1 : 0;
}
