#include "../src/compile_status.h"

int main() {
    using namespace std::chrono_literals;
    if (leonida::format_compile_status(1, 0s, 0) !=
        "[rt] compile: 1 in flight (oldest 0 s), 0 finished\n") return 1;
    if (leonida::format_compile_status(3, 130s, 312) !=
        "[rt] compile: 3 in flight (oldest 130 s), 312 finished\n") return 1;
    return 0;
}
