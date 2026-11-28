#include "../src/dev_status.h"

#include <cstdio>
#include <string>

#define EXPECT(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

int main() {
    using namespace leonida::dev_status;

    Builds local;
    local.name("cs_example");
    EXPECT(local.last() == "cs_example");

    // Truncates safely to buffer capacity (95 chars + NUL).
    const std::string long_name(120, 'x');
    local.name(long_name.c_str());
    EXPECT(local.last().size() == 95);

    // Regression: nullptr must clear the name safely instead of calling strlen(nullptr).
    local.name(nullptr);
    EXPECT(local.last().empty());

    // RAII Build guard with nullptr name must also work cleanly and track active/done counts.
    const auto done_before = builds().done.load();
    {
        Build b(nullptr);
        EXPECT(builds().active.load() == 1);
        EXPECT(builds().last().empty());
    }
    EXPECT(builds().active.load() == 0);
    EXPECT(builds().done.load() == done_before + 1);

    std::puts("dev_status_test: ok");
    return 0;
}
