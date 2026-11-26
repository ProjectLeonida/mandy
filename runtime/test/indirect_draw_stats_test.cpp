#include "../src/indirect_draw_stats.h"

#include <cstdio>
#include <cstring>

#define EXPECT(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

int main() {
    using namespace leonida::indirect_stats;

    EXPECT(topology_supported(1));
    EXPECT(topology_supported(5));
    EXPECT(topology_supported(10));
    EXPECT(topology_supported(17));
    EXPECT(!topology_supported(0));
    EXPECT(!topology_supported(6));
    EXPECT(!topology_supported(19));

    EXPECT(std::strcmp(reason_name(Reason::compound), "compound signature") == 0);
    EXPECT(std::strcmp(reason_tag(Reason::stateless), "stateless") == 0);
    EXPECT(std::strcmp(reason_name(Reason::reason_count), "unknown") == 0);
    EXPECT(std::strcmp(reason_tag(static_cast<Reason>(999)), "unknown") == 0);
    EXPECT(reason_skips(Reason::reason_count) == 0);

    // Regression: report(nullptr) must be a safe no-op rather than dereferencing null FILE*.
    Counts local{};
    local.report(nullptr);

    // Regression: out-of-range Event must not index past values[4].
    note(static_cast<Event>(4));
    note(static_cast<Event>(99));
    skip(static_cast<Event>(4), Reason::topology);
    EXPECT(reason_skips(Reason::topology) == 0);

    // Valid skip updates counts and writes formatted report cleanly.
    skip(Event::record_skip, Reason::topology);
    EXPECT(reason_skips(Reason::topology) == 1);
    std::FILE* tmp = std::tmpfile();
    EXPECT(tmp != nullptr);
    counts().report(tmp);
    std::rewind(tmp);
    char buf[512]{};
    const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, tmp);
    std::fclose(tmp);
    EXPECT(n > 0);
    EXPECT(std::strstr(buf, "record_skips=1") != nullptr);
    EXPECT(std::strstr(buf, "[topology=1]") != nullptr);

    std::puts("indirect_draw_stats_test: ok");
    return 0;
}
