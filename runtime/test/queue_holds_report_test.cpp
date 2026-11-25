// The stall report of QueueHolds (src/queue_holds.h, src/held_work_report.h): every instance answers
// report_held_work() with what its queues hold and the owner's extra lines, and stops answering once it is gone.
// Portable: no Vulkan, no Windows headers, no GPU, no game data.
#define _CRT_SECURE_NO_WARNINGS  // std::tmpfile for the report text (MSVC)
#include "../src/queue_holds.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

#define EXPECT(condition)                                                    \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);     \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

using Holds = leonida::QueueHolds<std::uint64_t>;

// One held operation that waits for (fence, value); it describes itself the way a real queue does.
struct TestHolder final : Holds::Holder {
    explicit TestHolder(Holds& holds) : holds(holds) {}
    Holds& holds;
    std::uint64_t fence = 0, value = 0;
    bool held = false;

    void hold(std::uint64_t f, std::uint64_t v) {
        fence = f;
        value = v;
        held = true;
        holds.held_changed(1);
    }
    bool release_ready() override {
        if (!held || holds.pending(fence, value)) return false;
        held = false;
        holds.held_changed(-1);
        return true;
    }
    void discard_waiting_on(std::uint64_t) override {}
    std::uint64_t oldest_held_seq() override { return held ? 1 : 0; }
    void describe_held(std::FILE* out) override {
        if (!held) return;
        std::uint64_t submitted = 0;
        const bool known = holds.submitted(fence, submitted);
        std::fprintf(out, "held: fence %llu >= %llu, submitted %s%llu\n", static_cast<unsigned long long>(fence),
                     static_cast<unsigned long long>(value), known ? "" : "untracked ",
                     static_cast<unsigned long long>(submitted));
    }
};

// What report_held_work() says now.
std::string report() {
    std::FILE* out = std::tmpfile();
    if (!out) return "<no tmpfile>";
    leonida::report_held_work(out);
    std::rewind(out);
    std::string text;
    char chunk[256];
    for (std::size_t got; (got = std::fread(chunk, 1, sizeof(chunk), out)) > 0;) text.append(chunk, got);
    std::fclose(out);
    return text;
}

bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

}  // namespace

int main() {
    EXPECT(report().empty());  // nothing registered, nothing said
    {
        auto holds = std::make_unique<Holds>();
        TestHolder holder(*holds);
        holds->attach(&holder);
        holds->add_fence(7, 0);
        EXPECT(report().empty());  // registered, but nothing held and no extra lines

        // A held operation names its fence, the value it waits for and the highest Signal submitted so far.
        holds->signaled(7, 2);
        holder.hold(7, 5);
        std::string text = report();
        EXPECT(has(text, "held: fence 7 >= 5, submitted 2"));

        // The owner's extra lines come first, and removing them is seen at once.
        holds->set_describe_extra([](std::FILE* out) { std::fprintf(out, "extra: cpu waits\n"); });
        text = report();
        EXPECT(has(text, "extra: cpu waits") && text.find("extra") < text.find("held:"));
        holds->set_describe_extra(nullptr);
        EXPECT(!has(report(), "extra"));

        // An unknown fence reads as untracked; once the Signal is submitted and released, nothing is held.
        std::uint64_t value = 0;
        EXPECT(!holds->submitted(99, value));
        holds->signaled(7, 5);
        holds->pump_ready();
        EXPECT(!holder.held && report().empty());

        // Two instances report independently.
        Holds other;
        TestHolder other_holder(other);
        other.attach(&other_holder);
        other.add_fence(3, 0);
        other_holder.hold(3, 1);
        holder.hold(7, 9);
        text = report();
        EXPECT(has(text, "held: fence 3 >= 1, submitted 0") && has(text, "held: fence 7 >= 9, submitted 5"));
        other.signaled(3, 1);
        other.pump_ready();
        other.detach(&other_holder);
        holds->signaled(7, 9);
        holds->pump_ready();
        holds->detach(&holder);
    }
    // Destroyed instances unregister: the report is empty again.
    EXPECT(report().empty());
    std::printf("queue_holds_report_test: a stall report names each held operation's fence, value and highest "
                "Signal submitted, with the owner's extra lines, per instance, until the instance is gone\n");
    return 0;
}
