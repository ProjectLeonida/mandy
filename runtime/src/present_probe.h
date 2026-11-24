#pragma once

// LEONIDA_PRESENT_PROBE (any non-empty value other than "0"): opt-in diagnostics for a frame that is presented but
// shows nothing. With it, stderr gets `[rt] present-probe:` lines that say what the title cleared and with which
// colour, which draws were recorded and which were dropped for want of a viewport, what each PresentX carried and
// which check refused it, and which path each present took through the presenter (blit, compose or blank) with the
// description of its source texture. The lines carry ids, sizes, formats, counters and result codes, never the
// contents of a buffer or texture. Unset, none of it runs: no line is written and no counter is kept.
//
// Each kind of line is written for the first few events and then for every n-th, so a long run stays readable.

#include "local_diagnostics.h"

#include <cstdint>
#include <cstdlib>

namespace leonida::present_probe {

// Read once per process, at the first call.
inline bool enabled() {
    static const bool on = [] {
        const char* value = std::getenv("LEONIDA_PRESENT_PROBE");
        return value && *value && *value != '0';
    }();
    return on || local_diagnostics();
}

// True for the first `first` events and for every `every`-th one after them.
inline bool sampled(std::uint64_t n, std::uint64_t first, std::uint64_t every) { return n <= first || n % every == 0; }

}  // namespace leonida::present_probe
