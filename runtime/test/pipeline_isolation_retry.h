#pragma once

#include <cstddef>
#include <string>

namespace leonida {

enum class IsolationAction { abort, complete, record, split };

template<class Run>
IsolationAction isolation_action(const Run& child, std::size_t count) {
    if (!child.runner_error.empty()) return IsolationAction::abort;
    if (child.done) return IsolationAction::complete;
    if (!child.ready) return IsolationAction::abort;
    return count == 1 ? IsolationAction::record : IsolationAction::split;
}

// The attempt handler commits complete results and records terminal failures.
// Interrupted batch results are replaced by their halves, with a batch failure retained
// when neither half reproduces the interruption.
template<class Attempt, class Unreproduced>
std::size_t isolate_range(std::size_t start, std::size_t count, bool& setup_failed,
                          Attempt& attempt, Unreproduced& unreproduced) {
    if (setup_failed) return 0;
    std::string split_reason;
    switch (attempt(start, count, split_reason)) {
    case IsolationAction::abort:
        setup_failed = true;
        return 0;
    case IsolationAction::complete:
        return 0;
    case IsolationAction::record:
        return 1;
    case IsolationAction::split:
        break;
    }
    const std::size_t half = count / 2;
    std::size_t found = isolate_range(start, half, setup_failed, attempt, unreproduced);
    found += isolate_range(start + half, count - half, setup_failed, attempt, unreproduced);
    if (found == 0 && !setup_failed) {
        unreproduced(start, count, split_reason);
        return 1;
    }
    return found;
}

}  // namespace leonida
