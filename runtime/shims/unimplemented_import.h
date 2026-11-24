// The per-import state of an unimplemented-import thunk (runtime/shims/registry.cpp). Portable C++: no Windows
// headers, so runtime/test/unimplemented_import_test.cpp builds it on any host.
//
// make_unimplemented_thunk makes one record per distinct "dll!name" and puts its address in the thunk (r10), so
// rt_unimplemented reads the import's return class and whether this is its first call from the record itself: one
// relaxed atomic load on a repeat call, with no process-wide lock and no lookup. Before this the per-import state was a
// std::map keyed by the `what` pointer under the mutex that also guards the list of called imports, so every call of
// every unimplemented import, from every thread, serialised on that mutex and walked the tree.
#pragma once

#include <atomic>
#include <cstdint>

namespace rt::shims {

struct UnimplementedImport {
    const char* what;      // "dll!name", lowercase DLL; lives for the process
    uint8_t return_class;  // a ReturnClass, fixed when the thunk is made
    std::atomic<bool> called{false};

    UnimplementedImport(const char* w, uint8_t cls) : what(w), return_class(cls) {}
    UnimplementedImport(const UnimplementedImport&) = delete;
    UnimplementedImport& operator=(const UnimplementedImport&) = delete;

    // True for exactly one caller, the first call of this import, however many threads call it at once. A repeat call
    // is one relaxed load; the exchange runs only while the flag still reads false.
    bool first_call() {
        return !called.load(std::memory_order_relaxed) && !called.exchange(true, std::memory_order_acq_rel);
    }
};

}  // namespace rt::shims
