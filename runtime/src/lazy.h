#pragma once

#include <utility>

namespace rt {

// A value worked out the first time it is asked for, then kept. get(make) runs make() only while no call has returned
// yet: a make() that throws leaves the value unset, so the next get() runs it again, exactly as calling make() each
// time would. Not thread-safe; the owner serialises access (the loader's binding runs under its mutex).
template <class T>
class Lazy {
public:
    template <class Make>
    const T& get(Make&& make) {
        if (!set_) {
            value_ = std::forward<Make>(make)();
            set_ = true;
        }
        return value_;
    }
    // Like get(), but keeps the slot unset when make() returns a falsy/null value so a failed probe can be retried
    // if a later dependency makes the target available.
    template <class Make>
    T get_if_truthy(Make&& make) {
        if (set_) return value_;
        T v = std::forward<Make>(make)();
        if (v) {
            value_ = std::move(v);
            set_ = true;
            return value_;
        }
        return v;
    }
    bool has_value() const { return set_; }

private:
    T value_{};
    bool set_ = false;
};

}  // namespace rt
