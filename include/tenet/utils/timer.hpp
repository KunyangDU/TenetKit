#pragma once
// include/tenet/utils/timer.hpp
//
// Simple RAII wall-clock timer.

#include <chrono>
#include <string>

namespace tenet {

class Timer {
public:
    explicit Timer(std::string label = "") : label_(std::move(label)) {}

    void start() { t0_ = Clock::now(); }
    void stop()  { elapsed_ += Clock::now() - t0_; }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(elapsed_).count();
    }

    const std::string& label() const noexcept { return label_; }

private:
    using Clock = std::chrono::high_resolution_clock;
    std::string                     label_;
    Clock::time_point               t0_{};
    Clock::duration                 elapsed_{};
};

// RAII scope timer: starts on construction, stops on destruction.
class ScopeTimer {
public:
    explicit ScopeTimer(Timer& t) : t_(t) { t_.start(); }
    ~ScopeTimer() { t_.stop(); }
private:
    Timer& t_;
};

} // namespace tenet
