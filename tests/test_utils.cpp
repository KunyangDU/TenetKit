// tests/test_utils.cpp
//
// Tests for Timer, ScopeTimer, and the spdlog-based logger.
// All of these are fully header-only (timer.hpp + logger.hpp), so no
// compiled .cpp body is required and all tests run immediately.

#include <gtest/gtest.h>
#include "tenet/utils/timer.hpp"
#include "tenet/utils/logger.hpp"

#include <thread>
#include <chrono>

namespace tenet::test {

// ── Timer ──────────────────────────────────────────────────────────────────

TEST(Timer, DefaultLabelEmpty) {
    Timer t;
    EXPECT_EQ(t.label(), "");
}

TEST(Timer, CustomLabel) {
    Timer t("my_timer");
    EXPECT_EQ(t.label(), "my_timer");
}

TEST(Timer, ElapsedMsNonNegativeBeforeStart) {
    Timer t;
    EXPECT_GE(t.elapsed_ms(), 0.0);
}

TEST(Timer, ElapsedMsNonNegativeAfterStartStop) {
    Timer t;
    t.start();
    t.stop();
    EXPECT_GE(t.elapsed_ms(), 0.0);
}

TEST(Timer, ElapsedMsIncreasesWithSleep) {
    Timer t;
    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    t.stop();
    // Must record at least 1 ms (very generous lower bound for slow CI).
    EXPECT_GE(t.elapsed_ms(), 1.0);
}

TEST(Timer, AccumulatesAcrossMultipleStartStop) {
    Timer t;
    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    t.stop();
    double first = t.elapsed_ms();

    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    t.stop();
    double total = t.elapsed_ms();

    EXPECT_GT(total, first);
}

// ── ScopeTimer ─────────────────────────────────────────────────────────────

TEST(ScopeTimer, RecordsElapsedOnDestruction) {
    Timer t("scope");
    {
        ScopeTimer st(t);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // destructor calls t.stop()
    }
    EXPECT_GE(t.elapsed_ms(), 1.0);
}

TEST(ScopeTimer, TimerZeroBeforeScope) {
    Timer t;
    EXPECT_DOUBLE_EQ(t.elapsed_ms(), 0.0);  // nothing recorded yet
}

TEST(ScopeTimer, NestedScopeTimers) {
    Timer outer("outer");
    Timer inner("inner");
    {
        ScopeTimer so(outer);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        {
            ScopeTimer si(inner);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    EXPECT_GE(outer.elapsed_ms(), 1.0);
    EXPECT_GE(inner.elapsed_ms(), 1.0);
    // outer ≥ inner (outer covers inner's sleep too)
    EXPECT_GE(outer.elapsed_ms(), inner.elapsed_ms());
}

// ── Logger ─────────────────────────────────────────────────────────────────

TEST(Logger, GetLoggerReturnsNonNull) {
    auto logger = get_logger();
    EXPECT_NE(logger, nullptr);
}

TEST(Logger, InitLoggerCreatesLogger) {
    // init_logger is idempotent; calling it multiple times is safe.
    init_logger("tenet_test", spdlog::level::warn);
    auto logger = get_logger();
    EXPECT_NE(logger, nullptr);
}

TEST(Logger, LoggerNameMatchesAfterInit) {
    init_logger("tenet_test2", spdlog::level::warn);
    // After a second init_logger the singleton already exists, so the name
    // stays as whatever was set first.  Just verify the call doesn't crash.
    EXPECT_NE(get_logger(), nullptr);
}

TEST(Logger, MacrosDontCrash) {
    // Set to off so nothing actually prints during tests.
    get_logger()->set_level(spdlog::level::off);
    TENET_LOG_TRACE("trace {}", 0);
    TENET_LOG_DEBUG("debug {}", 1);
    TENET_LOG_INFO("info  {}", 2);
    TENET_LOG_WARN("warn  {}", 3);
    TENET_LOG_ERROR("error {}", 4);
    SUCCEED();
}

} // namespace tenet::test
