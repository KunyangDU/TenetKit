// tests/test_tdvp.cpp
//
// Tests for TDVP / CBE configuration structs and sweep-info types (all
// inline/aggregate).  Actual time-evolution correctness requires tdvp.cpp +
// full Phase-1 — those tests carry the DISABLED_ prefix.

#include <gtest/gtest.h>
#include "tenet/algorithm/tdvp.hpp"
#include "tenet/process_control/config.hpp"
#include "tenet/process_control/sweep_info.hpp"

namespace tenet::test {

// ── TDVPConfig defaults ────────────────────────────────────────────────────

TEST(TDVPConfig, DefaultTol) {
    TDVPConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.tol, 1e-8);
}

TEST(TDVPConfig, DefaultKrylovDim) {
    TDVPConfig cfg;
    EXPECT_EQ(cfg.krylov_dim, 32);
}

TEST(TDVPConfig, DefaultMaxKrylovIter) {
    TDVPConfig cfg;
    EXPECT_EQ(cfg.max_krylov_iter, 1);
}

TEST(TDVPConfig, DefaultKrylovTol) {
    TDVPConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.krylov_tol, 1e-8);
}

TEST(TDVPConfig, TruncParamsDefaultNested) {
    TDVPConfig cfg;
    EXPECT_EQ(cfg.trunc.maxD, 0);
    EXPECT_DOUBLE_EQ(cfg.trunc.cutoff, 1e-12);
    EXPECT_FALSE(cfg.trunc.normalize);
}

TEST(TDVPConfig, DesignatedInit) {
    TDVPConfig cfg{.tol = 1e-12, .krylov_dim = 64};
    EXPECT_DOUBLE_EQ(cfg.tol, 1e-12);
    EXPECT_EQ(cfg.krylov_dim, 64);
    EXPECT_EQ(cfg.max_krylov_iter, 1);   // other fields keep default
}

// ── TDVPSweepInfo defaults ─────────────────────────────────────────────────

TEST(TDVPSweepInfo, DefaultStep) {
    TDVPSweepInfo info;
    EXPECT_EQ(info.step, 0);
}

TEST(TDVPSweepInfo, DefaultTime) {
    TDVPSweepInfo info;
    EXPECT_DOUBLE_EQ(info.time, 0.0);
}

TEST(TDVPSweepInfo, DefaultNormOne) {
    TDVPSweepInfo info;
    EXPECT_DOUBLE_EQ(info.norm, 1.0);
}

TEST(TDVPSweepInfo, DefaultNormError) {
    TDVPSweepInfo info;
    EXPECT_DOUBLE_EQ(info.norm_error, 0.0);
}

TEST(TDVPSweepInfo, DefaultTruncError) {
    TDVPSweepInfo info;
    EXPECT_DOUBLE_EQ(info.truncation_err, 0.0);
}

// ── CBEConfig defaults ─────────────────────────────────────────────────────

TEST(CBEConfig, DefaultTargetD) {
    CBEConfig cfg;
    EXPECT_EQ(cfg.target_D, 0);
}

TEST(CBEConfig, DefaultLambda) {
    CBEConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.lambda, 1.2);
}

TEST(CBEConfig, DefaultNBoundary) {
    CBEConfig cfg;
    EXPECT_EQ(cfg.n_boundary, 4);
}

TEST(CBEConfig, DefaultSchemeDynamic) {
    CBEConfig cfg;
    EXPECT_EQ(cfg.scheme, CBESVDScheme::DynamicSVD);
}

TEST(CBEConfig, SchemeEnumValues) {
    // Verify all three enum values are distinct and compile correctly.
    EXPECT_NE(CBESVDScheme::FullSVD,    CBESVDScheme::RandSVD);
    EXPECT_NE(CBESVDScheme::RandSVD,    CBESVDScheme::DynamicSVD);
    EXPECT_NE(CBESVDScheme::FullSVD,    CBESVDScheme::DynamicSVD);
}

// ── CBEInfo defaults ───────────────────────────────────────────────────────

TEST(CBEInfo, DefaultInitialD) {
    CBEInfo info;
    EXPECT_EQ(info.initial_D, 0);
}

TEST(CBEInfo, DefaultFinalD) {
    CBEInfo info;
    EXPECT_EQ(info.final_D, 0);
}

TEST(CBEInfo, DefaultSite) {
    CBEInfo info;
    EXPECT_EQ(info.site, 0);
}

// ── SETTNConfig defaults ───────────────────────────────────────────────────

TEST(SETTNConfig, DefaultMaxOrder) {
    SETTNConfig cfg;
    EXPECT_EQ(cfg.max_order, 10);
}

TEST(SETTNConfig, DefaultTol) {
    SETTNConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.tol, 1e-10);
}

TEST(SETTNConfig, TruncParamsDefaultNested) {
    SETTNConfig cfg;
    EXPECT_EQ(cfg.trunc.maxD, 0);
    EXPECT_DOUBLE_EQ(cfg.trunc.cutoff, 1e-12);
}

// ── DISABLED_: requires full Phase-1 implementation ───────────────────────

// Single-site TDVP must preserve ‖ψ‖ during real-time evolution.
TEST(DISABLED_TDVPIntegration, NormConservationRealTime) {
    FAIL() << "requires tdvp.cpp + full Phase-1 implementations";
}

// Two-site TDVP must agree with single-site for a product-state initial guess.
TEST(DISABLED_TDVPIntegration, Tdvp2AgreesTdvp1) {
    FAIL() << "requires tdvp.cpp + full Phase-1 implementations";
}

// Imaginary-time TDVP (tanTRG) must drive ‖ψ‖² toward exp(-βE_0).
TEST(DISABLED_TDVPIntegration, ImaginaryTimeConvergesToGround) {
    FAIL() << "requires tdvp.cpp + full Phase-1 implementations";
}

// CBE expand step must monotonically increase bond dimension until target_D.
TEST(DISABLED_CBEIntegration, BondGrowthMonotonic) {
    FAIL() << "requires cbe.cpp + full Phase-1 implementations";
}

// SETTN partition function at β→0 must approach 2^L (infinite-temperature).
TEST(DISABLED_SETTNIntegration, InfiniteTemperatureLimit) {
    FAIL() << "requires settn.cpp + full Phase-1 implementations";
}

} // namespace tenet::test
