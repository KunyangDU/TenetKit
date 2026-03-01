// tests/test_dmrg.cpp
//
// Tests for DMRG algorithm configuration structs (inline/aggregate) and tag
// types.  Actual DMRG sweep correctness requires dmrg.cpp + all Phase-1
// implementations — those tests carry the DISABLED_ prefix.

#include <gtest/gtest.h>
#include "tenet/algorithm/dmrg.hpp"
#include "tenet/local_space/spin.hpp"

namespace tenet::test {

// ── DMRGConfig defaults ────────────────────────────────────────────────────

TEST(DMRGConfig, DefaultMaxSweeps) {
    DMRGConfig cfg;
    EXPECT_EQ(cfg.max_sweeps, 20);
}

TEST(DMRGConfig, DefaultEnergyTol) {
    DMRGConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.E_tol, 1e-8);
}

TEST(DMRGConfig, DefaultEntropyTol) {
    DMRGConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.S_tol, 1e-5);
}

TEST(DMRGConfig, DefaultKrylovDim) {
    DMRGConfig cfg;
    EXPECT_EQ(cfg.krylov_dim, 8);
}

TEST(DMRGConfig, DefaultMaxKrylovIter) {
    DMRGConfig cfg;
    EXPECT_EQ(cfg.max_krylov_iter, 1);
}

TEST(DMRGConfig, DefaultKrylovTol) {
    DMRGConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.krylov_tol, 1e-6);
}

TEST(DMRGConfig, DefaultGCSweep) {
    DMRGConfig cfg;
    EXPECT_EQ(cfg.gc_sweep, 2);
}

TEST(DMRGConfig, TruncParamsDefaultNested) {
    DMRGConfig cfg;
    // TruncParams{} default: maxD=0, cutoff=1e-12, normalize=false
    EXPECT_EQ(cfg.trunc.maxD, 0);
    EXPECT_DOUBLE_EQ(cfg.trunc.cutoff, 1e-12);
    EXPECT_FALSE(cfg.trunc.normalize);
}

TEST(DMRGConfig, DesignatedInit) {
    DMRGConfig cfg{.max_sweeps = 50, .E_tol = 1e-10};
    EXPECT_EQ(cfg.max_sweeps, 50);
    EXPECT_DOUBLE_EQ(cfg.E_tol, 1e-10);
    // Other fields keep defaults
    EXPECT_DOUBLE_EQ(cfg.S_tol, 1e-5);
}

// ── DMRGSweepInfo defaults ─────────────────────────────────────────────────

TEST(DMRGSweepInfo, DefaultSweepZero) {
    DMRGSweepInfo info;
    EXPECT_EQ(info.sweep, 0);
}

TEST(DMRGSweepInfo, DefaultEnergyZero) {
    DMRGSweepInfo info;
    EXPECT_DOUBLE_EQ(info.ground_energy, 0.0);
}

TEST(DMRGSweepInfo, DefaultConvergedFalse) {
    DMRGSweepInfo info;
    EXPECT_FALSE(info.converged);
}

TEST(DMRGSweepInfo, DefaultEmptyVectors) {
    DMRGSweepInfo info;
    EXPECT_TRUE(info.site_energies.empty());
    EXPECT_TRUE(info.entanglement_entropy.empty());
}

// ── DMRGResult defaults ────────────────────────────────────────────────────

TEST(DMRGResult, DefaultGroundEnergyZero) {
    DMRGResult<> r;
    EXPECT_DOUBLE_EQ(r.ground_energy, 0.0);
}

TEST(DMRGResult, DefaultConvergedFalse) {
    DMRGResult<> r;
    EXPECT_FALSE(r.converged);
}

TEST(DMRGResult, DefaultHistoryEmpty) {
    DMRGResult<> r;
    EXPECT_TRUE(r.history.empty());
}

// ── Scheme tag types ───────────────────────────────────────────────────────

TEST(SchemeTags, SingleSiteDefaultConstruct) {
    [[maybe_unused]] SingleSite ss{};
    // Just verifying it compiles and is default-constructible.
    SUCCEED();
}

TEST(SchemeTags, DoubleSiteDefaultConstruct) {
    [[maybe_unused]] DoubleSite ds{};
    SUCCEED();
}

TEST(SchemeTags, L2RDefaultConstruct) {
    [[maybe_unused]] L2R lr{};
    SUCCEED();
}

TEST(SchemeTags, R2LDefaultConstruct) {
    [[maybe_unused]] R2L rl{};
    SUCCEED();
}

TEST(SchemeTags, TagsAreDistinctTypes) {
    EXPECT_FALSE((std::is_same_v<SingleSite, DoubleSite>));
    EXPECT_FALSE((std::is_same_v<L2R, R2L>));
}

// ── DISABLED_: requires full Phase-1 implementation ───────────────────────

// DMRG on Heisenberg L=10 spin-1/2 chain.
// Known exact ground energy: E_0/J ≈ -4.2580, first gap ≈ 0.3411.
TEST(DISABLED_DMRGIntegration, HeisenbergL10GroundEnergy) {
    FAIL() << "requires dmrg.cpp + full Phase-1 implementations";
}

// Single-site DMRG must be variational: E_0 < E_1 throughout all sweeps.
TEST(DISABLED_DMRGIntegration, EnergyMonotonicallyDecreasing) {
    FAIL() << "requires dmrg.cpp + full Phase-1 implementations";
}

// Two-site DMRG must agree with single-site on the same model to 1e-6.
TEST(DISABLED_DMRGIntegration, Dmrg2AgreesDmrg1) {
    FAIL() << "requires dmrg.cpp + full Phase-1 implementations";
}

// convergence flag must be set when |ΔE| < E_tol over two consecutive sweeps.
TEST(DISABLED_DMRGIntegration, ConvergenceFlagSet) {
    FAIL() << "requires dmrg.cpp + full Phase-1 implementations";
}

} // namespace tenet::test
