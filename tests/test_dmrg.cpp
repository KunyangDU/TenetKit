// tests/test_dmrg.cpp
//
// Tests for DMRG algorithm configuration structs (inline/aggregate) and tag
// types.  Integration tests exercise the full Phase-1 DMRG pipeline.

#include <gtest/gtest.h>
#include "tenet/algorithm/dmrg.hpp"
#include "tenet/local_space/spin.hpp"
#include "tenet/intr_tree/interaction_node.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/intr_tree/local_operator.hpp"
#include "tenet/core/space.hpp"

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

// ── Integration test helpers ───────────────────────────────────────────────

// Build the spin-1/2 Heisenberg chain Hamiltonian as a SparseMPO.
// H = J * Σ_i (Sz_i Sz_{i+1} + 0.5 Sp_i Sm_{i+1} + 0.5 Sm_i Sp_{i+1})
static SparseMPO<> make_heisenberg_mpo(int L, double J = 1.0)
{
    using namespace spin::half;
    InteractionTree<> tree(L);

    for (int i = 0; i < L - 1; ++i) {
        // Store as Op<> (= unique_ptr<AbstractLocalOperator<>>) to enable
        // template argument deduction for B=DenseBackend in add_intr2.

        // Sz_i * Sz_{i+1}
        Op<> sz_i   = std::make_unique<LocalOperator<>>(Sz(), "Sz", i);
        Op<> sz_ip1 = std::make_unique<LocalOperator<>>(Sz(), "Sz", i + 1);
        add_intr2(tree, std::move(sz_i), i, std::move(sz_ip1), i + 1, J);

        // 0.5 * Sp_i * Sm_{i+1}
        Op<> sp_i   = std::make_unique<LocalOperator<>>(Sp(), "Sp", i);
        Op<> sm_ip1 = std::make_unique<LocalOperator<>>(Sm(), "Sm", i + 1);
        add_intr2(tree, std::move(sp_i), i, std::move(sm_ip1), i + 1, 0.5 * J);

        // 0.5 * Sm_i * Sp_{i+1}
        Op<> sm_i   = std::make_unique<LocalOperator<>>(Sm(), "Sm", i);
        Op<> sp_ip1 = std::make_unique<LocalOperator<>>(Sp(), "Sp", i + 1);
        add_intr2(tree, std::move(sm_i), i, std::move(sp_ip1), i + 1, 0.5 * J);
    }

    return compile(tree);
}

// ── Integration tests ──────────────────────────────────────────────────────

// DMRG on Heisenberg L=10 spin-1/2 chain.
// Known exact ground energy (OBC, J=1): E_0 ≈ -4.2580.
TEST(DMRGIntegration, HeisenbergL10GroundEnergy) {
    const int    L = 10;
    const int    D = 20;   // bond dimension — large enough for exact result

    auto H = make_heisenberg_mpo(L);
    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi = DenseMPS<>::random(L, D, phys);

    DMRGConfig cfg;
    cfg.max_sweeps = 20;
    cfg.E_tol      = 1e-8;
    cfg.trunc      = TruncParams{D, 1e-12, false};

    Environment<> env(psi, H);
    auto result = dmrg1(env, cfg);

    // Exact E_0/J ≈ -4.2580 for L=10 OBC Heisenberg (J=1)
    EXPECT_NEAR(result.ground_energy, -4.2580, 5e-2);
}

// Single-site DMRG: the converged energy must be below the initial energy
// estimate (variational principle).
TEST(DMRGIntegration, EnergyMonotonicallyDecreasing) {
    const int L = 8;
    const int D = 16;

    auto H = make_heisenberg_mpo(L);
    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi = DenseMPS<>::random(L, D, phys);

    DMRGConfig cfg;
    cfg.max_sweeps = 10;
    cfg.E_tol      = 1e-6;
    cfg.trunc      = TruncParams{D, 1e-12, false};

    Environment<> env(psi, H);
    auto result = dmrg1(env, cfg);

    // Check that we have at least two sweeps of history
    ASSERT_GE(static_cast<int>(result.history.size()), 2);

    // The final energy must be below (or equal to) the first sweep energy
    double E_first = result.history[0].ground_energy;
    double E_final = result.ground_energy;
    EXPECT_LE(E_final, E_first + 1e-6)
        << "DMRG energy should not increase after first sweep";
}

// Two-site DMRG must agree with single-site on the same model.
TEST(DMRGIntegration, Dmrg2AgreesDmrg1) {
    const int L = 8;
    const int D = 16;

    auto H1 = make_heisenberg_mpo(L);
    auto H2 = make_heisenberg_mpo(L);

    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi1 = DenseMPS<>::random(L, D, phys);
    auto psi2 = DenseMPS<>::random(L, D, phys);

    DMRGConfig cfg;
    cfg.max_sweeps = 15;
    cfg.E_tol      = 1e-7;
    cfg.trunc      = TruncParams{D, 1e-12, false};

    Environment<> env1(psi1, H1);
    Environment<> env2(psi2, H2);

    auto r1 = dmrg1(env1, cfg);
    auto r2 = dmrg2(env2, cfg);

    // Both should converge to the same energy within 1e-3
    EXPECT_NEAR(r1.ground_energy, r2.ground_energy, 1e-3);
}

// Convergence flag must be set when |ΔE| < E_tol.
TEST(DMRGIntegration, ConvergenceFlagSet) {
    const int L = 6;
    const int D = 8;

    auto H = make_heisenberg_mpo(L);
    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi = DenseMPS<>::random(L, D, phys);

    DMRGConfig cfg;
    cfg.max_sweeps = 30;
    cfg.E_tol      = 1e-6;
    cfg.trunc      = TruncParams{D, 1e-12, false};

    Environment<> env(psi, H);
    auto result = dmrg1(env, cfg);

    EXPECT_TRUE(result.converged)
        << "DMRG should converge for L=" << L << " with D=" << D
        << " within " << cfg.max_sweeps << " sweeps";
}

} // namespace tenet::test
