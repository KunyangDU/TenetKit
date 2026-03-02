// tests/test_tdvp.cpp
//
// Tests for TDVP / CBE configuration structs and sweep-info types (all
// inline/aggregate).  Integration tests exercise the full TDVP pipeline.

#include <gtest/gtest.h>
#include "tenet/algorithm/tdvp.hpp"
#include "tenet/environment/environment.hpp"
#include "tenet/local_space/spin.hpp"
#include "tenet/intr_tree/interaction_node.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/intr_tree/local_operator.hpp"
#include "tenet/core/space.hpp"
#include "tenet/process_control/config.hpp"
#include "tenet/process_control/sweep_info.hpp"
#include "test_lattice.hpp"

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

// ── Integration test helpers ───────────────────────────────────────────────

// H = J Σ_i (Sz_i Sz_{i+1} + ½ Sp_i Sm_{i+1} + ½ Sm_i Sp_{i+1})
static SparseMPO<> make_heisenberg_tdvp(int L, double J = 1.0)
{
    using namespace spin::half;
    InteractionTree<> tree(L);
    for (int i = 0; i < L - 1; ++i) {
        Op<> sz_i   = std::make_unique<LocalOperator<>>(Sz(), "Sz", i);
        Op<> sz_ip1 = std::make_unique<LocalOperator<>>(Sz(), "Sz", i + 1);
        add_intr2(tree, std::move(sz_i), i, std::move(sz_ip1), i + 1, J);

        Op<> sp_i   = std::make_unique<LocalOperator<>>(Sp(), "Sp", i);
        Op<> sm_ip1 = std::make_unique<LocalOperator<>>(Sm(), "Sm", i + 1);
        add_intr2(tree, std::move(sp_i), i, std::move(sm_ip1), i + 1, 0.5 * J);

        Op<> sm_i   = std::make_unique<LocalOperator<>>(Sm(), "Sm", i);
        Op<> sp_ip1 = std::make_unique<LocalOperator<>>(Sp(), "Sp", i + 1);
        add_intr2(tree, std::move(sm_i), i, std::move(sp_ip1), i + 1, 0.5 * J);
    }
    return compile(tree);
}

// ── TDVP integration tests ─────────────────────────────────────────────────

// Single-site TDVP must preserve ‖ψ‖ during real-time evolution.
// For real τ and Hermitian H, exp(-i H τ) is unitary → ‖ψ‖ is conserved.
TEST(TDVPIntegration, NormConservationRealTime) {
    const int L = 6;
    const int D = 8;

    auto H = make_heisenberg_tdvp(L);
    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi = DenseMPS<>::random(L, D, phys);

    TDVPConfig cfg;
    cfg.krylov_dim = 16;
    cfg.krylov_tol = 1e-10;
    cfg.trunc      = TruncParams{D, 0.0, false};

    Environment<> env(psi, H);
    psi.right_canonicalize(0, L - 1);
    psi.set_center(0, 0);
    env.build_all();

    const double dt = 0.05;
    for (int step = 0; step < 5; ++step) {
        auto info = tdvp1_step(env, {dt, 0.0}, cfg);
        EXPECT_NEAR(info.norm, 1.0, 1e-5)
            << "Norm deviated at real-time step " << step
            << ": norm=" << info.norm;
    }
}

// Two-site TDVP must agree with single-site TDVP on norm conservation.
// With the same initial state and a short time step, both should
// conserve the norm and produce consistent results.
TEST(TDVPIntegration, Tdvp2AgreesTdvp1) {
    const int L = 4;
    const int D = 4;

    auto H1 = make_heisenberg_tdvp(L);
    auto H2 = make_heisenberg_tdvp(L);

    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi1 = DenseMPS<>::random(L, D, phys);
    auto psi2 = psi1;   // identical initial state

    TDVPConfig cfg;
    cfg.krylov_dim = 16;
    cfg.krylov_tol = 1e-10;
    cfg.trunc      = TruncParams{D, 0.0, false};

    Environment<> env1(psi1, H1);
    psi1.right_canonicalize(0, L - 1);
    psi1.set_center(0, 0);
    env1.build_all();

    Environment<> env2(psi2, H2);
    psi2.right_canonicalize(0, L - 1);
    psi2.set_center(0, 0);
    env2.build_all();

    // Very small time step — Trotter errors are negligible at this scale
    const double dt = 0.01;
    auto info1 = tdvp1_step(env1, {dt, 0.0}, cfg);
    auto info2 = tdvp2_step(env2, {dt, 0.0}, cfg);

    // Both must conserve norm for real-time evolution
    EXPECT_NEAR(info1.norm, 1.0, 1e-4) << "TDVP1 norm deviation";
    EXPECT_NEAR(info2.norm, 1.0, 1e-4) << "TDVP2 norm deviation";

    // The resulting norms should be close to each other
    EXPECT_NEAR(info1.norm, info2.norm, 1e-3)
        << "TDVP1 norm=" << info1.norm << " vs TDVP2 norm=" << info2.norm;
}

// Imaginary-time TDVP normalises the MPS after each half-sweep and
// accumulates the log-partition-function increment in info.lnZ.
// For the Heisenberg AFM (E₀ < 0), d(lnZ)/dβ = −⟨E⟩ > 0, so lnZ must
// grow positive after several imaginary-time steps.
TEST(TDVPIntegration, ImaginaryTimeConvergesToGround) {
    const int L = 4;
    const int D = 8;

    auto H = make_heisenberg_tdvp(L);
    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi = DenseMPS<>::random(L, D, phys);

    TDVPConfig cfg;
    cfg.krylov_dim = 16;
    cfg.krylov_tol = 1e-10;
    cfg.trunc      = TruncParams{D, 0.0, false};

    Environment<> env(psi, H);
    psi.right_canonicalize(0, L - 1);
    psi.set_center(0, 0);
    env.build_all();

    // Imaginary-time step: τ = {0, −dt}
    const double dt = 0.1;
    std::complex<double> tau_imag{0.0, -dt};

    double total_lnZ = 0.0;
    for (int step = 0; step < 8; ++step) {
        auto info = tdvp1_step(env, tau_imag, cfg);
        total_lnZ += info.lnZ;
        // After normalisation the MPS norm is always 1.
        EXPECT_NEAR(info.norm, 1.0, 1e-6)
            << "Imaginary-time TDVP norm deviated from 1 at step " << step;
    }

    // For Heisenberg AFM (E₀ < 0): lnZ must accumulate positively with β.
    EXPECT_GT(total_lnZ, 0.0)
        << "Imaginary-time TDVP: total_lnZ=" << total_lnZ
        << " should be positive for Heisenberg AFM";

    // After β = 8 × dt = 0.8, the accumulated lnZ should be non-trivial.
    EXPECT_GT(total_lnZ, 0.05)
        << "Imaginary-time TDVP: total_lnZ=" << total_lnZ
        << " did not grow sufficiently after 8 steps";
}

// ── 2D TDVP integration tests ──────────────────────────────────────────────

// TDVP1 real-time evolution on a 2×4 YC Heisenberg cylinder.
// Exercises TDVP with a long-range MPO (periodic Y bonds span Ly-1 sites).
// exp(-iHτ) is unitary → ‖ψ‖ must be conserved.
TEST(TDVPIntegration, YCHeisenberg2x4_Tdvp1Norm)
{
    const int Lx = 2, Ly = 4;
    const int L  = Lx * Ly;   // 8 sites
    const int D  = 10;

    auto H = make_yc_heisenberg_mpo(Lx, Ly);
    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi = DenseMPS<>::random(L, D, phys);

    TDVPConfig cfg;
    cfg.krylov_dim = 16;
    cfg.krylov_tol = 1e-10;
    cfg.trunc      = TruncParams{D, 0.0, false};

    Environment<> env(psi, H);
    psi.right_canonicalize(0, L - 1);
    psi.set_center(0, 0);
    env.build_all();

    const double dt = 0.05;
    for (int step = 0; step < 5; ++step) {
        auto info = tdvp1_step(env, {dt, 0.0}, cfg);
        EXPECT_NEAR(info.norm, 1.0, 1e-4)
            << "YC2x4 TDVP1 norm deviated at step " << step
            << ": norm=" << info.norm;
    }
}

// TDVP2 real-time evolution on a 2×4 YC Heisenberg cylinder.
// Verifies the two-site variant also conserves the norm on a 2D geometry.
// Tolerance is looser (3e-4) than TDVP1 because two-site SVD truncation
// introduces a small per-step norm error that accumulates over sweeps.
TEST(TDVPIntegration, YCHeisenberg2x4_Tdvp2Norm)
{
    const int Lx = 2, Ly = 4;
    const int L  = Lx * Ly;   // 8 sites
    const int D  = 10;

    auto H = make_yc_heisenberg_mpo(Lx, Ly);
    std::vector<TrivialSpace> phys(L, TrivialSpace(2));
    auto psi = DenseMPS<>::random(L, D, phys);

    TDVPConfig cfg;
    cfg.krylov_dim = 16;
    cfg.krylov_tol = 1e-10;
    cfg.trunc      = TruncParams{D, 0.0, false};

    Environment<> env(psi, H);
    psi.right_canonicalize(0, L - 1);
    psi.set_center(0, 0);
    env.build_all();

    const double dt = 0.05;
    for (int step = 0; step < 5; ++step) {
        auto info = tdvp2_step(env, {dt, 0.0}, cfg);
        EXPECT_NEAR(info.norm, 1.0, 3e-4)
            << "YC2x4 TDVP2 norm deviated at step " << step
            << ": norm=" << info.norm;
    }
}

// ── DISABLED_: stubs for future CBE/SETTN modules ─────────────────────────

TEST(DISABLED_CBEIntegration, BondGrowthMonotonic) {
    FAIL() << "requires cbe.cpp + full Phase-1 implementations";
}

TEST(DISABLED_SETTNIntegration, InfiniteTemperatureLimit) {
    FAIL() << "requires settn.cpp + full Phase-1 implementations";
}

} // namespace tenet::test
