// tests/test_settn.cpp
//
// Tests for SETTN (Series Expansion Tensor Network).
//
// Structure:
//   1. FreeFermionRef self-tests   — validate reference functions
//   2. MpoTrace                   — mpo_trace unit tests
//   3. MpoMul                     — mpo_mul unit tests
//   4. MpoAxpy                    — mpo_axpy unit tests
//   5. SETTNUnit                  — settn() unit tests (high-T, single-site)
//   6. FreeFermionValidation      — settn() vs exact free fermion (core integration)
//
// DISABLED tests require additional infrastructure not yet implemented:
//   - Energy_L6           : energy computation (needs Tr(H·ρ))
//   - LnZ_Heisenberg_L4   : needs exact-diagonalization reference value
//   - LnZMonotoneInBeta   : needs multi-β runner

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "free_fermion_ref.hpp"
#include "test_lattice.hpp"

#include "tenet/algorithm/settn.hpp"
#include "tenet/algebra/mpo_trace.hpp"
#include "tenet/algebra/mpo_mul.hpp"
#include "tenet/algebra/mpo_axpy.hpp"
#include "tenet/mps/dense_mpo.hpp"
#include "tenet/local_space/spin.hpp"
#include "tenet/intr_tree/interaction_node.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/intr_tree/local_operator.hpp"

namespace tenet::test {

// ── Test helpers ──────────────────────────────────────────────────────────────

// OBC 1D XX chain: H = -J/2 Σ_i (Sp_i Sm_{i+1} + Sm_i Sp_{i+1})
// After Jordan-Wigner: H_JW = -t Σ(c†_i c_{i+1} + h.c.) with t = J/2
// Single-particle energies: ε_k = -J·cos(kπ/(L+1)), k=1,...,L
static SparseMPO<> make_xx_mpo(int L, double J = 1.0)
{
    using namespace spin::half;
    InteractionTree<> tree(L);
    for (int i = 0; i < L - 1; ++i) {
        Op<> sp_i  = std::make_unique<LocalOperator<>>(Sp(), "Sp", i);
        Op<> sm_ip = std::make_unique<LocalOperator<>>(Sm(), "Sm", i + 1);
        add_intr2(tree, std::move(sp_i), i, std::move(sm_ip), i + 1, -0.5 * J);

        Op<> sm_i  = std::make_unique<LocalOperator<>>(Sm(), "Sm", i);
        Op<> sp_ip = std::make_unique<LocalOperator<>>(Sp(), "Sp", i + 1);
        add_intr2(tree, std::move(sm_i), i, std::move(sp_ip), i + 1, -0.5 * J);
    }
    return compile(tree);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Reference function self-tests (FreeFermionRef)
// ─────────────────────────────────────────────────────────────────────────────

// β→0 high-T: each level f_k → 1/2, lnZ → L·log(2)
TEST(FreeFermionRef, HighTempLimit_LnZ) {
    for (int L : {2, 4, 6}) {
        auto eps = xx_chain_energies(L);
        double lnZ = grand_canonical_lnZ(eps, /*beta=*/1e-4);
        double expected = L * std::log(2.0);
        EXPECT_NEAR(lnZ, expected, 1e-3)
            << "L=" << L << ": lnZ=" << lnZ << " expected=" << expected;
    }
}

// β→0 high-T: E → 0 (spectrum symmetric about 0)
TEST(FreeFermionRef, HighTempLimit_Energy) {
    for (int L : {2, 4, 6}) {
        auto eps = xx_chain_energies(L);
        double E = grand_canonical_energy(eps, /*beta=*/1e-4);
        EXPECT_NEAR(E, 0.0, 1e-3) << "L=" << L << ": E=" << E;
    }
}

// β→∞ low-T: lnZ → -β·E_0
TEST(FreeFermionRef, LowTempLimit_LnZ) {
    const int L = 4;
    auto eps = xx_chain_energies(L);
    double E0 = 0.0;
    for (double e : eps) if (e < 0) E0 += e;

    const double beta = 50.0;
    double lnZ     = grand_canonical_lnZ(eps, beta);
    double expected = -beta * E0;
    EXPECT_NEAR(lnZ, expected, std::abs(expected) * 0.01)
        << "L=" << L << " beta=" << beta;
}

// d(lnZ)/dβ = -E  (numerical differentiation consistency)
TEST(FreeFermionRef, EnergyConsistency_NumericalDiff) {
    const int L = 4;
    auto eps = xx_chain_energies(L);
    for (double beta : {0.5, 1.0, 2.0}) {
        const double dbeta = 1e-5;
        double lnZ_p = grand_canonical_lnZ(eps, beta + dbeta);
        double lnZ_m = grand_canonical_lnZ(eps, beta - dbeta);
        double E_num = -(lnZ_p - lnZ_m) / (2 * dbeta);
        double E_ref  = grand_canonical_energy(eps, beta);
        EXPECT_NEAR(E_num, E_ref, std::abs(E_ref) * 1e-6) << "beta=" << beta;
    }
}

// Σ_k ε_k = 0 (spectrum symmetry)
TEST(FreeFermionRef, SpectrumSymmetry) {
    for (int L : {2, 4, 6, 8}) {
        auto eps = xx_chain_energies(L);
        double sum = 0.0;
        for (double e : eps) sum += e;
        EXPECT_NEAR(sum, 0.0, 1e-12) << "L=" << L;
    }
}

// L=4 explicit energy levels
TEST(FreeFermionRef, EnergiesL4Explicit) {
    const int L = 4;
    auto eps = xx_chain_energies(L, /*J=*/1.0);
    ASSERT_EQ((int)eps.size(), L);
    const double pi = M_PI;
    EXPECT_NEAR(eps[0], -std::cos(1 * pi / 5), 1e-14);
    EXPECT_NEAR(eps[1], -std::cos(2 * pi / 5), 1e-14);
    EXPECT_NEAR(eps[2], -std::cos(3 * pi / 5), 1e-14);
    EXPECT_NEAR(eps[3], -std::cos(4 * pi / 5), 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. mpo_trace unit tests
// ─────────────────────────────────────────────────────────────────────────────

TEST(MpoTrace, IdentityTrace_L4d2) {
    auto rho = DenseMPO<>::identity(4, 2);
    EXPECT_NEAR(mpo_trace(rho).real(), 16.0, 1e-10);
}

TEST(MpoTrace, SingleSiteIdentity) {
    auto rho = DenseMPO<>::identity(1, 2);
    EXPECT_NEAR(mpo_trace(rho).real(), 2.0, 1e-10);
}

TEST(MpoTrace, IdentityTrace_L6d2) {
    auto rho = DenseMPO<>::identity(6, 2);
    EXPECT_NEAR(mpo_trace(rho).real(), 64.0, 1e-8);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. mpo_mul unit tests
// ─────────────────────────────────────────────────────────────────────────────

// I · H = H  →  Tr(I·H) = Tr(H) = 0 for XX chain (no diagonal terms)
TEST(MpoMul, IdentityTimesH_TraceZero) {
    const int L = 4;
    auto I_mpo = DenseMPO<>::identity(L, 2);
    auto H     = make_xx_mpo(L);
    DenseMPO<> C = DenseMPO<>::identity(L, 2);
    MulConfig cfg;
    cfg.trunc = TruncParams{16, 1e-12, false};
    mpo_mul(C, I_mpo, H, 1.0, cfg);
    // Tr(I · H) = Tr(H) = 0 since H has no diagonal terms
    EXPECT_NEAR(mpo_trace(C).real(), 0.0, 1e-6);
}

// alpha scaling: mpo_mul with alpha=2 → Tr(2·I·H) = 0
TEST(MpoMul, AlphaScaling) {
    const int L = 2;
    auto I_mpo = DenseMPO<>::identity(L, 2);
    auto H     = make_xx_mpo(L);
    DenseMPO<> C = DenseMPO<>::identity(L, 2);
    mpo_mul(C, I_mpo, H, 2.0, {});
    EXPECT_NEAR(mpo_trace(C).real(), 0.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. mpo_axpy unit tests
// ─────────────────────────────────────────────────────────────────────────────

// 2·I + I = 3·I → Tr = 3 * d^L = 12
TEST(MpoAxpy, ExactSmallBond_L2d2) {
    auto x = DenseMPO<>::identity(2, 2);   // Tr = 4
    auto y = DenseMPO<>::identity(2, 2);   // Tr = 4
    mpo_axpy(2.0, x, y, {});               // y ← 2·x + y = 3·I → Tr = 12
    EXPECT_NEAR(mpo_trace(y).real(), 12.0, 1e-8);
}

// Linearity: Tr(α·x + y) = α·Tr(x) + Tr(y)
TEST(MpoAxpy, TraceLinearity_L4) {
    const double alpha = 3.0;
    auto x = DenseMPO<>::identity(4, 2);   // Tr(x) = 16
    auto y = DenseMPO<>::identity(4, 2);   // Tr(y) = 16
    double tr_x = mpo_trace(x).real();
    double tr_y = mpo_trace(y).real();
    mpo_axpy(alpha, x, y, {});
    double tr_expected = alpha * tr_x + tr_y;  // 3*16 + 16 = 64
    EXPECT_NEAR(mpo_trace(y).real(), tr_expected, 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. SETTN unit tests
// ─────────────────────────────────────────────────────────────────────────────

// β→0 high-T limit: ρ ≈ I, lnZ ≈ L·log(d)
TEST(SETTNUnit, HighTemperatureLimit) {
    const int L = 4;
    auto H = make_xx_mpo(L);
    SETTNConfig cfg;
    cfg.max_order = 3;
    cfg.trunc = TruncParams{8, 1e-12, false};
    auto result = settn(H, /*beta=*/0.01, cfg);
    double lnZ_expected = L * std::log(2.0);
    ASSERT_FALSE(result.lnZ_values.empty());
    EXPECT_NEAR(result.lnZ_values.back(), lnZ_expected,
                std::abs(lnZ_expected) * 0.05);
}

// First-order truncation: coeff_1 = -β, Tr(H) = 0 → lnZ unchanged from order 0
TEST(SETTNUnit, FirstOrderXXChain) {
    const int L = 4;
    auto H = make_xx_mpo(L);
    SETTNConfig cfg;
    cfg.max_order = 1;
    cfg.trunc = TruncParams{32, 1e-12, false};
    auto result = settn(H, 0.1, cfg);
    // At n=1: ρ = I + (-0.1)*H. Tr(I + (-0.1)*H) = d^L since Tr(H)=0
    double lnZ_expected = L * std::log(2.0);
    ASSERT_GE((int)result.lnZ_values.size(), 2);
    EXPECT_NEAR(result.lnZ_values[1], lnZ_expected,
                std::abs(lnZ_expected) * 0.05);
    // lnZ > 0  for d=2, L=4
    EXPECT_GT(result.lnZ_values.back(), 0.0);
}

// converged_order is set when tol is loose
TEST(SETTNUnit, ConvergenceDetected) {
    const int L = 4;
    auto H = make_xx_mpo(L);
    SETTNConfig cfg;
    cfg.max_order = 20;
    cfg.tol = 1e-2;   // loose tolerance, should converge quickly at high T
    cfg.trunc = TruncParams{8, 1e-12, false};
    auto result = settn(H, 0.01, cfg);
    // With very small β, series converges quickly
    EXPECT_GE(result.converged_order, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. FreeFermionValidation — core integration tests
// ─────────────────────────────────────────────────────────────────────────────

// lnZ vs exact: L=4 XX chain, β ∈ {0.1, 0.5, 1.0, 2.0}, tolerance 1%
TEST(FreeFermionValidation, LnZ_L4) {
    const int L = 4;
    auto H   = make_xx_mpo(L);
    auto eps = xx_chain_energies(L);

    SETTNConfig cfg;
    cfg.max_order = 12;
    cfg.trunc = TruncParams{32, 1e-12, false};

    for (double beta : {0.1, 0.5, 1.0, 2.0}) {
        double lnZ_exact = grand_canonical_lnZ(eps, beta);
        auto result = settn(H, beta, cfg);
        ASSERT_FALSE(result.lnZ_values.empty())
            << "No lnZ values computed for beta=" << beta;
        EXPECT_NEAR(result.lnZ_values.back(), lnZ_exact,
                    std::abs(lnZ_exact) * 0.01)
            << "beta=" << beta
            << " lnZ_SETTN=" << result.lnZ_values.back()
            << " lnZ_exact=" << lnZ_exact;
    }
}

// lnZ for L=2 (smallest non-trivial case)
TEST(FreeFermionValidation, LnZ_L2) {
    const int L = 2;
    auto H   = make_xx_mpo(L);
    auto eps = xx_chain_energies(L);

    SETTNConfig cfg;
    cfg.max_order = 10;
    cfg.trunc = TruncParams{16, 1e-12, false};

    for (double beta : {0.5, 1.0, 2.0}) {
        double lnZ_exact = grand_canonical_lnZ(eps, beta);
        auto result = settn(H, beta, cfg);
        EXPECT_NEAR(result.lnZ_values.back(), lnZ_exact,
                    std::abs(lnZ_exact) * 0.01)
            << "L=" << L << " beta=" << beta;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. DISABLED tests (require additional infrastructure)
// ─────────────────────────────────────────────────────────────────────────────

// Needs energy computation Tr(H·ρ)/Tr(ρ) — not yet implemented
TEST(DISABLED_FreeFermionValidation, Energy_L6) {
    FAIL() << "energy computation not yet implemented";
}

// Needs exact-diagonalization reference value for Heisenberg at β=0.5
TEST(DISABLED_FreeFermionValidation, LnZ_Heisenberg_L4_Beta05) {
    FAIL() << "Heisenberg ED reference not yet available";
}

// Needs multi-β runner (run settn at several β values, check monotonicity)
TEST(DISABLED_SETTNUnit, LnZMonotoneInBeta) {
    FAIL() << "multi-beta runner not implemented";
}

// ── 2D SETTN integration tests ────────────────────────────────────────────────

// SETTN on a 2×4 YC Heisenberg cylinder at high T.
// At β→0, ρ→I regardless of H, so lnZ → L·log(d) = 8·log(2) ≈ 5.545.
// This test exercises SETTN with a long-range MPO (periodic Y bonds).
TEST(SETTNIntegration, YCHeisenberg2x4_HighT)
{
    const int Lx = 2, Ly = 4;
    const int L  = Lx * Ly;   // 8 sites

    auto H = make_yc_heisenberg_mpo(Lx, Ly);

    SETTNConfig cfg;
    cfg.max_order = 6;
    cfg.tol       = 1e-6;
    cfg.trunc     = TruncParams{32, 1e-12, false};

    const double beta = 0.01;   // high T → ρ ≈ I
    auto result = settn(H, beta, cfg);

    const double lnZ_expected = L * std::log(2.0);   // log(d^L) = 8·log(2)
    ASSERT_FALSE(result.lnZ_values.empty());
    EXPECT_NEAR(result.lnZ_values.back(), lnZ_expected,
                std::abs(lnZ_expected) * 0.02)
        << "YC2x4 high-T lnZ: got=" << result.lnZ_values.back()
        << " expected=" << lnZ_expected;
}

// lnZ of the Heisenberg AFM is increasing in β.
// d(lnZ)/dβ = -⟨E⟩ > 0 since E_0 < 0 for the AFM.
// We verify: lnZ(β₂) > lnZ(β₁) for β₂ > β₁ > 0.
// Low β values and modest max_order keep the test fast.
TEST(SETTNIntegration, YCHeisenberg2x4_LnZIncreasesWithBeta)
{
    const int Lx = 2, Ly = 4;

    auto H1 = make_yc_heisenberg_mpo(Lx, Ly);
    auto H2 = make_yc_heisenberg_mpo(Lx, Ly);

    SETTNConfig cfg;
    cfg.max_order = 5;
    cfg.tol       = 1e-4;
    cfg.trunc     = TruncParams{20, 1e-12, false};

    auto r_lo = settn(H1, 0.05, cfg);
    auto r_hi = settn(H2, 0.15, cfg);

    ASSERT_FALSE(r_lo.lnZ_values.empty());
    ASSERT_FALSE(r_hi.lnZ_values.empty());

    double lnZ_lo = r_lo.lnZ_values.back();
    double lnZ_hi = r_hi.lnZ_values.back();

    EXPECT_GT(lnZ_hi, lnZ_lo)
        << "lnZ should increase with β for Heisenberg AFM (E_0 < 0):\n"
        << "  lnZ(β=0.05)=" << lnZ_lo << "  lnZ(β=0.15)=" << lnZ_hi;
}

} // namespace tenet::test
