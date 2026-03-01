// tests/test_lanczos.cpp
//
// Tests for Krylov solver configuration structs (inline/aggregate) and result
// types.  Actual solver correctness requires lanczos.cpp / arnoldi.cpp which
// are not yet implemented — those tests carry the DISABLED_ prefix so they
// compile and register but are skipped automatically.

#include <gtest/gtest.h>
#include "tenet/krylov/lanczos.hpp"
#include "tenet/krylov/arnoldi.hpp"

namespace tenet::test {

// ── LanczosConfig defaults ─────────────────────────────────────────────────

TEST(LanczosConfig, DefaultKrylovDim) {
    LanczosConfig cfg;
    EXPECT_EQ(cfg.krylov_dim, 8);
}

TEST(LanczosConfig, DefaultMaxIter) {
    LanczosConfig cfg;
    EXPECT_EQ(cfg.max_iter, 1);
}

TEST(LanczosConfig, DefaultTol) {
    LanczosConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.tol, 1e-6);
}

TEST(LanczosConfig, DefaultEager) {
    LanczosConfig cfg;
    EXPECT_TRUE(cfg.eager);
}

TEST(LanczosConfig, CustomValues) {
    LanczosConfig cfg{/*.krylov_dim=*/16, /*.max_iter=*/5,
                      /*.tol=*/1e-10, /*.eager=*/false};
    EXPECT_EQ(cfg.krylov_dim, 16);
    EXPECT_EQ(cfg.max_iter, 5);
    EXPECT_DOUBLE_EQ(cfg.tol, 1e-10);
    EXPECT_FALSE(cfg.eager);
}

// ── EigPair ────────────────────────────────────────────────────────────────

TEST(EigPair, DefaultConstruct) {
    // EigPair has no user-provided constructor; DenseTensor is default-constructible.
    EigPair p{};
    EXPECT_EQ(p.eigenvalue, 0.0);
    EXPECT_EQ(p.eigenvector.rank(), 0);
}

// ── LinSolveResult defaults ────────────────────────────────────────────────

TEST(LinSolveResult, DefaultConvergedFalse) {
    LinSolveResult r;
    EXPECT_FALSE(r.converged);
}

TEST(LinSolveResult, DefaultIterations) {
    LinSolveResult r;
    EXPECT_EQ(r.iterations, 0);
}

TEST(LinSolveResult, DefaultResidual) {
    LinSolveResult r;
    EXPECT_DOUBLE_EQ(r.residual, 0.0);
}

// ── ArnoldiConfig defaults ─────────────────────────────────────────────────

TEST(ArnoldiConfig, DefaultKrylovDim) {
    ArnoldiConfig cfg;
    EXPECT_EQ(cfg.krylov_dim, 32);
}

TEST(ArnoldiConfig, DefaultMaxIter) {
    ArnoldiConfig cfg;
    EXPECT_EQ(cfg.max_iter, 1);
}

TEST(ArnoldiConfig, DefaultTol) {
    ArnoldiConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.tol, 1e-8);
}

TEST(ArnoldiConfig, CustomValues) {
    ArnoldiConfig cfg{/*.krylov_dim=*/64, /*.max_iter=*/3, /*.tol=*/1e-12};
    EXPECT_EQ(cfg.krylov_dim, 64);
    EXPECT_EQ(cfg.max_iter, 3);
    EXPECT_DOUBLE_EQ(cfg.tol, 1e-12);
}

// ── DISABLED_: requires lanczos.cpp / arnoldi.cpp implementation ───────────

// Verify that lanczos_eigs finds the correct lowest eigenvalue of the 2×2
// symmetric matrix [[2,1],[1,2]] → eigenvalues {1, 3} → lowest = 1.
TEST(DISABLED_LanczosEigs, TwoByTwoMatrix) {
    // matvec: [[2,1],[1,2]] applied to a rank-1 DenseTensor
    FAIL() << "requires lanczos.cpp implementation";
}

// For a diagonal 4×4 matrix with known spectrum, Lanczos should recover
// all four eigenvalues in at most 4 Krylov steps.
TEST(DISABLED_LanczosEigs, ConvergenceDiagonal) {
    FAIL() << "requires lanczos.cpp implementation";
}

// LinSolvResult should converge for a simple diagonally-dominant system.
TEST(DISABLED_LanczosLinSolve, SimpleDiagonal) {
    FAIL() << "requires lanczos.cpp implementation";
}

// exp(α H) v for a 2×2 Hermitian H should match the direct matrix-exp.
TEST(DISABLED_ArnoldiExpm, TwoByTwo) {
    FAIL() << "requires arnoldi.cpp implementation";
}

} // namespace tenet::test
