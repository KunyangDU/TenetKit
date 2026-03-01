// tests/test_factorization.cpp
//
// Tests for factorization config structs (fully aggregate-initialized)
// and DISABLED_ tests for QR/SVD/eigh that need implementation.

#include <gtest/gtest.h>

#include "tenet/core/factorization.hpp"

using namespace tenet;

// ── TruncParams defaults ──────────────────────────────────────────────────────

TEST(TruncParams, DefaultMaxD) {
    TruncParams p;
    EXPECT_EQ(p.maxD, 0);     // 0 = unlimited
}

TEST(TruncParams, DefaultCutoff) {
    TruncParams p;
    EXPECT_DOUBLE_EQ(p.cutoff, 1e-12);
}

TEST(TruncParams, DefaultNormalize) {
    TruncParams p;
    EXPECT_FALSE(p.normalize);
}

TEST(TruncParams, CustomValues) {
    TruncParams p{.maxD = 64, .cutoff = 1e-8, .normalize = true};
    EXPECT_EQ(p.maxD, 64);
    EXPECT_DOUBLE_EQ(p.cutoff, 1e-8);
    EXPECT_TRUE(p.normalize);
}

// ── SVDResult, QRResult, EigenResult struct layout ────────────────────────────

TEST(SVDResult, DefaultBondDim) {
    SVDResult r;
    EXPECT_EQ(r.bond_dim,      0);
    EXPECT_DOUBLE_EQ(r.truncation_err, 0.0);
}

TEST(QRResult, DefaultBondDim) {
    QRResult r;
    EXPECT_EQ(r.bond_dim, 0);
}

// ── von_neumann_entropy (DISABLED – needs implementation) ─────────────────────

TEST(Factorization, DISABLED_QRLeftOrthogonality) {
    // For a random matrix A = Q*R, verify Q†Q = I
    GTEST_SKIP() << "Needs qr() implementation";
}

TEST(Factorization, DISABLED_QRRightOrthogonality) {
    GTEST_SKIP() << "Needs rq() implementation";
}

TEST(Factorization, DISABLED_SVDRoundtrip) {
    // ||A - U * diag(S) * Vt|| < 1e-12 for a random tensor
    GTEST_SKIP() << "Needs svd() implementation";
}

TEST(Factorization, DISABLED_SVDSingularValuesDescending) {
    GTEST_SKIP() << "Needs svd() implementation";
}

TEST(Factorization, DISABLED_SVDTruncation) {
    // With maxD=2, only 2 singular values are kept; truncation_err > 0
    GTEST_SKIP() << "Needs svd() implementation";
}

TEST(Factorization, DISABLED_RandSVDAccuracy) {
    // ||A - U_rand * diag(S_rand) * Vt_rand|| < tol for low-rank A
    GTEST_SKIP() << "Needs rand_svd() implementation";
}

TEST(Factorization, DISABLED_EighHermitian) {
    // Eigenvalues of a 4×4 Hermitian matrix match std::complex Eigen solver
    GTEST_SKIP() << "Needs eigh() implementation";
}

TEST(Factorization, DISABLED_VonNeumannEntropy) {
    // For singular values {1/sqrt(2), 1/sqrt(2)}: entropy = log(2)
    GTEST_SKIP() << "Needs von_neumann_entropy() implementation";
}

TEST(Factorization, DISABLED_VonNeumannEntropyProductState) {
    // For singular values {1, 0, …}: entropy = 0
    GTEST_SKIP() << "Needs von_neumann_entropy() implementation";
}
