// tests/test_local_space.cpp
//
// Tests for the pre-built local operator matrices in spin.hpp and fermion.hpp.
// All operators are fully inline (Eigen expressions), so every test runs
// without any compiled .cpp body.
//
// Physics covered:
//   Spin-1/2 : SU(2) algebra, S^2 = 3/4 · I, correct dimensions
//   Spin-1   : SU(2) algebra, S^2 = 2 · I
//   Fermion  : CAR algebra, nilpotency, number-operator idempotency

#include <gtest/gtest.h>
#include "tenet/local_space/spin.hpp"
#include "tenet/local_space/fermion.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <complex>

namespace tenet::test {

// ── helpers ────────────────────────────────────────────────────────────────

static constexpr double kEps = 1e-12;

// Check that a complex matrix is zero within tolerance.
static void expect_zero(const Eigen::MatrixXcd& m, double tol = kEps) {
    EXPECT_LE(m.cwiseAbs().maxCoeff(), tol)
        << "Matrix is not zero (max |coeff| = " << m.cwiseAbs().maxCoeff() << ")";
}

// Check that two complex matrices are equal within tolerance.
static void expect_eq(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b,
                      double tol = kEps) {
    expect_zero(a - b, tol);
}

// ── spin::half ─────────────────────────────────────────────────────────────

TEST(SpinHalf, Dimensions) {
    EXPECT_EQ(spin::half::Sz().rows(), 2);
    EXPECT_EQ(spin::half::Sz().cols(), 2);
    EXPECT_EQ(spin::half::Sp().rows(), 2);
    EXPECT_EQ(spin::half::Sp().cols(), 2);
    EXPECT_EQ(spin::half::Sm().rows(), 2);
    EXPECT_EQ(spin::half::Sm().cols(), 2);
    EXPECT_EQ(spin::half::Id().rows(), 2);
    EXPECT_EQ(spin::half::Id().cols(), 2);
}

TEST(SpinHalf, IdIsIdentity) {
    expect_eq(spin::half::Id(), Eigen::MatrixXcd::Identity(2, 2));
}

TEST(SpinHalf, SzDiagonal) {
    auto Sz = spin::half::Sz();
    EXPECT_NEAR(Sz(0, 0).real(),  0.5, kEps);
    EXPECT_NEAR(Sz(1, 1).real(), -0.5, kEps);
    EXPECT_NEAR(Sz(0, 1).real(),  0.0, kEps);
    EXPECT_NEAR(Sz(1, 0).real(),  0.0, kEps);
}

TEST(SpinHalf, SpIsRaisingOperator) {
    // Sp|↓⟩ = |↑⟩  ↔  Sp * e_1 = e_0  (column indices: 0=↑, 1=↓)
    Eigen::Vector2cd down; down << 0.0, 1.0;
    Eigen::Vector2cd up;   up   << 1.0, 0.0;
    expect_eq((spin::half::Sp() * down).eval(), up);
}

TEST(SpinHalf, SmIsLoweringOperator) {
    Eigen::Vector2cd up;   up   << 1.0, 0.0;
    Eigen::Vector2cd down; down << 0.0, 1.0;
    expect_eq((spin::half::Sm() * up).eval(), down);
}

TEST(SpinHalf, SmIsAdjointOfSp) {
    expect_eq(spin::half::Sm(), spin::half::Sp().adjoint());
}

// [Sz, Sp] = Sp
TEST(SpinHalf, CommutatorSzSp) {
    auto Sz = spin::half::Sz();
    auto Sp = spin::half::Sp();
    expect_eq((Sz * Sp - Sp * Sz).eval(), Sp);
}

// [Sz, Sm] = -Sm
TEST(SpinHalf, CommutatorSzSm) {
    auto Sz = spin::half::Sz();
    auto Sm = spin::half::Sm();
    expect_eq((Sz * Sm - Sm * Sz).eval(), -Sm);
}

// [Sp, Sm] = 2 Sz
TEST(SpinHalf, CommutatorSpSm) {
    auto Sp = spin::half::Sp();
    auto Sm = spin::half::Sm();
    auto Sz = spin::half::Sz();
    expect_eq((Sp * Sm - Sm * Sp).eval(), 2.0 * Sz);
}

// S^2 = Sz^2 + (Sp*Sm + Sm*Sp)/2 = s(s+1) I = 3/4 · I  for s=1/2
TEST(SpinHalf, TotalSpinSquared) {
    auto Sz = spin::half::Sz();
    auto Sp = spin::half::Sp();
    auto Sm = spin::half::Sm();
    auto S2 = (Sz * Sz + 0.5 * (Sp * Sm + Sm * Sp)).eval();
    expect_eq(S2, 0.75 * Eigen::MatrixXcd::Identity(2, 2));
}

// ── spin::one ──────────────────────────────────────────────────────────────

TEST(SpinOne, Dimensions) {
    EXPECT_EQ(spin::one::Sz().rows(), 3);
    EXPECT_EQ(spin::one::Sp().rows(), 3);
    EXPECT_EQ(spin::one::Sm().rows(), 3);
    EXPECT_EQ(spin::one::Id().rows(), 3);
}

TEST(SpinOne, IdIsIdentity) {
    expect_eq(spin::one::Id(), Eigen::MatrixXcd::Identity(3, 3));
}

TEST(SpinOne, SzEigenvalues) {
    auto Sz = spin::one::Sz();
    EXPECT_NEAR(Sz(0, 0).real(),  1.0, kEps);   // m=+1
    EXPECT_NEAR(Sz(1, 1).real(),  0.0, kEps);   // m=0
    EXPECT_NEAR(Sz(2, 2).real(), -1.0, kEps);   // m=-1
}

TEST(SpinOne, SmIsAdjointOfSp) {
    expect_eq(spin::one::Sm(), spin::one::Sp().adjoint());
}

// [Sz, Sp] = Sp
TEST(SpinOne, CommutatorSzSp) {
    auto Sz = spin::one::Sz();
    auto Sp = spin::one::Sp();
    expect_eq((Sz * Sp - Sp * Sz).eval(), Sp);
}

// [Sz, Sm] = -Sm
TEST(SpinOne, CommutatorSzSm) {
    auto Sz = spin::one::Sz();
    auto Sm = spin::one::Sm();
    expect_eq((Sz * Sm - Sm * Sz).eval(), -Sm);
}

// [Sp, Sm] = 2 Sz
TEST(SpinOne, CommutatorSpSm) {
    auto Sp = spin::one::Sp();
    auto Sm = spin::one::Sm();
    auto Sz = spin::one::Sz();
    expect_eq((Sp * Sm - Sm * Sp).eval(), 2.0 * Sz);
}

// S^2 = s(s+1)·I = 2·I  for s=1
TEST(SpinOne, TotalSpinSquared) {
    auto Sz = spin::one::Sz();
    auto Sp = spin::one::Sp();
    auto Sm = spin::one::Sm();
    auto S2 = (Sz * Sz + 0.5 * (Sp * Sm + Sm * Sp)).eval();
    expect_eq(S2, 2.0 * Eigen::MatrixXcd::Identity(3, 3));
}

// Sp raises: Sp|m=-1⟩ = sqrt(2)|m=0⟩, Sp|m=0⟩ = sqrt(2)|m=+1⟩
TEST(SpinOne, SpRaisesCorrectly) {
    auto Sp = spin::one::Sp();
    Eigen::Vector3cd m_minus; m_minus << 0, 0, 1;
    Eigen::Vector3cd m_zero;  m_zero  << 0, 1, 0;
    Eigen::Vector3cd m_plus;  m_plus  << 1, 0, 0;

    expect_eq((Sp * m_minus).eval(), std::sqrt(2.0) * m_zero);
    expect_eq((Sp * m_zero).eval(),  std::sqrt(2.0) * m_plus);
}

// ── fermion::spinless ──────────────────────────────────────────────────────

TEST(FermionSpinless, Dimensions) {
    EXPECT_EQ(fermion::spinless::c().rows(), 2);
    EXPECT_EQ(fermion::spinless::c().cols(), 2);
    EXPECT_EQ(fermion::spinless::cdag().rows(), 2);
    EXPECT_EQ(fermion::spinless::n().rows(), 2);
    EXPECT_EQ(fermion::spinless::Z().rows(), 2);
    EXPECT_EQ(fermion::spinless::Id().rows(), 2);
}

TEST(FermionSpinless, IdIsIdentity) {
    expect_eq(fermion::spinless::Id(), Eigen::MatrixXcd::Identity(2, 2));
}

TEST(FermionSpinless, CdagIsAdjointOfC) {
    expect_eq(fermion::spinless::cdag(), fermion::spinless::c().adjoint());
}

// CAR: c · c† + c† · c = I
TEST(FermionSpinless, AntiCommutatorCCdag) {
    auto c    = fermion::spinless::c();
    auto cdag = fermion::spinless::cdag();
    expect_eq((c * cdag + cdag * c).eval(), Eigen::MatrixXcd::Identity(2, 2));
}

// Nilpotency: c² = 0,  c†² = 0
TEST(FermionSpinless, CNilpotent) {
    auto c = fermion::spinless::c();
    expect_zero((c * c).eval());
}

TEST(FermionSpinless, CdagNilpotent) {
    auto cdag = fermion::spinless::cdag();
    expect_zero((cdag * cdag).eval());
}

// Number operator idempotency: n² = n
TEST(FermionSpinless, NumberOpIdempotent) {
    auto n = fermion::spinless::n();
    expect_eq((n * n).eval(), n);
}

// n = c† c  (verify construction)
TEST(FermionSpinless, NumberOpDefinition) {
    auto n    = fermion::spinless::n();
    auto cdag = fermion::spinless::cdag();
    auto c    = fermion::spinless::c();
    expect_eq(n, (cdag * c).eval());
}

// Jordan-Wigner string: Z² = I,  Z = (-1)^n  → Z = I - 2n
TEST(FermionSpinless, ZWilsonLoop) {
    auto Z = fermion::spinless::Z();
    expect_eq((Z * Z).eval(), Eigen::MatrixXcd::Identity(2, 2));
}

TEST(FermionSpinless, ZIsMinusOneToN) {
    auto Z = fermion::spinless::Z();
    auto n = fermion::spinless::n();
    auto I = Eigen::MatrixXcd::Identity(2, 2);
    // Z = I - 2n
    expect_eq(Z, (I - 2.0 * n).eval());
}

// Z anti-commutes with c: Z c + c Z = 0
TEST(FermionSpinless, ZAnticommutesWithC) {
    auto Z = fermion::spinless::Z();
    auto c = fermion::spinless::c();
    expect_zero((Z * c + c * Z).eval());
}

} // namespace tenet::test
