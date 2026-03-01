// tests/test_mps.cpp
//
// Tests for DenseMPS<> and MPSTensor<> (inline methods).
// Construction, center management, operator[], Adjoint wrapper.

#include <gtest/gtest.h>

#include "tenet/mps/mps.hpp"
#include "tenet/mps/canonical.hpp"

using namespace tenet;

// ── MPSTensor<> ───────────────────────────────────────────────────────────────

TEST(MPSTensor, DefaultConstruct) {
    MPSTensor<> t;
    // default-constructed data is an empty DenseTensor (rank 0)
    EXPECT_EQ(t.data().rank(), 0);
}

// ── DenseMPS<> construction & metadata ───────────────────────────────────────

TEST(DenseMPS, LengthAfterConstruct) {
    DenseMPS<> psi(10);
    EXPECT_EQ(psi.length(), 10);
}

TEST(DenseMPS, CenterDefaultsToZero) {
    DenseMPS<> psi(8);
    EXPECT_EQ(psi.center_left(),  0);
    EXPECT_EQ(psi.center_right(), 0);
}

TEST(DenseMPS, SetCenter) {
    DenseMPS<> psi(10);
    psi.set_center(4, 4);
    EXPECT_EQ(psi.center_left(),  4);
    EXPECT_EQ(psi.center_right(), 4);
}

TEST(DenseMPS, IsCanonicalWhenCentersMatch) {
    DenseMPS<> psi(6);
    psi.set_center(3, 3);
    EXPECT_TRUE(psi.is_canonical());
}

TEST(DenseMPS, NotCanonicalWhenCentersMismatch) {
    DenseMPS<> psi(6);
    psi.set_center(2, 4);
    EXPECT_FALSE(psi.is_canonical());
}

TEST(DenseMPS, SiteAccessNoCrash) {
    DenseMPS<> psi(5);
    for (int i = 0; i < 5; ++i) {
        // operator[] must be callable; returns reference to default MPSTensor
        EXPECT_EQ(psi[i].data().rank(), 0);
    }
}

TEST(DenseMPS, ConstSiteAccess) {
    const DenseMPS<> psi(3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(psi[i].data().rank(), 0);
    }
}

TEST(DenseMPS, LengthOne) {
    DenseMPS<> psi(1);
    EXPECT_EQ(psi.length(), 1);
    EXPECT_EQ(psi[0].data().rank(), 0);
}

// ── Adjoint wrapper ───────────────────────────────────────────────────────────

TEST(AdjointMPS, LengthMatchesOriginal) {
    DenseMPS<> psi(7);
    auto psi_dag = psi.adjoint();
    EXPECT_EQ(psi_dag.length(), 7);
}

TEST(AdjointMPS, UnadjointReturnsOriginal) {
    DenseMPS<> psi(5);
    psi.set_center(2, 2);
    auto psi_dag = psi.adjoint();
    EXPECT_EQ(psi_dag.unadjoint().center_left(), 2);
}

// ── Tests requiring Phase-1 implementation ───────────────────────────────────

TEST(DenseMPS, DISABLED_RandomInit) {
    GTEST_SKIP() << "Needs DenseMPS::random() + max_bond_dim() implementation";
    // When enabled: random(10, 32, phys) → length=10, max_bond_dim=32
}

TEST(DenseMPS, DISABLED_NormAfterNormalize) {
    GTEST_SKIP() << "Needs DenseMPS::norm() + normalize() implementation";
}

TEST(DenseMPS, DISABLED_InnerProductSelfNorm) {
    // ⟨ψ|ψ⟩ == norm²
    GTEST_SKIP() << "Needs DenseMPS::inner() implementation";
}

TEST(DenseMPS, DISABLED_CanonicalOrthogonality) {
    // After left_canonicalize(0, site), Q†Q == I for each site < split.
    GTEST_SKIP() << "Needs left_canonicalize() implementation";
}

TEST(DenseMPS, DISABLED_MoveCenterPreservesNorm) {
    GTEST_SKIP() << "Needs move_center() implementation";
}
