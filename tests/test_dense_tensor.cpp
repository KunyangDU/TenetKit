// tests/test_dense_tensor.cpp
//
// Tests for:
//   - TrivialSpace  (fully inline in space.hpp)
//   - DenseTensor   (inline accessors only; non-inline ops in DISABLED_ tests)
//   - Concept checks (compile-time static_assert)
//
// DISABLED_ prefix: tests that require .cpp implementation (link during
// Phase 1 development once the body is written).

#include <gtest/gtest.h>

#include "tenet/core/backend.hpp"
#include "tenet/core/dense_tensor.hpp"
#include "tenet/core/space.hpp"
#include "tenet/core/tensor_ops.hpp"

using namespace tenet;

// ── Compile-time concept checks ───────────────────────────────────────────────
static_assert(SpaceLike<TrivialSpace>,     "TrivialSpace must satisfy SpaceLike");
static_assert(TensorBackend<DenseBackend>, "DenseBackend must satisfy TensorBackend");
static_assert(std::same_as<DefaultBackend, DenseBackend>);

// ── TrivialSpace ──────────────────────────────────────────────────────────────

TEST(TrivialSpace, Dim) {
    EXPECT_EQ(TrivialSpace(1).dim(), 1);
    EXPECT_EQ(TrivialSpace(2).dim(), 2);
    EXPECT_EQ(TrivialSpace(100).dim(), 100);
}

TEST(TrivialSpace, NotDualByDefault) {
    EXPECT_FALSE(TrivialSpace(4).is_dual());
}

TEST(TrivialSpace, DualFlipsFlag) {
    TrivialSpace s(5);
    EXPECT_TRUE(s.dual().is_dual());
    EXPECT_EQ(s.dual().dim(), 5);
}

TEST(TrivialSpace, DualIsInvolution) {
    for (int d : {1, 2, 5, 10}) {
        TrivialSpace s(d);
        EXPECT_EQ(s.dual().dual(), s);
    }
}

TEST(TrivialSpace, TrivialFactory) {
    auto s = TrivialSpace::trivial(7);
    EXPECT_EQ(s.dim(), 7);
    EXPECT_FALSE(s.is_dual());
}

TEST(TrivialSpace, EqualitySameDim) {
    EXPECT_EQ(TrivialSpace(3), TrivialSpace(3));
    EXPECT_EQ(TrivialSpace(3).dual(), TrivialSpace(3).dual());
}

TEST(TrivialSpace, InequalityDifferentDim) {
    EXPECT_NE(TrivialSpace(3), TrivialSpace(4));
}

TEST(TrivialSpace, InequalityDualVsNonDual) {
    EXPECT_NE(TrivialSpace(3), TrivialSpace(3).dual());
}

// ── DenseTensor inline accessors ──────────────────────────────────────────────

TEST(DenseTensor, DefaultConstruct) {
    DenseTensor t;
    EXPECT_EQ(t.rank(), 0);
}

TEST(DenseTensor, DefaultSpacesEmpty) {
    DenseTensor t;
    EXPECT_TRUE(t.spaces().empty());
    EXPECT_EQ(t.spaces().size(), 0u);
}

TEST(DenseTensor, DefaultDataNoCrash) {
    // data() on an empty tensor must not segfault
    const DenseTensor t;
    (void)t.data();
}

// ── DenseBackend type aliases ─────────────────────────────────────────────────

TEST(DenseBackend, TypeAliases) {
    static_assert(std::same_as<DenseBackend::Space,  TrivialSpace>);
    static_assert(std::same_as<DenseBackend::Tensor, DenseTensor>);
    static_assert(std::same_as<DenseBackend::Scalar, std::complex<double>>);
    SUCCEED();
}

// ── Tests requiring Phase-1 implementation (DISABLED until src/ is filled) ───

TEST(DenseTensor, DISABLED_ConstructFromSpaces) {
    GTEST_SKIP() << "Needs DenseTensor(vector<TrivialSpace>) + numel() implementation";
    // When enabled: t({d2,d3,d4}) → rank=3, numel=24
}

TEST(DenseTensor, DISABLED_PermuteConsistency) {
    // A.permute({1,0}) contracted with B should equal A contracted with B
    // (after adjusting indices). Verified by comparing against Eigen GEMM.
    GTEST_SKIP() << "Needs DenseTensor::permute() implementation";
}

TEST(DenseTensor, DISABLED_ContractCorrectness) {
    // C[i,k] = sum_j A[i,j] * B[j,k], compare against Eigen MatrixXcd multiply
    GTEST_SKIP() << "Needs contract() implementation";
}

TEST(DenseTensor, DISABLED_SVDRoundtrip) {
    // ||A - U*diag(S)*Vt|| < 1e-12
    GTEST_SKIP() << "Needs svd() implementation";
}

TEST(DenseTensor, DISABLED_NormAndNormalize) {
    GTEST_SKIP() << "Needs DenseTensor::norm() implementation";
}
