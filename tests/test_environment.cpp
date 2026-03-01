// tests/test_environment.cpp
//
// Tests for:
//   - SparseLeftEnvTensor / SparseRightEnvTensor (inline methods)
//   - LeftEnvTensor / RightEnvTensor construction from a DenseTensor
//   - Environment build/push (DISABLED – requires implementation)

#include <gtest/gtest.h>

#include "tenet/environment/env_tensor.hpp"
#include "tenet/environment/environment.hpp"

using namespace tenet;

// ── SparseLeftEnvTensor ───────────────────────────────────────────────────────

TEST(SparseLeftEnvTensor, DimAfterConstruct) {
    SparseLeftEnvTensor<> L(5);
    EXPECT_EQ(L.dim(), 5);
}

TEST(SparseLeftEnvTensor, AllNullAfterConstruct) {
    SparseLeftEnvTensor<> L(4);
    for (int i = 0; i < 4; ++i)
        EXPECT_FALSE(L.has(i)) << "Index " << i << " should be null";
}

TEST(SparseLeftEnvTensor, NullPtrOnEmptySlot) {
    SparseLeftEnvTensor<> L(3);
    EXPECT_EQ(L[0], nullptr);
    EXPECT_EQ(L[1], nullptr);
    EXPECT_EQ(L[2], nullptr);
}

TEST(SparseLeftEnvTensor, SetMakesHasTrue) {
    SparseLeftEnvTensor<> L(3);
    // Wrap a default DenseTensor in a LeftEnvTensor and store it
    L.set(1, std::make_unique<LeftEnvTensor<>>(DenseTensor{}));
    EXPECT_FALSE(L.has(0));
    EXPECT_TRUE( L.has(1));
    EXPECT_FALSE(L.has(2));
}

TEST(SparseLeftEnvTensor, NonConstAccessAfterSet) {
    SparseLeftEnvTensor<> L(2);
    L.set(0, std::make_unique<LeftEnvTensor<>>(DenseTensor{}));
    EXPECT_NE(L[0], nullptr);
    EXPECT_EQ(L[1], nullptr);
}

TEST(SparseLeftEnvTensor, ConstAccessAfterSet) {
    SparseLeftEnvTensor<> L(2);
    L.set(0, std::make_unique<LeftEnvTensor<>>(DenseTensor{}));
    const auto& cL = L;
    EXPECT_NE(cL[0], nullptr);
    EXPECT_EQ(cL[1], nullptr);
}

TEST(SparseLeftEnvTensor, DimOne) {
    SparseLeftEnvTensor<> L(1);
    EXPECT_EQ(L.dim(), 1);
    EXPECT_FALSE(L.has(0));
}

// ── SparseRightEnvTensor ──────────────────────────────────────────────────────

TEST(SparseRightEnvTensor, DimAfterConstruct) {
    SparseRightEnvTensor<> R(6);
    EXPECT_EQ(R.dim(), 6);
}

TEST(SparseRightEnvTensor, AllNullAfterConstruct) {
    SparseRightEnvTensor<> R(4);
    for (int i = 0; i < 4; ++i)
        EXPECT_FALSE(R.has(i));
}

TEST(SparseRightEnvTensor, SetMakesHasTrue) {
    SparseRightEnvTensor<> R(3);
    R.set(2, std::make_unique<RightEnvTensor<>>(DenseTensor{}));
    EXPECT_FALSE(R.has(0));
    EXPECT_FALSE(R.has(1));
    EXPECT_TRUE( R.has(2));
}

// ── LeftEnvTensor / RightEnvTensor construction ───────────────────────────────

TEST(LeftEnvTensor, ConstructFromDenseTensor) {
    LeftEnvTensor<> L(DenseTensor{});
    EXPECT_EQ(L.data().rank(), 0);
}

TEST(RightEnvTensor, ConstructFromDenseTensor) {
    RightEnvTensor<> R(DenseTensor{});
    EXPECT_EQ(R.data().rank(), 0);
}

TEST(LeftEnvTensor, MutableDataAccess) {
    LeftEnvTensor<> L(DenseTensor{});
    L.data() = DenseTensor{};   // assign should compile
    SUCCEED();
}

// ── Tests requiring Phase-1 implementation ───────────────────────────────────

TEST(Environment, DISABLED_PushRightConsistency) {
    // After push_right(site), left_env(site+1) must equal the contraction
    // of left_env(site) ⊗ psi[site] ⊗ H[site] ⊗ conj(psi[site]).
    GTEST_SKIP() << "Needs Environment::push_right() implementation";
}

TEST(Environment, DISABLED_PushLeftConsistency) {
    GTEST_SKIP() << "Needs Environment::push_left() implementation";
}

TEST(Environment, DISABLED_BuildAll_HeisenbergL4) {
    // For a L=4 Heisenberg chain with exact MPS, verify left_env(L) is a scalar.
    GTEST_SKIP() << "Needs Environment::build_all() implementation";
}
