// tests/test_mpo.cpp
//
// Tests for SparseMPOTensor<> and SparseMPO<> (inline methods).

#include <gtest/gtest.h>

#include "tenet/mps/mpo.hpp"

using namespace tenet;

// ── SparseMPOTensor<> ─────────────────────────────────────────────────────────

TEST(SparseMPOTensor, Dimensions) {
    SparseMPOTensor<> t(3, 4);
    EXPECT_EQ(t.d_in(),  3);
    EXPECT_EQ(t.d_out(), 4);
}

TEST(SparseMPOTensor, AllEntriesNullAfterConstruct) {
    SparseMPOTensor<> t(3, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_FALSE(t.has(i, j)) << "Entry (" << i << "," << j << ") should be null";
}

TEST(SparseMPOTensor, Trivial1x1) {
    SparseMPOTensor<> t(1, 1);
    EXPECT_EQ(t.d_in(),  1);
    EXPECT_EQ(t.d_out(), 1);
    EXPECT_FALSE(t.has(0, 0));
}

TEST(SparseMPOTensor, LargerDimensions) {
    SparseMPOTensor<> t(10, 10);
    EXPECT_EQ(t.d_in(),  10);
    EXPECT_EQ(t.d_out(), 10);
    for (int i = 0; i < 10; ++i)
        EXPECT_FALSE(t.has(i, i));
}

// ── SparseMPO<> ───────────────────────────────────────────────────────────────

TEST(SparseMPO, Length) {
    SparseMPO<> H(8);
    EXPECT_EQ(H.length(), 8);
}

TEST(SparseMPO, DefaultBondDim1x1) {
    SparseMPO<> H(5);
    for (int i = 0; i < 5; ++i) {
        auto [d_in, d_out] = H.bond_dim(i);
        EXPECT_EQ(d_in,  1);
        EXPECT_EQ(d_out, 1);
    }
}

TEST(SparseMPO, SiteAccessNoCrash) {
    SparseMPO<> H(4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(H[i].d_in(),  1);
        EXPECT_EQ(H[i].d_out(), 1);
    }
}

TEST(SparseMPO, ConstSiteAccess) {
    const SparseMPO<> H(3);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(H[i].has(0, 0));
    }
}

TEST(SparseMPO, LengthOne) {
    SparseMPO<> H(1);
    EXPECT_EQ(H.length(), 1);
    auto [d_in, d_out] = H.bond_dim(0);
    EXPECT_EQ(d_in,  1);
    EXPECT_EQ(d_out, 1);
}

// ── Tests requiring Phase-1 implementation ───────────────────────────────────

TEST(SparseMPOTensor, DISABLED_SetAndGetOperator) {
    GTEST_SKIP() << "Needs LocalOperator + SparseMPOTensor::set/operator() implementation";
}

TEST(SparseMPO, DISABLED_HeisenbergFromInteractionTree) {
    GTEST_SKIP() << "Needs InteractionTree compile() implementation";
}
