// tests/test_observables.cpp
//
// Unit tests for:
//   - add_obs / add_obs2  (ObservableTree building)
//   - cal_obs             (expectation value computation)
//
// Test strategy: use D=1 product states whose expectation values can be
// computed analytically, then compare with cal_obs output.
//
//   All-up state  |↑↑↑↑⟩  →  ⟨Sz_i⟩ = +0.5,  ⟨Sz_i Sz_j⟩ = +0.25
//   Néel state    |↑↓↑↓⟩  →  ⟨Sz_0⟩ = +0.5,  ⟨Sz_1⟩ = -0.5, …

#include <gtest/gtest.h>

#include "tenet/observables/add_observable.hpp"
#include "tenet/observables/cal_observable.hpp"
#include "tenet/observables/obs_node.hpp"
#include "tenet/environment/environment.hpp"
#include "tenet/mps/mps.hpp"
#include "tenet/mps/mpo.hpp"
#include "tenet/intr_tree/interaction_node.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/intr_tree/local_operator.hpp"
#include "tenet/local_space/spin.hpp"
#include "tenet/core/space.hpp"
#include "tenet/core/dense_tensor.hpp"

namespace tenet::test {

// ── Operator factory helpers ──────────────────────────────────────────────────
// add_obs/add_obs2 take unique_ptr<AbstractLocalOperator<B>>.
// unique_ptr<LocalOperator<B>> does not auto-deduce as that type, so we use
// factory functions that return the base-class unique_ptr explicitly.

static Op<> mkSz(int site) {
    return std::make_unique<LocalOperator<>>(spin::half::Sz(), "Sz", site);
}
static Op<> mkSp(int site) {
    return std::make_unique<LocalOperator<>>(spin::half::Sp(), "Sp", site);
}
static Op<> mkSm(int site) {
    return std::make_unique<LocalOperator<>>(spin::half::Sm(), "Sm", site);
}

// ── MPS / MPO helpers ─────────────────────────────────────────────────────────

// Build a spin-1/2 site tensor (D_l=1, d=2, D_r=1).
// up=true  → A[0,0,0]=1, A[0,1,0]=0   (σ=0 is spin-up per spin.hpp)
// up=false → A[0,0,0]=0, A[0,1,0]=1
static MPSTensor<> spin_tensor(bool up)
{
    DenseTensor::Scalar one{1.0, 0.0};
    DenseTensor::Scalar zero{0.0, 0.0};
    DenseTensor data({TrivialSpace(1), TrivialSpace(2), TrivialSpace(1)},
                     up ? std::vector<DenseTensor::Scalar>{one, zero}
                        : std::vector<DenseTensor::Scalar>{zero, one});
    MPSTensor<> t(TrivialSpace(1), TrivialSpace(2), TrivialSpace(1));
    t.data() = std::move(data);
    return t;
}

// Build a D=1 product state from a boolean spin pattern.
// pattern[i] = true → spin up at site i, false → spin down.
static DenseMPS<> product_state(std::vector<bool> pattern)
{
    const int L = static_cast<int>(pattern.size());
    DenseMPS<> psi(L);
    for (int i = 0; i < L; ++i)
        psi[i] = spin_tensor(pattern[i]);
    return psi;
}

// Build a minimal Heisenberg MPO (used only to satisfy Environment ctor;
// cal_obs does not access the Hamiltonian right environments).
static SparseMPO<> make_dummy_mpo(int L)
{
    InteractionTree<> tree(L);
    for (int i = 0; i < L - 1; ++i) {
        add_intr2(tree, mkSz(i), i, mkSz(i + 1), i + 1, 1.0);
    }
    return compile(tree);
}

// ── add_obs structural tests ──────────────────────────────────────────────────

TEST(AddObs, SingleSiteCreatesLeafUnderRoot)
{
    ObservableTree<> tree(4);
    add_obs(tree, mkSz(2), 2);

    ASSERT_EQ(tree.root->children.size(), 1u);
    const auto& node = *tree.root->children[0];
    EXPECT_EQ(node.op->site(), 2);
    EXPECT_EQ(node.op->name(), "Sz");
    ASSERT_TRUE(node.leaf.has_value());
    EXPECT_EQ(node.leaf->sites,    std::vector<int>{2});
    EXPECT_EQ(node.leaf->op_names, std::vector<std::string>{"Sz"});
    EXPECT_TRUE(node.children.empty());
}

TEST(AddObs, MultipleCallsAddSeparateChildren)
{
    ObservableTree<> tree(4);
    for (int i = 0; i < 4; ++i)
        add_obs(tree, mkSz(i), i);
    EXPECT_EQ(tree.root->children.size(), 4u);
}

// ── add_obs2 structural tests ─────────────────────────────────────────────────

TEST(AddObs2, TwoSitePathStructure)
{
    ObservableTree<> tree(4);
    add_obs2(tree, mkSz(0), 0, mkSz(2), 2);

    ASSERT_EQ(tree.root->children.size(), 1u);
    const auto& node1 = *tree.root->children[0];
    EXPECT_EQ(node1.op->site(), 0);
    EXPECT_FALSE(node1.leaf.has_value());  // intermediate node, not a leaf
    ASSERT_EQ(node1.children.size(), 1u);

    const auto& node2 = *node1.children[0];
    EXPECT_EQ(node2.op->site(), 2);
    ASSERT_TRUE(node2.leaf.has_value());
    EXPECT_EQ(node2.leaf->sites,    (std::vector<int>{0, 2}));
    EXPECT_EQ(node2.leaf->op_names, (std::vector<std::string>{"Sz", "Sz"}));
    EXPECT_TRUE(node2.children.empty());
}

// ── cal_obs: all-up product state |↑↑↑↑⟩ ────────────────────────────────────

class CalObsAllUp : public ::testing::Test {
protected:
    static constexpr int L = 4;
    DenseMPS<>    psi = product_state({true, true, true, true});
    SparseMPO<>   H   = make_dummy_mpo(L);
    Environment<> env{psi, H};
};

TEST_F(CalObsAllUp, SingleSiteAllSites)
{
    ObservableTree<> tree(L);
    for (int i = 0; i < L; ++i)
        add_obs(tree, mkSz(i), i);

    cal_obs(tree, env);

    for (const auto& child : tree.root->children) {
        ASSERT_TRUE(child->leaf.has_value());
        EXPECT_NEAR(child->leaf->value.real(), 0.5, 1e-12)
            << "site=" << child->op->site();
        EXPECT_NEAR(child->leaf->value.imag(), 0.0, 1e-12);
    }
}

TEST_F(CalObsAllUp, TwoSiteNearestNeighbor)
{
    ObservableTree<> tree(L);
    for (int i = 0; i < L - 1; ++i)
        add_obs2(tree, mkSz(i), i, mkSz(i + 1), i + 1);

    cal_obs(tree, env);

    for (const auto& child : tree.root->children) {
        const auto& leaf = child->children[0]->leaf;
        ASSERT_TRUE(leaf.has_value());
        EXPECT_NEAR(leaf->value.real(), 0.25, 1e-12);
        EXPECT_NEAR(leaf->value.imag(), 0.0,  1e-12);
    }
}

TEST_F(CalObsAllUp, TwoSiteNonAdjacent)
{
    // ⟨Sz@0 Sz@3⟩ = 0.25  (long-range; tests intermediate identity gap-filling)
    ObservableTree<> tree(L);
    add_obs2(tree, mkSz(0), 0, mkSz(3), 3);

    cal_obs(tree, env);

    const auto& leaf = tree.root->children[0]->children[0]->leaf;
    ASSERT_TRUE(leaf.has_value());
    EXPECT_NEAR(leaf->value.real(), 0.25, 1e-12);
}

// ── cal_obs: Néel state |↑↓↑↓⟩ ───────────────────────────────────────────────

class CalObsNeel : public ::testing::Test {
protected:
    static constexpr int L = 4;
    DenseMPS<>    psi = product_state({true, false, true, false});
    SparseMPO<>   H   = make_dummy_mpo(L);
    Environment<> env{psi, H};
};

TEST_F(CalObsNeel, SingleSiteExpectations)
{
    ObservableTree<> tree(L);
    for (int i = 0; i < L; ++i)
        add_obs(tree, mkSz(i), i);

    cal_obs(tree, env);

    // Even sites → spin up (+0.5), odd sites → spin down (-0.5)
    const std::vector<double> expected_by_site = {0.5, -0.5, 0.5, -0.5};
    for (const auto& child : tree.root->children) {
        const auto& leaf = child->leaf;
        ASSERT_TRUE(leaf.has_value());
        int site = child->op->site();
        EXPECT_NEAR(leaf->value.real(), expected_by_site[site], 1e-12)
            << "site=" << site;
        EXPECT_NEAR(leaf->value.imag(), 0.0, 1e-12);
    }
}

TEST_F(CalObsNeel, NearestNeighborAntiferromagnetic)
{
    // ⟨Sz@i Sz@{i+1}⟩ = -0.25  (antiparallel neighbors in Néel state)
    ObservableTree<> tree(L);
    for (int i = 0; i < L - 1; ++i)
        add_obs2(tree, mkSz(i), i, mkSz(i + 1), i + 1);

    cal_obs(tree, env);

    for (const auto& child : tree.root->children) {
        const auto& leaf = child->children[0]->leaf;
        ASSERT_TRUE(leaf.has_value());
        EXPECT_NEAR(leaf->value.real(), -0.25, 1e-12);
    }
}

TEST_F(CalObsNeel, NextNearestNeighborFerromagnetic)
{
    // ⟨Sz@0 Sz@2⟩ = +0.25  (both spin-up)
    // ⟨Sz@1 Sz@3⟩ = +0.25  (both spin-down: (-0.5)·(-0.5))
    ObservableTree<> tree(L);
    add_obs2(tree, mkSz(0), 0, mkSz(2), 2);
    add_obs2(tree, mkSz(1), 1, mkSz(3), 3);

    cal_obs(tree, env);

    for (const auto& child : tree.root->children) {
        const auto& leaf = child->children[0]->leaf;
        ASSERT_TRUE(leaf.has_value());
        EXPECT_NEAR(leaf->value.real(), 0.25, 1e-12);
    }
}

TEST_F(CalObsNeel, LongRangeCorrelator)
{
    // ⟨Sz@0 Sz@3⟩ = 0.5 * (-0.5) = -0.25  (site-0 up, site-3 down)
    ObservableTree<> tree(L);
    add_obs2(tree, mkSz(0), 0, mkSz(3), 3);

    cal_obs(tree, env);

    const auto& leaf = tree.root->children[0]->children[0]->leaf;
    ASSERT_TRUE(leaf.has_value());
    EXPECT_NEAR(leaf->value.real(), -0.25, 1e-12);
}

// ── cal_obs: Sp/Sm off-diagonal correlators ───────────────────────────────────

TEST(CalObsSpSm, SpKillsSpinUp)
{
    // |↑↑⟩: ⟨S+@0 S-@1⟩ — S+ applied to bra ⟨↑| gives 0 → result = 0
    const int L = 2;
    auto psi = product_state({true, true});
    auto H   = make_dummy_mpo(L);
    Environment<> env(psi, H);

    ObservableTree<> tree(L);
    add_obs2(tree, mkSp(0), 0, mkSm(1), 1);
    cal_obs(tree, env);

    const auto& leaf = tree.root->children[0]->children[0]->leaf;
    ASSERT_TRUE(leaf.has_value());
    EXPECT_NEAR(std::abs(leaf->value), 0.0, 1e-12);
}

TEST(CalObsSpSm, SzOnSpinDownSite)
{
    // |↓↑⟩: ⟨Sz@0⟩ = -0.5
    const int L = 2;
    auto psi = product_state({false, true});
    auto H   = make_dummy_mpo(L);
    Environment<> env(psi, H);

    ObservableTree<> tree(L);
    add_obs(tree, mkSz(0), 0);
    cal_obs(tree, env);

    const auto& leaf = tree.root->children[0]->leaf;
    ASSERT_TRUE(leaf.has_value());
    EXPECT_NEAR(leaf->value.real(), -0.5, 1e-12);
}

} // namespace tenet::test
