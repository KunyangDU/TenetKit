#pragma once
// include/tenet/intr_tree/interaction_node.hpp
//
// InteractionTreeLeave and InteractionTreeNode: the tree data structure used
// to build SparseMPO from user-supplied Hamiltonian terms.
// See docs/C++重构设计方案.md §10.

#include "tenet/intr_tree/local_operator.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace tenet {

// ── InteractionTreeLeave: one Hamiltonian term ────────────────────────────────
// Stores N operators on N sites with an overall coefficient.
template<TensorBackend B = DenseBackend, int N = 2>
struct InteractionTreeLeave {
    std::array<std::unique_ptr<AbstractLocalOperator<B>>, N> ops;
    std::array<int, N>    sites;
    std::array<bool, N>   fermionic;   // Jordan-Wigner string required?
    std::complex<double>  strength{1.0};
};

// ── InteractionTreeNode ───────────────────────────────────────────────────────
template<TensorBackend B = DenseBackend>
struct InteractionTreeNode {
    std::unique_ptr<AbstractLocalOperator<B>> op;       // nullptr for root
    std::vector<std::unique_ptr<InteractionTreeNode<B>>> children;
    InteractionTreeNode<B>* parent = nullptr;           // non-owning

    bool is_root() const noexcept { return parent == nullptr; }
};

// ── InteractionTree ───────────────────────────────────────────────────────────
template<TensorBackend B = DenseBackend>
struct InteractionTree {
    std::unique_ptr<InteractionTreeNode<B>> root;
    int L = 0;   // number of sites

    explicit InteractionTree(int L) : root(std::make_unique<InteractionTreeNode<B>>()), L(L) {}
};

} // namespace tenet
