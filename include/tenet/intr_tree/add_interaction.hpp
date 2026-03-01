#pragma once
// include/tenet/intr_tree/add_interaction.hpp
//
// Interface for adding Hamiltonian terms to an InteractionTree.
// The tree is later compiled to a SparseMPO.
// See docs/C++重构设计方案.md §10.

#include "tenet/intr_tree/interaction_node.hpp"
#include "tenet/mps/mpo.hpp"

#include <complex>
#include <initializer_list>

namespace tenet {

template<TensorBackend B = DenseBackend>
using Op = std::unique_ptr<AbstractLocalOperator<B>>;

// ── Add interaction terms ─────────────────────────────────────────────────────

template<TensorBackend B>
void add_intr1(InteractionTree<B>& tree, Op<B> op, int site,
               std::complex<double> strength = 1.0);

template<TensorBackend B>
void add_intr2(InteractionTree<B>& tree, Op<B> op1, int site1,
               Op<B> op2, int site2,
               std::complex<double> strength = 1.0,
               bool fermionic = false);

template<TensorBackend B>
void add_intr3(InteractionTree<B>& tree,
               Op<B> op1, int s1, Op<B> op2, int s2, Op<B> op3, int s3,
               std::complex<double> strength = 1.0);

// ── Compile tree → SparseMPO ──────────────────────────────────────────────────
template<TensorBackend B>
SparseMPO<B> compile(const InteractionTree<B>& tree);

} // namespace tenet
