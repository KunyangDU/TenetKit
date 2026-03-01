#pragma once
// include/tenet/hamiltonian/ham_action.hpp
//
// apply(): computes H_eff |ψ_site⟩ for use inside Krylov solvers.
// See docs/C++重构设计方案.md §9.

#include "tenet/hamiltonian/projective_ham.hpp"
#include "tenet/mps/mps_tensor.hpp"

namespace tenet {

// Single-site: H_eff |A⟩
template<TensorBackend B>
typename B::Tensor apply(const SparseProjectiveHamiltonian<B>& H_eff,
                         const typename B::Tensor&              psi);

// Two-site: H_eff |AB⟩
template<TensorBackend B>
typename B::Tensor apply2(const SparseProjectiveHamiltonian<B>& H_eff,
                           const typename B::Tensor&              psi_ab);

} // namespace tenet
