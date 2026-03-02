#pragma once
// include/tenet/hamiltonian/mpo_ham_action.hpp
//
// apply_H_eff_mpo: H_eff action on a DenseMPOTensor site.
//
// H_eff(ρ[site]) = Σ_{i,j: H[i,j]≠0}  el_i · ρ[site] · h_{i,j} · er_j
//
// el_i:  (D_out_l, D_in_l)   (left env component i)
// ρ[s]:  (D_in_l, d_bra, d_ket, D_in_r)
// h_{ij}:(d_out, d_in)       h acts on d_bra: d_in = d_bra, d_out = new d_bra
// er_j:  (D_in_r, D_out_r)   (right env component j)
//
// Result: (D_out_l, d_out, d_ket, D_out_r)

#include "tenet/environment/env_tensor.hpp"
#include "tenet/mps/dense_mpo_tensor.hpp"
#include "tenet/mps/mpo_tensor.hpp"

namespace tenet {

// Single-site H_eff action on a DenseMPOTensor.
template<TensorBackend B = DenseBackend>
typename B::Tensor
apply_H_eff_mpo(const SparseLeftEnvTensor<B>&   L_env,
                const DenseMPOTensor<B>&          rho_site,
                const SparseMPOTensor<B>&         H_site,
                const SparseRightEnvTensor<B>&    R_env);

} // namespace tenet
