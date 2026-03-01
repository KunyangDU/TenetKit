#pragma once
// include/tenet/environment/env_push.hpp
//
// Free functions for environment propagation during sweeps.
// The Environment class delegates to these internally.

#include "tenet/environment/env_tensor.hpp"
#include "tenet/mps/mps_tensor.hpp"
#include "tenet/mps/mpo_tensor.hpp"

namespace tenet {

// Extend a left environment by one site:
//   L_new = contract(L_old, psi[site], H[site], conj(psi[site]))
template<TensorBackend B>
SparseLeftEnvTensor<B> push_right(const SparseLeftEnvTensor<B>& L,
                                   const MPSTensor<B>&            psi_site,
                                   const SparseMPOTensor<B>&      H_site);

template<TensorBackend B>
SparseRightEnvTensor<B> push_left(const SparseRightEnvTensor<B>& R,
                                   const MPSTensor<B>&             psi_site,
                                   const SparseMPOTensor<B>&       H_site);

} // namespace tenet
