#pragma once
// include/tenet/environment/mpo_environment.hpp
//
// MPOEnvironment<B>: three-layer [ρ, H, ρ†] environment for tanTRG / mpo_mul.
//
// Left env el_i:  shape (D_ρ†_l, D_ρ_l)  — one block per SparseMPO bond index i
// Right env er_j: shape (D_ρ_r, D_ρ†_r)  — one block per SparseMPO bond index j
//
// Reuses SparseLeftEnvTensor / SparseRightEnvTensor infrastructure.

#include "tenet/environment/env_tensor.hpp"
#include "tenet/mps/dense_mpo.hpp"
#include "tenet/mps/mpo.hpp"

#include <vector>

namespace tenet {

template<TensorBackend B = DenseBackend>
class MPOEnvironment {
public:
    // rho: DenseMPO being optimised (non-owning reference)
    // H:   SparseMPO Hamiltonian (non-owning reference)
    MPOEnvironment(DenseMPO<B>& rho, SparseMPO<B>& H);

    DenseMPO<B>&        rho()       { return *rho_; }
    const DenseMPO<B>&  rho() const { return *rho_; }
    SparseMPO<B>&       H()         { return *H_;   }
    const SparseMPO<B>& H()   const { return *H_;   }
    int                 length() const noexcept { return L_; }

    // Access left/right environments.
    // left_env(0)  = left boundary; left_env(L) = rightmost left env.
    // right_env(L) = right boundary.
    SparseLeftEnvTensor<B>&        left_env(int bond);
    const SparseLeftEnvTensor<B>&  left_env(int bond) const;
    SparseRightEnvTensor<B>&       right_env(int bond);
    const SparseRightEnvTensor<B>& right_env(int bond) const;

    // Build all right environments (from right boundary inward).
    // Call after initialising rho (usually right-canonical from site 0).
    void build_right_envs();

    // Sweep propagation:
    //   push_right(s): left_env[s+1] ← contract(left_env[s], rho[s], H[s], rho†[s])
    //   push_left(s):  right_env[s]  ← contract(rho[s], H[s], rho†[s], right_env[s+1])
    void push_right(int site);
    void push_left(int site);

private:
    DenseMPO<B>*  rho_ = nullptr;
    SparseMPO<B>* H_   = nullptr;
    int           L_   = 0;

    std::vector<SparseLeftEnvTensor<B>>  left_envs_;   // size L+1
    std::vector<SparseRightEnvTensor<B>> right_envs_;  // size L+1
};

} // namespace tenet
