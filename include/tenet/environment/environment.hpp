#pragma once
// include/tenet/environment/environment.hpp
//
// Environment<B>: manages left and right partial contractions along the chain.
// See docs/C++重构设计方案.md §8.2.

#include "tenet/environment/env_tensor.hpp"
#include "tenet/mps/mps.hpp"
#include "tenet/mps/mpo.hpp"

#include <vector>

namespace tenet {

template<TensorBackend B = DenseBackend>
class Environment {
public:
    // Build environment wrapping psi and H.
    // psi and H are stored as non-owning references.
    Environment(DenseMPS<B>& psi, SparseMPO<B>& H);

    DenseMPS<B>&        psi()       { return *psi_; }
    const DenseMPS<B>&  psi() const { return *psi_; }
    SparseMPO<B>&       H()         { return *H_;   }
    const SparseMPO<B>& H()   const { return *H_;   }
    int                 length() const noexcept { return L_; }

    // Access cached environment tensors.
    // left_env(0)  = left boundary (always initialised).
    // right_env(L) = right boundary.
    SparseLeftEnvTensor<B>&  left_env(int site);
    SparseRightEnvTensor<B>& right_env(int site);

    int center() const noexcept { return psi_->center_left(); }

    // ── Initialisation ────────────────────────────────────────────────────────
    // Call once after psi is in canonical form to build all environments.
    void build_all();

    // ── Sweep propagation ─────────────────────────────────────────────────────
    // left_env[site+1] ← contract(left_env[site], psi[site], H[site])
    void push_right(int site);
    // right_env[site-1] ← contract(right_env[site], psi[site], H[site])
    void push_left(int site);

private:
    DenseMPS<B>* psi_ = nullptr;
    SparseMPO<B>* H_  = nullptr;
    int           L_  = 0;

    // left_envs_[i]  = environment to the left of site i  (size L+1)
    // right_envs_[i] = environment to the right of site i (size L+1)
    std::vector<SparseLeftEnvTensor<B>>  left_envs_;
    std::vector<SparseRightEnvTensor<B>> right_envs_;
};

} // namespace tenet
