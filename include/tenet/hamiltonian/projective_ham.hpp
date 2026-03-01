#pragma once
// include/tenet/hamiltonian/projective_ham.hpp
//
// SparseProjectiveHamiltonian<B>: the effective Hamiltonian H_eff seen by
// one or two sites during a DMRG/TDVP sweep.
// See docs/C++重构设计方案.md §9.

#include "tenet/environment/env_tensor.hpp"
#include "tenet/environment/environment.hpp"
#include "tenet/mps/mpo.hpp"

#include <optional>
#include <vector>

namespace tenet {

template<TensorBackend B = DenseBackend>
class SparseProjectiveHamiltonian {
public:
    using Tensor = typename B::Tensor;

    SparseProjectiveHamiltonian(SparseLeftEnvTensor<B>  envL,
                                SparseRightEnvTensor<B> envR,
                                std::optional<SparseMPO<B>*> H,
                                std::vector<std::pair<int,int>> valid_inds,
                                double E0 = 0.0,
                                int site1 = -1,
                                int site2 = -1);

    const SparseLeftEnvTensor<B>&  env_left()   const { return envL_; }
    const SparseRightEnvTensor<B>& env_right()  const { return envR_; }
    double                          energy_offset() const { return E0_; }

    // The number of active sites (0, 1, or 2)
    int n_sites() const noexcept { return n_sites_; }
    int site1()   const noexcept { return site1_; }
    int site2()   const noexcept { return site2_; }

    const std::optional<SparseMPO<B>*>&    H()          const { return H_; }
    const std::vector<std::pair<int,int>>& valid_inds() const { return valid_inds_; }

private:
    SparseLeftEnvTensor<B>           envL_;
    SparseRightEnvTensor<B>          envR_;
    std::optional<SparseMPO<B>*>     H_;
    std::vector<std::pair<int,int>>  valid_inds_;
    double                           E0_;
    int                              n_sites_ = 1;
    int                              site1_   = -1;
    int                              site2_   = -1;
};

// ── Factory functions ─────────────────────────────────────────────────────────
template<TensorBackend B>
SparseProjectiveHamiltonian<B> proj0(SparseLeftEnvTensor<B>  envL,
                                      SparseRightEnvTensor<B> envR,
                                      double E0 = 0.0);

template<TensorBackend B>
SparseProjectiveHamiltonian<B> proj1(const Environment<B>& env, int site,
                                      double E0 = 0.0);

template<TensorBackend B>
SparseProjectiveHamiltonian<B> proj2(const Environment<B>& env,
                                      int site1, int site2,
                                      double E0 = 0.0);

} // namespace tenet
