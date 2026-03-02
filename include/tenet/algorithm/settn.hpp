#pragma once
// include/tenet/algorithm/settn.hpp
//
// SETTN (Series Expansion Tensor Network): finite-temperature simulation.
//
// ρ(β) = e^{-βH} = Σ_{n=0}^{N} (-β)^n / n! · H^n
//
// Iterates:
//   Hn ← Hn · H      (mpo_mul, DenseMPO × SparseMPO)
//   ρ  ← ρ + coeff · Hn  (mpo_axpy)
// where coeff = (-β)^n / n!
//
// Free energy: F = -log(Tr(ρ)) / β
// Convergence: |ΔlnZ / lnZ| < tol

#include "tenet/mps/dense_mpo.hpp"
#include "tenet/mps/mpo.hpp"
#include "tenet/process_control/config.hpp"

#include <vector>

namespace tenet {

struct SETTNResult {
    std::vector<double> lnZ_values;       // lnZ at each completed order (0..n)
    std::vector<double> free_energies;    // F(β) = -lnZ / β at each order
    std::vector<double> truncation_errs;  // sum of mpo_mul + mpo_axpy errors per order
    int converged_order = -1;             // order n at which convergence was detected (-1 if not)
};

// Main SETTN driver: compute ρ(β) = e^{-βH} as a DenseMPO via Taylor expansion.
// H is the Hamiltonian as a SparseMPO.
// cfg.max_order:  maximum expansion order N
// cfg.tol:        convergence threshold on |ΔlnZ / lnZ|
// cfg.trunc:      SVD truncation parameters for mpo_mul and mpo_axpy
template<TensorBackend B = DenseBackend>
SETTNResult settn(SparseMPO<B>& H, double beta, const SETTNConfig& cfg = {});

} // namespace tenet
