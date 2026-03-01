#pragma once
// include/tenet/krylov/lanczos.hpp
//
// Lanczos eigensolver and linear system solver.
// Used by DMRG (ground-state search) and TDVP (matrix-exponential action).
// See docs/C++重构设计方案.md §9.

#include "tenet/core/dense_tensor.hpp"

#include <Eigen/Dense>
#include <functional>
#include <optional>

namespace tenet {

// ── Lanczos eigensolver ───────────────────────────────────────────────────────
// Finds the lowest few eigenvalues / eigenvectors of the linear map H.
//
// matvec: v ↦ H·v  (H must be Hermitian)
// v0:     initial vector
// Returns the lowest `n_eigs` (eigenvalue, eigenvector) pairs.

struct LanczosConfig {
    int    krylov_dim = 8;
    int    max_iter   = 1;
    double tol        = 1e-6;
    bool   eager      = true;   // Stop early if tolerance reached
};

struct EigPair {
    double      eigenvalue;
    DenseTensor eigenvector;
};

std::vector<EigPair> lanczos_eigs(
    std::function<DenseTensor(const DenseTensor&)> matvec,
    const DenseTensor&                              v0,
    int                                             n_eigs = 1,
    const LanczosConfig&                            cfg    = {});

// ── Lanczos linear-system solver (GMRES / Lanczos-based) ─────────────────────
// Solves  (H - shift·I)|x⟩ = |b⟩
struct LinSolveResult {
    DenseTensor solution;
    bool        converged   = false;
    int         iterations  = 0;
    double      residual    = 0.0;
};

LinSolveResult lanczos_solve(
    std::function<DenseTensor(const DenseTensor&)> matvec,
    const DenseTensor&                              b,
    std::complex<double>                            shift = 0.0,
    const LanczosConfig&                            cfg   = {});

} // namespace tenet
