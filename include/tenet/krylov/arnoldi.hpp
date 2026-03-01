#pragma once
// include/tenet/krylov/arnoldi.hpp
//
// Arnoldi-based matrix-exponential action: exp(α H) v.
// Used by TDVP for time propagation of site tensors.
// See docs/C++重构设计方案.md §9.

#include "tenet/core/dense_tensor.hpp"

#include <complex>
#include <functional>

namespace tenet {

struct ArnoldiConfig {
    int    krylov_dim = 32;
    int    max_iter   = 1;
    double tol        = 1e-8;
};

// Computes exp(α * H) * v  where H is Hermitian.
// matvec: v ↦ H·v
DenseTensor arnoldi_expm_vec(
    std::function<DenseTensor(const DenseTensor&)> matvec,
    const DenseTensor&                              v,
    std::complex<double>                            alpha,
    const ArnoldiConfig&                            cfg = {});

} // namespace tenet
