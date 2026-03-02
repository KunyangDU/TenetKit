#pragma once
// include/tenet/algebra/mpo_axpy.hpp
//
// mpo_axpy: variational MPO addition y ← α · x + y.
// Uses two-site double-site sweeps with TwoLayerMPOEnv.

#include "tenet/mps/dense_mpo.hpp"
#include "tenet/core/factorization.hpp"

namespace tenet {

struct AxpyConfig {
    int         max_sweeps = 3;
    double      tol        = 1e-8;
    TruncParams trunc      = {};
};

// Variational MPO addition: y ← α · x + y  (y updated in-place).
// Returns final residual estimate.
template<TensorBackend B = DenseBackend>
double mpo_axpy(double             alpha,
                const DenseMPO<B>& x,
                DenseMPO<B>&       y,
                const AxpyConfig&  cfg = {});

} // namespace tenet
