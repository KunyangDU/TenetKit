#pragma once
// include/tenet/algebra/mpo_mul.hpp
//
// mpo_mul: variational MPO multiplication C ← α · A · H.
// Minimises ‖C − α·A·H‖² using alternating single-site sweeps.
//
// Environment used: three-layer [A, H, C†] (MPOEnvironment).
//
// MulConfig: controls sweep count, tolerance, truncation.

#include "tenet/mps/dense_mpo.hpp"
#include "tenet/mps/mpo.hpp"
#include "tenet/core/factorization.hpp"

namespace tenet {

struct MulConfig {
    int         max_sweeps = 4;
    double      tol        = 1e-12;
    TruncParams trunc      = {};
};

// Variational MPO multiplication: C ← α · A · H  (C is updated in-place).
// Returns final relative residual ‖C_new − C_old‖ / ‖C_old‖.
template<TensorBackend B = DenseBackend>
double mpo_mul(DenseMPO<B>&        C,
               const DenseMPO<B>&  A,
               const SparseMPO<B>& H,
               double              alpha = 1.0,
               const MulConfig&    cfg   = {});

} // namespace tenet
