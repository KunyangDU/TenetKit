#pragma once
// include/tenet/environment/cbe_environment.hpp
//
// CBEEnvironment<B>: auxiliary environment data for the Correlated Basis
// Expansion (CBE) bond-dimension growth algorithm.
// See docs/C++重构设计方案.md §8.

#include "tenet/environment/env_tensor.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <optional>

namespace tenet {

template<TensorBackend B = DenseBackend>
struct CBEEnvironment {
    using Tensor = typename B::Tensor;

    // Reference (unmodified) left/right tensors
    SparseLeftEnvTensor<B>  tL0;
    SparseRightEnvTensor<B> tR0;

    // Working tensors (modified during CBE)
    std::optional<SparseLeftEnvTensor<B>>  tL;
    std::optional<SparseRightEnvTensor<B>> tR;

    int D_init = 0;   // Initial bond dimension
    int D_final = 0;  // Target bond dimension

    std::optional<Tensor> singular_values;  // Λ from randomised SVD
};

} // namespace tenet
