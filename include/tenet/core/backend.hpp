#pragma once
// include/tenet/core/backend.hpp
//
// Backend concept and DenseBackend (Phase-1 default).
//
// The entire algorithm layer is parameterised by a Backend type B that
// bundles:
//   B::Scalar   – numeric scalar type (std::complex<double> for Phase 1)
//   B::Space    – leg-space type (TrivialSpace for Phase 1)
//   B::Tensor   – tensor type (DenseTensor for Phase 1)
//
// Phase 2 will introduce SymmetricBackend<Charge> with block-sparse tensors;
// all algorithm code remains unchanged.

#include "tenet/core/dense_tensor.hpp"
#include "tenet/core/space.hpp"

#include <concepts>
#include <cstdint>
#include <optional>
#include <vector>

namespace tenet {

// ── TensorBackend concept ─────────────────────────────────────────────────────
template<typename B>
concept TensorBackend = requires {
    typename B::Scalar;
    typename B::Space;
    typename B::Tensor;
    requires SpaceLike<typename B::Space>;
} && requires(std::vector<typename B::Space> legs,
              std::optional<uint64_t>        seed) {
    // Factory functions – must be callable as static member functions.
    { B::zeros(legs)        } -> std::same_as<typename B::Tensor>;
    { B::random(legs, seed) } -> std::same_as<typename B::Tensor>;
    { B::identity(legs[0])  } -> std::same_as<typename B::Tensor>;
};

// ── DenseBackend ──────────────────────────────────────────────────────────────
struct DenseBackend {
    using Scalar = std::complex<double>;
    using Space  = TrivialSpace;
    using Tensor = DenseTensor;

    // Return a rank-N zero tensor with the given leg spaces.
    static DenseTensor zeros(const std::vector<TrivialSpace>& legs);

    // Return a random tensor.  If seed is provided the result is reproducible;
    // in multithreaded code each thread uses a thread_local engine.
    static DenseTensor random(const std::vector<TrivialSpace>& legs,
                               std::optional<uint64_t> seed = std::nullopt);

    // Return a (square) identity tensor for a single space.
    // Resulting tensor has shape (space, space.dual()) – rank 2.
    static DenseTensor identity(const TrivialSpace& space);
};

static_assert(TensorBackend<DenseBackend>, "DenseBackend must satisfy TensorBackend");

// Convenience alias – users of Phase-1 code can write DenseMPS<> instead
// of DenseMPS<DenseBackend>.
using DefaultBackend = DenseBackend;

} // namespace tenet
