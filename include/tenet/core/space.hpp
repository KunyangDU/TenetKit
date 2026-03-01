#pragma once
// include/tenet/core/space.hpp
//
// TrivialSpace: a plain integer dimension, no quantum-number structure.
// This satisfies the SpaceLike concept and serves as the Phase-1 backend.
//
// Future: VectorSpace<Charge> will carry sector labels and satisfy the same
// concept, allowing algorithm code to remain untouched.

#include <cassert>
#include <concepts>

namespace tenet {

// ── SpaceLike concept ────────────────────────────────────────────────────────
// Any type S satisfying this concept can be used as a "leg label" in tensors.
template<typename S>
concept SpaceLike = requires(S s, int d) {
    { s.dim()              } -> std::convertible_to<int>;
    { s.dual()             } -> std::same_as<S>;
    { S::trivial(d)        } -> std::same_as<S>;
    { s == s               } -> std::convertible_to<bool>;
};

// ── TrivialSpace ─────────────────────────────────────────────────────────────
// Phase-1 implementation: just a dimension + a dual flag.
class TrivialSpace {
public:
    explicit TrivialSpace(int dim) : dim_(dim) { assert(dim > 0); }

    int  dim()     const noexcept { return dim_; }
    bool is_dual() const noexcept { return dual_; }

    // Flip dual flag (physical vs virtual leg orientation).
    TrivialSpace dual() const noexcept { return TrivialSpace(dim_, !dual_); }

    // Factory: create a trivial space of given dimension.
    static TrivialSpace trivial(int dim) { return TrivialSpace(dim); }

    bool operator==(const TrivialSpace&) const = default;

private:
    TrivialSpace(int dim, bool dual) : dim_(dim), dual_(dual) {}

    int  dim_  = 1;
    bool dual_ = false;
};

static_assert(SpaceLike<TrivialSpace>, "TrivialSpace must satisfy SpaceLike");

} // namespace tenet
