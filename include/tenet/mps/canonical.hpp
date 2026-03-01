#pragma once
// include/tenet/mps/canonical.hpp
//
// Free functions for MPS canonicalisation.
// See docs/C++重构设计方案.md §7.4.

#include "tenet/mps/mps.hpp"

#include <Eigen/Dense>

namespace tenet {

// Left-canonicalise sites [from, to).  The boundary matrix is absorbed into
// site `to` (if in range).
template<TensorBackend B>
void left_canonicalize(DenseMPS<B>& psi, int from, int to);

// Right-canonicalise sites (from, to].  The boundary matrix is absorbed into
// site `from` (if in range).
template<TensorBackend B>
void right_canonicalize(DenseMPS<B>& psi, int from, int to);

// Move orthogonality centre to `target` via the shortest path.
template<TensorBackend B>
void move_center(DenseMPS<B>& psi, int target);

} // namespace tenet
