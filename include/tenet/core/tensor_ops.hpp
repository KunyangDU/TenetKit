#pragma once
// include/tenet/core/tensor_ops.hpp
//
// Free functions for DenseTensor arithmetic.
// All contraction ultimately calls BLAS ZGEMM via Eigen.

#include "tenet/core/dense_tensor.hpp"

#include <Eigen/Dense>
#include <complex>
#include <utility>
#include <vector>

namespace tenet {

// ── Two-tensor contraction ────────────────────────────────────────────────────
// C = A ⊗_contracted B
//
// contracted: list of (leg-index in A, leg-index in B) pairs to sum over.
// Open legs: remaining A legs (in original order) ++ remaining B legs.
//
// Internally: matricize → ZGEMM → from_matrix.
DenseTensor contract(const DenseTensor&                     A,
                     const DenseTensor&                     B,
                     const std::vector<std::pair<int,int>>& contracted);

// ── Inner product ⟨A|B⟩ = Σ conj(A_i) * B_i ─────────────────────────────────
// Tensors must have the same shape.
std::complex<double> inner(const DenseTensor& A, const DenseTensor& B);

// ── Outer (tensor) product ────────────────────────────────────────────────────
DenseTensor outer(const DenseTensor& A, const DenseTensor& B);

// ── Utilities ─────────────────────────────────────────────────────────────────
DenseTensor zeros_like (const DenseTensor& t);
DenseTensor random_like(const DenseTensor& t);

} // namespace tenet
