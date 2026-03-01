// src/core/tensor_ops.cpp
//
// Free functions for DenseTensor arithmetic.

#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/backend.hpp"

#include <cassert>
#include <complex>
#include <numeric>
#include <random>
#include <vector>

namespace tenet {

// ── Two-tensor contraction ────────────────────────────────────────────────────

DenseTensor contract(const DenseTensor& A,
                     const DenseTensor& B,
                     const std::vector<std::pair<int,int>>& contracted)
{
    int rA = A.rank(), rB = B.rank();
    int nc = static_cast<int>(contracted.size());

    // Mark contracted legs
    std::vector<bool> A_contr(rA, false), B_contr(rB, false);
    for (auto& [a, b] : contracted) {
        assert(a >= 0 && a < rA);
        assert(b >= 0 && b < rB);
        A_contr[a] = true;
        B_contr[b] = true;
    }

    // Free legs of A and B
    std::vector<int> A_free, B_free;
    for (int i = 0; i < rA; ++i) if (!A_contr[i]) A_free.push_back(i);
    for (int i = 0; i < rB; ++i) if (!B_contr[i]) B_free.push_back(i);

    // Contracted leg lists (in order given by contracted[])
    std::vector<int> A_cleg(nc), B_cleg(nc);
    for (int k = 0; k < nc; ++k) {
        A_cleg[k] = contracted[k].first;
        B_cleg[k] = contracted[k].second;
    }

    // Matricize: A → (free_A) × (contr_A), B → (contr_B) × (free_B)
    Eigen::MatrixXcd M_A = A.matricize(A_free, A_cleg);
    Eigen::MatrixXcd M_B = B.matricize(B_cleg, B_free);

    // Validate dimensions match
    assert(M_A.cols() == M_B.rows() && "contract: contracted dimensions do not match");

    // ZGEMM
    Eigen::MatrixXcd M_C = M_A * M_B;

    // Result legs: free_A legs ++ free_B legs
    std::vector<TrivialSpace> result_legs;
    result_legs.reserve(A_free.size() + B_free.size());
    for (int i : A_free) result_legs.push_back(A.space(i));
    for (int i : B_free) result_legs.push_back(B.space(i));

    return DenseTensor::from_matrix(M_C, std::move(result_legs), true);
}

// ── Inner product ─────────────────────────────────────────────────────────────

std::complex<double> inner(const DenseTensor& A, const DenseTensor& B) {
    assert(A.numel() == B.numel() && "inner: tensors must have the same number of elements");
    int64_t n = A.numel();
    std::complex<double> s{0.0, 0.0};
    for (int64_t i = 0; i < n; ++i)
        s += std::conj(A.data()[i]) * B.data()[i];
    return s;
}

// ── Outer (tensor) product ────────────────────────────────────────────────────

DenseTensor outer(const DenseTensor& A, const DenseTensor& B) {
    // Outer product: result has all legs of A followed by all legs of B
    std::vector<TrivialSpace> result_legs;
    result_legs.reserve(A.rank() + B.rank());
    for (int i = 0; i < A.rank(); ++i) result_legs.push_back(A.space(i));
    for (int i = 0; i < B.rank(); ++i) result_legs.push_back(B.space(i));

    int64_t nA = A.numel(), nB = B.numel();
    std::vector<DenseTensor::Scalar> data(nA * nB);
    for (int64_t i = 0; i < nA; ++i)
        for (int64_t j = 0; j < nB; ++j)
            data[i * nB + j] = A.data()[i] * B.data()[j];

    return DenseTensor(std::move(result_legs), std::move(data));
}

// ── Utilities ─────────────────────────────────────────────────────────────────

DenseTensor zeros_like(const DenseTensor& t) {
    return DenseBackend::zeros(t.spaces());
}

DenseTensor random_like(const DenseTensor& t) {
    return DenseBackend::random(t.spaces());
}

} // namespace tenet
