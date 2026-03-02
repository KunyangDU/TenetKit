// src/mps/dense_mpo.cpp
//
// DenseMPO<DenseBackend>: identity, canonicalization, normalize, trace_sq.

#include "tenet/mps/dense_mpo.hpp"
#include "tenet/core/space.hpp"
#include "tenet/core/tensor_ops.hpp"

#include <cassert>
#include <cmath>

namespace tenet {

// ── identity ──────────────────────────────────────────────────────────────────

template<>
DenseMPO<DenseBackend> DenseMPO<DenseBackend>::identity(int L, int d)
{
    DenseMPO<DenseBackend> mpo(L);
    for (int s = 0; s < L; ++s) {
        mpo[s] = DenseMPOTensor<DenseBackend>(1, d, 1);
        // Set diagonal: ρ[s]({0, σ, σ, 0}) = 1 for σ = 0..d-1
        for (int sigma = 0; sigma < d; ++sigma)
            mpo[s].data()({0, sigma, sigma, 0}) = DenseTensor::Scalar{1.0, 0.0};
    }
    mpo.center_ = 0;
    return mpo;
}

// ── left_canonicalize ─────────────────────────────────────────────────────────

template<>
void DenseMPO<DenseBackend>::left_canonicalize(int from, int to)
{
    // Canonicalize sites [from, to): QR at each site, push R right.
    for (int s = from; s < to; ++s) {
        Eigen::MatrixXcd R = sites_[s].left_canonicalize();

        if (s + 1 < L_) {
            // Absorb R into sites_[s+1] from the left
            // sites_[s+1] has shape (D_l, d, d, D_r); as_matrix_right() = D_l × (d*d*D_r)
            int Dr = sites_[s + 1].D_r();
            int d  = sites_[s + 1].d();
            int new_Dl = static_cast<int>(R.rows());
            Eigen::MatrixXcd M = sites_[s + 1].as_matrix_right();  // D_l × (d*d*D_r)
            Eigen::MatrixXcd new_M = R * M;
            sites_[s + 1] = DenseMPOTensor<DenseBackend>::from_right_matrix(
                new_M, new_Dl, d, Dr);
        }
    }
}

// ── right_canonicalize ────────────────────────────────────────────────────────

template<>
void DenseMPO<DenseBackend>::right_canonicalize(int from, int to)
{
    // Canonicalize sites (from, to]: RQ at each site from to down to from+1,
    // push L left.
    for (int s = to; s > from; --s) {
        Eigen::MatrixXcd L = sites_[s].right_canonicalize();

        if (s - 1 >= 0) {
            int Dl = sites_[s - 1].D_l();
            int d  = sites_[s - 1].d();
            int new_Dr = static_cast<int>(L.cols());
            Eigen::MatrixXcd M = sites_[s - 1].as_matrix_left();  // (D_l*d*d) × D_r
            Eigen::MatrixXcd new_M = M * L;
            sites_[s - 1] = DenseMPOTensor<DenseBackend>::from_left_matrix(
                new_M, Dl, d, new_Dr);
        }
    }
}

// ── move_center_to ────────────────────────────────────────────────────────────

template<>
void DenseMPO<DenseBackend>::move_center_to(int target)
{
    if (target > center_)
        left_canonicalize(center_, target);
    else if (target < center_)
        right_canonicalize(target, center_);
    center_ = target;
}

// ── normalize ────────────────────────────────────────────────────────────────
// Compute Frobenius norm of all site tensors (sum of |elem|^2),
// divide each element by norm, return log(norm).

template<>
double DenseMPO<DenseBackend>::normalize()
{
    // Frobenius norm^2 = sum over all sites of ||ρ[s]||_F^2
    double norm_sq = 0.0;
    for (int s = 0; s < L_; ++s) {
        const DenseTensor& t = sites_[s].data();
        for (int64_t i = 0; i < t.numel(); ++i) {
            auto v = t.data()[i];
            norm_sq += v.real() * v.real() + v.imag() * v.imag();
        }
    }
    double norm = std::sqrt(norm_sq);
    if (norm < 1e-300) return 0.0;

    double inv_norm = 1.0 / norm;
    for (int s = 0; s < L_; ++s) {
        DenseTensor& t = sites_[s].data();
        for (int64_t i = 0; i < t.numel(); ++i)
            t.data()[i] *= inv_norm;
    }
    return std::log(norm);
}

// ── trace_sq ─────────────────────────────────────────────────────────────────
// Tr(ρ†ρ) = sum over all ρ elements of |ρ_elem|^2 (for MPO in mixed canonical form
// with the center at a single site).
// For a general MPO, this is the Frobenius norm squared (which equals Tr(ρ†ρ)
// only when ρ is in canonical form).
template<>
double DenseMPO<DenseBackend>::trace_sq() const
{
    double norm_sq = 0.0;
    for (int s = 0; s < L_; ++s) {
        const DenseTensor& t = sites_[s].data();
        for (int64_t i = 0; i < t.numel(); ++i) {
            auto v = t.data()[i];
            norm_sq += v.real() * v.real() + v.imag() * v.imag();
        }
    }
    return norm_sq;
}

} // namespace tenet
