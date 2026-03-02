// src/krylov/arnoldi.cpp
//
// arnoldi_expm_vec: Krylov subspace approximation to exp(α H) v.
//
// Algorithm:
//   1. Build orthonormal Krylov basis V = [v_0, ..., v_{k-1}] via
//      Arnoldi iteration (modified Gram-Schmidt).
//   2. Collect the k×k upper-Hessenberg projection H_k of H.
//   3. Compute exp(α H_k) exactly (Padé via Eigen's .exp()).
//   4. Return V_k * (exp(α H_k) * e_1) * ‖v‖_orig.
//
// For Hermitian H the Hessenberg degenerates to tridiagonal (= Lanczos),
// but the code is written for the general case.

#include "tenet/krylov/arnoldi.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>  // .exp()

#include <cassert>

namespace tenet {

// ── Inner product ─────────────────────────────────────────────────────────────
// <u, v> = conj(u)^T v  (Hermitian inner product)
static std::complex<double> inner_product(const DenseTensor& u,
                                           const DenseTensor& v)
{
    assert(u.numel() == v.numel());
    int n = static_cast<int>(u.numel());
    Eigen::Map<const Eigen::VectorXcd> u_vec(u.data(), n);
    Eigen::Map<const Eigen::VectorXcd> v_vec(v.data(), n);
    return u_vec.dot(v_vec);   // Eigen dot = conj(this).T * other
}

// ── arnoldi_expm_vec ──────────────────────────────────────────────────────────

DenseTensor arnoldi_expm_vec(
    std::function<DenseTensor(const DenseTensor&)> matvec,
    const DenseTensor&                              v,
    std::complex<double>                            alpha,
    const ArnoldiConfig&                            cfg)
{
    double v_norm = v.norm();
    if (v_norm < 1e-14)
        return DenseTensor(v.spaces());   // zero

    int max_dim = std::min(cfg.krylov_dim,
                           static_cast<int>(v.numel()));

    // ── Build Krylov basis ────────────────────────────────────────────────────
    // V[j]: orthonormal Krylov vectors
    // H_hess(i, j): Hessenberg coefficients  (size (max_dim+1) × max_dim)
    std::vector<DenseTensor> V;
    V.reserve(max_dim);

    Eigen::MatrixXcd H_hess = Eigen::MatrixXcd::Zero(max_dim + 1, max_dim);

    DenseTensor q = v;
    q.normalize();
    V.push_back(q);

    int k = 1;  // actual Krylov dimension built so far

    for (int j = 0; j < max_dim; ++j) {
        DenseTensor w = matvec(V[j]);

        // Modified Gram-Schmidt: orthogonalize w against all previous V[i]
        for (int i = 0; i <= j; ++i) {
            std::complex<double> h_ij = inner_product(V[i], w);
            H_hess(i, j) = h_ij;
            // w -= h_ij * V[i]
            w.axpby({1.0, 0.0}, {-h_ij.real(), -h_ij.imag()}, V[i]);
        }

        double beta = w.norm();
        H_hess(j + 1, j) = beta;

        k = j + 1;

        if (beta < 1e-10)
            break;   // invariant subspace — exact result possible

        if (j < max_dim - 1) {
            w.normalize();
            V.push_back(std::move(w));
        }
    }

    // ── Compute exp(α * H_k) where H_k is k×k upper Hessenberg ──────────────
    Eigen::MatrixXcd H_k   = H_hess.topLeftCorner(k, k);
    Eigen::MatrixXcd expH  = (alpha * H_k).exp();   // Padé approx via Eigen

    // Coefficient vector: c = exp(α H_k) * e_1 * v_norm
    Eigen::VectorXcd c = expH.col(0) * v_norm;

    // ── Map back to original space: result = V * c ────────────────────────────
    DenseTensor result(v.spaces());   // zero tensor, same shape
    for (int j = 0; j < k; ++j)
        result.axpby({1.0, 0.0}, {c(j).real(), c(j).imag()}, V[j]);

    return result;
}

} // namespace tenet
