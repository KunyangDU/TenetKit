// src/krylov/lanczos.cpp
//
// lanczos_eigs: standard Lanczos iteration for the lowest eigenvalue(s)
//               of a Hermitian operator given as a matvec functor.
// lanczos_solve: Lanczos-based solver for (H - shift·I)|x⟩ = |b⟩.

#include "tenet/krylov/lanczos.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace tenet {

// ── Inner product ─────────────────────────────────────────────────────────────
// <u,v> = sum_i  conj(u_i) * v_i  (Hermitian inner product)

static std::complex<double> inner_product(const DenseTensor& u,
                                           const DenseTensor& v)
{
    assert(u.numel() == v.numel());
    int n = static_cast<int>(u.numel());
    Eigen::Map<const Eigen::VectorXcd> u_vec(u.data(), n);
    Eigen::Map<const Eigen::VectorXcd> v_vec(v.data(), n);
    return u_vec.dot(v_vec);   // Eigen dot = conj(this).T * other
}

// ── lanczos_eigs ──────────────────────────────────────────────────────────────

std::vector<EigPair> lanczos_eigs(
    std::function<DenseTensor(const DenseTensor&)> matvec,
    const DenseTensor&                              v0,
    int                                             n_eigs,
    const LanczosConfig&                            cfg)
{
    int max_dim = std::min(cfg.krylov_dim,
                           static_cast<int>(v0.numel()));

    // Guard: degenerate initial vector.
    {
        double nrm = v0.norm();
        if (nrm < 1e-14) {
            EigPair ep;
            ep.eigenvalue  = 0.0;
            ep.eigenvector = DenseTensor(v0.spaces());
            return {ep};
        }
    }

    std::vector<DenseTensor> V;   // Krylov basis (orthonormal)
    V.reserve(max_dim);

    std::vector<double> alpha_vec, beta_vec;

    // v1 = v0 / ||v0||
    DenseTensor v = v0;
    v.normalize();
    V.push_back(v);

    int actual_dim = 1;

    for (int j = 0; j < max_dim; ++j) {
        DenseTensor w = matvec(V[j]);

        // alpha_j = Re(<v_j, w>)  (Hermitian → real)
        double alpha_j = inner_product(V[j], w).real();
        alpha_vec.push_back(alpha_j);

        // Three-term recurrence: w -= alpha_j * v_j
        w.axpby({1, 0}, {-alpha_j, 0}, V[j]);

        // w -= beta_{j-1} * v_{j-1}
        if (j > 0)
            w.axpby({1, 0}, {-beta_vec[j - 1], 0}, V[j - 1]);

        actual_dim = j + 1;

        if (j < max_dim - 1) {
            double beta_j = w.norm();

            if (beta_j < 1e-10) {
                // Reached invariant subspace; stop (do NOT push near-zero beta).
                break;
            }
            beta_vec.push_back(beta_j);
            w.normalize();
            V.push_back(std::move(w));
        }
    }

    int k = actual_dim;

    // Build real symmetric tridiagonal T.
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i) {
        T(i, i) = alpha_vec[i];
        if (i < static_cast<int>(beta_vec.size())) {
            T(i, i + 1) = beta_vec[i];
            T(i + 1, i) = beta_vec[i];
        }
    }

    // Eigendecompose T (ascending order).
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);

    int n = std::min(n_eigs, k);
    std::vector<EigPair> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i) {
        double          eigenvalue = solver.eigenvalues()(i);
        Eigen::VectorXd eigvec_T  = solver.eigenvectors().col(i);

        // Map back to original space: x = V * y
        DenseTensor eigvec(v0.spaces());   // zero tensor, same shape
        for (int j = 0; j < k; ++j)
            eigvec.axpby({1, 0}, {eigvec_T(j), 0.0}, V[j]);

        result.push_back({eigenvalue, std::move(eigvec)});
    }

    return result;
}

// ── lanczos_solve ─────────────────────────────────────────────────────────────
// Solve (H - shift·I)|x⟩ = |b⟩ using the Lanczos projection.
// Builds the Krylov basis for (H - shift·I) and solves the projected system.

LinSolveResult lanczos_solve(
    std::function<DenseTensor(const DenseTensor&)> matvec,
    const DenseTensor&                              b,
    std::complex<double>                            shift,
    const LanczosConfig&                            cfg)
{
    double b_norm = b.norm();
    if (b_norm < 1e-14)
        return {DenseTensor(b.spaces()), true, 0, 0.0};

    int max_dim = std::min(cfg.krylov_dim,
                           static_cast<int>(b.numel()));

    // Shifted matvec: v ↦ H·v - shift·v
    auto matvec_shifted = [&](const DenseTensor& v) {
        DenseTensor w = matvec(v);
        if (std::norm(shift) > 0.0)
            w.axpby({1, 0}, {-shift.real(), -shift.imag()}, v);
        return w;
    };

    std::vector<DenseTensor> V;
    V.reserve(max_dim);
    std::vector<double> alpha_vec, beta_vec;

    DenseTensor v = b;
    v.normalize();
    V.push_back(v);

    int actual_dim = 1;

    for (int j = 0; j < max_dim; ++j) {
        DenseTensor w = matvec_shifted(V[j]);

        double alpha_j = inner_product(V[j], w).real();
        alpha_vec.push_back(alpha_j);

        w.axpby({1, 0}, {-alpha_j, 0}, V[j]);
        if (j > 0)
            w.axpby({1, 0}, {-beta_vec[j - 1], 0}, V[j - 1]);

        actual_dim = j + 1;

        if (j < max_dim - 1) {
            double beta_j = w.norm();
            if (beta_j < 1e-10) break;
            beta_vec.push_back(beta_j);
            w.normalize();
            V.push_back(std::move(w));
        }
    }

    int k = actual_dim;

    // Build tridiagonal (for shifted operator).
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i) {
        T(i, i) = alpha_vec[i];
        if (i < static_cast<int>(beta_vec.size())) {
            T(i, i + 1) = beta_vec[i];
            T(i + 1, i) = beta_vec[i];
        }
    }

    // RHS in Krylov basis: e_1 * ||b||  (V[0] = b / ||b||)
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(k);
    rhs(0) = b_norm;

    // Solve T * y = rhs
    Eigen::VectorXd y = T.colPivHouseholderQr().solve(rhs);

    // x = V * y
    DenseTensor x(b.spaces());
    for (int j = 0; j < k; ++j)
        x.axpby({1, 0}, {y(j), 0.0}, V[j]);

    // Compute residual ||(H - shift·I)x - b||
    DenseTensor Ax = matvec_shifted(x);
    Ax.axpby({1, 0}, {-1, 0}, b);
    double residual = Ax.norm();

    return {std::move(x), residual < cfg.tol, k, residual};
}

} // namespace tenet
