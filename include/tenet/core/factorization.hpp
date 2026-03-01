#pragma once
// include/tenet/core/factorization.hpp
//
// QR, SVD, eigendecomposition, and matrix-exponential-times-vector.
// All work on DenseTensor with row-major storage via Eigen.

#include "tenet/core/dense_tensor.hpp"

#include <Eigen/Dense>

namespace tenet {

// ── Truncation parameters ─────────────────────────────────────────────────────
struct TruncParams {
    int    maxD      = 0;       // Maximum bond dim kept (0 = unlimited)
    double cutoff    = 1e-12;   // Drop singular values with σ/σ_max < cutoff
    bool   normalize = false;   // Re-normalise after truncation
};

// ── QR decomposition ──────────────────────────────────────────────────────────
// T[legs 0..split_at-1 | legs split_at..rank-1] = Q * R
// Q: left-orthogonal (Q†Q = I), R: upper-triangular.
struct QRResult {
    DenseTensor Q;
    DenseTensor R;
    int         bond_dim = 0;
};

QRResult qr(const DenseTensor& T, int split_at);  // Q left-orthogonal
QRResult rq(const DenseTensor& T, int split_at);  // Q right-orthogonal (R*Q form)

// ── SVD ───────────────────────────────────────────────────────────────────────
// T ≈ U * diag(S) * Vt,  singular values in descending order.
struct SVDResult {
    DenseTensor     U;               // Left singular vectors
    Eigen::VectorXd S;               // Singular values (descending)
    DenseTensor     Vt;              // Right singular vectors (conjugate-transposed)
    int             bond_dim      = 0;
    double          truncation_err = 0.0;  // Σ discarded σ²
};

SVDResult svd(const DenseTensor& T, int split_at,
              const TruncParams& trunc = {});

// Randomised SVD (Halko et al.) for the CBE randSVD scheme.
// oversample: oversampling ratio ≥ 1.0.
SVDResult rand_svd(const DenseTensor& T, int split_at,
                   int target_rank, double oversample = 1.5,
                   const TruncParams& trunc = {});

// ── Hermitian eigendecomposition ──────────────────────────────────────────────
// Eigenvalues returned in ascending order.
struct EigenResult {
    Eigen::VectorXd eigenvalues;
    DenseTensor     eigenvectors;
};

EigenResult eigh(const DenseTensor& T);

// ── Matrix-exponential times vector (Krylov approximation) ───────────────────
// Computes exp(α * H) * v  where H is Hermitian (used by TDVP).
DenseTensor matrix_exp_times_vec(const DenseTensor& H,
                                  const DenseTensor& v,
                                  std::complex<double> alpha,
                                  double tol = 1e-10);

// ── Utility ───────────────────────────────────────────────────────────────────
double von_neumann_entropy(const Eigen::VectorXd& singular_values);

} // namespace tenet
