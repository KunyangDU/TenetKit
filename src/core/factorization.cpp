// src/core/factorization.cpp
//
// QR, SVD, eigh, and related factorizations for DenseTensor.

#include "tenet/core/factorization.hpp"
#include "tenet/core/tensor_ops.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace tenet {

// ── Helper: build row_legs and col_legs from split_at ────────────────────────
static void split_legs(const DenseTensor& T, int split_at,
                       std::vector<int>& row_legs, std::vector<int>& col_legs)
{
    row_legs.clear(); col_legs.clear();
    for (int i = 0; i < split_at;   ++i) row_legs.push_back(i);
    for (int i = split_at; i < T.rank(); ++i) col_legs.push_back(i);
}

// ── QR decomposition ──────────────────────────────────────────────────────────

QRResult qr(const DenseTensor& T, int split_at) {
    std::vector<int> row_legs, col_legs;
    split_legs(T, split_at, row_legs, col_legs);

    Eigen::MatrixXcd M = T.matricize(row_legs, col_legs);
    int m = static_cast<int>(M.rows());
    int n = static_cast<int>(M.cols());
    int k = std::min(m, n);

    // Thin QR via Householder
    Eigen::HouseholderQR<Eigen::MatrixXcd> decomp(M);

    // Extract Q: m×k
    Eigen::MatrixXcd Q = decomp.householderQ() * Eigen::MatrixXcd::Identity(m, k);

    // Extract R: k×n (upper triangular, zero out lower)
    Eigen::MatrixXcd R_full = decomp.matrixQR().topRows(k).template triangularView<Eigen::Upper>();
    Eigen::MatrixXcd R = R_full;

    TrivialSpace bond(k);

    std::vector<TrivialSpace> Q_legs, R_legs;
    for (int i : row_legs) Q_legs.push_back(T.space(i));
    Q_legs.push_back(bond);

    R_legs.push_back(bond);
    for (int i : col_legs) R_legs.push_back(T.space(i));

    return {DenseTensor::from_matrix(Q, std::move(Q_legs), true),
            DenseTensor::from_matrix(R, std::move(R_legs), true),
            k};
}

QRResult rq(const DenseTensor& T, int split_at) {
    std::vector<int> row_legs, col_legs;
    split_legs(T, split_at, row_legs, col_legs);

    Eigen::MatrixXcd M = T.matricize(row_legs, col_legs);
    int m = static_cast<int>(M.rows());
    int n = static_cast<int>(M.cols());
    int k = std::min(m, n);

    // RQ of M = QR of M†
    Eigen::HouseholderQR<Eigen::MatrixXcd> decomp(M.adjoint());  // n×m

    // Q_new: n×k
    Eigen::MatrixXcd Q_new = decomp.householderQ() * Eigen::MatrixXcd::Identity(n, k);
    // R_new: k×m
    Eigen::MatrixXcd R_new = decomp.matrixQR().topRows(k).template triangularView<Eigen::Upper>();

    // M = R_new† * Q_new†
    Eigen::MatrixXcd R_mat = R_new.adjoint();  // m×k
    Eigen::MatrixXcd Q_mat = Q_new.adjoint();  // k×n

    TrivialSpace bond(k);

    // QRResult convention: Q is right-orthogonal, R is left part
    std::vector<TrivialSpace> R_legs, Q_legs;
    for (int i : row_legs) R_legs.push_back(T.space(i));
    R_legs.push_back(bond);

    Q_legs.push_back(bond);
    for (int i : col_legs) Q_legs.push_back(T.space(i));

    return {DenseTensor::from_matrix(Q_mat, std::move(Q_legs), true),   // Q: bond × col
            DenseTensor::from_matrix(R_mat, std::move(R_legs), true),   // R: row × bond
            k};
}

// ── SVD ───────────────────────────────────────────────────────────────────────

SVDResult svd(const DenseTensor& T, int split_at, const TruncParams& trunc) {
    std::vector<int> row_legs, col_legs;
    split_legs(T, split_at, row_legs, col_legs);

    Eigen::MatrixXcd M = T.matricize(row_legs, col_legs);

    Eigen::BDCSVD<Eigen::MatrixXcd> solver(M, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::VectorXd S = solver.singularValues();   // descending
    Eigen::MatrixXcd U = solver.matrixU();
    Eigen::MatrixXcd V = solver.matrixV();

    int k = static_cast<int>(S.size());
    double trunc_err = 0.0;

    // Apply truncation
    if ((trunc.maxD > 0 || trunc.cutoff > 0) && k > 0) {
        int k_keep = k;

        // Cutoff threshold
        if (trunc.cutoff > 0 && S(0) > 0.0) {
            double thr = trunc.cutoff * S(0);
            for (int i = 0; i < k; ++i) {
                if (S(i) < thr) { k_keep = i; break; }
            }
        }

        // MaxD
        if (trunc.maxD > 0) k_keep = std::min(k_keep, trunc.maxD);
        k_keep = std::max(k_keep, 1);  // keep at least 1

        // Truncation error = Σ_{i>=k_keep} σ_i^2
        for (int i = k_keep; i < k; ++i) trunc_err += S(i) * S(i);

        S = S.head(k_keep);
        U = U.leftCols(k_keep);
        V = V.leftCols(k_keep);
        k = k_keep;
    }

    if (trunc.normalize && S.norm() > 0) S /= S.norm();

    TrivialSpace bond(k);

    std::vector<TrivialSpace> U_legs, Vt_legs;
    for (int i : row_legs) U_legs.push_back(T.space(i));
    U_legs.push_back(bond);

    Vt_legs.push_back(bond);
    for (int i : col_legs) Vt_legs.push_back(T.space(i));

    return {DenseTensor::from_matrix(U, std::move(U_legs), true),
            S,
            DenseTensor::from_matrix(V.adjoint(), std::move(Vt_legs), true),
            k,
            trunc_err};
}

// ── Randomised SVD ────────────────────────────────────────────────────────────

SVDResult rand_svd(const DenseTensor& T, int split_at,
                   int target_rank, double oversample,
                   const TruncParams& trunc)
{
    std::vector<int> row_legs, col_legs;
    split_legs(T, split_at, row_legs, col_legs);

    Eigen::MatrixXcd M = T.matricize(row_legs, col_legs);
    int m = static_cast<int>(M.rows());
    int n = static_cast<int>(M.cols());
    int l = std::min(static_cast<int>(target_rank * oversample) + 4, std::min(m, n));

    // Random Gaussian sketch
    static thread_local std::mt19937_64 eng{std::random_device{}()};
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXcd Omega(n, l);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < l; ++j)
            Omega(i, j) = {nd(eng), nd(eng)};

    Eigen::MatrixXcd Y = M * Omega;
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr_Y(Y);
    Eigen::MatrixXcd Q = qr_Y.householderQ() * Eigen::MatrixXcd::Identity(m, l);

    Eigen::MatrixXcd B = Q.adjoint() * M;
    Eigen::BDCSVD<Eigen::MatrixXcd> solver(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::VectorXd S = solver.singularValues();
    Eigen::MatrixXcd U_B = solver.matrixU();
    Eigen::MatrixXcd V_B = solver.matrixV();

    Eigen::MatrixXcd U = Q * U_B;

    int k = std::min(target_rank, static_cast<int>(S.size()));
    double trunc_err = 0.0;
    for (int i = k; i < static_cast<int>(S.size()); ++i) trunc_err += S(i) * S(i);
    S = S.head(k); U = U.leftCols(k); V_B = V_B.leftCols(k);

    if (trunc.maxD > 0 && k > trunc.maxD) {
        for (int i = trunc.maxD; i < k; ++i) trunc_err += S(i) * S(i);
        k = trunc.maxD;
        S = S.head(k); U = U.leftCols(k); V_B = V_B.leftCols(k);
    }

    TrivialSpace bond(k);
    std::vector<TrivialSpace> U_legs, Vt_legs;
    for (int i : row_legs) U_legs.push_back(T.space(i));
    U_legs.push_back(bond);
    Vt_legs.push_back(bond);
    for (int i : col_legs) Vt_legs.push_back(T.space(i));

    return {DenseTensor::from_matrix(U, std::move(U_legs), true),
            S,
            DenseTensor::from_matrix(V_B.adjoint(), std::move(Vt_legs), true),
            k,
            trunc_err};
}

// ── Hermitian eigendecomposition ──────────────────────────────────────────────

EigenResult eigh(const DenseTensor& T) {
    // T must be rank-2 square Hermitian
    assert(T.rank() == 2);
    assert(T.space(0).dim() == T.space(1).dim());

    int n = T.space(0).dim();
    Eigen::MatrixXcd M = T.matricize({0}, {1});

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(M);
    Eigen::VectorXd evals = solver.eigenvalues();  // ascending
    Eigen::MatrixXcd evecs = solver.eigenvectors();

    // Eigenvectors as DenseTensor: shape (n × n)
    TrivialSpace bond(n);
    std::vector<TrivialSpace> evec_legs = {TrivialSpace(n), TrivialSpace(n)};

    return {evals, DenseTensor::from_matrix(evecs, std::move(evec_legs), true)};
}

// ── Matrix exponential times vector ──────────────────────────────────────────

DenseTensor matrix_exp_times_vec(const DenseTensor& H,
                                  const DenseTensor& v,
                                  std::complex<double> alpha,
                                  double /*tol*/)
{
    // H must be rank-2 square; v rank-1
    assert(H.rank() == 2);
    int n = H.space(0).dim();
    Eigen::MatrixXcd M = H.matricize({0}, {1});
    Eigen::MatrixXcd alphaM = alpha * M;
    Eigen::MatrixXcd Mv = alphaM.exp();

    // v as column vector
    Eigen::VectorXcd vv(n);
    for (int i = 0; i < n; ++i) vv(i) = v.data()[i];

    Eigen::VectorXcd result = Mv * vv;

    std::vector<DenseTensor::Scalar> data(n);
    for (int i = 0; i < n; ++i) data[i] = result(i);

    return DenseTensor(v.spaces(), std::move(data));
}

// ── Von Neumann entropy ───────────────────────────────────────────────────────

double von_neumann_entropy(const Eigen::VectorXd& singular_values) {
    // S = -Σ p_i log(p_i) where p_i = σ_i^2 / Σ σ_j^2
    double norm_sq = singular_values.squaredNorm();
    if (norm_sq < 1e-15) return 0.0;

    double entropy = 0.0;
    for (int i = 0; i < singular_values.size(); ++i) {
        double p = (singular_values(i) * singular_values(i)) / norm_sq;
        if (p > 1e-15) entropy -= p * std::log(p);
    }
    return entropy;
}

} // namespace tenet
