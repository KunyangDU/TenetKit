// src/mps/canonical.cpp
//
// MPSTensor canonicalization methods and DenseMPS sweep functions.

#include "tenet/mps/canonical.hpp"
#include "tenet/mps/mps.hpp"
#include "tenet/core/factorization.hpp"
#include "tenet/core/tensor_ops.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <vector>

namespace tenet {

// ── MPSTensor<DenseBackend> construction ──────────────────────────────────────

template<>
MPSTensor<DenseBackend>::MPSTensor(Space vl, Space phys, Space vr)
{
    data_ = DenseBackend::zeros({vl, phys, vr});
}

// ── MPSTensor<DenseBackend> shape ─────────────────────────────────────────────

template<> int MPSTensor<DenseBackend>::left_dim()  const { return data_.space(0).dim(); }
template<> int MPSTensor<DenseBackend>::phys_dim()  const { return data_.space(1).dim(); }
template<> int MPSTensor<DenseBackend>::right_dim() const { return data_.space(2).dim(); }

template<> const MPSTensor<DenseBackend>::Space& MPSTensor<DenseBackend>::left_space()  const { return data_.space(0); }
template<> const MPSTensor<DenseBackend>::Space& MPSTensor<DenseBackend>::phys_space()  const { return data_.space(1); }
template<> const MPSTensor<DenseBackend>::Space& MPSTensor<DenseBackend>::right_space() const { return data_.space(2); }

// ── Matrix views ─────────────────────────────────────────────────────────────

template<>
Eigen::MatrixXcd MPSTensor<DenseBackend>::as_matrix_left() const {
    // (D_l * d) × D_r
    return data_.matricize({0, 1}, {2});
}

template<>
Eigen::MatrixXcd MPSTensor<DenseBackend>::as_matrix_right() const {
    // D_l × (d * D_r)
    return data_.matricize({0}, {1, 2});
}

// ── Left canonicalize ─────────────────────────────────────────────────────────
// QR of as_matrix_left(); data becomes Q; returns R.

template<>
Eigen::MatrixXcd MPSTensor<DenseBackend>::left_canonicalize() {
    Eigen::MatrixXcd M = as_matrix_left();
    int m = static_cast<int>(M.rows());
    int n = static_cast<int>(M.cols());
    int k = std::min(m, n);

    Eigen::HouseholderQR<Eigen::MatrixXcd> decomp(M);
    Eigen::MatrixXcd Q = decomp.householderQ() * Eigen::MatrixXcd::Identity(m, k);
    Eigen::MatrixXcd R = decomp.matrixQR().topRows(k).template triangularView<Eigen::Upper>();

    // Update data: shape (D_l, d, D_r) → (D_l, d, k) → reshape
    TrivialSpace new_vl = data_.space(0);
    TrivialSpace new_phys = data_.space(1);
    TrivialSpace new_vr(k);
    data_ = DenseTensor::from_matrix(Q, {new_vl, new_phys, new_vr}, true);

    return R;  // k × D_r
}

// ── Right canonicalize ────────────────────────────────────────────────────────
// RQ of as_matrix_right(); data becomes Q (right-orthogonal); returns L.

template<>
Eigen::MatrixXcd MPSTensor<DenseBackend>::right_canonicalize() {
    Eigen::MatrixXcd M = as_matrix_right();
    int m = static_cast<int>(M.rows());
    int n = static_cast<int>(M.cols());
    int k = std::min(m, n);

    // RQ via QR of M†: M† = Q_new * R_new → M = R_new† * Q_new†
    Eigen::HouseholderQR<Eigen::MatrixXcd> decomp(M.adjoint());  // n × m
    Eigen::MatrixXcd Q_new = decomp.householderQ() * Eigen::MatrixXcd::Identity(n, k);  // n × k
    Eigen::MatrixXcd R_new = decomp.matrixQR().topRows(k).template triangularView<Eigen::Upper>(); // k × m

    Eigen::MatrixXcd L = R_new.adjoint();  // m × k  (left part, absorbed to the left)
    Eigen::MatrixXcd Q = Q_new.adjoint();  // k × n  (right-orthogonal)

    // Update data: shape (k, d, D_r)
    TrivialSpace new_vl(k);
    TrivialSpace new_phys = data_.space(1);
    TrivialSpace new_vr = data_.space(2);
    data_ = DenseTensor::from_matrix(Q, {new_vl, new_phys, new_vr}, true);

    return L;  // D_l × k (absorbed into site to the left)
}

// ── From matrix constructors ──────────────────────────────────────────────────

template<>
MPSTensor<DenseBackend>
MPSTensor<DenseBackend>::from_left_matrix(const Eigen::MatrixXcd& mat,
                                           Space vl, Space phys, Space vr)
{
    MPSTensor<DenseBackend> t;
    t.data_ = DenseTensor::from_matrix(mat, {vl, phys, vr}, true);
    return t;
}

template<>
MPSTensor<DenseBackend>
MPSTensor<DenseBackend>::from_right_matrix(const Eigen::MatrixXcd& mat,
                                            Space vl, Space phys, Space vr)
{
    MPSTensor<DenseBackend> t;
    t.data_ = DenseTensor::from_matrix(mat, {vl, phys, vr}, true);
    return t;
}

// ── Adjoint ───────────────────────────────────────────────────────────────────
// Returns tensor with legs (D_r, D_l, d) and conjugated values.
// This matches Julia's AdjointMPSTensor convention.

template<>
MPSTensor<DenseBackend> MPSTensor<DenseBackend>::adjoint() const {
    MPSTensor<DenseBackend> t;
    // Permute (D_l, d, D_r) → (D_r, D_l, d) and conjugate
    t.data_ = data_.permute({2, 0, 1}).conj();
    return t;
}

// ── DenseMPS canonicalization (free functions) ────────────────────────────────

// Left-canonicalize sites [from, to): for each site, QR, push boundary to right.
template<>
void left_canonicalize(DenseMPS<DenseBackend>& psi, int from, int to) {
    for (int site = from; site < to; ++site) {
        Eigen::MatrixXcd R = psi[site].left_canonicalize();  // modifies psi[site]

        // Absorb R into psi[site+1] from the left
        if (site + 1 < psi.length()) {
            // psi[site+1] data has shape (D_l, d, D_r); multiply R from left on D_l
            Eigen::MatrixXcd M = psi[site + 1].as_matrix_right();
            // M is D_l × (d * D_r); D_l of site+1 may equal cols of R
            // new shape: (k, d, D_r) where k = R.rows()
            Eigen::MatrixXcd new_M = R * M;
            int new_dl = static_cast<int>(new_M.rows());
            int dr = psi[site + 1].right_dim();
            int d  = psi[site + 1].phys_dim();
            psi[site + 1] = MPSTensor<DenseBackend>::from_right_matrix(
                new_M,
                TrivialSpace(new_dl),
                TrivialSpace(d),
                TrivialSpace(dr));
        }
    }
}

// Right-canonicalize sites (from, to]: for each site from `to` to `from+1`,
// RQ, push boundary to the left.
template<>
void right_canonicalize(DenseMPS<DenseBackend>& psi, int from, int to) {
    for (int site = to; site > from; --site) {
        Eigen::MatrixXcd L = psi[site].right_canonicalize();  // modifies psi[site]

        // Absorb L into psi[site-1] from the right
        if (site - 1 >= 0) {
            Eigen::MatrixXcd M = psi[site - 1].as_matrix_left();
            // M is (D_l * d) × D_r; D_r may equal rows of L
            Eigen::MatrixXcd new_M = M * L;
            int dl = psi[site - 1].left_dim();
            int d  = psi[site - 1].phys_dim();
            int new_dr = static_cast<int>(new_M.cols());
            psi[site - 1] = MPSTensor<DenseBackend>::from_left_matrix(
                new_M,
                TrivialSpace(dl),
                TrivialSpace(d),
                TrivialSpace(new_dr));
        }
    }
}

// Move orthogonality centre to target.
template<>
void move_center(DenseMPS<DenseBackend>& psi, int target) {
    int current = psi.center_left();
    if (target > current)
        left_canonicalize(psi, current, target);
    else if (target < current)
        right_canonicalize(psi, target, current);
    psi.set_center(target, target);
}

// Member function wrappers
template<>
void DenseMPS<DenseBackend>::left_canonicalize(int from, int to) {
    tenet::left_canonicalize(*this, from, to);
}

template<>
void DenseMPS<DenseBackend>::right_canonicalize(int from, int to) {
    tenet::right_canonicalize(*this, from, to);
}

template<>
void DenseMPS<DenseBackend>::move_center_to(int target) {
    tenet::move_center(*this, target);
}

} // namespace tenet
