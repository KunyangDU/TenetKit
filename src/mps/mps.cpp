// src/mps/mps.cpp
//
// DenseMPS<DenseBackend> non-inline methods.

#include "tenet/mps/mps.hpp"
#include "tenet/mps/canonical.hpp"
#include "tenet/core/tensor_ops.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <complex>

namespace tenet {

// ── Bond dimensions ───────────────────────────────────────────────────────────

template<>
int DenseMPS<DenseBackend>::bond_dim(int bond) const {
    assert(bond >= 0 && bond <= L_);
    if (bond == 0)  return (L_ > 0) ? sites_[0].left_dim()  : 1;
    if (bond == L_) return (L_ > 0) ? sites_[L_-1].right_dim() : 1;
    return sites_[bond].left_dim();
}

template<>
int DenseMPS<DenseBackend>::max_bond_dim() const {
    int maxD = 1;
    for (int i = 0; i < L_; ++i) {
        maxD = std::max(maxD, sites_[i].left_dim());
        maxD = std::max(maxD, sites_[i].right_dim());
    }
    return maxD;
}

// ── Inner product ⟨other|this⟩ ───────────────────────────────────────────────
// Contracts bra·ket from left to right via transfer matrix.

template<>
std::complex<double> DenseMPS<DenseBackend>::inner(const DenseMPS<DenseBackend>& other) const
{
    assert(L_ == other.L_);

    // Transfer matrix T[α', α] = 1×1 identity at start
    Eigen::MatrixXcd T = Eigen::MatrixXcd::Identity(1, 1);

    for (int i = 0; i < L_; ++i) {
        int d  = sites_[i].phys_dim();
        int dr      = sites_[i].right_dim();
        int bra_dr  = other.sites_[i].right_dim();

        // as_matrix_left: (dl*d) × dr, row index = α*d + σ
        Eigen::MatrixXcd bra_full = other.sites_[i].as_matrix_left();  // (dl*d) × bra_dr
        Eigen::MatrixXcd ket_full = sites_[i].as_matrix_left();         // (dl*d) × dr
        int dl = static_cast<int>(bra_full.rows()) / d;

        // T_new[β', β] = Σ_{α',α,σ} T[α',α] * conj(bra[α',σ,β']) * ket[α,σ,β]
        // Rewritten: T_new = Σ_σ bra_σ†  * T * ket_σ
        // where bra_σ[α', β'] = bra[α',σ,β']  and  ket_σ[α, β] = ket[α,σ,β]
        Eigen::MatrixXcd T_new = Eigen::MatrixXcd::Zero(bra_dr, dr);
        for (int sigma = 0; sigma < d; ++sigma) {
            // Extract σ-slice: rows with (row mod d == sigma) when ordered as (α, σ)
            // row = α * d + σ → for fixed σ, rows are {σ, d+σ, 2d+σ, ...}
            Eigen::MatrixXcd bra_sigma(dl, bra_dr);
            Eigen::MatrixXcd ket_sigma(dl, dr);
            for (int a = 0; a < dl; ++a) {
                bra_sigma.row(a) = bra_full.row(a * d + sigma);
                ket_sigma.row(a) = ket_full.row(a * d + sigma);
            }
            T_new += bra_sigma.adjoint() * T * ket_sigma;
        }
        T = T_new;
    }
    return T(0, 0);
}

template<>
double DenseMPS<DenseBackend>::norm() const {
    return std::sqrt(std::abs(inner(*this)));
}

template<>
void DenseMPS<DenseBackend>::normalize() {
    double n = norm();
    if (n > 0.0 && L_ > 0) {
        // Scale all elements of site 0 by 1/n
        DenseTensor zero_t = zeros_like(sites_[0].data());
        sites_[0].data().axpby(
            std::complex<double>(1.0 / n, 0.0),
            std::complex<double>(0.0, 0.0),
            zero_t);
    }
}

// ── Random MPS ───────────────────────────────────────────────────────────────

template<>
DenseMPS<DenseBackend>
DenseMPS<DenseBackend>::random(int L, int D, const std::vector<Space>& phys_spaces)
{
    assert(static_cast<int>(phys_spaces.size()) == L);
    DenseMPS<DenseBackend> psi(L);

    for (int i = 0; i < L; ++i) {
        // Compute bond dimensions capped at D
        int dl = 1, dr = 1;
        int d = phys_spaces[i].dim();

        if (i > 0) {
            // Left bond = right bond of previous site
            dl = psi[i-1].right_dim();
            // If this is the first site, dl = 1 (set below)
        }
        if (i == 0) {
            dl = 1;
            dr = std::min(D, d);
        } else if (i == L - 1) {
            dl = psi[i-1].right_dim();
            dr = 1;
        } else {
            dl = psi[i-1].right_dim();
            // Grow dimension up to D
            dr = std::min(D, dl * d);
        }

        Space vl(dl), vr(dr);
        psi[i] = MPSTensor<DenseBackend>(vl, phys_spaces[i], vr);
        psi[i].data() = DenseBackend::random({vl, phys_spaces[i], vr});
    }

    // Left-canonicalize to normalize
    psi.left_canonicalize(0, L - 1);
    psi.set_center(L - 1, L - 1);

    // Normalize the last site
    double n = psi.norm();
    if (n > 0.0) {
        DenseTensor zero_t = zeros_like(psi[L-1].data());
        psi[L-1].data().axpby(
            std::complex<double>(1.0 / n, 0.0),
            std::complex<double>(0.0, 0.0),
            zero_t);
    }

    return psi;
}

// ── Member function wrappers for canonicalization ────────────────────────────
// Implemented in canonical.cpp

} // namespace tenet
