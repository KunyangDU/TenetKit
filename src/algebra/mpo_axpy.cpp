// src/algebra/mpo_axpy.cpp
//
// mpo_axpy: y ← α · x + y  using direct-sum MPO construction + optional SVD compression.
//
// Algorithm:
//   1. Save y_old = y.
//   2. Build direct-sum MPO: y_new = α*x ⊕ y_old (block-diagonal concatenation).
//      At s=0:     shape (1, d, d, Dxr+Dyr)       -- horizontal concat of x and y_old cols
//      At 0<s<L-1: shape (Dxl+Dyl, d, d, Dxr+Dyr) -- block diagonal
//      At s=L-1:   shape (Dxl+Dyl, d, d, 1)       -- vertical concat of x and y_old rows
//   3. If truncation requested, apply L2R SVD sweep to compress bond dims.
//
// This approach is exact (no variational approximation) and avoids the metric
// mismatch that plagues variational ALS when environments are not identity matrices.

#include "tenet/algebra/mpo_axpy.hpp"
#include "tenet/core/factorization.hpp"
#include "tenet/core/space.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <complex>
#include <vector>

namespace tenet {

template<>
double mpo_axpy(double                          alpha,
                const DenseMPO<DenseBackend>&   x,
                DenseMPO<DenseBackend>&          y,
                const AxpyConfig&                cfg)
{
    const int L = x.length();
    assert(L == y.length());
    assert(L >= 1);

    // Snapshot of the original y (will be overwritten in-place)
    DenseMPO<DenseBackend> y_old = y;

    // ── Step 1: Direct-sum MPO construction ──────────────────────────────────────
    //
    // For each site s, the new tensor is the block-diagonal "direct sum" of
    // α·x[s] (top-left block) and y_old[s] (bottom-right block), with boundary
    // adaptations at s=0 and s=L-1 to respect the D_l=1 / D_r=1 conventions.

    for (int s = 0; s < L; ++s) {
        const DenseTensor& Xs = x[s].data();      // (Dxl, d, d, Dxr)
        const DenseTensor& Ys = y_old[s].data();  // (Dyl, d, d, Dyr)

        const int Dxl = Xs.dim(0);
        const int d   = Xs.dim(1);   // d_bra (== d_ket for density-matrix MPOs)
        const int Dxr = Xs.dim(3);
        const int Dyl = Ys.dim(0);
        const int Dyr = Ys.dim(3);

        // New bond dimensions
        const int Dnl = (s == 0)     ? 1          : (Dxl + Dyl);
        const int Dnr = (s == L - 1) ? 1          : (Dxr + Dyr);

        // Allocate zero-initialised data (row-major: last index varies fastest)
        const std::size_t total = static_cast<std::size_t>(Dnl) * d * d * Dnr;
        std::vector<std::complex<double>> data(total, {0.0, 0.0});

        auto flat = [&](int il, int ib, int ik, int ir) -> std::size_t {
            return static_cast<std::size_t>(il) * (d * d * Dnr)
                 + static_cast<std::size_t>(ib) * (d * Dnr)
                 + static_cast<std::size_t>(ik) * Dnr
                 + static_cast<std::size_t>(ir);
        };

        // Fill x[s] block (scaled by α only at the LEFT boundary s=0).
        //
        // In an MPO direct sum, the scalar α must appear exactly once along each
        // path through the x-bond indices.  We place it at s=0 so that
        // contracting the entire chain gives α·x[0]·x[1]·…·x[L-1] = α·x.
        // At s=0 both x and y_old share D_l=1 → nil=0 for all contributions.
        // At s=L-1 both share D_r=1 → nir=0 for all contributions.
        const double scale = (s == 0) ? alpha : 1.0;
        for (int il = 0; il < Dxl; ++il)
            for (int ib = 0; ib < d; ++ib)
                for (int ik = 0; ik < d; ++ik)
                    for (int ir = 0; ir < Dxr; ++ir) {
                        const int nil = (s == 0)     ? 0 : il;
                        const int nir = (s == L - 1) ? 0 : ir;
                        data[flat(nil, ib, ik, nir)] +=
                            scale * Xs({il, ib, ik, ir});
                    }

        // Fill y_old[s] block (offset by Dxl / Dxr at interior sites)
        for (int il = 0; il < Dyl; ++il)
            for (int ib = 0; ib < d; ++ib)
                for (int ik = 0; ik < d; ++ik)
                    for (int ir = 0; ir < Dyr; ++ir) {
                        const int nil = (s == 0)     ? 0         : (Dxl + il);
                        const int nir = (s == L - 1) ? 0         : (Dxr + ir);
                        data[flat(nil, ib, ik, nir)] += Ys({il, ib, ik, ir});
                    }

        std::vector<TrivialSpace> spaces = {
            TrivialSpace(Dnl), TrivialSpace(d), TrivialSpace(d), TrivialSpace(Dnr)
        };
        y[s].data() = DenseTensor(std::move(spaces), std::move(data));
    }

    // ── Step 2: L2R SVD compression sweep (only if truncation is requested) ──────
    //
    // A single left-to-right SVD sweep puts the MPO in left-canonical form and
    // truncates the bond dimension according to cfg.trunc.  The result is exact
    // when no truncation is applied.

    if (cfg.trunc.maxD > 0 || cfg.trunc.cutoff > 0) {
        for (int s = 0; s < L - 1; ++s) {
            const int D_l      = y[s].D_l();
            const int d        = y[s].d();
            const int D_r_next = y[s + 1].D_r();

            // SVD: (D_l, d, d, D_r) split at 3 → U (D_l*d*d × k), S (k), Vt (k × D_r)
            SVDResult res = svd(y[s].data(), 3, cfg.trunc);
            const int k = res.bond_dim;

            // y[s] ← left-canonical U reshaped to (D_l, d, d, k)
            y[s] = DenseMPOTensor<DenseBackend>::from_left_matrix(
                res.U.matricize({0, 1, 2}, {3}), D_l, d, k);

            // Absorb S·Vt into y[s+1]:
            //   SVt: k × D_r
            //   Y_next (current y[s+1] as matrix): D_r × (d*d*D_r_next)
            //   New_Y = SVt * Y_next: k × (d*d*D_r_next)
            Eigen::MatrixXcd SVt =
                res.S.asDiagonal() * res.Vt.matricize({0}, {1});       // k × D_r
            Eigen::MatrixXcd Y_next =
                y[s + 1].data().matricize({0}, {1, 2, 3});              // D_r × (d²·D_r_next)
            Eigen::MatrixXcd New_Y = SVt * Y_next;                      // k × (d²·D_r_next)

            y[s + 1] = DenseMPOTensor<DenseBackend>::from_right_matrix(
                New_Y, k, d, D_r_next);
        }
    }

    return 0.0;   // Direct-sum gives exact result; no iterative residual.
}

} // namespace tenet
