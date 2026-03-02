// src/algebra/mpo_mul.cpp
//
// mpo_mul: compute C ← α · A · H  (direct product + L2R SVD truncation).
//
// Algorithm:
//   Step 1 — Full direct product (no truncation, bonds consistent by construction):
//     C_work[s] shape = (D_A_l * D_H_in, d, d, D_A_r * D_H_out)
//     for each (i,j) nonzero in H[s]:
//       C_work[s][al*D_H_in+i, σ', σ, ar*D_H_out+j]
//         += α * Σ_{σ''} A[s][al, σ', σ'', ar] · h_{i,j}(σ'', σ)
//
//   Step 2 — L2R SVD sweep to truncate (absorb bond into next site):
//     SVD C_work[s] at split=3 → U (left-canonical), S·Vt (bond)
//     Absorb R = S·Vt into C_work[s+1] via matricize({0},{1,2,3})

#include "tenet/algebra/mpo_mul.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/space.hpp"

#include <cassert>

namespace tenet {

namespace {

// Build the full direct product of one DenseMPO site with one SparseMPO site.
// Returns tensor of shape (D_A_l * D_H_in, d, d, D_A_r * D_H_out).
static DenseTensor mpo_site_product(const DenseTensor& A_site,
                                     const SparseMPOTensor<DenseBackend>& H_site,
                                     double alpha)
{
    int D_A_l   = A_site.dim(0);
    int d       = A_site.dim(1);
    int D_A_r   = A_site.dim(3);
    int D_H_in  = H_site.d_in();
    int D_H_out = H_site.d_out();

    int D_C_l = D_A_l * D_H_in;
    int D_C_r = D_A_r * D_H_out;

    DenseTensor C({TrivialSpace(D_C_l),
                   TrivialSpace(d),
                   TrivialSpace(d),
                   TrivialSpace(D_C_r)});

    for (int i = 0; i < D_H_in; ++i) {
        for (int j = 0; j < D_H_out; ++j) {
            if (!H_site.has(i, j)) continue;
            const auto* op = H_site(i, j);

            for (int al = 0; al < D_A_l; ++al) {
                int cl = al * D_H_in + i;
                for (int ar = 0; ar < D_A_r; ++ar) {
                    int cr = ar * D_H_out + j;
                    for (int sb = 0; sb < d; ++sb) {
                        for (int sk = 0; sk < d; ++sk) {
                            DenseTensor::Scalar val{0.0, 0.0};
                            if (op->is_identity()) {
                                val = A_site({al, sb, sk, ar});
                            } else {
                                const auto& mat = op->matrix();
                                for (int sk2 = 0; sk2 < d; ++sk2) {
                                    val += A_site({al, sb, sk2, ar}) *
                                           DenseTensor::Scalar{mat(sk2, sk).real(),
                                                               mat(sk2, sk).imag()};
                                }
                            }
                            C({cl, sb, sk, cr}) += val * DenseTensor::Scalar{alpha, 0.0};
                        }
                    }
                }
            }
        }
    }
    return C;
}

} // anonymous namespace

template<>
double mpo_mul(DenseMPO<DenseBackend>&       C,
               const DenseMPO<DenseBackend>&  A_in,
               const SparseMPO<DenseBackend>& H,
               double                         alpha,
               const MulConfig&               cfg)
{
    const int L = A_in.length();
    assert(L == H.length());
    assert(C.length() == L);

    // Step 1: full direct product (bonds consistent, no truncation)
    std::vector<DenseTensor> C_work(L);
    for (int s = 0; s < L; ++s)
        C_work[s] = mpo_site_product(A_in[s].data(), H[s], alpha);

    // Step 2: L2R SVD sweep to truncate and canonicalize
    for (int s = 0; s < L - 1; ++s) {
        int D_C_l      = C_work[s].dim(0);
        int d          = C_work[s].dim(1);
        int D_C_r_next = C_work[s + 1].dim(3);

        SVDResult svd_res = svd(C_work[s], 3, cfg.trunc);
        int D_new = svd_res.bond_dim;

        C[s] = DenseMPOTensor<DenseBackend>::from_left_matrix(
            svd_res.U.matricize({0, 1, 2}, {3}),
            D_C_l, d, D_new);

        // Bond R = S·Vt, shape (D_new, D_C_r(s))
        Eigen::MatrixXcd R = svd_res.S.asDiagonal()
                           * svd_res.Vt.matricize({0}, {1});

        // Absorb R into C_work[s+1]: new left bond = D_new
        Eigen::MatrixXcd M_next = C_work[s + 1].matricize({0}, {1, 2, 3});
        Eigen::MatrixXcd M_new  = R * M_next;

        C_work[s + 1] = DenseTensor::from_matrix(
            M_new,
            {TrivialSpace(D_new),
             TrivialSpace(d),
             TrivialSpace(d),
             TrivialSpace(D_C_r_next)},
            true);
    }

    // Last site: assign directly
    {
        int s     = L - 1;
        int D_C_l = C_work[s].dim(0);
        int d     = C_work[s].dim(1);
        int D_C_r = C_work[s].dim(3);
        C[s] = DenseMPOTensor<DenseBackend>::from_left_matrix(
            C_work[s].matricize({0, 1, 2}, {3}),
            D_C_l, d, D_C_r);
    }

    return 0.0;
}

} // namespace tenet
