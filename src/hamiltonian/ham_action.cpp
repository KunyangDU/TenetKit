// src/hamiltonian/ham_action.cpp
//
// apply():  H_eff |psi_site> for single-site DMRG.
// apply2(): H_eff |psi_ab>   for two-site DMRG.
//
// Single-site contraction (from Julia Hamiltonian/contract.jl):
//   With local op h:
//     x[-1,-2;-3] = El[-1,1] * psi[1,2,3] * h[-2,2] * Er[3,-3]
//   With identity:
//     x[-1,-2;-3] = El[-1,1] * psi[1,-2,2] * Er[2,-3]
//
// Two-site contraction:
//   With local ops hl, hr:
//     x[-1,-2,-3;-4] = El[-1,1]*psi[1,2,3,4]*hl[-2,2]*hr[-3,3]*Er[4,-4]

#include "tenet/hamiltonian/ham_action.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <cassert>

namespace tenet {

// ── Helpers ──────────────────────────────────────────────────────────────────

static DenseTensor matrix_to_tensor(const Eigen::MatrixXcd& m) {
    return DenseTensor::from_matrix(
        m, {TrivialSpace(static_cast<int>(m.rows())),
             TrivialSpace(static_cast<int>(m.cols()))}, true);
}

// El:(D_bra,D_ket), psi:(D_l,d_in,D_r), h:(d_out,d_in), Er:(D_ket,D_bra)
// → (D_bra, d_out, D_r_bra)
static DenseTensor action1_local(const DenseTensor& el,
                                  const DenseTensor& psi,
                                  const Eigen::MatrixXcd& h_mat,
                                  const DenseTensor& er)
{
    DenseTensor h_t = matrix_to_tensor(h_mat);
    DenseTensor T1  = contract(el, psi, {{1, 0}});          // (D_bra, d_in, D_r)
    DenseTensor T2  = contract(T1, h_t, {{1, 1}});          // (D_bra, D_r, d_out)
    T2 = T2.permute({0, 2, 1});                             // (D_bra, d_out, D_r)
    return contract(T2, er, {{2, 0}});                      // (D_bra, d_out, D_r_bra)
}

static DenseTensor action1_identity(const DenseTensor& el,
                                     const DenseTensor& psi,
                                     const DenseTensor& er)
{
    DenseTensor T1 = contract(el, psi, {{1, 0}});            // (D_bra, d, D_r)
    return contract(T1, er, {{2, 0}});                       // (D_bra, d, D_r_bra)
}

// psi_ab:(D_l,d_l,d_r,D_r)
// hl:(d_out_l,d_in_l), hr:(d_out_r,d_in_r)
// → (D_bra, d_out_l, d_out_r, D_r_bra)
static DenseTensor action2_local(const DenseTensor& el,
                                  const DenseTensor& psi_ab,
                                  const Eigen::MatrixXcd& hl_mat,
                                  const Eigen::MatrixXcd& hr_mat,
                                  const DenseTensor& er)
{
    DenseTensor hl_t = matrix_to_tensor(hl_mat);
    DenseTensor hr_t = matrix_to_tensor(hr_mat);
    DenseTensor T1 = contract(el, psi_ab, {{1, 0}});         // (D_bra, d_in_l, d_in_r, D_r)
    DenseTensor T2 = contract(T1, hl_t, {{1, 1}});           // (D_bra, d_in_r, D_r, d_out_l)
    T2 = T2.permute({0, 3, 1, 2});                           // (D_bra, d_out_l, d_in_r, D_r)
    DenseTensor T3 = contract(T2, hr_t, {{2, 1}});           // (D_bra, d_out_l, D_r, d_out_r)
    T3 = T3.permute({0, 1, 3, 2});                           // (D_bra, d_out_l, d_out_r, D_r)
    return contract(T3, er, {{3, 0}});                       // (D_bra, d_out_l, d_out_r, D_r_bra)
}

static DenseTensor action2_id_local(const DenseTensor& el,
                                     const DenseTensor& psi_ab,
                                     const Eigen::MatrixXcd& hr_mat,
                                     const DenseTensor& er)
{
    DenseTensor hr_t = matrix_to_tensor(hr_mat);
    DenseTensor T1 = contract(el, psi_ab, {{1, 0}});         // (D_bra, d_l, d_in_r, D_r)
    DenseTensor T2 = contract(T1, hr_t, {{2, 1}});           // (D_bra, d_l, D_r, d_out_r)
    T2 = T2.permute({0, 1, 3, 2});                           // (D_bra, d_l, d_out_r, D_r)
    return contract(T2, er, {{3, 0}});                       // (D_bra, d_l, d_out_r, D_r_bra)
}

static DenseTensor action2_local_id(const DenseTensor& el,
                                     const DenseTensor& psi_ab,
                                     const Eigen::MatrixXcd& hl_mat,
                                     const DenseTensor& er)
{
    DenseTensor hl_t = matrix_to_tensor(hl_mat);
    DenseTensor T1 = contract(el, psi_ab, {{1, 0}});         // (D_bra, d_in_l, d_r, D_r)
    DenseTensor T2 = contract(T1, hl_t, {{1, 1}});           // (D_bra, d_r, D_r, d_out_l)
    T2 = T2.permute({0, 3, 1, 2});                           // (D_bra, d_out_l, d_r, D_r)
    return contract(T2, er, {{3, 0}});                       // (D_bra, d_out_l, d_r, D_r_bra)
}

static DenseTensor action2_id_id(const DenseTensor& el,
                                  const DenseTensor& psi_ab,
                                  const DenseTensor& er)
{
    DenseTensor T1 = contract(el, psi_ab, {{1, 0}});          // (D_bra, d_l, d_r, D_r)
    return contract(T1, er, {{3, 0}});                        // (D_bra, d_l, d_r, D_r_bra)
}

// ── Public API ────────────────────────────────────────────────────────────────

template<>
DenseTensor apply(const SparseProjectiveHamiltonian<DenseBackend>& H_eff,
                  const DenseTensor& psi)
{
    assert(H_eff.H().has_value());
    const SparseMPO<DenseBackend>& H = **H_eff.H();
    const SparseMPOTensor<DenseBackend>& Hs = H[H_eff.site1()];

    DenseTensor result(psi.spaces());

    for (auto [i, j] : H_eff.valid_inds()) {
        const auto* el_ptr = H_eff.env_left()[i];
        const auto* er_ptr = H_eff.env_right()[j];
        if (!el_ptr || !er_ptr) continue;
        const auto* op = Hs(i, j);
        if (!op) continue;

        DenseTensor contrib = op->is_identity()
            ? action1_identity(el_ptr->data(), psi, er_ptr->data())
            : action1_local(el_ptr->data(), psi, op->matrix(), er_ptr->data());

        result.axpby({1, 0}, {1, 0}, contrib);
    }

    if (H_eff.energy_offset() != 0.0)
        result.axpby({1, 0}, {-H_eff.energy_offset(), 0}, psi);

    return result;
}

template<>
DenseTensor apply2(const SparseProjectiveHamiltonian<DenseBackend>& H_eff,
                   const DenseTensor& psi_ab)
{
    assert(H_eff.H().has_value());
    int s1 = H_eff.site1(), s2 = H_eff.site2();
    assert(s2 == s1 + 1);

    const SparseMPO<DenseBackend>&       H   = **H_eff.H();
    const SparseMPOTensor<DenseBackend>& Hs1 = H[s1];
    const SparseMPOTensor<DenseBackend>& Hs2 = H[s2];
    int D_mid = Hs1.d_out();

    DenseTensor result(psi_ab.spaces());

    for (auto [i, j] : H_eff.valid_inds()) {
        const auto* el_ptr = H_eff.env_left()[i];
        const auto* er_ptr = H_eff.env_right()[j];
        if (!el_ptr || !er_ptr) continue;
        const DenseTensor& el = el_ptr->data();
        const DenseTensor& er = er_ptr->data();

        for (int k = 0; k < D_mid; ++k) {
            if (!Hs1.has(i, k) || !Hs2.has(k, j)) continue;
            const auto* op1 = Hs1(i, k);
            const auto* op2 = Hs2(k, j);
            if (!op1 || !op2) continue;

            bool id1 = op1->is_identity(), id2 = op2->is_identity();
            DenseTensor contrib;
            if      (!id1 && !id2) contrib = action2_local(el, psi_ab, op1->matrix(), op2->matrix(), er);
            else if ( id1 && !id2) contrib = action2_id_local(el, psi_ab, op2->matrix(), er);
            else if (!id1 &&  id2) contrib = action2_local_id(el, psi_ab, op1->matrix(), er);
            else                   contrib = action2_id_id(el, psi_ab, er);

            result.axpby({1, 0}, {1, 0}, contrib);
        }
    }

    if (H_eff.energy_offset() != 0.0)
        result.axpby({1, 0}, {-H_eff.energy_offset(), 0}, psi_ab);

    return result;
}

} // namespace tenet
