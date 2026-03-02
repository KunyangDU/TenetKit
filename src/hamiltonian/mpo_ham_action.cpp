// src/hamiltonian/mpo_ham_action.cpp
//
// apply_H_eff_mpo: H_eff action on a DenseMPOTensor.
//
// Contraction (h acts on bra index):
//   T1 = contract(el_i, ρ[s], {{1,0}})        → (D_out_l, d_bra, d_ket, D_in_r)
//   T2 = contract(T1, h_t, {{1,1}})            → (D_out_l, d_ket, D_in_r, d_out)
//   T2 = T2.permute({0,3,1,2})                 → (D_out_l, d_out, d_ket, D_in_r)
//   result += contract(T2, er_j, {{3,0}})      → (D_out_l, d_out, d_ket, D_out_r)
//
// For identity h: T1 is already correct, just contract with er_j.

#include "tenet/hamiltonian/mpo_ham_action.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <cassert>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenet {

namespace {

static DenseTensor mat_to_tensor(const Eigen::MatrixXcd& m)
{
    return DenseTensor::from_matrix(
        m, {TrivialSpace(static_cast<int>(m.rows())),
             TrivialSpace(static_cast<int>(m.cols()))}, true);
}

// el: (D_out_l, D_in_l), rho: (D_in_l, d_bra, d_ket, D_in_r)
// h_mat: (d_out, d_in), er: (D_in_r, D_out_r)
// returns: (D_out_l, d_out, d_ket, D_out_r)
static DenseTensor action_mpo_local(const DenseTensor& el,
                                     const DenseTensor& rho,
                                     const Eigen::MatrixXcd& h_mat,
                                     const DenseTensor& er)
{
    DenseTensor h_t = mat_to_tensor(h_mat);
    DenseTensor T1  = contract(el, rho, {{1, 0}});          // (D_out_l, d_bra, d_ket, D_in_r)
    DenseTensor T2  = contract(T1, h_t, {{1, 1}});          // (D_out_l, d_ket, D_in_r, d_out)
    T2 = T2.permute({0, 3, 1, 2});                          // (D_out_l, d_out, d_ket, D_in_r)
    return contract(T2, er, {{3, 0}});                      // (D_out_l, d_out, d_ket, D_out_r)
}

static DenseTensor action_mpo_identity(const DenseTensor& el,
                                        const DenseTensor& rho,
                                        const DenseTensor& er)
{
    DenseTensor T1 = contract(el, rho, {{1, 0}});            // (D_out_l, d_bra, d_ket, D_in_r)
    return contract(T1, er, {{3, 0}});                       // (D_out_l, d_bra, d_ket, D_out_r)
}

} // anonymous namespace

template<>
DenseTensor
apply_H_eff_mpo(const SparseLeftEnvTensor<DenseBackend>&  L_env,
                const DenseMPOTensor<DenseBackend>&         rho_site,
                const SparseMPOTensor<DenseBackend>&        H_site,
                const SparseRightEnvTensor<DenseBackend>&   R_env)
{
    int D_in  = H_site.d_in();
    int D_out = H_site.d_out();
    assert(L_env.dim() == D_in);
    assert(R_env.dim() == D_out);

    const DenseTensor& rho = rho_site.data();

    using Task = std::tuple<int, int, const AbstractLocalOperator<DenseBackend>*>;
    std::vector<Task> tasks;
    for (int i = 0; i < D_in; ++i) {
        if (!L_env.has(i)) continue;
        for (int j = 0; j < D_out; ++j) {
            if (!R_env.has(j)) continue;
            if (!H_site.has(i, j)) continue;
            tasks.emplace_back(i, j, H_site(i, j));
        }
    }

    DenseTensor result(rho.spaces());

    #pragma omp parallel for schedule(dynamic) if(tasks.size() > 4)
    for (int t = 0; t < static_cast<int>(tasks.size()); ++t) {
        auto [i, j, op] = tasks[t];
        const DenseTensor& el = L_env[i]->data();
        const DenseTensor& er = R_env[j]->data();
        DenseTensor contrib = op->is_identity()
            ? action_mpo_identity(el, rho, er)
            : action_mpo_local(el, rho, op->matrix(), er);

        #pragma omp critical(mpo_ham_action_accum)
        {
            result.axpby({1, 0}, {1, 0}, contrib);
        }
    }

    return result;
}

} // namespace tenet
