// src/environment/env_push.cpp
//
// push_right / push_left: extend a sparse environment by one site.
//
// Tensor leg conventions (derived from Julia @tensor reference):
//   MPSTensor ket:  (D_l, d, D_r)          legs 0,1,2
//   MPSTensor bra:  (D_r, D_l, d) conjg    legs 0,1,2  (from adjoint())
//   LeftEnvTensor:  (D_bra, D_ket)          legs 0,1
//   RightEnvTensor: (D_ket, D_bra)          legs 0,1
//
// push_right: new_EL[j] = sum_i contract(EL[i], ket, H[i,j], bra)
// push_left:  new_ER[i] = sum_j contract(ket, H[i,j], bra, ER[j])
//
// Parallelism (§14.2 of design doc):
//   Non-zero MPO entries are independent → parallelise over them with OpenMP.
//   Heavy tensor contractions run in parallel; accumulation into per-slot
//   accumulators is protected by a named critical section.

#include "tenet/environment/env_push.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <cassert>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenet {

// ── Helpers ──────────────────────────────────────────────────────────────────

static DenseTensor matrix_to_tensor(const Eigen::MatrixXcd& m) {
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    return DenseTensor::from_matrix(m, {TrivialSpace(rows), TrivialSpace(cols)}, true);
}

// ── push_right helpers ────────────────────────────────────────────────────────
//
// Julia: @tensor tmp[-1;-2] = A.A[3,4,-2] * mpot.A[2,4] * B.A[-1,1,2] * EnvL.A[1,3]
// Result: new LeftEnvTensor (D_r_bra, D_r_ket)
//
static DenseTensor push_right_local(const DenseTensor& el,
                                     const DenseTensor& ket,
                                     const Eigen::MatrixXcd& h_mat,
                                     const DenseTensor& bra)
{
    DenseTensor h_t = matrix_to_tensor(h_mat);             // (d_out, d_in)
    DenseTensor T1  = contract(el, ket, {{1, 0}});          // (D_bra, d_in, D_r)
    DenseTensor T2  = contract(T1, h_t, {{1, 1}});          // (D_bra, D_r, d_out)
    T2 = T2.permute({0, 2, 1});                             // (D_bra, d_out, D_r)
    DenseTensor T_res = contract(T2, bra, {{0, 1}, {1, 2}}); // (D_r, D_r_bra)
    return T_res.permute({1, 0});                           // (D_r_bra, D_r)
}

// Julia: @tensor tmp[-1;-2] = A.A[3,2,-2] * B.A[-1,1,2] * EnvL.A[1,3]
static DenseTensor push_right_identity(const DenseTensor& el,
                                        const DenseTensor& ket,
                                        const DenseTensor& bra)
{
    DenseTensor T1    = contract(el, ket, {{1, 0}});          // (D_bra, d, D_r)
    DenseTensor T_res = contract(T1, bra, {{0, 1}, {1, 2}}); // (D_r, D_r_bra)
    return T_res.permute({1, 0});                             // (D_r_bra, D_r)
}

// ── push_left helpers ─────────────────────────────────────────────────────────
//
// Julia: @tensor tmp[-1;-2] = A.A[-1,2,1] * mpot.A[4,2] * B.A[3,-2,4] * EnvR.A[1,3]
// Result: new RightEnvTensor (D_l_ket, D_l_bra) = (D_ket, D_bra)
//
static DenseTensor push_left_local(const DenseTensor& ket,
                                    const Eigen::MatrixXcd& h_mat,
                                    const DenseTensor& bra,
                                    const DenseTensor& er)
{
    DenseTensor h_t = matrix_to_tensor(h_mat);               // (d_out, d_in)
    DenseTensor T1  = contract(ket, er, {{2, 0}});            // (D_l, d_in, D_r_bra)
    DenseTensor T2  = contract(T1, h_t, {{1, 1}});            // (D_l, D_r_bra, d_out)
    T2 = T2.permute({0, 2, 1});                               // (D_l, d_out, D_r_bra)
    return contract(T2, bra, {{2, 0}, {1, 2}});               // (D_l, D_l_bra)
}

// Julia: @tensor tmp[-1;-2] = A.A[-1,2,1] * B.A[3,-2,2] * EnvR.A[1,3]
static DenseTensor push_left_identity(const DenseTensor& ket,
                                       const DenseTensor& bra,
                                       const DenseTensor& er)
{
    DenseTensor T1 = contract(ket, er, {{2, 0}});              // (D_l, d, D_r_bra)
    return contract(T1, bra, {{2, 0}, {1, 2}});                // (D_l, D_l_bra)
}

// ── Public API ────────────────────────────────────────────────────────────────

template<>
SparseLeftEnvTensor<DenseBackend>
push_right(const SparseLeftEnvTensor<DenseBackend>& L,
           const MPSTensor<DenseBackend>&            psi_site,
           const SparseMPOTensor<DenseBackend>&      H_site)
{
    int D_in  = H_site.d_in();
    int D_out = H_site.d_out();
    assert(L.dim() == D_in);

    const DenseTensor& ket      = psi_site.data();
    DenseTensor        bra_data = psi_site.adjoint().data();

    // Collect non-zero (row, col, op*) triplets — serial, fast
    using Task = std::tuple<int, int, const AbstractLocalOperator<DenseBackend>*>;
    std::vector<Task> tasks;
    tasks.reserve(D_in * D_out);
    for (int i = 0; i < D_in; ++i) {
        if (!L.has(i)) continue;
        for (int j = 0; j < D_out; ++j) {
            if (!H_site.has(i, j)) continue;
            tasks.emplace_back(i, j, H_site(i, j));
        }
    }

    // Per-column accumulators
    std::vector<std::optional<DenseTensor>> accum(D_out);

    // Parallel over independent MPO entries; accumulation is critical
    #pragma omp parallel for schedule(dynamic) if(tasks.size() > 4)
    for (int t = 0; t < static_cast<int>(tasks.size()); ++t) {
        auto [i, j, op] = tasks[t];
        const DenseTensor& el = L[i]->data();
        DenseTensor contrib = op->is_identity()
            ? push_right_identity(el, ket, bra_data)
            : push_right_local(el, ket, op->matrix(), bra_data);

        #pragma omp critical(env_push_accum)
        {
            if (!accum[j].has_value())
                accum[j] = std::move(contrib);
            else
                accum[j]->axpby({1, 0}, {1, 0}, contrib);
        }
    }

    SparseLeftEnvTensor<DenseBackend> result(D_out);
    for (int j = 0; j < D_out; ++j) {
        if (accum[j].has_value())
            result.set(j, std::make_unique<LeftEnvTensor<DenseBackend>>(
                            std::move(*accum[j])));
    }
    return result;
}

template<>
SparseRightEnvTensor<DenseBackend>
push_left(const SparseRightEnvTensor<DenseBackend>& R,
          const MPSTensor<DenseBackend>&             psi_site,
          const SparseMPOTensor<DenseBackend>&       H_site)
{
    int D_in  = H_site.d_in();
    int D_out = H_site.d_out();
    assert(R.dim() == D_out);

    const DenseTensor& ket      = psi_site.data();
    DenseTensor        bra_data = psi_site.adjoint().data();

    // Collect non-zero (row, col, op*) triplets — serial, fast
    using Task = std::tuple<int, int, const AbstractLocalOperator<DenseBackend>*>;
    std::vector<Task> tasks;
    tasks.reserve(D_in * D_out);
    for (int i = 0; i < D_in; ++i) {
        for (int j = 0; j < D_out; ++j) {
            if (!H_site.has(i, j)) continue;
            if (!R.has(j)) continue;
            tasks.emplace_back(i, j, H_site(i, j));
        }
    }

    // Per-row accumulators
    std::vector<std::optional<DenseTensor>> accum(D_in);

    // Parallel over independent MPO entries; accumulation is critical
    #pragma omp parallel for schedule(dynamic) if(tasks.size() > 4)
    for (int t = 0; t < static_cast<int>(tasks.size()); ++t) {
        auto [i, j, op] = tasks[t];
        const DenseTensor& er = R[j]->data();
        DenseTensor contrib = op->is_identity()
            ? push_left_identity(ket, bra_data, er)
            : push_left_local(ket, op->matrix(), bra_data, er);

        #pragma omp critical(env_push_accum)
        {
            if (!accum[i].has_value())
                accum[i] = std::move(contrib);
            else
                accum[i]->axpby({1, 0}, {1, 0}, contrib);
        }
    }

    SparseRightEnvTensor<DenseBackend> result(D_in);
    for (int i = 0; i < D_in; ++i) {
        if (accum[i].has_value())
            result.set(i, std::make_unique<RightEnvTensor<DenseBackend>>(
                            std::move(*accum[i])));
    }
    return result;
}

} // namespace tenet
