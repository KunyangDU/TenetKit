// src/environment/mpo_env_push.cpp
//
// Three-layer push_right / push_left for MPOEnvironment.
//
// Layer layout: [ρ, H, ρ†]
//   Left env el_i:  (D_ρ†_l, D_ρ_l)   — index i over SparseMPO bond
//   Right env er_j: (D_ρ_r, D_ρ†_r)   — index j over SparseMPO bond
//
// push_right_mpo_local (MPO entry h_{i,j}):
//   new_el_j[β_R, α_R] = Σ el[β_L, α_L] · ρ[α_L,σ',σ,α_R] · h[σ'',σ'] · conj(ρ[β_L,σ'',σ,β_R])
//
// C++ derivation (indices: el=(β_L,α_L), ρ=(D_l,d_bra,d_ket,D_r), ρ†=(D_r,d_ket,d_bra,D_l)):
//   T1 = contract(el, rho, {{1,0}})         → (β_L, d_bra, d_ket, α_R)
//   T2 = contract(T1, h_t, {{1,1}})         → (β_L, d_ket, α_R, d_out)
//   T2 = T2.permute({0,3,2,1})              → (β_L, d_out, α_R, d_ket)
//   T3 = contract(T2, ρ†, {{0,3},{1,2},{3,1}}) → (α_R, β_R)
//   return T3.permute({1,0})                → (β_R, α_R)
//
// push_right_mpo_identity (h = I):
//   T1 = contract(el, rho, {{1,0}})         → (β_L, d_bra, d_ket, α_R)
//   T3 = contract(T1, ρ†, {{0,3},{1,2},{2,1}}) → (α_R, β_R)
//   return T3.permute({1,0})                → (β_R, α_R)
//
// push_left is the mirror: from er_j we build er_i contracting site s from the right.

#include "tenet/environment/mpo_environment.hpp"
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

namespace {

static DenseTensor mat_to_tensor(const Eigen::MatrixXcd& m)
{
    return DenseTensor::from_matrix(
        m, {TrivialSpace(static_cast<int>(m.rows())),
             TrivialSpace(static_cast<int>(m.cols()))}, true);
}

// push_right for a single MPO entry h_{i,j}
//   el:     (D_ρ†_l, D_ρ_l)
//   rho:    (D_ρ_l, d_bra, d_ket, D_ρ_r)
//   h_mat:  (d_out, d_in)  where d_in = d_bra, d_out = d_bra_new
//   rho_adj: rho.adjoint() = (D_ρ_r, d_ket, d_bra, D_ρ_l)†  [conj, perm {3,2,1,0}]
// returns: (D_ρ†_r, D_ρ_r)
static DenseTensor push_right_mpo_local(const DenseTensor& el,
                                         const DenseTensor& rho,
                                         const Eigen::MatrixXcd& h_mat,
                                         const DenseTensor& rho_adj)
{
    DenseTensor h_t = mat_to_tensor(h_mat);
    DenseTensor T1  = contract(el, rho, {{1, 0}});          // (β_L, d_bra, d_ket, α_R)
    DenseTensor T2  = contract(T1, h_t, {{1, 1}});          // (β_L, d_ket, α_R, d_out)
    T2 = T2.permute({0, 3, 2, 1});                          // (β_L, d_out, α_R, d_ket)
    // rho_adj shape: (D_ρ_r, d_ket, d_bra, D_ρ_l) [= (β_R, d_ket, d_bra=d_out, β_L)]
    DenseTensor T3 = contract(T2, rho_adj, {{0, 3}, {1, 2}, {3, 1}}); // (α_R, β_R)
    return T3.permute({1, 0});                              // (β_R, α_R) = (D_ρ†_r, D_ρ_r)
}

// push_right for identity MPO entry
static DenseTensor push_right_mpo_identity(const DenseTensor& el,
                                            const DenseTensor& rho,
                                            const DenseTensor& rho_adj)
{
    DenseTensor T1 = contract(el, rho, {{1, 0}});           // (β_L, d_bra, d_ket, α_R)
    // Contract with rho_adj over (β_L, d_bra, d_ket)
    DenseTensor T3 = contract(T1, rho_adj, {{0, 3}, {1, 2}, {2, 1}}); // (α_R, β_R)
    return T3.permute({1, 0});                              // (β_R, α_R)
}

// push_left for a single MPO entry h_{i,j}
//   rho:    (D_ρ_l, d_bra, d_ket, D_ρ_r)
//   h_mat:  (d_out, d_in)
//   rho_adj: (D_ρ_r, d_ket, d_bra, D_ρ_l)
//   er:     (D_ρ_r, D_ρ†_r)
// returns: (D_ρ_l, D_ρ†_l) = (α_L, β_L)
static DenseTensor push_left_mpo_local(const DenseTensor& rho,
                                        const Eigen::MatrixXcd& h_mat,
                                        const DenseTensor& rho_adj,
                                        const DenseTensor& er)
{
    DenseTensor h_t = mat_to_tensor(h_mat);
    // er: (D_ρ_r, D_ρ†_r) = (α_R, β_R)
    // Absorb er into rho from the right
    DenseTensor T1 = contract(rho, er, {{3, 0}});           // (D_l, d_bra, d_ket, β_R)
    DenseTensor T2 = contract(T1, h_t, {{1, 1}});           // (D_l, d_ket, β_R, d_out)
    T2 = T2.permute({0, 3, 1, 2});                          // (D_l, d_out, d_ket, β_R)
    // Contract with rho_adj: (D_ρ_r, d_ket, d_bra, D_ρ_l) = (β_R_adj, d_ket, d_out, β_L)
    // Wait — rho_adj.leg0 = D_r of original, but here we want D_l.
    // For push_left: we're coming from the right, so:
    //   new_er[α_L, β_L] = Σ rho[α_L, σ', σ, α_R] · h[σ'', σ'] · conj(ρ[β_L, σ'', σ, β_R]) · er[α_R, β_R]
    // With rho_adj = ρ†: (D_r, d_ket, d_bra, D_l) where rho_adj[β_R, σ, σ'', β_L] = conj(ρ[β_L, σ'', σ, β_R])
    // Need to contract T2 with rho_adj over (β_R, d_ket, d_out):
    // T2 shape: (D_l, d_out, d_ket, β_R) = legs (0,1,2,3)
    // rho_adj: (D_r=β_R, d_ket, d_bra=d_out, D_l=β_L)
    DenseTensor T3 = contract(T2, rho_adj, {{3, 0}, {2, 1}, {1, 2}}); // (D_l, β_L) = (α_L, β_L)
    return T3;  // (α_L, β_L) = (D_ρ_l, D_ρ†_l)
}

static DenseTensor push_left_mpo_identity(const DenseTensor& rho,
                                           const DenseTensor& rho_adj,
                                           const DenseTensor& er)
{
    DenseTensor T1 = contract(rho, er, {{3, 0}});           // (D_l, d_bra, d_ket, β_R)
    // Contract with rho_adj over (β_R, d_ket, d_bra)
    DenseTensor T3 = contract(T1, rho_adj, {{3, 0}, {2, 1}, {1, 2}}); // (D_l, β_L)
    return T3;  // (α_L, β_L)
}

} // anonymous namespace

// ── MPOEnvironment<DenseBackend> ──────────────────────────────────────────────

template<>
MPOEnvironment<DenseBackend>::MPOEnvironment(DenseMPO<DenseBackend>& rho,
                                              SparseMPO<DenseBackend>& H)
    : rho_(&rho), H_(&H), L_(rho.length())
{
    assert(rho.length() == H.length());

    int D_left  = H[0].d_in();
    int D_right = H[L_ - 1].d_out();

    left_envs_.reserve(L_ + 1);
    right_envs_.reserve(L_ + 1);
    for (int i = 0; i <= L_; ++i) {
        left_envs_.emplace_back(D_left);
        right_envs_.emplace_back(D_right);
    }

    left_envs_[0]   = SparseLeftEnvTensor<DenseBackend>::boundary(D_left);
    right_envs_[L_] = SparseRightEnvTensor<DenseBackend>::boundary(D_right);
}

template<>
SparseLeftEnvTensor<DenseBackend>& MPOEnvironment<DenseBackend>::left_env(int bond)
{
    assert(bond >= 0 && bond <= L_);
    return left_envs_[bond];
}

template<>
const SparseLeftEnvTensor<DenseBackend>& MPOEnvironment<DenseBackend>::left_env(int bond) const
{
    assert(bond >= 0 && bond <= L_);
    return left_envs_[bond];
}

template<>
SparseRightEnvTensor<DenseBackend>& MPOEnvironment<DenseBackend>::right_env(int bond)
{
    assert(bond >= 0 && bond <= L_);
    return right_envs_[bond];
}

template<>
const SparseRightEnvTensor<DenseBackend>& MPOEnvironment<DenseBackend>::right_env(int bond) const
{
    assert(bond >= 0 && bond <= L_);
    return right_envs_[bond];
}

template<>
void MPOEnvironment<DenseBackend>::push_right(int site)
{
    assert(site >= 0 && site < L_);
    const auto& Hs = (*H_)[site];
    int D_in  = Hs.d_in();
    int D_out = Hs.d_out();
    const SparseLeftEnvTensor<DenseBackend>& L_env = left_envs_[site];
    assert(L_env.dim() == D_in);

    const DenseTensor& rho_site = (*rho_)[site].data();
    DenseTensor rho_adj = (*rho_)[site].adjoint().data();

    using Task = std::tuple<int, int, const AbstractLocalOperator<DenseBackend>*>;
    std::vector<Task> tasks;
    for (int i = 0; i < D_in; ++i) {
        if (!L_env.has(i)) continue;
        for (int j = 0; j < D_out; ++j) {
            if (!Hs.has(i, j)) continue;
            tasks.emplace_back(i, j, Hs(i, j));
        }
    }

    std::vector<std::optional<DenseTensor>> accum(D_out);

    #pragma omp parallel for schedule(dynamic) if(tasks.size() > 4)
    for (int t = 0; t < static_cast<int>(tasks.size()); ++t) {
        auto [i, j, op] = tasks[t];
        const DenseTensor& el = L_env[i]->data();
        DenseTensor contrib = op->is_identity()
            ? push_right_mpo_identity(el, rho_site, rho_adj)
            : push_right_mpo_local(el, rho_site, op->matrix(), rho_adj);

        #pragma omp critical(mpo_env_push_accum)
        {
            if (!accum[j].has_value())
                accum[j] = std::move(contrib);
            else
                accum[j]->axpby({1, 0}, {1, 0}, contrib);
        }
    }

    int D_new_out = (*H_)[site].d_out();
    left_envs_[site + 1] = SparseLeftEnvTensor<DenseBackend>(D_new_out);
    for (int j = 0; j < D_out; ++j) {
        if (accum[j].has_value())
            left_envs_[site + 1].set(
                j, std::make_unique<LeftEnvTensor<DenseBackend>>(std::move(*accum[j])));
    }
}

template<>
void MPOEnvironment<DenseBackend>::push_left(int site)
{
    assert(site >= 0 && site < L_);
    const auto& Hs = (*H_)[site];
    int D_in  = Hs.d_in();
    int D_out = Hs.d_out();
    const SparseRightEnvTensor<DenseBackend>& R_env = right_envs_[site + 1];
    assert(R_env.dim() == D_out);

    const DenseTensor& rho_site = (*rho_)[site].data();
    DenseTensor rho_adj = (*rho_)[site].adjoint().data();

    using Task = std::tuple<int, int, const AbstractLocalOperator<DenseBackend>*>;
    std::vector<Task> tasks;
    for (int i = 0; i < D_in; ++i) {
        for (int j = 0; j < D_out; ++j) {
            if (!Hs.has(i, j)) continue;
            if (!R_env.has(j)) continue;
            tasks.emplace_back(i, j, Hs(i, j));
        }
    }

    std::vector<std::optional<DenseTensor>> accum(D_in);

    #pragma omp parallel for schedule(dynamic) if(tasks.size() > 4)
    for (int t = 0; t < static_cast<int>(tasks.size()); ++t) {
        auto [i, j, op] = tasks[t];
        const DenseTensor& er = R_env[j]->data();
        DenseTensor contrib = op->is_identity()
            ? push_left_mpo_identity(rho_site, rho_adj, er)
            : push_left_mpo_local(rho_site, op->matrix(), rho_adj, er);

        #pragma omp critical(mpo_env_push_accum)
        {
            if (!accum[i].has_value())
                accum[i] = std::move(contrib);
            else
                accum[i]->axpby({1, 0}, {1, 0}, contrib);
        }
    }

    right_envs_[site] = SparseRightEnvTensor<DenseBackend>(D_in);
    for (int i = 0; i < D_in; ++i) {
        if (accum[i].has_value())
            right_envs_[site].set(
                i, std::make_unique<RightEnvTensor<DenseBackend>>(std::move(*accum[i])));
    }
}

template<>
void MPOEnvironment<DenseBackend>::build_right_envs()
{
    int D_left  = (*H_)[0].d_in();
    int D_right = (*H_)[L_ - 1].d_out();

    left_envs_[0]   = SparseLeftEnvTensor<DenseBackend>::boundary(D_left);
    right_envs_[L_] = SparseRightEnvTensor<DenseBackend>::boundary(D_right);

    for (int site = L_ - 1; site >= 0; --site)
        push_left(site);
}

} // namespace tenet
