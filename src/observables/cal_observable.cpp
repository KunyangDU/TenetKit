// src/observables/cal_observable.cpp
//
// cal_obs: DFS traversal of an ObservableTree to compute all leaf expectation
// values ⟨ψ|O_1 … O_k|ψ⟩.
//
// Algorithm overview (§3 of TODO.md):
//
//   1. Precompute plain "observable" right environments R_obs[0..L]:
//        R_obs[L]   = [[1]]  (right boundary, 1×1)
//        R_obs[s]   = push_left_identity(psi[s], R_obs[s+1])
//      Shape: (D_bond_s, D_bond_s) at bond (s-1, s).
//
//   2. DFS traversal of the tree.  Each node carries:
//        - an operator op at site op->site()
//        - a parent frame supplying the accumulated left env
//      For a node at site `s` with parent site `p`:
//        a. Apply identity at sites p+1, …, s-1  (gap filling)
//        b. Apply op at site s
//        c. Store the new left env in node->env_cache
//        d. If leaf: value = Tr(L_s · R_obs[s+1])
//        e. Push children onto the stack with the new left env
//
// The left env tensor has shape (D_bra, D_ket) at the right bond of the last
// contracted site.  The right obs env has shape (D_ket, D_bra) at the left
// bond of the first NOT-yet-contracted site.  The scalar is Tr(L · R).

#include "tenet/observables/cal_observable.hpp"
#include "tenet/observables/obs_node.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/dense_tensor.hpp"
#include "tenet/environment/env_tensor.hpp"

#include <cassert>
#include <complex>
#include <stack>
#include <vector>

namespace tenet {

// ── Internal env push helpers ─────────────────────────────────────────────────
//
// These implement the same contractions as the static helpers in env_push.cpp
// but without the sparse MPO bond dimension (plain D_bra × D_ket envs).

// Extend a left env past site `s` with operator h_mat.
//   new_L[β',β] = Σ_{α',α,σ',σ} L[α',α] · ψ[α,σ,β] · h[σ',σ] · ψ*[α',σ',β']
// In: el(D_bra, D_ket), ket(D_l, d, D_r), h_mat(d_out, d_in), bra(D_r, D_l, d)
// Out: (D_r_bra, D_r_ket)
static DenseTensor obs_push_right_local(const DenseTensor&      el,
                                         const DenseTensor&      ket,
                                         const Eigen::MatrixXcd& h_mat,
                                         const DenseTensor&      bra)
{
    int d_out = static_cast<int>(h_mat.rows());
    int d_in  = static_cast<int>(h_mat.cols());
    DenseTensor h_t   = DenseTensor::from_matrix(
                            h_mat, {TrivialSpace(d_out), TrivialSpace(d_in)}, true);
    DenseTensor T1    = contract(el, ket, {{1, 0}});           // (D_bra, d_in, D_r)
    DenseTensor T2    = contract(T1, h_t, {{1, 1}});           // (D_bra, D_r, d_out)
    T2                = T2.permute({0, 2, 1});                 // (D_bra, d_out, D_r)
    DenseTensor T_res = contract(T2, bra, {{0, 1}, {1, 2}});  // (D_r_ket, D_r_bra)
    return T_res.permute({1, 0});                              // (D_r_bra, D_r_ket)
}

// Extend a left env past site `s` with the identity operator.
//   new_L[β',β] = Σ_{α',α,σ} L[α',α] · ψ[α,σ,β] · ψ*[α',σ,β']
// In: el(D_bra, D_ket), ket(D_l, d, D_r), bra(D_r, D_l, d)
// Out: (D_r_bra, D_r_ket)
static DenseTensor obs_push_right_identity(const DenseTensor& el,
                                            const DenseTensor& ket,
                                            const DenseTensor& bra)
{
    DenseTensor T1    = contract(el, ket, {{1, 0}});           // (D_bra, d, D_r)
    DenseTensor T_res = contract(T1, bra, {{0, 1}, {1, 2}});  // (D_r_ket, D_r_bra)
    return T_res.permute({1, 0});                              // (D_r_bra, D_r_ket)
}

// Extend a right obs env from the right (push left by one site, no operator).
//   new_R[α,α'] = Σ_{β,β',σ} ψ[α,σ,β] · R[β,β'] · ψ*[α',σ,β']
// In: ket(D_l, d, D_r), bra(D_r, D_l, d), er(D_ket, D_bra)
// Out: (D_l_ket, D_l_bra)
static DenseTensor obs_push_left_identity(const DenseTensor& ket,
                                           const DenseTensor& bra,
                                           const DenseTensor& er)
{
    DenseTensor T1 = contract(ket, er, {{2, 0}});              // (D_l, d, D_bra)
    return contract(T1, bra, {{2, 0}, {1, 2}});                // (D_l_ket, D_l_bra)
}

// ── cal_obs ───────────────────────────────────────────────────────────────────

template<>
void cal_obs(ObservableTree<DenseBackend>& tree, const Environment<DenseBackend>& env)
{
    const DenseMPS<DenseBackend>& psi = env.psi();
    const int L = psi.length();
    assert(L > 0);

    // ── Step 1: Precompute plain right obs envs R_obs[0..L] ──────────────────
    // R_obs[s] has shape (D_bond_s_ket, D_bond_s_bra) at bond (s-1, s).
    // R_obs[L] = [[1]] (1×1 right boundary).
    std::vector<DenseTensor> R_obs(L + 1);
    R_obs[L] = DenseTensor({TrivialSpace(1), TrivialSpace(1)},
                            {DenseTensor::Scalar{1.0, 0.0}});
    for (int s = L - 1; s >= 0; --s) {
        const DenseTensor& ket = psi[s].data();
        DenseTensor        bra = psi[s].adjoint().data();
        R_obs[s] = obs_push_left_identity(ket, bra, R_obs[s + 1]);
    }

    // ── Step 2: DFS traversal ─────────────────────────────────────────────────
    struct Frame {
        ObservableTreeNode<DenseBackend>* node;
        int         parent_site;  // -1 = before site 0 (root's children)
        DenseTensor left_env;     // accumulated left env up to parent's right bond
    };

    // Left boundary: 1×1 [[1]]
    const DenseTensor root_env({TrivialSpace(1), TrivialSpace(1)},
                                {DenseTensor::Scalar{1.0, 0.0}});

    std::stack<Frame> stk;
    for (auto& child : tree.root->children)
        stk.push({child.get(), -1, root_env});

    while (!stk.empty()) {
        auto [node, parent_site, init_env] = stk.top();
        stk.pop();

        assert(node->op != nullptr);
        const int this_site = node->op->site();
        assert(this_site > parent_site && this_site < L);

        DenseTensor cur_env = std::move(init_env);

        // ── Apply identity at gap sites (parent_site+1, …, this_site-1) ──────
        for (int s = parent_site + 1; s < this_site; ++s) {
            const DenseTensor& ket = psi[s].data();
            DenseTensor        bra = psi[s].adjoint().data();
            cur_env = obs_push_right_identity(cur_env, ket, bra);
        }

        // ── Apply this node's operator at this_site ───────────────────────────
        {
            const DenseTensor& ket = psi[this_site].data();
            DenseTensor        bra = psi[this_site].adjoint().data();
            cur_env = obs_push_right_local(cur_env, ket, node->op->matrix(), bra);
        }

        // Store left env in node's env_cache
        node->env_cache = LeftEnvTensor<DenseBackend>(cur_env);

        // ── Leaf: compute ⟨O_path⟩ = Tr(L · R_obs[this_site+1]) ─────────────
        if (node->leaf) {
            // L has shape (D_bra_right, D_ket_right) at bond (this_site, this_site+1)
            // R has shape (D_ket_left,  D_bra_left)  at bond (this_site, this_site+1)
            // scalar = Σ_{bra,ket} L[bra,ket] * R[ket,bra] = Tr(L_mat * R_mat)
            const DenseTensor& R    = R_obs[this_site + 1];
            Eigen::MatrixXcd   Lmat = cur_env.matricize({0}, {1});  // (D_bra, D_ket)
            Eigen::MatrixXcd   Rmat = R.matricize({0}, {1});         // (D_ket, D_bra)
            node->leaf->value = (Lmat * Rmat).trace();
        }

        // Push children with the current accumulated left env
        for (auto& child : node->children)
            stk.push({child.get(), this_site, cur_env});
    }
}

} // namespace tenet
