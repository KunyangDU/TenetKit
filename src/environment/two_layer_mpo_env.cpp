// src/environment/two_layer_mpo_env.cpp
//
// TwoLayerMPOEnv<DenseBackend>: [x, y†] environment for mpo_axpy.
//
// push_right:
//   el: (D_y_l, D_x_l)
//   x[s]: (D_x_l, d_bra, d_ket, D_x_r)
//   y†[s]: adjoint = (D_y_r, d_ket, d_bra, D_y_l)   [conj+permute {3,2,1,0}]
//
//   T1 = contract(el, x_site, {{1,0}})             → (D_y_l, d_bra, d_ket, D_x_r)
//   T3 = contract(T1, y_adj, {{0,3},{1,2},{2,1}})  → (D_x_r, D_y_r)
//   new_el = T3.permute({1,0})? No: shape is (D_x_r, D_y_r)
//   Actually result = T3.permute({1,0}) → (D_y_r, D_x_r)
//
// push_left:
//   er: (D_x_r, D_y_r)
//   x[s]: (D_x_l, d_bra, d_ket, D_x_r)
//   y†[s]: (D_y_r, d_ket, d_bra, D_y_l)
//
//   T1 = contract(x_site, er, {{3,0}})             → (D_x_l, d_bra, d_ket, D_y_r)
//   T3 = contract(T1, y_adj, {{2,1},{1,2},{3,0}})  → (D_x_l, D_y_l)
//   new_er = T3  (shape D_x_l, D_y_l)

#include "tenet/environment/two_layer_mpo_env.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/space.hpp"

#include <cassert>

namespace tenet {

template<>
TwoLayerMPOEnv<DenseBackend>::TwoLayerMPOEnv(const DenseMPO<DenseBackend>& x,
                                               const DenseMPO<DenseBackend>& y)
    : x_(&x), y_(&y), L_(x.length())
{
    assert(x.length() == y.length());

    // Left boundary: el[0] = 1×1 identity = [[1]]
    // Right boundary: er[L] = 1×1 identity = [[1]]
    // Initialize all with zero placeholders; boundaries are set below.
    for (int i = 0; i <= L_; ++i) {
        left_envs_.emplace_back();
        right_envs_.emplace_back();
    }

    // Left boundary: shape (D_y_l=1, D_x_l=1) with value 1
    left_envs_[0] = DenseTensor({TrivialSpace(1), TrivialSpace(1)},
                                 {DenseTensor::Scalar{1.0, 0.0}});
    // Right boundary: shape (D_x_r=1, D_y_r=1) with value 1
    right_envs_[L_] = DenseTensor({TrivialSpace(1), TrivialSpace(1)},
                                    {DenseTensor::Scalar{1.0, 0.0}});
}

template<>
void TwoLayerMPOEnv<DenseBackend>::push_right(int site)
{
    assert(site >= 0 && site < L_);
    const DenseTensor& el       = left_envs_[site];      // (D_y_l, D_x_l)
    const DenseTensor& x_site   = (*x_)[site].data();    // (D_x_l, d, d, D_x_r)
    DenseTensor        y_adj    = (*y_)[site].adjoint().data(); // (D_y_r, d_ket, d_bra, D_y_l)

    // T1 = contract(el, x_site, {{1,0}}) → (D_y_l, d_bra, d_ket, D_x_r)
    DenseTensor T1 = contract(el, x_site, {{1, 0}});
    // Contract T1 with y_adj: sum over (D_y_l, d_bra, d_ket)
    // y_adj legs: (D_y_r=0, d_ket=1, d_bra=2, D_y_l=3)
    // T1 legs:    (D_y_l=0, d_bra=1, d_ket=2, D_x_r=3)
    // Contract: T1.0 ↔ y_adj.3 (D_y_l), T1.1 ↔ y_adj.2 (d_bra), T1.2 ↔ y_adj.1 (d_ket)
    DenseTensor T3 = contract(T1, y_adj, {{0, 3}, {1, 2}, {2, 1}}); // (D_x_r, D_y_r)
    left_envs_[site + 1] = T3.permute({1, 0});  // (D_y_r, D_x_r)
}

template<>
void TwoLayerMPOEnv<DenseBackend>::push_left(int site)
{
    assert(site >= 0 && site < L_);
    const DenseTensor& er       = right_envs_[site + 1]; // (D_x_r, D_y_r)
    const DenseTensor& x_site   = (*x_)[site].data();    // (D_x_l, d, d, D_x_r)
    DenseTensor        y_adj    = (*y_)[site].adjoint().data(); // (D_y_r, d_ket, d_bra, D_y_l)

    // T1 = contract(x_site, er, {{3,0}}) → (D_x_l, d_bra, d_ket, D_y_r)
    DenseTensor T1 = contract(x_site, er, {{3, 0}});
    // Contract T1 with y_adj: sum over (D_y_r, d_ket, d_bra)
    // T1 legs:    (D_x_l=0, d_bra=1, d_ket=2, D_y_r=3)
    // y_adj legs: (D_y_r=0, d_ket=1, d_bra=2, D_y_l=3)
    // Contract: T1.3 ↔ y_adj.0 (D_y_r), T1.2 ↔ y_adj.1 (d_ket), T1.1 ↔ y_adj.2 (d_bra)
    DenseTensor T3 = contract(T1, y_adj, {{3, 0}, {2, 1}, {1, 2}}); // (D_x_l, D_y_l)
    right_envs_[site] = T3;  // (D_x_l, D_y_l)
}

template<>
void TwoLayerMPOEnv<DenseBackend>::build_right_envs()
{
    // Reset boundaries
    right_envs_[L_] = DenseTensor({TrivialSpace(1), TrivialSpace(1)},
                                    {DenseTensor::Scalar{1.0, 0.0}});
    for (int site = L_ - 1; site >= 0; --site)
        push_left(site);
}

template<>
const DenseTensor& TwoLayerMPOEnv<DenseBackend>::left_env(int bond) const
{
    assert(bond >= 0 && bond <= L_);
    return left_envs_[bond];
}

template<>
const DenseTensor& TwoLayerMPOEnv<DenseBackend>::right_env(int bond) const
{
    assert(bond >= 0 && bond <= L_);
    return right_envs_[bond];
}

} // namespace tenet
