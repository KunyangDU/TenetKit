#pragma once
// include/tenet/environment/two_layer_mpo_env.hpp
//
// TwoLayerMPOEnv<B>: two-layer [x, y†] DenseMPO environment for mpo_axpy.
//
// Left env  el[s]: shape (D_y_l, D_x_l) — "overlap" from the left up to bond s
// Right env er[s]: shape (D_x_r, D_y_r) — "overlap" from the right from bond s
//
// push_right (el update):
//   new_el[β_R, α_R] = Σ el[β_L, α_L] · x[α_L,σ',σ,α_R] · conj(y[β_L,σ',σ,β_R])
//
// push_left (er update):
//   new_er[α_L, β_L] = Σ x[α_L,σ',σ,α_R] · conj(y[β_L,σ',σ,β_R]) · er[α_R, β_R]

#include "tenet/mps/dense_mpo.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <vector>

namespace tenet {

template<TensorBackend B = DenseBackend>
class TwoLayerMPOEnv {
public:
    // x and y are non-owning references.
    TwoLayerMPOEnv(const DenseMPO<B>& x, const DenseMPO<B>& y);

    // Build all right environments from the right boundary inward.
    void build_right_envs();

    // Sweep propagation.
    void push_right(int site);  // left_envs_[site+1] ← updated
    void push_left(int site);   // right_envs_[site]  ← updated

    // Environment access.
    const DenseTensor& left_env(int bond)  const;  // shape (D_y, D_x)
    const DenseTensor& right_env(int bond) const;  // shape (D_x, D_y)

    const DenseMPO<B>& x() const { return *x_; }
    const DenseMPO<B>& y() const { return *y_; }
    int length() const noexcept { return L_; }

private:
    const DenseMPO<B>* x_;
    const DenseMPO<B>* y_;
    int L_;

    // left_envs_[s]:  (D_y_l, D_x_l) for bond to left of site s  (size L+1)
    // right_envs_[s]: (D_x_r, D_y_r) for bond to right of site s (size L+1)
    std::vector<DenseTensor> left_envs_;
    std::vector<DenseTensor> right_envs_;
};

} // namespace tenet
