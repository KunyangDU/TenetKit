// src/environment/environment.cpp
//
// LeftEnvTensor / RightEnvTensor boundary factories.
// Environment<DenseBackend>: constructor, build_all, push_right, push_left.

#include "tenet/environment/environment.hpp"
#include "tenet/environment/env_push.hpp"
#include "tenet/core/dense_tensor.hpp"

#include <cassert>
#include <memory>

namespace tenet {

// ── Boundary factories ────────────────────────────────────────────────────────

template<>
LeftEnvTensor<DenseBackend> LeftEnvTensor<DenseBackend>::boundary() {
    DenseTensor t({TrivialSpace(1), TrivialSpace(1)},
                  {DenseTensor::Scalar{1.0, 0.0}});
    return LeftEnvTensor<DenseBackend>(std::move(t));
}

template<>
RightEnvTensor<DenseBackend> RightEnvTensor<DenseBackend>::boundary() {
    DenseTensor t({TrivialSpace(1), TrivialSpace(1)},
                  {DenseTensor::Scalar{1.0, 0.0}});
    return RightEnvTensor<DenseBackend>(std::move(t));
}

template<>
SparseLeftEnvTensor<DenseBackend>
SparseLeftEnvTensor<DenseBackend>::boundary(int D) {
    SparseLeftEnvTensor<DenseBackend> env(D);
    env.set(0, std::make_unique<LeftEnvTensor<DenseBackend>>(
                    LeftEnvTensor<DenseBackend>::boundary()));
    return env;
}

template<>
SparseRightEnvTensor<DenseBackend>
SparseRightEnvTensor<DenseBackend>::boundary(int D) {
    SparseRightEnvTensor<DenseBackend> env(D);
    env.set(D - 1, std::make_unique<RightEnvTensor<DenseBackend>>(
                        RightEnvTensor<DenseBackend>::boundary()));
    return env;
}

// ── Environment<DenseBackend> ─────────────────────────────────────────────────

template<>
Environment<DenseBackend>::Environment(DenseMPS<DenseBackend>& psi,
                                        SparseMPO<DenseBackend>& H)
    : psi_(&psi), H_(&H), L_(psi.length())
{
    assert(psi.length() == H.length());

    int D_left  = H[0].d_in();
    int D_right = H[L_ - 1].d_out();

    // Allocate L+1 slots for left and right environments
    left_envs_.reserve(L_ + 1);
    right_envs_.reserve(L_ + 1);

    // Fill with placeholder empty envs (will be set by build_all)
    for (int i = 0; i <= L_; ++i) {
        left_envs_.emplace_back(SparseLeftEnvTensor<DenseBackend>(D_left));
        right_envs_.emplace_back(SparseRightEnvTensor<DenseBackend>(D_right));
    }

    // Set boundaries
    left_envs_[0]   = SparseLeftEnvTensor<DenseBackend>::boundary(D_left);
    right_envs_[L_] = SparseRightEnvTensor<DenseBackend>::boundary(D_right);
}

template<>
SparseLeftEnvTensor<DenseBackend>& Environment<DenseBackend>::left_env(int site) {
    assert(site >= 0 && site <= L_);
    return left_envs_[site];
}

template<>
SparseRightEnvTensor<DenseBackend>& Environment<DenseBackend>::right_env(int site) {
    assert(site >= 0 && site <= L_);
    return right_envs_[site];
}

template<>
void Environment<DenseBackend>::push_right(int site) {
    assert(site >= 0 && site < L_);
    // Resize to match output bond dim of H[site]
    int D_out = (*H_)[site].d_out();
    left_envs_[site + 1] = SparseLeftEnvTensor<DenseBackend>(D_out);
    left_envs_[site + 1] = tenet::push_right(left_envs_[site],
                                               (*psi_)[site],
                                               (*H_)[site]);
}

template<>
void Environment<DenseBackend>::push_left(int site) {
    assert(site >= 0 && site < L_);
    // Resize to match input bond dim of H[site]
    int D_in = (*H_)[site].d_in();
    right_envs_[site] = SparseRightEnvTensor<DenseBackend>(D_in);
    right_envs_[site] = tenet::push_left(right_envs_[site + 1],
                                           (*psi_)[site],
                                           (*H_)[site]);
}

// build_all: psi must be in canonical form. Build all right environments.
template<>
void Environment<DenseBackend>::build_all() {
    int D_left  = H_->operator[](0).d_in();
    int D_right = H_->operator[](L_ - 1).d_out();

    // Reset boundaries
    left_envs_[0]   = SparseLeftEnvTensor<DenseBackend>::boundary(D_left);
    right_envs_[L_] = SparseRightEnvTensor<DenseBackend>::boundary(D_right);

    // Build all right envs: from rightmost site to leftmost
    for (int site = L_ - 1; site >= 0; --site) {
        push_left(site);
    }
}

} // namespace tenet
