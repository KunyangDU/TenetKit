#pragma once
// include/tenet/environment/env_tensor.hpp
//
// Left/Right environment tensors (dense and sparse variants).
// See docs/C++重构设计方案.md §8.1.

#include "tenet/core/backend.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace tenet {

// ── Dense left environment: ⟨ψ|H|ψ⟩ partially contracted from the left ───────
// Shape: (D_bra × D_mpo × D_ket), matricised as (D_bra*D_mpo) × D_ket.
template<TensorBackend B = DenseBackend>
class LeftEnvTensor {
public:
    using Tensor = typename B::Tensor;

    explicit LeftEnvTensor(Tensor data) : data_(std::move(data)) {}

    Tensor&       data()       { return data_; }
    const Tensor& data() const { return data_; }

    Eigen::MatrixXcd as_matrix() const;

    // Boundary value: (D_mpo=1, D_bra=1, D_ket=1), entry = 1.
    static LeftEnvTensor boundary();

private:
    Tensor data_;
};

template<TensorBackend B = DenseBackend>
class RightEnvTensor {
public:
    using Tensor = typename B::Tensor;

    explicit RightEnvTensor(Tensor data) : data_(std::move(data)) {}

    Tensor&       data()       { return data_; }
    const Tensor& data() const { return data_; }

    Eigen::MatrixXcd as_matrix() const;

    static RightEnvTensor boundary();

private:
    Tensor data_;
};

// ── Sparse left environment: one LeftEnvTensor per MPO bond index ─────────────
template<TensorBackend B = DenseBackend>
class SparseLeftEnvTensor {
public:
    explicit SparseLeftEnvTensor(int D) : D_(D), envs_(D) {}

    int dim() const noexcept { return D_; }

    LeftEnvTensor<B>*       operator[](int i)       { return envs_[i].get(); }
    const LeftEnvTensor<B>* operator[](int i) const { return envs_[i].get(); }

    void set(int i, std::unique_ptr<LeftEnvTensor<B>> e) { envs_[i] = std::move(e); }
    bool has(int i) const { return envs_[i] != nullptr; }

    static SparseLeftEnvTensor boundary(int D);

private:
    int D_;
    std::vector<std::unique_ptr<LeftEnvTensor<B>>> envs_;
};

template<TensorBackend B = DenseBackend>
class SparseRightEnvTensor {
public:
    explicit SparseRightEnvTensor(int D) : D_(D), envs_(D) {}

    int dim() const noexcept { return D_; }

    RightEnvTensor<B>*       operator[](int i)       { return envs_[i].get(); }
    const RightEnvTensor<B>* operator[](int i) const { return envs_[i].get(); }

    void set(int i, std::unique_ptr<RightEnvTensor<B>> e) { envs_[i] = std::move(e); }
    bool has(int i) const { return envs_[i] != nullptr; }

    static SparseRightEnvTensor boundary(int D);

private:
    int D_;
    std::vector<std::unique_ptr<RightEnvTensor<B>>> envs_;
};

} // namespace tenet
