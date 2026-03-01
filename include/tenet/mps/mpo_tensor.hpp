#pragma once
// include/tenet/mps/mpo_tensor.hpp
//
// SparseMPOTensor<B>: a (D_in × D_out) matrix of LocalOperators.
// Elements are owned via unique_ptr; nullptr means zero contribution.
// See docs/C++重构设计方案.md §7.3.

#include "tenet/core/backend.hpp"
#include "tenet/intr_tree/local_operator.hpp"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace tenet {

template<TensorBackend B = DenseBackend>
class SparseMPOTensor {
public:
    SparseMPOTensor(int d_in, int d_out)
        : d_in_(d_in), d_out_(d_out), ops_(d_in * d_out) {}

    // Element access
    AbstractLocalOperator<B>*       operator()(int i, int j);
    const AbstractLocalOperator<B>* operator()(int i, int j) const;

    void set(int i, int j, std::unique_ptr<AbstractLocalOperator<B>> op);
    bool has(int i, int j) const { return ops_[i * d_out_ + j] != nullptr; }

    int d_in()  const noexcept { return d_in_; }
    int d_out() const noexcept { return d_out_; }

    // Iterate non-zero entries: fn(row, col, op)
    void for_each_nonzero(
        std::function<void(int, int, const AbstractLocalOperator<B>&)> fn) const;

private:
    int d_in_, d_out_;
    std::vector<std::unique_ptr<AbstractLocalOperator<B>>> ops_;
};

} // namespace tenet
