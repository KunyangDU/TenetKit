#pragma once
// include/tenet/mps/mps_tensor.hpp
//
// MPSTensor<B>: a rank-3 tensor (D_left × d_phys × D_right).
// See docs/C++重构设计方案.md §7.1.

#include "tenet/core/backend.hpp"
#include "tenet/core/factorization.hpp"

#include <Eigen/Dense>

namespace tenet {

template<TensorBackend B = DenseBackend>
class MPSTensor {
public:
    using Tensor = typename B::Tensor;
    using Space  = typename B::Space;

    MPSTensor() = default;
    MPSTensor(Space vl, Space phys, Space vr);

    // Shape
    int left_dim()  const;
    int phys_dim()  const;
    int right_dim() const;

    const Space& left_space()  const;
    const Space& phys_space()  const;
    const Space& right_space() const;

    // Data
    Tensor&       data()       { return data_; }
    const Tensor& data() const { return data_; }

    // Matrix views for QR/SVD
    Eigen::MatrixXcd as_matrix_left()  const;   // (D_l * d) × D_r
    Eigen::MatrixXcd as_matrix_right() const;   // D_l × (d * D_r)

    // In-place canonicalisation; returns the boundary matrix absorbed right/left.
    Eigen::MatrixXcd left_canonicalize();
    Eigen::MatrixXcd right_canonicalize();

    static MPSTensor from_left_matrix (const Eigen::MatrixXcd& mat, Space vl, Space phys, Space vr);
    static MPSTensor from_right_matrix(const Eigen::MatrixXcd& mat, Space vl, Space phys, Space vr);

    MPSTensor adjoint() const;

private:
    Tensor data_;   // rank-3
};

} // namespace tenet
