#pragma once
// include/tenet/mps/dense_mpo_tensor.hpp
//
// DenseMPOTensor<B>: rank-4 MPO site tensor (D_l, d_bra, d_ket, D_r).
// Index convention (C++): (left_bond, bra_phys, ket_phys, right_bond)
// Julia convention:        (ket_phys,  left_bond, right_bond, bra_phys)
//
// adjoint(): permute {3,2,1,0} + conjugate → shape (D_r, d_ket, d_bra, D_l)

#include "tenet/core/backend.hpp"
#include "tenet/core/dense_tensor.hpp"
#include "tenet/core/space.hpp"

#include <Eigen/Dense>

namespace tenet {

template<TensorBackend B = DenseBackend>
class DenseMPOTensor {
public:
    using Tensor = typename B::Tensor;

    DenseMPOTensor() = default;

    // Construct zero tensor of shape (D_l, d, d, D_r).
    DenseMPOTensor(int D_l, int d, int D_r);

    // Shape
    int D_l() const noexcept { return data_.rank() > 0 ? data_.dim(0) : 0; }
    int d()   const noexcept { return data_.rank() > 0 ? data_.dim(1) : 0; }
    int D_r() const noexcept { return data_.rank() > 0 ? data_.dim(3) : 0; }

    // Raw tensor access
    Tensor&       data()       { return data_; }
    const Tensor& data() const { return data_; }

    // Conjugate-transpose: permute {3,2,1,0} + conjugate data.
    // Returns shape (D_r, d_ket, d_bra, D_l).
    DenseMPOTensor adjoint() const;

    // Matrix form for QR/SVD: fuse (D_l, d_bra, d_ket) × D_r → (D_l*d*d) × D_r
    Eigen::MatrixXcd as_matrix_left()  const;   // (D_l*d*d) × D_r
    Eigen::MatrixXcd as_matrix_right() const;   // D_l × (d*d*D_r)

    // In-place canonicalization; returns the boundary matrix.
    // left_canonicalize:  QR of as_matrix_left(); site becomes Q; returns R.
    // right_canonicalize: RQ of as_matrix_right(); site becomes Q; returns L.
    Eigen::MatrixXcd left_canonicalize();
    Eigen::MatrixXcd right_canonicalize();

    // Construct from matrix
    static DenseMPOTensor from_left_matrix(const Eigen::MatrixXcd& mat,
                                            int D_l, int d, int D_r);
    static DenseMPOTensor from_right_matrix(const Eigen::MatrixXcd& mat,
                                             int D_l, int d, int D_r);

private:
    Tensor data_;   // shape (D_l, d_bra, d_ket, D_r)
};

} // namespace tenet
