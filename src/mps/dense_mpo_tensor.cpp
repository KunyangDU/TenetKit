// src/mps/dense_mpo_tensor.cpp
//
// DenseMPOTensor<DenseBackend>: rank-4 MPO site tensor (D_l, d_bra, d_ket, D_r).

#include "tenet/mps/dense_mpo_tensor.hpp"
#include "tenet/core/space.hpp"

#include <cassert>

namespace tenet {

template<>
DenseMPOTensor<DenseBackend>::DenseMPOTensor(int D_l, int d, int D_r)
{
    data_ = DenseTensor({TrivialSpace(D_l),
                         TrivialSpace(d),
                         TrivialSpace(d),
                         TrivialSpace(D_r)});
}

template<>
DenseMPOTensor<DenseBackend> DenseMPOTensor<DenseBackend>::adjoint() const
{
    // Permute (D_l, d_bra, d_ket, D_r) → (D_r, d_ket, d_bra, D_l)
    // then conjugate data.
    DenseMPOTensor<DenseBackend> result;
    result.data_ = data_.permute({3, 2, 1, 0}).conj();
    return result;
}

template<>
Eigen::MatrixXcd DenseMPOTensor<DenseBackend>::as_matrix_left() const
{
    // (D_l * d_bra * d_ket) × D_r
    return data_.matricize({0, 1, 2}, {3});
}

template<>
Eigen::MatrixXcd DenseMPOTensor<DenseBackend>::as_matrix_right() const
{
    // D_l × (d_bra * d_ket * D_r)
    return data_.matricize({0}, {1, 2, 3});
}

template<>
Eigen::MatrixXcd DenseMPOTensor<DenseBackend>::left_canonicalize()
{
    Eigen::MatrixXcd M = as_matrix_left();  // (D_l*d*d) × D_r
    int m = static_cast<int>(M.rows());
    int n = static_cast<int>(M.cols());
    int k = std::min(m, n);

    Eigen::HouseholderQR<Eigen::MatrixXcd> decomp(M);
    Eigen::MatrixXcd Q = decomp.householderQ() * Eigen::MatrixXcd::Identity(m, k);
    Eigen::MatrixXcd R = decomp.matrixQR().topRows(k)
                            .template triangularView<Eigen::Upper>();

    int Dl = data_.dim(0), d = data_.dim(1);
    data_ = DenseTensor::from_matrix(Q, {TrivialSpace(Dl),
                                          TrivialSpace(d),
                                          TrivialSpace(d),
                                          TrivialSpace(k)}, true);
    return R;  // k × D_r
}

template<>
Eigen::MatrixXcd DenseMPOTensor<DenseBackend>::right_canonicalize()
{
    Eigen::MatrixXcd M = as_matrix_right();  // D_l × (d*d*D_r)
    int m = static_cast<int>(M.rows());
    int n = static_cast<int>(M.cols());
    int k = std::min(m, n);

    // RQ via QR of M†
    Eigen::HouseholderQR<Eigen::MatrixXcd> decomp(M.adjoint());
    Eigen::MatrixXcd Q_t = decomp.householderQ() * Eigen::MatrixXcd::Identity(n, k);
    Eigen::MatrixXcd R_t = decomp.matrixQR().topRows(k)
                              .template triangularView<Eigen::Upper>();

    Eigen::MatrixXcd L = R_t.adjoint();  // m × k
    Eigen::MatrixXcd Q = Q_t.adjoint();  // k × n

    int d = data_.dim(1), Dr = data_.dim(3);
    data_ = DenseTensor::from_matrix(Q, {TrivialSpace(k),
                                          TrivialSpace(d),
                                          TrivialSpace(d),
                                          TrivialSpace(Dr)}, true);
    return L;  // D_l × k
}

template<>
DenseMPOTensor<DenseBackend>
DenseMPOTensor<DenseBackend>::from_left_matrix(const Eigen::MatrixXcd& mat,
                                                int D_l, int d, int D_r)
{
    DenseMPOTensor<DenseBackend> t;
    t.data_ = DenseTensor::from_matrix(mat,
        {TrivialSpace(D_l), TrivialSpace(d), TrivialSpace(d), TrivialSpace(D_r)},
        true);
    return t;
}

template<>
DenseMPOTensor<DenseBackend>
DenseMPOTensor<DenseBackend>::from_right_matrix(const Eigen::MatrixXcd& mat,
                                                 int D_l, int d, int D_r)
{
    DenseMPOTensor<DenseBackend> t;
    t.data_ = DenseTensor::from_matrix(mat,
        {TrivialSpace(D_l), TrivialSpace(d), TrivialSpace(d), TrivialSpace(D_r)},
        true);
    return t;
}

} // namespace tenet
