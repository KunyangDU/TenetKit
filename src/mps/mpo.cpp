// src/mps/mpo.cpp
//
// SparseMPOTensor non-inline methods.

#include "tenet/mps/mpo_tensor.hpp"
#include "tenet/intr_tree/local_operator.hpp"

#include <cassert>
#include <stdexcept>

namespace tenet {

// ── SparseMPOTensor<DenseBackend> explicit instantiation ─────────────────────

template<>
AbstractLocalOperator<DenseBackend>*
SparseMPOTensor<DenseBackend>::operator()(int i, int j) {
    assert(i >= 0 && i < d_in_ && j >= 0 && j < d_out_);
    return ops_[i * d_out_ + j].get();
}

template<>
const AbstractLocalOperator<DenseBackend>*
SparseMPOTensor<DenseBackend>::operator()(int i, int j) const {
    assert(i >= 0 && i < d_in_ && j >= 0 && j < d_out_);
    return ops_[i * d_out_ + j].get();
}

template<>
void SparseMPOTensor<DenseBackend>::set(
    int i, int j, std::unique_ptr<AbstractLocalOperator<DenseBackend>> op)
{
    assert(i >= 0 && i < d_in_ && j >= 0 && j < d_out_);
    ops_[i * d_out_ + j] = std::move(op);
}

template<>
void SparseMPOTensor<DenseBackend>::for_each_nonzero(
    std::function<void(int, int, const AbstractLocalOperator<DenseBackend>&)> fn) const
{
    for (int i = 0; i < d_in_; ++i)
        for (int j = 0; j < d_out_; ++j)
            if (ops_[i * d_out_ + j])
                fn(i, j, *ops_[i * d_out_ + j]);
}

// ── LocalOperator<DenseBackend> ───────────────────────────────────────────────

template<>
LocalOperator<DenseBackend>::LocalOperator(
    Eigen::MatrixXcd mat, std::string name, int site,
    std::optional<std::complex<double>> strength)
    : mat_(std::move(mat))
    , name_(std::move(name))
    , site_(site)
    , strength_(strength)
{
    if (strength_.has_value()) mat_ *= strength_.value();
}

template<>
std::unique_ptr<AbstractLocalOperator<DenseBackend>>
LocalOperator<DenseBackend>::clone() const {
    auto c = std::make_unique<LocalOperator<DenseBackend>>(mat_, name_, site_);
    return c;
}

// ── IdentityOperator<DenseBackend> ───────────────────────────────────────────

template<>
IdentityOperator<DenseBackend>::IdentityOperator(int site, int dim)
    : site_(site)
{
    mat_ = Eigen::MatrixXcd::Identity(dim, dim);
}

template<>
std::unique_ptr<AbstractLocalOperator<DenseBackend>>
IdentityOperator<DenseBackend>::clone() const {
    auto c = std::make_unique<IdentityOperator<DenseBackend>>(site_, static_cast<int>(mat_.rows()));
    return c;
}

} // namespace tenet
