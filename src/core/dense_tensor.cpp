// src/core/dense_tensor.cpp
//
// Implementation of DenseTensor and DenseBackend (Phase 1).

#include "tenet/core/dense_tensor.hpp"
#include "tenet/core/backend.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace tenet {

// ── DenseTensor construction ──────────────────────────────────────────────────

DenseTensor::DenseTensor(std::vector<TrivialSpace> legs)
    : legs_(std::move(legs))
{
    int64_t n = 1;
    for (auto& s : legs_) n *= s.dim();
    storage_.assign(n, Scalar{0.0, 0.0});
}

DenseTensor::DenseTensor(std::vector<TrivialSpace> legs, std::vector<Scalar> data)
    : legs_(std::move(legs)), storage_(std::move(data))
{
    // Validate size matches
    int64_t n = 1;
    for (auto& s : legs_) n *= s.dim();
    assert(static_cast<int64_t>(storage_.size()) == n &&
           "data size does not match product of leg dimensions");
}

// ── Shape queries ─────────────────────────────────────────────────────────────

int64_t DenseTensor::numel() const noexcept {
    if (legs_.empty()) return 0;
    int64_t n = 1;
    for (auto& s : legs_) n *= s.dim();
    return n;
}

// ── Data access ───────────────────────────────────────────────────────────────

int64_t DenseTensor::linear_idx(const std::vector<int>& idx) const {
    assert(static_cast<int>(idx.size()) == rank());
    int64_t lin = 0;
    int64_t stride = 1;
    for (int k = rank() - 1; k >= 0; --k) {
        lin += idx[k] * stride;
        stride *= legs_[k].dim();
    }
    return lin;
}

DenseTensor::Scalar& DenseTensor::operator()(std::initializer_list<int> idx) {
    return storage_[linear_idx(std::vector<int>(idx))];
}

const DenseTensor::Scalar& DenseTensor::operator()(std::initializer_list<int> idx) const {
    return storage_[linear_idx(std::vector<int>(idx))];
}

// ── Core operations ───────────────────────────────────────────────────────────

DenseTensor DenseTensor::permute(const std::vector<int>& perm) const {
    int r = rank();
    assert(static_cast<int>(perm.size()) == r);

    // New legs
    std::vector<TrivialSpace> new_legs;
    new_legs.reserve(r);
    for (int k : perm) new_legs.push_back(legs_[k]);

    // Compute old strides (row-major)
    std::vector<int64_t> old_strides(r);
    if (r > 0) {
        old_strides[r - 1] = 1;
        for (int k = r - 2; k >= 0; --k)
            old_strides[k] = old_strides[k + 1] * legs_[k + 1].dim();
    }

    // Compute new strides
    std::vector<int64_t> new_strides(r);
    if (r > 0) {
        new_strides[r - 1] = 1;
        for (int k = r - 2; k >= 0; --k)
            new_strides[k] = new_strides[k + 1] * new_legs[k + 1].dim();
    }

    int64_t n = numel();
    std::vector<Scalar> new_data(n);

    // Map new index → old index via perm
    std::vector<int> multi(r, 0);
    for (int64_t flat = 0; flat < n; ++flat) {
        // Compute old linear index:
        //   new tensor's multi[k] corresponds to old leg perm[k]
        int64_t old_flat = 0;
        for (int k = 0; k < r; ++k)
            old_flat += multi[k] * old_strides[perm[k]];
        new_data[flat] = storage_[old_flat];

        // Increment multi-index
        for (int k = r - 1; k >= 0; --k) {
            ++multi[k];
            if (multi[k] < new_legs[k].dim()) break;
            multi[k] = 0;
        }
    }

    return DenseTensor(std::move(new_legs), std::move(new_data));
}

DenseTensor DenseTensor::reshape(std::vector<TrivialSpace> new_legs) const {
    // Validate total size
    int64_t old_n = numel();
    int64_t new_n = 1;
    for (auto& s : new_legs) new_n *= s.dim();
    assert(old_n == new_n && "reshape: total elements must be preserved");
    return DenseTensor(std::move(new_legs), storage_);
}

Eigen::MatrixXcd DenseTensor::matricize(const std::vector<int>& row_legs,
                                         const std::vector<int>& col_legs) const
{
    // Build permutation: row_legs first, then col_legs
    std::vector<int> perm;
    perm.insert(perm.end(), row_legs.begin(), row_legs.end());
    perm.insert(perm.end(), col_legs.begin(), col_legs.end());

    // Compute row_dim and col_dim
    int64_t rows = 1;
    for (int leg : row_legs) rows *= legs_[leg].dim();
    int64_t cols = 1;
    for (int leg : col_legs) cols *= legs_[leg].dim();

    // Permute (even if identity perm, this ensures data is contiguous in right order)
    DenseTensor tmp = permute(perm);

    // Convert row-major storage to column-major Eigen matrix
    Eigen::MatrixXcd result(rows, cols);
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            result(i, j) = tmp.storage_[i * cols + j];
    return result;
}

DenseTensor DenseTensor::from_matrix(const Eigen::MatrixXcd& mat,
                                      std::vector<TrivialSpace> legs,
                                      bool /*row_is_left*/)
{
    int64_t rows = mat.rows(), cols = mat.cols();
    std::vector<Scalar> data(rows * cols);
    // Convert column-major Eigen → row-major storage
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            data[i * cols + j] = mat(i, j);
    return DenseTensor(std::move(legs), std::move(data));
}

DenseTensor DenseTensor::adjoint() const {
    // Conjugate data + dual all legs
    std::vector<TrivialSpace> new_legs;
    new_legs.reserve(legs_.size());
    for (auto& s : legs_) new_legs.push_back(s.dual());

    std::vector<Scalar> new_data(storage_.size());
    for (size_t i = 0; i < storage_.size(); ++i)
        new_data[i] = std::conj(storage_[i]);

    return DenseTensor(std::move(new_legs), std::move(new_data));
}

DenseTensor& DenseTensor::conj() {
    for (auto& x : storage_) x = std::conj(x);
    return *this;
}

DenseTensor& DenseTensor::axpby(Scalar alpha, Scalar beta, const DenseTensor& other) {
    assert(storage_.size() == other.storage_.size());
    for (size_t i = 0; i < storage_.size(); ++i)
        storage_[i] = alpha * storage_[i] + beta * other.storage_[i];
    return *this;
}

double DenseTensor::norm() const {
    double s = 0.0;
    for (auto& x : storage_) s += std::norm(x);  // std::norm = |x|^2
    return std::sqrt(s);
}

DenseTensor& DenseTensor::normalize() {
    double n = norm();
    if (n > 0.0) {
        Scalar scale{1.0 / n, 0.0};
        for (auto& x : storage_) x *= scale;
    }
    return *this;
}

// ── Leg fusion / split ────────────────────────────────────────────────────────

DenseTensor DenseTensor::fuse(int from, int to) const {
    assert(from >= 0 && to <= rank() && from < to);
    int64_t fused_dim = 1;
    for (int k = from; k < to; ++k) fused_dim *= legs_[k].dim();

    std::vector<TrivialSpace> new_legs;
    for (int k = 0; k < from; ++k)          new_legs.push_back(legs_[k]);
    new_legs.push_back(TrivialSpace(static_cast<int>(fused_dim)));
    for (int k = to; k < rank(); ++k)        new_legs.push_back(legs_[k]);

    return DenseTensor(std::move(new_legs), storage_);  // same row-major data
}

DenseTensor DenseTensor::split(int leg, const std::vector<TrivialSpace>& sub_spaces) const {
    assert(leg >= 0 && leg < rank());
    int64_t product = 1;
    for (auto& s : sub_spaces) product *= s.dim();
    assert(product == legs_[leg].dim() && "split: product of sub-dims must equal leg dim");

    std::vector<TrivialSpace> new_legs;
    for (int k = 0; k < leg; ++k)              new_legs.push_back(legs_[k]);
    for (auto& s : sub_spaces)                 new_legs.push_back(s);
    for (int k = leg + 1; k < rank(); ++k)     new_legs.push_back(legs_[k]);

    return DenseTensor(std::move(new_legs), storage_);
}

// ── DenseBackend static methods ───────────────────────────────────────────────

DenseTensor DenseBackend::zeros(const std::vector<TrivialSpace>& legs) {
    return DenseTensor(legs);  // constructor already zero-fills
}

DenseTensor DenseBackend::random(const std::vector<TrivialSpace>& legs,
                                  std::optional<uint64_t> seed)
{
    int64_t n = 1;
    for (auto& s : legs) n *= s.dim();

    // Use thread_local engine for thread safety
    static thread_local std::mt19937_64 eng{std::random_device{}()};
    if (seed) eng.seed(*seed);

    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<DenseTensor::Scalar> data(n);
    for (auto& x : data)
        x = {dist(eng), dist(eng)};

    return DenseTensor(legs, std::move(data));
}

DenseTensor DenseBackend::identity(const TrivialSpace& space) {
    int d = space.dim();
    std::vector<TrivialSpace> legs = {TrivialSpace(d), TrivialSpace(d)};
    std::vector<DenseTensor::Scalar> data(d * d, DenseTensor::Scalar{0.0, 0.0});
    for (int i = 0; i < d; ++i)
        data[i * d + i] = {1.0, 0.0};
    return DenseTensor(std::move(legs), std::move(data));
}

} // namespace tenet
