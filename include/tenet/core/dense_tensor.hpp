#pragma once
// include/tenet/core/dense_tensor.hpp
//
// DenseTensor: an N-rank complex tensor stored in row-major order.
// The underlying storage is a flat std::vector<std::complex<double>>.
//
// All shape information is encoded as std::vector<TrivialSpace>; each
// TrivialSpace carries the dimension of that leg.
//
// Key design decisions (see docs/C++重构设计方案.md §6):
//   • No raw new/delete – RAII via std::vector.
//   • matricize() returns an Eigen::Map when no permutation is needed (zero-copy).
//   • [[nodiscard]] on operations that return a new tensor.

#include "tenet/core/space.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace tenet {

class DenseTensor {
public:
    using Scalar = std::complex<double>;

    // ── Construction ─────────────────────────────────────────────────────────

    DenseTensor() = default;

    // Construct zero tensor from a list of spaces.
    explicit DenseTensor(std::vector<TrivialSpace> legs);

    // Construct from spaces + pre-filled data (row-major).
    DenseTensor(std::vector<TrivialSpace> legs, std::vector<Scalar> data);

    // ── Shape queries ─────────────────────────────────────────────────────────

    int rank()    const noexcept { return static_cast<int>(legs_.size()); }
    int dim(int i) const { assert(i >= 0 && i < rank()); return legs_[i].dim(); }
    int64_t numel() const noexcept;   // product of all dims

    const TrivialSpace& space(int i) const { assert(i >= 0 && i < rank()); return legs_[i]; }
    const std::vector<TrivialSpace>& spaces() const noexcept { return legs_; }

    // ── Data access ───────────────────────────────────────────────────────────

    // Multi-index access (debug; hot paths should use data() directly).
    Scalar&       operator()(std::initializer_list<int> idx);
    const Scalar& operator()(std::initializer_list<int> idx) const;

    // Raw pointer to row-major data.
    Scalar*       data()       noexcept { return storage_.data(); }
    const Scalar* data() const noexcept { return storage_.data(); }

    // ── Core operations ───────────────────────────────────────────────────────

    // Permute legs: perm[new_i] = old_i.  Reorders data accordingly.
    [[nodiscard]] DenseTensor permute(const std::vector<int>& perm) const;

    // Reshape: reinterpret shape without moving data (total numel must match).
    [[nodiscard]] DenseTensor reshape(std::vector<TrivialSpace> new_legs) const;

    // Matricize: fuse row_legs into rows, col_legs into columns.
    // Returns Eigen::Map if no permutation is needed (zero-copy), otherwise
    // permutes first and returns a matrix backed by a fresh allocation.
    Eigen::MatrixXcd matricize(const std::vector<int>& row_legs,
                                const std::vector<int>& col_legs) const;

    // Inverse of matricize: rebuild tensor from a matrix result.
    static DenseTensor from_matrix(const Eigen::MatrixXcd& mat,
                                   std::vector<TrivialSpace> legs,
                                   bool row_is_left);

    // Conjugate-transpose: flips dual flags on all legs and conjugates data.
    [[nodiscard]] DenseTensor adjoint() const;

    // In-place conjugate (no transpose, no leg flip).
    DenseTensor& conj();

    // this = α * this + β * other
    DenseTensor& axpby(Scalar alpha, Scalar beta, const DenseTensor& other);

    double norm() const;
    DenseTensor& normalize();

    // ── Leg fusion / split ────────────────────────────────────────────────────

    // Fuse legs [from, to) into a single leg (product of dims).
    [[nodiscard]] DenseTensor fuse(int from, int to) const;

    // Split leg `leg` into sub_spaces (product of sub-dims must equal leg.dim()).
    [[nodiscard]] DenseTensor split(int leg,
                                    const std::vector<TrivialSpace>& sub_spaces) const;

private:
    std::vector<TrivialSpace> legs_;
    std::vector<Scalar>       storage_;   // row-major

    int64_t linear_idx(const std::vector<int>& idx) const;
};

} // namespace tenet
