#pragma once
// include/tenet/mps/mps.hpp
//
// DenseMPS<B>: matrix product state, L site tensors.
// See docs/C++重构设计方案.md §7.2.

#include "tenet/mps/mps_tensor.hpp"

#include <complex>
#include <string>
#include <utility>
#include <vector>

namespace tenet {

// Forward-declare the zero-copy adjoint wrapper (defined in adjoint.hpp).
template<typename Base> class Adjoint;

template<TensorBackend B = DenseBackend>
class DenseMPS {
public:
    using Space      = typename B::Space;
    using SiteTensor = MPSTensor<B>;

    explicit DenseMPS(int L);

    // Site access (0-indexed)
    SiteTensor&       operator[](int i)       { return sites_[i]; }
    const SiteTensor& operator[](int i) const { return sites_[i]; }

    int length() const noexcept { return L_; }

    // Orthogonality centre
    int  center_left()  const noexcept { return cleft_; }
    int  center_right() const noexcept { return cright_; }
    void set_center(int l, int r)      { cleft_ = l; cright_ = r; }
    bool is_canonical() const noexcept { return cleft_ == cright_; }

    void move_center_to(int target);
    void left_canonicalize(int from, int to);
    void right_canonicalize(int from, int to);

    // Norms / inner product
    std::complex<double> inner(const DenseMPS& other) const;
    double norm() const;
    void   normalize();

    // Bond dimensions
    int bond_dim(int bond) const;     // bond ∈ [0, L]
    int max_bond_dim()     const;

    // Zero-copy adjoint (⟨ψ|)
    Adjoint<DenseMPS> adjoint();

    // I/O (HDF5)
    void save(const std::string& path) const;
    static DenseMPS load(const std::string& path);

    // Random initialisation
    static DenseMPS random(int L, int D, const std::vector<Space>& phys_spaces);

private:
    int L_;
    std::vector<SiteTensor> sites_;
    int cleft_{0}, cright_{0};
};

// ── Adjoint wrapper (zero-copy reference) ────────────────────────────────────
template<typename Base>
class Adjoint {
public:
    explicit Adjoint(Base& b) : base_(b) {}
    Base& unadjoint() { return base_; }
    int   length()  const { return base_.length(); }
    auto  operator[](int i) const { return base_[i].adjoint(); }

private:
    Base& base_;
};

// Default alias (users write MPS, not DenseMPS<DenseBackend>)
using MPS = DenseMPS<DenseBackend>;

} // namespace tenet
