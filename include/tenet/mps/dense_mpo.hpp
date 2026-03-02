#pragma once
// include/tenet/mps/dense_mpo.hpp
//
// DenseMPO<B>: L-site dense matrix product operator.
// Each site tensor has shape (D_l, d_bra, d_ket, D_r).
// Used as density-matrix representation in tanTRG and SETTN.

#include "tenet/mps/dense_mpo_tensor.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace tenet {

template<TensorBackend B = DenseBackend>
class DenseMPO {
public:
    using SiteTensor = DenseMPOTensor<B>;

    explicit DenseMPO(int L) : L_(L), sites_(L) {}

    // Site access (0-indexed)
    SiteTensor&       operator[](int i)       { return sites_[i]; }
    const SiteTensor& operator[](int i) const { return sites_[i]; }

    int length() const noexcept { return L_; }

    // Orthogonality centre
    int  center()     const noexcept { return center_; }
    void set_center(int i)           { center_ = i; }

    // Build identity MPO: ρ[s] = δ_{σ',σ}, shape (1, d, d, 1).
    static DenseMPO identity(int L, int d);

    // Sweep canonicalization (in-place)
    void left_canonicalize(int from, int to);    // sites [from, to)
    void right_canonicalize(int from, int to);   // sites (from, to]
    void move_center_to(int target);

    // Normalize all data by Frobenius norm; returns log(norm).
    // Use for lnZ accumulation in tanTRG: lnZ += 2 * normalize().
    double normalize();

    // Tr(ρ†ρ) — useful for norm validation.
    double trace_sq() const;

    // I/O (stub)
    void save(const std::string&) const {}

private:
    int L_;
    std::vector<SiteTensor> sites_;
    int center_{0};
};

// Default alias
using MPODense = DenseMPO<DenseBackend>;

} // namespace tenet
