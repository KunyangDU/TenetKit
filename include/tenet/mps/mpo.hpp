#pragma once
// include/tenet/mps/mpo.hpp
//
// SparseMPO<B>: L-site matrix product operator.
// See docs/C++重构设计方案.md §7.3.

#include "tenet/mps/mpo_tensor.hpp"

#include <utility>
#include <vector>

namespace tenet {

template<TensorBackend B = DenseBackend>
class SparseMPO {
public:
    explicit SparseMPO(int L) : L_(L), sites_(L, SparseMPOTensor<B>(1, 1)) {}

    SparseMPOTensor<B>&       operator[](int i)       { return sites_[i]; }
    const SparseMPOTensor<B>& operator[](int i) const { return sites_[i]; }

    int length() const noexcept { return L_; }

    // Returns (D_in, D_out) bond dimensions at site i.
    std::pair<int,int> bond_dim(int i) const {
        return {sites_[i].d_in(), sites_[i].d_out()};
    }

private:
    int L_;
    std::vector<SparseMPOTensor<B>> sites_;
};

// Default alias
using MPO = SparseMPO<DenseBackend>;

} // namespace tenet
