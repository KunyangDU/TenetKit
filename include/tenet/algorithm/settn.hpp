#pragma once
// include/tenet/algorithm/settn.hpp
//
// Series Expansion Tensor Network (SETTN): finite-temperature simulation.
// See docs/C++重构设计方案.md §13.

#include "tenet/mps/mps.hpp"
#include "tenet/mps/mpo.hpp"
#include "tenet/process_control/config.hpp"

#include <vector>

namespace tenet {

struct SETTNResult {
    std::vector<double> free_energies;
    std::vector<double> energies;
    std::vector<double> beta_grid;
};

template<TensorBackend B = DenseBackend>
SETTNResult settn(DenseMPS<B>& rho, SparseMPO<B>& H,
                   const std::vector<double>& beta_grid,
                   const SETTNConfig& cfg = {});

} // namespace tenet
