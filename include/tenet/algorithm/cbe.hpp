#pragma once
// include/tenet/algorithm/cbe.hpp
//
// Correlated Basis Expansion: adaptive bond-dimension growth used inside DMRG.
// See docs/C++重构设计方案.md §13.

#include "tenet/environment/cbe_environment.hpp"
#include "tenet/environment/environment.hpp"
#include "tenet/process_control/config.hpp"
#include "tenet/process_control/sweep_info.hpp"

namespace tenet {

// Expand bond at (site, site+1) using the specified SVD scheme.
template<TensorBackend B = DenseBackend>
CBEInfo cbe_expand(Environment<B>& env, int site,
                   const CBEConfig& cfg = {});

} // namespace tenet
