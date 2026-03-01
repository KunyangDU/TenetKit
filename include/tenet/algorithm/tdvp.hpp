#pragma once
// include/tenet/algorithm/tdvp.hpp
//
// TDVP time evolution (real-time and imaginary-time / tanTRG).
// See docs/C++重构设计方案.md §13.

#include "tenet/environment/environment.hpp"
#include "tenet/process_control/config.hpp"
#include "tenet/process_control/sweep_info.hpp"

#include <complex>
#include <vector>

namespace tenet {

// ── Single time step ──────────────────────────────────────────────────────────
// Propagates by τ via a full L2R + R2L sweep.
template<TensorBackend B = DenseBackend>
TDVPSweepInfo tdvp1_step(Environment<B>& env, std::complex<double> tau,
                          const TDVPConfig& cfg = {});

template<TensorBackend B = DenseBackend>
TDVPSweepInfo tdvp2_step(Environment<B>& env, std::complex<double> tau,
                          const TDVPConfig& cfg = {});

// ── Multi-step wrappers ───────────────────────────────────────────────────────
// time_steps: list of time values τ_1, τ_2, … (can be complex for imaginary time)
template<TensorBackend B = DenseBackend>
std::vector<TDVPSweepInfo> tdvp(DenseMPS<B>& psi, SparseMPO<B>& H,
                                  const std::vector<std::complex<double>>& time_steps,
                                  const TDVPConfig& cfg = {});

// ── tanTRG (thermal density matrix via imaginary-time TDVP on DenseMPO) ──────
// Returns (free_energies, energies) per beta step.
// TODO: DenseMPO type to be added in Phase 1.

} // namespace tenet
