#pragma once
// include/tenet/algorithm/dmrg.hpp
//
// DMRG ground-state search (single-site and two-site variants).
// See docs/C++重构设计方案.md §13.

#include "tenet/environment/environment.hpp"
#include "tenet/process_control/config.hpp"
#include "tenet/process_control/scheme.hpp"
#include "tenet/process_control/sweep_info.hpp"

#include <vector>

namespace tenet {

// ── Result type ───────────────────────────────────────────────────────────────
template<TensorBackend B = DenseBackend>
struct DMRGResult {
    double                      ground_energy = 0.0;
    bool                        converged     = false;
    std::vector<DMRGSweepInfo>  history;
};

// ── Public interface ──────────────────────────────────────────────────────────

// Single-site DMRG (modifies psi in-place via env)
template<TensorBackend B = DenseBackend>
DMRGResult<B> dmrg1(Environment<B>& env, const DMRGConfig& cfg = {});

// Two-site DMRG
template<TensorBackend B = DenseBackend>
DMRGResult<B> dmrg2(Environment<B>& env, const DMRGConfig& cfg = {});

// Convenience wrappers that accept psi and H directly
template<TensorBackend B = DenseBackend>
DMRGResult<B> dmrg(DenseMPS<B>& psi, SparseMPO<B>& H, const DMRGConfig& cfg = {});

// ── Internal sweep (compile-time dispatch via tag types) ─────────────────────
// Deleted primary template forces an error for invalid Scheme/Dir combinations.
template<typename Scheme, typename Dir, TensorBackend B>
void dmrg_sweep(Environment<B>&, const DMRGConfig&, DMRGSweepInfo&) = delete;

template<TensorBackend B>
void dmrg_sweep(Environment<B>& env, const DMRGConfig& cfg, DMRGSweepInfo& info,
                SingleSite, L2R);

template<TensorBackend B>
void dmrg_sweep(Environment<B>& env, const DMRGConfig& cfg, DMRGSweepInfo& info,
                SingleSite, R2L);

template<TensorBackend B>
void dmrg_sweep(Environment<B>& env, const DMRGConfig& cfg, DMRGSweepInfo& info,
                DoubleSite, L2R);

template<TensorBackend B>
void dmrg_sweep(Environment<B>& env, const DMRGConfig& cfg, DMRGSweepInfo& info,
                DoubleSite, R2L);

} // namespace tenet
