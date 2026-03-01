#pragma once
// include/tenet/process_control/sweep_info.hpp
//
// Convergence tracking structs for DMRG and TDVP sweeps.

#include <vector>

namespace tenet {

struct DMRGSweepInfo {
    int    sweep           = 0;
    double ground_energy   = 0.0;
    double delta_energy    = 0.0;
    double max_entropy     = 0.0;
    double delta_entropy   = 0.0;
    bool   converged       = false;

    std::vector<double> site_energies;
    std::vector<double> entanglement_entropy;
};

struct TDVPSweepInfo {
    int    step         = 0;
    double time         = 0.0;
    double norm         = 1.0;
    double norm_error   = 0.0;
    double truncation_err = 0.0;
};

struct CBEInfo {
    int initial_D = 0;
    int final_D   = 0;
    int site      = 0;
};

} // namespace tenet
