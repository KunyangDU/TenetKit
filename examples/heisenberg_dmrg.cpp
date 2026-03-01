// examples/heisenberg_dmrg.cpp
// Demonstrates: build Heisenberg Hamiltonian, run DMRG, measure energy.
//
// Build:  cmake --build build && ./build/examples/heisenberg_dmrg
//
// TODO: implement after Phase 1 (core + MPS + algorithm) is complete.

#include "tenet/algorithm/dmrg.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/local_space/spin.hpp"
#include "tenet/utils/logger.hpp"

int main() {
    tenet::init_logger("heisenberg_dmrg");
    TENET_LOG_INFO("Heisenberg DMRG example – not yet implemented.");
    return 0;
}
