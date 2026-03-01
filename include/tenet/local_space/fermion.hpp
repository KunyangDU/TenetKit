#pragma once
// include/tenet/local_space/fermion.hpp
//
// Pre-built spinless and spinful fermion operator matrices (no symmetry, Phase 1).
// See docs/C++重构设计方案.md §12.

#include <Eigen/Dense>
#include <complex>

namespace tenet::fermion {

// ── Spinless fermion (dim=2: |0⟩, |1⟩) ──────────────────────────────────────
namespace spinless {
inline Eigen::MatrixXcd c()   // annihilation
{
    Eigen::MatrixXcd m(2, 2);
    m << 0.0, 1.0,
         0.0, 0.0;
    return m;
}
inline Eigen::MatrixXcd cdag() { return c().adjoint(); }
inline Eigen::MatrixXcd n()   { return cdag() * c(); }
inline Eigen::MatrixXcd Z()   // Jordan-Wigner string
{
    Eigen::MatrixXcd m(2, 2);
    m << 1.0, 0.0,
         0.0, -1.0;
    return m;
}
inline Eigen::MatrixXcd Id() { return Eigen::MatrixXcd::Identity(2, 2); }
} // namespace spinless

// ── Spinful fermion (dim=4: |0⟩, |↑⟩, |↓⟩, |↑↓⟩) ───────────────────────────
namespace spinful {
// TODO: implement for Hubbard model (Phase 1 milestone)
} // namespace spinful

} // namespace tenet::fermion
