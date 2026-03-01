#pragma once
// include/tenet/local_space/spin.hpp
//
// Pre-built spin operator matrices (dense, no symmetry).
// Phase 1 uses plain Eigen matrices; symmetry will be in Phase 2.
// See docs/C++重构设计方案.md §12.

#include <Eigen/Dense>
#include <complex>
#include <cmath>

namespace tenet::spin {

// ── Spin-1/2 ──────────────────────────────────────────────────────────────────
namespace half {
inline Eigen::MatrixXcd Sz() {
    Eigen::MatrixXcd m(2, 2);
    m << 0.5, 0.0,
         0.0, -0.5;
    return m;
}
inline Eigen::MatrixXcd Sp() {
    Eigen::MatrixXcd m(2, 2);
    m << 0.0, 1.0,
         0.0, 0.0;
    return m;
}
inline Eigen::MatrixXcd Sm() { return Sp().adjoint(); }
inline Eigen::MatrixXcd Id() { return Eigen::MatrixXcd::Identity(2, 2); }
// S·S coupling term (returns {Sz*Sz, 0.5*Sp*Sm, 0.5*Sm*Sp} for addIntr2 use)
} // namespace half

// ── Spin-1 ───────────────────────────────────────────────────────────────────
namespace one {
inline Eigen::MatrixXcd Sz() {
    Eigen::MatrixXcd m = Eigen::MatrixXcd::Zero(3, 3);
    m(0, 0) = 1.0;  m(2, 2) = -1.0;
    return m;
}
inline Eigen::MatrixXcd Sp() {
    Eigen::MatrixXcd m = Eigen::MatrixXcd::Zero(3, 3);
    m(0, 1) = std::sqrt(2.0);  m(1, 2) = std::sqrt(2.0);
    return m;
}
inline Eigen::MatrixXcd Sm() { return Sp().adjoint(); }
inline Eigen::MatrixXcd Id() { return Eigen::MatrixXcd::Identity(3, 3); }
} // namespace one

} // namespace tenet::spin
