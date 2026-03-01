#pragma once
// include/tenet/process_control/config.hpp
//
// Algorithm configuration structs (replaces Julia's DMRGalgo, TDVPalgo, etc.)
// See docs/C++重构设计方案.md §13.

#include "tenet/core/factorization.hpp"

#include <cstdint>

namespace tenet {

// ── DMRG ─────────────────────────────────────────────────────────────────────
struct DMRGConfig {
    int         max_sweeps  = 20;
    double      E_tol       = 1e-8;    // Energy convergence threshold
    double      S_tol       = 1e-5;    // Entropy convergence threshold
    TruncParams trunc       = {};
    int         krylov_dim  = 8;
    int         max_krylov_iter = 1;
    double      krylov_tol  = 1e-6;
    int         gc_sweep    = 2;       // GC every N sweeps (stub; N/A in C++)
};

// ── TDVP ─────────────────────────────────────────────────────────────────────
struct TDVPConfig {
    TruncParams trunc    = {};
    double      tol      = 1e-8;
    int         krylov_dim = 32;
    int         max_krylov_iter = 1;
    double      krylov_tol = 1e-8;
};

// ── CBE ──────────────────────────────────────────────────────────────────────
enum class CBESVDScheme { FullSVD, RandSVD, DynamicSVD };

struct CBEConfig {
    int          target_D   = 0;      // Target bond dimension (0 = unlimited)
    double       lambda     = 1.2;    // Oversampling ratio for randSVD
    int          n_boundary = 4;      // Sites near boundary using fullSVD
    CBESVDScheme scheme     = CBESVDScheme::DynamicSVD;
};

// ── SETTN ─────────────────────────────────────────────────────────────────────
struct SETTNConfig {
    int         max_order = 10;
    double      tol       = 1e-10;
    TruncParams trunc     = {};
};

} // namespace tenet
