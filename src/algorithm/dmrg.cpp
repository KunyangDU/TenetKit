// src/algorithm/dmrg.cpp
//
// DMRG ground-state optimisation: single-site and two-site variants.
// See docs/C++重构设计方案.md §13.

#include "tenet/algorithm/dmrg.hpp"
#include "tenet/core/factorization.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/environment/environment.hpp"
#include "tenet/hamiltonian/ham_action.hpp"
#include "tenet/hamiltonian/projective_ham.hpp"
#include "tenet/krylov/lanczos.hpp"
#include "tenet/mps/canonical.hpp"
#include "tenet/mps/mps_tensor.hpp"

#include <cassert>
#include <cmath>
#include <limits>

namespace tenet {

// ── Helper ────────────────────────────────────────────────────────────────────

namespace {
LanczosConfig make_lanczos_cfg(const DMRGConfig& cfg)
{
    LanczosConfig lc;
    lc.krylov_dim = cfg.krylov_dim;
    lc.max_iter   = cfg.max_krylov_iter;
    lc.tol        = cfg.krylov_tol;
    return lc;
}
} // anonymous namespace

// ── dmrg_sweep: SingleSite + L2R ──────────────────────────────────────────────

template<>
void dmrg_sweep<DenseBackend>(Environment<DenseBackend>& env,
                               const DMRGConfig&           cfg,
                               DMRGSweepInfo&              info,
                               SingleSite, L2R)
{
    int L = env.length();
    auto& psi = env.psi();
    LanczosConfig lcfg = make_lanczos_cfg(cfg);

    info.site_energies.assign(L, 0.0);

    for (int site = 0; site < L; ++site) {
        // Build 1-site effective Hamiltonian and optimise
        auto H_eff = proj1(env, site, 0.0);
        auto results = lanczos_eigs(
            [&H_eff](const DenseTensor& v) { return apply(H_eff, v); },
            psi[site].data(), 1, lcfg);

        info.site_energies[site] = results[0].eigenvalue;
        psi[site].data() = std::move(results[0].eigenvector);

        if (site < L - 1) {
            // Left-canonicalize psi[site], push R into psi[site+1]
            Eigen::MatrixXcd R     = psi[site].left_canonicalize();
            int              new_dl = static_cast<int>(R.rows());
            Eigen::MatrixXcd new_M  = R * psi[site + 1].as_matrix_right();

            psi[site + 1] = MPSTensor<DenseBackend>::from_right_matrix(
                new_M,
                TrivialSpace(new_dl),
                TrivialSpace(psi[site + 1].phys_dim()),
                TrivialSpace(psi[site + 1].right_dim()));

            env.push_right(site);
            psi.set_center(site + 1, site + 1);
        }
    }

    info.ground_energy = info.site_energies[L - 1];
}

// ── dmrg_sweep: SingleSite + R2L ──────────────────────────────────────────────

template<>
void dmrg_sweep<DenseBackend>(Environment<DenseBackend>& env,
                               const DMRGConfig&           cfg,
                               DMRGSweepInfo&              info,
                               SingleSite, R2L)
{
    int L = env.length();
    auto& psi = env.psi();
    LanczosConfig lcfg = make_lanczos_cfg(cfg);

    info.site_energies.assign(L, 0.0);

    for (int site = L - 1; site >= 0; --site) {
        // Build 1-site effective Hamiltonian and optimise
        auto H_eff = proj1(env, site, 0.0);
        auto results = lanczos_eigs(
            [&H_eff](const DenseTensor& v) { return apply(H_eff, v); },
            psi[site].data(), 1, lcfg);

        info.site_energies[site] = results[0].eigenvalue;
        psi[site].data() = std::move(results[0].eigenvector);

        if (site > 0) {
            // Right-canonicalize psi[site], push L into psi[site-1]
            Eigen::MatrixXcd L_mat  = psi[site].right_canonicalize();
            int              new_dr = static_cast<int>(L_mat.cols());
            Eigen::MatrixXcd new_M  = psi[site - 1].as_matrix_left() * L_mat;

            psi[site - 1] = MPSTensor<DenseBackend>::from_left_matrix(
                new_M,
                TrivialSpace(psi[site - 1].left_dim()),
                TrivialSpace(psi[site - 1].phys_dim()),
                TrivialSpace(new_dr));

            // push_left(site) updates right_envs_[site] using psi[site]
            env.push_left(site);
            psi.set_center(site - 1, site - 1);
        }
    }

    info.ground_energy = info.site_energies[0];
}

// ── dmrg_sweep: DoubleSite + L2R ──────────────────────────────────────────────

template<>
void dmrg_sweep<DenseBackend>(Environment<DenseBackend>& env,
                               const DMRGConfig&           cfg,
                               DMRGSweepInfo&              info,
                               DoubleSite, L2R)
{
    int L = env.length();
    assert(L >= 2);
    auto& psi = env.psi();
    LanczosConfig lcfg = make_lanczos_cfg(cfg);

    info.site_energies.assign(L - 1, 0.0);

    for (int site = 0; site < L - 1; ++site) {
        // Merge two-site tensor: (D_l, d_l, D_m) x (D_m, d_r, D_r) → (D_l, d_l, d_r, D_r)
        DenseTensor psi_ab = contract(psi[site].data(), psi[site + 1].data(), {{2, 0}});

        auto H_eff = proj2(env, site, site + 1, 0.0);
        auto results = lanczos_eigs(
            [&H_eff](const DenseTensor& v) { return apply2(H_eff, v); },
            psi_ab, 1, lcfg);

        info.site_energies[site] = results[0].eigenvalue;
        psi_ab = std::move(results[0].eigenvector);

        // SVD with truncation; split at leg 2: (D_l, d_l | d_r, D_r)
        SVDResult s = svd(psi_ab, 2, cfg.trunc);
        int D_new = s.bond_dim;

        // psi[site] = U (left-canonical), shape (D_l, d_l, D_new)
        psi[site].data() = s.U;

        // psi[site+1] = diag(S) * Vt, shape (D_new, d_r, D_r)
        int d_r = psi[site + 1].phys_dim();
        int D_r = psi[site + 1].right_dim();
        Eigen::MatrixXcd Vt_mat  = s.Vt.matricize({0}, {1, 2});    // (D_new, d_r*D_r)
        Eigen::MatrixXcd SVt_mat = s.S.asDiagonal() * Vt_mat;      // (D_new, d_r*D_r)

        psi[site + 1] = MPSTensor<DenseBackend>::from_right_matrix(
            SVt_mat,
            TrivialSpace(D_new),
            TrivialSpace(d_r),
            TrivialSpace(D_r));

        // push_right(site) updates left_envs_[site+1] using psi[site]=U
        env.push_right(site);
        psi.set_center(site + 1, site + 1);
    }

    info.ground_energy = info.site_energies[L - 2];
}

// ── dmrg_sweep: DoubleSite + R2L ──────────────────────────────────────────────

template<>
void dmrg_sweep<DenseBackend>(Environment<DenseBackend>& env,
                               const DMRGConfig&           cfg,
                               DMRGSweepInfo&              info,
                               DoubleSite, R2L)
{
    int L = env.length();
    assert(L >= 2);
    auto& psi = env.psi();
    LanczosConfig lcfg = make_lanczos_cfg(cfg);

    info.site_energies.assign(L - 1, 0.0);

    for (int site = L - 2; site >= 0; --site) {
        // Merge two-site tensor
        DenseTensor psi_ab = contract(psi[site].data(), psi[site + 1].data(), {{2, 0}});

        auto H_eff = proj2(env, site, site + 1, 0.0);
        auto results = lanczos_eigs(
            [&H_eff](const DenseTensor& v) { return apply2(H_eff, v); },
            psi_ab, 1, lcfg);

        info.site_energies[site] = results[0].eigenvalue;
        psi_ab = std::move(results[0].eigenvector);

        // SVD with truncation; split at leg 2
        SVDResult s = svd(psi_ab, 2, cfg.trunc);
        int D_new = s.bond_dim;

        // psi[site] = U * diag(S), shape (D_l, d_l, D_new)
        int d_l = psi[site].phys_dim();
        int D_l = psi[site].left_dim();
        Eigen::MatrixXcd U_mat  = s.U.matricize({0, 1}, {2});    // (D_l*d_l, D_new)
        Eigen::MatrixXcd US_mat = U_mat * s.S.asDiagonal();      // (D_l*d_l, D_new)

        psi[site] = MPSTensor<DenseBackend>::from_left_matrix(
            US_mat,
            TrivialSpace(D_l),
            TrivialSpace(d_l),
            TrivialSpace(D_new));

        // psi[site+1] = Vt (right-canonical), shape (D_new, d_r, D_r)
        psi[site + 1].data() = s.Vt;

        // push_left(site+1) updates right_envs_[site+1] using psi[site+1]=Vt
        env.push_left(site + 1);
        psi.set_center(site, site);
    }

    info.ground_energy = info.site_energies[0];
}

// ── dmrg1 ─────────────────────────────────────────────────────────────────────

template<>
DMRGResult<DenseBackend> dmrg1(Environment<DenseBackend>& env, const DMRGConfig& cfg)
{
    int L = env.length();
    auto& psi = env.psi();

    // Initialise: right-canonical form with center at site 0
    psi.right_canonicalize(0, L - 1);
    psi.set_center(0, 0);
    env.build_all();

    DMRGResult<DenseBackend> result;
    double E_prev = std::numeric_limits<double>::max();

    for (int sw = 0; sw < cfg.max_sweeps; ++sw) {
        DMRGSweepInfo info_lr, info_rl;
        info_lr.sweep = 2 * sw;
        info_rl.sweep = 2 * sw + 1;

        dmrg_sweep<DenseBackend>(env, cfg, info_lr, SingleSite{}, L2R{});
        dmrg_sweep<DenseBackend>(env, cfg, info_rl, SingleSite{}, R2L{});

        double E_new = info_rl.ground_energy;
        double dE    = std::abs(E_new - E_prev);

        info_rl.delta_energy = dE;
        info_rl.converged    = (sw > 0) && (dE < cfg.E_tol);

        result.history.push_back(info_lr);
        result.history.push_back(info_rl);

        if (info_rl.converged) {
            result.converged     = true;
            result.ground_energy = E_new;
            break;
        }

        E_prev = E_new;
    }

    if (!result.converged && !result.history.empty())
        result.ground_energy = result.history.back().ground_energy;

    return result;
}

// ── dmrg2 ─────────────────────────────────────────────────────────────────────

template<>
DMRGResult<DenseBackend> dmrg2(Environment<DenseBackend>& env, const DMRGConfig& cfg)
{
    int L = env.length();
    auto& psi = env.psi();

    // Initialise: right-canonical form with center at site 0
    psi.right_canonicalize(0, L - 1);
    psi.set_center(0, 0);
    env.build_all();

    DMRGResult<DenseBackend> result;
    double E_prev = std::numeric_limits<double>::max();

    for (int sw = 0; sw < cfg.max_sweeps; ++sw) {
        DMRGSweepInfo info_lr, info_rl;
        info_lr.sweep = 2 * sw;
        info_rl.sweep = 2 * sw + 1;

        dmrg_sweep<DenseBackend>(env, cfg, info_lr, DoubleSite{}, L2R{});
        dmrg_sweep<DenseBackend>(env, cfg, info_rl, DoubleSite{}, R2L{});

        double E_new = info_rl.ground_energy;
        double dE    = std::abs(E_new - E_prev);

        info_rl.delta_energy = dE;
        info_rl.converged    = (sw > 0) && (dE < cfg.E_tol);

        result.history.push_back(info_lr);
        result.history.push_back(info_rl);

        if (info_rl.converged) {
            result.converged     = true;
            result.ground_energy = E_new;
            break;
        }

        E_prev = E_new;
    }

    if (!result.converged && !result.history.empty())
        result.ground_energy = result.history.back().ground_energy;

    return result;
}

// ── dmrg ─────────────────────────────────────────────────────────────────────

template<>
DMRGResult<DenseBackend> dmrg(DenseMPS<DenseBackend>& psi,
                               SparseMPO<DenseBackend>& H,
                               const DMRGConfig& cfg)
{
    Environment<DenseBackend> env(psi, H);
    return dmrg1(env, cfg);
}

} // namespace tenet
