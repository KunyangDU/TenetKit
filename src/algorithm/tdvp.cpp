// src/algorithm/tdvp.cpp
//
// TDVP time evolution: single-site (TDVP1) and two-site (TDVP2).
// Reference: Haegeman et al., Phys. Rev. B 94, 165116 (2016).
//
// TDVP1 (one full time step τ = L2R half-sweep + R2L half-sweep):
//   L2R: for s=0..L-2: forward exp(-i H_eff τ/2) on site, QR, push_right,
//                       backward exp(+i H_bond τ/2) on bond, absorb into s+1.
//        for s=L-1:     forward exp(-i H_eff τ/2) only (boundary).
//   R2L: for s=L-1..1:  forward exp(-i H_eff τ/2) on site, RQ, push_left,
//                       backward exp(+i H_bond τ/2) on bond, absorb into s-1.
//        for s=0:        forward exp(-i H_eff τ/2) only (boundary).
//
// Bond Hamiltonian at bond (s, s+1):
//   After push_right(s): use left_env(s+1) and right_env(s).
//   After push_left(s):  use left_env(s) and right_env(s-1).
//
// TDVP2 (two-site):
//   L2R: forward exp on two-site tensor, SVD, push_right,
//        backward exp on single-site psi[s+1] (except last bond).
//   R2L: symmetric, backward exp on psi[s] (except first site).

#include "tenet/algorithm/tdvp.hpp"
#include "tenet/core/dense_tensor.hpp"
#include "tenet/core/factorization.hpp"
#include "tenet/core/space.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/environment/environment.hpp"
#include "tenet/hamiltonian/ham_action.hpp"
#include "tenet/hamiltonian/projective_ham.hpp"
#include "tenet/krylov/arnoldi.hpp"
#include "tenet/mps/mps_tensor.hpp"

#include <cassert>
#include <cmath>

namespace tenet {

// ── Helpers ───────────────────────────────────────────────────────────────────

namespace {

ArnoldiConfig make_arnoldi_cfg(const TDVPConfig& cfg)
{
    ArnoldiConfig ac;
    ac.krylov_dim = cfg.krylov_dim;
    ac.max_iter   = cfg.max_krylov_iter;
    ac.tol        = cfg.krylov_tol;
    return ac;
}

DenseTensor mat_to_tensor(const Eigen::MatrixXcd& m)
{
    return DenseTensor::from_matrix(
        m, {TrivialSpace(static_cast<int>(m.rows())),
             TrivialSpace(static_cast<int>(m.cols()))}, true);
}

// ── Zero-site (bond) Hamiltonian action ──────────────────────────────────────
//
// H_bond * C = Σ_i  El[i] * C * Er[i]
//
// El[i]: (D_bra_l, D_ket_l),  Er[i]: (D_ket_r, D_bra_r),  C: (D_ket_l, D_ket_r)
// The MPO auxiliary index threads through the bond unchanged (identity at bond).
//
// Environments to pass:
//   L2R bond at (s, s+1) after push_right(s): L=left_env(s+1), R=right_env(s)
//   R2L bond at (s-1,s) after push_left(s):   L=left_env(s),   R=right_env(s-1)
//
static DenseTensor apply_bond(const SparseLeftEnvTensor<DenseBackend>&  L,
                               const SparseRightEnvTensor<DenseBackend>& R,
                               const DenseTensor&                         C)
{
    int D = L.dim();
    assert(R.dim() == D);
    DenseTensor result(C.spaces());
    for (int i = 0; i < D; ++i) {
        if (!L.has(i) || !R.has(i)) continue;
        const DenseTensor& el = L[i]->data();            // (D_bra_l, D_ket_l)
        const DenseTensor& er = R[i]->data();            // (D_ket_r, D_bra_r)
        DenseTensor T1 = contract(el, C,  {{1, 0}});     // (D_bra_l, D_ket_r)
        DenseTensor T2 = contract(T1, er, {{1, 0}});     // (D_bra_l, D_bra_r)
        result.axpby({1.0, 0.0}, {1.0, 0.0}, T2);
    }
    return result;
}

} // anonymous namespace


// ── tdvp1_step ────────────────────────────────────────────────────────────────

template<>
TDVPSweepInfo tdvp1_step<DenseBackend>(Environment<DenseBackend>& env,
                                        std::complex<double>         tau,
                                        const TDVPConfig&            cfg)
{
    int L = env.length();
    auto& psi = env.psi();
    ArnoldiConfig acfg = make_arnoldi_cfg(cfg);

    // α_fwd = -i*tau/2 → exp(α_fwd * H) = exp(-i H tau/2)  [forward]
    std::complex<double> alpha_fwd{ tau.imag() / 2.0, -tau.real() / 2.0};
    // α_bwd = +i*tau/2 → exp(α_bwd * H) = exp(+i H tau/2)  [backward]
    std::complex<double> alpha_bwd{-tau.imag() / 2.0,  tau.real() / 2.0};

    // ── L2R half-sweep ────────────────────────────────────────────────────────
    for (int site = 0; site < L; ++site) {
        auto H_eff = proj1(env, site);

        psi[site].data() = arnoldi_expm_vec(
            [&H_eff](const DenseTensor& v) { return apply(H_eff, v); },
            psi[site].data(), alpha_fwd, acfg);

        if (site < L - 1) {
            Eigen::MatrixXcd R_mat = psi[site].left_canonicalize();
            int D_new = static_cast<int>(R_mat.rows());

            env.push_right(site);
            psi.set_center(site + 1, site + 1);

            // Bond at (site, site+1): L=left_env(site+1), R=right_env(site+1)
            DenseTensor R_t   = mat_to_tensor(R_mat);
            const auto& L_env = env.left_env(site + 1);
            const auto& R_env = env.right_env(site + 1);
            R_t = arnoldi_expm_vec(
                [&L_env, &R_env](const DenseTensor& C) {
                    return apply_bond(L_env, R_env, C);
                },
                R_t, alpha_bwd, acfg);

            Eigen::MatrixXcd R_evo = R_t.matricize({0}, {1});
            Eigen::MatrixXcd new_M = R_evo * psi[site + 1].as_matrix_right();
            psi[site + 1] = MPSTensor<DenseBackend>::from_right_matrix(
                new_M,
                TrivialSpace(D_new),
                TrivialSpace(psi[site + 1].phys_dim()),
                TrivialSpace(psi[site + 1].right_dim()));
        }
    }

    // ── R2L half-sweep ────────────────────────────────────────────────────────
    for (int site = L - 1; site >= 0; --site) {
        auto H_eff = proj1(env, site);

        psi[site].data() = arnoldi_expm_vec(
            [&H_eff](const DenseTensor& v) { return apply(H_eff, v); },
            psi[site].data(), alpha_fwd, acfg);

        if (site > 0) {
            Eigen::MatrixXcd L_mat = psi[site].right_canonicalize();
            int D_new = static_cast<int>(L_mat.cols());

            env.push_left(site);
            psi.set_center(site - 1, site - 1);

            // Bond at (site-1, site): L=left_env(site), R=right_env(site)
            DenseTensor L_t   = mat_to_tensor(L_mat);
            const auto& L_env = env.left_env(site);
            const auto& R_env = env.right_env(site);
            L_t = arnoldi_expm_vec(
                [&L_env, &R_env](const DenseTensor& C) {
                    return apply_bond(L_env, R_env, C);
                },
                L_t, alpha_bwd, acfg);

            Eigen::MatrixXcd L_evo = L_t.matricize({0}, {1});
            Eigen::MatrixXcd new_M = psi[site - 1].as_matrix_left() * L_evo;
            psi[site - 1] = MPSTensor<DenseBackend>::from_left_matrix(
                new_M,
                TrivialSpace(psi[site - 1].left_dim()),
                TrivialSpace(psi[site - 1].phys_dim()),
                TrivialSpace(D_new));
        }
    }

    psi.set_center(0, 0);

    double nrm = psi.norm();
    TDVPSweepInfo info;
    info.norm       = nrm;
    info.norm_error = std::abs(nrm - 1.0);
    return info;
}


// ── tdvp2_step ────────────────────────────────────────────────────────────────

template<>
TDVPSweepInfo tdvp2_step<DenseBackend>(Environment<DenseBackend>& env,
                                        std::complex<double>         tau,
                                        const TDVPConfig&            cfg)
{
    int L = env.length();
    assert(L >= 2);
    auto& psi = env.psi();
    ArnoldiConfig acfg = make_arnoldi_cfg(cfg);

    std::complex<double> alpha_fwd{ tau.imag() / 2.0, -tau.real() / 2.0};
    std::complex<double> alpha_bwd{-tau.imag() / 2.0,  tau.real() / 2.0};

    double total_trunc = 0.0;

    // ── L2R sweep ─────────────────────────────────────────────────────────────
    for (int site = 0; site < L - 1; ++site) {
        DenseTensor psi_ab = contract(psi[site].data(), psi[site + 1].data(),
                                      {{2, 0}});
        auto H_eff2 = proj2(env, site, site + 1);

        psi_ab = arnoldi_expm_vec(
            [&H_eff2](const DenseTensor& v) { return apply2(H_eff2, v); },
            psi_ab, alpha_fwd, acfg);

        SVDResult s   = svd(psi_ab, 2, cfg.trunc);
        total_trunc  += s.truncation_err;
        int D_new     = s.bond_dim;
        int d_r       = psi[site + 1].phys_dim();
        int D_r       = psi[site + 1].right_dim();

        psi[site].data() = s.U;

        env.push_right(site);
        psi.set_center(site + 1, site + 1);

        Eigen::MatrixXcd Vt_mat  = s.Vt.matricize({0}, {1, 2});
        Eigen::MatrixXcd SVt_mat = s.S.asDiagonal() * Vt_mat;

        psi[site + 1] = MPSTensor<DenseBackend>::from_right_matrix(
            SVt_mat,
            TrivialSpace(D_new),
            TrivialSpace(d_r),
            TrivialSpace(D_r));

        if (site < L - 2) {
            // Backward evolve psi[site+1] as single-site tensor
            auto H_eff1 = proj1(env, site + 1);
            psi[site + 1].data() = arnoldi_expm_vec(
                [&H_eff1](const DenseTensor& v) { return apply(H_eff1, v); },
                psi[site + 1].data(), alpha_bwd, acfg);
        }
        // At last bond (site == L-2): no backward single-site step needed.
    }

    // ── R2L sweep ─────────────────────────────────────────────────────────────
    for (int site = L - 2; site >= 0; --site) {
        DenseTensor psi_ab = contract(psi[site].data(), psi[site + 1].data(),
                                      {{2, 0}});
        auto H_eff2 = proj2(env, site, site + 1);

        psi_ab = arnoldi_expm_vec(
            [&H_eff2](const DenseTensor& v) { return apply2(H_eff2, v); },
            psi_ab, alpha_fwd, acfg);

        SVDResult s   = svd(psi_ab, 2, cfg.trunc);
        total_trunc  += s.truncation_err;
        int D_new     = s.bond_dim;
        int d_l       = psi[site].phys_dim();
        int D_l       = psi[site].left_dim();

        psi[site + 1].data() = s.Vt;

        env.push_left(site + 1);
        psi.set_center(site, site);

        Eigen::MatrixXcd U_mat  = s.U.matricize({0, 1}, {2});
        Eigen::MatrixXcd US_mat = U_mat * s.S.asDiagonal();

        psi[site] = MPSTensor<DenseBackend>::from_left_matrix(
            US_mat,
            TrivialSpace(D_l),
            TrivialSpace(d_l),
            TrivialSpace(D_new));

        if (site > 0) {
            // Backward evolve psi[site] as single-site tensor
            auto H_eff1 = proj1(env, site);
            psi[site].data() = arnoldi_expm_vec(
                [&H_eff1](const DenseTensor& v) { return apply(H_eff1, v); },
                psi[site].data(), alpha_bwd, acfg);
        }
        // At first site (site == 0): no backward single-site step needed.
    }

    psi.set_center(0, 0);

    double nrm = psi.norm();
    TDVPSweepInfo info;
    info.norm           = nrm;
    info.norm_error     = std::abs(nrm - 1.0);
    info.truncation_err = total_trunc;
    return info;
}


// ── tdvp ─────────────────────────────────────────────────────────────────────

template<>
std::vector<TDVPSweepInfo> tdvp<DenseBackend>(
    DenseMPS<DenseBackend>&                  psi,
    SparseMPO<DenseBackend>&                 H,
    const std::vector<std::complex<double>>& time_steps,
    const TDVPConfig&                        cfg)
{
    Environment<DenseBackend> env(psi, H);

    psi.right_canonicalize(0, psi.length() - 1);
    psi.set_center(0, 0);
    env.build_all();

    std::vector<TDVPSweepInfo> history;
    history.reserve(time_steps.size());

    double t = 0.0;
    for (int i = 0; i < static_cast<int>(time_steps.size()); ++i) {
        TDVPSweepInfo info = tdvp1_step(env, time_steps[i], cfg);
        info.step = i;
        info.time = t + time_steps[i].real();
        t         = info.time;
        history.push_back(info);
    }

    return history;
}

} // namespace tenet
