// src/algorithm/tdvp.cpp
//
// TDVP time evolution: single-site (TDVP1) and two-site (TDVP2).
// Reference: Haegeman et al., Phys. Rev. B 94, 165116 (2016).
//
// Numerical details (matching Julia reference):
//   • Energy shift E₀ = ⟨ψ|H|ψ⟩ passed to proj1/proj2 for Arnoldi stability.
//   • After each forward Arnoldi step: multiply by exp(α_fwd·E₀) to undo shift.
//   • Before QR/SVD: normalise tensor, save norm nmt.
//   • After QR/SVD:  restore nmt to the bond/centre tensor.
//   • Imaginary-time (τ.real() == 0): after each half-sweep normalise the
//     centre tensor and accumulate info.lnZ += 2·log(norm).

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

// Scale a DenseTensor in-place by complex scalar c.
// axpby(c, 0, self) is self-safe when beta==0.
inline void scale(DenseTensor& t, std::complex<double> c)
{
    t.axpby(c, {0.0, 0.0}, t);
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

    // Imaginary-time when τ has no real component (τ = {0, -dt}).
    bool is_imag_time = (tau.real() == 0.0);

    // ── Compute E₀ = ⟨ψ|H|ψ⟩ ─────────────────────────────────────────────────
    // Centre is at site 0 after build_all().  Unshifted H_eff at site 0.
    double E0;
    {
        auto H0 = proj1(env, 0, 0.0);
        E0 = inner(psi[0].data(), apply(H0, psi[0].data())).real();
    }

    double lnZ_accum = 0.0;

    // ── L2R half-sweep ────────────────────────────────────────────────────────
    for (int site = 0; site < L; ++site) {
        auto H_eff = proj1(env, site, E0);

        psi[site].data() = arnoldi_expm_vec(
            [&H_eff](const DenseTensor& v) { return apply(H_eff, v); },
            psi[site].data(), alpha_fwd, acfg);

        // Undo the energy shift: exp(α_fwd·E₀) · exp(α_fwd·(H−E₀)) = exp(α_fwd·H).
        scale(psi[site].data(), std::exp(alpha_fwd * E0));

        if (site < L - 1) {
            // Normalise before QR, save norm factor nmt.
            double nmt = psi[site].data().norm();
            psi[site].data().normalize();

            Eigen::MatrixXcd R_mat = psi[site].left_canonicalize();
            int D_new = static_cast<int>(R_mat.rows());

            // Restore nmt to the R factor (bond/centre tensor).
            R_mat *= nmt;

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

    // After L2R (centre at L-1): normalise centre for imaginary-time.
    // Factor of 2: lnZ = log(Tr ρ) = log ‖ψ‖² = 2 log ‖ψ‖ (purification).
    if (is_imag_time) {
        double d = psi[L - 1].data().norm();
        if (d > 0.0) {
            psi[L - 1].data().normalize();
            lnZ_accum += 2.0 * std::log(d);
        }
    }

    // ── R2L half-sweep ────────────────────────────────────────────────────────
    for (int site = L - 1; site >= 0; --site) {
        auto H_eff = proj1(env, site, E0);

        psi[site].data() = arnoldi_expm_vec(
            [&H_eff](const DenseTensor& v) { return apply(H_eff, v); },
            psi[site].data(), alpha_fwd, acfg);

        // Energy correction.
        scale(psi[site].data(), std::exp(alpha_fwd * E0));

        if (site > 0) {
            // Normalise before RQ.
            double nmt = psi[site].data().norm();
            psi[site].data().normalize();

            Eigen::MatrixXcd L_mat = psi[site].right_canonicalize();
            int D_new = static_cast<int>(L_mat.cols());

            // Restore nmt to the L factor.
            L_mat *= nmt;

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

    // After R2L (centre at 0): normalise for imaginary-time.
    if (is_imag_time) {
        double d = psi[0].data().norm();
        if (d > 0.0) {
            psi[0].data().normalize();
            lnZ_accum += 2.0 * std::log(d);
        }
    }

    psi.set_center(0, 0);

    double nrm = psi.norm();
    TDVPSweepInfo info;
    info.norm       = nrm;
    info.norm_error = std::abs(nrm - 1.0);
    info.E          = E0;
    info.lnZ        = lnZ_accum;
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

    bool is_imag_time = (tau.real() == 0.0);

    // ── Compute E₀ ────────────────────────────────────────────────────────────
    double E0;
    {
        auto H0 = proj1(env, 0, 0.0);
        E0 = inner(psi[0].data(), apply(H0, psi[0].data())).real();
    }

    double lnZ_accum  = 0.0;
    double total_trunc = 0.0;

    // ── L2R sweep ─────────────────────────────────────────────────────────────
    for (int site = 0; site < L - 1; ++site) {
        DenseTensor psi_ab = contract(psi[site].data(), psi[site + 1].data(),
                                      {{2, 0}});
        auto H_eff2 = proj2(env, site, site + 1, E0);

        psi_ab = arnoldi_expm_vec(
            [&H_eff2](const DenseTensor& v) { return apply2(H_eff2, v); },
            psi_ab, alpha_fwd, acfg);

        // Energy correction for the two-site forward step.
        scale(psi_ab, std::exp(alpha_fwd * E0));

        // Normalise psi_ab before SVD; restore nmt to S·Vt (centre-right).
        double nmt = psi_ab.norm();
        psi_ab.normalize();

        SVDResult s   = svd(psi_ab, 2, cfg.trunc);
        total_trunc  += s.truncation_err;
        int D_new     = s.bond_dim;
        int d_r       = psi[site + 1].phys_dim();
        int D_r       = psi[site + 1].right_dim();

        psi[site].data() = s.U;

        env.push_right(site);
        psi.set_center(site + 1, site + 1);

        // S·Vt scaled by nmt carries the orthogonality centre.
        Eigen::MatrixXcd Vt_mat  = s.Vt.matricize({0}, {1, 2});
        Eigen::MatrixXcd SVt_mat = (s.S.asDiagonal() * Vt_mat) * nmt;

        psi[site + 1] = MPSTensor<DenseBackend>::from_right_matrix(
            SVt_mat,
            TrivialSpace(D_new),
            TrivialSpace(d_r),
            TrivialSpace(D_r));

        if (site < L - 2) {
            // Backward single-site evolve on psi[site+1] with energy shift.
            auto H_eff1 = proj1(env, site + 1, E0);
            psi[site + 1].data() = arnoldi_expm_vec(
                [&H_eff1](const DenseTensor& v) { return apply(H_eff1, v); },
                psi[site + 1].data(), alpha_bwd, acfg);
            scale(psi[site + 1].data(), std::exp(alpha_bwd * E0));
        }
        // At last bond (site == L-2): no backward single-site step needed.
    }

    // After L2R (centre at L-1): imaginary-time normalisation.
    if (is_imag_time) {
        double d = psi[L - 1].data().norm();
        if (d > 0.0) {
            psi[L - 1].data().normalize();
            lnZ_accum += 2.0 * std::log(d);
        }
    }

    // ── R2L sweep ─────────────────────────────────────────────────────────────
    for (int site = L - 2; site >= 0; --site) {
        DenseTensor psi_ab = contract(psi[site].data(), psi[site + 1].data(),
                                      {{2, 0}});
        auto H_eff2 = proj2(env, site, site + 1, E0);

        psi_ab = arnoldi_expm_vec(
            [&H_eff2](const DenseTensor& v) { return apply2(H_eff2, v); },
            psi_ab, alpha_fwd, acfg);

        // Energy correction.
        scale(psi_ab, std::exp(alpha_fwd * E0));

        // Normalise before SVD; restore nmt to U·S (centre-left).
        double nmt = psi_ab.norm();
        psi_ab.normalize();

        SVDResult s   = svd(psi_ab, 2, cfg.trunc);
        total_trunc  += s.truncation_err;
        int D_new     = s.bond_dim;
        int d_l       = psi[site].phys_dim();
        int D_l       = psi[site].left_dim();

        psi[site + 1].data() = s.Vt;

        env.push_left(site + 1);
        psi.set_center(site, site);

        // U·S scaled by nmt.
        Eigen::MatrixXcd U_mat  = s.U.matricize({0, 1}, {2});
        Eigen::MatrixXcd US_mat = (U_mat * s.S.asDiagonal()) * nmt;

        psi[site] = MPSTensor<DenseBackend>::from_left_matrix(
            US_mat,
            TrivialSpace(D_l),
            TrivialSpace(d_l),
            TrivialSpace(D_new));

        if (site > 0) {
            // Backward single-site evolve on psi[site] with energy shift.
            auto H_eff1 = proj1(env, site, E0);
            psi[site].data() = arnoldi_expm_vec(
                [&H_eff1](const DenseTensor& v) { return apply(H_eff1, v); },
                psi[site].data(), alpha_bwd, acfg);
            scale(psi[site].data(), std::exp(alpha_bwd * E0));
        }
        // At first site (site == 0): no backward single-site step needed.
    }

    // After R2L (centre at 0): imaginary-time normalisation.
    if (is_imag_time) {
        double d = psi[0].data().norm();
        if (d > 0.0) {
            psi[0].data().normalize();
            lnZ_accum += 2.0 * std::log(d);
        }
    }

    psi.set_center(0, 0);

    double nrm = psi.norm();
    TDVPSweepInfo info;
    info.norm           = nrm;
    info.norm_error     = std::abs(nrm - 1.0);
    info.truncation_err = total_trunc;
    info.E              = E0;
    info.lnZ            = lnZ_accum;
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
