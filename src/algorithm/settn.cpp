// src/algorithm/settn.cpp
//
// SETTN main loop: ρ(β) = Σ_{n=0}^{N} (-β)^n/n! · H^n
//
// Convergence criterion: |ΔlnZ / lnZ| < cfg.tol

#include "tenet/algorithm/settn.hpp"
#include "tenet/algebra/mpo_mul.hpp"
#include "tenet/algebra/mpo_axpy.hpp"
#include "tenet/algebra/mpo_trace.hpp"

#include <cassert>
#include <cmath>

namespace tenet {

namespace {

// Infer physical dimension d from the SparseMPO.
// Iterates over all nonzero operators; returns their matrix size.
// Falls back to 2 (spin-1/2) if no operator is found.
static int get_phys_dim(const SparseMPO<DenseBackend>& H)
{
    for (int s = 0; s < H.length(); ++s) {
        int din  = H[s].d_in();
        int dout = H[s].d_out();
        for (int i = 0; i < din; ++i) {
            for (int j = 0; j < dout; ++j) {
                if (H[s].has(i, j)) {
                    int d = static_cast<int>(H[s](i, j)->matrix().rows());
                    if (d > 0) return d;
                }
            }
        }
    }
    return 2;  // default
}

} // anonymous namespace

template<>
SETTNResult settn<DenseBackend>(SparseMPO<DenseBackend>& H,
                                 double                   beta,
                                 const SETTNConfig&        cfg)
{
    const int L = H.length();
    assert(L >= 1);

    const int d = get_phys_dim(H);

    // Initialize ρ = I,  Hn = I
    DenseMPO<DenseBackend> rho = DenseMPO<DenseBackend>::identity(L, d);
    DenseMPO<DenseBackend> Hn  = DenseMPO<DenseBackend>::identity(L, d);

    // Configs for sub-algorithms (inherit truncation from SETTNConfig)
    MulConfig  mul_cfg;
    mul_cfg.trunc = cfg.trunc;
    AxpyConfig axpy_cfg;
    axpy_cfg.trunc = cfg.trunc;

    SETTNResult result;

    // n=0: lnZ = log(d^L) = L * log(d)
    double lnZ_prev = static_cast<double>(L) * std::log(static_cast<double>(d));
    result.lnZ_values.push_back(lnZ_prev);
    result.free_energies.push_back(-lnZ_prev / beta);

    double neg_beta_pow = 1.0;   // (-β)^0 = 1
    double factorial    = 1.0;   // 0! = 1

    for (int n = 1; n <= cfg.max_order; ++n) {
        // Step 1: Hn ← Hn · H
        double mul_err = mpo_mul(Hn, Hn, H, 1.0, mul_cfg);

        // Step 2: coeff = (-β)^n / n!
        neg_beta_pow *= (-beta);
        factorial    *= static_cast<double>(n);
        double coeff  = neg_beta_pow / factorial;

        // Step 3: ρ ← ρ + coeff · Hn
        double axpy_err = mpo_axpy(coeff, Hn, rho, axpy_cfg);

        result.truncation_errs.push_back(mul_err + axpy_err);

        // Step 4: lnZ = log(|Tr(ρ)|)
        std::complex<double> tr = mpo_trace(rho);
        double tr_re = tr.real();
        if (tr_re <= 0.0) tr_re = std::abs(tr_re);
        double lnZ = std::log(tr_re);

        result.lnZ_values.push_back(lnZ);
        result.free_energies.push_back(-lnZ / beta);

        // Step 5: convergence check |ΔlnZ / lnZ| < tol
        double denom = std::abs(lnZ_prev) > 1e-15 ? std::abs(lnZ_prev) : 1.0;
        double delta = std::abs(lnZ - lnZ_prev) / denom;
        lnZ_prev = lnZ;

        // Require n >= 2 before declaring convergence: at n=1 the change may be
        // exactly zero when Tr(H)=0 (e.g. XX chain), which would incorrectly
        // terminate the series before any physical correction is accumulated.
        // Also require delta > 0: for Hamiltonians with odd-moment symmetry
        // (e.g. Tr(H^{2k+1})=0), odd-order steps produce zero delta, which
        // should NOT be treated as convergence — the series is still building up.
        if (n >= 2 && delta > 0.0 && delta < cfg.tol) {
            result.converged_order = n;
            break;
        }
    }

    return result;
}

} // namespace tenet
