#pragma once
// tests/free_fermion_ref.hpp
//
// 自由费米子精确参考：用于 SETTN 集成测试。
//
// 数学基础：OBC 下 1D XX 链经 Jordan-Wigner 变换映射到自由费米子
//
//   H_XX = -J/2 · Σ (Sp_i Sm_{i+1} + h.c.)
//        ↕  JW（OBC，无边界相因子）
//   H_JW = -J/2 · Σ (c†_i c_{i+1} + h.c.)   （t = J/2）
//
// 因此：
//   Tr_spin[ e^{-β H_XX} ] = Π_{k=1}^{L} (1 + e^{-β ε_k})
//   lnZ_exact = Σ_k log(1 + e^{-β ε_k})
//   E_exact   = Σ_k ε_k / (1 + e^{β ε_k})
//
// 单粒子能级（OBC，t = J/2）：
//   ε_k = -J · cos(kπ/(L+1)),  k = 1, ..., L
//
// 这些函数纯粹依赖 <cmath> 和 <vector>，不依赖 tenet 库。

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace tenet::test {

// 返回 OBC 1D XX 链的 L 个单粒子能级 ε_k = -J·cos(kπ/(L+1))，k=1,...,L
inline std::vector<double> xx_chain_energies(int L, double J = 1.0) {
    std::vector<double> eps;
    eps.reserve(L);
    for (int k = 1; k <= L; ++k)
        eps.push_back(-J * std::cos(k * M_PI / (L + 1)));
    return eps;
}

// 数值稳定的 log(1 + exp(-x))
static inline double _ff_log1p_exp_neg(double x) {
    if (x >  30.0) return std::exp(-x);
    if (x < -30.0) return -x + std::exp(x);
    return (x >= 0) ? std::log1p(std::exp(-x))
                    : -x + std::log1p(std::exp(x));
}

// 数值稳定的 Fermi-Dirac：f(ε, μ=0) = 1/(e^{βε} + 1)
static inline double _ff_fermi(double eps, double beta) {
    double x = beta * eps;
    if (x >  500.0) return 0.0;
    if (x < -500.0) return 1.0;
    return 1.0 / (std::exp(x) + 1.0);
}

// 巨正则配分函数对数（μ=0）：lnZ = Σ_k log(1 + e^{-β ε_k})
// 这正是 SETTN 计算的 log Tr(e^{-βH}) 在 XX 链上的精确值。
inline double grand_canonical_lnZ(const std::vector<double>& eps, double beta) {
    double lnZ = 0.0;
    for (double e : eps)
        lnZ += _ff_log1p_exp_neg(beta * e);
    return lnZ;
}

// 内能（μ=0）：E = Σ_k ε_k · f_k，f_k = 1/(e^{βε_k}+1)
inline double grand_canonical_energy(const std::vector<double>& eps, double beta) {
    double E = 0.0;
    for (double e : eps)
        E += e * _ff_fermi(e, beta);
    return E;
}

} // namespace tenet::test
