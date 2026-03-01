// src/hamiltonian/projective_ham.cpp
//
// SparseProjectiveHamiltonian: constructor and factory functions proj0/proj1/proj2.

#include "tenet/hamiltonian/projective_ham.hpp"

#include <cassert>
#include <memory>
#include <optional>
#include <vector>
#include <utility>

namespace tenet {

// ── Constructor ───────────────────────────────────────────────────────────────

template<>
SparseProjectiveHamiltonian<DenseBackend>::SparseProjectiveHamiltonian(
    SparseLeftEnvTensor<DenseBackend>  envL,
    SparseRightEnvTensor<DenseBackend> envR,
    std::optional<SparseMPO<DenseBackend>*> H,
    std::vector<std::pair<int,int>> valid_inds,
    double E0,
    int site1,
    int site2)
    : envL_(std::move(envL))
    , envR_(std::move(envR))
    , H_(H)
    , valid_inds_(std::move(valid_inds))
    , E0_(E0)
    , n_sites_(H.has_value() ? (site2 >= 0 ? 2 : 1) : 0)
    , site1_(site1)
    , site2_(site2)
{}

// ── proj0 ─────────────────────────────────────────────────────────────────────

template<>
SparseProjectiveHamiltonian<DenseBackend>
proj0(SparseLeftEnvTensor<DenseBackend>  envL,
      SparseRightEnvTensor<DenseBackend> envR,
      double E0)
{
    return SparseProjectiveHamiltonian<DenseBackend>(
        std::move(envL), std::move(envR),
        std::nullopt, {}, E0, -1, -1);
}

// ── proj1 ─────────────────────────────────────────────────────────────────────

template<>
SparseProjectiveHamiltonian<DenseBackend>
proj1(const Environment<DenseBackend>& env, int site, double E0)
{
    auto& menv = const_cast<Environment<DenseBackend>&>(env);
    const auto& L  = menv.left_env(site);
    const auto& R  = menv.right_env(site + 1);
    const auto& Hs = menv.H()[site];

    int D_in  = Hs.d_in();
    int D_out = Hs.d_out();

    std::vector<std::pair<int,int>> vinds;
    for (int i = 0; i < D_in; ++i) {
        if (!L.has(i)) continue;
        for (int j = 0; j < D_out; ++j) {
            if (!R.has(j)) continue;
            if (Hs.has(i, j)) vinds.emplace_back(i, j);
        }
    }

    SparseLeftEnvTensor<DenseBackend>  envL_copy(D_in);
    SparseRightEnvTensor<DenseBackend> envR_copy(D_out);

    for (int i = 0; i < D_in; ++i)
        if (L.has(i))
            envL_copy.set(i, std::make_unique<LeftEnvTensor<DenseBackend>>(L[i]->data()));
    for (int j = 0; j < D_out; ++j)
        if (R.has(j))
            envR_copy.set(j, std::make_unique<RightEnvTensor<DenseBackend>>(R[j]->data()));

    SparseMPO<DenseBackend>* H_ptr = &menv.H();
    return SparseProjectiveHamiltonian<DenseBackend>(
        std::move(envL_copy), std::move(envR_copy),
        std::make_optional(H_ptr), std::move(vinds),
        E0, site, -1);
}

// ── proj2 ─────────────────────────────────────────────────────────────────────

template<>
SparseProjectiveHamiltonian<DenseBackend>
proj2(const Environment<DenseBackend>& env, int site1, int site2, double E0)
{
    assert(site2 == site1 + 1);
    auto& menv = const_cast<Environment<DenseBackend>&>(env);

    const auto& L   = menv.left_env(site1);
    const auto& R   = menv.right_env(site2 + 1);
    const auto& Hs1 = menv.H()[site1];
    const auto& Hs2 = menv.H()[site2];

    int D_in1  = Hs1.d_in();
    int D_mid  = Hs1.d_out();
    int D_out2 = Hs2.d_out();

    // valid_inds: (i, j) outer bonds; k (middle) is summed over in apply2
    std::vector<std::pair<int,int>> vinds;
    for (int i = 0; i < D_in1; ++i) {
        if (!L.has(i)) continue;
        for (int j = 0; j < D_out2; ++j) {
            if (!R.has(j)) continue;
            bool ok = false;
            for (int k = 0; k < D_mid && !ok; ++k)
                if (Hs1.has(i, k) && Hs2.has(k, j)) ok = true;
            if (ok) vinds.emplace_back(i, j);
        }
    }

    SparseLeftEnvTensor<DenseBackend>  envL_copy(D_in1);
    SparseRightEnvTensor<DenseBackend> envR_copy(D_out2);

    for (int i = 0; i < D_in1; ++i)
        if (L.has(i))
            envL_copy.set(i, std::make_unique<LeftEnvTensor<DenseBackend>>(L[i]->data()));
    for (int j = 0; j < D_out2; ++j)
        if (R.has(j))
            envR_copy.set(j, std::make_unique<RightEnvTensor<DenseBackend>>(R[j]->data()));

    SparseMPO<DenseBackend>* H_ptr = &menv.H();
    return SparseProjectiveHamiltonian<DenseBackend>(
        std::move(envL_copy), std::move(envR_copy),
        std::make_optional(H_ptr), std::move(vinds),
        E0, site1, site2);
}

} // namespace tenet
