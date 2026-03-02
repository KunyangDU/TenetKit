#pragma once
// include/tenet/algebra/mpo_trace.hpp
//
// mpo_trace: compute Tr(ρ) = Σ_σ ρ[σ_1,σ_1; ...; σ_L,σ_L] for a DenseMPO.

#include "tenet/mps/dense_mpo.hpp"
#include <complex>

namespace tenet {

template<TensorBackend B = DenseBackend>
std::complex<double> mpo_trace(const DenseMPO<B>& rho);

} // namespace tenet
