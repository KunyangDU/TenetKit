// src/algebra/mpo_trace.cpp
//
// mpo_trace: Tr(ρ) = Σ_σ ρ[σ,σ] contracted right-to-left.
//
// Algorithm:
//   R = [[1]]  (1×1 boundary)
//   for s = L-1 downto 0:
//       T[αL, αR] = Σ_σ ρ[s][αL, σ, σ, αR]   (trace over physical legs)
//       R = T * R                               (matrix multiply)
//   return R[0,0]

#include "tenet/algebra/mpo_trace.hpp"
#include "tenet/core/tensor_ops.hpp"
#include "tenet/core/space.hpp"

#include <cassert>

namespace tenet {

template<>
std::complex<double> mpo_trace(const DenseMPO<DenseBackend>& rho)
{
    const int L = rho.length();
    assert(L > 0);

    // R: right boundary, shape (D_r=1, 1), value [[1]]
    Eigen::MatrixXcd R = Eigen::MatrixXcd::Identity(1, 1);

    for (int s = L - 1; s >= 0; --s) {
        const DenseTensor& A = rho[s].data();  // (D_l, d_bra, d_ket, D_r)
        int Dl = A.dim(0);
        int d  = A.dim(1);
        int Dr = A.dim(3);

        // T[αL, αR] = Σ_σ A[αL, σ, σ, αR]
        Eigen::MatrixXcd T(Dl, Dr);
        T.setZero();
        for (int al = 0; al < Dl; ++al)
            for (int ar = 0; ar < Dr; ++ar)
                for (int sigma = 0; sigma < d; ++sigma)
                    T(al, ar) += A({al, sigma, sigma, ar});

        // R ← T * R  (Dl×Dr * Dr×1 = Dl×1)
        R = T * R;
    }

    return R(0, 0);
}

} // namespace tenet
