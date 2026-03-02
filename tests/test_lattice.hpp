#pragma once
// tests/test_lattice.hpp
//
// Shared test helper: Heisenberg MPO on a YC cylinder.
//
// Site numbering (column-major, 0-indexed):
//   s = col * Ly + row
//   Example Lx=2, Ly=4:
//     col 0   col 1
//       0       4     <- row 0
//       1       5     <- row 1
//       2       6     <- row 2
//       3       7     <- row 3
// X direction: open BC (bonds between adjacent columns at same row)
// Y direction: periodic BC (bonds within each column, row 0 wraps to row Ly-1)

#include "tenet/local_space/spin.hpp"
#include "tenet/intr_tree/interaction_node.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/intr_tree/local_operator.hpp"
#include "tenet/mps/mpo.hpp"

namespace tenet::test {

// H = J Σ_{<i,j>} (Sz_i Sz_j + ½ Sp_i Sm_j + ½ Sm_i Sp_j)
// Bonds i < j always passed to add_intr2 (Heisenberg is symmetric under swap).
inline SparseMPO<> make_yc_heisenberg_mpo(int Lx, int Ly, double J = 1.0)
{
    using namespace spin::half;
    const int L = Lx * Ly;
    InteractionTree<> tree(L);

    auto add_bond = [&](int i, int j) {
        if (i > j) std::swap(i, j);

        Op<> sz_i = std::make_unique<LocalOperator<>>(Sz(), "Sz", i);
        Op<> sz_j = std::make_unique<LocalOperator<>>(Sz(), "Sz", j);
        add_intr2(tree, std::move(sz_i), i, std::move(sz_j), j, J);

        Op<> sp_i = std::make_unique<LocalOperator<>>(Sp(), "Sp", i);
        Op<> sm_j = std::make_unique<LocalOperator<>>(Sm(), "Sm", j);
        add_intr2(tree, std::move(sp_i), i, std::move(sm_j), j, 0.5 * J);

        Op<> sm_i = std::make_unique<LocalOperator<>>(Sm(), "Sm", i);
        Op<> sp_j = std::make_unique<LocalOperator<>>(Sp(), "Sp", j);
        add_intr2(tree, std::move(sm_i), i, std::move(sp_j), j, 0.5 * J);
    };

    // Y bonds: within each column, periodic
    for (int col = 0; col < Lx; ++col)
        for (int row = 0; row < Ly; ++row)
            add_bond(col * Ly + row, col * Ly + (row + 1) % Ly);

    // X bonds: between adjacent columns, open BC
    for (int col = 0; col < Lx - 1; ++col)
        for (int row = 0; row < Ly; ++row)
            add_bond(col * Ly + row, (col + 1) * Ly + row);

    return compile(tree);
}

} // namespace tenet::test
