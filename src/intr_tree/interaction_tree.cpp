// src/intr_tree/interaction_tree.cpp
//
// InteractionTree manipulation (add_intr1, add_intr2, add_intr3) and
// AutomataSparseMPO compilation.  Ported from Julia's Automata.jl /
// addIntr{1,2}.jl using the same layer-by-layer traversal.

#include "tenet/intr_tree/interaction_node.hpp"
#include "tenet/intr_tree/add_interaction.hpp"
#include "tenet/mps/mpo.hpp"
#include "tenet/mps/mpo_tensor.hpp"

#include <array>
#include <cassert>
#include <map>
#include <utility>

namespace tenet {

// ── Helpers ──────────────────────────────────────────────────────────────────

// Check whether two local operators are "equal" (same site, same matrix).
template<TensorBackend B>
static bool ops_match(const AbstractLocalOperator<B>& a,
                      const AbstractLocalOperator<B>& b)
{
    if (a.site() != b.site()) return false;
    if (a.is_identity() && b.is_identity()) return true;
    if (a.is_identity() || b.is_identity()) return false;
    return a.matrix().isApprox(b.matrix(), 1e-14);
}

// Return the child of `node` whose op matches `op`, or create a new one.
template<TensorBackend B>
static InteractionTreeNode<B>* find_or_create_child(
    InteractionTreeNode<B>*                   node,
    std::unique_ptr<AbstractLocalOperator<B>> op)
{
    for (auto& child : node->children)
        if (ops_match(*child->op, *op))
            return child.get();

    auto new_node    = std::make_unique<InteractionTreeNode<B>>();
    new_node->op     = std::move(op);
    new_node->parent = node;
    node->children.push_back(std::move(new_node));
    return node->children.back().get();
}

// Depth-first search: return the op of the first leaf found.
template<TensorBackend B>
static const AbstractLocalOperator<B>* find_any_leaf_op(
    const InteractionTreeNode<B>* node)
{
    for (auto& child : node->children) {
        if (child->children.empty()) return child->op.get();
        auto* r = find_any_leaf_op(child.get());
        if (r) return r;
    }
    return nullptr;
}

// ── add_intr1 ─────────────────────────────────────────────────────────────────

template<TensorBackend B>
void add_intr1(InteractionTree<B>& tree, Op<B> op, int site,
               std::complex<double> strength)
{
    assert(site >= 0 && site < tree.L);
    int d = static_cast<int>(op->matrix().rows());

    auto* current = tree.root.get();

    // Walk from root to site-1, finding/creating Identity nodes.
    for (int s = 0; s < site; ++s) {
        auto id_op = std::make_unique<IdentityOperator<B>>(s, d);
        current    = find_or_create_child<B>(current, std::move(id_op));
    }

    // Always append a new leaf (no sharing for leaf nodes).
    Eigen::MatrixXcd mat = op->matrix() * strength;
    auto leaf    = std::make_unique<InteractionTreeNode<B>>();
    leaf->op     = std::make_unique<LocalOperator<B>>(std::move(mat), op->name(), site);
    leaf->parent = current;
    current->children.push_back(std::move(leaf));
}

// ── add_intr2 ─────────────────────────────────────────────────────────────────

template<TensorBackend B>
void add_intr2(InteractionTree<B>& tree,
               Op<B> op1, int site1,
               Op<B> op2, int site2,
               std::complex<double> strength,
               bool /*fermionic*/)
{
    assert(site1 != site2);
    assert(site1 >= 0 && site2 >= 0 && site1 < tree.L && site2 < tree.L);

    // Sort so site1 < site2.
    if (site1 > site2) {
        std::swap(op1, op2);
        std::swap(site1, site2);
    }

    int d      = static_cast<int>(op1->matrix().rows());
    auto* current = tree.root.get();

    // Walk from root up to (but not including) site2.
    for (int s = 0; s < site2; ++s) {
        std::unique_ptr<AbstractLocalOperator<B>> temp;
        if (s < site1)
            temp = std::make_unique<IdentityOperator<B>>(s, d);
        else if (s == site1)
            temp = std::make_unique<LocalOperator<B>>(op1->matrix(), op1->name(), s);
        else
            temp = std::make_unique<IdentityOperator<B>>(s, d);

        current = find_or_create_child<B>(current, std::move(temp));
    }

    // Leaf at site2 with strength absorbed into op2.
    Eigen::MatrixXcd mat2 = op2->matrix() * strength;
    auto leaf    = std::make_unique<InteractionTreeNode<B>>();
    leaf->op     = std::make_unique<LocalOperator<B>>(std::move(mat2), op2->name(), site2);
    leaf->parent = current;
    current->children.push_back(std::move(leaf));
}

// ── add_intr3 ─────────────────────────────────────────────────────────────────

template<TensorBackend B>
void add_intr3(InteractionTree<B>& tree,
               Op<B> op1, int s1,
               Op<B> op2, int s2,
               Op<B> op3, int s3,
               std::complex<double> strength)
{
    // Collect and sort by site index.
    std::array<int, 3> sites = {s1, s2, s3};
    std::array<std::unique_ptr<AbstractLocalOperator<B>>, 3> ops;
    ops[0] = std::move(op1); ops[1] = std::move(op2); ops[2] = std::move(op3);

    // Bubble sort
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2 - i; ++j)
            if (sites[j] > sites[j + 1]) {
                std::swap(sites[j], sites[j + 1]);
                std::swap(ops[j],   ops[j + 1]);
            }

    int d      = static_cast<int>(ops[0]->matrix().rows());
    auto* current = tree.root.get();
    int op_idx = 0;

    // Walk up to (but not including) sites[2].
    for (int s = 0; s < sites[2]; ++s) {
        std::unique_ptr<AbstractLocalOperator<B>> temp;
        if (op_idx < 2 && s == sites[op_idx]) {
            temp = std::make_unique<LocalOperator<B>>(
                ops[op_idx]->matrix(), ops[op_idx]->name(), s);
            ++op_idx;
        } else {
            temp = std::make_unique<IdentityOperator<B>>(s, d);
        }
        current = find_or_create_child<B>(current, std::move(temp));
    }

    // Leaf at sites[2] with strength.
    Eigen::MatrixXcd mat = ops[2]->matrix() * strength;
    auto leaf    = std::make_unique<InteractionTreeNode<B>>();
    leaf->op     = std::make_unique<LocalOperator<B>>(std::move(mat), ops[2]->name(), sites[2]);
    leaf->parent = current;
    current->children.push_back(std::move(leaf));
}

// ── compile ───────────────────────────────────────────────────────────────────
//
// Implements Julia's AutomataSparseMPO: layer-by-layer traversal of the tree.
//
// State encoding (0-indexed):
//   "inverse_root" (last_inv / next_inv): 1 if an "end" state exists at col/row 0.
//   Internal node at push position i (0-indexed) → col = i + next_inv.
//   Leaf node → col = 0 (the end state; only valid when next_inv == 1).
//   End-state propagation: m[0,0] = Identity when last_inv == 1 and m[0,0] is empty.

template<TensorBackend B>
SparseMPO<B> compile(const InteractionTree<B>& tree)
{
    int L = tree.L;
    SparseMPO<B> mpo(L);

    if (tree.root->children.empty()) return mpo;

    // Physical dimension from first leaf.
    const auto* any_op = find_any_leaf_op(tree.root.get());
    assert(any_op != nullptr);
    int d = static_cast<int>(any_op->matrix().rows());

    std::vector<InteractionTreeNode<B>*> last_roots;
    last_roots.push_back(tree.root.get());
    int last_inv = 0, next_inv = 0;

    for (int site = 0; site < L; ++site) {
        std::vector<InteractionTreeNode<B>*> next_roots;

        // (row,col) → accumulated matrix for leaves (may accumulate if same row)
        std::map<std::pair<int,int>, Eigen::MatrixXcd> leaf_mats;
        // (row,col) → raw op pointer for internal nodes (always unique (row,col))
        std::map<std::pair<int,int>, const AbstractLocalOperator<B>*> internal_ops;

        // Determine next_inv: 1 if any child of current roots is a leaf.
        if (next_inv == 0) {
            for (auto* r : last_roots) {
                for (auto& c : r->children)
                    if (c->children.empty()) { next_inv = 1; break; }
                if (next_inv) break;
            }
        }

        // Process children of each active root.
        for (int lastind = 0; lastind < static_cast<int>(last_roots.size()); ++lastind) {
            int row   = lastind + last_inv;
            auto* lr  = last_roots[lastind];

            for (auto& child_ptr : lr->children) {
                auto* child = child_ptr.get();

                if (child->children.empty()) {
                    // Leaf → end-state column (0 when next_inv == 1).
                    auto& mat = leaf_mats[{row, 0}];
                    if (mat.size() == 0)
                        mat = child->op->matrix();
                    else
                        mat += child->op->matrix();
                } else {
                    // Internal node: gets a unique new column.
                    next_roots.push_back(child);
                    int col = static_cast<int>(next_roots.size()) - 1 + next_inv;
                    internal_ops[{row, col}] = child->op.get();
                }
            }
        }

        int dim_rows = static_cast<int>(last_roots.size()) + last_inv;
        int dim_cols = static_cast<int>(next_roots.size()) + next_inv;

        // Replace placeholder (1×1) tensor with properly sized one.
        mpo[site] = SparseMPOTensor<B>(dim_rows, dim_cols);

        // Fill leaf entries (with identity detection).
        Eigen::MatrixXcd Id_mat = Eigen::MatrixXcd::Identity(d, d);
        for (auto& [key, mat] : leaf_mats) {
            auto [row, col] = key;
            if (mat.isApprox(Id_mat, 1e-14))
                mpo[site].set(row, col,
                    std::make_unique<IdentityOperator<B>>(site, d));
            else
                mpo[site].set(row, col,
                    std::make_unique<LocalOperator<B>>(mat, "H", site));
        }

        // Fill internal node entries.
        for (auto& [key, op_ptr] : internal_ops) {
            auto [row, col] = key;
            mpo[site].set(row, col, op_ptr->clone());
        }

        // End-state propagation: m[0,0] = Identity if end state exists in rows.
        if (last_inv == 1 && !mpo[site].has(0, 0))
            mpo[site].set(0, 0, std::make_unique<IdentityOperator<B>>(site, d));

        last_roots = std::move(next_roots);
        last_inv   = next_inv;
    }

    return mpo;
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template void add_intr1<DenseBackend>(
    InteractionTree<DenseBackend>&, Op<DenseBackend>, int, std::complex<double>);

template void add_intr2<DenseBackend>(
    InteractionTree<DenseBackend>&,
    Op<DenseBackend>, int, Op<DenseBackend>, int,
    std::complex<double>, bool);

template void add_intr3<DenseBackend>(
    InteractionTree<DenseBackend>&,
    Op<DenseBackend>, int, Op<DenseBackend>, int, Op<DenseBackend>, int,
    std::complex<double>);

template SparseMPO<DenseBackend> compile<DenseBackend>(
    const InteractionTree<DenseBackend>&);

} // namespace tenet
