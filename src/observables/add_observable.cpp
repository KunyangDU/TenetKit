// src/observables/add_observable.cpp
//
// add_obs / add_obs2: insert single-site or two-site observables into an
// ObservableTree.
//
// Design: the tree only stores nodes where a non-trivial operator acts.
// cal_obs handles intermediate identity contractions automatically, so there
// is no need to insert explicit identity nodes for gap sites.

#include "tenet/observables/add_observable.hpp"
#include "tenet/observables/obs_node.hpp"

#include <limits>
#include <memory>
#include <string>

namespace tenet {

// ── add_obs ──────────────────────────────────────────────────────────────────
// Adds a single-site observable at op->site().
// Creates one leaf node directly under the tree root.

template<>
void add_obs(ObservableTree<DenseBackend>& tree,
             std::unique_ptr<AbstractLocalOperator<DenseBackend>> op,
             int /*site*/)
{
    std::string op_name = op->name();
    int         site    = op->site();

    auto node    = std::make_unique<ObservableTreeNode<DenseBackend>>();
    node->op     = std::move(op);
    node->parent = tree.root.get();
    node->leaf   = ObservableLeaf{
        {site},
        {std::move(op_name)},
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0)
    };
    tree.root->children.push_back(std::move(node));
}

// ── add_obs2 ─────────────────────────────────────────────────────────────────
// Adds a two-site observable: op1 at site1, op2 at site2 (site1 < site2).
// Creates the path root → node(op1) → leaf(op2).

template<>
void add_obs2(ObservableTree<DenseBackend>& tree,
              std::unique_ptr<AbstractLocalOperator<DenseBackend>> op1, int /*site1*/,
              std::unique_ptr<AbstractLocalOperator<DenseBackend>> op2, int /*site2*/)
{
    std::string op1_name = op1->name();
    std::string op2_name = op2->name();
    int         site1    = op1->site();
    int         site2    = op2->site();

    // Intermediate node: op1 at site1 (env cache built during cal_obs)
    auto node1    = std::make_unique<ObservableTreeNode<DenseBackend>>();
    node1->op     = std::move(op1);
    node1->parent = tree.root.get();

    // Leaf node: op2 at site2
    auto node2    = std::make_unique<ObservableTreeNode<DenseBackend>>();
    node2->op     = std::move(op2);
    node2->parent = node1.get();
    node2->leaf   = ObservableLeaf{
        {site1, site2},
        {std::move(op1_name), std::move(op2_name)},
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 0.0)
    };

    node1->children.push_back(std::move(node2));
    tree.root->children.push_back(std::move(node1));
}

} // namespace tenet
