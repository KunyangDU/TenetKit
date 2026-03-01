#pragma once
// include/tenet/observables/obs_node.hpp
//
// ObservableTreeNode and ObservableTreeLeave: tree structure used to compute
// expectation values ⟨ψ|O|ψ⟩ efficiently during / after a sweep.
// See docs/C++重构设计方案.md §11.

#include "tenet/intr_tree/local_operator.hpp"
#include "tenet/environment/env_tensor.hpp"

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tenet {

// A single measured expectation value.
struct ObservableLeaf {
    std::vector<int>         sites;
    std::vector<std::string> op_names;
    std::complex<double>     value{std::numeric_limits<double>::quiet_NaN(), 0.0};
};

template<TensorBackend B = DenseBackend>
struct ObservableTreeNode {
    std::unique_ptr<AbstractLocalOperator<B>>      op;       // nullptr for root

    // Cached partial environment (nullptr / env tensor / disk path)
    using EnvCache = std::variant<std::monostate,
                                  LeftEnvTensor<B>,
                                  std::string>;
    EnvCache env_cache;

    ObservableTreeNode<B>*                         parent = nullptr;
    std::vector<std::unique_ptr<ObservableTreeNode<B>>> children;
    std::optional<ObservableLeaf>                  leaf;   // non-null for leaf nodes
};

template<TensorBackend B = DenseBackend>
struct ObservableTree {
    std::unique_ptr<ObservableTreeNode<B>> root;
    int L = 0;

    explicit ObservableTree(int L) : root(std::make_unique<ObservableTreeNode<B>>()), L(L) {}
};

} // namespace tenet
