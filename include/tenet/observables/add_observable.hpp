#pragma once
// include/tenet/observables/add_observable.hpp

#include "tenet/observables/obs_node.hpp"

namespace tenet {

template<TensorBackend B = DenseBackend>
void add_obs(ObservableTree<B>& tree,
             std::unique_ptr<AbstractLocalOperator<B>> op,
             int site);

template<TensorBackend B = DenseBackend>
void add_obs2(ObservableTree<B>& tree,
              std::unique_ptr<AbstractLocalOperator<B>> op1, int site1,
              std::unique_ptr<AbstractLocalOperator<B>> op2, int site2);

} // namespace tenet
