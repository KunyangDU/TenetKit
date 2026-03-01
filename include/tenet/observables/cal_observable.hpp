#pragma once
// include/tenet/observables/cal_observable.hpp

#include "tenet/observables/obs_node.hpp"
#include "tenet/environment/environment.hpp"

namespace tenet {

// Traverse the observable tree, computing all leaf expectation values.
// Results are written into ObservableLeaf::value of each leaf node.
template<TensorBackend B = DenseBackend>
void cal_obs(ObservableTree<B>& tree, const Environment<B>& env);

} // namespace tenet
