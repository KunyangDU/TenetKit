// tests/test_dense_tensor.cpp
// Stub test file – replace with real test cases as Phase 1 is implemented.
// See docs/C++重构设计方案.md §17 for the required test cases.

#include <gtest/gtest.h>
#include "tenet/core/dense_tensor.hpp"
#include "tenet/core/tensor_ops.hpp"

namespace tenet::test {

TEST(DenseTensor, DefaultConstruct) {
    DenseTensor t;
    EXPECT_EQ(t.rank(), 0);
}

// TODO: ContractCorrectness, SVD_Roundtrip, Permute_Consistency

} // namespace tenet::test
