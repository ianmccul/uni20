#include "gtest/gtest.h"
#include "test_contract_common.hpp"

using namespace uni20::test;
using namespace uni20;

template<typename T>
class TensorContractTypedTest_CPU : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(TensorContractTypedTest_CPU, ScalarTypes);

TYPED_TEST(TensorContractTypedTest_CPU, Rank2) {
  test_rank2_contraction_correctness<TypeParam, cpu_tag>();
}
