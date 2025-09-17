#include "gtest/gtest.h"
#include "test_contract_common.hpp"
#include <iostream>
using namespace uni20::test;
using namespace uni20;

template<typename T>
class TensorContractTypedTest_BLAS : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;  // no int!
TYPED_TEST_SUITE(TensorContractTypedTest_BLAS, ScalarTypes);

TYPED_TEST(TensorContractTypedTest_BLAS, Rank2) {
  test_rank2_contraction_correctness<TypeParam, blas_tag>();
  std::cout<<"XXXXX"<<std::endl;
}
