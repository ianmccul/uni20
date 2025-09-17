#include "test_contract_common.hpp"
#include "gtest/gtest.h"
#include <iostream>
using namespace uni20::test;
using namespace uni20;

template <typename T> class TensorContractTypedTest_matmul_colunm_major12345678 : public ::testing::Test {};
using ScalarTypes = ::testing::Types<float, double>; //, std::complex<float>, std::complex<double>>;  // no int!
TYPED_TEST_SUITE(TensorContractTypedTest_matmul_colunm_major12345678, ScalarTypes);

TYPED_TEST(TensorContractTypedTest_matmul_colunm_major12345678, Rank2)
{
  test_colunm_major_matmul_12345678_correctness<TypeParam, blas_tag>();
  std::cout << "XXXXX" << std::endl;
}
