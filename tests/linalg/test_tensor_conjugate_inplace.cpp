#include <uni20/core/types.hpp>
#include <uni20/tensor/conjugate_inplace.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>

namespace
{

TEST(TensorConjugateInplaceTest, ConjugatesComplexStridedTensorThroughDefaultDispatch)
{
  using scalar_type = uni20::complex<double>;
  using matrix_type = uni20::Tensor<scalar_type, 2>;
  matrix_type matrix(matrix_type::extents_type{2, 2}, std::array<uni20::index_type, 2>{3, 1});
  matrix[0, 0] = scalar_type{1.0, 2.0};
  matrix[0, 1] = scalar_type{3.0, -4.0};
  matrix[1, 0] = scalar_type{-5.0, 6.0};
  matrix[1, 1] = scalar_type{7.0, 8.0};

  uni20::conjugate_inplace(matrix);

  EXPECT_EQ((matrix[0, 0]), (scalar_type{1.0, -2.0}));
  EXPECT_EQ((matrix[0, 1]), (scalar_type{3.0, 4.0}));
  EXPECT_EQ((matrix[1, 0]), (scalar_type{-5.0, -6.0}));
  EXPECT_EQ((matrix[1, 1]), (scalar_type{7.0, -8.0}));
}

TEST(TensorConjugateInplaceTest, RealMdspanIsAnInPlaceIdentityOperation)
{
  uni20::DenseMatrix<double> matrix(2, 2);
  matrix[0, 0] = 1.0;
  matrix[0, 1] = 2.0;
  matrix[1, 0] = 3.0;
  matrix[1, 1] = 4.0;
  auto span = matrix.mdspan();

  uni20::conjugate_inplace(uni20::linalg::CpuReferenceBackend{}, span);

  EXPECT_DOUBLE_EQ((matrix[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((matrix[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((matrix[1, 0]), 3.0);
  EXPECT_DOUBLE_EQ((matrix[1, 1]), 4.0);
}

} // namespace
