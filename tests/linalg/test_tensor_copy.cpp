#include <uni20/core/types.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/tensor/copy.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <type_traits>

namespace
{
using complex_type = uni20::complex<double>;

void fill_complex_matrix(uni20::DenseMatrix<complex_type, uni20::RowMajor>& matrix)
{
  matrix[0, 0] = complex_type{1.0, 2.0};
  matrix[0, 1] = complex_type{3.0, -4.0};
  matrix[1, 0] = complex_type{-5.0, 6.0};
  matrix[1, 1] = complex_type{7.0, 8.0};
}
} // namespace

TEST(TensorCopyTest, CopyResizesOutputAndObservesLazyConjugation)
{
  uni20::DenseMatrix<complex_type, uni20::RowMajor> input(2, 2);
  uni20::DenseMatrix<complex_type> output;
  fill_complex_matrix(input);

  auto conjugated = uni20::conj(input);
  uni20::copy(output, conjugated);

  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 2);
  EXPECT_EQ((output[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((output[0, 1]), (complex_type{3.0, 4.0}));
  EXPECT_EQ((output[1, 0]), (complex_type{-5.0, -6.0}));
  EXPECT_EQ((output[1, 1]), (complex_type{7.0, -8.0}));
}

TEST(TensorCopyTest, MakeTensorInfersScalarExtentsAndSourceLayout)
{
  uni20::DenseMatrix<complex_type, uni20::RowMajor> input(2, 2);
  fill_complex_matrix(input);

  auto result = uni20::make_tensor(uni20::conj(input));

  using expected_type = uni20::Tensor<complex_type, 2, uni20::VectorStorage, uni20::RowMajor>;
  static_assert(std::same_as<decltype(result), expected_type>);
  EXPECT_EQ((result[0, 1]), (complex_type{3.0, 4.0}));
  EXPECT_EQ(result.mapping().stride(0), 2);
  EXPECT_EQ(result.mapping().stride(1), 1);
}

TEST(TensorCopyTest, MakeTensorAcceptsExplicitLayoutAndBareMdspanSelector)
{
  uni20::DenseMatrix<complex_type, uni20::RowMajor> input(2, 2);
  fill_complex_matrix(input);
  auto conjugated = uni20::conj(input.mdspan());

  auto result = uni20::make_tensor<uni20::ColumnMajor>(uni20::linalg::CpuReferenceBackend{}, conjugated);

  static_assert(std::same_as<typename decltype(result)::layout_type, uni20::ColumnMajor>);
  EXPECT_EQ((result[1, 0]), (complex_type{-5.0, -6.0}));
  EXPECT_EQ(result.mapping().stride(0), 1);
  EXPECT_EQ(result.mapping().stride(1), 2);
}

TEST(TensorCopyTest, MakeTensorMaterializesStaticExtentsAsGeneralPurposeTensor)
{
  using fixed_extents = stdex::extents<uni20::index_type, 2, 3>;
  using fixed_tensor = uni20::BasicTensor<double, fixed_extents, uni20::VectorStorage, uni20::RowMajor>;
  fixed_tensor input(fixed_extents{});
  input[0, 0] = 1.0;
  input[1, 2] = 6.0;

  auto result = uni20::make_tensor(input);

  using expected_type = uni20::Tensor<double, 2, uni20::VectorStorage, uni20::RowMajor>;
  static_assert(std::same_as<decltype(result), expected_type>);
  EXPECT_EQ(result.rows(), 2);
  EXPECT_EQ(result.cols(), 3);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((result[1, 2]), 6.0);
}

TEST(TensorCopyTest, MakeTensorUsesDefaultLayoutForNoncanonicalSourceType)
{
  using input_type = uni20::StridedTensor<double, 2>;
  input_type input(input_type::extents_type{2, 2}, std::array<uni20::index_type, 2>{2, 1});
  input[0, 0] = 1.0;
  input[0, 1] = 2.0;
  input[1, 0] = 3.0;
  input[1, 1] = 4.0;

  auto result = uni20::make_tensor(input);

  static_assert(std::same_as<typename decltype(result)::layout_type, uni20::ColumnMajor>);
  EXPECT_EQ(result.mapping().stride(0), 1);
  EXPECT_EQ(result.mapping().stride(1), 2);
  EXPECT_DOUBLE_EQ((result[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 3.0);
}

TEST(TensorCopyTest, CpuReferenceCopySupportsScalarConversion)
{
  uni20::Tensor<float, 1> input(3);
  uni20::Tensor<double, 1> output;
  input[0] = 1.25F;
  input[1] = -2.5F;
  input[2] = 4.0F;

  uni20::copy(output, input);

  EXPECT_EQ(output.extent(0), 3);
  EXPECT_DOUBLE_EQ(output[0], 1.25);
  EXPECT_DOUBLE_EQ(output[1], -2.5);
  EXPECT_DOUBLE_EQ(output[2], 4.0);
}
