#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/gemv.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <limits>
#include <utility>
#include <vector>

namespace
{
using uni20::linalg::KernelAttempt;

using extents_1d = stdex::dextents<uni20::index_type, 1>;
using extents_2d = stdex::dextents<uni20::index_type, 2>;

template <class Matrix> void fill_matrix(Matrix span, std::initializer_list<typename Matrix::value_type> values)
{
  auto value = values.begin();
  for (uni20::index_type row = 0; row < span.extent(0); ++row)
  {
    for (uni20::index_type col = 0; col < span.extent(1); ++col)
    {
      span[row, col] = *value;
      ++value;
    }
  }
}
} // namespace

TEST(BlasGemvTest, MultipliesWithStridedInputAndOutputVectors)
{
  std::vector<double> matrix_storage(6);
  std::vector<double> input_storage(5, -100.0);
  std::vector<double> output_storage(3, -100.0);

  stdex::mdspan<double, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 3);
  stdex::layout_stride::mapping<extents_1d> input_mapping(extents_1d{3}, std::array<uni20::index_type, 1>{2});
  stdex::layout_stride::mapping<extents_1d> output_mapping(extents_1d{2}, std::array<uni20::index_type, 1>{2});
  stdex::mdspan<double, extents_1d, stdex::layout_stride> input(input_storage.data(), input_mapping);
  stdex::mdspan<double, extents_1d, stdex::layout_stride> output(output_storage.data(), output_mapping);

  fill_matrix(matrix, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  input[0] = 1.0;
  input[1] = 2.0;
  input[2] = 3.0;
  output[0] = 10.0;
  output[1] = 20.0;

  EXPECT_EQ(uni20::linalg::blas::try_gemv(output, 2.0, matrix, input, 0.5), KernelAttempt::success);
  EXPECT_DOUBLE_EQ(output[0], 33.0);
  EXPECT_DOUBLE_EQ(output[1], 74.0);
  EXPECT_DOUBLE_EQ(output_storage[1], -100.0);
}

TEST(BlasGemvTest, LowersConjugatedRowMajorMatrixToConjugateTranspose)
{
  using Scalar = uni20::complex<double>;
  std::vector<Scalar> matrix_storage(4);
  std::vector<Scalar> input_storage{Scalar{1.0, 0.0}, Scalar{2.0, 0.0}};
  std::vector<Scalar> output_storage(2);

  stdex::mdspan<Scalar, extents_2d, stdex::layout_right> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> output(output_storage.data(), 2);
  fill_matrix(matrix, {Scalar{1.0, 1.0}, Scalar{2.0, 0.0}, Scalar{0.0, 3.0}, Scalar{4.0, -1.0}});

  EXPECT_EQ(uni20::linalg::blas::try_gemv(output, Scalar{1.0, 0.0}, uni20::conj(matrix), input, Scalar{}),
            KernelAttempt::success);
  EXPECT_EQ(output[0], (Scalar{5.0, -1.0}));
  EXPECT_EQ(output[1], (Scalar{8.0, -1.0}));
}

TEST(BlasGemvTest, DeclinesConjugateOnlyMatrixAndInputVector)
{
  using Scalar = uni20::complex<double>;
  std::vector<Scalar> matrix_storage(4, Scalar{1.0, 1.0});
  std::vector<Scalar> input_storage(2, Scalar{1.0, -1.0});
  std::vector<Scalar> output_storage(2, Scalar{7.0, 0.0});

  stdex::mdspan<Scalar, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<Scalar, extents_1d, stdex::layout_left> output(output_storage.data(), 2);

  EXPECT_EQ(uni20::linalg::blas::try_gemv(output, Scalar{1.0}, uni20::conj(matrix), input, Scalar{}),
            KernelAttempt::unsupported_transform);
  EXPECT_EQ(uni20::linalg::blas::try_gemv(output, Scalar{1.0}, matrix, uni20::conj(input), Scalar{}),
            KernelAttempt::unsupported_transform);
  EXPECT_EQ(output[0], (Scalar{7.0, 0.0}));
  EXPECT_EQ(output[1], (Scalar{7.0, 0.0}));
}

TEST(BlasGemvTest, AlphaZeroDoesNotReadProductOperands)
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> matrix_storage(4, nan);
  std::vector<double> input_storage(2, nan);
  std::vector<double> output_storage{2.0, -3.0};

  stdex::mdspan<double, extents_2d, stdex::layout_right> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> output(output_storage.data(), 2);

  EXPECT_EQ(uni20::linalg::blas::try_gemv(output, 0.0, matrix, input, 2.0), KernelAttempt::success);
  EXPECT_DOUBLE_EQ(output[0], 4.0);
  EXPECT_DOUBLE_EQ(output[1], -6.0);
}

TEST(BlasGemvTest, BetaZeroDoesNotReadOutput)
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> matrix_storage(4);
  std::vector<double> input_storage{2.0, 3.0};
  std::vector<double> output_storage(2, nan);

  stdex::mdspan<double, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> output(output_storage.data(), 2);
  fill_matrix(matrix, {1.0, 2.0, 3.0, 4.0});

  uni20::linalg::blas::gemv(output, 1.0, matrix, input, 0.0);
  EXPECT_DOUBLE_EQ(output[0], 8.0);
  EXPECT_DOUBLE_EQ(output[1], 18.0);
}

TEST(BlasGemvTest, EmptyInnerDimensionStillScalesOutput)
{
  std::vector<double> matrix_storage;
  std::vector<double> input_storage;
  std::vector<double> output_storage{2.0, -3.0};

  stdex::mdspan<double, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 0);
  stdex::mdspan<double, extents_1d, stdex::layout_left> input(input_storage.data(), 0);
  stdex::mdspan<double, extents_1d, stdex::layout_left> output(output_storage.data(), 2);

  EXPECT_EQ(uni20::linalg::blas::try_gemv(output, 1.0, matrix, input, 3.0), KernelAttempt::success);
  EXPECT_DOUBLE_EQ(output[0], 6.0);
  EXPECT_DOUBLE_EQ(output[1], -9.0);
}

TEST(BlasGemvTest, CheckedWrapperRejectsMismatchedDimensions)
{
  std::vector<double> matrix_storage(4);
  std::vector<double> input_storage(2);
  std::vector<double> output_storage(3);

  stdex::mdspan<double, extents_2d, stdex::layout_left> matrix(matrix_storage.data(), 2, 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> input(input_storage.data(), 2);
  stdex::mdspan<double, extents_1d, stdex::layout_left> output(output_storage.data(), 3);

  EXPECT_DEATH(uni20::linalg::blas::gemv(output, 1.0, matrix, input, 0.0), "output.size");
}
