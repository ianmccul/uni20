#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

namespace
{
struct NormalOnlySvdBackend
{
    static constexpr std::string_view name = "normal_only_svd";
};

template <uni20::MutableRankedMdspecLike<1> SingularValueMdspan, uni20::MutableRankedMdspecLike<2> LeftMdspan,
          uni20::MutableRankedMdspecLike<2> MatrixMdspan>
  requires uni20::HostWritableMdspec<SingularValueMdspan> && uni20::HostWritableMdspec<LeftMdspan> &&
           uni20::HostWritableMdspec<MatrixMdspan>
consteval auto kernel_accepts_types(NormalOnlySvdBackend const&, uni20::linalg::svd_left_op const&,
                                    SingularValueMdspan&, LeftMdspan&, MatrixMdspan&)
{
  return uni20::linalg::kernel_types_maybe;
}

template <uni20::MutableRankedMdspecLike<1> SingularValueMdspan, uni20::MutableRankedMdspecLike<2> LeftMdspan,
          uni20::MutableRankedMdspecLike<2> MatrixMdspan>
  requires uni20::HostWritableMdspec<SingularValueMdspan> && uni20::HostWritableMdspec<LeftMdspan> &&
           uni20::HostWritableMdspec<MatrixMdspan>
uni20::linalg::KernelAttempt try_kernel(NormalOnlySvdBackend, uni20::linalg::svd_left_op const& operation,
                                        SingularValueMdspan& singular_values, LeftMdspan& left_singular_vectors,
                                        MatrixMdspan& matrix_work)
{
  return uni20::linalg::try_kernel(uni20::linalg::LapackBackend{}, operation, singular_values, left_singular_vectors,
                                   matrix_work);
}

template <uni20::MutableRankedMdspecLike<1> SingularValueMdspan, uni20::MutableRankedMdspecLike<2> LeftMdspan,
          uni20::MutableRankedMdspecLike<2> RightAdjointMdspan, uni20::MutableRankedMdspecLike<2> MatrixMdspan>
  requires uni20::HostWritableMdspec<SingularValueMdspan> && uni20::HostWritableMdspec<LeftMdspan> &&
           uni20::HostWritableMdspec<RightAdjointMdspan> && uni20::HostWritableMdspec<MatrixMdspan>
consteval auto kernel_accepts_types(NormalOnlySvdBackend const&, uni20::linalg::svd_op const&, SingularValueMdspan&,
                                    LeftMdspan&, RightAdjointMdspan&, MatrixMdspan&)
{
  return uni20::linalg::kernel_types_maybe;
}

template <uni20::MutableRankedMdspecLike<1> SingularValueMdspan, uni20::MutableRankedMdspecLike<2> LeftMdspan,
          uni20::MutableRankedMdspecLike<2> RightAdjointMdspan, uni20::MutableRankedMdspecLike<2> MatrixMdspan>
  requires uni20::HostWritableMdspec<SingularValueMdspan> && uni20::HostWritableMdspec<LeftMdspan> &&
           uni20::HostWritableMdspec<RightAdjointMdspan> && uni20::HostWritableMdspec<MatrixMdspan>
uni20::linalg::KernelAttempt try_kernel(NormalOnlySvdBackend, uni20::linalg::svd_op const& operation,
                                        SingularValueMdspan& singular_values, LeftMdspan& left_singular_vectors,
                                        RightAdjointMdspan& right_singular_vectors_adjoint, MatrixMdspan& matrix_work)
{
  return uni20::linalg::try_kernel(uni20::linalg::LapackBackend{}, operation, singular_values, left_singular_vectors,
                                   right_singular_vectors_adjoint, matrix_work);
}

template <class Scalar> double scalar_error(Scalar const& actual, Scalar const& expected)
{
  return static_cast<double>(std::abs(actual - expected));
}

TEST(SvdTest, DestructiveSvdAcquiresNormalizedDeferredOperands)
{
  uni20::test::DeferredHostTensor<double, 2> matrix(2, 2);
  uni20::test::DeferredHostTensor<double, 1> values(2);
  uni20::test::DeferredHostTensor<double, 2> left(2, 2);
  uni20::test::DeferredHostTensor<double, 2> right(2, 2);
  matrix.storage() = {3.0, 0.0, 0.0, 2.0};

  uni20::linalg::singular_value_decomposition(uni20::linalg::LapackBackend{}, values, left, right, matrix);

  EXPECT_NEAR(values.storage()[0], 3.0, 1.0e-13);
  EXPECT_NEAR(values.storage()[1], 2.0, 1.0e-13);
}

template <class Matrix, class Result>
void expect_svd_reconstruction(Matrix const& original, Result const& result, double tolerance)
{
  using scalar_type = typename Matrix::value_type;
  std::size_t const rank = static_cast<std::size_t>(result.singular_values.extent(0));
  for (uni20::index_type row = 0; row < original.rows(); ++row)
  {
    for (uni20::index_type column = 0; column < original.cols(); ++column)
    {
      scalar_type reconstructed{};
      for (std::size_t inner = 0; inner < rank; ++inner)
      {
        auto const index = static_cast<uni20::index_type>(inner);
        reconstructed += result.left_singular_vectors[row, index] * result.singular_values[index] *
                         result.right_singular_vectors_adjoint[index, column];
      }
      EXPECT_LE(scalar_error(reconstructed, original[row, column]), tolerance);
    }
  }
}

template <class Matrix> void expect_orthonormal_columns(Matrix const& matrix, double tolerance)
{
  using scalar_type = typename Matrix::value_type;
  for (uni20::index_type lhs = 0; lhs < matrix.cols(); ++lhs)
  {
    for (uni20::index_type rhs = 0; rhs < matrix.cols(); ++rhs)
    {
      scalar_type inner{};
      for (uni20::index_type row = 0; row < matrix.rows(); ++row)
        inner += uni20::conj(matrix[row, lhs]) * matrix[row, rhs];
      scalar_type const expected = lhs == rhs ? scalar_type{1} : scalar_type{};
      EXPECT_LE(scalar_error(inner, expected), tolerance);
    }
  }
}

template <class Matrix> void expect_orthonormal_rows(Matrix const& matrix, double tolerance)
{
  using scalar_type = typename Matrix::value_type;
  for (uni20::index_type lhs = 0; lhs < matrix.rows(); ++lhs)
  {
    for (uni20::index_type rhs = 0; rhs < matrix.rows(); ++rhs)
    {
      scalar_type inner{};
      for (uni20::index_type column = 0; column < matrix.cols(); ++column)
        inner += matrix[lhs, column] * uni20::conj(matrix[rhs, column]);
      scalar_type const expected = lhs == rhs ? scalar_type{1} : scalar_type{};
      EXPECT_LE(scalar_error(inner, expected), tolerance);
    }
  }
}

template <class Matrix> void expect_identity(Matrix const& matrix)
{
  using scalar_type = typename Matrix::value_type;
  ASSERT_EQ(matrix.rows(), matrix.cols());
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
    for (uni20::index_type column = 0; column < matrix.cols(); ++column)
      EXPECT_EQ((matrix[row, column]), row == column ? scalar_type{1} : scalar_type{});
}

template <class Matrix, class Left, class SingularValues>
void expect_left_singular_equations(Matrix const& matrix, Left const& left, SingularValues const& singular_values,
                                    double tolerance)
{
  using scalar_type = typename Matrix::value_type;
  for (uni20::index_type index = 0; index < singular_values.extent(0); ++index)
  {
    for (uni20::index_type row = 0; row < matrix.rows(); ++row)
    {
      scalar_type transformed{};
      for (uni20::index_type column = 0; column < matrix.cols(); ++column)
      {
        scalar_type projected{};
        for (uni20::index_type inner = 0; inner < matrix.rows(); ++inner)
          projected += uni20::conj(matrix[inner, column]) * left[inner, index];
        transformed += matrix[row, column] * projected;
      }
      auto const expected = singular_values[index] * singular_values[index] * left[row, index];
      EXPECT_LE(scalar_error(transformed, expected), tolerance);
    }
  }
}

template <class Matrix, class SingularValues, class RightAdjoint>
void expect_right_singular_equations(Matrix const& matrix, SingularValues const& singular_values,
                                     RightAdjoint const& right_adjoint, double tolerance)
{
  using scalar_type = typename Matrix::value_type;
  for (uni20::index_type index = 0; index < singular_values.extent(0); ++index)
  {
    for (uni20::index_type column = 0; column < matrix.cols(); ++column)
    {
      scalar_type transformed{};
      for (uni20::index_type row = 0; row < matrix.rows(); ++row)
      {
        scalar_type projected{};
        for (uni20::index_type inner = 0; inner < matrix.cols(); ++inner)
          projected += matrix[row, inner] * uni20::conj(right_adjoint[index, inner]);
        transformed += uni20::conj(matrix[row, column]) * projected;
      }
      auto const expected = singular_values[index] * singular_values[index] * uni20::conj(right_adjoint[index, column]);
      EXPECT_LE(scalar_error(transformed, expected), tolerance);
    }
  }
}
} // namespace

TEST(SvdTest, ReducedValueApiPreservesRealRowMajorInput)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix(3, 2);
  matrix[0, 0] = 3.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = -2.0;
  matrix[1, 1] = 4.0;
  matrix[2, 0] = 0.5;
  matrix[2, 1] = -1.0;
  auto const original = matrix;

  auto result = uni20::linalg::svd(matrix);

  static_assert(std::same_as<typename decltype(result.left_singular_vectors)::layout_type, uni20::ColumnMajor>);
  static_assert(
      std::same_as<typename decltype(result.right_singular_vectors_adjoint)::layout_type, uni20::ColumnMajor>);
  ASSERT_EQ(result.left_singular_vectors.rows(), 3);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.singular_values.extent(0), 2);
  ASSERT_EQ(result.right_singular_vectors_adjoint.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_adjoint.cols(), 2);
  EXPECT_GE(result.singular_values[0], result.singular_values[1]);
  EXPECT_GE(result.singular_values[1], 0.0);
  expect_svd_reconstruction(original, result, 1.0e-12);
  expect_orthonormal_columns(result.left_singular_vectors, 1.0e-12);
  expect_orthonormal_rows(result.right_singular_vectors_adjoint, 1.0e-12);
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
    for (uni20::index_type column = 0; column < matrix.cols(); ++column)
      EXPECT_DOUBLE_EQ((matrix[row, column]), (original[row, column]));
}

TEST(SvdTest, ComplexValueApiReturnsConjugateTransposedRightVectors)
{
  using scalar_type = uni20::complex<double>;
  uni20::DenseMatrix<scalar_type> matrix(2, 3);
  matrix[0, 0] = scalar_type{1.0, 2.0};
  matrix[0, 1] = scalar_type{-3.0, 0.5};
  matrix[0, 2] = scalar_type{0.0, -1.0};
  matrix[1, 0] = scalar_type{2.0, -1.0};
  matrix[1, 1] = scalar_type{0.5, 4.0};
  matrix[1, 2] = scalar_type{-2.0, 3.0};

  auto result = uni20::linalg::svd(matrix);

  static_assert(std::same_as<typename decltype(result.singular_values)::value_type, double>);
  ASSERT_EQ(result.left_singular_vectors.rows(), 2);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.right_singular_vectors_adjoint.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_adjoint.cols(), 3);
  expect_svd_reconstruction(matrix, result, 2.0e-12);
  expect_orthonormal_columns(result.left_singular_vectors, 2.0e-12);
  expect_orthonormal_rows(result.right_singular_vectors_adjoint, 2.0e-12);
}

TEST(SvdTest, FullLeftAndRightExtentsAreIndependent)
{
  uni20::DenseMatrix<double> tall(3, 2);
  tall[0, 0] = 1.0;
  tall[1, 1] = 2.0;
  tall[2, 0] = 3.0;
  auto full_left = uni20::linalg::svd(tall, uni20::linalg::SvdOptions{.left = uni20::linalg::SvdVectorExtent::Full});

  EXPECT_EQ(full_left.left_singular_vectors.rows(), 3);
  EXPECT_EQ(full_left.left_singular_vectors.cols(), 3);
  EXPECT_EQ(full_left.right_singular_vectors_adjoint.rows(), 2);
  EXPECT_EQ(full_left.right_singular_vectors_adjoint.cols(), 2);
  expect_svd_reconstruction(tall, full_left, 1.0e-12);
  expect_orthonormal_columns(full_left.left_singular_vectors, 1.0e-12);

  uni20::DenseMatrix<double> wide(2, 3);
  wide[0, 0] = 1.0;
  wide[0, 2] = -2.0;
  wide[1, 1] = 3.0;
  auto full_right = uni20::linalg::svd(wide, uni20::linalg::SvdOptions{.right = uni20::linalg::SvdVectorExtent::Full});

  EXPECT_EQ(full_right.left_singular_vectors.rows(), 2);
  EXPECT_EQ(full_right.left_singular_vectors.cols(), 2);
  EXPECT_EQ(full_right.right_singular_vectors_adjoint.rows(), 3);
  EXPECT_EQ(full_right.right_singular_vectors_adjoint.cols(), 3);
  expect_svd_reconstruction(wide, full_right, 1.0e-12);
  expect_orthonormal_rows(full_right.right_singular_vectors_adjoint, 1.0e-12);
}

TEST(SvdTest, AcceptsSingletonRowColumnMajorMatrix)
{
  uni20::DenseMatrix<double> matrix(1, 2);
  matrix[0, 0] = 3.0;
  matrix[0, 1] = 4.0;
  ASSERT_DOUBLE_EQ((matrix[0, 0]), 3.0);
  ASSERT_DOUBLE_EQ((matrix[0, 1]), 4.0);

  auto copy = uni20::make_tensor<uni20::ColumnMajor>(matrix);
  ASSERT_DOUBLE_EQ((copy[0, 0]), 3.0);
  ASSERT_DOUBLE_EQ((copy[0, 1]), 4.0);
  auto provider = uni20::linalg::blas::try_lapack_writable_matrix(copy.mdspan());
  ASSERT_TRUE(provider);
  EXPECT_EQ(provider->rows, 1);
  EXPECT_EQ(provider->cols, 2);
  EXPECT_EQ(provider->leading_dimension, 1);
  EXPECT_DOUBLE_EQ(provider->data[0], 3.0);
  EXPECT_DOUBLE_EQ(provider->data[1], 4.0);

  auto result = uni20::linalg::svd(matrix);

  ASSERT_EQ(result.singular_values.extent(0), 1);
  EXPECT_DOUBLE_EQ(result.singular_values[0], 5.0);
  expect_svd_reconstruction(matrix, result, 1.0e-12);
}

TEST(SvdTest, ValuesOnlyAndOneSidedApisComputeOnlyRequestedFactors)
{
  using scalar_type = uni20::complex<double>;
  uni20::DenseMatrix<scalar_type> matrix(3, 2);
  matrix[0, 0] = scalar_type{3.0, 1.0};
  matrix[0, 1] = scalar_type{1.0, -2.0};
  matrix[1, 0] = scalar_type{-2.0, 0.5};
  matrix[1, 1] = scalar_type{4.0, 1.0};
  matrix[2, 0] = scalar_type{0.5, -1.0};
  matrix[2, 1] = scalar_type{-1.0, 2.0};

  auto values = uni20::linalg::singular_values(matrix);
  auto [left, left_values] = uni20::linalg::svd_left(matrix);
  auto [right_values, right_adjoint] = uni20::linalg::svd_right(matrix);

  ASSERT_EQ(values.extent(0), 2);
  ASSERT_EQ(left.rows(), 3);
  ASSERT_EQ(left.cols(), 2);
  ASSERT_EQ(right_adjoint.rows(), 2);
  ASSERT_EQ(right_adjoint.cols(), 2);
  for (uni20::index_type index = 0; index < values.extent(0); ++index)
  {
    EXPECT_NEAR(values[index], left_values[index], 1.0e-12);
    EXPECT_NEAR(values[index], right_values[index], 1.0e-12);
  }
  expect_orthonormal_columns(left, 2.0e-12);
  expect_orthonormal_rows(right_adjoint, 2.0e-12);
  expect_left_singular_equations(matrix, left, left_values, 1.0e-10);
  expect_right_singular_equations(matrix, right_values, right_adjoint, 1.0e-10);
}

TEST(SvdTest, OneSidedFullFactorsHaveIndependentExtentPolicies)
{
  uni20::DenseMatrix<double> tall(3, 2);
  tall[0, 0] = 1.0;
  tall[1, 1] = 2.0;
  tall[2, 0] = 3.0;
  auto [left, left_values] = uni20::linalg::svd_left(tall, uni20::linalg::SvdVectorExtent::Full);
  EXPECT_EQ(left.rows(), 3);
  EXPECT_EQ(left.cols(), 3);
  EXPECT_EQ(left_values.extent(0), 2);
  expect_orthonormal_columns(left, 1.0e-12);

  uni20::DenseMatrix<double> wide(2, 3);
  wide[0, 0] = 1.0;
  wide[0, 2] = -2.0;
  wide[1, 1] = 3.0;
  auto [right_values, right_adjoint] = uni20::linalg::svd_right(wide, uni20::linalg::SvdVectorExtent::Full);
  EXPECT_EQ(right_values.extent(0), 2);
  EXPECT_EQ(right_adjoint.rows(), 3);
  EXPECT_EQ(right_adjoint.cols(), 3);
  expect_orthonormal_rows(right_adjoint, 1.0e-12);
}

TEST(SvdTest, ConsumingReducedFactorsAdoptInputStorage)
{
  uni20::DenseMatrix<double> tall(3, 2);
  tall[0, 0] = 3.0;
  tall[1, 1] = 2.0;
  tall[2, 0] = 1.0;
  auto* tall_storage = tall.mdspan().data_handle();
  auto [left, left_values] = uni20::linalg::svd_left(std::move(tall));
  EXPECT_EQ(left.mdspan().data_handle(), tall_storage);
  EXPECT_EQ(left.mapping().stride(0), 1);
  EXPECT_EQ(left.mapping().stride(1), 3);
  expect_orthonormal_columns(left, 1.0e-12);
  EXPECT_EQ(left_values.extent(0), 2);

  uni20::DenseMatrix<double> wide(2, 3);
  wide[0, 0] = 4.0;
  wide[0, 2] = -1.0;
  wide[1, 1] = 2.0;
  auto* wide_storage = wide.mdspan().data_handle();
  auto [right_values, right_adjoint] = uni20::linalg::svd_right(std::move(wide));
  EXPECT_EQ(right_adjoint.mdspan().data_handle(), wide_storage);
  EXPECT_EQ(right_adjoint.mapping().stride(0), 1);
  EXPECT_EQ(right_adjoint.mapping().stride(1), 2);
  expect_orthonormal_rows(right_adjoint, 1.0e-12);
  EXPECT_EQ(right_values.extent(0), 2);
}

TEST(SvdTest, ConsumingExplicitBackendMayDeclineOverwriteOptimizationAtCompileTime)
{
  uni20::DenseMatrix<double> matrix(3, 2);
  matrix[0, 0] = 3.0;
  matrix[1, 1] = 2.0;
  matrix[2, 0] = 1.0;
  auto* input_storage = matrix.mdspan().data_handle();

  auto [left, singular_values] = uni20::linalg::svd_left(NormalOnlySvdBackend{}, std::move(matrix));

  EXPECT_NE(left.mdspan().data_handle(), input_storage);
  EXPECT_EQ(left.rows(), 3);
  EXPECT_EQ(left.cols(), 2);
  EXPECT_EQ(singular_values.extent(0), 2);
  expect_orthonormal_columns(left, 1.0e-12);
}

TEST(SvdTest, ConsumingCompleteSvdFallsBackWhenExplicitBackendHasNoOverwriteSignature)
{
  uni20::DenseMatrix<double> matrix(3, 2);
  matrix[0, 0] = 3.0;
  matrix[0, 1] = -1.0;
  matrix[1, 0] = 2.0;
  matrix[1, 1] = 4.0;
  matrix[2, 0] = 0.5;
  matrix[2, 1] = -2.0;
  auto const original = matrix;
  auto* input_storage = matrix.mdspan().data_handle();

  auto result = uni20::linalg::svd(NormalOnlySvdBackend{}, std::move(matrix));

  EXPECT_NE(result.left_singular_vectors.mdspan().data_handle(), input_storage);
  EXPECT_NE(result.right_singular_vectors_adjoint.mdspan().data_handle(), input_storage);
  expect_svd_reconstruction(original, result, 1.0e-12);
}

TEST(SvdTest, ConsumingCompleteSvdPreservesPaddedLeadingDimension)
{
  using matrix_type = uni20::StridedTensor<double, 2>;
  using extents_type = typename matrix_type::extents_type;
  matrix_type matrix(extents_type{3, 2}, std::array<uni20::index_type, 2>{1, 5});
  matrix[0, 0] = 3.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = -2.0;
  matrix[1, 1] = 4.0;
  matrix[2, 0] = 0.5;
  matrix[2, 1] = -1.0;
  auto const original = matrix;
  auto* storage = matrix.mdspan().data_handle();
  auto result = uni20::linalg::svd(std::move(matrix));

  EXPECT_EQ(result.left_singular_vectors.mdspan().data_handle(), storage);
  EXPECT_EQ(result.left_singular_vectors.mapping().stride(0), 1);
  EXPECT_EQ(result.left_singular_vectors.mapping().stride(1), 5);
  expect_svd_reconstruction(original, result, 1.0e-12);
}

TEST(SvdTest, ConsumingCompleteSvdCanOverwriteEitherReducedFactor)
{
  uni20::DenseMatrix<double> wide(2, 3);
  wide[0, 0] = 3.0;
  wide[0, 2] = 1.0;
  wide[1, 1] = 2.0;
  auto const wide_original = wide;
  auto* wide_storage = wide.mdspan().data_handle();
  auto wide_result = uni20::linalg::svd(std::move(wide));
  EXPECT_EQ(wide_result.right_singular_vectors_adjoint.mdspan().data_handle(), wide_storage);
  expect_svd_reconstruction(wide_original, wide_result, 1.0e-12);

  uni20::DenseMatrix<double> tall(3, 2);
  tall[0, 0] = 3.0;
  tall[1, 1] = 2.0;
  tall[2, 0] = 1.0;
  auto const tall_original = tall;
  auto* tall_storage = tall.mdspan().data_handle();
  auto tall_result =
      uni20::linalg::svd(std::move(tall), uni20::linalg::SvdOptions{.left = uni20::linalg::SvdVectorExtent::Full});
  EXPECT_EQ(tall_result.right_singular_vectors_adjoint.mdspan().data_handle(), tall_storage);
  EXPECT_EQ(tall_result.left_singular_vectors.cols(), 3);
  expect_svd_reconstruction(tall_original, tall_result, 1.0e-12);
}

TEST(SvdTest, ConsumingRowMajorInputMaterializesCompatibleWorkStorage)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix(3, 2);
  matrix[0, 0] = 3.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = -2.0;
  matrix[1, 1] = 4.0;
  matrix[2, 0] = 0.5;
  matrix[2, 1] = -1.0;
  auto const original = matrix;
  auto* input_storage = matrix.mdspan().data_handle();

  auto result = uni20::linalg::svd(std::move(matrix));

  EXPECT_NE(result.left_singular_vectors.mdspan().data_handle(), input_storage);
  EXPECT_NE(result.right_singular_vectors_adjoint.mdspan().data_handle(), input_storage);
  expect_svd_reconstruction(original, result, 1.0e-12);
}

TEST(SvdTest, DestructiveApiResizesOutputs)
{
  uni20::DenseMatrix<double> matrix_work(2, 2);
  matrix_work[0, 0] = 4.0;
  matrix_work[1, 1] = 2.0;
  uni20::Tensor<double, 1> singular_values;
  uni20::DenseMatrix<double> left_singular_vectors;
  uni20::DenseMatrix<double> right_singular_vectors_adjoint;

  uni20::linalg::singular_value_decomposition(singular_values, left_singular_vectors, right_singular_vectors_adjoint,
                                              matrix_work);

  ASSERT_EQ(singular_values.extent(0), 2);
  EXPECT_NEAR(singular_values[0], 4.0, 1.0e-13);
  EXPECT_NEAR(singular_values[1], 2.0, 1.0e-13);
  EXPECT_EQ(left_singular_vectors.rows(), 2);
  EXPECT_EQ(left_singular_vectors.cols(), 2);
  EXPECT_EQ(right_singular_vectors_adjoint.rows(), 2);
  EXPECT_EQ(right_singular_vectors_adjoint.cols(), 2);
}

TEST(SvdTest, DirectLapackBackendDeclinesRowMajorBeforeMutation)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix_work(2, 2);
  matrix_work[0, 0] = 4.0;
  matrix_work[0, 1] = 1.0;
  matrix_work[1, 0] = -2.0;
  matrix_work[1, 1] = 3.0;
  auto const original_matrix = matrix_work;

  uni20::Tensor<double, 1> singular_values(2);
  singular_values[0] = -1.0;
  singular_values[1] = -2.0;
  uni20::DenseMatrix<double, uni20::RowMajor> left_singular_vectors(2, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> right_singular_vectors_adjoint(2, 2);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      left_singular_vectors[row, column] = 7.0;
      right_singular_vectors_adjoint[row, column] = 9.0;
    }
  }

  auto singular_value_span = singular_values.mdspan();
  auto left_span = left_singular_vectors.mdspan();
  auto right_span = right_singular_vectors_adjoint.mdspan();
  auto matrix_span = matrix_work.mdspan();
  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(uni20::linalg::LapackBackend{}, uni20::linalg::svd_op{},
                                                  singular_value_span, left_span, right_span, matrix_span));

  EXPECT_EQ(singular_values[0], -1.0);
  EXPECT_EQ(singular_values[1], -2.0);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      EXPECT_EQ((left_singular_vectors[row, column]), 7.0);
      EXPECT_EQ((right_singular_vectors_adjoint[row, column]), 9.0);
      EXPECT_EQ((matrix_work[row, column]), (original_matrix[row, column]));
    }
  }
}

TEST(SvdTest, ZeroExtentFullFactorsUseIdentityCompletion)
{
  uni20::DenseMatrix<double> no_rows(0, 3);
  auto right_completion =
      uni20::linalg::svd(no_rows, uni20::linalg::SvdOptions{.right = uni20::linalg::SvdVectorExtent::Full});

  EXPECT_EQ(right_completion.left_singular_vectors.rows(), 0);
  EXPECT_EQ(right_completion.left_singular_vectors.cols(), 0);
  EXPECT_EQ(right_completion.singular_values.extent(0), 0);
  expect_identity(right_completion.right_singular_vectors_adjoint);

  uni20::DenseMatrix<double> no_columns(3, 0);
  auto left_completion =
      uni20::linalg::svd(no_columns, uni20::linalg::SvdOptions{.left = uni20::linalg::SvdVectorExtent::Full});

  expect_identity(left_completion.left_singular_vectors);
  EXPECT_EQ(left_completion.singular_values.extent(0), 0);
  EXPECT_EQ(left_completion.right_singular_vectors_adjoint.rows(), 0);
  EXPECT_EQ(left_completion.right_singular_vectors_adjoint.cols(), 0);
}
