#include <mplapack_config.h>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/krylov/dense_subspace_unused.hpp>
#include <uni20/krylov/tridiagonal.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using Binary128 = mplapack_binary128_t;
template <typename Scalar> using DenseMatrix = uni20::DenseMatrix<Scalar>;

template <typename Scalar, std::integral Rows, std::integral Columns>
DenseMatrix<Scalar> zero_matrix(Rows rows, Columns columns)
{
  DenseMatrix<Scalar> result(rows, columns);
  std::fill_n(result.data(), result.size(), Scalar{});
  return result;
}

Binary128 abs_error(Binary128 actual, Binary128 expected) { return std::abs(actual - expected); }

Binary128 complex_abs(uni20::complex<Binary128> const& value) { return std::abs(value); }

Binary128 tolerance() { return static_cast<Binary128>(1.0e-30L); }

Binary128 schur_right_eigenvector_residual_max(DenseMatrix<Binary128> const& schur_form,
                                               uni20::complex<Binary128> eigenvalue,
                                               DenseMatrix<uni20::complex<Binary128>> const& eigenvectors,
                                               std::size_t column)
{
  Binary128 residual_max{};
  for (uni20::index_type row = 0; row < schur_form.rows(); ++row)
  {
    uni20::complex<Binary128> applied{};
    for (uni20::index_type inner = 0; inner < schur_form.cols(); ++inner)
    {
      applied += uni20::complex<Binary128>{schur_form[row, inner], Binary128{}} * eigenvectors[inner, column];
    }
    residual_max = std::max(residual_max, complex_abs(applied - eigenvalue * eigenvectors[row, column]));
  }
  return residual_max;
}

Binary128 generalized_schur_right_eigenvector_residual_max(DenseMatrix<Binary128> const& matrix_schur_form,
                                                           DenseMatrix<Binary128> const& metric_schur_form,
                                                           uni20::complex<Binary128> eigenvalue,
                                                           DenseMatrix<uni20::complex<Binary128>> const& eigenvectors,
                                                           std::size_t column)
{
  Binary128 residual_max{};
  for (uni20::index_type row = 0; row < matrix_schur_form.rows(); ++row)
  {
    uni20::complex<Binary128> matrix_applied{};
    uni20::complex<Binary128> metric_applied{};
    for (uni20::index_type inner = 0; inner < matrix_schur_form.cols(); ++inner)
    {
      matrix_applied +=
          uni20::complex<Binary128>{matrix_schur_form[row, inner], Binary128{}} * eigenvectors[inner, column];
      metric_applied +=
          uni20::complex<Binary128>{metric_schur_form[row, inner], Binary128{}} * eigenvectors[inner, column];
    }
    residual_max = std::max(residual_max, complex_abs(matrix_applied - eigenvalue * metric_applied));
  }
  return residual_max;
}

std::span<Binary128 const> const_span(std::vector<Binary128> const& values)
{
  return std::span<Binary128 const>(values.data(), values.size());
}

std::span<Binary128> mutable_span(std::vector<Binary128>& values)
{
  return std::span<Binary128>(values.data(), values.size());
}

DenseMatrix<Binary128> multiply_for_test(DenseMatrix<Binary128> const& lhs, DenseMatrix<Binary128> const& rhs)
{
  if (lhs.cols() != rhs.rows())
  {
    throw std::invalid_argument("test matrix dimensions do not agree");
  }

  auto result = zero_matrix<Binary128>(lhs.rows(), rhs.cols());
  for (uni20::index_type row = 0; row < lhs.rows(); ++row)
  {
    for (uni20::index_type inner = 0; inner < lhs.cols(); ++inner)
    {
      Binary128 const factor = lhs[row, inner];
      for (uni20::index_type col = 0; col < rhs.cols(); ++col)
      {
        result[row, col] += factor * rhs[inner, col];
      }
    }
  }
  return result;
}

DenseMatrix<Binary128> transpose(DenseMatrix<Binary128> const& matrix)
{
  auto result = zero_matrix<Binary128>(matrix.cols(), matrix.rows());
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.cols(); ++col)
    {
      result[col, row] = matrix[row, col];
    }
  }
  return result;
}

DenseMatrix<Binary128> identity_matrix(std::size_t order)
{
  auto result = zero_matrix<Binary128>(order, order);
  for (std::size_t index = 0; index < order; ++index)
  {
    result[index, index] = Binary128{1};
  }
  return result;
}

Binary128 binary_power_of_two(int exponent)
{
  Binary128 result{1};
  if (exponent >= 0)
  {
    for (int i = 0; i < exponent; ++i)
    {
      result *= Binary128{2};
    }
  }
  else
  {
    for (int i = 0; i < -exponent; ++i)
    {
      result /= Binary128{2};
    }
  }
  return result;
}

Binary128 below_double_resolution_gap() { return binary_power_of_two(-80); }

Binary128 below_double_minimum_value() { return binary_power_of_two(-1200); }

void expect_gap_is_binary128_only(Binary128 gap)
{
  EXPECT_TRUE(Binary128{1} + gap > Binary128{1});
  EXPECT_EQ(static_cast<double>(Binary128{1} + gap), 1.0);
}

void expect_value_underflows_to_double_zero(Binary128 value)
{
  EXPECT_GT(value, Binary128{});
  EXPECT_EQ(static_cast<double>(value), 0.0);
}

TEST(MplapackBinary128DenseSubspaceTest, ScalarConceptsAcceptBackendReal)
{
  static_assert(uni20::Real<Binary128>);
  static_assert(uni20::BlasReal<Binary128>);
  static_assert(uni20::LapackComplex<uni20::complex<Binary128>>);
  static_assert(uni20::LapackComplexReal<Binary128>);
  static_assert(uni20::LapackScalar<uni20::complex<Binary128>>);
  static_assert(std::same_as<uni20::make_complex_t<Binary128>, uni20::complex<Binary128>>);
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexNonsymmetricEigensystemResolvesGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);
  Complex const first{Binary128{1} + delta, Binary128{1} + delta};
  Complex const second{Binary128{1} + Binary128{2} * delta, Binary128{1} + Binary128{2} * delta};
  EXPECT_EQ(static_cast<double>(first.real()), 1.0);
  EXPECT_EQ(static_cast<double>(first.imag()), 1.0);
  EXPECT_EQ(static_cast<double>(second.real()), 1.0);
  EXPECT_EQ(static_cast<double>(second.imag()), 1.0);

  auto matrix = zero_matrix<Complex>(2, 2);
  matrix[0, 0] = first;
  matrix[0, 1] = Complex{delta, -delta};
  matrix[1, 1] = second;

  auto result = uni20::krylov::complex_nonsymmetric_eigensystem<Binary128>(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_TRUE(std::min(complex_abs(result.eigenvalues[0] - first), complex_abs(result.eigenvalues[1] - first)) <=
              Binary128{1000} * tolerance());
  EXPECT_TRUE(std::min(complex_abs(result.eigenvalues[0] - second), complex_abs(result.eigenvalues[1] - second)) <=
              Binary128{1000} * tolerance());
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexSchurResolvesGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);
  Complex const first{Binary128{1} + delta, Binary128{1} + delta};
  Complex const second{Binary128{1} + Binary128{2} * delta, Binary128{1} + Binary128{2} * delta};

  auto matrix = zero_matrix<Complex>(2, 2);
  matrix[0, 0] = first;
  matrix[0, 1] = Complex{delta, -delta};
  matrix[1, 1] = second;

  auto result = uni20::krylov::complex_schur<Binary128>(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.schur_form.rows(), 2);
  ASSERT_EQ(result.schur_form.cols(), 2);
  ASSERT_EQ(result.schur_vectors.rows(), 2);
  ASSERT_EQ(result.schur_vectors.cols(), 2);
  EXPECT_TRUE(std::min(complex_abs(result.eigenvalues[0] - first), complex_abs(result.eigenvalues[1] - first)) <=
              Binary128{1000} * tolerance());
  EXPECT_TRUE(std::min(complex_abs(result.eigenvalues[0] - second), complex_abs(result.eigenvalues[1] - second)) <=
              Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexSchurReordersBinary128SeparatedEntry)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);
  Complex const first{Binary128{1} + delta, Binary128{1} + delta};
  Complex const second{Binary128{1} + Binary128{2} * delta, Binary128{1} + Binary128{2} * delta};
  Complex const third{Binary128{1} + Binary128{3} * delta, Binary128{1} + Binary128{3} * delta};

  auto matrix = zero_matrix<Complex>(3, 3);
  matrix[0, 0] = first;
  matrix[1, 1] = second;
  matrix[2, 2] = third;

  auto schur = uni20::krylov::complex_schur<Binary128>(matrix, true);
  std::size_t third_index = schur.eigenvalues.size();
  for (std::size_t index = 0; index < schur.eigenvalues.size(); ++index)
  {
    if (complex_abs(schur.eigenvalues[index] - third) <= Binary128{1000} * tolerance())
    {
      third_index = index;
      break;
    }
  }
  ASSERT_LT(third_index, schur.eigenvalues.size());

  auto reordered = uni20::krylov::reorder_complex_schur<Binary128>(std::move(schur), {third_index});

  ASSERT_EQ(reordered.eigenvalues.size(), 3);
  EXPECT_TRUE(complex_abs(reordered.eigenvalues[0] - third) <= Binary128{1000} * tolerance());
  ASSERT_EQ(reordered.schur_vectors.rows(), 3);
  ASSERT_EQ(reordered.schur_vectors.cols(), 3);
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexHermitianEigensystemResolvesGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Complex>(2, 2);
  matrix[0, 0] = Complex{Binary128{1}, Binary128{}};
  matrix[0, 1] = Complex{Binary128{}, offdiagonal};
  matrix[1, 1] = Complex{Binary128{1} + delta, Binary128{}};

  auto result = uni20::krylov::complex_hermitian_eigensystem<Binary128>(matrix, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Complex const x0 = result.eigenvectors[0, col];
    Complex const x1 = result.eigenvectors[1, col];
    Complex const residual0 = x0 + Complex{Binary128{}, offdiagonal} * x1 - lambda * x0;
    Complex const residual1 = Complex{Binary128{}, -offdiagonal} * x0 + (Binary128{1} + delta) * x1 - lambda * x1;
    EXPECT_TRUE(std::sqrt(std::norm(residual0) + std::norm(residual1)) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexHermitianDivideAndConquerEigensystemResolvesGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Complex>(2, 2);
  matrix[0, 0] = Complex{Binary128{1}, Binary128{}};
  matrix[0, 1] = Complex{Binary128{}, offdiagonal};
  matrix[1, 1] = Complex{Binary128{1} + delta, Binary128{}};

  auto result = uni20::krylov::complex_hermitian_eigensystem_divide_and_conquer<Binary128>(matrix, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedComplexHermitianEigensystemResolvesGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Complex>(3, 3);
  matrix[0, 0] = Complex{Binary128{1}, Binary128{}};
  matrix[1, 1] = Complex{Binary128{1} + delta, Binary128{}};
  matrix[2, 2] = Complex{Binary128{2}, Binary128{}};

  auto result = uni20::krylov::complex_hermitian_eigensystem_index_range<Binary128>(matrix, 0, 1, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[1]), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexGeneralizedHermitianEigensystemResolvesMetricGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  Binary128 const root_two = std::sqrt(Binary128{2});
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Complex>(2, 2);
  matrix[0, 0] = Complex{Binary128{1}, Binary128{}};
  matrix[0, 1] = Complex{Binary128{}, root_two * offdiagonal};
  matrix[1, 1] = Complex{Binary128{2} * (Binary128{1} + delta), Binary128{}};
  EXPECT_EQ(static_cast<double>(matrix[1, 1].real()), 2.0);

  auto metric = zero_matrix<Complex>(2, 2);
  metric[0, 0] = Complex{Binary128{1}, Binary128{}};
  metric[1, 1] = Complex{Binary128{2}, Binary128{}};

  auto result = uni20::krylov::complex_generalized_hermitian_eigensystem<Binary128>(matrix, metric, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= Binary128{1000} * tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Complex const x0 = result.eigenvectors[0, col];
    Complex const x1 = result.eigenvectors[1, col];
    Complex const residual0 = x0 + Complex{Binary128{}, root_two * offdiagonal} * x1 - lambda * x0;
    Complex const residual1 = Complex{Binary128{}, -root_two * offdiagonal} * x0 +
                              Binary128{2} * (Binary128{1} + delta) * x1 - lambda * Binary128{2} * x1;
    Binary128 const metric_norm = std::norm(x0) + Binary128{2} * std::norm(x1);
    EXPECT_TRUE(std::sqrt(std::norm(residual0) + std::norm(residual1)) <= Binary128{1000} * tolerance());
    EXPECT_TRUE(abs_error(metric_norm, Binary128{1}) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest,
     ComplexGeneralizedHermitianDivideAndConquerEigensystemResolvesMetricGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  Binary128 const root_two = std::sqrt(Binary128{2});
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Complex>(2, 2);
  matrix[0, 0] = Complex{Binary128{1}, Binary128{}};
  matrix[0, 1] = Complex{Binary128{}, root_two * offdiagonal};
  matrix[1, 1] = Complex{Binary128{2} * (Binary128{1} + delta), Binary128{}};
  EXPECT_EQ(static_cast<double>(matrix[1, 1].real()), 2.0);

  auto metric = zero_matrix<Complex>(2, 2);
  metric[0, 0] = Complex{Binary128{1}, Binary128{}};
  metric[1, 1] = Complex{Binary128{2}, Binary128{}};

  auto result =
      uni20::krylov::complex_generalized_hermitian_eigensystem_divide_and_conquer<Binary128>(matrix, metric, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest,
     SelectedComplexGeneralizedHermitianEigensystemResolvesMetricGapBelowDoublePrecision)
{
  using Complex = uni20::complex<Binary128>;

  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Complex>(3, 3);
  matrix[0, 0] = Complex{Binary128{1}, Binary128{}};
  matrix[1, 1] = Complex{Binary128{2} * (Binary128{1} + delta), Binary128{}};
  matrix[2, 2] = Complex{Binary128{6}, Binary128{}};
  EXPECT_EQ(static_cast<double>(matrix[1, 1].real()), 2.0);

  auto metric = zero_matrix<Complex>(3, 3);
  metric[0, 0] = Complex{Binary128{1}, Binary128{}};
  metric[1, 1] = Complex{Binary128{2}, Binary128{}};
  metric[2, 2] = Complex{Binary128{3}, Binary128{}};

  auto result =
      uni20::krylov::complex_generalized_hermitian_eigensystem_index_range<Binary128>(matrix, metric, 0, 1, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[1]), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], Binary128{1}) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], Binary128{1} + delta) <= Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, MatrixNormsPreserveBinary128OnlyIncrement)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);
  Binary128 const one_plus_delta = Binary128{1} + delta;

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = one_plus_delta;
  matrix[1, 1] = Binary128{1};

  Binary128 const result = uni20::krylov::real_matrix_norm(matrix, uni20::krylov::MatrixNorm::MaxAbs);

  EXPECT_TRUE(abs_error(result, one_plus_delta) <= tolerance());
  EXPECT_GT(result, Binary128{1});
  EXPECT_EQ(static_cast<double>(result), 1.0);
}

TEST(MplapackBinary128DenseSubspaceTest, MatrixFrobeniusNormScalesAboveDoubleRange)
{
  Binary128 const huge = binary_power_of_two(1200);
  EXPECT_TRUE(std::isinf(static_cast<double>(huge)));

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = huge;
  matrix[1, 1] = huge;

  Binary128 const result = uni20::krylov::real_matrix_norm(matrix, uni20::krylov::MatrixNorm::Frobenius);
  Binary128 const expected_scale = std::sqrt(Binary128{2});

  EXPECT_TRUE(abs_error(result / huge, expected_scale) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(std::isinf(static_cast<double>(result)));
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricMatrixNormPreservesBinary128OnlyColumnSum)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);
  Binary128 const expected = Binary128{1} + Binary128{2} * delta;

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[0, 1] = delta;
  matrix[1, 0] = binary_power_of_two(800);
  matrix[1, 1] = Binary128{1};

  Binary128 const result = uni20::krylov::real_symmetric_matrix_norm(matrix, uni20::krylov::MatrixNorm::One,
                                                                     uni20::krylov::MatrixFill::Upper);

  EXPECT_TRUE(abs_error(result, expected) <= Binary128{1000} * tolerance());
  EXPECT_GT(result, Binary128{1});
  EXPECT_EQ(static_cast<double>(result), 1.0);
}

TEST(MplapackBinary128DenseSubspaceTest, TriangularMatrixNormPreservesBinary128OnlyDiagonalAndUnitFlag)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);
  Binary128 const stored_diagonal_max = Binary128{1} + Binary128{3} * delta;

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1} + Binary128{2} * delta;
  matrix[0, 1] = delta;
  matrix[1, 0] = binary_power_of_two(800);
  matrix[1, 1] = stored_diagonal_max;

  Binary128 const nonunit = uni20::krylov::real_triangular_matrix_norm(matrix, uni20::krylov::MatrixNorm::MaxAbs,
                                                                       uni20::krylov::MatrixFill::Upper);
  Binary128 const unit = uni20::krylov::real_triangular_matrix_norm(matrix, uni20::krylov::MatrixNorm::MaxAbs,
                                                                    uni20::krylov::MatrixFill::Upper, true);

  EXPECT_TRUE(abs_error(nonunit, stored_diagonal_max) <= Binary128{1000} * tolerance());
  EXPECT_GT(nonunit, Binary128{1});
  EXPECT_EQ(static_cast<double>(nonunit), 1.0);
  EXPECT_TRUE(abs_error(unit, Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralBandMatrixNormPreservesValuesBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{2} * tiny;
  matrix[1, 0] = Binary128{0.5} * tiny;
  matrix[2, 0] = binary_power_of_two(800);
  matrix[0, 1] = Binary128{0.25} * tiny;
  matrix[1, 1] = Binary128{3} * tiny;
  matrix[2, 1] = Binary128{0.75} * tiny;
  matrix[1, 2] = Binary128{-0.25} * tiny;
  matrix[2, 2] = Binary128{4} * tiny;

  auto band = uni20::krylov::real_general_band_from_dense(matrix, 1, 1);
  Binary128 const result = uni20::krylov::real_general_band_matrix_norm(band, uni20::krylov::MatrixNorm::One);

  expect_value_underflows_to_double_zero(result);
  EXPECT_TRUE(abs_error(result / tiny, Binary128{4.25}) <= Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralBandSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{2} * tiny;
  matrix[1, 0] = Binary128{0.5} * tiny;
  matrix[0, 1] = Binary128{0.25} * tiny;
  matrix[1, 1] = Binary128{3} * tiny;
  matrix[2, 1] = Binary128{0.75} * tiny;
  matrix[1, 2] = Binary128{-0.25} * tiny;
  matrix[2, 2] = Binary128{4} * tiny;

  auto expected_solution = zero_matrix<Binary128>(3, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto direct_solution = uni20::krylov::real_general_band_solve(
      uni20::krylov::real_general_band_from_dense(matrix, 1, 1), right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(direct_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto factorization =
      uni20::krylov::real_general_band_factorization(uni20::krylov::real_general_band_from_dense(matrix, 1, 1));
  auto factorized_solution = uni20::krylov::real_general_band_solve(factorization, right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(factorized_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralTridiagonalSolveAcceptsPivotingBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{0.25} * tiny;
  matrix[0, 1] = Binary128{2} * tiny;
  matrix[1, 0] = Binary128{3} * tiny;
  matrix[1, 1] = Binary128{4} * tiny;
  matrix[1, 2] = Binary128{-1} * tiny;
  matrix[2, 1] = Binary128{0.5} * tiny;
  matrix[2, 2] = Binary128{5} * tiny;

  auto tridiagonal = uni20::krylov::real_general_tridiagonal_from_dense(matrix);
  ASSERT_EQ(tridiagonal.diagonal.size(), 3);
  ASSERT_EQ(tridiagonal.lower_diagonal.size(), 2);
  ASSERT_EQ(tridiagonal.upper_diagonal.size(), 2);
  for (Binary128 const value : tridiagonal.diagonal)
  {
    expect_value_underflows_to_double_zero(std::abs(value));
  }
  for (Binary128 const value : tridiagonal.lower_diagonal)
  {
    expect_value_underflows_to_double_zero(std::abs(value));
  }
  for (Binary128 const value : tridiagonal.upper_diagonal)
  {
    expect_value_underflows_to_double_zero(std::abs(value));
  }

  auto expected_solution = zero_matrix<Binary128>(3, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto direct_solution = uni20::krylov::real_general_tridiagonal_solve(tridiagonal, right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(direct_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto factorization = uni20::krylov::real_general_tridiagonal_factorization(tridiagonal);
  ASSERT_EQ(factorization.pivot_rows.size(), 3);
  EXPECT_EQ(factorization.pivot_rows[0], 1);
  auto factorized_solution = uni20::krylov::real_general_tridiagonal_solve(factorization, right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(factorized_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralTridiagonalConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto tridiagonal = uni20::krylov::real_general_tridiagonal_from_dense(matrix);
  Binary128 const norm = uni20::krylov::real_general_tridiagonal_one_norm(tridiagonal);
  EXPECT_TRUE(abs_error(norm, Binary128{1}) <= tolerance());

  auto factorization = uni20::krylov::real_general_tridiagonal_factorization(tridiagonal);
  Binary128 const factorized_rcond =
      uni20::krylov::real_general_tridiagonal_one_norm_reciprocal_condition_number(factorization, norm);
  Binary128 const direct_rcond =
      uni20::krylov::real_general_tridiagonal_one_norm_reciprocal_condition_number(tridiagonal);

  expect_value_underflows_to_double_zero(factorized_rcond);
  expect_value_underflows_to_double_zero(direct_rcond);
  EXPECT_TRUE(abs_error(factorized_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(direct_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralTridiagonalRefinedSolveUsesMplapackGtrfsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{0.25} * tiny;
  matrix[0, 1] = Binary128{2} * tiny;
  matrix[1, 0] = Binary128{3} * tiny;
  matrix[1, 1] = Binary128{4} * tiny;
  matrix[1, 2] = Binary128{-1} * tiny;
  matrix[2, 1] = Binary128{0.5} * tiny;
  matrix[2, 2] = Binary128{5} * tiny;

  auto expected_solution = zero_matrix<Binary128>(3, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto result = uni20::krylov::real_general_tridiagonal_refined_solve(
      uni20::krylov::real_general_tridiagonal_from_dense(matrix), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 3);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_GE(result.forward_error_bounds[0], Binary128{});
  EXPECT_GE(result.backward_error_bounds[0], Binary128{});
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(result.solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto reconstructed = multiply_for_test(matrix, result.solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(reconstructed[row, 0], right_hand_side[row, 0]) <= tiny * Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralTridiagonalExpertSolveUsesMplapackGtsvxBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto expected_solution = zero_matrix<Binary128>(2, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{2};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  EXPECT_TRUE(abs_error(right_hand_side[0, 0], Binary128{1}) <= tolerance());
  expect_value_underflows_to_double_zero(std::abs(right_hand_side[1, 0]));

  auto result = uni20::krylov::real_general_tridiagonal_expert_linear_solve(
      uni20::krylov::real_general_tridiagonal_from_dense(matrix), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  ASSERT_EQ(result.factorization.pivot_rows.size(), 2);
  EXPECT_TRUE(result.reciprocal_condition_below_machine_precision);
  expect_value_underflows_to_double_zero(result.reciprocal_condition);
  EXPECT_TRUE(abs_error(result.reciprocal_condition / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(result.solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralBandReciprocalConditionNumberStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto band = uni20::krylov::real_general_band_from_dense(matrix, 0, 0);
  Binary128 const norm = uni20::krylov::real_general_band_matrix_norm(band, uni20::krylov::MatrixNorm::One);
  EXPECT_TRUE(abs_error(norm, Binary128{1}) <= tolerance());

  auto factorization = uni20::krylov::real_general_band_factorization(band);
  Binary128 const factorized_rcond =
      uni20::krylov::real_general_band_one_norm_reciprocal_condition_number(factorization, norm);
  Binary128 const direct_rcond = uni20::krylov::real_general_band_one_norm_reciprocal_condition_number(band);

  expect_value_underflows_to_double_zero(factorized_rcond);
  expect_value_underflows_to_double_zero(direct_rcond);
  EXPECT_TRUE(abs_error(factorized_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(direct_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralBandRefinedSolveUsesMplapackGbrfsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{2} * tiny;
  matrix[1, 0] = Binary128{0.5} * tiny;
  matrix[0, 1] = Binary128{0.25} * tiny;
  matrix[1, 1] = Binary128{3} * tiny;
  matrix[2, 1] = Binary128{0.75} * tiny;
  matrix[1, 2] = Binary128{-0.25} * tiny;
  matrix[2, 2] = Binary128{4} * tiny;

  auto expected_solution = zero_matrix<Binary128>(3, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto result = uni20::krylov::real_general_band_refined_solve(
      uni20::krylov::real_general_band_from_dense(matrix, 1, 1), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 3);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_GE(result.forward_error_bounds[0], Binary128{});
  EXPECT_GE(result.backward_error_bounds[0], Binary128{});
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(result.solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto reconstructed = multiply_for_test(matrix, result.solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(reconstructed[row, 0], right_hand_side[row, 0]) <= tiny * Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralBandExpertSolveUsesMplapackGbsvxBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto right_hand_side = zero_matrix<Binary128>(2, 1);
  right_hand_side[0, 0] = Binary128{1};
  right_hand_side[1, 0] = tiny;
  expect_value_underflows_to_double_zero(right_hand_side[1, 0]);

  auto result = uni20::krylov::real_general_band_expert_linear_solve(
      uni20::krylov::real_general_band_from_dense(matrix, 0, 0), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factors.storage.rows(), 1);
  ASSERT_EQ(result.factors.storage.cols(), 2);
  ASSERT_EQ(result.pivot_rows.size(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_EQ(result.equilibration, 'N');
  EXPECT_TRUE(result.reciprocal_condition_below_machine_precision);
  expect_value_underflows_to_double_zero(result.reciprocal_condition);
  EXPECT_TRUE(abs_error(result.reciprocal_condition / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(matrix, result.solution);
  EXPECT_TRUE(abs_error(reconstructed[0, 0], right_hand_side[0, 0]) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 0], right_hand_side[1, 0]) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralBandEquilibrationPreservesBinary128OnlyScales)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto band = uni20::krylov::real_general_band_from_dense(matrix, 0, 0);
  auto result = uni20::krylov::real_general_band_equilibration(band);
  auto power_result = uni20::krylov::real_general_band_power_of_two_equilibration(band);

  for (auto const& equilibration : {result, power_result})
  {
    ASSERT_EQ(equilibration.row_scale.size(), 2);
    ASSERT_EQ(equilibration.column_scale.size(), 2);
    EXPECT_TRUE(equilibration.row_scale[0] > Binary128{});
    EXPECT_FALSE(std::isfinite(static_cast<double>(equilibration.row_scale[0])));
    EXPECT_TRUE(abs_error(equilibration.row_scale[0] * tiny, Binary128{1}) <= tolerance());
    EXPECT_TRUE(abs_error(equilibration.row_scale[1], Binary128{1}) <= tolerance());
    EXPECT_TRUE(abs_error(equilibration.column_scale[0], Binary128{1}) <= tolerance());
    EXPECT_TRUE(abs_error(equilibration.column_scale[1], Binary128{1}) <= tolerance());
    EXPECT_TRUE(abs_error(equilibration.row_condition, tiny) <= tiny * Binary128{1000} * tolerance());
    EXPECT_TRUE(abs_error(equilibration.column_condition, Binary128{1}) <= tolerance());
    EXPECT_TRUE(abs_error(equilibration.max_abs, Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, TridiagonalHelperAcceptsBackendReal)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::krylov::symmetric_tridiagonal_matrix<Binary128> matrix(3);
  matrix.diagonal[0] = Binary128{1};
  matrix.diagonal[1] = Binary128{1} + delta;
  matrix.diagonal[2] = Binary128{2};
  matrix.offdiagonal[0] = delta;
  matrix.offdiagonal[1] = Binary128{3};

  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 1.0);
  EXPECT_TRUE((matrix[1, 1] > Binary128{1}));
  EXPECT_TRUE(abs_error((matrix.view()[0, 1]), delta) <= tolerance());
  EXPECT_TRUE(abs_error((matrix.view()[1, 0]), delta) <= tolerance());
  EXPECT_TRUE(abs_error((matrix.view()[1, 2]), Binary128{3}) <= tolerance());
  EXPECT_TRUE(abs_error((matrix.view()[0, 2]), Binary128{}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralEquilibrationPreservesBinary128OnlyScales)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto result = uni20::krylov::real_equilibration(matrix);

  ASSERT_EQ(result.row_scale.size(), 2);
  ASSERT_EQ(result.column_scale.size(), 2);
  EXPECT_TRUE(result.row_scale[0] > Binary128{});
  EXPECT_FALSE(std::isfinite(static_cast<double>(result.row_scale[0])));
  EXPECT_TRUE(abs_error(result.row_scale[0] * tiny, Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.row_scale[1], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.column_scale[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.column_scale[1], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.row_condition, tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.column_condition, Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.max_abs, Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, RefinedLinearSolvePreservesBinary128OnlyRightHandSide)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2};

  auto result = uni20::krylov::real_refined_linear_solve(matrix, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_EQ(static_cast<double>(result.solution[0, 0]), 0.0);
  EXPECT_TRUE(abs_error(result.solution[0, 0], tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{2}) <= tolerance());
  EXPECT_TRUE(result.forward_error_bounds[0] >= Binary128{});
  EXPECT_TRUE(result.backward_error_bounds[0] >= Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, RealSymmetricEigensystemResolvesDenseGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = offdiagonal;
  matrix[1, 0] = Binary128{};
  matrix[1, 1] = Binary128{1} + delta;

  auto result = uni20::krylov::real_symmetric_eigensystem(matrix, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const residual0 = x0 + offdiagonal * x1 - lambda * x0;
    Binary128 const residual1 = offdiagonal * x0 + (Binary128{1} + delta) * x1 - lambda * x1;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    EXPECT_TRUE(residual_norm <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealSymmetricDivideAndConquerEigensystemResolvesDenseGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = offdiagonal;
  matrix[1, 0] = Binary128{};
  matrix[1, 1] = Binary128{1} + delta;

  auto result = uni20::krylov::real_symmetric_eigensystem_divide_and_conquer(matrix, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const residual0 = x0 + offdiagonal * x1 - lambda * x0;
    Binary128 const residual1 = offdiagonal * x0 + (Binary128{1} + delta) * x1 - lambda * x1;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    EXPECT_TRUE(residual_norm <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedRealSymmetricEigensystemResolvesDenseGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 2] = Binary128{2};

  auto result = uni20::krylov::real_symmetric_eigensystem_index_range(matrix, 0, 1, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[1]), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], Binary128{1} + delta) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 residual_norm{};
    Binary128 vector_norm{};
    for (uni20::index_type row = 0; row < result.eigenvectors.rows(); ++row)
    {
      Binary128 const diagonal = row == 0 ? Binary128{1} : (row == 1 ? Binary128{1} + delta : Binary128{2});
      Binary128 const residual = diagonal * result.eigenvectors[row, col] - lambda * result.eigenvectors[row, col];
      residual_norm += residual * residual;
      vector_norm += result.eigenvectors[row, col] * result.eigenvectors[row, col];
    }
    EXPECT_TRUE(std::sqrt(residual_norm) <= tolerance());
    EXPECT_TRUE(abs_error(std::sqrt(vector_norm), Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealGeneralizedSymmetricEigensystemResolvesMetricGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{2} * (Binary128{1} + delta);
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 2.0);

  auto metric = zero_matrix<Binary128>(2, 2);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{2};

  auto result = uni20::krylov::real_generalized_symmetric_eigensystem(matrix, metric, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[1]), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(result.eigenvalues[1] - result.eigenvalues[0] > delta / Binary128{2});
  EXPECT_TRUE(abs_error(result.eigenvalues[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], Binary128{1} + delta) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const residual0 = x0 - lambda * x0;
    Binary128 const residual1 = Binary128{2} * (Binary128{1} + delta) * x1 - lambda * Binary128{2} * x1;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Binary128 const metric_norm = x0 * x0 + Binary128{2} * x1 * x1;
    EXPECT_TRUE(residual_norm <= tolerance());
    EXPECT_TRUE(abs_error(metric_norm, Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest,
     RealGeneralizedSymmetricDivideAndConquerEigensystemResolvesMetricGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{2} * (Binary128{1} + delta);
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 2.0);

  auto metric = zero_matrix<Binary128>(2, 2);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{2};

  auto result = uni20::krylov::real_generalized_symmetric_eigensystem_divide_and_conquer(matrix, metric, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[1]), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(result.eigenvalues[1] - result.eigenvalues[0] > delta / Binary128{2});
  EXPECT_TRUE(abs_error(result.eigenvalues[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], Binary128{1} + delta) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const residual0 = x0 - lambda * x0;
    Binary128 const residual1 = Binary128{2} * (Binary128{1} + delta) * x1 - lambda * Binary128{2} * x1;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Binary128 const metric_norm = x0 * x0 + Binary128{2} * x1 * x1;
    EXPECT_TRUE(residual_norm <= tolerance());
    EXPECT_TRUE(abs_error(metric_norm, Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest,
     SelectedRealGeneralizedSymmetricEigensystemResolvesMetricGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{2} * (Binary128{1} + delta);
  matrix[2, 2] = Binary128{6};
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 2.0);

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{2};
  metric[2, 2] = Binary128{3};

  auto result = uni20::krylov::real_generalized_symmetric_eigensystem_index_range(matrix, metric, 0, 1, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.eigenvalues[1]), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(result.eigenvalues[1] - result.eigenvalues[0] > delta / Binary128{2});
  EXPECT_TRUE(abs_error(result.eigenvalues[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], Binary128{1} + delta) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 residual_norm{};
    Binary128 metric_norm{};
    for (uni20::index_type row = 0; row < result.eigenvectors.rows(); ++row)
    {
      Binary128 const matrix_diagonal =
          row == 0 ? Binary128{1} : (row == 1 ? Binary128{2} * (Binary128{1} + delta) : Binary128{6});
      Binary128 const metric_diagonal = Binary128{1} + static_cast<Binary128>(row);
      Binary128 const value = result.eigenvectors[row, col];
      Binary128 const residual = matrix_diagonal * value - lambda * metric_diagonal * value;
      residual_norm += residual * residual;
      metric_norm += metric_diagonal * value * value;
    }
    EXPECT_TRUE(std::sqrt(residual_norm) <= tolerance());
    EXPECT_TRUE(abs_error(metric_norm, Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealSingularValueDecompositionResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 1] = Binary128{1};

  auto result = uni20::krylov::real_singular_value_decomposition(matrix, true);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.left_singular_vectors.rows(), 2);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.singular_values[1]), 1.0);
  EXPECT_TRUE(result.singular_values[0] > result.singular_values[1]);
  EXPECT_TRUE(result.singular_values[0] - result.singular_values[1] > delta / Binary128{2});
  EXPECT_TRUE(abs_error(result.singular_values[0], Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(result.singular_values[1], Binary128{1}) <= tolerance());

  auto sigma = zero_matrix<Binary128>(2, 2);
  sigma[0, 0] = result.singular_values[0];
  sigma[1, 1] = result.singular_values[1];
  auto reconstructed = multiply_for_test(multiply_for_test(result.left_singular_vectors, sigma),
                                         result.right_singular_vectors_transpose);
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(reconstructed[row, col], matrix[row, col]) <= tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealDivideAndConquerSvdResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 1] = Binary128{1};

  auto result = uni20::krylov::real_singular_value_decomposition_divide_and_conquer(matrix, true);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.left_singular_vectors.rows(), 2);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.singular_values[1]), 1.0);
  EXPECT_TRUE(result.singular_values[0] > result.singular_values[1]);
  EXPECT_TRUE(result.singular_values[0] - result.singular_values[1] > delta / Binary128{2});
  EXPECT_TRUE(abs_error(result.singular_values[0], Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(result.singular_values[1], Binary128{1}) <= tolerance());

  auto sigma = zero_matrix<Binary128>(2, 2);
  sigma[0, 0] = result.singular_values[0];
  sigma[1, 1] = result.singular_values[1];
  auto reconstructed = multiply_for_test(multiply_for_test(result.left_singular_vectors, sigma),
                                         result.right_singular_vectors_transpose);
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(reconstructed[row, col], matrix[row, col]) <= tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedRealSvdResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 2] = Binary128{2};

  auto values_only = uni20::krylov::real_singular_value_decomposition_index_range(matrix, 1, 2, false);
  auto result = uni20::krylov::real_singular_value_decomposition_index_range(matrix, 1, 2, true);

  ASSERT_EQ(values_only.singular_values.size(), 2);
  EXPECT_EQ(static_cast<double>(values_only.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(values_only.singular_values[1]), 1.0);
  EXPECT_TRUE(values_only.singular_values[0] > values_only.singular_values[1]);
  EXPECT_TRUE(abs_error(values_only.singular_values[0] - values_only.singular_values[1], delta) <= tolerance());
  EXPECT_EQ(values_only.left_singular_vectors.rows(), 0);
  EXPECT_EQ(values_only.right_singular_vectors_transpose.cols(), 0);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.left_singular_vectors.rows(), 3);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.cols(), 3);
  EXPECT_EQ(static_cast<double>(result.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.singular_values[1]), 1.0);
  EXPECT_TRUE(result.singular_values[0] > result.singular_values[1]);
  EXPECT_TRUE(abs_error(result.singular_values[0] - result.singular_values[1], delta) <= tolerance());

  auto sigma = zero_matrix<Binary128>(2, 2);
  sigma[0, 0] = result.singular_values[0];
  sigma[1, 1] = result.singular_values[1];
  auto selected_contribution = multiply_for_test(multiply_for_test(result.left_singular_vectors, sigma),
                                                 result.right_singular_vectors_transpose);
  EXPECT_TRUE(abs_error(selected_contribution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[0, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[0, 2], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[1, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[1, 1], Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[1, 2], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[2, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[2, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[2, 2], Binary128{}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = tiny;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2} * tiny;

  auto solution = uni20::krylov::real_symmetric_positive_definite_solve(matrix, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteFactorizationSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = tiny;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2} * tiny;

  auto factorization = uni20::krylov::real_symmetric_positive_definite_factorization(matrix);
  auto solution = uni20::krylov::real_symmetric_positive_definite_solve(factorization, rhs);

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  EXPECT_EQ(factorization.triangle, uni20::krylov::MatrixFill::Upper);
  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteBandSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{2} * tiny;
  matrix[0, 1] = Binary128{0.25} * tiny;
  matrix[1, 0] = matrix[0, 1];
  matrix[1, 1] = Binary128{3} * tiny;
  matrix[1, 2] = Binary128{-0.25} * tiny;
  matrix[2, 1] = matrix[1, 2];
  matrix[2, 2] = Binary128{4} * tiny;

  auto expected_solution = zero_matrix<Binary128>(3, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto upper_band =
      uni20::krylov::real_symmetric_positive_definite_band_from_dense(matrix, 1, uni20::krylov::MatrixFill::Upper);
  auto direct_solution = uni20::krylov::real_symmetric_positive_definite_band_solve(upper_band, right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(direct_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto lower_factorization = uni20::krylov::real_symmetric_positive_definite_band_factorization(
      uni20::krylov::real_symmetric_positive_definite_band_from_dense(matrix, 1, uni20::krylov::MatrixFill::Lower));
  auto factorized_solution =
      uni20::krylov::real_symmetric_positive_definite_band_solve(lower_factorization, right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(factorized_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteBandConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto band = uni20::krylov::real_symmetric_positive_definite_band_from_dense(matrix, 0);
  Binary128 const norm = uni20::krylov::real_symmetric_positive_definite_band_one_norm(band);
  EXPECT_TRUE(abs_error(norm, Binary128{1}) <= tolerance());

  auto factorization = uni20::krylov::real_symmetric_positive_definite_band_factorization(band);
  Binary128 const factorized_rcond =
      uni20::krylov::real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number(factorization, norm);
  Binary128 const direct_rcond =
      uni20::krylov::real_symmetric_positive_definite_band_one_norm_reciprocal_condition_number(band);

  expect_value_underflows_to_double_zero(factorized_rcond);
  expect_value_underflows_to_double_zero(direct_rcond);
  EXPECT_TRUE(abs_error(factorized_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(direct_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteBandRefinedSolveUsesMplapackPbrfs)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{2} * tiny;
  matrix[0, 1] = Binary128{0.25} * tiny;
  matrix[1, 0] = matrix[0, 1];
  matrix[1, 1] = Binary128{3} * tiny;
  matrix[1, 2] = Binary128{-0.25} * tiny;
  matrix[2, 1] = matrix[1, 2];
  matrix[2, 2] = Binary128{4} * tiny;

  auto expected_solution = zero_matrix<Binary128>(3, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto result = uni20::krylov::real_symmetric_positive_definite_band_refined_solve(
      uni20::krylov::real_symmetric_positive_definite_band_from_dense(matrix, 1), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 3);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_GE(result.forward_error_bounds[0], Binary128{});
  EXPECT_GE(result.backward_error_bounds[0], Binary128{});
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(result.solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto reconstructed = multiply_for_test(matrix, result.solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(reconstructed[row, 0], right_hand_side[row, 0]) <= tiny * Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteBandExpertSolveUsesMplapackPbsvx)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto expected_solution = zero_matrix<Binary128>(2, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{2};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  EXPECT_TRUE(abs_error(right_hand_side[0, 0], Binary128{1}) <= tolerance());
  expect_value_underflows_to_double_zero(std::abs(right_hand_side[1, 0]));

  auto result = uni20::krylov::real_symmetric_positive_definite_band_expert_linear_solve(
      uni20::krylov::real_symmetric_positive_definite_band_from_dense(matrix, 0), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factorization.factors.storage.rows(), 1);
  ASSERT_EQ(result.factorization.factors.storage.cols(), 2);
  ASSERT_EQ(result.scale.size(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_EQ(result.equilibration, 'N');
  EXPECT_TRUE(result.reciprocal_condition_below_machine_precision);
  expect_value_underflows_to_double_zero(result.reciprocal_condition);
  EXPECT_TRUE(abs_error(result.reciprocal_condition / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(result.solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteTridiagonalSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(4, 4);
  matrix[0, 0] = Binary128{2} * tiny;
  matrix[0, 1] = Binary128{0.25} * tiny;
  matrix[1, 0] = matrix[0, 1];
  matrix[1, 1] = Binary128{3} * tiny;
  matrix[1, 2] = Binary128{-0.125} * tiny;
  matrix[2, 1] = matrix[1, 2];
  matrix[2, 2] = Binary128{4} * tiny;
  matrix[2, 3] = Binary128{0.5} * tiny;
  matrix[3, 2] = matrix[2, 3];
  matrix[3, 3] = Binary128{5} * tiny;

  auto expected_solution = zero_matrix<Binary128>(4, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  expected_solution[3, 0] = Binary128{-4};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto tridiagonal = uni20::krylov::real_symmetric_positive_definite_tridiagonal_from_dense(matrix);
  ASSERT_EQ(tridiagonal.diagonal.size(), 4);
  ASSERT_EQ(tridiagonal.offdiagonal.size(), 3);

  auto direct_solution =
      uni20::krylov::real_symmetric_positive_definite_tridiagonal_solve(tridiagonal, right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(direct_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto factorization = uni20::krylov::real_symmetric_positive_definite_tridiagonal_factorization(tridiagonal);
  auto factorized_solution =
      uni20::krylov::real_symmetric_positive_definite_tridiagonal_solve(factorization, right_hand_side);
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(factorized_solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteTridiagonalConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto tridiagonal = uni20::krylov::real_symmetric_positive_definite_tridiagonal_from_dense(matrix);
  Binary128 const norm = uni20::krylov::real_symmetric_positive_definite_tridiagonal_one_norm(tridiagonal);
  EXPECT_TRUE(abs_error(norm, Binary128{1}) <= tolerance());

  auto factorization = uni20::krylov::real_symmetric_positive_definite_tridiagonal_factorization(tridiagonal);
  Binary128 const factorized_rcond =
      uni20::krylov::real_symmetric_positive_definite_tridiagonal_one_norm_reciprocal_condition_number(factorization,
                                                                                                       norm);
  Binary128 const direct_rcond =
      uni20::krylov::real_symmetric_positive_definite_tridiagonal_one_norm_reciprocal_condition_number(tridiagonal);

  expect_value_underflows_to_double_zero(factorized_rcond);
  expect_value_underflows_to_double_zero(direct_rcond);
  EXPECT_TRUE(abs_error(factorized_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(direct_rcond / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteTridiagonalRefinedSolveUsesMplapackPtrfs)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(4, 4);
  matrix[0, 0] = Binary128{2} * tiny;
  matrix[0, 1] = Binary128{0.25} * tiny;
  matrix[1, 0] = matrix[0, 1];
  matrix[1, 1] = Binary128{3} * tiny;
  matrix[1, 2] = Binary128{-0.125} * tiny;
  matrix[2, 1] = matrix[1, 2];
  matrix[2, 2] = Binary128{4} * tiny;
  matrix[2, 3] = Binary128{0.5} * tiny;
  matrix[3, 2] = matrix[2, 3];
  matrix[3, 3] = Binary128{5} * tiny;

  auto expected_solution = zero_matrix<Binary128>(4, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{-2};
  expected_solution[2, 0] = Binary128{3};
  expected_solution[3, 0] = Binary128{-4};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    expect_value_underflows_to_double_zero(std::abs(right_hand_side[row, 0]));
  }

  auto result = uni20::krylov::real_symmetric_positive_definite_tridiagonal_refined_solve(
      uni20::krylov::real_symmetric_positive_definite_tridiagonal_from_dense(matrix), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 4);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_GE(result.forward_error_bounds[0], Binary128{});
  EXPECT_GE(result.backward_error_bounds[0], Binary128{});
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(result.solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }

  auto reconstructed = multiply_for_test(matrix, result.solution);
  for (uni20::index_type row = 0; row < right_hand_side.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(reconstructed[row, 0], right_hand_side[row, 0]) <= tiny * Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteTridiagonalExpertSolveUsesMplapackPtsvx)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto expected_solution = zero_matrix<Binary128>(2, 1);
  expected_solution[0, 0] = Binary128{1};
  expected_solution[1, 0] = Binary128{2};
  DenseMatrix<Binary128> right_hand_side = multiply_for_test(matrix, expected_solution);
  EXPECT_TRUE(abs_error(right_hand_side[0, 0], Binary128{1}) <= tolerance());
  expect_value_underflows_to_double_zero(std::abs(right_hand_side[1, 0]));

  auto result = uni20::krylov::real_symmetric_positive_definite_tridiagonal_expert_linear_solve(
      uni20::krylov::real_symmetric_positive_definite_tridiagonal_from_dense(matrix), right_hand_side);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factorization.factors.diagonal.size(), 2);
  ASSERT_EQ(result.factorization.factors.offdiagonal.size(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_TRUE(result.reciprocal_condition_below_machine_precision);
  expect_value_underflows_to_double_zero(result.reciprocal_condition);
  EXPECT_TRUE(abs_error(result.reciprocal_condition / tiny, Binary128{1}) <= Binary128{1000} * tolerance());
  for (uni20::index_type row = 0; row < expected_solution.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(result.solution[row, 0], expected_solution[row, 0]) <= Binary128{1000} * tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, PivotedCholeskyKeepsPivotBelowDoubleMinimumWithBinary128Tolerance)
{
  Binary128 const tiny = below_double_minimum_value();
  Binary128 const tinier = tiny * tiny;
  expect_value_underflows_to_double_zero(tiny);
  expect_value_underflows_to_double_zero(tinier);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tinier;

  auto factorization = uni20::krylov::real_pivoted_cholesky_factorization(matrix, tinier / Binary128{2});

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  ASSERT_EQ(factorization.pivot_order.size(), 2);
  EXPECT_EQ(factorization.rank, 2);
  EXPECT_FALSE(factorization.rank_deficient);
  EXPECT_EQ(factorization.pivot_order[0], 0);
  EXPECT_EQ(factorization.pivot_order[1], 1);

  Binary128 const small_factor = factorization.factors[1, 1];
  EXPECT_GT(small_factor, Binary128{});
  EXPECT_EQ(static_cast<double>(small_factor), 0.0);
  EXPECT_TRUE(abs_error(small_factor * small_factor, tinier) <= tinier * Binary128{1000} * tolerance());

  auto upper = zero_matrix<Binary128>(2, 2);
  upper[0, 0] = factorization.factors[0, 0];
  upper[0, 1] = factorization.factors[0, 1];
  upper[1, 1] = factorization.factors[1, 1];
  auto reconstructed = multiply_for_test(transpose(upper), upper);
  EXPECT_TRUE(abs_error(reconstructed[0, 0], matrix[0, 0]) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[0, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 1], tinier) <= tinier * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteExpertSolveUsesMplapackPosvxBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2};

  auto result = uni20::krylov::real_symmetric_positive_definite_expert_solve(matrix, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factors.rows(), 2);
  ASSERT_EQ(result.factors.cols(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  ASSERT_EQ(result.scale.size(), 2);
  EXPECT_EQ(result.equilibration, 'N');
  EXPECT_TRUE(result.reciprocal_condition_below_machine_precision);
  EXPECT_GT(result.reciprocal_condition, Binary128{});
  EXPECT_EQ(static_cast<double>(result.reciprocal_condition), 0.0);
  EXPECT_TRUE(abs_error(result.reciprocal_condition, tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteEquilibrationPreservesBinary128OnlyScales)
{
  Binary128 const tiny = below_double_minimum_value();
  Binary128 const tinier = tiny * tiny;
  expect_value_underflows_to_double_zero(tinier);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tinier;
  matrix[1, 1] = Binary128{1};

  auto result = uni20::krylov::real_symmetric_positive_definite_equilibration(matrix);

  ASSERT_EQ(result.scale.size(), 2);
  EXPECT_TRUE(result.scale[0] > Binary128{});
  EXPECT_FALSE(std::isfinite(static_cast<double>(result.scale[0])));
  EXPECT_TRUE(abs_error(result.scale[0] * tiny, Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.scale[1], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.scale_condition, tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.max_abs, Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteRefinedSolvePreservesBinary128OnlyRightHandSide)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2};

  auto result = uni20::krylov::real_symmetric_positive_definite_refined_solve(matrix, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_EQ(static_cast<double>(result.solution[0, 0]), 0.0);
  EXPECT_TRUE(abs_error(result.solution[0, 0], tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{2}) <= tolerance());
  EXPECT_TRUE(result.forward_error_bounds[0] >= Binary128{});
  EXPECT_TRUE(result.backward_error_bounds[0] >= Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  Binary128 const rcond = uni20::krylov::real_symmetric_positive_definite_one_norm_reciprocal_condition_number(matrix);

  EXPECT_GT(rcond, Binary128{});
  EXPECT_EQ(static_cast<double>(rcond), 0.0);
  EXPECT_TRUE(abs_error(rcond, tiny) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteFactorizedConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto factorization = uni20::krylov::real_symmetric_positive_definite_factorization(matrix);
  Binary128 const rcond =
      uni20::krylov::real_symmetric_positive_definite_one_norm_reciprocal_condition_number(factorization, Binary128{1});

  EXPECT_GT(rcond, Binary128{});
  EXPECT_EQ(static_cast<double>(rcond), 0.0);
  EXPECT_TRUE(abs_error(rcond, tiny) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricPositiveDefiniteInverseAcceptsPivotBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto inverse = uni20::krylov::real_symmetric_positive_definite_inverse(matrix);
  auto product = multiply_for_test(matrix, inverse);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  EXPECT_TRUE((inverse[0, 0] > Binary128{1}));
  EXPECT_FALSE(std::isfinite(static_cast<double>(inverse[0, 0])));
  EXPECT_TRUE(abs_error(inverse[0, 0] * tiny, Binary128{1}) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(inverse[1, 1], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(product[0, 0], Binary128{1}) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(product[1, 1], Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricIndefiniteSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = -tiny;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = -Binary128{2} * tiny;

  auto solution = uni20::krylov::real_symmetric_indefinite_solve(matrix, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricIndefiniteFactorizationSolveAcceptsPivotsBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = -tiny;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = -Binary128{2} * tiny;

  auto factorization = uni20::krylov::real_symmetric_indefinite_factorization(matrix);
  auto solution = uni20::krylov::real_symmetric_indefinite_solve(factorization, rhs);

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  ASSERT_EQ(factorization.pivot_blocks.size(), 2);
  EXPECT_EQ(factorization.triangle, uni20::krylov::MatrixFill::Upper);
  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricIndefiniteRefinedSolvePreservesBinary128OnlyRightHandSide)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{-1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{-2};

  auto result = uni20::krylov::real_symmetric_indefinite_refined_solve(matrix, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_EQ(static_cast<double>(result.solution[0, 0]), 0.0);
  EXPECT_TRUE(abs_error(result.solution[0, 0], tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{2}) <= tolerance());
  EXPECT_TRUE(result.forward_error_bounds[0] >= Binary128{});
  EXPECT_TRUE(result.backward_error_bounds[0] >= Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricIndefiniteInverseAcceptsPivotBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = -tiny;

  auto inverse = uni20::krylov::real_symmetric_indefinite_inverse(matrix);
  auto product = multiply_for_test(matrix, inverse);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  EXPECT_TRUE((inverse[0, 0] > Binary128{1}));
  EXPECT_FALSE(std::isfinite(static_cast<double>(inverse[0, 0])));
  EXPECT_TRUE((inverse[1, 1] < Binary128{-1}));
  EXPECT_FALSE(std::isfinite(static_cast<double>(inverse[1, 1])));
  EXPECT_TRUE(abs_error(inverse[0, 0] * tiny, Binary128{1}) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(inverse[1, 1] * tiny, Binary128{-1}) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(product[0, 0], Binary128{1}) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(product[1, 1], Binary128{1}) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricIndefiniteExpertSolveUsesMplapackSysvxBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = -tiny;
  matrix[1, 1] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = -tiny;
  rhs[1, 0] = Binary128{2};

  auto result = uni20::krylov::real_symmetric_indefinite_expert_solve(matrix, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factors.rows(), 2);
  ASSERT_EQ(result.factors.cols(), 2);
  ASSERT_EQ(result.pivot_blocks.size(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_TRUE(result.reciprocal_condition_below_machine_precision);
  EXPECT_GT(result.reciprocal_condition, Binary128{});
  EXPECT_EQ(static_cast<double>(result.reciprocal_condition), 0.0);
  EXPECT_TRUE(abs_error(result.reciprocal_condition, tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricIndefiniteConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = -tiny;
  matrix[1, 1] = Binary128{1};

  Binary128 const rcond = uni20::krylov::real_symmetric_indefinite_one_norm_reciprocal_condition_number(matrix);

  EXPECT_GT(rcond, Binary128{});
  EXPECT_EQ(static_cast<double>(rcond), 0.0);
  EXPECT_TRUE(abs_error(rcond, tiny) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricIndefiniteFactorizedConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = -tiny;
  matrix[1, 1] = Binary128{1};

  auto factorization = uni20::krylov::real_symmetric_indefinite_factorization(matrix);
  Binary128 const rcond =
      uni20::krylov::real_symmetric_indefinite_one_norm_reciprocal_condition_number(factorization, Binary128{1});

  EXPECT_GT(rcond, Binary128{});
  EXPECT_EQ(static_cast<double>(rcond), 0.0);
  EXPECT_TRUE(abs_error(rcond, tiny) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, TriangularSolveAcceptsDiagonalBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2};

  auto solution = uni20::krylov::real_triangular_solve(matrix, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, TriangularRefinedSolveAcceptsDiagonalBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = tiny;
  rhs[1, 0] = Binary128{2};

  auto result = uni20::krylov::real_triangular_refined_solve(matrix, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{2}) <= tolerance());
  EXPECT_TRUE(result.forward_error_bounds[0] >= Binary128{});
  EXPECT_TRUE(result.backward_error_bounds[0] >= Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, TriangularInverseAcceptsDiagonalBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto inverse = uni20::krylov::real_triangular_inverse(matrix);
  auto product = multiply_for_test(matrix, inverse);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  EXPECT_TRUE((inverse[0, 0] > Binary128{1}));
  EXPECT_FALSE(std::isfinite(static_cast<double>(inverse[0, 0])));
  EXPECT_TRUE(abs_error(inverse[0, 0] * tiny, Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(product[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(product[1, 1], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(product[0, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(product[1, 0], Binary128{}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, TriangularConditionEstimateStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  Binary128 const rcond = uni20::krylov::real_triangular_one_norm_reciprocal_condition_number(matrix);

  EXPECT_GT(rcond, Binary128{});
  EXPECT_EQ(static_cast<double>(rcond), 0.0);
  EXPECT_TRUE(abs_error(rcond, tiny) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SylvesterSolvePreservesRightHandSideBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto left = zero_matrix<Binary128>(1, 1);
  left[0, 0] = Binary128{1};

  auto right = zero_matrix<Binary128>(1, 1);
  right[0, 0] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(1, 1);
  rhs[0, 0] = tiny;
  EXPECT_EQ(static_cast<double>(rhs[0, 0]), 0.0);

  auto result = uni20::krylov::real_sylvester_solve(left, right, rhs);

  ASSERT_EQ(result.solution.rows(), 1);
  ASSERT_EQ(result.solution.cols(), 1);
  EXPECT_TRUE(abs_error(result.solution[0, 0], tiny / Binary128{2}) <= tiny * tolerance());
  EXPECT_TRUE(abs_error(result.scale, Binary128{1}) <= tolerance());
  EXPECT_FALSE(result.separation_perturbed);
}

TEST(MplapackBinary128DenseSubspaceTest, RealQrFactorizationPreservesColumnNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 1);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 0] = Binary128{-1};

  auto result = uni20::krylov::real_qr_factorization(matrix);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(result.q.rows(), 2);
  ASSERT_EQ(result.q.cols(), 1);
  ASSERT_EQ(result.r.rows(), 1);
  ASSERT_EQ(result.r.cols(), 1);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(std::abs(result.r[0, 0]) > Binary128{1});
  EXPECT_TRUE(abs_error(std::abs(result.r[0, 0]), expected_norm) <= tolerance());

  Binary128 const q_norm = result.q[0, 0] * result.q[0, 0] + result.q[1, 0] * result.q[1, 0];
  EXPECT_TRUE(abs_error(q_norm, Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(result.q, result.r);
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(reconstructed[row, 0], matrix[row, 0]) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealQrFactorApplicationPreservesColumnNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 1);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 0] = Binary128{-1};

  auto compact = uni20::krylov::real_compact_qr_factorization(matrix);
  auto rotated = uni20::krylov::apply_real_qr_factor(compact, matrix, uni20::krylov::MatrixSide::Left,
                                                     uni20::krylov::MatrixTranspose::Transpose);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(compact.rank, 1);
  ASSERT_EQ(compact.tau.size(), 1);
  ASSERT_EQ(rotated.rows(), 2);
  ASSERT_EQ(rotated.cols(), 1);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(abs_error(std::abs(rotated[0, 0]), expected_norm) <= tolerance());
  EXPECT_TRUE(abs_error(rotated[1, 0], Binary128{}) <= tolerance());

  auto identity = identity_matrix(2);
  auto q = uni20::krylov::apply_real_qr_factor(compact, identity, uni20::krylov::MatrixSide::Left);
  auto recovered_identity = uni20::krylov::apply_real_qr_factor(compact, q, uni20::krylov::MatrixSide::Left,
                                                                uni20::krylov::MatrixTranspose::Transpose);
  for (uni20::index_type row = 0; row < identity.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < identity.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(recovered_identity[row, col], identity[row, col]) <= tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealLqFactorizationPreservesRowNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(1, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[0, 1] = Binary128{-1};

  auto result = uni20::krylov::real_lq_factorization(matrix);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(result.l.rows(), 1);
  ASSERT_EQ(result.l.cols(), 1);
  ASSERT_EQ(result.q.rows(), 1);
  ASSERT_EQ(result.q.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(std::abs(result.l[0, 0]) > Binary128{1});
  EXPECT_TRUE(abs_error(std::abs(result.l[0, 0]), expected_norm) <= tolerance());

  Binary128 const q_norm = result.q[0, 0] * result.q[0, 0] + result.q[0, 1] * result.q[0, 1];
  EXPECT_TRUE(abs_error(q_norm, Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(result.l, result.q);
  for (uni20::index_type col = 0; col < matrix.cols(); ++col)
  {
    EXPECT_TRUE(abs_error(reconstructed[0, col], matrix[0, col]) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealLqFactorApplicationPreservesRowNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(1, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[0, 1] = Binary128{-1};

  auto compact = uni20::krylov::real_compact_lq_factorization(matrix);
  auto rotated = uni20::krylov::apply_real_lq_factor(compact, matrix, uni20::krylov::MatrixSide::Right,
                                                     uni20::krylov::MatrixTranspose::Transpose);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(compact.rank, 1);
  ASSERT_EQ(compact.tau.size(), 1);
  ASSERT_EQ(rotated.rows(), 1);
  ASSERT_EQ(rotated.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(abs_error(std::abs(rotated[0, 0]), expected_norm) <= tolerance());
  EXPECT_TRUE(abs_error(rotated[0, 1], Binary128{}) <= tolerance());

  auto identity = identity_matrix(2);
  auto q = uni20::krylov::apply_real_lq_factor(compact, identity, uni20::krylov::MatrixSide::Right);
  auto recovered_identity = uni20::krylov::apply_real_lq_factor(compact, q, uni20::krylov::MatrixSide::Right,
                                                                uni20::krylov::MatrixTranspose::Transpose);
  for (uni20::index_type row = 0; row < identity.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < identity.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(recovered_identity[row, col], identity[row, col]) <= tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealQlFactorizationPreservesColumnNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 1);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 0] = Binary128{-1};

  auto result = uni20::krylov::real_ql_factorization(matrix);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(result.q.rows(), 2);
  ASSERT_EQ(result.q.cols(), 1);
  ASSERT_EQ(result.l.rows(), 1);
  ASSERT_EQ(result.l.cols(), 1);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(std::abs(result.l[0, 0]) > Binary128{1});
  EXPECT_TRUE(abs_error(std::abs(result.l[0, 0]), expected_norm) <= tolerance());

  Binary128 const q_norm = result.q[0, 0] * result.q[0, 0] + result.q[1, 0] * result.q[1, 0];
  EXPECT_TRUE(abs_error(q_norm, Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(result.q, result.l);
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    EXPECT_TRUE(abs_error(reconstructed[row, 0], matrix[row, 0]) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealQlFactorApplicationPreservesColumnNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 1);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 0] = Binary128{-1};

  auto compact = uni20::krylov::real_compact_ql_factorization(matrix);
  auto rotated = uni20::krylov::apply_real_ql_factor(compact, matrix, uni20::krylov::MatrixSide::Left,
                                                     uni20::krylov::MatrixTranspose::Transpose);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(compact.rank, 1);
  ASSERT_EQ(compact.tau.size(), 1);
  ASSERT_EQ(rotated.rows(), 2);
  ASSERT_EQ(rotated.cols(), 1);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(abs_error(rotated[0, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(std::abs(rotated[1, 0]), expected_norm) <= tolerance());

  auto identity = identity_matrix(2);
  auto q = uni20::krylov::apply_real_ql_factor(compact, identity, uni20::krylov::MatrixSide::Left);
  auto recovered_identity = uni20::krylov::apply_real_ql_factor(compact, q, uni20::krylov::MatrixSide::Left,
                                                                uni20::krylov::MatrixTranspose::Transpose);
  for (uni20::index_type row = 0; row < identity.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < identity.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(recovered_identity[row, col], identity[row, col]) <= tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealRqFactorizationPreservesRowNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(1, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[0, 1] = Binary128{-1};

  auto result = uni20::krylov::real_rq_factorization(matrix);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(result.r.rows(), 1);
  ASSERT_EQ(result.r.cols(), 1);
  ASSERT_EQ(result.q.rows(), 1);
  ASSERT_EQ(result.q.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(std::abs(result.r[0, 0]) > Binary128{1});
  EXPECT_TRUE(abs_error(std::abs(result.r[0, 0]), expected_norm) <= tolerance());

  Binary128 const q_norm = result.q[0, 0] * result.q[0, 0] + result.q[0, 1] * result.q[0, 1];
  EXPECT_TRUE(abs_error(q_norm, Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(result.r, result.q);
  for (uni20::index_type col = 0; col < matrix.cols(); ++col)
  {
    EXPECT_TRUE(abs_error(reconstructed[0, col], matrix[0, col]) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealRqFactorApplicationPreservesRowNormBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(1, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[0, 1] = Binary128{-1};

  auto compact = uni20::krylov::real_compact_rq_factorization(matrix);
  auto rotated = uni20::krylov::apply_real_rq_factor(compact, matrix, uni20::krylov::MatrixSide::Right,
                                                     uni20::krylov::MatrixTranspose::Transpose);
  Binary128 const expected_norm = std::sqrt((Binary128{1} + delta) * (Binary128{1} + delta) + Binary128{1});

  ASSERT_EQ(compact.rank, 1);
  ASSERT_EQ(compact.tau.size(), 1);
  ASSERT_EQ(rotated.rows(), 1);
  ASSERT_EQ(rotated.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_norm), std::sqrt(2.0));
  EXPECT_TRUE(abs_error(rotated[0, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(std::abs(rotated[0, 1]), expected_norm) <= tolerance());

  auto identity = identity_matrix(2);
  auto q = uni20::krylov::apply_real_rq_factor(compact, identity, uni20::krylov::MatrixSide::Right);
  auto recovered_identity = uni20::krylov::apply_real_rq_factor(compact, q, uni20::krylov::MatrixSide::Right,
                                                                uni20::krylov::MatrixTranspose::Transpose);
  for (uni20::index_type row = 0; row < identity.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < identity.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(recovered_identity[row, col], identity[row, col]) <= tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, BidiagonalReductionReconstructsEntryBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 0] = Binary128{2};
  matrix[2, 0] = Binary128{-1};
  matrix[0, 1] = Binary128{3};
  matrix[1, 1] = Binary128{-2};
  matrix[2, 1] = Binary128{5};

  auto reduction = uni20::krylov::real_bidiagonal_reduction(matrix);
  auto q = uni20::krylov::real_bidiagonal_left_orthogonal_factor(reduction);
  auto pt = uni20::krylov::real_bidiagonal_right_orthogonal_factor_transpose(reduction);
  auto reconstructed = multiply_for_test(multiply_for_test(q, reduction.bidiagonal), pt);

  ASSERT_EQ(reduction.bidiagonal.rows(), 3);
  ASSERT_EQ(reduction.bidiagonal.cols(), 2);
  ASSERT_EQ(reduction.diagonal.size(), 2);
  ASSERT_EQ(reduction.offdiagonal.size(), 1);
  EXPECT_TRUE(reduction.upper);
  EXPECT_EQ(static_cast<double>(reconstructed[0, 0]), 1.0);
  EXPECT_TRUE((reconstructed[0, 0] > Binary128{1}));
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(reconstructed[row, col], matrix[row, col]) <= Binary128{1000} * tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, BidiagonalFactorApplicationPreservesEntryBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 2);
  matrix[0, 0] = Binary128{1} + delta;
  matrix[1, 0] = Binary128{2};
  matrix[2, 0] = Binary128{-1};
  matrix[0, 1] = Binary128{3};
  matrix[1, 1] = Binary128{-2};
  matrix[2, 1] = Binary128{5};

  auto reduction = uni20::krylov::real_bidiagonal_reduction(matrix);
  auto q = uni20::krylov::apply_real_bidiagonal_left_factor(reduction, identity_matrix(3));
  auto pt = uni20::krylov::apply_real_bidiagonal_right_factor(
      reduction, identity_matrix(2), uni20::krylov::MatrixSide::Left, uni20::krylov::MatrixTranspose::Transpose);
  auto reconstructed = multiply_for_test(multiply_for_test(q, reduction.bidiagonal), pt);
  auto recovered_q = uni20::krylov::apply_real_bidiagonal_left_factor(reduction, q, uni20::krylov::MatrixSide::Left,
                                                                      uni20::krylov::MatrixTranspose::Transpose);
  auto recovered_p = uni20::krylov::apply_real_bidiagonal_right_factor(reduction, pt, uni20::krylov::MatrixSide::Left);
  auto identity_q = identity_matrix(3);
  auto identity_p = identity_matrix(2);

  EXPECT_EQ(static_cast<double>(reconstructed[0, 0]), 1.0);
  EXPECT_TRUE((reconstructed[0, 0] > Binary128{1}));
  for (uni20::index_type row = 0; row < matrix.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(reconstructed[row, col], matrix[row, col]) <= Binary128{1000} * tolerance());
    }
  }
  for (uni20::index_type row = 0; row < recovered_q.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < recovered_q.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(recovered_q[row, col], identity_q[row, col]) <= Binary128{1000} * tolerance());
    }
  }
  for (uni20::index_type row = 0; row < recovered_p.rows(); ++row)
  {
    for (uni20::index_type col = 0; col < recovered_p.cols(); ++col)
    {
      EXPECT_TRUE(abs_error(recovered_p[row, col], identity_p[row, col]) <= Binary128{1000} * tolerance());
    }
  }
}

TEST(MplapackBinary128DenseSubspaceTest, BidiagonalSingularValuesResolveGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  std::vector<Binary128> diagonal{Binary128{1}, Binary128{1} + delta};
  std::vector<Binary128> offdiagonal{Binary128{}};

  auto singular_values = uni20::krylov::real_bidiagonal_singular_values(diagonal, offdiagonal);

  ASSERT_EQ(singular_values.size(), 2);
  EXPECT_EQ(static_cast<double>(singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(singular_values[1]), 1.0);
  EXPECT_TRUE(singular_values[0] > singular_values[1]);
  EXPECT_TRUE(abs_error(singular_values[0] - singular_values[1], delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, BidiagonalDivideAndConquerSvdResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  std::vector<Binary128> diagonal{Binary128{1}, Binary128{1} + delta};
  std::vector<Binary128> offdiagonal{Binary128{}};

  auto values_only = uni20::krylov::real_bidiagonal_svd(diagonal, offdiagonal, false);
  auto result = uni20::krylov::real_bidiagonal_svd(diagonal, offdiagonal, true);

  ASSERT_EQ(values_only.singular_values.size(), 2);
  EXPECT_EQ(static_cast<double>(values_only.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(values_only.singular_values[1]), 1.0);
  EXPECT_TRUE(values_only.singular_values[0] > values_only.singular_values[1]);
  EXPECT_TRUE(abs_error(values_only.singular_values[0] - values_only.singular_values[1], delta) <= tolerance());
  EXPECT_EQ(values_only.u.rows(), 0);
  EXPECT_EQ(values_only.vt.cols(), 0);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.u.rows(), 2);
  ASSERT_EQ(result.u.cols(), 2);
  ASSERT_EQ(result.vt.rows(), 2);
  ASSERT_EQ(result.vt.cols(), 2);
  EXPECT_EQ(static_cast<double>(result.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.singular_values[1]), 1.0);
  EXPECT_TRUE(result.singular_values[0] > result.singular_values[1]);
  EXPECT_TRUE(abs_error(result.singular_values[0] - result.singular_values[1], delta) <= tolerance());

  auto sigma = zero_matrix<Binary128>(2, 2);
  sigma[0, 0] = result.singular_values[0];
  sigma[1, 1] = result.singular_values[1];
  auto reconstructed = multiply_for_test(multiply_for_test(result.u, sigma), result.vt);
  EXPECT_TRUE(abs_error(reconstructed[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[0, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 1], Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedBidiagonalSvdResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  std::vector<Binary128> diagonal{Binary128{1}, Binary128{1} + delta, Binary128{2}};
  std::vector<Binary128> offdiagonal{Binary128{}, Binary128{}};

  auto values_only = uni20::krylov::real_bidiagonal_svd_index_range(diagonal, offdiagonal, 1, 2, false);
  auto result = uni20::krylov::real_bidiagonal_svd_index_range(diagonal, offdiagonal, 1, 2, true);

  ASSERT_EQ(values_only.singular_values.size(), 2);
  EXPECT_EQ(static_cast<double>(values_only.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(values_only.singular_values[1]), 1.0);
  EXPECT_TRUE(values_only.singular_values[0] > values_only.singular_values[1]);
  EXPECT_TRUE(abs_error(values_only.singular_values[0] - values_only.singular_values[1], delta) <= tolerance());
  EXPECT_EQ(values_only.u.rows(), 0);
  EXPECT_EQ(values_only.vt.cols(), 0);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.u.rows(), 3);
  ASSERT_EQ(result.u.cols(), 2);
  ASSERT_EQ(result.vt.rows(), 2);
  ASSERT_EQ(result.vt.cols(), 3);
  EXPECT_EQ(static_cast<double>(result.singular_values[0]), 1.0);
  EXPECT_EQ(static_cast<double>(result.singular_values[1]), 1.0);
  EXPECT_TRUE(result.singular_values[0] > result.singular_values[1]);
  EXPECT_TRUE(abs_error(result.singular_values[0] - result.singular_values[1], delta) <= tolerance());

  auto sigma = zero_matrix<Binary128>(2, 2);
  sigma[0, 0] = result.singular_values[0];
  sigma[1, 1] = result.singular_values[1];
  auto selected_contribution = multiply_for_test(multiply_for_test(result.u, sigma), result.vt);
  EXPECT_TRUE(abs_error(selected_contribution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[0, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[0, 2], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[1, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[1, 1], Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[1, 2], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[2, 0], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[2, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(selected_contribution[2, 2], Binary128{}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, HessenbergReductionPreservesBinary128OnlyEntry)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{3};
  matrix[1, 0] = Binary128{2};
  matrix[2, 0] = Binary128{1} / Binary128{5};
  matrix[0, 1] = Binary128{-1};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 1] = Binary128{4};
  matrix[0, 2] = Binary128{2};
  matrix[1, 2] = Binary128{-3};
  matrix[2, 2] = Binary128{5};
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 1.0);

  auto reduction = uni20::krylov::real_hessenberg_reduction(matrix);
  auto q = uni20::krylov::real_hessenberg_orthogonal_factor(reduction);
  auto reconstructed = multiply_for_test(multiply_for_test(q, reduction.hessenberg), transpose(q));
  auto orthogonality = multiply_for_test(transpose(q), q);
  auto identity = identity_matrix(3);

  ASSERT_EQ(reduction.hessenberg.rows(), 3);
  ASSERT_EQ(reduction.hessenberg.cols(), 3);
  ASSERT_EQ(reduction.reflectors.rows(), 3);
  ASSERT_EQ(reduction.reflectors.cols(), 3);
  ASSERT_EQ(reduction.tau.size(), 2);
  EXPECT_TRUE(abs_error(reduction.hessenberg[2, 0], Binary128{}) <= tolerance());
  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 3; ++col)
    {
      EXPECT_TRUE(abs_error(orthogonality[row, col], identity[row, col]) <= tolerance());
      EXPECT_TRUE(abs_error(reconstructed[row, col], matrix[row, col]) <= tolerance());
    }
  }
  EXPECT_EQ(static_cast<double>(reconstructed[1, 1]), 1.0);
  EXPECT_TRUE((reconstructed[1, 1] > Binary128{1}));
  EXPECT_TRUE(abs_error(reconstructed[1, 1], Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, HessenbergOrthogonalFactorApplicationPreservesTinyTarget)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{3};
  matrix[1, 0] = Binary128{2};
  matrix[2, 0] = Binary128{1} / Binary128{5};
  matrix[0, 1] = Binary128{-1};
  matrix[1, 1] = Binary128{1};
  matrix[2, 1] = Binary128{4};
  matrix[0, 2] = Binary128{2};
  matrix[1, 2] = Binary128{-3};
  matrix[2, 2] = Binary128{5};

  auto reduction = uni20::krylov::real_hessenberg_reduction(matrix);
  auto q = uni20::krylov::real_hessenberg_orthogonal_factor(reduction);

  auto target = zero_matrix<Binary128>(3, 1);
  target[1, 0] = tiny;
  target[2, 0] = Binary128{2} * tiny;

  auto applied = uni20::krylov::apply_real_hessenberg_orthogonal_factor(reduction, target);
  auto expected = multiply_for_test(q, target);
  auto recovered = uni20::krylov::apply_real_hessenberg_orthogonal_factor(
      reduction, applied, uni20::krylov::MatrixSide::Left, uni20::krylov::MatrixTranspose::Transpose);

  ASSERT_EQ(applied.rows(), 3);
  ASSERT_EQ(applied.cols(), 1);
  for (uni20::index_type row = 0; row < target.rows(); ++row)
  {
    Binary128 const scale = std::max(tiny, std::abs(expected[row, 0]));
    EXPECT_EQ(static_cast<double>(expected[row, 0]), 0.0);
    EXPECT_TRUE(abs_error(applied[row, 0], expected[row, 0]) <= scale * Binary128{1000} * tolerance());
    EXPECT_TRUE(abs_error(recovered[row, 0], target[row, 0]) <= tiny * Binary128{1000} * tolerance());
  }
  EXPECT_GT(std::abs(applied[1, 0]) + std::abs(applied[2, 0]), Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricTridiagonalReductionPreservesTinyOffdiagonal)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 0] = tiny;
  matrix[0, 1] = tiny;
  matrix[1, 1] = Binary128{2};
  matrix[2, 1] = Binary128{1} / Binary128{3};
  matrix[1, 2] = Binary128{1} / Binary128{3};
  matrix[2, 2] = Binary128{3};

  auto reduction = uni20::krylov::real_symmetric_tridiagonal_reduction(matrix, uni20::krylov::MatrixFill::Lower);

  ASSERT_EQ(reduction.diagonal.size(), 3);
  ASSERT_EQ(reduction.offdiagonal.size(), 2);
  EXPECT_TRUE(abs_error(reduction.diagonal[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(reduction.diagonal[1], Binary128{2}) <= tolerance());
  EXPECT_TRUE(abs_error(reduction.diagonal[2], Binary128{3}) <= tolerance());
  EXPECT_TRUE(abs_error(reduction.offdiagonal[0], tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(reduction.offdiagonal[1], Binary128{1} / Binary128{3}) <= tolerance());
  EXPECT_EQ(static_cast<double>(reduction.offdiagonal[0]), 0.0);
  EXPECT_TRUE(abs_error(reduction.tridiagonal[1, 0], tiny) <= tiny * Binary128{1000} * tolerance());
  EXPECT_TRUE(abs_error(reduction.tridiagonal[0, 1], tiny) <= tiny * Binary128{1000} * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricTridiagonalOrthogonalFactorApplicationPreservesTinyTarget)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{3};
  matrix[1, 0] = Binary128{-1};
  matrix[2, 0] = Binary128{2};
  matrix[0, 1] = Binary128{-1};
  matrix[1, 1] = Binary128{1};
  matrix[2, 1] = Binary128{4};
  matrix[0, 2] = Binary128{2};
  matrix[1, 2] = Binary128{4};
  matrix[2, 2] = Binary128{5};

  auto reduction = uni20::krylov::real_symmetric_tridiagonal_reduction(matrix);
  auto q = uni20::krylov::real_symmetric_tridiagonal_orthogonal_factor(reduction);

  auto target = zero_matrix<Binary128>(3, 1);
  target[1, 0] = tiny;
  target[2, 0] = Binary128{2} * tiny;

  auto applied = uni20::krylov::apply_real_symmetric_tridiagonal_orthogonal_factor(reduction, target);
  auto expected = multiply_for_test(q, target);
  auto recovered = uni20::krylov::apply_real_symmetric_tridiagonal_orthogonal_factor(
      reduction, applied, uni20::krylov::MatrixSide::Left, uni20::krylov::MatrixTranspose::Transpose);

  ASSERT_EQ(applied.rows(), 3);
  ASSERT_EQ(applied.cols(), 1);
  for (uni20::index_type row = 0; row < target.rows(); ++row)
  {
    Binary128 const scale = std::max(tiny, std::abs(expected[row, 0]));
    EXPECT_EQ(static_cast<double>(expected[row, 0]), 0.0);
    EXPECT_TRUE(abs_error(applied[row, 0], expected[row, 0]) <= scale * Binary128{1000} * tolerance());
    EXPECT_TRUE(abs_error(recovered[row, 0], target[row, 0]) <= tiny * Binary128{1000} * tolerance());
  }
  EXPECT_GT(std::abs(applied[1, 0]) + std::abs(applied[2, 0]), Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, RealPivotedQrFactorizationPreservesColumnBelowDoubleMinimum)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = tiny;
  matrix[1, 1] = Binary128{1};

  auto result = uni20::krylov::real_pivoted_qr_factorization(matrix);

  ASSERT_EQ(result.q.rows(), 2);
  ASSERT_EQ(result.q.cols(), 2);
  ASSERT_EQ(result.r.rows(), 2);
  ASSERT_EQ(result.r.cols(), 2);
  ASSERT_EQ(result.pivot_columns.size(), 2);
  EXPECT_EQ(result.pivot_columns[0], 1);
  EXPECT_EQ(result.pivot_columns[1], 0);

  auto pivoted_reconstructed = multiply_for_test(result.q, result.r);
  for (uni20::index_type col = 0; col < matrix.cols(); ++col)
  {
    std::size_t const original_col = result.pivot_columns[col];
    for (uni20::index_type row = 0; row < matrix.rows(); ++row)
    {
      Binary128 const expected = matrix[row, original_col];
      Binary128 const scale = std::max(tiny, std::abs(expected));
      EXPECT_TRUE(abs_error(pivoted_reconstructed[row, col], expected) <= scale * tolerance());
    }
  }
  EXPECT_TRUE(abs_error(pivoted_reconstructed[0, 1], tiny) <= tiny * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, RealLeastSquaresPreservesSolutionBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const one_plus_delta = Binary128{1} + delta;
  expect_gap_is_binary128_only(delta);

  auto coefficients = zero_matrix<Binary128>(2, 1);
  coefficients[0, 0] = Binary128{1};
  coefficients[1, 0] = Binary128{1};

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = one_plus_delta;
  rhs[1, 0] = one_plus_delta;
  EXPECT_EQ(static_cast<double>(rhs[0, 0]), 1.0);
  EXPECT_EQ(static_cast<double>(rhs[1, 0]), 1.0);

  auto solution = uni20::krylov::real_least_squares(coefficients, rhs);

  ASSERT_EQ(solution.rows(), 1);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_EQ(static_cast<double>(solution[0, 0]), 1.0);
  EXPECT_TRUE((solution[0, 0] > Binary128{1}));
  EXPECT_TRUE(abs_error(solution[0, 0], one_plus_delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SvdLeastSquaresKeepsTinyBinary128SingularValue)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto coefficients = zero_matrix<Binary128>(2, 2);
  coefficients[0, 0] = Binary128{1};
  coefficients[1, 1] = tiny;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = Binary128{1};
  rhs[1, 0] = tiny;

  Binary128 const rcond = tiny / Binary128{4};
  EXPECT_EQ(static_cast<double>(rcond), 0.0);

  auto result = uni20::krylov::real_svd_least_squares(coefficients, rhs, rcond);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.singular_values.size(), 2);
  EXPECT_EQ(result.rank, 2);
  EXPECT_TRUE(abs_error(result.singular_values[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.singular_values[1], tiny) <= tiny * tolerance());
  EXPECT_EQ(static_cast<double>(result.singular_values[1]), 0.0);
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(coefficients, result.solution);
  EXPECT_TRUE(abs_error(reconstructed[0, 0], rhs[0, 0]) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 0], rhs[1, 0]) <= tiny * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, DivideAndConquerSvdLeastSquaresKeepsTinyBinary128SingularValue)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto coefficients = zero_matrix<Binary128>(2, 2);
  coefficients[0, 0] = Binary128{1};
  coefficients[1, 1] = tiny;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = Binary128{1};
  rhs[1, 0] = tiny;

  Binary128 const rcond = tiny / Binary128{4};
  EXPECT_EQ(static_cast<double>(rcond), 0.0);

  auto result = uni20::krylov::real_divide_and_conquer_svd_least_squares(coefficients, rhs, rcond);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.singular_values.size(), 2);
  EXPECT_EQ(result.rank, 2);
  EXPECT_TRUE(abs_error(result.singular_values[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.singular_values[1], tiny) <= tiny * tolerance());
  EXPECT_EQ(static_cast<double>(result.singular_values[1]), 0.0);
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(coefficients, result.solution);
  EXPECT_TRUE(abs_error(reconstructed[0, 0], rhs[0, 0]) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 0], rhs[1, 0]) <= tiny * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, RankRevealingLeastSquaresKeepsTinyBinary128Column)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto coefficients = zero_matrix<Binary128>(2, 2);
  coefficients[0, 0] = Binary128{1};
  coefficients[1, 1] = tiny;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = Binary128{1};
  rhs[1, 0] = tiny;

  Binary128 const rcond = tiny / Binary128{4};
  EXPECT_EQ(static_cast<double>(rcond), 0.0);

  auto result = uni20::krylov::real_rank_revealing_least_squares(coefficients, rhs, rcond);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.pivot_columns.size(), 2);
  EXPECT_EQ(result.rank, 2);
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(coefficients, result.solution);
  EXPECT_TRUE(abs_error(reconstructed[0, 0], rhs[0, 0]) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 0], rhs[1, 0]) <= tiny * tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, RealDenseSolveUsesMplapackGesvBelowDoubleResolution)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 1.0);

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = Binary128{2};
  rhs[1, 0] = Binary128{2} + delta;

  auto solution = uni20::krylov::real_dense_solve_linear_system(matrix, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, ExpertDenseSolveUsesMplapackGesvxBelowDoubleResolution)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 1.0);

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = Binary128{2};
  rhs[1, 0] = Binary128{2} + delta;

  auto result = uni20::krylov::real_expert_linear_solve(matrix, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factors.rows(), 2);
  ASSERT_EQ(result.factors.cols(), 2);
  ASSERT_EQ(result.pivot_rows.size(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_TRUE(result.equilibration == 'N' || result.equilibration == 'R' || result.equilibration == 'C' ||
              result.equilibration == 'B');
  EXPECT_TRUE(result.reciprocal_condition > Binary128{});
  EXPECT_TRUE(abs_error(result.solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.solution[1, 0], Binary128{1}) <= tolerance());

  auto reconstructed = multiply_for_test(matrix, result.solution);
  EXPECT_TRUE(abs_error(reconstructed[0, 0], rhs[0, 0]) <= tolerance());
  EXPECT_TRUE(abs_error(reconstructed[1, 0], rhs[1, 0]) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, RealLuSolveUsesMplapackGetrsBelowDoubleResolution)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 1.0);

  auto rhs = zero_matrix<Binary128>(2, 2);
  rhs[0, 0] = Binary128{2};
  rhs[1, 0] = Binary128{2} + delta;
  rhs[0, 1] = Binary128{1};
  rhs[1, 1] = Binary128{1} + delta;

  auto factorization = uni20::krylov::real_lu_factorization(matrix);
  auto solution = uni20::krylov::real_lu_solve(factorization, rhs);

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  ASSERT_EQ(factorization.pivot_rows.size(), 2);
  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 2);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[0, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 1], Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, RealReciprocalConditionNumberStaysInBinary128)
{
  Binary128 const tiny = below_double_minimum_value();
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = tiny;

  auto factorization = uni20::krylov::real_lu_factorization(matrix);
  Binary128 const from_factorization =
      uni20::krylov::real_lu_one_norm_reciprocal_condition_number(factorization, Binary128{1});
  Binary128 const direct = uni20::krylov::real_one_norm_reciprocal_condition_number(matrix);

  EXPECT_TRUE(abs_error(from_factorization, tiny) <= tiny * tolerance());
  EXPECT_TRUE(abs_error(direct, tiny) <= tiny * tolerance());
  EXPECT_EQ(static_cast<double>(from_factorization), 0.0);
  EXPECT_EQ(static_cast<double>(direct), 0.0);
}

TEST(MplapackBinary128DenseSubspaceTest, RealDenseInverseAcceptsPivotGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 1.0);

  auto inverse = uni20::krylov::real_dense_inverse(matrix);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  EXPECT_TRUE(abs_error(inverse[0, 0] * delta, Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(inverse[1, 0] * delta, Binary128{-1}) <= tolerance());
  EXPECT_TRUE(abs_error(inverse[0, 1] * delta, Binary128{-1}) <= tolerance());
  EXPECT_TRUE(abs_error(inverse[1, 1] * delta, Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, LocalDenseOpsPreserveBinary128OnlyIncrements)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const one_plus_delta = Binary128{1} + delta;
  expect_gap_is_binary128_only(delta);

  std::vector<Binary128> cancellation_x{one_plus_delta, Binary128{-1}};
  std::vector<Binary128> cancellation_y{Binary128{1}, Binary128{1}};
  Binary128 const dot_result = uni20::krylov::dot(const_span(cancellation_x), const_span(cancellation_y));
  EXPECT_TRUE(abs_error(dot_result, delta) <= tolerance());

  std::vector<Binary128> axpy_source{delta};
  std::vector<Binary128> axpy_destination{Binary128{1}};
  uni20::krylov::axpy(mutable_span(axpy_destination), Binary128{1}, const_span(axpy_source));
  EXPECT_TRUE(abs_error(axpy_destination[0], one_plus_delta) <= tolerance());

  auto matrix = zero_matrix<Binary128>(1, 2);
  matrix[0, 0] = one_plus_delta;
  matrix[0, 1] = Binary128{-1};
  std::vector<Binary128> vector{Binary128{1}, Binary128{1}};
  std::vector<Binary128> output{Binary128{}};
  uni20::krylov::gemv(mutable_span(output), Binary128{1}, matrix, const_span(vector), Binary128{});
  EXPECT_TRUE(abs_error(output[0], delta) <= tolerance());

  auto rank_one = zero_matrix<Binary128>(1, 1);
  std::vector<Binary128> left{one_plus_delta};
  std::vector<Binary128> right{Binary128{1}};
  uni20::krylov::geru(rank_one, Binary128{1}, const_span(left), const_span(right));
  EXPECT_TRUE(abs_error(rank_one[0, 0], one_plus_delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SolvesSymmetricTridiagonalEigenvectors)
{
  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem(std::vector<Binary128>{Binary128{2}, Binary128{2}},
                                                                 std::vector<Binary128>{Binary128{1}}, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], Binary128{3}) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const residual0 = Binary128{2} * x0 + x1 - lambda * x0;
    Binary128 const residual1 = x0 + Binary128{2} * x1 - lambda * x1;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Binary128 const vector_norm = std::sqrt(x0 * x0 + x1 * x1);
    EXPECT_TRUE(residual_norm <= tolerance());
    EXPECT_TRUE(abs_error(vector_norm, Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricTridiagonalEigenvaluesResolveGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto eigenvalues = uni20::krylov::symmetric_tridiagonal_eigenvalues(
      std::vector<Binary128>{Binary128{1}, Binary128{1} + delta}, std::vector<Binary128>{offdiagonal});

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(eigenvalues.size(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(eigenvalues[1] > eigenvalues[0]);
  EXPECT_TRUE(eigenvalues[1] - eigenvalues[0] > delta);
  EXPECT_TRUE(abs_error(eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1], expected_high) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, ResolvesSymmetricTridiagonalGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem(
      std::vector<Binary128>{Binary128{1}, Binary128{1} + delta}, std::vector<Binary128>{offdiagonal}, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(result.eigenvalues[1] - result.eigenvalues[0] > delta);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const residual0 = x0 + offdiagonal * x1 - lambda * x0;
    Binary128 const residual1 = offdiagonal * x0 + (Binary128{1} + delta) * x1 - lambda * x1;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    EXPECT_TRUE(residual_norm <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, DivideAndConquerSymmetricTridiagonalResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem_divide_and_conquer(
      std::vector<Binary128>{Binary128{1}, Binary128{1} + delta}, std::vector<Binary128>{offdiagonal}, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(result.eigenvalues[1] - result.eigenvalues[0] > delta);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const residual0 = x0 + offdiagonal * x1 - lambda * x0;
    Binary128 const residual1 = offdiagonal * x0 + (Binary128{1} + delta) * x1 - lambda * x1;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Binary128 const vector_norm = std::sqrt(x0 * x0 + x1 * x1);
    EXPECT_TRUE(residual_norm <= tolerance());
    EXPECT_TRUE(abs_error(vector_norm, Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedSymmetricTridiagonalResolvesGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  Binary128 const offdiagonal = delta / Binary128{4};
  expect_gap_is_binary128_only(delta);

  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem_index_range(
      std::vector<Binary128>{Binary128{1}, Binary128{1} + delta, Binary128{2}},
      std::vector<Binary128>{offdiagonal, Binary128{}}, 0, 1, true);

  Binary128 const center = Binary128{1} + delta / Binary128{2};
  Binary128 const radius = std::sqrt((delta / Binary128{2}) * (delta / Binary128{2}) + offdiagonal * offdiagonal);
  Binary128 const expected_low = center - radius;
  Binary128 const expected_high = center + radius;

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_EQ(static_cast<double>(expected_low), 1.0);
  EXPECT_EQ(static_cast<double>(expected_high), 1.0);
  EXPECT_TRUE(result.eigenvalues[1] > result.eigenvalues[0]);
  EXPECT_TRUE(result.eigenvalues[1] - result.eigenvalues[0] > delta);
  EXPECT_TRUE(abs_error(result.eigenvalues[0], expected_low) <= tolerance());
  EXPECT_TRUE(abs_error(result.eigenvalues[1], expected_high) <= tolerance());

  for (uni20::index_type col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Binary128 const lambda = result.eigenvalues[col];
    Binary128 const x0 = result.eigenvectors[0, col];
    Binary128 const x1 = result.eigenvectors[1, col];
    Binary128 const x2 = result.eigenvectors[2, col];
    Binary128 const residual0 = x0 + offdiagonal * x1 - lambda * x0;
    Binary128 const residual1 = offdiagonal * x0 + (Binary128{1} + delta) * x1 - lambda * x1;
    Binary128 const residual2 = Binary128{2} * x2 - lambda * x2;
    Binary128 const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1 + residual2 * residual2);
    Binary128 const vector_norm = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    EXPECT_TRUE(residual_norm <= tolerance());
    EXPECT_TRUE(abs_error(vector_norm, Binary128{1}) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, ComputesRealSchurDecomposition)
{
  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[1, 0] = Binary128{-2};
  matrix[0, 1] = Binary128{2};
  matrix[1, 1] = Binary128{1};

  auto result = uni20::krylov::real_schur(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.blocks.size(), 1);
  EXPECT_EQ(result.blocks[0].begin, 0);
  EXPECT_EQ(result.blocks[0].size, 2);
  EXPECT_TRUE(abs_error(result.eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(complex_abs(result.eigenvalues[0]), std::sqrt(Binary128{5})) <= tolerance());
  EXPECT_TRUE(complex_abs(result.eigenvalues[1] - std::conj(result.eigenvalues[0])) <= tolerance());
  ASSERT_EQ(result.schur_vectors.rows(), 2);
  ASSERT_EQ(result.schur_vectors.cols(), 2);
}

TEST(MplapackBinary128DenseSubspaceTest, HessenbergSchurResolvesDiagonalGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto hessenberg = zero_matrix<Binary128>(3, 3);
  hessenberg[0, 0] = Binary128{1};
  hessenberg[1, 1] = Binary128{1} + delta;
  hessenberg[2, 2] = Binary128{2};
  hessenberg[0, 1] = Binary128{1};
  hessenberg[1, 2] = Binary128{1} / Binary128{3};

  auto result = uni20::krylov::real_hessenberg_schur(std::move(hessenberg), true);
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(eigenvalues.size(), 3);
  ASSERT_EQ(result.blocks.size(), 3);
  ASSERT_EQ(result.schur_vectors.rows(), 3);
  ASSERT_EQ(result.schur_vectors.cols(), 3);
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[2].real(), Binary128{2}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SolvesRealNonsymmetricEigenvalues)
{
  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{3};
  matrix[1, 1] = Binary128{-1};
  matrix[2, 2] = Binary128{2};

  auto result = uni20::krylov::real_nonsymmetric_eigensystem(std::move(matrix), false);
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(eigenvalues.size(), 3);
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{-1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{2}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[2].real(), Binary128{3}) <= tolerance());
  EXPECT_TRUE(complex_abs(eigenvalues[0] - uni20::complex<Binary128>{Binary128{-1}, Binary128{}}) <= tolerance());
  EXPECT_EQ(result.right_eigenvectors.rows(), 0);
  EXPECT_EQ(result.right_eigenvectors.cols(), 0);
}

TEST(MplapackBinary128DenseSubspaceTest, ResolvesRealNonsymmetricEigenvalueGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;

  auto result = uni20::krylov::real_nonsymmetric_eigensystem(std::move(matrix), false);
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(eigenvalues.size(), 2);
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[0].imag(), Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].imag(), Binary128{}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, ExpertRealNonsymmetricEigensystemReportsBinary128GapDiagnostics)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;

  auto result = uni20::krylov::real_nonsymmetric_expert_eigensystem(std::move(matrix), true);
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(eigenvalues.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 2);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 2);
  ASSERT_EQ(result.balance_scale.size(), 2);
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(result.balanced_matrix_norm > Binary128{});
  EXPECT_LT(result.balanced_first, result.balanced_last_exclusive);
  EXPECT_LE(result.balanced_last_exclusive, 2);
  for (std::size_t index = 0; index < 2; ++index)
  {
    EXPECT_TRUE(result.reciprocal_eigenvalue_condition_numbers[index] > Binary128{});
    EXPECT_TRUE(result.reciprocal_eigenvector_condition_numbers[index] > Binary128{});
  }
}

TEST(MplapackBinary128DenseSubspaceTest, BalancesRealNonsymmetricMatrixWithTinyBinary128Offdiagonal)
{
  Binary128 const tiny = binary_power_of_two(-8000);
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 1] = tiny;
  matrix[1, 0] = Binary128{1};

  auto result = uni20::krylov::real_nonsymmetric_balance(matrix, uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  ASSERT_EQ(result.balanced_matrix.rows(), 2);
  ASSERT_EQ(result.balanced_matrix.cols(), 2);
  ASSERT_EQ(result.scale.size(), 2);
  EXPECT_LT(result.balanced_first, result.balanced_last_exclusive);
  EXPECT_LE(result.balanced_last_exclusive, 2);
  EXPECT_TRUE((result.balanced_matrix[0, 1] > Binary128{}));
  EXPECT_EQ(static_cast<double>(result.balanced_matrix[0, 1]), 0.0);
  EXPECT_TRUE((result.balanced_matrix[1, 0] > Binary128{}));

  auto vectors = zero_matrix<Binary128>(2, 1);
  vectors[0, 0] = Binary128{1};
  auto transformed = uni20::krylov::real_nonsymmetric_balance_backtransform_right_vectors(
      vectors, result, uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  ASSERT_EQ(transformed.rows(), 2);
  ASSERT_EQ(transformed.cols(), 1);
  EXPECT_TRUE((transformed[0, 0] != Binary128{}));
}

TEST(MplapackBinary128DenseSubspaceTest, BalancesRealGeneralizedNonsymmetricPencilWithTinyBinary128Offdiagonal)
{
  Binary128 const tiny = binary_power_of_two(-8000);
  expect_value_underflows_to_double_zero(tiny);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 1] = tiny;
  matrix[1, 0] = Binary128{1};

  auto metric = zero_matrix<Binary128>(2, 2);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{2};

  auto result = uni20::krylov::real_generalized_nonsymmetric_balance(matrix, metric,
                                                                     uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  ASSERT_EQ(result.balanced_matrix.rows(), 2);
  ASSERT_EQ(result.balanced_matrix.cols(), 2);
  ASSERT_EQ(result.balanced_metric.rows(), 2);
  ASSERT_EQ(result.balanced_metric.cols(), 2);
  ASSERT_EQ(result.left_scale.size(), 2);
  ASSERT_EQ(result.right_scale.size(), 2);
  EXPECT_LT(result.balanced_first, result.balanced_last_exclusive);
  EXPECT_LE(result.balanced_last_exclusive, 2);
  EXPECT_TRUE((result.balanced_matrix[0, 1] > Binary128{}));
  EXPECT_EQ(static_cast<double>(result.balanced_matrix[0, 1]), 0.0);
  EXPECT_TRUE((result.balanced_matrix[1, 0] > Binary128{}));

  auto vectors = zero_matrix<Binary128>(2, 1);
  vectors[0, 0] = Binary128{1};
  auto transformed = uni20::krylov::real_generalized_nonsymmetric_balance_backtransform_right_vectors(
      vectors, result, uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  ASSERT_EQ(transformed.rows(), 2);
  ASSERT_EQ(transformed.cols(), 1);
  EXPECT_TRUE((transformed[0, 0] != Binary128{}));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedHessenbergReductionPreservesBinary128OnlyEntry)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{3};
  matrix[1, 0] = Binary128{2};
  matrix[2, 0] = Binary128{-1};
  matrix[0, 1] = Binary128{4};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 1] = Binary128{5};
  matrix[0, 2] = Binary128{-2};
  matrix[1, 2] = Binary128{1};
  matrix[2, 2] = Binary128{6};
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 1.0);

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{2};
  metric[0, 1] = Binary128{-1};
  metric[1, 1] = Binary128{3};
  metric[0, 2] = Binary128{1};
  metric[1, 2] = Binary128{2};
  metric[2, 2] = Binary128{4};

  auto result = uni20::krylov::real_generalized_hessenberg_reduction(matrix, metric, true);
  auto reconstructed_matrix =
      multiply_for_test(multiply_for_test(result.left_orthogonal_vectors, result.matrix_hessenberg_form),
                        transpose(result.right_orthogonal_vectors));
  auto reconstructed_metric =
      multiply_for_test(multiply_for_test(result.left_orthogonal_vectors, result.metric_triangular_form),
                        transpose(result.right_orthogonal_vectors));

  ASSERT_EQ(result.matrix_hessenberg_form.rows(), 3);
  ASSERT_EQ(result.matrix_hessenberg_form.cols(), 3);
  ASSERT_EQ(result.metric_triangular_form.rows(), 3);
  ASSERT_EQ(result.metric_triangular_form.cols(), 3);
  EXPECT_TRUE(abs_error(result.matrix_hessenberg_form[2, 0], Binary128{}) <= tolerance());
  for (std::size_t col = 0; col < 3; ++col)
  {
    for (std::size_t row = col + 1; row < 3; ++row)
    {
      EXPECT_TRUE(abs_error(result.metric_triangular_form[row, col], Binary128{}) <= tolerance());
      EXPECT_TRUE(abs_error(reconstructed_matrix[row, col], matrix[row, col]) <= tolerance());
      EXPECT_TRUE(abs_error(reconstructed_metric[row, col], metric[row, col]) <= tolerance());
    }
  }
  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 3; ++col)
    {
      EXPECT_TRUE(abs_error(reconstructed_matrix[row, col], matrix[row, col]) <= tolerance());
      EXPECT_TRUE(abs_error(reconstructed_metric[row, col], metric[row, col]) <= tolerance());
    }
  }
  EXPECT_EQ(static_cast<double>(reconstructed_matrix[1, 1]), 1.0);
  EXPECT_TRUE((reconstructed_matrix[1, 1] > Binary128{1}));
  EXPECT_TRUE(abs_error(reconstructed_matrix[1, 1], Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedHessenbergSchurResolvesMetricGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto hessenberg = zero_matrix<Binary128>(3, 3);
  hessenberg[0, 0] = Binary128{1};
  hessenberg[0, 1] = delta / Binary128{4};
  hessenberg[1, 1] = Binary128{1};
  hessenberg[1, 2] = Binary128{3} * delta;
  hessenberg[2, 2] = Binary128{2};

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[0, 2] = delta / Binary128{8};
  metric[1, 1] = Binary128{1} / (Binary128{1} + delta);
  metric[2, 2] = Binary128{1};
  EXPECT_EQ(static_cast<double>(metric[1, 1]), 1.0);

  auto result = uni20::krylov::real_generalized_hessenberg_schur(hessenberg, metric, true);
  auto reconstructed_matrix = multiply_for_test(multiply_for_test(result.left_schur_vectors, result.matrix_schur_form),
                                                transpose(result.right_schur_vectors));
  auto reconstructed_metric = multiply_for_test(multiply_for_test(result.left_schur_vectors, result.metric_schur_form),
                                                transpose(result.right_schur_vectors));
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(result.alpha.size(), 3);
  ASSERT_EQ(result.beta.size(), 3);
  ASSERT_EQ(eigenvalues.size(), 3);
  ASSERT_EQ(result.left_schur_vectors.rows(), 3);
  ASSERT_EQ(result.left_schur_vectors.cols(), 3);
  ASSERT_EQ(result.right_schur_vectors.rows(), 3);
  ASSERT_EQ(result.right_schur_vectors.cols(), 3);
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[2].real(), Binary128{2}) <= tolerance());

  for (std::size_t col = 0; col < 3; ++col)
  {
    for (std::size_t row = 0; row < 3; ++row)
    {
      EXPECT_TRUE(abs_error(reconstructed_matrix[row, col], hessenberg[row, col]) <= tolerance());
      EXPECT_TRUE(abs_error(reconstructed_metric[row, col], metric[row, col]) <= tolerance());
    }
  }
  for (std::size_t index = 0; index < result.alpha.size(); ++index)
  {
    EXPECT_TRUE(abs_error(result.beta[index], Binary128{}) > tolerance());
    EXPECT_TRUE(complex_abs(result.alpha[index] / result.beta[index] - result.eigenvalues[index]) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, ResolvesRealGeneralizedNonsymmetricEigenvalueGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{2} * (Binary128{1} + delta);
  matrix[2, 2] = Binary128{6};
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 2.0);

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{2};
  metric[2, 2] = Binary128{3};

  auto result = uni20::krylov::real_generalized_nonsymmetric_eigensystem(std::move(matrix), std::move(metric), true);
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(result.alpha.size(), 3);
  ASSERT_EQ(result.beta.size(), 3);
  ASSERT_EQ(eigenvalues.size(), 3);
  ASSERT_EQ(result.right_eigenvectors.rows(), 3);
  ASSERT_EQ(result.right_eigenvectors.cols(), 3);
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[2].real(), Binary128{2}) <= tolerance());

  for (std::size_t col = 0; col < result.eigenvalues.size(); ++col)
  {
    EXPECT_TRUE(abs_error(result.beta[col], Binary128{}) > tolerance());
    EXPECT_TRUE(complex_abs(result.alpha[col] / result.beta[col] - result.eigenvalues[col]) <= tolerance());

    auto const lambda = result.eigenvalues[col];
    Binary128 residual_norm{};
    for (uni20::index_type row = 0; row < result.right_eigenvectors.rows(); ++row)
    {
      Binary128 const matrix_diagonal =
          row == 0 ? Binary128{1} : (row == 1 ? Binary128{2} * (Binary128{1} + delta) : Binary128{6});
      Binary128 const metric_diagonal = Binary128{1} + static_cast<Binary128>(row);
      auto const value = result.right_eigenvectors[row, col];
      auto const residual = matrix_diagonal * value - lambda * metric_diagonal * value;
      Binary128 const residual_abs = complex_abs(residual);
      residual_norm += residual_abs * residual_abs;
    }
    EXPECT_TRUE(std::sqrt(residual_norm) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, ExpertRealGeneralizedNonsymmetricEigensystemReportsBinary128GapDiagnostics)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{2} * (Binary128{1} + delta);
  matrix[2, 2] = Binary128{6};
  EXPECT_EQ(static_cast<double>(matrix[1, 1]), 2.0);

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{2};
  metric[2, 2] = Binary128{3};

  auto result =
      uni20::krylov::real_generalized_nonsymmetric_expert_eigensystem(std::move(matrix), std::move(metric), true);
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(result.alpha.size(), 3);
  ASSERT_EQ(result.beta.size(), 3);
  ASSERT_EQ(eigenvalues.size(), 3);
  ASSERT_EQ(result.right_eigenvectors.rows(), 3);
  ASSERT_EQ(result.right_eigenvectors.cols(), 3);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 3);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 3);
  ASSERT_EQ(result.left_balance_scale.size(), 3);
  ASSERT_EQ(result.right_balance_scale.size(), 3);
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[2].real(), Binary128{2}) <= tolerance());
  EXPECT_TRUE(result.balanced_matrix_norm > Binary128{});
  EXPECT_TRUE(result.balanced_metric_norm > Binary128{});
  EXPECT_LT(result.balanced_first, result.balanced_last_exclusive);
  EXPECT_LE(result.balanced_last_exclusive, 3);
  for (std::size_t index = 0; index < 3; ++index)
  {
    EXPECT_TRUE(result.reciprocal_eigenvalue_condition_numbers[index] > Binary128{});
    EXPECT_TRUE(result.reciprocal_eigenvector_condition_numbers[index] > Binary128{});
  }
}

TEST(MplapackBinary128DenseSubspaceTest, RealGeneralizedSchurResolvesMetricGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1};
  matrix[2, 2] = Binary128{2};

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{1} / (Binary128{1} + delta);
  metric[2, 2] = Binary128{1};

  auto result = uni20::krylov::real_generalized_schur(std::move(matrix), std::move(metric), false);
  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });

  ASSERT_EQ(eigenvalues.size(), 3);
  ASSERT_EQ(result.alpha.size(), 3);
  ASSERT_EQ(result.beta.size(), 3);
  ASSERT_EQ(result.blocks.size(), 3);
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[2].real(), Binary128{2}) <= tolerance());
  EXPECT_EQ(result.left_schur_vectors.rows(), 0);
  EXPECT_EQ(result.right_schur_vectors.rows(), 0);
  for (std::size_t index = 0; index < result.alpha.size(); ++index)
  {
    EXPECT_TRUE(abs_error(result.beta[index], Binary128{}) > tolerance());
    EXPECT_TRUE(complex_abs(result.alpha[index] / result.beta[index] - result.eigenvalues[index]) <= tolerance());
  }
}

TEST(MplapackBinary128DenseSubspaceTest, ReordersGeneralizedSchurBlockSeparatedOnlyInBinary128)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1};
  matrix[2, 2] = Binary128{2};

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{1} / (Binary128{1} + delta);
  metric[2, 2] = Binary128{1};

  auto schur = uni20::krylov::real_generalized_schur(std::move(matrix), std::move(metric), true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (abs_error(schur.blocks[index].first_eigenvalue.real(), Binary128{1} + delta) <= tolerance())
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto reordered =
      uni20::krylov::reorder_real_generalized_schur(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_GE(reordered.eigenvalues.size(), 3);
  ASSERT_GE(reordered.blocks.size(), 3);
  EXPECT_EQ(static_cast<double>(reordered.eigenvalues.front().real()), 1.0);
  EXPECT_TRUE(reordered.eigenvalues.front().real() > Binary128{1});
  EXPECT_TRUE(abs_error(reordered.eigenvalues.front().real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(complex_abs(reordered.alpha.front() / reordered.beta.front() - reordered.eigenvalues.front()) <=
              tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedGeneralizedSchurSubspaceKeepsBinary128SeparatedBlock)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1};
  matrix[2, 2] = Binary128{2};

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{1} / (Binary128{1} + delta);
  metric[2, 2] = Binary128{1};

  auto schur = uni20::krylov::real_generalized_schur(std::move(matrix), std::move(metric), true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (abs_error(schur.blocks[index].first_eigenvalue.real(), Binary128{1} + delta) <= tolerance())
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto result =
      uni20::krylov::real_generalized_schur_selected_subspace(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_GE(result.decomposition.eigenvalues.size(), 3);
  EXPECT_EQ(result.selected_dimension, 1);
  EXPECT_EQ(static_cast<double>(result.decomposition.eigenvalues.front().real()), 1.0);
  EXPECT_TRUE(result.decomposition.eigenvalues.front().real() > Binary128{1});
  EXPECT_TRUE(abs_error(result.decomposition.eigenvalues.front().real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(complex_abs(result.decomposition.alpha.front() / result.decomposition.beta.front() -
                          result.decomposition.eigenvalues.front()) <= tolerance());
  EXPECT_TRUE(result.left_projection_lower_bound > Binary128{});
  EXPECT_TRUE(result.right_projection_lower_bound > Binary128{});
  EXPECT_TRUE(result.upper_deflating_subspace_separation > Binary128{});
  EXPECT_TRUE(result.lower_deflating_subspace_separation > Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSchurRightEigenvectorsUseBinary128OnlyGap)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1};
  matrix[2, 2] = Binary128{2};

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{1} / (Binary128{1} + delta);
  metric[2, 2] = Binary128{1};

  auto schur = uni20::krylov::real_generalized_schur(std::move(matrix), std::move(metric), false);
  auto matrix_schur_form = schur.matrix_schur_form;
  auto metric_schur_form = schur.metric_schur_form;
  auto eigenvalues = schur.eigenvalues;
  auto eigenvectors = uni20::krylov::real_generalized_schur_right_eigenvectors(std::move(schur));

  ASSERT_EQ(eigenvectors.computed_vectors, 3);
  ASSERT_EQ(eigenvectors.right_eigenvectors.rows(), 3);
  std::size_t target_column = eigenvalues.size();
  for (std::size_t col = 0; col < eigenvalues.size(); ++col)
  {
    if (abs_error(eigenvalues[col].real(), Binary128{1} + delta) <= tolerance())
    {
      target_column = col;
    }
    EXPECT_TRUE(generalized_schur_right_eigenvector_residual_max(matrix_schur_form, metric_schur_form, eigenvalues[col],
                                                                 eigenvectors.right_eigenvectors, col) <= tolerance());
  }
  ASSERT_LT(target_column, eigenvalues.size());
  EXPECT_EQ(static_cast<double>(eigenvalues[target_column].real()), 1.0);
  EXPECT_TRUE(eigenvalues[target_column].real() > Binary128{1});
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSchurConditionEstimatesUseBinary128OnlyGap)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1};
  matrix[2, 2] = Binary128{2};

  auto metric = zero_matrix<Binary128>(3, 3);
  metric[0, 0] = Binary128{1};
  metric[1, 1] = Binary128{1} / (Binary128{1} + delta);
  metric[2, 2] = Binary128{1};

  auto schur = uni20::krylov::real_generalized_schur(std::move(matrix), std::move(metric), false);
  auto eigenvalues = schur.eigenvalues;
  auto result = uni20::krylov::real_generalized_schur_condition_estimates(std::move(schur));

  ASSERT_EQ(result.computed_estimates, 3);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 3);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 3);
  std::size_t target_column = eigenvalues.size();
  for (std::size_t col = 0; col < eigenvalues.size(); ++col)
  {
    if (abs_error(eigenvalues[col].real(), Binary128{1} + delta) <= tolerance())
    {
      target_column = col;
    }
    EXPECT_TRUE(result.reciprocal_eigenvalue_condition_numbers[col] > Binary128{});
    EXPECT_TRUE(result.reciprocal_eigenvector_condition_numbers[col] >= Binary128{});
  }
  ASSERT_LT(target_column, eigenvalues.size());
  Binary128 const min_eigenvector_condition =
      *std::ranges::min_element(result.reciprocal_eigenvector_condition_numbers);
  EXPECT_TRUE(min_eigenvector_condition > Binary128{});
  EXPECT_TRUE(min_eigenvector_condition < static_cast<Binary128>(1.0e-20L));
  EXPECT_EQ(static_cast<double>(eigenvalues[target_column].real()), 1.0);
  EXPECT_TRUE(eigenvalues[target_column].real() > Binary128{1});
}

TEST(MplapackBinary128DenseSubspaceTest, ReordersRealSchurBlocks)
{
  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{3};
  matrix[2, 2] = Binary128{2};

  auto schur = uni20::krylov::real_schur(matrix, true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (abs_error(schur.blocks[index].first_eigenvalue.real(), Binary128{3}) <= tolerance())
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto reordered = uni20::krylov::reorder_real_schur(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_FALSE(reordered.eigenvalues.empty());
  EXPECT_TRUE(abs_error(reordered.eigenvalues.front().real(), Binary128{3}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, ReordersSchurBlockSeparatedOnlyInBinary128)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 2] = Binary128{2};

  auto schur = uni20::krylov::real_schur(matrix, true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (abs_error(schur.blocks[index].first_eigenvalue.real(), Binary128{1} + delta) <= tolerance())
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto reordered = uni20::krylov::reorder_real_schur(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_GE(reordered.eigenvalues.size(), 3);
  EXPECT_EQ(static_cast<double>(reordered.eigenvalues.front().real()), 1.0);
  EXPECT_TRUE(reordered.eigenvalues.front().real() > Binary128{1});
  EXPECT_TRUE(abs_error(reordered.eigenvalues.front().real(), Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedSchurSubspaceKeepsBinary128SeparatedBlock)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(3, 3);
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 2] = Binary128{2};

  auto schur = uni20::krylov::real_schur(matrix, true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (abs_error(schur.blocks[index].first_eigenvalue.real(), Binary128{1} + delta) <= tolerance())
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto result = uni20::krylov::real_schur_selected_subspace(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_EQ(result.selected_dimension, 1);
  ASSERT_GE(result.decomposition.eigenvalues.size(), 3);
  EXPECT_EQ(static_cast<double>(result.decomposition.eigenvalues.front().real()), 1.0);
  EXPECT_TRUE(result.decomposition.eigenvalues.front().real() > Binary128{1});
  EXPECT_TRUE(abs_error(result.decomposition.eigenvalues.front().real(), Binary128{1} + delta) <= tolerance());
  EXPECT_TRUE(result.reciprocal_eigenvalue_cluster_condition > Binary128{});
  EXPECT_TRUE(result.reciprocal_invariant_subspace_condition > Binary128{});
}

TEST(MplapackBinary128DenseSubspaceTest, SchurRightEigenvectorsUseBinary128OnlyGap)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::krylov::RealSchurDecomposition<Binary128> schur;
  schur.schur_form = zero_matrix<Binary128>(2, 2);
  schur.schur_form[0, 0] = Binary128{1};
  schur.schur_form[0, 1] = Binary128{1};
  schur.schur_form[1, 1] = Binary128{1} + delta;
  schur.schur_vectors = identity_matrix(2);
  schur.eigenvalues =
      std::vector<uni20::complex<Binary128>>{uni20::complex<Binary128>{Binary128{1}, Binary128{}},
                                             uni20::complex<Binary128>{Binary128{1} + delta, Binary128{}}};
  schur.blocks = std::vector<uni20::krylov::RealSchurBlock<Binary128>>{
      uni20::krylov::RealSchurBlock<Binary128>{
          .begin = 0, .size = 1, .first_eigenvalue = schur.eigenvalues[0], .second_eigenvalue = {}},
      uni20::krylov::RealSchurBlock<Binary128>{
          .begin = 1, .size = 1, .first_eigenvalue = schur.eigenvalues[1], .second_eigenvalue = {}}};

  auto const schur_form = schur.schur_form;
  auto const eigenvalues = schur.eigenvalues;
  auto result = uni20::krylov::real_schur_right_eigenvectors(std::move(schur));

  ASSERT_EQ(result.computed_vectors, 2);
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
  for (std::size_t column = 0; column < 2; ++column)
  {
    EXPECT_TRUE(schur_right_eigenvector_residual_max(schur_form, eigenvalues[column], result.right_eigenvectors,
                                                     column) <= Binary128{100} * tolerance());
  }

  auto const x0 = result.right_eigenvectors[0, 1];
  auto const x1 = result.right_eigenvectors[1, 1];
  ASSERT_GT(complex_abs(x0), Binary128{});
  auto const ratio = x1 / x0;
  EXPECT_TRUE(abs_error(ratio.real(), delta) <= Binary128{100} * tolerance());
  EXPECT_TRUE(abs_error(ratio.imag(), Binary128{}) <= Binary128{100} * tolerance());
  EXPECT_TRUE(complex_abs(x1) > Binary128{});
  EXPECT_TRUE(complex_abs(x1) < Binary128{1} / binary_power_of_two(60));
}

TEST(MplapackBinary128DenseSubspaceTest, SchurConditionEstimatesUseBinary128OnlyGap)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::krylov::RealSchurDecomposition<Binary128> schur;
  schur.schur_form = zero_matrix<Binary128>(2, 2);
  schur.schur_form[0, 0] = Binary128{1};
  schur.schur_form[0, 1] = Binary128{1};
  schur.schur_form[1, 1] = Binary128{1} + delta;
  schur.schur_vectors = identity_matrix(2);
  schur.eigenvalues =
      std::vector<uni20::complex<Binary128>>{uni20::complex<Binary128>{Binary128{1}, Binary128{}},
                                             uni20::complex<Binary128>{Binary128{1} + delta, Binary128{}}};
  schur.blocks = std::vector<uni20::krylov::RealSchurBlock<Binary128>>{
      uni20::krylov::RealSchurBlock<Binary128>{
          .begin = 0, .size = 1, .first_eigenvalue = schur.eigenvalues[0], .second_eigenvalue = {}},
      uni20::krylov::RealSchurBlock<Binary128>{
          .begin = 1, .size = 1, .first_eigenvalue = schur.eigenvalues[1], .second_eigenvalue = {}}};

  auto result = uni20::krylov::real_schur_condition_estimates(std::move(schur));

  ASSERT_EQ(result.computed_estimates, 2);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 2);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 2);
  Binary128 min_vector_condition = result.reciprocal_eigenvector_condition_numbers[0];
  for (std::size_t index = 0; index < 2; ++index)
  {
    EXPECT_TRUE(result.reciprocal_eigenvalue_condition_numbers[index] > Binary128{});
    EXPECT_TRUE(result.reciprocal_eigenvector_condition_numbers[index] > Binary128{});
    min_vector_condition = std::min(min_vector_condition, result.reciprocal_eigenvector_condition_numbers[index]);
  }
  EXPECT_TRUE(min_vector_condition < Binary128{1} / binary_power_of_two(60));
}

TEST(MplapackBinary128DenseSubspaceTest, SchurConditionWorkspaceQueryAcceptsNullIntegerWork)
{
  std::vector<uni20::blas_int> select{1, 0};
  std::vector<Binary128> schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> schur_vectors{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> wr(2);
  std::vector<Binary128> wi(2);
  std::vector<Binary128> work(1);
  uni20::blas_int selected_dimension = 0;
  Binary128 reciprocal_eigenvalue_cluster_condition{};
  Binary128 reciprocal_invariant_subspace_condition{};

  EXPECT_NO_THROW(uni20::lapack::trsen('B', 'V', select.data(), 2, schur_form.data(), 2, schur_vectors.data(), 2,
                                       wr.data(), wi.data(), selected_dimension,
                                       reciprocal_eigenvalue_cluster_condition, reciprocal_invariant_subspace_condition,
                                       work.data(), -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, SchurConditionQuickReturnAcceptsNullSelection)
{
  std::vector<Binary128> schur_form(1);
  std::vector<Binary128> schur_vectors(1);
  std::vector<Binary128> wr(1);
  std::vector<Binary128> wi(1);
  std::vector<Binary128> work(1);
  std::vector<uni20::blas_int> iwork(1);
  uni20::blas_int selected_dimension = 0;
  Binary128 reciprocal_eigenvalue_cluster_condition{};
  Binary128 reciprocal_invariant_subspace_condition{};

  EXPECT_NO_THROW(uni20::lapack::trsen('N', 'N', nullptr, 0, schur_form.data(), 1, schur_vectors.data(), 1, wr.data(),
                                       wi.data(), selected_dimension, reciprocal_eigenvalue_cluster_condition,
                                       reciprocal_invariant_subspace_condition, work.data(), 1, iwork.data(), 1));
  EXPECT_EQ(selected_dimension, 0);
}

TEST(MplapackBinary128DenseSubspaceTest, SchurEigenvectorsAllModeAcceptsNullSelection)
{
  std::vector<Binary128> schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> left_vectors(4);
  std::vector<Binary128> right_vectors(4);
  std::vector<Binary128> work(6);
  uni20::blas_int computed_vectors = 0;

  EXPECT_NO_THROW(uni20::lapack::trevc('R', 'A', nullptr, 2, schur_form.data(), 2, left_vectors.data(), 2,
                                       right_vectors.data(), 2, 2, computed_vectors, work.data()));
}

TEST(MplapackBinary128DenseSubspaceTest, SchurConditionAllModeAcceptsNullSelection)
{
  std::vector<Binary128> schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> left_vectors{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> right_vectors{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> reciprocal_eigenvalue_condition_numbers(2);
  std::vector<Binary128> reciprocal_eigenvector_condition_numbers(2);
  std::vector<Binary128> work(16);
  std::vector<uni20::blas_int> iwork(2);
  uni20::blas_int computed_estimates = 0;

  EXPECT_NO_THROW(uni20::lapack::trsna('B', 'A', nullptr, 2, schur_form.data(), 2, left_vectors.data(), 2,
                                       right_vectors.data(), 2, reciprocal_eigenvalue_condition_numbers.data(),
                                       reciprocal_eigenvector_condition_numbers.data(), 2, computed_estimates,
                                       work.data(), 2, iwork.data()));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedEigensystemNoConditionModeAcceptsNullBooleanWork)
{
  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> metric{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> alphar(2);
  std::vector<Binary128> alphai(2);
  std::vector<Binary128> beta(2);
  std::vector<Binary128> left_vectors(1);
  std::vector<Binary128> right_vectors(1);
  std::vector<Binary128> lscale(2);
  std::vector<Binary128> rscale(2);
  std::vector<Binary128> rconde(2);
  std::vector<Binary128> rcondv(2);
  std::vector<Binary128> work(1);
  std::vector<uni20::blas_int> iwork(1);
  uni20::blas_int first = 0;
  uni20::blas_int last = 0;
  Binary128 matrix_norm{};
  Binary128 metric_norm{};

  EXPECT_NO_THROW(uni20::lapack::ggevx('N', 'N', 'N', 'N', 2, matrix.data(), 2, metric.data(), 2, alphar.data(),
                                       alphai.data(), beta.data(), left_vectors.data(), 1, right_vectors.data(), 1,
                                       first, last, lscale.data(), rscale.data(), matrix_norm, metric_norm,
                                       rconde.data(), rcondv.data(), work.data(), -1, iwork.data(), nullptr));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSchurUnsortedModeAcceptsNullBooleanWork)
{
  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> metric{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> alphar(2);
  std::vector<Binary128> alphai(2);
  std::vector<Binary128> beta(2);
  std::vector<Binary128> left_schur_vectors(1);
  std::vector<Binary128> right_schur_vectors(1);
  std::vector<Binary128> work(1);
  uni20::blas_int selected_dimension = 0;

  EXPECT_NO_THROW(uni20::lapack::gges('N', 'N', 'N', 2, matrix.data(), 2, metric.data(), 2, selected_dimension,
                                      alphar.data(), alphai.data(), beta.data(), left_schur_vectors.data(), 1,
                                      right_schur_vectors.data(), 1, work.data(), -1, nullptr));
}

TEST(MplapackBinary128DenseSubspaceTest, DivideAndConquerLeastSquaresDoesNotWritePastCallerIntegerWork)
{
  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{}, Binary128{},  Binary128{1},
                                Binary128{},  Binary128{}, Binary128{}, Binary128{}, Binary128{1}, Binary128{}};
  std::vector<Binary128> rhs{Binary128{1}, Binary128{2}, Binary128{3}, Binary128{}};
  std::vector<Binary128> singular_values(3);
  std::vector<Binary128> work(1);
  std::vector<uni20::blas_int> intentionally_tiny_iwork(1);
  uni20::blas_int rank = 0;

  EXPECT_NO_THROW(uni20::lapack::gelsd(4, 3, 1, matrix.data(), 4, rhs.data(), 4, singular_values.data(), Binary128{-1},
                                       rank, work.data(), -1, intentionally_tiny_iwork.data()));
}

TEST(MplapackBinary128DenseSubspaceTest, SymmetricDivideAndConquerWorkspaceQueryAcceptsNullIntegerWork)
{
  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Binary128> work(1);

  EXPECT_NO_THROW(
      uni20::lapack::syevd('N', 'U', 2, matrix.data(), 2, eigenvalues.data(), work.data(), -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, QuerySizedIntegerWorkspaceOnlyCopiesFirstElementBack)
{
  std::vector<Binary128> query_matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> query_eigenvalues(2);
  std::vector<Binary128> work_query(1);
  std::vector<uni20::blas_int> iwork_query(1);

  ASSERT_NO_THROW(uni20::lapack::syevd('V', 'U', 2, query_matrix.data(), 2, query_eigenvalues.data(), work_query.data(),
                                       -1, iwork_query.data(), -1));

  auto const lwork = static_cast<uni20::blas_int>(work_query[0]);
  auto const liwork = iwork_query[0];
  ASSERT_GT(lwork, 0);
  ASSERT_GT(liwork, 1);

  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Binary128> work(static_cast<std::size_t>(lwork));
  uni20::blas_int const sentinel = -12345;
  std::vector<uni20::blas_int> iwork(static_cast<std::size_t>(liwork), sentinel);

  ASSERT_NO_THROW(uni20::lapack::syevd('V', 'U', 2, matrix.data(), 2, eigenvalues.data(), work.data(), lwork,
                                       iwork.data(), liwork));

  for (std::size_t index = 1; index < iwork.size(); ++index)
  {
    EXPECT_EQ(iwork[index], sentinel);
  }
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedSymmetricWorkspaceQueryAcceptsNullSupportAndIntegerWork)
{
  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Binary128> eigenvectors(1);
  std::vector<Binary128> work(1);
  uni20::blas_int selected_count = 0;

  EXPECT_NO_THROW(uni20::lapack::syevr('N', 'A', 'U', 2, matrix.data(), 2, Binary128{}, Binary128{}, 0, 0, Binary128{},
                                       selected_count, eigenvalues.data(), eigenvectors.data(), 1, nullptr, work.data(),
                                       -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSymmetricDivideAndConquerWorkspaceQueryAcceptsNullIntegerWork)
{
  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> metric{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Binary128> work(1);

  EXPECT_NO_THROW(uni20::lapack::sygvd(1, 'N', 'U', 2, matrix.data(), 2, metric.data(), 2, eigenvalues.data(),
                                       work.data(), -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSymmetricSelectedWorkspaceAcceptsNullIntegerOutputs)
{
  std::vector<Binary128> matrix{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> metric{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Binary128> eigenvectors(1);
  std::vector<Binary128> work(1);
  uni20::blas_int selected_count = 0;

  EXPECT_NO_THROW(uni20::lapack::sygvx(1, 'N', 'A', 'U', 2, matrix.data(), 2, metric.data(), 2, Binary128{},
                                       Binary128{}, 0, 0, Binary128{}, selected_count, eigenvalues.data(),
                                       eigenvectors.data(), 1, work.data(), -1, nullptr, nullptr));
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexHermitianDivideAndConquerWorkspaceQueryAcceptsNullIntegerWork)
{
  using Complex = uni20::complex<Binary128>;

  std::vector<Complex> matrix{Complex{Binary128{1}, Binary128{}}, Complex{}, Complex{},
                              Complex{Binary128{2}, Binary128{}}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Complex> work(1);
  std::vector<Binary128> rwork(1);

  EXPECT_NO_THROW(uni20::lapack::heevd('N', 'U', 2, matrix.data(), 2, eigenvalues.data(), work.data(), -1, rwork.data(),
                                       -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedComplexHermitianWorkspaceQueryAcceptsNullSupportAndIntegerWork)
{
  using Complex = uni20::complex<Binary128>;

  std::vector<Complex> matrix{Complex{Binary128{1}, Binary128{}}, Complex{}, Complex{},
                              Complex{Binary128{2}, Binary128{}}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Complex> eigenvectors(1);
  std::vector<Complex> work(1);
  std::vector<Binary128> rwork(1);
  uni20::blas_int selected_count = 0;

  EXPECT_NO_THROW(uni20::lapack::heevr('N', 'A', 'U', 2, matrix.data(), 2, Binary128{}, Binary128{}, 0, 0, Binary128{},
                                       selected_count, eigenvalues.data(), eigenvectors.data(), 1, nullptr, work.data(),
                                       -1, rwork.data(), -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest,
     ComplexGeneralizedHermitianDivideAndConquerWorkspaceQueryAcceptsNullIntegerWork)
{
  using Complex = uni20::complex<Binary128>;

  std::vector<Complex> matrix{Complex{Binary128{1}, Binary128{}}, Complex{}, Complex{},
                              Complex{Binary128{2}, Binary128{}}};
  std::vector<Complex> metric{Complex{Binary128{1}, Binary128{}}, Complex{}, Complex{},
                              Complex{Binary128{1}, Binary128{}}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Complex> work(1);
  std::vector<Binary128> rwork(1);

  EXPECT_NO_THROW(uni20::lapack::hegvd(1, 'N', 'U', 2, matrix.data(), 2, metric.data(), 2, eigenvalues.data(),
                                       work.data(), -1, rwork.data(), -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, ComplexGeneralizedHermitianSelectedWorkspaceAcceptsNullIntegerOutputs)
{
  using Complex = uni20::complex<Binary128>;

  std::vector<Complex> matrix{Complex{Binary128{1}, Binary128{}}, Complex{}, Complex{},
                              Complex{Binary128{2}, Binary128{}}};
  std::vector<Complex> metric{Complex{Binary128{1}, Binary128{}}, Complex{}, Complex{},
                              Complex{Binary128{1}, Binary128{}}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Complex> eigenvectors(1);
  std::vector<Complex> work(1);
  std::vector<Binary128> rwork(7);
  uni20::blas_int selected_count = 0;

  EXPECT_NO_THROW(uni20::lapack::hegvx(1, 'N', 'A', 'U', 2, matrix.data(), 2, metric.data(), 2, Binary128{},
                                       Binary128{}, 0, 0, Binary128{}, selected_count, eigenvalues.data(),
                                       eigenvectors.data(), 1, work.data(), -1, rwork.data(), nullptr, nullptr));
}

TEST(MplapackBinary128DenseSubspaceTest, TridiagonalDivideAndConquerWorkspaceQueryAcceptsNullIntegerWork)
{
  std::vector<Binary128> diagonal{Binary128{1}, Binary128{2}};
  std::vector<Binary128> offdiagonal{Binary128{}};
  std::vector<Binary128> eigenvectors(1);
  std::vector<Binary128> work(1);

  EXPECT_NO_THROW(uni20::lapack::stevd('N', 2, diagonal.data(), offdiagonal.data(), eigenvectors.data(), 1, work.data(),
                                       -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, SelectedTridiagonalWorkspaceQueryAcceptsNullSupportAndIntegerWork)
{
  std::vector<Binary128> diagonal{Binary128{1}, Binary128{2}};
  std::vector<Binary128> offdiagonal{Binary128{}};
  std::vector<Binary128> eigenvalues(2);
  std::vector<Binary128> eigenvectors(1);
  std::vector<Binary128> work(1);
  uni20::blas_int selected_count = 0;

  EXPECT_NO_THROW(uni20::lapack::stevr('N', 'A', 2, diagonal.data(), offdiagonal.data(), Binary128{}, Binary128{}, 0, 0,
                                       Binary128{}, selected_count, eigenvalues.data(), eigenvectors.data(), 1, nullptr,
                                       work.data(), -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSchurConditionWorkspaceQueryAcceptsNullIntegerWork)
{
  std::vector<uni20::blas_int> select{1, 0};
  std::vector<Binary128> matrix_schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> metric_schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> alphar(2);
  std::vector<Binary128> alphai(2);
  std::vector<Binary128> beta(2);
  std::vector<Binary128> left_schur_vectors{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> right_schur_vectors{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> dif(2);
  std::vector<Binary128> work(1);
  uni20::blas_int selected_dimension = 0;
  Binary128 pl{};
  Binary128 pr{};

  EXPECT_NO_THROW(uni20::lapack::tgsen(1, true, true, select.data(), 2, matrix_schur_form.data(), 2,
                                       metric_schur_form.data(), 2, alphar.data(), alphai.data(), beta.data(),
                                       left_schur_vectors.data(), 2, right_schur_vectors.data(), 2, selected_dimension,
                                       pl, pr, dif.data(), work.data(), -1, nullptr, -1));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSchurConditionQuickReturnAcceptsNullSelection)
{
  std::vector<Binary128> matrix_schur_form(1);
  std::vector<Binary128> metric_schur_form(1);
  std::vector<Binary128> alphar(1);
  std::vector<Binary128> alphai(1);
  std::vector<Binary128> beta(1);
  std::vector<Binary128> left_schur_vectors(1);
  std::vector<Binary128> right_schur_vectors(1);
  std::vector<Binary128> dif(2);
  std::vector<Binary128> work(1);
  std::vector<uni20::blas_int> iwork(1);
  uni20::blas_int selected_dimension = 0;
  Binary128 pl{};
  Binary128 pr{};

  EXPECT_NO_THROW(uni20::lapack::tgsen(0, false, false, nullptr, 0, matrix_schur_form.data(), 1,
                                       metric_schur_form.data(), 1, alphar.data(), alphai.data(), beta.data(),
                                       left_schur_vectors.data(), 1, right_schur_vectors.data(), 1, selected_dimension,
                                       pl, pr, dif.data(), work.data(), -1, iwork.data(), -1));
  EXPECT_EQ(selected_dimension, 0);
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSchurEigenvectorsAllModeAcceptsNullSelection)
{
  std::vector<Binary128> matrix_schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> metric_schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> left_vectors(4);
  std::vector<Binary128> right_vectors(4);
  std::vector<Binary128> work(12);
  uni20::blas_int computed_vectors = 0;

  EXPECT_NO_THROW(uni20::lapack::tgevc('R', 'A', nullptr, 2, matrix_schur_form.data(), 2, metric_schur_form.data(), 2,
                                       left_vectors.data(), 2, right_vectors.data(), 2, 2, computed_vectors,
                                       work.data()));
}

TEST(MplapackBinary128DenseSubspaceTest, GeneralizedSchurConditionAllModeAcceptsNullSelection)
{
  std::vector<Binary128> matrix_schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{2}};
  std::vector<Binary128> metric_schur_form{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> left_vectors{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> right_vectors{Binary128{1}, Binary128{}, Binary128{}, Binary128{1}};
  std::vector<Binary128> reciprocal_eigenvalue_condition_numbers(2);
  std::vector<Binary128> reciprocal_eigenvector_condition_numbers(2);
  std::vector<Binary128> work(32);
  std::vector<uni20::blas_int> iwork(8);
  uni20::blas_int computed_estimates = 0;

  EXPECT_NO_THROW(uni20::lapack::tgsna(
      'B', 'A', nullptr, 2, matrix_schur_form.data(), 2, metric_schur_form.data(), 2, left_vectors.data(), 2,
      right_vectors.data(), 2, reciprocal_eigenvalue_condition_numbers.data(),
      reciprocal_eigenvector_condition_numbers.data(), 2, computed_estimates, work.data(), 32, iwork.data()));
}

TEST(MplapackBinary128DenseSubspaceTest, SolvesNearlySingularDenseSystemBelowDoubleResolution)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{1};
  matrix[0, 1] = Binary128{1};
  matrix[1, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;

  auto rhs = zero_matrix<Binary128>(2, 1);
  rhs[0, 0] = Binary128{2};
  rhs[1, 0] = Binary128{2} + delta;

  auto solution = uni20::linalg::solve(uni20::linalg::CpuReferenceBackend{}, matrix, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_TRUE(abs_error(solution[0, 0], Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(solution[1, 0], Binary128{1}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, ScalarMatrixExponentialResolvesBelowDoublePrecisionIncrement)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(1, 1);
  matrix[0, 0] = delta;

  auto result = uni20::linalg::backends::cpu::matrix_exponential(matrix, Binary128{1});

  ASSERT_EQ(result.rows(), 1);
  ASSERT_EQ(result.cols(), 1);
  EXPECT_EQ(static_cast<double>(result[0, 0]), 1.0);
  EXPECT_TRUE((result[0, 0] > Binary128{1}));
  EXPECT_TRUE(abs_error((result[0, 0] - Binary128{1}), delta) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, UpperTriangularMatrixExponentialPreservesBinary128OnlyIncrements)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = delta;
  matrix[0, 1] = delta;
  matrix[1, 0] = Binary128{};
  matrix[1, 1] = Binary128{2} * delta;

  auto result = uni20::linalg::backends::cpu::matrix_exponential(matrix, Binary128{1});

  ASSERT_EQ(result.rows(), 2);
  ASSERT_EQ(result.cols(), 2);
  EXPECT_EQ(static_cast<double>(result[0, 0]), 1.0);
  EXPECT_EQ(static_cast<double>(result[1, 1]), 1.0);
  EXPECT_TRUE((result[0, 0] > Binary128{1}));
  EXPECT_TRUE((result[1, 1] > Binary128{1}));
  EXPECT_TRUE(abs_error((result[0, 0] - Binary128{1}), delta) <= tolerance());
  EXPECT_TRUE(abs_error((result[1, 1] - Binary128{1}), Binary128{2} * delta) <= tolerance());
  EXPECT_TRUE(abs_error(result[0, 1], delta) <= tolerance());
  EXPECT_TRUE(abs_error(result[1, 0], Binary128{}) <= tolerance());
}

TEST(MplapackBinary128DenseSubspaceTest, HighNormMatrixExponentialPreservesSmallBinary128BlockIncrement)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  auto matrix = zero_matrix<Binary128>(2, 2);
  matrix[0, 0] = Binary128{6};
  matrix[0, 1] = Binary128{};
  matrix[1, 0] = Binary128{};
  matrix[1, 1] = delta;

  auto result = uni20::linalg::backends::cpu::matrix_exponential(matrix, Binary128{1});

  ASSERT_EQ(result.rows(), 2);
  ASSERT_EQ(result.cols(), 2);
  EXPECT_TRUE((result[0, 0] > Binary128{400}));
  EXPECT_EQ(static_cast<double>(result[1, 1]), 1.0);
  EXPECT_TRUE((result[1, 1] > Binary128{1}));
  EXPECT_TRUE(abs_error((result[1, 1] - Binary128{1}), delta) <= tolerance());
  EXPECT_TRUE(abs_error(result[0, 1], Binary128{}) <= tolerance());
  EXPECT_TRUE(abs_error(result[1, 0], Binary128{}) <= tolerance());
}

} // namespace
