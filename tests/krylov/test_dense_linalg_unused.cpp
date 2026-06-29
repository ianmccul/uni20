#include <uni20/krylov/dense_subspace_unused.hpp>

#include "krylov_test_types.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

template <typename Scalar> class KrylovDenseSubspaceTypedTest : public ::testing::Test {};
template <typename Scalar> class KrylovDenseSubspaceComplexTypedTest : public ::testing::Test {};

using RealTypes = uni20::krylov::test::KrylovRealTestTypes;
using ComplexTypes = uni20::krylov::test::KrylovComplexTestTypes;
TYPED_TEST_SUITE(KrylovDenseSubspaceTypedTest, RealTypes);
TYPED_TEST_SUITE(KrylovDenseSubspaceComplexTypedTest, ComplexTypes);

template <typename Scalar> double tolerance()
{
  if constexpr (std::is_same_v<Scalar, float>)
  {
    return 1.0e-5;
  }
  else
  {
    return 1.0e-12;
  }
}

template <typename Scalar> double scaled_tolerance(double factor) { return factor * tolerance<Scalar>(); }

template <typename Scalar> double abs_as_double(Scalar const& value) { return static_cast<double>(std::abs(value)); }

template <typename Scalar>
void expect_vector_near_values(std::vector<Scalar> const& actual, std::vector<Scalar> const& expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index)
  {
    EXPECT_NEAR(static_cast<double>(actual[index]), static_cast<double>(expected[index]), tolerance<Scalar>());
  }
}

template <typename Scalar>
uni20::krylov::Matrix<Scalar> multiply_for_test(uni20::krylov::Matrix<Scalar> const& lhs,
                                                uni20::krylov::Matrix<Scalar> const& rhs)
{
  if (lhs.cols() != rhs.rows())
  {
    throw std::invalid_argument("test matrix dimensions do not agree");
  }

  uni20::krylov::Matrix<Scalar> result(lhs.rows(), rhs.cols());
  for (std::size_t row = 0; row < lhs.rows(); ++row)
  {
    for (std::size_t inner = 0; inner < lhs.cols(); ++inner)
    {
      Scalar const factor = lhs(row, inner);
      for (std::size_t col = 0; col < rhs.cols(); ++col)
      {
        result(row, col) += factor * rhs(inner, col);
      }
    }
  }
  return result;
}

template <typename Scalar> uni20::krylov::Matrix<Scalar> transpose(uni20::krylov::Matrix<Scalar> const& matrix)
{
  uni20::krylov::Matrix<Scalar> result(matrix.cols(), matrix.rows());
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      result(col, row) = matrix(row, col);
    }
  }
  return result;
}

template <typename Scalar> uni20::krylov::Matrix<Scalar> identity_matrix(std::size_t order)
{
  uni20::krylov::Matrix<Scalar> result(order, order);
  for (std::size_t index = 0; index < order; ++index)
  {
    result(index, index) = Scalar{1};
  }
  return result;
}

template <typename Scalar>
uni20::krylov::Matrix<Scalar> conjugate_transpose(uni20::krylov::Matrix<Scalar> const& matrix)
{
  uni20::krylov::Matrix<Scalar> result(matrix.cols(), matrix.rows());
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      if constexpr (uni20::krylov::detail::is_complex_v<Scalar>)
      {
        result(col, row) = std::conj(matrix(row, col));
      }
      else
      {
        result(col, row) = matrix(row, col);
      }
    }
  }
  return result;
}

template <typename Scalar>
void expect_reconstructs(uni20::krylov::Matrix<Scalar> const& original, uni20::krylov::Matrix<Scalar> const& schur_form,
                         uni20::krylov::Matrix<Scalar> const& schur_vectors)
{
  auto reconstructed = multiply_for_test(multiply_for_test(schur_vectors, schur_form), transpose(schur_vectors));
  for (std::size_t row = 0; row < original.rows(); ++row)
  {
    for (std::size_t col = 0; col < original.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(original(row, col)),
                  scaled_tolerance<Scalar>(20.0));
    }
  }
}

template <typename Scalar>
void expect_complex_reconstructs(uni20::krylov::Matrix<Scalar> const& original,
                                 uni20::krylov::Matrix<Scalar> const& schur_form,
                                 uni20::krylov::Matrix<Scalar> const& schur_vectors)
{
  auto reconstructed =
      multiply_for_test(multiply_for_test(schur_vectors, schur_form), conjugate_transpose(schur_vectors));
  using Real = typename Scalar::value_type;
  for (std::size_t row = 0; row < original.rows(); ++row)
  {
    for (std::size_t col = 0; col < original.cols(); ++col)
    {
      EXPECT_LT(abs_as_double(reconstructed(row, col) - original(row, col)), scaled_tolerance<Real>(50.0));
    }
  }
}

template <typename Scalar>
void expect_generalized_schur_reconstructs(
    uni20::krylov::Matrix<Scalar> const& original_matrix, uni20::krylov::Matrix<Scalar> const& original_metric,
    uni20::krylov::RealGeneralizedSchurDecomposition<Scalar> const& decomposition)
{
  auto reconstructed_matrix =
      multiply_for_test(multiply_for_test(decomposition.left_schur_vectors, decomposition.matrix_schur_form),
                        transpose(decomposition.right_schur_vectors));
  auto reconstructed_metric =
      multiply_for_test(multiply_for_test(decomposition.left_schur_vectors, decomposition.metric_schur_form),
                        transpose(decomposition.right_schur_vectors));
  for (std::size_t row = 0; row < original_matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < original_matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed_matrix(row, col)), static_cast<double>(original_matrix(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      EXPECT_NEAR(static_cast<double>(reconstructed_metric(row, col)), static_cast<double>(original_metric(row, col)),
                  scaled_tolerance<Scalar>(100.0));
    }
  }
}

template <typename Scalar>
Scalar schur_right_eigenvector_residual_max(uni20::krylov::Matrix<Scalar> const& schur_form,
                                            uni20::complex<Scalar> eigenvalue,
                                            uni20::krylov::Matrix<uni20::complex<Scalar>> const& eigenvectors,
                                            std::size_t column)
{
  Scalar residual_max{};
  for (std::size_t row = 0; row < schur_form.rows(); ++row)
  {
    uni20::complex<Scalar> applied{};
    for (std::size_t inner = 0; inner < schur_form.cols(); ++inner)
    {
      applied += uni20::complex<Scalar>{schur_form(row, inner), Scalar{}} * eigenvectors(inner, column);
    }
    residual_max =
        std::max(residual_max, static_cast<Scalar>(std::abs(applied - eigenvalue * eigenvectors(row, column))));
  }
  return residual_max;
}

template <typename Scalar>
Scalar generalized_schur_right_eigenvector_residual_max(
    uni20::krylov::Matrix<Scalar> const& matrix_schur_form, uni20::krylov::Matrix<Scalar> const& metric_schur_form,
    uni20::complex<Scalar> eigenvalue, uni20::krylov::Matrix<uni20::complex<Scalar>> const& eigenvectors,
    std::size_t column)
{
  Scalar residual_max{};
  for (std::size_t row = 0; row < matrix_schur_form.rows(); ++row)
  {
    uni20::complex<Scalar> matrix_applied{};
    uni20::complex<Scalar> metric_applied{};
    for (std::size_t inner = 0; inner < matrix_schur_form.cols(); ++inner)
    {
      matrix_applied += uni20::complex<Scalar>{matrix_schur_form(row, inner), Scalar{}} * eigenvectors(inner, column);
      metric_applied += uni20::complex<Scalar>{metric_schur_form(row, inner), Scalar{}} * eigenvectors(inner, column);
    }
    residual_max = std::max(residual_max, static_cast<Scalar>(std::abs(matrix_applied - eigenvalue * metric_applied)));
  }
  return residual_max;
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, StoresMatrixColumnMajor)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 3);
  matrix(0, 0) = Scalar{1};
  matrix(0, 1) = Scalar{2};
  matrix(0, 2) = Scalar{3};
  matrix(1, 0) = Scalar{4};
  matrix(1, 1) = Scalar{5};
  matrix(1, 2) = Scalar{6};

  expect_vector_near_values(std::vector<Scalar>(matrix.data(), matrix.data() + matrix.size()),
                            std::vector<Scalar>{Scalar{1}, Scalar{4}, Scalar{2}, Scalar{5}, Scalar{3}, Scalar{6}});
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDenseRealMatrixNorms)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{4};
  matrix(0, 1) = Scalar{-2};
  matrix(1, 1) = Scalar{-5};
  matrix(0, 2) = Scalar{3};
  matrix(1, 2) = Scalar{6};

  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_matrix_norm(matrix, uni20::krylov::MatrixNorm::MaxAbs)), 6.0,
              scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_matrix_norm(matrix, uni20::krylov::MatrixNorm::One)), 9.0,
              scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_matrix_norm(matrix, uni20::krylov::MatrixNorm::Infinity)), 15.0,
              scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_matrix_norm(matrix, uni20::krylov::MatrixNorm::Frobenius)),
              std::sqrt(91.0), scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDenseRealSymmetricMatrixNorms)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(0, 1) = Scalar{-3};
  matrix(1, 0) = Scalar{100};
  matrix(1, 1) = Scalar{5};

  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_symmetric_matrix_norm(matrix, uni20::krylov::MatrixNorm::MaxAbs,
                                                                            uni20::krylov::MatrixFill::Upper)),
              5.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_symmetric_matrix_norm(matrix, uni20::krylov::MatrixNorm::One,
                                                                            uni20::krylov::MatrixFill::Upper)),
              8.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_symmetric_matrix_norm(matrix, uni20::krylov::MatrixNorm::Infinity,
                                                                            uni20::krylov::MatrixFill::Upper)),
              8.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_symmetric_matrix_norm(
                  matrix, uni20::krylov::MatrixNorm::Frobenius, uni20::krylov::MatrixFill::Upper)),
              std::sqrt(47.0), scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDenseRealTriangularMatrixNorms)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{3};
  matrix(0, 1) = Scalar{2};
  matrix(1, 0) = Scalar{100};
  matrix(1, 1) = Scalar{4};

  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_triangular_matrix_norm(matrix, uni20::krylov::MatrixNorm::MaxAbs,
                                                                             uni20::krylov::MatrixFill::Upper)),
              4.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_triangular_matrix_norm(matrix, uni20::krylov::MatrixNorm::One,
                                                                             uni20::krylov::MatrixFill::Upper)),
              6.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_triangular_matrix_norm(
                  matrix, uni20::krylov::MatrixNorm::Infinity, uni20::krylov::MatrixFill::Upper)),
              5.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_triangular_matrix_norm(
                  matrix, uni20::krylov::MatrixNorm::Frobenius, uni20::krylov::MatrixFill::Upper)),
              std::sqrt(29.0), scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(uni20::krylov::real_triangular_matrix_norm(matrix, uni20::krylov::MatrixNorm::One,
                                                                             uni20::krylov::MatrixFill::Upper, true)),
              3.0, scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDenseRealEquilibration)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(0, 1) = Scalar{};
  matrix(1, 0) = Scalar{};
  matrix(1, 1) = Scalar{4};

  auto result = uni20::krylov::real_equilibration(matrix);

  ASSERT_EQ(result.row_scale.size(), 2);
  ASSERT_EQ(result.column_scale.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.row_scale[0]), 0.5, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.row_scale[1]), 0.25, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.column_scale[0]), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.column_scale[1]), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.row_condition), 0.5, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.column_condition), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.max_abs), 4.0, scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealLinearSystemWithPivoting)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{2};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 2);
  expected(0, 0) = Scalar{4};
  expected(1, 0) = Scalar{-1};
  expected(0, 1) = Scalar{-2};
  expected(1, 1) = Scalar{5};

  auto rhs = multiply_for_test(coefficients, expected);
  auto solution = uni20::krylov::real_dense_solve_linear_system(coefficients, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 2);
  for (std::size_t row = 0; row < solution.rows(); ++row)
  {
    for (std::size_t col = 0; col < solution.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(solution(row, col)), static_cast<double>(expected(row, col)),
                  tolerance<Scalar>());
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, RefinesDenseRealLinearSystemWithDiagnostics)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{2};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{0.2};
  expected(1, 0) = Scalar{0.6};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_refined_linear_solve(coefficients, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), static_cast<double>(expected(0, 0)),
              scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), static_cast<double>(expected(1, 0)),
              scaled_tolerance<Scalar>(50.0));
  EXPECT_GE(result.forward_error_bounds[0], Scalar{});
  EXPECT_GE(result.backward_error_bounds[0], Scalar{});
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.forward_error_bounds[0])));
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.backward_error_bounds[0])));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealExpertLinearSystemWithDiagnostics)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{2};
  coefficients(1, 0) = Scalar{1};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{2};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_expert_linear_solve(coefficients, rhs);
  auto reconstructed = multiply_for_test(coefficients, result.solution);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factors.rows(), 2);
  ASSERT_EQ(result.factors.cols(), 2);
  ASSERT_EQ(result.pivot_rows.size(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_EQ(result.equilibration, 'N');
  EXPECT_FALSE(result.reciprocal_condition_below_machine_precision);
  EXPECT_GT(static_cast<double>(result.reciprocal_condition), 0.0);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), 2.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(0, 0)), static_cast<double>(rhs(0, 0)), scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(1, 0)), static_cast<double>(rhs(1, 0)), scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesFromDenseRealLuFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{2};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 2);
  expected(0, 0) = Scalar{4};
  expected(1, 0) = Scalar{-1};
  expected(0, 1) = Scalar{-2};
  expected(1, 1) = Scalar{5};

  auto rhs = multiply_for_test(coefficients, expected);
  auto factorization = uni20::krylov::real_lu_factorization(coefficients);
  auto solution = uni20::krylov::real_lu_solve(factorization, rhs);

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  ASSERT_EQ(factorization.pivot_rows.size(), 2);
  EXPECT_EQ(factorization.pivot_rows[0], 1);
  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 2);
  for (std::size_t row = 0; row < solution.rows(); ++row)
  {
    for (std::size_t col = 0; col < solution.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(solution(row, col)), static_cast<double>(expected(row, col)),
                  tolerance<Scalar>());
    }
  }

  uni20::krylov::Matrix<Scalar> transposed_expected(2, 1);
  transposed_expected(0, 0) = Scalar{3};
  transposed_expected(1, 0) = Scalar{-2};
  auto transposed_rhs = multiply_for_test(transpose(coefficients), transposed_expected);
  auto transposed_solution =
      uni20::krylov::real_lu_solve(factorization, transposed_rhs, uni20::krylov::MatrixTranspose::Transpose);
  for (std::size_t row = 0; row < transposed_solution.rows(); ++row)
  {
    EXPECT_NEAR(static_cast<double>(transposed_solution(row, 0)), static_cast<double>(transposed_expected(row, 0)),
                tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesDenseRealReciprocalConditionNumber)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{1};
  matrix(1, 1) = Scalar{0.25};

  auto factorization = uni20::krylov::real_lu_factorization(matrix);
  Scalar const from_factorization =
      uni20::krylov::real_lu_one_norm_reciprocal_condition_number(factorization, Scalar{1});
  Scalar const direct = uni20::krylov::real_one_norm_reciprocal_condition_number(matrix);

  EXPECT_NEAR(static_cast<double>(from_factorization), 0.25, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(direct), 0.25, scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealTriangularSystem)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{2};
  coefficients(0, 1) = Scalar{-1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 2);
  expected(0, 0) = Scalar{4};
  expected(1, 0) = Scalar{-1};
  expected(0, 1) = Scalar{-2};
  expected(1, 1) = Scalar{5};

  auto rhs = multiply_for_test(coefficients, expected);
  auto solution = uni20::krylov::real_triangular_solve(coefficients, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 2);
  for (std::size_t row = 0; row < solution.rows(); ++row)
  {
    for (std::size_t col = 0; col < solution.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(solution(row, col)), static_cast<double>(expected(row, col)),
                  scaled_tolerance<Scalar>(20.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, RefinesDenseRealTriangularSystemWithDiagnostics)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{2};
  coefficients(0, 1) = Scalar{-1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{4};
  expected(1, 0) = Scalar{-1};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_triangular_refined_solve(coefficients, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), 4.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), -1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_GE(result.forward_error_bounds[0], Scalar{});
  EXPECT_GE(result.backward_error_bounds[0], Scalar{});
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.forward_error_bounds[0])));
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.backward_error_bounds[0])));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, InvertsDenseRealTriangularMatrix)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(0, 1) = Scalar{-1};
  matrix(1, 1) = Scalar{3};

  auto inverse = uni20::krylov::real_triangular_inverse(matrix);
  auto product = multiply_for_test(matrix, inverse);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  EXPECT_NEAR(static_cast<double>(inverse(1, 0)), 0.0, tolerance<Scalar>());
  for (std::size_t row = 0; row < product.rows(); ++row)
  {
    for (std::size_t col = 0; col < product.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(product(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesDenseRealTriangularReciprocalConditionNumber)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(1, 1) = Scalar{4};

  Scalar const rcond = uni20::krylov::real_triangular_one_norm_reciprocal_condition_number(matrix);

  EXPECT_NEAR(static_cast<double>(rcond), 0.5, scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealSylvesterEquation)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> left(1, 1);
  left(0, 0) = Scalar{2};

  uni20::krylov::Matrix<Scalar> right(1, 1);
  right(0, 0) = Scalar{3};

  uni20::krylov::Matrix<Scalar> rhs(1, 1);
  rhs(0, 0) = Scalar{10};

  auto result = uni20::krylov::real_sylvester_solve(left, right, rhs);

  ASSERT_EQ(result.solution.rows(), 1);
  ASSERT_EQ(result.solution.cols(), 1);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), 2.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.scale), 1.0, tolerance<Scalar>());
  EXPECT_FALSE(result.separation_perturbed);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, InvertsDenseRealMatrix)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{4};
  matrix(1, 0) = Scalar{2};
  matrix(0, 1) = Scalar{7};
  matrix(1, 1) = Scalar{6};

  auto inverse = uni20::krylov::real_dense_inverse(matrix);
  auto product = multiply_for_test(matrix, inverse);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  for (std::size_t row = 0; row < product.rows(); ++row)
  {
    for (std::size_t col = 0; col < product.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(product(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealLeastSquaresProblem)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(3, 2);
  coefficients(0, 0) = Scalar{1};
  coefficients(1, 0) = Scalar{};
  coefficients(2, 0) = Scalar{1};
  coefficients(0, 1) = Scalar{};
  coefficients(1, 1) = Scalar{1};
  coefficients(2, 1) = Scalar{1};

  uni20::krylov::Matrix<Scalar> expected(2, 2);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{3};
  expected(0, 1) = Scalar{-2};
  expected(1, 1) = Scalar{0.5};

  auto rhs = multiply_for_test(coefficients, expected);
  auto solution = uni20::krylov::real_least_squares(coefficients, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 2);
  for (std::size_t row = 0; row < solution.rows(); ++row)
  {
    for (std::size_t col = 0; col < solution.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(solution(row, col)), static_cast<double>(expected(row, col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseSvdLeastSquaresProblem)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{1};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{2};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_svd_least_squares(coefficients, rhs, static_cast<Scalar>(1.0e-4));
  auto reconstructed = multiply_for_test(coefficients, result.solution);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.singular_values.size(), 2);
  EXPECT_EQ(result.rank, 1);
  EXPECT_NEAR(static_cast<double>(result.singular_values[0]), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.singular_values[1]), 0.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), 2.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), 0.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(0, 0)), static_cast<double>(rhs(0, 0)), scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(1, 0)), static_cast<double>(rhs(1, 0)), scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseDivideAndConquerSvdLeastSquaresProblem)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{1};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{2};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result =
      uni20::krylov::real_divide_and_conquer_svd_least_squares(coefficients, rhs, static_cast<Scalar>(1.0e-4));
  auto reconstructed = multiply_for_test(coefficients, result.solution);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.singular_values.size(), 2);
  EXPECT_EQ(result.rank, 1);
  EXPECT_NEAR(static_cast<double>(result.singular_values[0]), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.singular_values[1]), 0.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), 2.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), 0.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(0, 0)), static_cast<double>(rhs(0, 0)), scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(1, 0)), static_cast<double>(rhs(1, 0)), scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRankRevealingLeastSquaresProblem)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(3, 2);
  coefficients(0, 0) = Scalar{1};
  coefficients(1, 0) = Scalar{2};
  coefficients(2, 0) = Scalar{3};
  coefficients(0, 1) = Scalar{2};
  coefficients(1, 1) = Scalar{4};
  coefficients(2, 1) = Scalar{6};

  uni20::krylov::Matrix<Scalar> expected_one_solution(2, 1);
  expected_one_solution(0, 0) = Scalar{3};
  expected_one_solution(1, 0) = Scalar{-1};
  auto rhs = multiply_for_test(coefficients, expected_one_solution);

  auto result = uni20::krylov::real_rank_revealing_least_squares(coefficients, rhs, static_cast<Scalar>(1.0e-4));
  auto reconstructed = multiply_for_test(coefficients, result.solution);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  EXPECT_EQ(result.rank, 1);
  ASSERT_EQ(result.pivot_columns.size(), 2);
  auto sorted_pivots = result.pivot_columns;
  std::sort(sorted_pivots.begin(), sorted_pivots.end());
  EXPECT_EQ(sorted_pivots[0], 0);
  EXPECT_EQ(sorted_pivots[1], 1);
  for (std::size_t row = 0; row < rhs.rows(); ++row)
  {
    EXPECT_NEAR(static_cast<double>(reconstructed(row, 0)), static_cast<double>(rhs(row, 0)),
                scaled_tolerance<Scalar>(100.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseSymmetricPositiveDefiniteSystem)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 2);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{3};
  expected(0, 1) = Scalar{-2};
  expected(1, 1) = Scalar{0.5};

  auto rhs = multiply_for_test(coefficients, expected);
  auto solution = uni20::krylov::real_symmetric_positive_definite_solve(coefficients, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 2);
  for (std::size_t row = 0; row < solution.rows(); ++row)
  {
    for (std::size_t col = 0; col < solution.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(solution(row, col)), static_cast<double>(expected(row, col)),
                  scaled_tolerance<Scalar>(20.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseSymmetricPositiveDefiniteSystemFromFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{2};

  auto rhs = multiply_for_test(coefficients, expected);
  auto factorization = uni20::krylov::real_symmetric_positive_definite_factorization(coefficients);
  auto solution = uni20::krylov::real_symmetric_positive_definite_solve(factorization, rhs);

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  EXPECT_EQ(factorization.triangle, uni20::krylov::MatrixFill::Upper);
  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_NEAR(static_cast<double>(solution(0, 0)), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(solution(1, 0)), 2.0, scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDensePivotedCholeskyFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(1, 1) = Scalar{1};

  auto factorization = uni20::krylov::real_pivoted_cholesky_factorization(coefficients, Scalar{});

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  ASSERT_EQ(factorization.pivot_order.size(), 2);
  EXPECT_EQ(factorization.rank, 2);
  EXPECT_FALSE(factorization.rank_deficient);
  EXPECT_EQ(factorization.triangle, uni20::krylov::MatrixFill::Upper);

  uni20::krylov::Matrix<Scalar> upper(2, 2);
  for (std::size_t row = 0; row < 2; ++row)
  {
    for (std::size_t col = row; col < 2; ++col)
    {
      upper(row, col) = factorization.factors(row, col);
    }
  }
  auto reconstructed = multiply_for_test(transpose(upper), upper);
  for (std::size_t row = 0; row < 2; ++row)
  {
    for (std::size_t col = 0; col < 2; ++col)
    {
      Scalar const expected = coefficients(factorization.pivot_order[row], factorization.pivot_order[col]);
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ReportsRankDeficientPivotedCholeskyFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};

  auto factorization = uni20::krylov::real_pivoted_cholesky_factorization(coefficients);

  ASSERT_EQ(factorization.pivot_order.size(), 2);
  EXPECT_EQ(factorization.rank, 1);
  EXPECT_TRUE(factorization.rank_deficient);
  EXPECT_EQ(factorization.pivot_order[0], 0);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, RefinesDenseSymmetricPositiveDefiniteSystemWithDiagnostics)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{1} / Scalar{11};
  expected(1, 0) = Scalar{7} / Scalar{11};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_symmetric_positive_definite_refined_solve(coefficients, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), static_cast<double>(expected(0, 0)),
              scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), static_cast<double>(expected(1, 0)),
              scaled_tolerance<Scalar>(50.0));
  EXPECT_GE(result.forward_error_bounds[0], Scalar{});
  EXPECT_GE(result.backward_error_bounds[0], Scalar{});
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.forward_error_bounds[0])));
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.backward_error_bounds[0])));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseSymmetricPositiveDefiniteExpertSystemWithDiagnostics)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{2};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_symmetric_positive_definite_expert_solve(coefficients, rhs);
  auto reconstructed = multiply_for_test(coefficients, result.solution);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factors.rows(), 2);
  ASSERT_EQ(result.factors.cols(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  ASSERT_EQ(result.scale.size(), 2);
  EXPECT_EQ(result.equilibration, 'N');
  EXPECT_FALSE(result.reciprocal_condition_below_machine_precision);
  EXPECT_GT(static_cast<double>(result.reciprocal_condition), 0.0);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), 1.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), 2.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(0, 0)), static_cast<double>(rhs(0, 0)),
              scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(1, 0)), static_cast<double>(rhs(1, 0)),
              scaled_tolerance<Scalar>(100.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDenseSymmetricPositiveDefiniteEquilibration)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{};
  coefficients(1, 0) = Scalar{};
  coefficients(1, 1) = Scalar{9};

  auto result = uni20::krylov::real_symmetric_positive_definite_equilibration(coefficients);

  ASSERT_EQ(result.scale.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.scale[0]), 0.5, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.scale[1]), 1.0 / 3.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.scale_condition), 2.0 / 3.0, scaled_tolerance<Scalar>(20.0));
  EXPECT_NEAR(static_cast<double>(result.max_abs), 9.0, scaled_tolerance<Scalar>(20.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesDenseSymmetricPositiveDefiniteReciprocalConditionNumber)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  Scalar const rcond = uni20::krylov::real_symmetric_positive_definite_one_norm_reciprocal_condition_number(
      coefficients, uni20::krylov::MatrixFill::Upper);

  EXPECT_NEAR(static_cast<double>(rcond), 11.0 / 25.0, scaled_tolerance<Scalar>(50.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesDenseSymmetricPositiveDefiniteFactorizedReciprocalConditionNumber)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  auto factorization = uni20::krylov::real_symmetric_positive_definite_factorization(coefficients);
  Scalar const rcond =
      uni20::krylov::real_symmetric_positive_definite_one_norm_reciprocal_condition_number(factorization, Scalar{5});

  EXPECT_NEAR(static_cast<double>(rcond), 11.0 / 25.0, scaled_tolerance<Scalar>(50.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, InvertsDenseSymmetricPositiveDefiniteMatrix)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{4};
  coefficients(0, 1) = Scalar{1};
  coefficients(1, 0) = Scalar{1};
  coefficients(1, 1) = Scalar{3};

  auto inverse =
      uni20::krylov::real_symmetric_positive_definite_inverse(coefficients, uni20::krylov::MatrixFill::Upper);
  auto product = multiply_for_test(coefficients, inverse);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  EXPECT_NEAR(static_cast<double>(inverse(0, 0)), 3.0 / 11.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(inverse(0, 1)), -1.0 / 11.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(inverse(1, 0)), -1.0 / 11.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(inverse(1, 1)), 4.0 / 11.0, scaled_tolerance<Scalar>(50.0));
  for (std::size_t row = 0; row < product.rows(); ++row)
  {
    for (std::size_t col = 0; col < product.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(product(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(100.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseSymmetricIndefiniteSystem)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{};
  coefficients(0, 1) = Scalar{2};
  coefficients(1, 0) = Scalar{2};
  coefficients(1, 1) = Scalar{-3};

  uni20::krylov::Matrix<Scalar> expected(2, 2);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{3};
  expected(0, 1) = Scalar{-2};
  expected(1, 1) = Scalar{0.5};

  auto rhs = multiply_for_test(coefficients, expected);
  auto solution = uni20::krylov::real_symmetric_indefinite_solve(coefficients, rhs);

  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 2);
  for (std::size_t row = 0; row < solution.rows(); ++row)
  {
    for (std::size_t col = 0; col < solution.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(solution(row, col)), static_cast<double>(expected(row, col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseSymmetricIndefiniteSystemFromFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{};
  coefficients(0, 1) = Scalar{2};
  coefficients(1, 0) = Scalar{2};
  coefficients(1, 1) = Scalar{-3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{3};

  auto rhs = multiply_for_test(coefficients, expected);
  auto factorization = uni20::krylov::real_symmetric_indefinite_factorization(coefficients);
  auto solution = uni20::krylov::real_symmetric_indefinite_solve(factorization, rhs);

  ASSERT_EQ(factorization.factors.rows(), 2);
  ASSERT_EQ(factorization.factors.cols(), 2);
  ASSERT_EQ(factorization.pivot_blocks.size(), 2);
  EXPECT_EQ(factorization.triangle, uni20::krylov::MatrixFill::Upper);
  ASSERT_EQ(solution.rows(), 2);
  ASSERT_EQ(solution.cols(), 1);
  EXPECT_NEAR(static_cast<double>(solution(0, 0)), 1.0, scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(solution(1, 0)), 3.0, scaled_tolerance<Scalar>(100.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, RefinesDenseSymmetricIndefiniteSystemWithDiagnostics)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{};
  coefficients(0, 1) = Scalar{2};
  coefficients(1, 0) = Scalar{2};
  coefficients(1, 1) = Scalar{-3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{3};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_symmetric_indefinite_refined_solve(coefficients, rhs);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), static_cast<double>(expected(0, 0)),
              scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), static_cast<double>(expected(1, 0)),
              scaled_tolerance<Scalar>(100.0));
  EXPECT_GE(result.forward_error_bounds[0], Scalar{});
  EXPECT_GE(result.backward_error_bounds[0], Scalar{});
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.forward_error_bounds[0])));
  EXPECT_TRUE(std::isfinite(static_cast<double>(result.backward_error_bounds[0])));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, InvertsDenseSymmetricIndefiniteMatrix)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{};
  coefficients(0, 1) = Scalar{2};
  coefficients(1, 0) = Scalar{2};
  coefficients(1, 1) = Scalar{-3};

  auto inverse = uni20::krylov::real_symmetric_indefinite_inverse(coefficients);
  auto product = multiply_for_test(coefficients, inverse);

  ASSERT_EQ(inverse.rows(), 2);
  ASSERT_EQ(inverse.cols(), 2);
  EXPECT_NEAR(static_cast<double>(inverse(0, 0)), 0.75, scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(inverse(0, 1)), 0.5, scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(inverse(1, 0)), 0.5, scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(inverse(1, 1)), 0.0, scaled_tolerance<Scalar>(100.0));
  for (std::size_t row = 0; row < product.rows(); ++row)
  {
    for (std::size_t col = 0; col < product.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(product(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(100.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseSymmetricIndefiniteExpertSystemWithDiagnostics)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{};
  coefficients(0, 1) = Scalar{2};
  coefficients(1, 0) = Scalar{2};
  coefficients(1, 1) = Scalar{-3};

  uni20::krylov::Matrix<Scalar> expected(2, 1);
  expected(0, 0) = Scalar{1};
  expected(1, 0) = Scalar{3};

  auto rhs = multiply_for_test(coefficients, expected);
  auto result = uni20::krylov::real_symmetric_indefinite_expert_solve(coefficients, rhs);
  auto reconstructed = multiply_for_test(coefficients, result.solution);

  ASSERT_EQ(result.solution.rows(), 2);
  ASSERT_EQ(result.solution.cols(), 1);
  ASSERT_EQ(result.factors.rows(), 2);
  ASSERT_EQ(result.factors.cols(), 2);
  ASSERT_EQ(result.pivot_blocks.size(), 2);
  ASSERT_EQ(result.forward_error_bounds.size(), 1);
  ASSERT_EQ(result.backward_error_bounds.size(), 1);
  EXPECT_FALSE(result.reciprocal_condition_below_machine_precision);
  EXPECT_GT(static_cast<double>(result.reciprocal_condition), 0.0);
  EXPECT_NEAR(static_cast<double>(result.solution(0, 0)), 1.0, scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(result.solution(1, 0)), 3.0, scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(0, 0)), static_cast<double>(rhs(0, 0)),
              scaled_tolerance<Scalar>(100.0));
  EXPECT_NEAR(static_cast<double>(reconstructed(1, 0)), static_cast<double>(rhs(1, 0)),
              scaled_tolerance<Scalar>(100.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesDenseSymmetricIndefiniteReciprocalConditionNumber)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{-2};
  coefficients(0, 1) = Scalar{};
  coefficients(1, 0) = Scalar{99};
  coefficients(1, 1) = Scalar{4};

  Scalar const rcond = uni20::krylov::real_symmetric_indefinite_one_norm_reciprocal_condition_number(
      coefficients, uni20::krylov::MatrixFill::Upper);

  EXPECT_NEAR(static_cast<double>(rcond), 0.5, scaled_tolerance<Scalar>(50.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesDenseSymmetricIndefiniteFactorizedReciprocalConditionNumber)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> coefficients(2, 2);
  coefficients(0, 0) = Scalar{-2};
  coefficients(0, 1) = Scalar{};
  coefficients(1, 0) = Scalar{99};
  coefficients(1, 1) = Scalar{4};

  auto factorization = uni20::krylov::real_symmetric_indefinite_factorization(coefficients);
  Scalar const rcond =
      uni20::krylov::real_symmetric_indefinite_one_norm_reciprocal_condition_number(factorization, Scalar{4});

  EXPECT_NEAR(static_cast<double>(rcond), 0.5, scaled_tolerance<Scalar>(50.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealSymmetricEigenvectors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(0, 1) = Scalar{1};
  matrix(1, 0) = Scalar{};
  matrix(1, 1) = Scalar{2};

  auto result = uni20::krylov::real_symmetric_eigensystem(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar const x0 = result.eigenvectors(0, col);
    Scalar const x1 = result.eigenvectors(1, col);
    Scalar const residual0 = Scalar{2} * x0 + x1 - lambda * x0;
    Scalar const residual1 = x0 + Scalar{2} * x1 - lambda * x1;
    Scalar const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Scalar const vector_norm = std::sqrt(x0 * x0 + x1 * x1);
    EXPECT_LT(static_cast<double>(residual_norm), tolerance<Scalar>());
    EXPECT_NEAR(static_cast<double>(vector_norm), 1.0, tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealSymmetricEigenvaluesFromLowerTriangle)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(0, 1) = Scalar{99};
  matrix(1, 0) = Scalar{1};
  matrix(1, 1) = Scalar{2};

  auto result = uni20::krylov::real_symmetric_eigensystem(matrix, false, uni20::krylov::MatrixFill::Lower);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());
  EXPECT_EQ(result.eigenvectors.rows(), 0);
  EXPECT_EQ(result.eigenvectors.cols(), 0);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseRealSymmetricDivideAndConquerEigenvectors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(0, 1) = Scalar{1};
  matrix(1, 0) = Scalar{};
  matrix(1, 1) = Scalar{2};

  auto result = uni20::krylov::real_symmetric_eigensystem_divide_and_conquer(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar const x0 = result.eigenvectors(0, col);
    Scalar const x1 = result.eigenvectors(1, col);
    Scalar const residual0 = Scalar{2} * x0 + x1 - lambda * x0;
    Scalar const residual1 = x0 + Scalar{2} * x1 - lambda * x1;
    Scalar const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Scalar const vector_norm = std::sqrt(x0 * x0 + x1 * x1);
    EXPECT_LT(static_cast<double>(residual_norm), tolerance<Scalar>());
    EXPECT_NEAR(static_cast<double>(vector_norm), 1.0, tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesSelectedDenseRealSymmetricEigenvectors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 1) = Scalar{2};
  matrix(2, 2) = Scalar{3};

  auto result = uni20::krylov::real_symmetric_eigensystem_index_range(matrix, 1, 2, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar residual_norm{};
    Scalar vector_norm{};
    for (std::size_t row = 0; row < result.eigenvectors.rows(); ++row)
    {
      Scalar const diagonal = Scalar{1} + static_cast<Scalar>(row);
      Scalar const residual = diagonal * result.eigenvectors(row, col) - lambda * result.eigenvectors(row, col);
      residual_norm += residual * residual;
      vector_norm += result.eigenvectors(row, col) * result.eigenvectors(row, col);
    }
    EXPECT_LT(static_cast<double>(std::sqrt(residual_norm)), tolerance<Scalar>());
    EXPECT_NEAR(static_cast<double>(std::sqrt(vector_norm)), 1.0, tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseGeneralizedRealSymmetricEigenvectors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(1, 1) = Scalar{9};

  uni20::krylov::Matrix<Scalar> metric(2, 2);
  metric(0, 0) = Scalar{1};
  metric(1, 1) = Scalar{3};

  auto result = uni20::krylov::real_generalized_symmetric_eigensystem(matrix, metric, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar const x0 = result.eigenvectors(0, col);
    Scalar const x1 = result.eigenvectors(1, col);
    Scalar const residual0 = Scalar{2} * x0 - lambda * x0;
    Scalar const residual1 = Scalar{9} * x1 - lambda * Scalar{3} * x1;
    Scalar const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Scalar const metric_norm = x0 * x0 + Scalar{3} * x1 * x1;
    EXPECT_LT(static_cast<double>(residual_norm), scaled_tolerance<Scalar>(20.0));
    EXPECT_NEAR(static_cast<double>(metric_norm), 1.0, scaled_tolerance<Scalar>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesDenseGeneralizedRealSymmetricDivideAndConquerEigenvectors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(1, 1) = Scalar{9};

  uni20::krylov::Matrix<Scalar> metric(2, 2);
  metric(0, 0) = Scalar{1};
  metric(1, 1) = Scalar{3};

  auto result = uni20::krylov::real_generalized_symmetric_eigensystem_divide_and_conquer(matrix, metric, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar const x0 = result.eigenvectors(0, col);
    Scalar const x1 = result.eigenvectors(1, col);
    Scalar const residual0 = Scalar{2} * x0 - lambda * x0;
    Scalar const residual1 = Scalar{9} * x1 - lambda * Scalar{3} * x1;
    Scalar const residual_norm = std::sqrt(residual0 * residual0 + residual1 * residual1);
    Scalar const metric_norm = x0 * x0 + Scalar{3} * x1 * x1;
    EXPECT_LT(static_cast<double>(residual_norm), scaled_tolerance<Scalar>(20.0));
    EXPECT_NEAR(static_cast<double>(metric_norm), 1.0, scaled_tolerance<Scalar>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesSelectedDenseGeneralizedRealSymmetricEigenvectors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 3);
  matrix(0, 0) = Scalar{2};
  matrix(1, 1) = Scalar{6};
  matrix(2, 2) = Scalar{12};

  uni20::krylov::Matrix<Scalar> metric(3, 3);
  metric(0, 0) = Scalar{1};
  metric(1, 1) = Scalar{2};
  metric(2, 2) = Scalar{3};

  auto result = uni20::krylov::real_generalized_symmetric_eigensystem_index_range(matrix, metric, 1, 2, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 3.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 4.0, tolerance<Scalar>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar residual_norm{};
    Scalar metric_norm{};
    for (std::size_t row = 0; row < result.eigenvectors.rows(); ++row)
    {
      Scalar const matrix_diagonal = row == 0 ? Scalar{2} : (row == 1 ? Scalar{6} : Scalar{12});
      Scalar const metric_diagonal = Scalar{1} + static_cast<Scalar>(row);
      Scalar const value = result.eigenvectors(row, col);
      Scalar const residual = matrix_diagonal * value - lambda * metric_diagonal * value;
      residual_norm += residual * residual;
      metric_norm += metric_diagonal * value * value;
    }
    EXPECT_LT(static_cast<double>(std::sqrt(residual_norm)), scaled_tolerance<Scalar>(20.0));
    EXPECT_NEAR(static_cast<double>(metric_norm), 1.0, scaled_tolerance<Scalar>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, SolvesDenseComplexHermitianEigenvectors)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  uni20::krylov::Matrix<Complex> matrix(2, 2);
  matrix(0, 0) = Complex{2};
  matrix(0, 1) = Complex{0, 1};
  matrix(1, 1) = Complex{2};

  auto result = uni20::krylov::complex_hermitian_eigensystem(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Real>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Real const lambda = result.eigenvalues[col];
    Complex const x0 = result.eigenvectors(0, col);
    Complex const x1 = result.eigenvectors(1, col);
    Complex const residual0 = Complex{2} * x0 + Complex{0, 1} * x1 - lambda * x0;
    Complex const residual1 = Complex{0, -1} * x0 + Complex{2} * x1 - lambda * x1;
    Real const residual_norm = std::sqrt(std::norm(residual0) + std::norm(residual1));
    Real const vector_norm = std::sqrt(std::norm(x0) + std::norm(x1));
    EXPECT_LT(static_cast<double>(residual_norm), scaled_tolerance<Real>(20.0));
    EXPECT_NEAR(static_cast<double>(vector_norm), 1.0, scaled_tolerance<Real>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, SolvesDenseComplexHermitianDivideAndConquerEigenvectors)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  uni20::krylov::Matrix<Complex> matrix(2, 2);
  matrix(0, 0) = Complex{2};
  matrix(0, 1) = Complex{0, 1};
  matrix(1, 1) = Complex{2};

  auto result = uni20::krylov::complex_hermitian_eigensystem_divide_and_conquer(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Real>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Real const lambda = result.eigenvalues[col];
    Complex const x0 = result.eigenvectors(0, col);
    Complex const x1 = result.eigenvectors(1, col);
    Complex const residual0 = Complex{2} * x0 + Complex{0, 1} * x1 - lambda * x0;
    Complex const residual1 = Complex{0, -1} * x0 + Complex{2} * x1 - lambda * x1;
    Real const residual_norm = std::sqrt(std::norm(residual0) + std::norm(residual1));
    Real const vector_norm = std::sqrt(std::norm(x0) + std::norm(x1));
    EXPECT_LT(static_cast<double>(residual_norm), scaled_tolerance<Real>(20.0));
    EXPECT_NEAR(static_cast<double>(vector_norm), 1.0, scaled_tolerance<Real>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, SolvesSelectedDenseComplexHermitianEigenvectors)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  uni20::krylov::Matrix<Complex> matrix(3, 3);
  matrix(0, 0) = Complex{1};
  matrix(1, 1) = Complex{2};
  matrix(2, 2) = Complex{3};

  auto result = uni20::krylov::complex_hermitian_eigensystem_index_range(matrix, 1, 2, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Real>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Real const lambda = result.eigenvalues[col];
    Real residual_norm{};
    Real vector_norm{};
    for (std::size_t row = 0; row < result.eigenvectors.rows(); ++row)
    {
      Real const diagonal = Real{1} + static_cast<Real>(row);
      Complex const value = result.eigenvectors(row, col);
      residual_norm += std::norm(diagonal * value - lambda * value);
      vector_norm += std::norm(value);
    }
    EXPECT_LT(static_cast<double>(std::sqrt(residual_norm)), scaled_tolerance<Real>(20.0));
    EXPECT_NEAR(static_cast<double>(std::sqrt(vector_norm)), 1.0, scaled_tolerance<Real>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, SolvesDenseGeneralizedComplexHermitianEigenvectors)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  Real const root_three = std::sqrt(Real{3});
  uni20::krylov::Matrix<Complex> matrix(2, 2);
  matrix(0, 0) = Complex{2};
  matrix(0, 1) = Complex{0, root_three};
  matrix(1, 1) = Complex{6};

  uni20::krylov::Matrix<Complex> metric(2, 2);
  metric(0, 0) = Complex{1};
  metric(1, 1) = Complex{3};

  auto result = uni20::krylov::complex_generalized_hermitian_eigensystem(matrix, metric, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, scaled_tolerance<Real>(20.0));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, scaled_tolerance<Real>(20.0));

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Real const lambda = result.eigenvalues[col];
    Complex const x0 = result.eigenvectors(0, col);
    Complex const x1 = result.eigenvectors(1, col);
    Complex const residual0 = Complex{2} * x0 + Complex{0, root_three} * x1 - lambda * x0;
    Complex const residual1 = Complex{0, -root_three} * x0 + Complex{6} * x1 - lambda * Complex{3} * x1;
    Real const residual_norm = std::sqrt(std::norm(residual0) + std::norm(residual1));
    Real const metric_norm = std::norm(x0) + Real{3} * std::norm(x1);
    EXPECT_LT(static_cast<double>(residual_norm), scaled_tolerance<Real>(50.0));
    EXPECT_NEAR(static_cast<double>(metric_norm), 1.0, scaled_tolerance<Real>(50.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, SolvesDenseGeneralizedComplexHermitianDivideAndConquerEigenvectors)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  Real const root_three = std::sqrt(Real{3});
  uni20::krylov::Matrix<Complex> matrix(2, 2);
  matrix(0, 0) = Complex{2};
  matrix(0, 1) = Complex{0, root_three};
  matrix(1, 1) = Complex{6};

  uni20::krylov::Matrix<Complex> metric(2, 2);
  metric(0, 0) = Complex{1};
  metric(1, 1) = Complex{3};

  auto result = uni20::krylov::complex_generalized_hermitian_eigensystem_divide_and_conquer(matrix, metric, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, scaled_tolerance<Real>(20.0));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, scaled_tolerance<Real>(20.0));

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Real const lambda = result.eigenvalues[col];
    Complex const x0 = result.eigenvectors(0, col);
    Complex const x1 = result.eigenvectors(1, col);
    Complex const residual0 = Complex{2} * x0 + Complex{0, root_three} * x1 - lambda * x0;
    Complex const residual1 = Complex{0, -root_three} * x0 + Complex{6} * x1 - lambda * Complex{3} * x1;
    Real const residual_norm = std::sqrt(std::norm(residual0) + std::norm(residual1));
    Real const metric_norm = std::norm(x0) + Real{3} * std::norm(x1);
    EXPECT_LT(static_cast<double>(residual_norm), scaled_tolerance<Real>(50.0));
    EXPECT_NEAR(static_cast<double>(metric_norm), 1.0, scaled_tolerance<Real>(50.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, SolvesSelectedDenseGeneralizedComplexHermitianEigenvectors)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  uni20::krylov::Matrix<Complex> matrix(3, 3);
  matrix(0, 0) = Complex{2};
  matrix(1, 1) = Complex{6};
  matrix(2, 2) = Complex{12};

  uni20::krylov::Matrix<Complex> metric(3, 3);
  metric(0, 0) = Complex{1};
  metric(1, 1) = Complex{2};
  metric(2, 2) = Complex{3};

  auto result = uni20::krylov::complex_generalized_hermitian_eigensystem_index_range(matrix, metric, 1, 2, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 3.0, scaled_tolerance<Real>(20.0));
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 4.0, scaled_tolerance<Real>(20.0));

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Real const lambda = result.eigenvalues[col];
    Real residual_norm{};
    Real metric_norm{};
    for (std::size_t row = 0; row < result.eigenvectors.rows(); ++row)
    {
      Real const matrix_diagonal = row == 0 ? Real{2} : (row == 1 ? Real{6} : Real{12});
      Real const metric_diagonal = Real{1} + static_cast<Real>(row);
      Complex const value = result.eigenvectors(row, col);
      residual_norm += std::norm(matrix_diagonal * value - lambda * metric_diagonal * value);
      metric_norm += metric_diagonal * std::norm(value);
    }
    EXPECT_LT(static_cast<double>(std::sqrt(residual_norm)), scaled_tolerance<Real>(50.0));
    EXPECT_NEAR(static_cast<double>(metric_norm), 1.0, scaled_tolerance<Real>(50.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesReducedRealQrFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 2);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(2, 0) = Scalar{};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(2, 1) = Scalar{1};

  auto result = uni20::krylov::real_qr_factorization(matrix);

  ASSERT_EQ(result.q.rows(), 3);
  ASSERT_EQ(result.q.cols(), 2);
  ASSERT_EQ(result.r.rows(), 2);
  ASSERT_EQ(result.r.cols(), 2);

  auto reconstructed = multiply_for_test(result.q, result.r);
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }

  auto gram = multiply_for_test(transpose(result.q), result.q);
  for (std::size_t row = 0; row < gram.rows(); ++row)
  {
    for (std::size_t col = 0; col < gram.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(gram(row, col)), static_cast<double>(expected), scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesReducedRealLqFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(0, 2) = Scalar{};
  matrix(1, 2) = Scalar{1};

  auto result = uni20::krylov::real_lq_factorization(matrix);

  ASSERT_EQ(result.l.rows(), 2);
  ASSERT_EQ(result.l.cols(), 2);
  ASSERT_EQ(result.q.rows(), 2);
  ASSERT_EQ(result.q.cols(), 3);

  auto reconstructed = multiply_for_test(result.l, result.q);
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }

  auto gram = multiply_for_test(result.q, transpose(result.q));
  for (std::size_t row = 0; row < gram.rows(); ++row)
  {
    for (std::size_t col = 0; col < gram.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(gram(row, col)), static_cast<double>(expected), scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesReducedRealQlFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 2);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(2, 0) = Scalar{};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(2, 1) = Scalar{1};

  auto result = uni20::krylov::real_ql_factorization(matrix);

  ASSERT_EQ(result.q.rows(), 3);
  ASSERT_EQ(result.q.cols(), 2);
  ASSERT_EQ(result.l.rows(), 2);
  ASSERT_EQ(result.l.cols(), 2);

  auto reconstructed = multiply_for_test(result.q, result.l);
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }

  auto gram = multiply_for_test(transpose(result.q), result.q);
  for (std::size_t row = 0; row < gram.rows(); ++row)
  {
    for (std::size_t col = 0; col < gram.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(gram(row, col)), static_cast<double>(expected), scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesReducedRealRqFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(0, 2) = Scalar{};
  matrix(1, 2) = Scalar{1};

  auto result = uni20::krylov::real_rq_factorization(matrix);

  ASSERT_EQ(result.r.rows(), 2);
  ASSERT_EQ(result.r.cols(), 2);
  ASSERT_EQ(result.q.rows(), 2);
  ASSERT_EQ(result.q.cols(), 3);

  auto reconstructed = multiply_for_test(result.r, result.q);
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }

  auto gram = multiply_for_test(result.q, transpose(result.q));
  for (std::size_t row = 0; row < gram.rows(); ++row)
  {
    for (std::size_t col = 0; col < gram.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(gram(row, col)), static_cast<double>(expected), scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealBidiagonalReductionTallMatrix)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(4, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{2};
  matrix(2, 0) = Scalar{-1};
  matrix(3, 0) = Scalar{4};
  matrix(0, 1) = Scalar{3};
  matrix(1, 1) = Scalar{-2};
  matrix(2, 1) = Scalar{5};
  matrix(3, 1) = Scalar{1};
  matrix(0, 2) = Scalar{-1};
  matrix(1, 2) = Scalar{6};
  matrix(2, 2) = Scalar{2};
  matrix(3, 2) = Scalar{-3};

  auto reduction = uni20::krylov::real_bidiagonal_reduction(matrix);
  auto q = uni20::krylov::real_bidiagonal_left_orthogonal_factor(reduction);
  auto pt = uni20::krylov::real_bidiagonal_right_orthogonal_factor_transpose(reduction);
  auto reconstructed = multiply_for_test(multiply_for_test(q, reduction.bidiagonal), pt);
  auto q_gram = multiply_for_test(transpose(q), q);
  auto p_gram = multiply_for_test(pt, transpose(pt));

  EXPECT_TRUE(reduction.upper);
  ASSERT_EQ(reduction.bidiagonal.rows(), 4);
  ASSERT_EQ(reduction.bidiagonal.cols(), 3);
  ASSERT_EQ(reduction.diagonal.size(), 3);
  ASSERT_EQ(reduction.offdiagonal.size(), 2);
  for (std::size_t row = 0; row < reduction.bidiagonal.rows(); ++row)
  {
    for (std::size_t col = 0; col < reduction.bidiagonal.cols(); ++col)
    {
      bool const allowed = row == col || col == row + 1;
      if (!allowed)
      {
        EXPECT_NEAR(static_cast<double>(reduction.bidiagonal(row, col)), 0.0, scaled_tolerance<Scalar>(50.0));
      }
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(500.0));
    }
  }
  for (std::size_t row = 0; row < q_gram.rows(); ++row)
  {
    for (std::size_t col = 0; col < q_gram.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(q_gram(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(500.0));
    }
  }
  for (std::size_t row = 0; row < p_gram.rows(); ++row)
  {
    for (std::size_t col = 0; col < p_gram.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(p_gram(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(500.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealBidiagonalReductionWideMatrix)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 4);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{2};
  matrix(2, 0) = Scalar{-1};
  matrix(0, 1) = Scalar{3};
  matrix(1, 1) = Scalar{-2};
  matrix(2, 1) = Scalar{5};
  matrix(0, 2) = Scalar{-1};
  matrix(1, 2) = Scalar{6};
  matrix(2, 2) = Scalar{2};
  matrix(0, 3) = Scalar{4};
  matrix(1, 3) = Scalar{1};
  matrix(2, 3) = Scalar{-3};

  auto reduction = uni20::krylov::real_bidiagonal_reduction(matrix);
  auto q = uni20::krylov::real_bidiagonal_left_orthogonal_factor(reduction);
  auto pt = uni20::krylov::real_bidiagonal_right_orthogonal_factor_transpose(reduction);
  auto reconstructed = multiply_for_test(multiply_for_test(q, reduction.bidiagonal), pt);

  EXPECT_FALSE(reduction.upper);
  ASSERT_EQ(reduction.bidiagonal.rows(), 3);
  ASSERT_EQ(reduction.bidiagonal.cols(), 4);
  ASSERT_EQ(reduction.diagonal.size(), 3);
  ASSERT_EQ(reduction.offdiagonal.size(), 2);
  for (std::size_t row = 0; row < reduction.bidiagonal.rows(); ++row)
  {
    for (std::size_t col = 0; col < reduction.bidiagonal.cols(); ++col)
    {
      bool const allowed = row == col || row == col + 1;
      if (!allowed)
      {
        EXPECT_NEAR(static_cast<double>(reduction.bidiagonal(row, col)), 0.0, scaled_tolerance<Scalar>(50.0));
      }
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(500.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealBidiagonalSingularValues)
{
  using Scalar = TypeParam;

  std::vector<Scalar> diagonal{Scalar{1}, Scalar{4}, Scalar{2}};
  std::vector<Scalar> offdiagonal{Scalar{}, Scalar{}};

  auto singular_values = uni20::krylov::real_bidiagonal_singular_values(diagonal, offdiagonal);

  ASSERT_EQ(singular_values.size(), 3);
  EXPECT_NEAR(static_cast<double>(singular_values[0]), 4.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(singular_values[1]), 2.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(singular_values[2]), 1.0, scaled_tolerance<Scalar>(50.0));
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealBidiagonalDivideAndConquerSvd)
{
  using Scalar = TypeParam;

  std::vector<Scalar> diagonal{Scalar{1}, Scalar{4}, Scalar{2}};
  std::vector<Scalar> offdiagonal{Scalar{}, Scalar{}};

  auto values_only = uni20::krylov::real_bidiagonal_svd(diagonal, offdiagonal, false);
  auto result = uni20::krylov::real_bidiagonal_svd(diagonal, offdiagonal, true);

  ASSERT_EQ(values_only.singular_values.size(), 3);
  EXPECT_NEAR(static_cast<double>(values_only.singular_values[0]), 4.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(values_only.singular_values[1]), 2.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(values_only.singular_values[2]), 1.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_EQ(values_only.u.rows(), 0);
  EXPECT_EQ(values_only.vt.cols(), 0);

  ASSERT_EQ(result.singular_values.size(), 3);
  ASSERT_EQ(result.u.rows(), 3);
  ASSERT_EQ(result.u.cols(), 3);
  ASSERT_EQ(result.vt.rows(), 3);
  ASSERT_EQ(result.vt.cols(), 3);

  uni20::krylov::Matrix<Scalar> sigma(3, 3);
  for (std::size_t index = 0; index < result.singular_values.size(); ++index)
  {
    sigma(index, index) = result.singular_values[index];
  }
  auto reconstructed = multiply_for_test(multiply_for_test(result.u, sigma), result.vt);

  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 3; ++col)
    {
      Scalar const expected = row == col ? diagonal[row] : Scalar{};
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(100.0));
    }
  }
}

TEST(KrylovDenseSubspace, RejectsInconsistentBidiagonalSizes)
{
  EXPECT_THROW(uni20::krylov::real_bidiagonal_svd(std::vector<double>{1.0, 2.0}, std::vector<double>{1.0, 2.0}, false),
               std::invalid_argument);
  EXPECT_THROW(uni20::krylov::real_bidiagonal_svd_index_range(std::vector<double>{1.0, 2.0},
                                                              std::vector<double>{1.0, 2.0}, 0, 1, false),
               std::invalid_argument);
  EXPECT_THROW(uni20::krylov::real_bidiagonal_svd_index_range(std::vector<double>{1.0, 2.0}, std::vector<double>{1.0},
                                                              1, 2, false),
               std::invalid_argument);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesSelectedRealBidiagonalSvd)
{
  using Scalar = TypeParam;

  std::vector<Scalar> diagonal{Scalar{1}, Scalar{4}, Scalar{2}};
  std::vector<Scalar> offdiagonal{Scalar{}, Scalar{}};

  auto values_only = uni20::krylov::real_bidiagonal_svd_index_range(diagonal, offdiagonal, 1, 2, false);
  auto result = uni20::krylov::real_bidiagonal_svd_index_range(diagonal, offdiagonal, 1, 2, true);

  ASSERT_EQ(values_only.singular_values.size(), 2);
  EXPECT_NEAR(static_cast<double>(values_only.singular_values[0]), 2.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(values_only.singular_values[1]), 1.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_EQ(values_only.u.rows(), 0);
  EXPECT_EQ(values_only.vt.cols(), 0);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.u.rows(), 3);
  ASSERT_EQ(result.u.cols(), 2);
  ASSERT_EQ(result.vt.rows(), 2);
  ASSERT_EQ(result.vt.cols(), 3);
  EXPECT_NEAR(static_cast<double>(result.singular_values[0]), 2.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(result.singular_values[1]), 1.0, scaled_tolerance<Scalar>(50.0));

  uni20::krylov::Matrix<Scalar> sigma(2, 2);
  for (std::size_t index = 0; index < result.singular_values.size(); ++index)
  {
    sigma(index, index) = result.singular_values[index];
  }
  auto selected_contribution = multiply_for_test(multiply_for_test(result.u, sigma), result.vt);

  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 3; ++col)
    {
      Scalar const expected = row == col && row != 1 ? diagonal[row] : Scalar{};
      EXPECT_NEAR(static_cast<double>(selected_contribution(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(100.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealHessenbergReductionAndOrthogonalFactor)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(4, 4);
  matrix(0, 0) = Scalar{4};
  matrix(1, 0) = Scalar{3};
  matrix(2, 0) = Scalar{-2};
  matrix(3, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{-1};
  matrix(2, 1) = Scalar{5};
  matrix(3, 1) = Scalar{2};
  matrix(0, 2) = Scalar{2};
  matrix(1, 2) = Scalar{6};
  matrix(2, 2) = Scalar{3};
  matrix(3, 2) = Scalar{-4};
  matrix(0, 3) = Scalar{-3};
  matrix(1, 3) = Scalar{2};
  matrix(2, 3) = Scalar{1};
  matrix(3, 3) = Scalar{7};

  auto reduction = uni20::krylov::real_hessenberg_reduction(matrix);

  ASSERT_EQ(reduction.hessenberg.rows(), 4);
  ASSERT_EQ(reduction.hessenberg.cols(), 4);
  ASSERT_EQ(reduction.reflectors.rows(), 4);
  ASSERT_EQ(reduction.reflectors.cols(), 4);
  ASSERT_EQ(reduction.tau.size(), 3);
  EXPECT_EQ(reduction.first, 0);
  EXPECT_EQ(reduction.last_exclusive, 4);
  for (std::size_t col = 0; col < reduction.hessenberg.cols(); ++col)
  {
    for (std::size_t row = col + 2; row < reduction.hessenberg.rows(); ++row)
    {
      EXPECT_NEAR(static_cast<double>(reduction.hessenberg(row, col)), 0.0, tolerance<Scalar>());
    }
  }

  auto q = uni20::krylov::real_hessenberg_orthogonal_factor(reduction);
  auto orthogonality = multiply_for_test(transpose(q), q);
  auto identity = identity_matrix<Scalar>(4);
  for (std::size_t row = 0; row < 4; ++row)
  {
    for (std::size_t col = 0; col < 4; ++col)
    {
      EXPECT_NEAR(static_cast<double>(orthogonality(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }

  auto reconstructed = multiply_for_test(multiply_for_test(q, reduction.hessenberg), transpose(q));
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(100.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, AppliesRealHessenbergOrthogonalFactor)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(4, 4);
  matrix(0, 0) = Scalar{4};
  matrix(1, 0) = Scalar{3};
  matrix(2, 0) = Scalar{-2};
  matrix(3, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{-1};
  matrix(2, 1) = Scalar{5};
  matrix(3, 1) = Scalar{2};
  matrix(0, 2) = Scalar{2};
  matrix(1, 2) = Scalar{6};
  matrix(2, 2) = Scalar{3};
  matrix(3, 2) = Scalar{-4};
  matrix(0, 3) = Scalar{-3};
  matrix(1, 3) = Scalar{2};
  matrix(2, 3) = Scalar{1};
  matrix(3, 3) = Scalar{7};

  auto reduction = uni20::krylov::real_hessenberg_reduction(matrix);
  auto q = uni20::krylov::real_hessenberg_orthogonal_factor(reduction);

  uni20::krylov::Matrix<Scalar> left_target(4, 2);
  left_target(0, 0) = Scalar{1};
  left_target(1, 0) = Scalar{-2};
  left_target(2, 0) = Scalar{3};
  left_target(3, 0) = Scalar{4};
  left_target(0, 1) = Scalar{-1};
  left_target(1, 1) = Scalar{5};
  left_target(2, 1) = Scalar{2};
  left_target(3, 1) = Scalar{-3};

  auto left_applied = uni20::krylov::apply_real_hessenberg_orthogonal_factor(reduction, left_target);
  auto expected_left = multiply_for_test(q, left_target);
  auto recovered_left = uni20::krylov::apply_real_hessenberg_orthogonal_factor(
      reduction, left_applied, uni20::krylov::MatrixSide::Left, uni20::krylov::MatrixTranspose::Transpose);

  uni20::krylov::Matrix<Scalar> right_target(2, 4);
  right_target(0, 0) = Scalar{2};
  right_target(1, 0) = Scalar{-1};
  right_target(0, 1) = Scalar{3};
  right_target(1, 1) = Scalar{4};
  right_target(0, 2) = Scalar{-5};
  right_target(1, 2) = Scalar{1};
  right_target(0, 3) = Scalar{6};
  right_target(1, 3) = Scalar{-2};

  auto right_applied =
      uni20::krylov::apply_real_hessenberg_orthogonal_factor(reduction, right_target, uni20::krylov::MatrixSide::Right);
  auto expected_right = multiply_for_test(right_target, q);
  auto recovered_right = uni20::krylov::apply_real_hessenberg_orthogonal_factor(
      reduction, right_applied, uni20::krylov::MatrixSide::Right, uni20::krylov::MatrixTranspose::Transpose);

  for (std::size_t row = 0; row < left_applied.rows(); ++row)
  {
    for (std::size_t col = 0; col < left_applied.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(left_applied(row, col)), static_cast<double>(expected_left(row, col)),
                  scaled_tolerance<Scalar>(200.0));
      EXPECT_NEAR(static_cast<double>(recovered_left(row, col)), static_cast<double>(left_target(row, col)),
                  scaled_tolerance<Scalar>(200.0));
    }
  }

  for (std::size_t row = 0; row < right_applied.rows(); ++row)
  {
    for (std::size_t col = 0; col < right_applied.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(right_applied(row, col)), static_cast<double>(expected_right(row, col)),
                  scaled_tolerance<Scalar>(200.0));
      EXPECT_NEAR(static_cast<double>(recovered_right(row, col)), static_cast<double>(right_target(row, col)),
                  scaled_tolerance<Scalar>(200.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealSymmetricTridiagonalReductionAndOrthogonalFactor)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(4, 4);
  matrix(0, 0) = Scalar{4};
  matrix(1, 0) = Scalar{1};
  matrix(2, 0) = Scalar{2};
  matrix(3, 0) = Scalar{-3};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{-1};
  matrix(2, 1) = Scalar{5};
  matrix(3, 1) = Scalar{2};
  matrix(0, 2) = Scalar{2};
  matrix(1, 2) = Scalar{5};
  matrix(2, 2) = Scalar{3};
  matrix(3, 2) = Scalar{-4};
  matrix(0, 3) = Scalar{-3};
  matrix(1, 3) = Scalar{2};
  matrix(2, 3) = Scalar{-4};
  matrix(3, 3) = Scalar{7};

  auto reduction = uni20::krylov::real_symmetric_tridiagonal_reduction(matrix, uni20::krylov::MatrixFill::Lower);
  auto q = uni20::krylov::real_symmetric_tridiagonal_orthogonal_factor(reduction);

  for (std::size_t col = 0; col < reduction.tridiagonal.cols(); ++col)
  {
    for (std::size_t row = 0; row < reduction.tridiagonal.rows(); ++row)
    {
      if (row + 1 < col || col + 1 < row)
      {
        EXPECT_NEAR(static_cast<double>(reduction.tridiagonal(row, col)), 0.0, tolerance<Scalar>());
      }
    }
  }

  auto orthogonality = multiply_for_test(transpose(q), q);
  auto identity = identity_matrix<Scalar>(q.rows());
  auto reconstructed = multiply_for_test(multiply_for_test(q, reduction.tridiagonal), transpose(q));
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(orthogonality(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(300.0));
    }
  }

  uni20::krylov::Matrix<Scalar> left_target(4, 2);
  left_target(0, 0) = Scalar{1};
  left_target(1, 0) = Scalar{-2};
  left_target(2, 0) = Scalar{3};
  left_target(3, 0) = Scalar{4};
  left_target(0, 1) = Scalar{-1};
  left_target(1, 1) = Scalar{5};
  left_target(2, 1) = Scalar{2};
  left_target(3, 1) = Scalar{-3};

  auto left_applied = uni20::krylov::apply_real_symmetric_tridiagonal_orthogonal_factor(reduction, left_target);
  auto expected_left = multiply_for_test(q, left_target);
  auto recovered_left = uni20::krylov::apply_real_symmetric_tridiagonal_orthogonal_factor(
      reduction, left_applied, uni20::krylov::MatrixSide::Left, uni20::krylov::MatrixTranspose::Transpose);

  uni20::krylov::Matrix<Scalar> right_target(2, 4);
  right_target(0, 0) = Scalar{2};
  right_target(1, 0) = Scalar{-1};
  right_target(0, 1) = Scalar{3};
  right_target(1, 1) = Scalar{4};
  right_target(0, 2) = Scalar{-5};
  right_target(1, 2) = Scalar{1};
  right_target(0, 3) = Scalar{6};
  right_target(1, 3) = Scalar{-2};

  auto right_applied = uni20::krylov::apply_real_symmetric_tridiagonal_orthogonal_factor(
      reduction, right_target, uni20::krylov::MatrixSide::Right);
  auto expected_right = multiply_for_test(right_target, q);
  auto recovered_right = uni20::krylov::apply_real_symmetric_tridiagonal_orthogonal_factor(
      reduction, right_applied, uni20::krylov::MatrixSide::Right, uni20::krylov::MatrixTranspose::Transpose);

  for (std::size_t row = 0; row < left_applied.rows(); ++row)
  {
    for (std::size_t col = 0; col < left_applied.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(left_applied(row, col)), static_cast<double>(expected_left(row, col)),
                  scaled_tolerance<Scalar>(300.0));
      EXPECT_NEAR(static_cast<double>(recovered_left(row, col)), static_cast<double>(left_target(row, col)),
                  scaled_tolerance<Scalar>(300.0));
    }
  }

  for (std::size_t row = 0; row < right_applied.rows(); ++row)
  {
    for (std::size_t col = 0; col < right_applied.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(right_applied(row, col)), static_cast<double>(expected_right(row, col)),
                  scaled_tolerance<Scalar>(300.0));
      EXPECT_NEAR(static_cast<double>(recovered_right(row, col)), static_cast<double>(right_target(row, col)),
                  scaled_tolerance<Scalar>(300.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, AppliesCompactRealQrFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 2);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(2, 0) = Scalar{};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(2, 1) = Scalar{1};

  auto compact = uni20::krylov::real_compact_qr_factorization(matrix);
  auto explicit_qr = uni20::krylov::real_qr_factorization(matrix);
  auto identity = identity_matrix<Scalar>(3);
  auto left_q = uni20::krylov::apply_real_qr_factor(compact, identity, uni20::krylov::MatrixSide::Left);
  auto right_q = uni20::krylov::apply_real_qr_factor(compact, identity, uni20::krylov::MatrixSide::Right);
  auto recovered_identity = uni20::krylov::apply_real_qr_factor(compact, left_q, uni20::krylov::MatrixSide::Left,
                                                                uni20::krylov::MatrixTranspose::Transpose);

  ASSERT_EQ(compact.rank, 2);
  ASSERT_EQ(compact.tau.size(), 2);
  ASSERT_EQ(left_q.rows(), 3);
  ASSERT_EQ(left_q.cols(), 3);
  for (std::size_t row = 0; row < left_q.rows(); ++row)
  {
    for (std::size_t col = 0; col < left_q.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(left_q(row, col)), static_cast<double>(right_q(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      EXPECT_NEAR(static_cast<double>(recovered_identity(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      if (col < explicit_qr.q.cols())
      {
        EXPECT_NEAR(static_cast<double>(left_q(row, col)), static_cast<double>(explicit_qr.q(row, col)),
                    scaled_tolerance<Scalar>(100.0));
      }
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, AppliesCompactRealLqFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(0, 2) = Scalar{};
  matrix(1, 2) = Scalar{1};

  auto compact = uni20::krylov::real_compact_lq_factorization(matrix);
  auto explicit_lq = uni20::krylov::real_lq_factorization(matrix);
  auto identity = identity_matrix<Scalar>(3);
  auto left_q = uni20::krylov::apply_real_lq_factor(compact, identity, uni20::krylov::MatrixSide::Left);
  auto right_q = uni20::krylov::apply_real_lq_factor(compact, identity, uni20::krylov::MatrixSide::Right);
  auto recovered_identity = uni20::krylov::apply_real_lq_factor(compact, right_q, uni20::krylov::MatrixSide::Right,
                                                                uni20::krylov::MatrixTranspose::Transpose);

  ASSERT_EQ(compact.rank, 2);
  ASSERT_EQ(compact.tau.size(), 2);
  ASSERT_EQ(left_q.rows(), 3);
  ASSERT_EQ(left_q.cols(), 3);
  for (std::size_t row = 0; row < left_q.rows(); ++row)
  {
    for (std::size_t col = 0; col < left_q.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(left_q(row, col)), static_cast<double>(right_q(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      EXPECT_NEAR(static_cast<double>(recovered_identity(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      if (row < explicit_lq.q.rows())
      {
        EXPECT_NEAR(static_cast<double>(left_q(row, col)), static_cast<double>(explicit_lq.q(row, col)),
                    scaled_tolerance<Scalar>(100.0));
      }
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, AppliesCompactRealQlFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 2);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(2, 0) = Scalar{};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(2, 1) = Scalar{1};

  auto compact = uni20::krylov::real_compact_ql_factorization(matrix);
  auto explicit_ql = uni20::krylov::real_ql_factorization(matrix);
  auto identity = identity_matrix<Scalar>(3);
  auto left_q = uni20::krylov::apply_real_ql_factor(compact, identity, uni20::krylov::MatrixSide::Left);
  auto right_q = uni20::krylov::apply_real_ql_factor(compact, identity, uni20::krylov::MatrixSide::Right);
  auto recovered_identity = uni20::krylov::apply_real_ql_factor(compact, left_q, uni20::krylov::MatrixSide::Left,
                                                                uni20::krylov::MatrixTranspose::Transpose);
  std::size_t const first_reduced_col = left_q.cols() - explicit_ql.q.cols();

  ASSERT_EQ(compact.rank, 2);
  ASSERT_EQ(compact.tau.size(), 2);
  ASSERT_EQ(left_q.rows(), 3);
  ASSERT_EQ(left_q.cols(), 3);
  for (std::size_t row = 0; row < left_q.rows(); ++row)
  {
    for (std::size_t col = 0; col < left_q.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(left_q(row, col)), static_cast<double>(right_q(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      EXPECT_NEAR(static_cast<double>(recovered_identity(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      if (col >= first_reduced_col)
      {
        EXPECT_NEAR(static_cast<double>(left_q(row, col)),
                    static_cast<double>(explicit_ql.q(row, col - first_reduced_col)), scaled_tolerance<Scalar>(100.0));
      }
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, AppliesCompactRealRqFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{};
  matrix(0, 2) = Scalar{};
  matrix(1, 2) = Scalar{1};

  auto compact = uni20::krylov::real_compact_rq_factorization(matrix);
  auto explicit_rq = uni20::krylov::real_rq_factorization(matrix);
  auto identity = identity_matrix<Scalar>(3);
  auto left_q = uni20::krylov::apply_real_rq_factor(compact, identity, uni20::krylov::MatrixSide::Left);
  auto right_q = uni20::krylov::apply_real_rq_factor(compact, identity, uni20::krylov::MatrixSide::Right);
  auto recovered_identity = uni20::krylov::apply_real_rq_factor(compact, right_q, uni20::krylov::MatrixSide::Right,
                                                                uni20::krylov::MatrixTranspose::Transpose);
  std::size_t const first_reduced_row = right_q.rows() - explicit_rq.q.rows();

  ASSERT_EQ(compact.rank, 2);
  ASSERT_EQ(compact.tau.size(), 2);
  ASSERT_EQ(right_q.rows(), 3);
  ASSERT_EQ(right_q.cols(), 3);
  for (std::size_t row = 0; row < right_q.rows(); ++row)
  {
    for (std::size_t col = 0; col < right_q.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(left_q(row, col)), static_cast<double>(right_q(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      EXPECT_NEAR(static_cast<double>(recovered_identity(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Scalar>(100.0));
      if (row >= first_reduced_row)
      {
        EXPECT_NEAR(static_cast<double>(right_q(row, col)),
                    static_cast<double>(explicit_rq.q(row - first_reduced_row, col)), scaled_tolerance<Scalar>(100.0));
      }
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, AppliesRealBidiagonalOrthogonalFactors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(4, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{2};
  matrix(2, 0) = Scalar{-1};
  matrix(3, 0) = Scalar{4};
  matrix(0, 1) = Scalar{3};
  matrix(1, 1) = Scalar{-2};
  matrix(2, 1) = Scalar{5};
  matrix(3, 1) = Scalar{1};
  matrix(0, 2) = Scalar{-1};
  matrix(1, 2) = Scalar{6};
  matrix(2, 2) = Scalar{2};
  matrix(3, 2) = Scalar{-3};

  auto reduction = uni20::krylov::real_bidiagonal_reduction(matrix);
  auto q = uni20::krylov::real_bidiagonal_left_orthogonal_factor(reduction);
  auto pt = uni20::krylov::real_bidiagonal_right_orthogonal_factor_transpose(reduction);
  auto identity_q = identity_matrix<Scalar>(4);
  auto identity_p = identity_matrix<Scalar>(3);
  auto applied_q = uni20::krylov::apply_real_bidiagonal_left_factor(reduction, identity_q);
  auto recovered_q = uni20::krylov::apply_real_bidiagonal_left_factor(
      reduction, applied_q, uni20::krylov::MatrixSide::Left, uni20::krylov::MatrixTranspose::Transpose);
  auto applied_pt = uni20::krylov::apply_real_bidiagonal_right_factor(
      reduction, identity_p, uni20::krylov::MatrixSide::Left, uni20::krylov::MatrixTranspose::Transpose);
  auto recovered_p =
      uni20::krylov::apply_real_bidiagonal_right_factor(reduction, applied_pt, uni20::krylov::MatrixSide::Left);

  for (std::size_t row = 0; row < q.rows(); ++row)
  {
    for (std::size_t col = 0; col < q.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(applied_q(row, col)), static_cast<double>(q(row, col)),
                  scaled_tolerance<Scalar>(500.0));
      EXPECT_NEAR(static_cast<double>(recovered_q(row, col)), static_cast<double>(identity_q(row, col)),
                  scaled_tolerance<Scalar>(500.0));
    }
  }
  for (std::size_t row = 0; row < pt.rows(); ++row)
  {
    for (std::size_t col = 0; col < pt.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(applied_pt(row, col)), static_cast<double>(pt(row, col)),
                  scaled_tolerance<Scalar>(500.0));
      EXPECT_NEAR(static_cast<double>(recovered_p(row, col)), static_cast<double>(identity_p(row, col)),
                  scaled_tolerance<Scalar>(500.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesReducedRealPivotedQrFactorization)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 2);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{};
  matrix(2, 0) = Scalar{};
  matrix(0, 1) = Scalar{2};
  matrix(1, 1) = Scalar{3};
  matrix(2, 1) = Scalar{};

  auto result = uni20::krylov::real_pivoted_qr_factorization(matrix);

  ASSERT_EQ(result.q.rows(), 3);
  ASSERT_EQ(result.q.cols(), 2);
  ASSERT_EQ(result.r.rows(), 2);
  ASSERT_EQ(result.r.cols(), 2);
  ASSERT_EQ(result.pivot_columns.size(), 2);
  EXPECT_EQ(result.pivot_columns[0], 1);
  EXPECT_EQ(result.pivot_columns[1], 0);

  auto pivoted_reconstructed = multiply_for_test(result.q, result.r);
  for (std::size_t col = 0; col < matrix.cols(); ++col)
  {
    std::size_t const original_col = result.pivot_columns[col];
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
      EXPECT_NEAR(static_cast<double>(pivoted_reconstructed(row, col)), static_cast<double>(matrix(row, original_col)),
                  scaled_tolerance<Scalar>(50.0));
    }
  }

  auto gram = multiply_for_test(transpose(result.q), result.q);
  for (std::size_t row = 0; row < gram.rows(); ++row)
  {
    for (std::size_t col = 0; col < gram.cols(); ++col)
    {
      Scalar const expected = row == col ? Scalar{1} : Scalar{};
      EXPECT_NEAR(static_cast<double>(gram(row, col)), static_cast<double>(expected), scaled_tolerance<Scalar>(50.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDenseRealSingularValueDecomposition)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(1, 1) = Scalar{3};

  auto result = uni20::krylov::real_singular_value_decomposition(matrix, true);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.left_singular_vectors.rows(), 2);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.singular_values[0]), 3.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.singular_values[1]), 2.0, tolerance<Scalar>());

  uni20::krylov::Matrix<Scalar> sigma(2, 2);
  sigma(0, 0) = result.singular_values[0];
  sigma(1, 1) = result.singular_values[1];
  auto reconstructed = multiply_for_test(multiply_for_test(result.left_singular_vectors, sigma),
                                         result.right_singular_vectors_transpose);
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(20.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesDenseRealDivideAndConquerSingularValueDecomposition)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{2};
  matrix(1, 1) = Scalar{3};

  auto result = uni20::krylov::real_singular_value_decomposition_divide_and_conquer(matrix, true);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.left_singular_vectors.rows(), 2);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.singular_values[0]), 3.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.singular_values[1]), 2.0, tolerance<Scalar>());

  uni20::krylov::Matrix<Scalar> sigma(2, 2);
  sigma(0, 0) = result.singular_values[0];
  sigma(1, 1) = result.singular_values[1];
  auto reconstructed = multiply_for_test(multiply_for_test(result.left_singular_vectors, sigma),
                                         result.right_singular_vectors_transpose);
  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Scalar>(20.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesSelectedDenseRealSingularValueDecomposition)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 1) = Scalar{4};
  matrix(2, 2) = Scalar{2};

  auto values_only = uni20::krylov::real_singular_value_decomposition_index_range(matrix, 1, 2, false);
  auto result = uni20::krylov::real_singular_value_decomposition_index_range(matrix, 1, 2, true);

  ASSERT_EQ(values_only.singular_values.size(), 2);
  EXPECT_NEAR(static_cast<double>(values_only.singular_values[0]), 2.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(values_only.singular_values[1]), 1.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_EQ(values_only.left_singular_vectors.rows(), 0);
  EXPECT_EQ(values_only.right_singular_vectors_transpose.cols(), 0);

  ASSERT_EQ(result.singular_values.size(), 2);
  ASSERT_EQ(result.left_singular_vectors.rows(), 3);
  ASSERT_EQ(result.left_singular_vectors.cols(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.rows(), 2);
  ASSERT_EQ(result.right_singular_vectors_transpose.cols(), 3);
  EXPECT_NEAR(static_cast<double>(result.singular_values[0]), 2.0, scaled_tolerance<Scalar>(50.0));
  EXPECT_NEAR(static_cast<double>(result.singular_values[1]), 1.0, scaled_tolerance<Scalar>(50.0));

  uni20::krylov::Matrix<Scalar> sigma(2, 2);
  sigma(0, 0) = result.singular_values[0];
  sigma(1, 1) = result.singular_values[1];
  auto selected_contribution = multiply_for_test(multiply_for_test(result.left_singular_vectors, sigma),
                                                 result.right_singular_vectors_transpose);

  for (std::size_t row = 0; row < matrix.rows(); ++row)
  {
    for (std::size_t col = 0; col < matrix.cols(); ++col)
    {
      Scalar const expected = row == col && row != 1 ? matrix(row, col) : Scalar{};
      EXPECT_NEAR(static_cast<double>(selected_contribution(row, col)), static_cast<double>(expected),
                  scaled_tolerance<Scalar>(100.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealSchurDecompositionWithComplexBlock)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{1};
  matrix(1, 0) = Scalar{-2};
  matrix(0, 1) = Scalar{2};
  matrix(1, 1) = Scalar{1};

  auto result = uni20::krylov::real_schur(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.blocks.size(), 1);
  EXPECT_EQ(result.blocks[0].begin, 0);
  EXPECT_EQ(result.blocks[0].size, 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].real()), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(std::abs(result.eigenvalues[0].imag())), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(std::abs(result.eigenvalues[1] - std::conj(result.eigenvalues[0]))), 0.0,
              tolerance<Scalar>());
  ASSERT_EQ(result.schur_vectors.rows(), 2);
  ASSERT_EQ(result.schur_vectors.cols(), 2);
  expect_reconstructs(matrix, result.schur_form, result.schur_vectors);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealHessenbergSchurDecomposition)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> hessenberg(3, 3);
  hessenberg(0, 0) = Scalar{1};
  hessenberg(1, 0) = Scalar{-2};
  hessenberg(0, 1) = Scalar{2};
  hessenberg(1, 1) = Scalar{1};
  hessenberg(0, 2) = Scalar{1} / Scalar{2};
  hessenberg(1, 2) = Scalar{3};
  hessenberg(2, 2) = Scalar{4};

  auto result = uni20::krylov::real_hessenberg_schur(hessenberg, true);

  ASSERT_EQ(result.eigenvalues.size(), 3);
  ASSERT_EQ(result.blocks.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].real()), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(std::abs(result.eigenvalues[0].imag())), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(std::abs(result.eigenvalues[1] - std::conj(result.eigenvalues[0]))), 0.0,
              tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[2].real()), 4.0, tolerance<Scalar>());
  ASSERT_EQ(result.schur_vectors.rows(), 3);
  ASSERT_EQ(result.schur_vectors.cols(), 3);
  expect_reconstructs(hessenberg, result.schur_form, result.schur_vectors);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesLayoutRightSchurForLogicalMatrix)
{
  using Scalar = TypeParam;

  uni20::krylov::RightMatrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{1};
  matrix(0, 1) = Scalar{10};
  matrix(1, 0) = Scalar{0};
  matrix(1, 1) = Scalar{2};

  auto result = uni20::krylov::real_schur_layout_right(matrix, true);
  uni20::krylov::Matrix<Scalar> left_matrix = uni20::krylov::copy_right_to_left(matrix);
  uni20::krylov::Matrix<Scalar> left_schur_form = uni20::krylov::copy_right_to_left(result.schur_form);
  uni20::krylov::Matrix<Scalar> left_schur_vectors = uni20::krylov::copy_right_to_left(result.schur_vectors);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].imag()), 0.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1].imag()), 0.0, tolerance<Scalar>());
  expect_reconstructs(left_matrix, left_schur_form, left_schur_vectors);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ReordersRealSchurOneByOneBlocks)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 1) = Scalar{3};
  matrix(2, 2) = Scalar{2};

  auto schur = uni20::krylov::real_schur(matrix, true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (std::abs(schur.blocks[index].first_eigenvalue - uni20::complex<Scalar>{Scalar{3}, Scalar{}}) <
        scaled_tolerance<Scalar>(10.0))
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto reordered = uni20::krylov::reorder_real_schur(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_EQ(reordered.blocks.front().size, 1);
  EXPECT_NEAR(static_cast<double>(reordered.eigenvalues.front().real()), 3.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(reordered.eigenvalues.front().imag()), 0.0, tolerance<Scalar>());
  expect_reconstructs(matrix, reordered.schur_form, reordered.schur_vectors);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SelectsRealSchurSubspaceWithConditionEstimates)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 3);
  matrix(0, 0) = Scalar{1};
  matrix(1, 1) = Scalar{3};
  matrix(2, 2) = Scalar{2};

  auto schur = uni20::krylov::real_schur(matrix, true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (std::abs(schur.blocks[index].first_eigenvalue - uni20::complex<Scalar>{Scalar{3}, Scalar{}}) <
        scaled_tolerance<Scalar>(10.0))
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto result = uni20::krylov::real_schur_selected_subspace(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_EQ(result.selected_dimension, 1);
  ASSERT_EQ(result.decomposition.blocks.front().size, 1);
  EXPECT_NEAR(static_cast<double>(result.decomposition.eigenvalues.front().real()), 3.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.decomposition.eigenvalues.front().imag()), 0.0, tolerance<Scalar>());
  EXPECT_GT(static_cast<double>(result.reciprocal_eigenvalue_cluster_condition), 0.0);
  EXPECT_GT(static_cast<double>(result.reciprocal_invariant_subspace_condition), 0.0);
  expect_reconstructs(matrix, result.decomposition.schur_form, result.decomposition.schur_vectors);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealSchurRightEigenvectors)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{3};

  auto schur = uni20::krylov::real_schur(matrix, true);
  auto const schur_form = schur.schur_form;
  auto const eigenvalues = schur.eigenvalues;
  auto result = uni20::krylov::real_schur_right_eigenvectors(std::move(schur));

  ASSERT_EQ(result.computed_vectors, 2);
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
  for (std::size_t column = 0; column < result.right_eigenvectors.cols(); ++column)
  {
    Scalar vector_max{};
    for (std::size_t row = 0; row < result.right_eigenvectors.rows(); ++row)
    {
      vector_max = std::max(vector_max, static_cast<Scalar>(std::abs(result.right_eigenvectors(row, column))));
    }
    EXPECT_GT(static_cast<double>(vector_max), 0.0);
    EXPECT_LT(static_cast<double>(schur_right_eigenvector_residual_max(schur_form, eigenvalues[column],
                                                                       result.right_eigenvectors, column)),
              scaled_tolerance<Scalar>(100.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesRealSchurEigenpairConditioning)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(2, 2);
  matrix(0, 0) = Scalar{1};
  matrix(0, 1) = Scalar{1};
  matrix(1, 1) = Scalar{3};

  auto schur = uni20::krylov::real_schur(matrix, true);
  auto result = uni20::krylov::real_schur_condition_estimates(std::move(schur));

  ASSERT_EQ(result.computed_estimates, 2);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 2);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 2);
  for (std::size_t index = 0; index < 2; ++index)
  {
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvalue_condition_numbers[index]), 0.0);
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvector_condition_numbers[index]), 0.0);
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ReordersRealSchurComplexBlockAsUnit)
{
  using Scalar = TypeParam;

  uni20::krylov::Matrix<Scalar> matrix(3, 3);
  matrix(0, 0) = Scalar{5};
  matrix(1, 1) = Scalar{1};
  matrix(2, 1) = Scalar{-2};
  matrix(1, 2) = Scalar{2};
  matrix(2, 2) = Scalar{1};

  auto schur = uni20::krylov::real_schur(matrix, true);
  std::size_t complex_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (schur.blocks[index].size == 2)
    {
      complex_block = index;
    }
  }
  ASSERT_LT(complex_block, schur.blocks.size());

  auto reordered = uni20::krylov::reorder_real_schur(std::move(schur), std::vector<std::size_t>{complex_block});

  ASSERT_GE(reordered.blocks.size(), 2);
  EXPECT_EQ(reordered.blocks[0].begin, 0);
  EXPECT_EQ(reordered.blocks[0].size, 2);
  EXPECT_EQ(reordered.blocks[1].begin, 2);
  EXPECT_NEAR(static_cast<double>(reordered.eigenvalues[0].real()), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(std::abs(reordered.eigenvalues[0].imag())), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(std::abs(reordered.eigenvalues[1] - std::conj(reordered.eigenvalues[0]))), 0.0,
              tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(reordered.eigenvalues[2].real()), 5.0, tolerance<Scalar>());
  expect_reconstructs(matrix, reordered.schur_form, reordered.schur_vectors);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesSymmetricTridiagonalEigenvalues)
{
  using Scalar = TypeParam;

  auto eigenvalues = uni20::krylov::symmetric_tridiagonal_eigenvalues(std::vector<Scalar>{Scalar{2}, Scalar{2}},
                                                                      std::vector<Scalar>{Scalar{1}});

  ASSERT_EQ(eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(eigenvalues[0]), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(eigenvalues[1]), 3.0, tolerance<Scalar>());
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesSymmetricTridiagonalEigensystemWithoutVectors)
{
  using Scalar = TypeParam;

  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem(std::vector<Scalar>{Scalar{2}, Scalar{2}},
                                                                 std::vector<Scalar>{Scalar{1}}, false);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());
  EXPECT_EQ(result.eigenvectors.rows(), 0);
  EXPECT_EQ(result.eigenvectors.cols(), 0);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesSymmetricTridiagonalEigenvectors)
{
  using Scalar = TypeParam;

  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem(std::vector<Scalar>{Scalar{2}, Scalar{2}},
                                                                 std::vector<Scalar>{Scalar{1}}, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar const x0 = result.eigenvectors(0, col);
    Scalar const x1 = result.eigenvectors(1, col);
    Scalar const residual0 = Scalar{2} * x0 + x1 - lambda * x0;
    Scalar const residual1 = x0 + Scalar{2} * x1 - lambda * x1;
    double const residual_norm = std::hypot(static_cast<double>(residual0), static_cast<double>(residual1));
    EXPECT_LT(residual_norm, tolerance<Scalar>());

    double const vector_norm = std::hypot(static_cast<double>(x0), static_cast<double>(x1));
    EXPECT_NEAR(vector_norm, 1.0, tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesSymmetricTridiagonalDivideAndConquerEigensystem)
{
  using Scalar = TypeParam;

  auto eigenvalues_only = uni20::krylov::symmetric_tridiagonal_eigensystem_divide_and_conquer(
      std::vector<Scalar>{Scalar{2}, Scalar{2}}, std::vector<Scalar>{Scalar{1}}, false);
  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem_divide_and_conquer(
      std::vector<Scalar>{Scalar{2}, Scalar{2}}, std::vector<Scalar>{Scalar{1}}, true);

  ASSERT_EQ(eigenvalues_only.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(eigenvalues_only.eigenvalues[0]), 1.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(eigenvalues_only.eigenvalues[1]), 3.0, tolerance<Scalar>());
  EXPECT_EQ(eigenvalues_only.eigenvectors.rows(), 0);
  EXPECT_EQ(eigenvalues_only.eigenvectors.cols(), 0);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 2);
  ASSERT_EQ(result.eigenvectors.cols(), 2);

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar const x0 = result.eigenvectors(0, col);
    Scalar const x1 = result.eigenvectors(1, col);
    Scalar const residual0 = Scalar{2} * x0 + x1 - lambda * x0;
    Scalar const residual1 = x0 + Scalar{2} * x1 - lambda * x1;
    double const residual_norm = std::hypot(static_cast<double>(residual0), static_cast<double>(residual1));
    EXPECT_LT(residual_norm, tolerance<Scalar>());

    double const vector_norm = std::hypot(static_cast<double>(x0), static_cast<double>(x1));
    EXPECT_NEAR(vector_norm, 1.0, tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesSelectedSymmetricTridiagonalEigenvectors)
{
  using Scalar = TypeParam;

  auto eigenvalues_only = uni20::krylov::symmetric_tridiagonal_eigensystem_index_range(
      std::vector<Scalar>{Scalar{1}, Scalar{2}, Scalar{3}}, std::vector<Scalar>{Scalar{}, Scalar{}}, 1, 2, false);
  auto result = uni20::krylov::symmetric_tridiagonal_eigensystem_index_range(
      std::vector<Scalar>{Scalar{1}, Scalar{2}, Scalar{3}}, std::vector<Scalar>{Scalar{}, Scalar{}}, 1, 2, true);

  ASSERT_EQ(eigenvalues_only.eigenvalues.size(), 2);
  EXPECT_NEAR(static_cast<double>(eigenvalues_only.eigenvalues[0]), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(eigenvalues_only.eigenvalues[1]), 3.0, tolerance<Scalar>());
  EXPECT_EQ(eigenvalues_only.eigenvectors.rows(), 0);
  EXPECT_EQ(eigenvalues_only.eigenvectors.cols(), 0);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.rows(), 3);
  ASSERT_EQ(result.eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0]), 2.0, tolerance<Scalar>());
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[1]), 3.0, tolerance<Scalar>());

  for (std::size_t col = 0; col < result.eigenvectors.cols(); ++col)
  {
    Scalar const lambda = result.eigenvalues[col];
    Scalar residual_norm{};
    Scalar vector_norm{};
    for (std::size_t row = 0; row < result.eigenvectors.rows(); ++row)
    {
      Scalar const diagonal = Scalar{1} + static_cast<Scalar>(row);
      Scalar const entry = result.eigenvectors(row, col);
      Scalar const residual = diagonal * entry - lambda * entry;
      residual_norm += residual * residual;
      vector_norm += entry * entry;
    }
    EXPECT_LT(static_cast<double>(std::sqrt(residual_norm)), tolerance<Scalar>());
    EXPECT_NEAR(static_cast<double>(std::sqrt(vector_norm)), 1.0, tolerance<Scalar>());
  }
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, SolvesComplexNonsymmetricEigenvalues)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  uni20::krylov::Matrix<Complex> matrix(2, 2);
  matrix(0, 0) = Complex{Real{1}, Real{2}};
  matrix(1, 0) = Complex{};
  matrix(0, 1) = Complex{Real{3}, Real{-1}};
  matrix(1, 1) = Complex{Real{-2}, Real{0.5}};

  auto result = uni20::krylov::complex_nonsymmetric_eigensystem<Real>(std::move(matrix), true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  Complex const first{Real{1}, Real{2}};
  Complex const second{Real{-2}, Real{0.5}};
  EXPECT_LT(
      static_cast<double>(std::min(std::abs(result.eigenvalues[0] - first), std::abs(result.eigenvalues[1] - first))),
      tolerance<Real>());
  EXPECT_LT(
      static_cast<double>(std::min(std::abs(result.eigenvalues[0] - second), std::abs(result.eigenvalues[1] - second))),
      tolerance<Real>());
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, ComputesComplexSchurDecomposition)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  uni20::krylov::Matrix<Complex> matrix(2, 2);
  matrix(0, 0) = Complex{Real{1}, Real{2}};
  matrix(1, 0) = Complex{};
  matrix(0, 1) = Complex{Real{3}, Real{-1}};
  matrix(1, 1) = Complex{Real{-2}, Real{0.5}};

  auto result = uni20::krylov::complex_schur<Real>(matrix, true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.schur_vectors.rows(), 2);
  ASSERT_EQ(result.schur_vectors.cols(), 2);
  EXPECT_LT(static_cast<double>(std::min(std::abs(result.eigenvalues[0] - Complex{Real{1}, Real{2}}),
                                         std::abs(result.eigenvalues[1] - Complex{Real{1}, Real{2}}))),
            tolerance<Real>());
  EXPECT_LT(static_cast<double>(std::min(std::abs(result.eigenvalues[0] - Complex{Real{-2}, Real{0.5}}),
                                         std::abs(result.eigenvalues[1] - Complex{Real{-2}, Real{0.5}}))),
            tolerance<Real>());
  expect_complex_reconstructs(matrix, result.schur_form, result.schur_vectors);
}

TYPED_TEST(KrylovDenseSubspaceComplexTypedTest, ReordersComplexSchurEntries)
{
  using Complex = TypeParam;
  using Real = typename Complex::value_type;

  uni20::krylov::Matrix<Complex> matrix(3, 3);
  matrix(0, 0) = Complex{Real{1}, Real{2}};
  matrix(1, 0) = Complex{};
  matrix(2, 0) = Complex{};
  matrix(0, 1) = Complex{Real{0.5}, Real{-0.25}};
  matrix(1, 1) = Complex{Real{-2}, Real{0.5}};
  matrix(2, 1) = Complex{};
  matrix(0, 2) = Complex{Real{-1}, Real{0.75}};
  matrix(1, 2) = Complex{Real{0.25}, Real{1}};
  matrix(2, 2) = Complex{Real{3}, Real{-1}};

  auto schur = uni20::krylov::complex_schur<Real>(matrix, true);
  std::size_t target_index = schur.eigenvalues.size();
  for (std::size_t index = 0; index < schur.eigenvalues.size(); ++index)
  {
    if (std::abs(schur.eigenvalues[index] - Complex{Real{3}, Real{-1}}) < scaled_tolerance<Real>(10.0))
    {
      target_index = index;
    }
  }
  ASSERT_LT(target_index, schur.eigenvalues.size());

  auto reordered = uni20::krylov::reorder_complex_schur<Real>(std::move(schur), std::vector<std::size_t>{target_index});

  ASSERT_EQ(reordered.eigenvalues.size(), 3);
  EXPECT_LT(abs_as_double(reordered.eigenvalues.front() - Complex{Real{3}, Real{-1}}), scaled_tolerance<Real>(10.0));
  expect_complex_reconstructs(matrix, reordered.schur_form, reordered.schur_vectors);
}

TEST(KrylovDenseSubspace, RejectsInconsistentTridiagonalSizes)
{
  EXPECT_THROW(
      uni20::krylov::symmetric_tridiagonal_eigenvalues(std::vector<double>{1.0, 2.0}, std::vector<double>{1.0, 2.0}),
      std::invalid_argument);
  EXPECT_THROW(uni20::krylov::symmetric_tridiagonal_eigensystem(std::vector<double>{1.0, 2.0},
                                                                std::vector<double>{1.0, 2.0}, false),
               std::invalid_argument);
  EXPECT_THROW(uni20::krylov::symmetric_tridiagonal_eigensystem_divide_and_conquer(
                   std::vector<double>{1.0, 2.0}, std::vector<double>{1.0, 2.0}, false),
               std::invalid_argument);
  EXPECT_THROW(uni20::krylov::symmetric_tridiagonal_eigensystem_index_range(std::vector<double>{1.0, 2.0},
                                                                            std::vector<double>{1.0, 2.0}, 0, 1, false),
               std::invalid_argument);
  EXPECT_THROW(uni20::krylov::symmetric_tridiagonal_eigensystem_index_range(std::vector<double>{1.0, 2.0},
                                                                            std::vector<double>{1.0}, 1, 2, false),
               std::invalid_argument);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesRealNonsymmetricDiagonalEigenvalues)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(3, 3);
  matrix(0, 0) = Real{3};
  matrix(1, 1) = Real{-1};
  matrix(2, 2) = Real{2};

  auto result = uni20::krylov::real_nonsymmetric_eigensystem(std::move(matrix), false);

  std::vector<uni20::complex<Real>> eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  ASSERT_EQ(eigenvalues.size(), 3);
  EXPECT_NEAR(static_cast<double>(eigenvalues[0].real()), -1.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(eigenvalues[1].real()), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(eigenvalues[2].real()), 3.0, tolerance<Real>());
  EXPECT_EQ(result.right_eigenvectors.rows(), 0);
  EXPECT_EQ(result.right_eigenvectors.cols(), 0);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, BalancesRealNonsymmetricMatrixAndBacktransformsRightVectors)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(2, 2);
  matrix(0, 0) = Real{};
  matrix(0, 1) = Real{1000000};
  matrix(1, 0) = Real{1} / Real{1000};
  matrix(1, 1) = Real{};

  auto balance = uni20::krylov::real_nonsymmetric_balance(matrix, uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  ASSERT_EQ(balance.balanced_matrix.rows(), 2);
  ASSERT_EQ(balance.balanced_matrix.cols(), 2);
  ASSERT_EQ(balance.scale.size(), 2);
  EXPECT_LT(balance.balanced_first, balance.balanced_last_exclusive);
  EXPECT_LE(balance.balanced_last_exclusive, 2);
  EXPECT_GT(static_cast<double>(std::abs(balance.balanced_matrix(0, 1))), 0.0);
  EXPECT_GT(static_cast<double>(std::abs(balance.balanced_matrix(1, 0))), 0.0);

  uni20::krylov::Matrix<Real> vectors(2, 2);
  vectors(0, 0) = Real{1};
  vectors(1, 1) = Real{1};

  uni20::krylov::RealNonsymmetricBalance<Real> manual_balance;
  manual_balance.scale = std::vector<Real>{Real{2}, Real{4}};
  manual_balance.balanced_first = 0;
  manual_balance.balanced_last_exclusive = 2;

  auto transformed = uni20::krylov::real_nonsymmetric_balance_backtransform_right_vectors(
      vectors, manual_balance, uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  EXPECT_NEAR(static_cast<double>(transformed(0, 0)), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(transformed(1, 0)), 0.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(transformed(0, 1)), 0.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(transformed(1, 1)), 4.0, tolerance<Real>());
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, BalancesRealGeneralizedNonsymmetricPencilAndBacktransformsRightVectors)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(2, 2);
  matrix(0, 0) = Real{};
  matrix(0, 1) = Real{1000000};
  matrix(1, 0) = Real{1} / Real{1000};
  matrix(1, 1) = Real{};

  uni20::krylov::Matrix<Real> metric(2, 2);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{2};

  auto balance = uni20::krylov::real_generalized_nonsymmetric_balance(matrix, metric,
                                                                      uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  ASSERT_EQ(balance.balanced_matrix.rows(), 2);
  ASSERT_EQ(balance.balanced_matrix.cols(), 2);
  ASSERT_EQ(balance.balanced_metric.rows(), 2);
  ASSERT_EQ(balance.balanced_metric.cols(), 2);
  ASSERT_EQ(balance.left_scale.size(), 2);
  ASSERT_EQ(balance.right_scale.size(), 2);
  EXPECT_LT(balance.balanced_first, balance.balanced_last_exclusive);
  EXPECT_LE(balance.balanced_last_exclusive, 2);
  EXPECT_GT(static_cast<double>(std::abs(balance.balanced_matrix(0, 1))), 0.0);
  EXPECT_GT(static_cast<double>(std::abs(balance.balanced_matrix(1, 0))), 0.0);

  uni20::krylov::Matrix<Real> vectors(2, 2);
  vectors(0, 0) = Real{1};
  vectors(1, 1) = Real{1};

  uni20::krylov::RealGeneralizedNonsymmetricBalance<Real> manual_balance;
  manual_balance.left_scale = std::vector<Real>{Real{1}, Real{1}};
  manual_balance.right_scale = std::vector<Real>{Real{2}, Real{4}};
  manual_balance.balanced_first = 0;
  manual_balance.balanced_last_exclusive = 2;

  auto transformed = uni20::krylov::real_generalized_nonsymmetric_balance_backtransform_right_vectors(
      vectors, manual_balance, uni20::krylov::RealNonsymmetricBalanceJob::Scale);

  EXPECT_NEAR(static_cast<double>(transformed(0, 0)), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(transformed(1, 0)), 0.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(transformed(0, 1)), 0.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(transformed(1, 1)), 4.0, tolerance<Real>());
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealGeneralizedHessenbergReduction)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(3, 3);
  matrix(0, 0) = Real{4};
  matrix(1, 0) = Real{2};
  matrix(2, 0) = Real{-1};
  matrix(0, 1) = Real{3};
  matrix(1, 1) = Real{5};
  matrix(2, 1) = Real{7};
  matrix(0, 2) = Real{-2};
  matrix(1, 2) = Real{1};
  matrix(2, 2) = Real{6};

  uni20::krylov::Matrix<Real> metric(3, 3);
  metric(0, 0) = Real{2};
  metric(0, 1) = Real{-1};
  metric(1, 1) = Real{3};
  metric(0, 2) = Real{1};
  metric(1, 2) = Real{2};
  metric(2, 2) = Real{4};

  auto result = uni20::krylov::real_generalized_hessenberg_reduction(matrix, metric, true);

  ASSERT_EQ(result.matrix_hessenberg_form.rows(), 3);
  ASSERT_EQ(result.matrix_hessenberg_form.cols(), 3);
  ASSERT_EQ(result.metric_triangular_form.rows(), 3);
  ASSERT_EQ(result.metric_triangular_form.cols(), 3);
  ASSERT_EQ(result.left_orthogonal_vectors.rows(), 3);
  ASSERT_EQ(result.left_orthogonal_vectors.cols(), 3);
  ASSERT_EQ(result.right_orthogonal_vectors.rows(), 3);
  ASSERT_EQ(result.right_orthogonal_vectors.cols(), 3);
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.last_exclusive, 3);
  EXPECT_NEAR(static_cast<double>(result.matrix_hessenberg_form(2, 0)), 0.0, tolerance<Real>());
  for (std::size_t col = 0; col < 3; ++col)
  {
    for (std::size_t row = col + 1; row < 3; ++row)
    {
      EXPECT_NEAR(static_cast<double>(result.metric_triangular_form(row, col)), 0.0, tolerance<Real>());
    }
  }

  auto q_orthogonality = multiply_for_test(transpose(result.left_orthogonal_vectors), result.left_orthogonal_vectors);
  auto z_orthogonality = multiply_for_test(transpose(result.right_orthogonal_vectors), result.right_orthogonal_vectors);
  auto identity = identity_matrix<Real>(3);
  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 3; ++col)
    {
      EXPECT_NEAR(static_cast<double>(q_orthogonality(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Real>(50.0));
      EXPECT_NEAR(static_cast<double>(z_orthogonality(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Real>(50.0));
    }
  }

  auto reconstructed_matrix =
      multiply_for_test(multiply_for_test(result.left_orthogonal_vectors, result.matrix_hessenberg_form),
                        transpose(result.right_orthogonal_vectors));
  auto reconstructed_metric =
      multiply_for_test(multiply_for_test(result.left_orthogonal_vectors, result.metric_triangular_form),
                        transpose(result.right_orthogonal_vectors));
  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 3; ++col)
    {
      EXPECT_NEAR(static_cast<double>(reconstructed_matrix(row, col)), static_cast<double>(matrix(row, col)),
                  scaled_tolerance<Real>(100.0));
      EXPECT_NEAR(static_cast<double>(reconstructed_metric(row, col)), static_cast<double>(metric(row, col)),
                  scaled_tolerance<Real>(100.0));
    }
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealGeneralizedHessenbergSchurDecomposition)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> hessenberg(3, 3);
  hessenberg(0, 0) = Real{3};
  hessenberg(1, 0) = Real{2};
  hessenberg(0, 1) = Real{1};
  hessenberg(1, 1) = Real{4};
  hessenberg(2, 1) = Real{-1};
  hessenberg(0, 2) = Real{-2};
  hessenberg(1, 2) = Real{5};
  hessenberg(2, 2) = Real{6};

  uni20::krylov::Matrix<Real> metric(3, 3);
  metric(0, 0) = Real{2};
  metric(0, 1) = Real{-1};
  metric(1, 1) = Real{3};
  metric(0, 2) = Real{1};
  metric(1, 2) = Real{2};
  metric(2, 2) = Real{4};

  auto result = uni20::krylov::real_generalized_hessenberg_schur(hessenberg, metric, true);

  ASSERT_EQ(result.matrix_schur_form.rows(), 3);
  ASSERT_EQ(result.matrix_schur_form.cols(), 3);
  ASSERT_EQ(result.metric_schur_form.rows(), 3);
  ASSERT_EQ(result.metric_schur_form.cols(), 3);
  ASSERT_EQ(result.left_schur_vectors.rows(), 3);
  ASSERT_EQ(result.left_schur_vectors.cols(), 3);
  ASSERT_EQ(result.right_schur_vectors.rows(), 3);
  ASSERT_EQ(result.right_schur_vectors.cols(), 3);
  ASSERT_EQ(result.alpha.size(), 3);
  ASSERT_EQ(result.beta.size(), 3);
  ASSERT_EQ(result.eigenvalues.size(), 3);
  EXPECT_EQ(result.selected_dimension, 0);
  EXPECT_NEAR(static_cast<double>(result.matrix_schur_form(2, 0)), 0.0, scaled_tolerance<Real>(100.0));
  for (std::size_t col = 0; col < 3; ++col)
  {
    for (std::size_t row = col + 1; row < 3; ++row)
    {
      EXPECT_NEAR(static_cast<double>(result.metric_schur_form(row, col)), 0.0, scaled_tolerance<Real>(100.0));
    }
  }

  for (std::size_t index = 0; index < result.alpha.size(); ++index)
  {
    EXPECT_GT(static_cast<double>(std::abs(result.beta[index])), 0.0);
    EXPECT_LT(static_cast<double>(std::abs(result.alpha[index] / result.beta[index] - result.eigenvalues[index])),
              scaled_tolerance<Real>(100.0));
  }

  auto q_orthogonality = multiply_for_test(transpose(result.left_schur_vectors), result.left_schur_vectors);
  auto z_orthogonality = multiply_for_test(transpose(result.right_schur_vectors), result.right_schur_vectors);
  auto identity = identity_matrix<Real>(3);
  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 3; ++col)
    {
      EXPECT_NEAR(static_cast<double>(q_orthogonality(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Real>(100.0));
      EXPECT_NEAR(static_cast<double>(z_orthogonality(row, col)), static_cast<double>(identity(row, col)),
                  scaled_tolerance<Real>(100.0));
    }
  }

  expect_generalized_schur_reconstructs(hessenberg, metric, result);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealGeneralizedSchurDecomposition)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(2, 2);
  matrix(0, 0) = Real{2};
  matrix(1, 1) = Real{6};

  uni20::krylov::Matrix<Real> metric(2, 2);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{2};

  auto result = uni20::krylov::real_generalized_schur(matrix, metric, true);

  std::vector<uni20::complex<Real>> eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  ASSERT_EQ(eigenvalues.size(), 2);
  ASSERT_EQ(result.alpha.size(), 2);
  ASSERT_EQ(result.beta.size(), 2);
  ASSERT_EQ(result.blocks.size(), 2);
  ASSERT_EQ(result.left_schur_vectors.rows(), 2);
  ASSERT_EQ(result.left_schur_vectors.cols(), 2);
  ASSERT_EQ(result.right_schur_vectors.rows(), 2);
  ASSERT_EQ(result.right_schur_vectors.cols(), 2);
  EXPECT_EQ(result.selected_dimension, 0);
  EXPECT_NEAR(static_cast<double>(eigenvalues[0].real()), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(eigenvalues[1].real()), 3.0, tolerance<Real>());
  for (std::size_t index = 0; index < result.alpha.size(); ++index)
  {
    EXPECT_GT(static_cast<double>(std::abs(result.beta[index])), 0.0);
    EXPECT_LT(static_cast<double>(std::abs(result.alpha[index] / result.beta[index] - result.eigenvalues[index])),
              scaled_tolerance<Real>(10.0));
  }
  expect_generalized_schur_reconstructs(matrix, metric, result);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ReordersRealGeneralizedSchurBlocks)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(3, 3);
  matrix(0, 0) = Real{1};
  matrix(1, 1) = Real{3};
  matrix(2, 2) = Real{2};

  uni20::krylov::Matrix<Real> metric(3, 3);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{1};
  metric(2, 2) = Real{1};

  auto schur = uni20::krylov::real_generalized_schur(matrix, metric, true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (std::abs(schur.blocks[index].first_eigenvalue - uni20::complex<Real>{Real{3}, Real{}}) <
        scaled_tolerance<Real>(10.0))
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto reordered =
      uni20::krylov::reorder_real_generalized_schur(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_FALSE(reordered.eigenvalues.empty());
  ASSERT_FALSE(reordered.blocks.empty());
  EXPECT_EQ(reordered.blocks.front().size, 1);
  EXPECT_NEAR(static_cast<double>(reordered.eigenvalues.front().real()), 3.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(reordered.eigenvalues.front().imag()), 0.0, tolerance<Real>());
  EXPECT_NEAR(
      static_cast<double>(std::abs(reordered.alpha.front() / reordered.beta.front() - reordered.eigenvalues.front())),
      0.0, scaled_tolerance<Real>(10.0));
  expect_generalized_schur_reconstructs(matrix, metric, reordered);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SelectsRealGeneralizedSchurSubspaceWithConditionEstimates)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(3, 3);
  matrix(0, 0) = Real{1};
  matrix(1, 1) = Real{3};
  matrix(2, 2) = Real{2};

  uni20::krylov::Matrix<Real> metric(3, 3);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{1};
  metric(2, 2) = Real{1};

  auto schur = uni20::krylov::real_generalized_schur(matrix, metric, true);
  std::size_t target_block = schur.blocks.size();
  for (std::size_t index = 0; index < schur.blocks.size(); ++index)
  {
    if (std::abs(schur.blocks[index].first_eigenvalue - uni20::complex<Real>{Real{3}, Real{}}) <
        scaled_tolerance<Real>(10.0))
    {
      target_block = index;
    }
  }
  ASSERT_LT(target_block, schur.blocks.size());

  auto result =
      uni20::krylov::real_generalized_schur_selected_subspace(std::move(schur), std::vector<std::size_t>{target_block});

  ASSERT_FALSE(result.decomposition.eigenvalues.empty());
  EXPECT_EQ(result.selected_dimension, 1);
  EXPECT_NEAR(static_cast<double>(result.decomposition.eigenvalues.front().real()), 3.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(result.decomposition.eigenvalues.front().imag()), 0.0, tolerance<Real>());
  EXPECT_GT(static_cast<double>(result.left_projection_lower_bound), 0.0);
  EXPECT_GT(static_cast<double>(result.right_projection_lower_bound), 0.0);
  EXPECT_GT(static_cast<double>(result.upper_deflating_subspace_separation), 0.0);
  EXPECT_GT(static_cast<double>(result.lower_deflating_subspace_separation), 0.0);
  expect_generalized_schur_reconstructs(matrix, metric, result.decomposition);
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, ComputesRealGeneralizedSchurRightEigenvectors)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(3, 3);
  matrix(0, 0) = Real{1};
  matrix(1, 1) = Real{3};
  matrix(2, 2) = Real{2};

  uni20::krylov::Matrix<Real> metric(3, 3);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{1};
  metric(2, 2) = Real{1};

  auto schur = uni20::krylov::real_generalized_schur(matrix, metric, false);
  auto matrix_schur_form = schur.matrix_schur_form;
  auto metric_schur_form = schur.metric_schur_form;
  auto eigenvalues = schur.eigenvalues;
  auto eigenvectors = uni20::krylov::real_generalized_schur_right_eigenvectors(std::move(schur));

  ASSERT_EQ(eigenvectors.computed_vectors, 3);
  ASSERT_EQ(eigenvectors.right_eigenvectors.rows(), 3);
  ASSERT_EQ(eigenvectors.right_eigenvectors.cols(), 3);
  for (std::size_t col = 0; col < eigenvalues.size(); ++col)
  {
    EXPECT_LE(static_cast<double>(generalized_schur_right_eigenvector_residual_max(
                  matrix_schur_form, metric_schur_form, eigenvalues[col], eigenvectors.right_eigenvectors, col)),
              scaled_tolerance<Real>(100.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, EstimatesRealGeneralizedSchurEigenpairConditioning)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(3, 3);
  matrix(0, 0) = Real{1};
  matrix(1, 1) = Real{3};
  matrix(2, 2) = Real{2};

  uni20::krylov::Matrix<Real> metric(3, 3);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{1};
  metric(2, 2) = Real{1};

  auto schur = uni20::krylov::real_generalized_schur(std::move(matrix), std::move(metric), false);
  auto result = uni20::krylov::real_generalized_schur_condition_estimates(std::move(schur));

  ASSERT_EQ(result.computed_estimates, 3);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 3);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 3);
  for (std::size_t index = 0; index < result.computed_estimates; ++index)
  {
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvalue_condition_numbers[index]), 0.0);
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvector_condition_numbers[index]), 0.0);
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesRealNonsymmetricExpertEigenvectorsWithDiagnostics)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(2, 2);
  matrix(0, 0) = Real{2};
  matrix(0, 1) = Real{1};
  matrix(1, 1) = Real{3};

  auto result = uni20::krylov::real_nonsymmetric_expert_eigensystem(matrix, true);

  std::vector<uni20::complex<Real>> eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  ASSERT_EQ(eigenvalues.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 2);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 2);
  ASSERT_EQ(result.balance_scale.size(), 2);
  EXPECT_NEAR(static_cast<double>(eigenvalues[0].real()), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(eigenvalues[1].real()), 3.0, tolerance<Real>());
  EXPECT_GT(static_cast<double>(result.balanced_matrix_norm), 0.0);
  EXPECT_LT(result.balanced_first, result.balanced_last_exclusive);
  EXPECT_LE(result.balanced_last_exclusive, 2);
  for (std::size_t index = 0; index < 2; ++index)
  {
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvalue_condition_numbers[index]), 0.0);
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvector_condition_numbers[index]), 0.0);
  }

  for (std::size_t col = 0; col < result.eigenvalues.size(); ++col)
  {
    auto const lambda = result.eigenvalues[col];
    auto const x0 = result.right_eigenvectors(0, col);
    auto const x1 = result.right_eigenvectors(1, col);
    auto const residual0 = Real{2} * x0 + x1 - lambda * x0;
    auto const residual1 = Real{3} * x1 - lambda * x1;
    double const residual_norm = std::hypot(abs_as_double(residual0), abs_as_double(residual1));
    EXPECT_LT(residual_norm, scaled_tolerance<Real>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesRealGeneralizedNonsymmetricDiagonalEigenvectors)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(2, 2);
  matrix(0, 0) = Real{2};
  matrix(1, 1) = Real{9};

  uni20::krylov::Matrix<Real> metric(2, 2);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{3};

  auto result = uni20::krylov::real_generalized_nonsymmetric_eigensystem(std::move(matrix), std::move(metric), true);

  std::vector<uni20::complex<Real>> eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  ASSERT_EQ(eigenvalues.size(), 2);
  ASSERT_EQ(result.alpha.size(), 2);
  ASSERT_EQ(result.beta.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(eigenvalues[0].real()), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(eigenvalues[1].real()), 3.0, tolerance<Real>());

  for (std::size_t col = 0; col < result.eigenvalues.size(); ++col)
  {
    auto const lambda = result.eigenvalues[col];
    auto const x0 = result.right_eigenvectors(0, col);
    auto const x1 = result.right_eigenvectors(1, col);
    auto const residual0 = Real{2} * x0 - lambda * x0;
    auto const residual1 = Real{9} * x1 - lambda * Real{3} * x1;
    double const residual_norm = std::hypot(abs_as_double(residual0), abs_as_double(residual1));
    EXPECT_LT(residual_norm, scaled_tolerance<Real>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesRealGeneralizedNonsymmetricExpertEigenvectorsWithDiagnostics)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(2, 2);
  matrix(0, 0) = Real{2};
  matrix(1, 1) = Real{9};

  uni20::krylov::Matrix<Real> metric(2, 2);
  metric(0, 0) = Real{1};
  metric(1, 1) = Real{3};

  auto result =
      uni20::krylov::real_generalized_nonsymmetric_expert_eigensystem(std::move(matrix), std::move(metric), true);

  std::vector<uni20::complex<Real>> eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  ASSERT_EQ(eigenvalues.size(), 2);
  ASSERT_EQ(result.alpha.size(), 2);
  ASSERT_EQ(result.beta.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
  ASSERT_EQ(result.reciprocal_eigenvalue_condition_numbers.size(), 2);
  ASSERT_EQ(result.reciprocal_eigenvector_condition_numbers.size(), 2);
  ASSERT_EQ(result.left_balance_scale.size(), 2);
  ASSERT_EQ(result.right_balance_scale.size(), 2);
  EXPECT_NEAR(static_cast<double>(eigenvalues[0].real()), 2.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(eigenvalues[1].real()), 3.0, tolerance<Real>());
  EXPECT_GT(static_cast<double>(result.balanced_matrix_norm), 0.0);
  EXPECT_GT(static_cast<double>(result.balanced_metric_norm), 0.0);
  EXPECT_LT(result.balanced_first, result.balanced_last_exclusive);
  EXPECT_LE(result.balanced_last_exclusive, 2);
  for (std::size_t index = 0; index < 2; ++index)
  {
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvalue_condition_numbers[index]), 0.0);
    EXPECT_GT(static_cast<double>(result.reciprocal_eigenvector_condition_numbers[index]), 0.0);
  }

  for (std::size_t col = 0; col < result.eigenvalues.size(); ++col)
  {
    auto const lambda = result.eigenvalues[col];
    auto const x0 = result.right_eigenvectors(0, col);
    auto const x1 = result.right_eigenvectors(1, col);
    auto const residual0 = Real{2} * x0 - lambda * x0;
    auto const residual1 = Real{9} * x1 - lambda * Real{3} * x1;
    double const residual_norm = std::hypot(abs_as_double(residual0), abs_as_double(residual1));
    EXPECT_LT(residual_norm, scaled_tolerance<Real>(20.0));
  }
}

TYPED_TEST(KrylovDenseSubspaceTypedTest, SolvesRealNonsymmetricComplexConjugatePair)
{
  using Real = TypeParam;

  uni20::krylov::Matrix<Real> matrix(2, 2);
  matrix(0, 0) = Real{1};
  matrix(1, 0) = Real{-2};
  matrix(0, 1) = Real{2};
  matrix(1, 1) = Real{1};

  auto result = uni20::krylov::real_nonsymmetric_eigensystem(std::move(matrix), true);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.rows(), 2);
  ASSERT_EQ(result.right_eigenvectors.cols(), 2);
  EXPECT_NEAR(static_cast<double>(result.eigenvalues[0].real()), 1.0, tolerance<Real>());
  EXPECT_NEAR(static_cast<double>(std::abs(result.eigenvalues[0].imag())), 2.0, tolerance<Real>());
  EXPECT_NEAR(abs_as_double(result.eigenvalues[1] - std::conj(result.eigenvalues[0])), 0.0, tolerance<Real>());

  for (std::size_t col = 0; col < result.eigenvalues.size(); ++col)
  {
    auto const lambda = result.eigenvalues[col];
    auto const x0 = result.right_eigenvectors(0, col);
    auto const x1 = result.right_eigenvectors(1, col);
    auto const residual0 = x0 + Real{2} * x1 - lambda * x0;
    auto const residual1 = Real{-2} * x0 + x1 - lambda * x1;
    double const residual_norm = std::hypot(abs_as_double(residual0), abs_as_double(residual1));
    EXPECT_LT(residual_norm, tolerance<Real>());
  }
}

} // namespace
