#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <cmath>
#include <concepts>

namespace
{

template <class Matrix> void initialize_coefficients(Matrix& matrix)
{
  matrix[0, 0] = 3.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;
}

template <class Matrix> void initialize_rhs(Matrix& rhs)
{
  rhs[0, 0] = 9.0;
  rhs[1, 0] = 8.0;
  rhs[0, 1] = 1.0;
  rhs[1, 1] = 0.0;
}

template <class Matrix> void check_solution(Matrix const& solution)
{
  EXPECT_NEAR((solution[0, 0]), 2.0, 1.0e-14);
  EXPECT_NEAR((solution[1, 0]), 3.0, 1.0e-14);
  EXPECT_NEAR((solution[0, 1]), 0.4, 1.0e-14);
  EXPECT_NEAR((solution[1, 1]), -0.2, 1.0e-14);
}

} // namespace

TEST(LinearSolveTest, ValueApiPreservesInputsAndSolvesMultipleRightHandSides)
{
  uni20::DenseMatrix<double, uni20::RowMajor> coefficients(2, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> rhs(2, 2);
  initialize_coefficients(coefficients);
  initialize_rhs(rhs);

  auto solution = uni20::linalg::solve(coefficients, rhs);

  static_assert(std::same_as<typename decltype(solution)::layout_type, uni20::ColumnMajor>);
  check_solution(solution);
  EXPECT_DOUBLE_EQ((coefficients[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 1]), 2.0);
  EXPECT_DOUBLE_EQ((rhs[0, 0]), 9.0);
  EXPECT_DOUBLE_EQ((rhs[1, 1]), 0.0);
}

TEST(LinearSolveTest, LapackInplaceApiOverwritesWorkspaces)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  initialize_coefficients(coefficients);
  initialize_rhs(rhs);

  uni20::linalg::solve_inplace(uni20::linalg::LapackBackend{}, coefficients, rhs);

  check_solution(rhs);
  EXPECT_NE((coefficients[1, 0]), 1.0);
}

TEST(LinearSolveTest, CpuReferencePerformsPartialPivoting)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  coefficients[0, 0] = 0.0;
  coefficients[0, 1] = 2.0;
  coefficients[1, 0] = 1.0;
  coefficients[1, 1] = 3.0;
  uni20::DenseMatrix<double> rhs(2, 1);
  rhs[0, 0] = 4.0;
  rhs[1, 0] = 7.0;

  uni20::linalg::solve_inplace(uni20::linalg::CpuReferenceBackend{}, coefficients, rhs);

  EXPECT_NEAR((rhs[0, 0]), 1.0, 1.0e-14);
  EXPECT_NEAR((rhs[1, 0]), 2.0, 1.0e-14);
}

TEST(LinearSolveTest, CpuReferenceSupportsComplexSystems)
{
  using complex_type = uni20::complex<double>;
  uni20::DenseMatrix<complex_type> coefficients(2, 2);
  coefficients[0, 0] = complex_type{1.0, 1.0};
  coefficients[0, 1] = complex_type{};
  coefficients[1, 0] = complex_type{};
  coefficients[1, 1] = complex_type{2.0, -1.0};
  uni20::DenseMatrix<complex_type> rhs(2, 1);
  rhs[0, 0] = complex_type{2.0, 2.0};
  rhs[1, 0] = complex_type{5.0, 0.0};

  auto solution = uni20::linalg::solve(uni20::linalg::CpuReferenceBackend{}, coefficients, rhs);

  EXPECT_NEAR(std::abs(solution[0, 0] - complex_type{2.0, 0.0}), 0.0, 1.0e-14);
  EXPECT_NEAR(std::abs(solution[1, 0] - complex_type{2.0, 1.0}), 0.0, 1.0e-14);
}

TEST(LinearSolveTest, LapackDeclinesRowMajorWorkspacesWithoutMutation)
{
  uni20::DenseMatrix<double, uni20::RowMajor> coefficients(2, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> rhs(2, 2);
  initialize_coefficients(coefficients);
  initialize_rhs(rhs);
  auto coefficient_descriptor = uni20::mdspec_of(coefficients);
  auto rhs_descriptor = uni20::mdspec_of(rhs);

  bool const accepted = uni20::linalg::try_dispatch_kernel(
      uni20::linalg::LapackBackend{}, uni20::linalg::linear_solve_op{}, coefficient_descriptor, rhs_descriptor);

  EXPECT_FALSE(accepted);
  EXPECT_DOUBLE_EQ((coefficients[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((rhs[0, 0]), 9.0);
  EXPECT_DOUBLE_EQ((rhs[1, 1]), 0.0);
}

TEST(LinearSolveTest, AcquiresDeferredHostWorkspaces)
{
  uni20::test::DeferredHostTensor<double, 2> coefficients(2, 2);
  uni20::test::DeferredHostTensor<double, 2> rhs(2, 2);
  {
    auto coefficient_access = uni20::test::acquire_host_write_access_sync(coefficients);
    auto rhs_access = uni20::test::acquire_host_write_access_sync(rhs);
    auto coefficient_span = coefficient_access.mdspan();
    auto rhs_span = rhs_access.mdspan();
    initialize_coefficients(coefficient_span);
    initialize_rhs(rhs_span);
  }

  uni20::linalg::solve_inplace(coefficients, rhs);

  auto rhs_access = uni20::test::acquire_host_read_access_sync(rhs);
  check_solution(rhs_access.mdspan());
}

TEST(LinearSolveTest, SingularSystemIsTerminal)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  coefficients[0, 0] = 1.0;
  coefficients[0, 1] = 2.0;
  coefficients[1, 0] = 2.0;
  coefficients[1, 1] = 4.0;
  uni20::DenseMatrix<double> rhs(2, 1);
  rhs[0, 0] = 1.0;
  rhs[1, 0] = 2.0;

  EXPECT_DEATH(
      { uni20::linalg::solve_inplace(uni20::linalg::CpuReferenceBackend{}, coefficients, rhs); },
      "singular matrix in solve");
}

TEST(LinearSolveTest, RejectsMismatchedShapesBeforeDispatch)
{
  uni20::DenseMatrix<double> coefficients(2, 3);
  uni20::DenseMatrix<double> rhs(2, 1);

  EXPECT_DEATH({ (void)uni20::linalg::solve(coefficients, rhs); }, "solve requires a square coefficient matrix");
}
