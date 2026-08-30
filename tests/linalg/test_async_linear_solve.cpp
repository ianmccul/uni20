#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <stdexcept>
#include <utility>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;
using row_major_matrix_type = uni20::DenseMatrix<double, uni20::RowMajor>;
using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;
using async_matrix_type = uni20::async::Async<matrix_type>;
using async_deferred_matrix_type = uni20::async::Async<deferred_matrix_type>;

static_assert(requires(async_deferred_matrix_type const& coefficients, async_deferred_matrix_type const& rhs) {
  uni20::linalg::solve(coefficients, rhs);
});
static_assert(requires(async_deferred_matrix_type& coefficients, async_deferred_matrix_type& rhs) {
  uni20::linalg::solve_inplace(coefficients, rhs);
});

class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

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

template <class Matrix> void expect_solution(Matrix const& solution)
{
  EXPECT_NEAR((solution[0, 0]), 2.0, 1.0e-14);
  EXPECT_NEAR((solution[1, 0]), 3.0, 1.0e-14);
  EXPECT_NEAR((solution[0, 1]), 0.4, 1.0e-14);
  EXPECT_NEAR((solution[1, 1]), -0.2, 1.0e-14);
}

matrix_type make_coefficients()
{
  matrix_type matrix(2, 2);
  initialize_coefficients(matrix);
  return matrix;
}

matrix_type make_rhs()
{
  matrix_type matrix(2, 2);
  initialize_rhs(matrix);
  return matrix;
}

deferred_matrix_type make_deferred_coefficients()
{
  deferred_matrix_type matrix(2, 2);
  auto lease = uni20::test::acquire_host_write_access_sync(matrix);
  initialize_coefficients(lease.mdspan());
  return matrix;
}

deferred_matrix_type make_deferred_rhs()
{
  deferred_matrix_type matrix(2, 2);
  auto lease = uni20::test::acquire_host_write_access_sync(matrix);
  initialize_rhs(lease.mdspan());
  return matrix;
}
} // namespace

TEST(AsyncLinearSolveTest, PreservingSolveReturnsIndependentSolutionAndPreservesInputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type coefficients = make_coefficients();
  async_matrix_type rhs = make_rhs();

  auto solution = uni20::linalg::solve(coefficients, rhs);

  static_assert(std::same_as<decltype(solution), async_matrix_type>);
  expect_solution(solution.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((coefficients.get_wait(scheduler)[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((rhs.get_wait(scheduler)[1, 1]), 0.0);
}

TEST(AsyncLinearSolveTest, PreservingSolveAcceptsDeferredTensorViews)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_deferred_matrix_type coefficients = make_deferred_coefficients();
  async_deferred_matrix_type rhs = make_deferred_rhs();

  auto solution = uni20::linalg::solve(uni20::linalg::LapackBackend{}, coefficients, rhs);

  expect_solution(solution.get_wait(scheduler));
  auto coefficient_lease = uni20::test::acquire_host_read_access_sync(coefficients.get_wait(scheduler));
  auto rhs_lease = uni20::test::acquire_host_read_access_sync(rhs.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((coefficient_lease.mdspan()[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((rhs_lease.mdspan()[1, 1]), 0.0);
}

TEST(AsyncLinearSolveTest, InplaceSolveOverwritesBothWorkspaceEpochs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type coefficients = make_coefficients();
  async_matrix_type rhs = make_rhs();

  uni20::linalg::solve_inplace(coefficients, rhs);

  expect_solution(rhs.get_wait(scheduler));
  EXPECT_NE((coefficients.get_wait(scheduler)[1, 0]), 1.0);
}

TEST(AsyncLinearSolveTest, InplaceSolveWritesThroughMutableAliases)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type coefficient_parent = make_coefficients();
  async_matrix_type rhs_parent = make_rhs();
  auto coefficients = uni20::async::reshape_view(coefficient_parent, 2, 2);
  auto rhs = uni20::async::reshape_view(rhs_parent, 2, 2);

  uni20::linalg::solve_inplace(uni20::linalg::LapackBackend{}, coefficients, rhs);

  expect_solution(rhs_parent.get_wait(scheduler));
  EXPECT_NE((coefficient_parent.get_wait(scheduler)[1, 0]), 1.0);
}

TEST(AsyncLinearSolveTest, InplaceFailureIsPublishedToBothWorkspaceEpochs)
{
  matrix_type singular(2, 2);
  singular[0, 0] = 1.0;
  singular[0, 1] = 2.0;
  singular[1, 0] = 2.0;
  singular[1, 1] = 4.0;
  matrix_type rhs_value(2, 1);
  rhs_value[0, 0] = 1.0;
  rhs_value[1, 0] = 2.0;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type coefficients = std::move(singular);
  async_matrix_type rhs = std::move(rhs_value);
  ErrorModeGuard const error_mode;

  uni20::linalg::solve_inplace(uni20::linalg::LapackBackend{}, coefficients, rhs);

  EXPECT_THROW((void)coefficients.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)rhs.get_wait(scheduler), std::runtime_error);
}

TEST(AsyncLinearSolveTest, RejectsSharedQueueBeforeWriterEnrollment)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_coefficients();
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::linalg::solve_inplace(matrix, matrix), std::runtime_error);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[0, 0]), 3.0);
}
