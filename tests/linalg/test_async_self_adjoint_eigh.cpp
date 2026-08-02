#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <stdexcept>
#include <utility>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;
using eigenvalue_type = uni20::Tensor<double, 1>;
using async_matrix_type = uni20::async::Async<matrix_type>;
using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;

matrix_type make_symmetric_matrix()
{
  matrix_type matrix(2, 2);
  matrix[0, 0] = 2.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;
  return matrix;
}

deferred_matrix_type make_deferred_symmetric_matrix()
{
  deferred_matrix_type matrix(2, 2);
  auto lease = uni20::test::acquire_host_write_access_sync(matrix);
  auto& span = lease.mdspan();
  span[0, 0] = 2.0;
  span[0, 1] = 1.0;
  span[1, 0] = 1.0;
  span[1, 1] = 2.0;
  return matrix;
}

void expect_eigensystem(eigenvalue_type const& eigenvalues, matrix_type const& eigenvectors)
{
  ASSERT_EQ(eigenvalues.extent(0), 2);
  ASSERT_EQ(eigenvectors.rows(), 2);
  ASSERT_EQ(eigenvectors.cols(), 2);
  EXPECT_NEAR(eigenvalues[0], 1.0, 1e-13);
  EXPECT_NEAR(eigenvalues[1], 3.0, 1e-13);

  for (uni20::index_type col = 0; col < 2; ++col)
  {
    EXPECT_NEAR((2.0 * eigenvectors[0, col] + eigenvectors[1, col]), (eigenvalues[col] * eigenvectors[0, col]), 1e-12);
    EXPECT_NEAR((eigenvectors[0, col] + 2.0 * eigenvectors[1, col]), (eigenvalues[col] * eigenvectors[1, col]), 1e-12);
  }
}

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
} // namespace

TEST(AsyncSelfAdjointEighTest, PreservingSolveReturnsIndependentAsyncOutputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_symmetric_matrix();

  auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(matrix);

  static_assert(std::same_as<decltype(eigenvalues), uni20::async::Async<eigenvalue_type>>);
  static_assert(std::same_as<decltype(eigenvectors), uni20::async::Async<matrix_type>>);
  expect_eigensystem(eigenvalues.get_wait(scheduler), eigenvectors.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[0, 1]), 1.0);
}

TEST(AsyncSelfAdjointEighTest, PreservingSolveAcceptsDeferredTensorViews)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<deferred_matrix_type> matrix = make_deferred_symmetric_matrix();

  auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(uni20::linalg::LapackBackend{}, matrix);

  expect_eigensystem(eigenvalues.get_wait(scheduler), eigenvectors.get_wait(scheduler));
  auto preserved = uni20::test::acquire_host_read_access_sync(matrix.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((preserved.mdspan()[0, 1]), 1.0);
}

TEST(AsyncSelfAdjointEighTest, ConsumingSolveReusesInputAllocation)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix_type matrix_value = make_symmetric_matrix();
  double* original_storage = matrix_value.mutable_handle();
  async_matrix_type matrix = std::move(matrix_value);

  auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(std::move(matrix));

  auto const& eigenvector_value = eigenvectors.get_wait(scheduler);
  EXPECT_EQ(eigenvector_value.handle(), original_storage);
  expect_eigensystem(eigenvalues.get_wait(scheduler), eigenvector_value);
  EXPECT_THROW((void)matrix.get_wait(scheduler), uni20::async::buffer_read_uninitialized);
}

TEST(AsyncSelfAdjointEighTest, ConsumingFailurePropagatesToBothOutputsAndInput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix(matrix_type(2, 3));
  ErrorModeGuard const error_mode;

  auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(std::move(matrix));

  EXPECT_THROW((void)eigenvalues.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)eigenvectors.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)matrix.get_wait(scheduler), std::runtime_error);
}

#if UNI20_FLOAT128_PROVIDER_MPLAPACK
TEST(AsyncSelfAdjointEighTest, SupportsConfiguredFloat128Backend)
{
  using real_type = uni20::float128;
  using matrix128_type = uni20::DenseMatrix<real_type>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix128_type matrix_value(2, 2);
  matrix_value[0, 0] = real_type{2};
  matrix_value[0, 1] = real_type{};
  matrix_value[1, 0] = real_type{};
  matrix_value[1, 1] = real_type{3};
  uni20::async::Async<matrix128_type> matrix = std::move(matrix_value);

  auto [eigenvalues, eigenvectors] = uni20::linalg::eigh(std::move(matrix));

  auto const& values = eigenvalues.get_wait(scheduler);
  auto const& vectors = eigenvectors.get_wait(scheduler);
  EXPECT_TRUE(values[0] == real_type{2});
  EXPECT_TRUE(values[1] == real_type{3});
  EXPECT_EQ(vectors.rows(), 2);
  EXPECT_EQ(vectors.cols(), 2);
}
#endif
