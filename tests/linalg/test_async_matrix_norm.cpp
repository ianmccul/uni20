#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>

namespace
{
using matrix_type = uni20::DenseMatrix<double, uni20::RowMajor>;
using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;

template <class Matrix> void initialize_matrix(Matrix& matrix)
{
  matrix[0, 0] = 1.0;
  matrix[0, 1] = -2.0;
  matrix[1, 0] = 3.0;
  matrix[1, 1] = -4.0;
}

deferred_matrix_type make_deferred_matrix()
{
  deferred_matrix_type matrix(2, 2);
  auto lease = uni20::test::acquire_host_write_access_sync(matrix);
  initialize_matrix(lease.mdspan());
  return matrix;
}
} // namespace

TEST(AsyncMatrixNormTest, ReturnsStoragePreservingTensorAndHostScalar)
{
  matrix_type matrix_value(2, 2);
  initialize_matrix(matrix_value);
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<matrix_type> matrix = std::move(matrix_value);

  auto tensor_result = uni20::linalg::matrix_norm(matrix, uni20::linalg::MatrixNorm::Frobenius);
  auto host_result = uni20::linalg::matrix_norm_host(matrix, uni20::linalg::MatrixNorm::One);

  using expected_tensor_type = uni20::ScalarTensor<double>;
  static_assert(std::same_as<decltype(tensor_result), uni20::async::Async<expected_tensor_type>>);
  static_assert(std::same_as<decltype(host_result), uni20::async::Async<double>>);
  EXPECT_NEAR(tensor_result.get_wait(scheduler)[], std::sqrt(30.0), 1.0e-14);
  EXPECT_DOUBLE_EQ(host_result.get_wait(scheduler), 6.0);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[1, 1]), -4.0);
}

TEST(AsyncMatrixNormTest, DeferredInputUsesMdspecDispatch)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<deferred_matrix_type> matrix = make_deferred_matrix();

  auto result =
      uni20::linalg::matrix_norm_host(uni20::linalg::LapackBackend{}, matrix, uni20::linalg::MatrixNorm::Infinity);

  EXPECT_DOUBLE_EQ(result.get_wait(scheduler), 7.0);
  auto lease = uni20::test::acquire_host_read_access_sync(matrix.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((lease.mdspan()[0, 1]), -2.0);
}

TEST(AsyncMatrixNormTest, ConstructsExplicitScalarOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix_type matrix_value(2, 2);
  initialize_matrix(matrix_value);
  uni20::async::Async<matrix_type> matrix = std::move(matrix_value);
  uni20::async::Async<uni20::ScalarTensor<double>> output;

  uni20::linalg::matrix_norm(output, matrix, uni20::linalg::MatrixNorm::Infinity);

  EXPECT_DOUBLE_EQ(output.get_wait(scheduler)[], 7.0);
}
