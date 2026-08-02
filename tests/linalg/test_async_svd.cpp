#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/gtest.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/linalg/dispatch_error.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;
using singular_value_type = uni20::Tensor<double, 1>;
using async_matrix_type = uni20::async::Async<matrix_type>;
using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;
using async_deferred_matrix_type = uni20::async::Async<deferred_matrix_type>;

template <class AsyncTensor>
concept CanConsumeAsyncSvd = requires(AsyncTensor&& matrix) {
  uni20::linalg::singular_values(std::move(matrix));
  uni20::linalg::svd_left(std::move(matrix));
  uni20::linalg::svd_right(std::move(matrix));
  uni20::linalg::svd(std::move(matrix));
};

static_assert(CanConsumeAsyncSvd<async_deferred_matrix_type>);

matrix_type make_matrix()
{
  matrix_type matrix(3, 2);
  matrix[0, 0] = 3.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = -2.0;
  matrix[1, 1] = 4.0;
  matrix[2, 0] = 0.5;
  matrix[2, 1] = -1.0;
  return matrix;
}

deferred_matrix_type make_deferred_matrix()
{
  deferred_matrix_type matrix(3, 2);
  auto lease = uni20::test::acquire_host_write_access_sync(matrix);
  auto& span = lease.mdspan();
  span[0, 0] = 3.0;
  span[0, 1] = 1.0;
  span[1, 0] = -2.0;
  span[1, 1] = 4.0;
  span[2, 0] = 0.5;
  span[2, 1] = -1.0;
  return matrix;
}

template <class T> uni20::async::AsyncTask co_publish(uni20::async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
}

template <class Matrix, class Left, class SingularValues, class RightAdjoint>
void expect_reconstruction(Matrix const& original, Left const& left, SingularValues const& singular_values,
                           RightAdjoint const& right_adjoint, double tolerance)
{
  std::size_t const rank = static_cast<std::size_t>(singular_values.extent(0));
  for (uni20::index_type row = 0; row < original.rows(); ++row)
  {
    for (uni20::index_type column = 0; column < original.cols(); ++column)
    {
      typename Matrix::value_type reconstructed{};
      for (std::size_t inner = 0; inner < rank; ++inner)
      {
        auto const index = static_cast<uni20::index_type>(inner);
        reconstructed += left[row, index] * singular_values[index] * right_adjoint[index, column];
      }
      EXPECT_LE(static_cast<double>(std::abs(reconstructed - original[row, column])), tolerance);
    }
  }
}

struct DecliningSvdBackend
{
    static constexpr std::string_view name = "declining_svd";
};

template <class Operation>
concept SvdOperation = std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::singular_values_op> ||
                       std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::svd_left_op> ||
                       std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::svd_right_op> ||
                       std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::svd_op>;

template <SvdOperation Operation, class... Args>
consteval auto kernel_accepts_types(DecliningSvdBackend const&, Operation const&, Args&...)
{
  return uni20::linalg::kernel_types_maybe;
}

template <SvdOperation Operation, class... Args>
uni20::linalg::KernelAttempt try_kernel(DecliningSvdBackend, Operation const&, Args&&...)
{
  return uni20::linalg::KernelAttempt::unsupported_layout;
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

TEST(AsyncSvdTest, PreservingSolveAwaitsInputAndReturnsIndependentOutputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix;
  matrix_type const expected = make_matrix();
  uni20::async::schedule(co_publish(matrix.write(), expected));

  auto [left, singular_values, right_adjoint] = uni20::linalg::svd(matrix);

  static_assert(std::same_as<decltype(left), uni20::async::Async<matrix_type>>);
  static_assert(std::same_as<decltype(singular_values), uni20::async::Async<singular_value_type>>);
  static_assert(std::same_as<decltype(right_adjoint), uni20::async::Async<matrix_type>>);

  auto const& left_value = left.get_wait(scheduler);
  auto const& singular_value = singular_values.get_wait(scheduler);
  auto const& right_value = right_adjoint.get_wait(scheduler);
  ASSERT_EQ(left_value.rows(), 3);
  ASSERT_EQ(left_value.cols(), 2);
  ASSERT_EQ(singular_value.extent(0), 2);
  ASSERT_EQ(right_value.rows(), 2);
  ASSERT_EQ(right_value.cols(), 2);
  expect_reconstruction(expected, left_value, singular_value, right_value, 1.0e-12);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[1, 1]), 4.0);
}

TEST(AsyncSvdTest, ComplexFullFactorsHonorIndependentOptions)
{
  using scalar_type = uni20::complex<double>;
  using complex_matrix_type = uni20::DenseMatrix<scalar_type>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  complex_matrix_type matrix_value(2, 3);
  matrix_value[0, 0] = scalar_type{1.0, 2.0};
  matrix_value[0, 1] = scalar_type{-3.0, 0.5};
  matrix_value[0, 2] = scalar_type{0.0, -1.0};
  matrix_value[1, 0] = scalar_type{2.0, -1.0};
  matrix_value[1, 1] = scalar_type{0.5, 4.0};
  matrix_value[1, 2] = scalar_type{-2.0, 3.0};
  uni20::async::Async<complex_matrix_type> matrix = matrix_value;

  auto [left, singular_values, right_adjoint] =
      uni20::linalg::svd(matrix, uni20::linalg::SvdOptions{.right = uni20::linalg::SvdVectorExtent::Full});

  auto const& left_value = left.get_wait(scheduler);
  auto const& singular_value = singular_values.get_wait(scheduler);
  auto const& right_value = right_adjoint.get_wait(scheduler);
  ASSERT_EQ(left_value.rows(), 2);
  ASSERT_EQ(left_value.cols(), 2);
  ASSERT_EQ(singular_value.extent(0), 2);
  ASSERT_EQ(right_value.rows(), 3);
  ASSERT_EQ(right_value.cols(), 3);
  expect_reconstruction(matrix_value, left_value, singular_value, right_value, 2.0e-12);
}

TEST(AsyncSvdTest, ValuesOnlyAndOneSidedOutputsAreIndependent)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix_type const matrix_value = make_matrix();
  async_matrix_type matrix = matrix_value;

  auto singular_values = uni20::linalg::singular_values(matrix);
  auto [left, left_values] = uni20::linalg::svd_left(matrix);
  auto [right_values, right_adjoint] = uni20::linalg::svd_right(matrix);

  auto const& values = singular_values.get_wait(scheduler);
  auto const& left_matrix = left.get_wait(scheduler);
  auto const& left_singular_values = left_values.get_wait(scheduler);
  auto const& right_singular_values = right_values.get_wait(scheduler);
  auto const& right_matrix = right_adjoint.get_wait(scheduler);
  ASSERT_EQ(values.extent(0), 2);
  ASSERT_EQ(left_matrix.rows(), 3);
  ASSERT_EQ(left_matrix.cols(), 2);
  ASSERT_EQ(right_matrix.rows(), 2);
  ASSERT_EQ(right_matrix.cols(), 2);
  for (uni20::index_type index = 0; index < values.extent(0); ++index)
  {
    EXPECT_NEAR(values[index], left_singular_values[index], 1.0e-12);
    EXPECT_NEAR(values[index], right_singular_values[index], 1.0e-12);
  }
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[1, 1]), (matrix_value[1, 1]));
}

TEST(AsyncSvdTest, PreservingOperationsAcceptDeferredTensorViews)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<deferred_matrix_type> matrix = make_deferred_matrix();

  auto values_only = uni20::linalg::singular_values(matrix);
  auto [left, left_values] = uni20::linalg::svd_left(uni20::linalg::LapackBackend{}, matrix);
  auto [right_values, right_adjoint] = uni20::linalg::svd_right(matrix);
  auto [complete_left, complete_values, complete_right] = uni20::linalg::svd(uni20::linalg::LapackBackend{}, matrix);

  auto const& values = values_only.get_wait(scheduler);
  auto const& left_matrix = left.get_wait(scheduler);
  auto const& left_singular_values = left_values.get_wait(scheduler);
  auto const& right_singular_values = right_values.get_wait(scheduler);
  auto const& right_matrix = right_adjoint.get_wait(scheduler);
  auto const& complete_left_matrix = complete_left.get_wait(scheduler);
  auto const& complete_singular_values = complete_values.get_wait(scheduler);
  auto const& complete_right_matrix = complete_right.get_wait(scheduler);

  ASSERT_EQ(values.extent(0), 2);
  ASSERT_EQ(left_matrix.rows(), 3);
  ASSERT_EQ(left_matrix.cols(), 2);
  ASSERT_EQ(right_matrix.rows(), 2);
  ASSERT_EQ(right_matrix.cols(), 2);
  ASSERT_EQ(complete_left_matrix.rows(), 3);
  ASSERT_EQ(complete_left_matrix.cols(), 2);
  ASSERT_EQ(complete_right_matrix.rows(), 2);
  ASSERT_EQ(complete_right_matrix.cols(), 2);
  for (uni20::index_type index = 0; index < values.extent(0); ++index)
  {
    EXPECT_NEAR(values[index], left_singular_values[index], 1.0e-12);
    EXPECT_NEAR(values[index], right_singular_values[index], 1.0e-12);
    EXPECT_NEAR(values[index], complete_singular_values[index], 1.0e-12);
  }

  auto preserved = uni20::test::acquire_host_read_access_sync(matrix.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((preserved.mdspan()[1, 1]), 4.0);
}

TEST(AsyncSvdTest, ConsumingReducedLeftFactorAdoptsStoredAllocation)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_matrix();
  auto* storage = matrix.get_wait(scheduler).mdspan().data_handle();

  auto [left, singular_values] = uni20::linalg::svd_left(std::move(matrix));

  auto const& left_value = left.get_wait(scheduler);
  EXPECT_EQ(left_value.mdspan().data_handle(), storage);
  EXPECT_EQ(left_value.rows(), 3);
  EXPECT_EQ(left_value.cols(), 2);
  EXPECT_EQ(singular_values.get_wait(scheduler).extent(0), 2);
}

TEST(AsyncSvdTest, ConsumingDeferredTensorMaterializesAfterTakingInput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_deferred_matrix_type matrix = make_deferred_matrix();

  auto [left, singular_values, right_adjoint] = uni20::linalg::svd(std::move(matrix));

  EXPECT_EQ(left.get_wait(scheduler).rows(), 3);
  EXPECT_EQ(singular_values.get_wait(scheduler).extent(0), 2);
  EXPECT_EQ(right_adjoint.get_wait(scheduler).cols(), 2);
  EXPECT_THROW((void)matrix.get_wait(scheduler), uni20::async::buffer_read_uninitialized);
}

TEST(AsyncSvdTest, DispatchFailurePropagatesToAllOutputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_matrix();
  ErrorModeGuard const error_mode;

  auto [left, singular_values, right_adjoint] = uni20::linalg::svd(DecliningSvdBackend{}, matrix);

  EXPECT_THROW((void)left.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)singular_values.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)right_adjoint.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[0, 0]), 3.0);
}

TEST(AsyncSvdTest, ConsumingDispatchFailurePropagatesToInputAndOutputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_matrix();
  ErrorModeGuard const error_mode;

  auto [left, singular_values, right_adjoint] = uni20::linalg::svd(DecliningSvdBackend{}, std::move(matrix));

  EXPECT_THROW((void)left.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)singular_values.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)right_adjoint.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)matrix.get_wait(scheduler), uni20::linalg::KernelDispatchError);
}

TEST(AsyncSvdTest, PartialDispatchFailurePropagatesToEveryResult)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_matrix();
  ErrorModeGuard const error_mode;

  auto [left, singular_values] = uni20::linalg::svd_left(DecliningSvdBackend{}, matrix);

  EXPECT_THROW((void)left.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)singular_values.get_wait(scheduler), uni20::linalg::KernelDispatchError);
}

#if UNI20_FLOAT128_PROVIDER_MPLAPACK
TEST(AsyncSvdTest, SupportsConfiguredFloat128Backend)
{
  using real_type = uni20::float128;
  using matrix128_type = uni20::DenseMatrix<real_type>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix128_type matrix_value(2, 2);
  matrix_value[0, 0] = real_type{3};
  matrix_value[1, 1] = real_type{2};
  uni20::async::Async<matrix128_type> matrix = matrix_value;

  auto values_only = uni20::linalg::singular_values(matrix);
  auto [left, singular_values, right_adjoint] = uni20::linalg::svd(matrix);

  auto const& direct_values = values_only.get_wait(scheduler);
  auto const& values = singular_values.get_wait(scheduler);
  EXPECT_FLOATING_EQ(direct_values[0], real_type{3});
  EXPECT_FLOATING_EQ(direct_values[1], real_type{2});
  EXPECT_FLOATING_EQ(values[0], real_type{3});
  EXPECT_FLOATING_EQ(values[1], real_type{2});
  EXPECT_EQ(left.get_wait(scheduler).rows(), 2);
  EXPECT_EQ(right_adjoint.get_wait(scheduler).cols(), 2);
}
#endif
