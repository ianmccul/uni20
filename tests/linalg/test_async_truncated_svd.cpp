#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/linalg/dispatch_error.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <string_view>
#include <type_traits>
#include <utility>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;
using async_matrix_type = uni20::async::Async<matrix_type>;
using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;
using async_deferred_matrix_type = uni20::async::Async<deferred_matrix_type>;

static_assert(requires(async_deferred_matrix_type&& matrix) { uni20::linalg::truncated_svd(std::move(matrix)); });

matrix_type make_matrix()
{
  matrix_type matrix(3, 3);
  matrix[0, 0] = 4.0;
  matrix[1, 1] = 2.0;
  matrix[2, 2] = 0.5;
  return matrix;
}

deferred_matrix_type make_deferred_matrix()
{
  deferred_matrix_type matrix(3, 3);
  auto lease = uni20::test::acquire_host_write_access_sync(matrix);
  auto& span = lease.mdspan();
  span[0, 0] = 4.0;
  span[1, 1] = 2.0;
  span[2, 2] = 0.5;
  return matrix;
}

template <class T> uni20::async::AsyncTask co_publish(uni20::async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
}

struct DecliningSvdBackend
{
    static constexpr std::string_view name = "declining_svd";
};

template <class Operation>
  requires std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::svd_op>
consteval auto kernel_accepts_types(DecliningSvdBackend const&, Operation const&, auto&...)
{
  return uni20::linalg::kernel_types_maybe;
}

template <class Operation>
  requires std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::svd_op>
uni20::linalg::KernelAttempt try_kernel(DecliningSvdBackend, Operation const&, auto&&...)
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

TEST(AsyncTruncatedSvdTest, PreservingCallAwaitsInputAndReturnsIndependentOutputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix;
  uni20::async::schedule(co_publish(matrix.write(), make_matrix()));

  auto [left, singular_values, right_adjoint, truncation] = uni20::linalg::truncated_svd(
      matrix, uni20::linalg::SvdTruncationPolicy<double>{.maximum_discarded_weight = 0.02});

  static_assert(std::same_as<decltype(left), uni20::async::Async<matrix_type>>);
  static_assert(std::same_as<decltype(singular_values), uni20::async::Async<uni20::Tensor<double, 1>>>);
  static_assert(std::same_as<decltype(right_adjoint), uni20::async::Async<matrix_type>>);
  static_assert(std::same_as<decltype(truncation), uni20::async::Async<uni20::linalg::SvdTruncationInfo<double>>>);

  EXPECT_EQ(left.get_wait(scheduler).cols(), 2);
  EXPECT_EQ(singular_values.get_wait(scheduler).extent(0), 2);
  EXPECT_EQ(right_adjoint.get_wait(scheduler).rows(), 2);
  auto const& info = truncation.get_wait(scheduler);
  EXPECT_EQ(info.available_rank, 3);
  EXPECT_EQ(info.retained_rank, 2);
  EXPECT_NEAR(info.discarded_weight, 0.25 / 20.25, 1.0e-15);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[0, 0]), 4.0);
}

TEST(AsyncTruncatedSvdTest, PreservingCallAcceptsDeferredTensorViews)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<deferred_matrix_type> matrix = make_deferred_matrix();

  auto [left, singular_values, right_adjoint, truncation] = uni20::linalg::truncated_svd(
      uni20::linalg::LapackBackend{}, matrix, uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 2});

  EXPECT_EQ(left.get_wait(scheduler).cols(), 2);
  EXPECT_EQ(singular_values.get_wait(scheduler).extent(0), 2);
  EXPECT_EQ(right_adjoint.get_wait(scheduler).rows(), 2);
  EXPECT_EQ(truncation.get_wait(scheduler).retained_rank, 2);
  auto preserved = uni20::test::acquire_host_read_access_sync(matrix.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((preserved.mdspan()[0, 0]), 4.0);
}

TEST(AsyncTruncatedSvdTest, ConsumingCallConsumesInputAndPublishesFourResults)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_matrix();

  auto [left, singular_values, right_adjoint, truncation] = uni20::linalg::truncated_svd(
      std::move(matrix), uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 1});

  EXPECT_EQ(left.get_wait(scheduler).cols(), 1);
  EXPECT_EQ(singular_values.get_wait(scheduler).extent(0), 1);
  EXPECT_EQ(right_adjoint.get_wait(scheduler).rows(), 1);
  EXPECT_EQ(truncation.get_wait(scheduler).retained_rank, 1);
}

TEST(AsyncTruncatedSvdTest, ConsumingDeferredTensorMaterializesAfterTakingInput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_deferred_matrix_type matrix = make_deferred_matrix();

  auto [left, singular_values, right_adjoint, truncation] = uni20::linalg::truncated_svd(
      std::move(matrix), uni20::linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 2});

  EXPECT_EQ(left.get_wait(scheduler).cols(), 2);
  EXPECT_EQ(singular_values.get_wait(scheduler).extent(0), 2);
  EXPECT_EQ(right_adjoint.get_wait(scheduler).rows(), 2);
  EXPECT_EQ(truncation.get_wait(scheduler).retained_rank, 2);
  EXPECT_THROW((void)matrix.get_wait(scheduler), uni20::async::buffer_read_uninitialized);
}

TEST(AsyncTruncatedSvdTest, DispatchFailurePropagatesToInputAndAllOutputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_matrix();
  ErrorModeGuard const error_mode;

  auto [left, singular_values, right_adjoint, truncation] =
      uni20::linalg::truncated_svd(DecliningSvdBackend{}, std::move(matrix));

  EXPECT_THROW((void)left.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)singular_values.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)right_adjoint.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)truncation.get_wait(scheduler), uni20::linalg::KernelDispatchError);
  EXPECT_THROW((void)matrix.get_wait(scheduler), uni20::linalg::KernelDispatchError);
}

TEST(AsyncTruncatedSvdTest, InvalidPolicyPropagatesToEveryOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_matrix();
  ErrorModeGuard const error_mode;

  auto [left, singular_values, right_adjoint, truncation] = uni20::linalg::truncated_svd(
      matrix, uni20::linalg::SvdTruncationPolicy<double>{.minimum_retained_extent = 2, .maximum_retained_extent = 1});

  EXPECT_THROW((void)left.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)singular_values.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)right_adjoint.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)truncation.get_wait(scheduler), std::runtime_error);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[0, 0]), 4.0);
}
