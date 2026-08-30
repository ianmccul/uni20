#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <concepts>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;
using row_major_matrix_type = uni20::DenseMatrix<double, uni20::RowMajor>;
using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;
using async_deferred_matrix_type = uni20::async::Async<deferred_matrix_type>;
using async_matrix_type = uni20::async::Async<matrix_type>;

static_assert(requires(async_deferred_matrix_type const& matrix) {
  uni20::linalg::qr(matrix);
  uni20::linalg::lq(matrix);
});
static_assert(requires(async_deferred_matrix_type&& matrix) {
  uni20::linalg::qr(std::move(matrix));
  uni20::linalg::lq(std::move(matrix));
});

template <class Matrix> void initialize_matrix(Matrix& matrix)
{
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.extent(1); ++column)
      matrix[row, column] = static_cast<double>((row + 1) * (column + 2)) + (row == column ? 1.0 : 0.0);
  }
}

template <class Matrix, class Q, class R> void expect_qr(Matrix const& matrix, Q const& q, R const& r)
{
  ASSERT_EQ(q.extent(0), matrix.extent(0));
  ASSERT_EQ(q.extent(1), r.extent(0));
  ASSERT_EQ(r.extent(1), matrix.extent(1));
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.extent(1); ++column)
    {
      double reconstructed{};
      for (uni20::index_type inner = 0; inner < q.extent(1); ++inner)
        reconstructed += q[row, inner] * r[inner, column];
      EXPECT_NEAR(reconstructed, (matrix[row, column]), 1.0e-12);
    }
  }
}

template <class Matrix, class L, class Q> void expect_lq(Matrix const& matrix, L const& l, Q const& q)
{
  ASSERT_EQ(l.extent(0), matrix.extent(0));
  ASSERT_EQ(l.extent(1), q.extent(0));
  ASSERT_EQ(q.extent(1), matrix.extent(1));
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.extent(1); ++column)
    {
      double reconstructed{};
      for (uni20::index_type inner = 0; inner < q.extent(0); ++inner)
        reconstructed += l[row, inner] * q[inner, column];
      EXPECT_NEAR(reconstructed, (matrix[row, column]), 1.0e-12);
    }
  }
}

struct DecliningQrLqBackend
{
    static constexpr std::string_view name = "declining_qr_lq";
};

struct RecordingQrBackend
{
    static constexpr std::string_view name = "recording_qr";
    static inline double* matrix_handle = nullptr;
};

template <class... Args>
consteval auto kernel_accepts_types(RecordingQrBackend const&, uni20::linalg::qr_op const&, Args&...)
{
  return uni20::linalg::kernel_types_maybe;
}

template <class Q, class R, class Matrix>
uni20::linalg::KernelAttempt try_kernel(RecordingQrBackend, uni20::linalg::qr_op const& operation, Q&& q, R&& r,
                                        Matrix&& matrix)
{
  RecordingQrBackend::matrix_handle = matrix.data_handle();
  return uni20::linalg::try_kernel(uni20::linalg::LapackBackend{}, operation, std::forward<Q>(q), std::forward<R>(r),
                                   std::forward<Matrix>(matrix));
}

template <class Operation, class... Args>
  requires(std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::qr_op> ||
           std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::lq_op>)
consteval auto kernel_accepts_types(DecliningQrLqBackend const&, Operation const&, Args&...)
{
  return uni20::linalg::kernel_types_maybe;
}

template <class Operation, class... Args>
  requires(std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::qr_op> ||
           std::same_as<std::remove_cvref_t<Operation>, uni20::linalg::lq_op>)
uni20::linalg::KernelAttempt try_kernel(DecliningQrLqBackend, Operation const&, Args&&...)
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

TEST(AsyncQrLqTest, PreservingQrReturnsIndependentAsyncFactors)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  row_major_matrix_type matrix_value(4, 3);
  initialize_matrix(matrix_value);
  auto const original = matrix_value;
  uni20::async::Async<row_major_matrix_type> matrix = std::move(matrix_value);

  auto [q, r] = uni20::linalg::qr(matrix);

  static_assert(std::same_as<decltype(q), uni20::async::Async<matrix_type>>);
  static_assert(std::same_as<decltype(r), uni20::async::Async<matrix_type>>);
  expect_qr(original, q.get_wait(scheduler), r.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[2, 1]), (original[2, 1]));
}

TEST(AsyncQrLqTest, PreservingLqAcceptsDeferredTensorView)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  deferred_matrix_type matrix_value(3, 4);
  {
    auto access = uni20::test::acquire_host_write_access_sync(matrix_value);
    initialize_matrix(access.mdspan());
  }
  auto const original = matrix_value.storage();
  async_deferred_matrix_type matrix = std::move(matrix_value);

  auto [l, q] = uni20::linalg::lq(uni20::linalg::LapackBackend{}, matrix);

  stdex::mdspan<double const, stdex::dextents<uni20::index_type, 2>, stdex::layout_left> original_span(original.data(),
                                                                                                       3, 4);
  expect_lq(original_span, l.get_wait(scheduler), q.get_wait(scheduler));
  auto preserved = uni20::test::acquire_host_read_access_sync(matrix.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((preserved.mdspan()[2, 1]), (original_span[2, 1]));
}

TEST(AsyncQrLqTest, ConsumingQrUsesColumnMajorInputAsWorkspace)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix_type matrix_value(4, 3);
  initialize_matrix(matrix_value);
  auto const original = matrix_value;
  double* const original_handle = matrix_value.mutable_handle();
  async_matrix_type matrix = std::move(matrix_value);
  RecordingQrBackend::matrix_handle = nullptr;

  auto [q, r] = uni20::linalg::qr(RecordingQrBackend{}, std::move(matrix));

  expect_qr(original, q.get_wait(scheduler), r.get_wait(scheduler));
  EXPECT_EQ(RecordingQrBackend::matrix_handle, original_handle);
  EXPECT_THROW((void)matrix.get_wait(scheduler), uni20::async::buffer_read_uninitialized);
}

TEST(AsyncQrLqTest, ConsumingLqMaterializesDeferredInput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  deferred_matrix_type matrix_value(3, 4);
  {
    auto access = uni20::test::acquire_host_write_access_sync(matrix_value);
    initialize_matrix(access.mdspan());
  }
  auto const original = matrix_value.storage();
  async_deferred_matrix_type matrix = std::move(matrix_value);

  auto [l, q] = uni20::linalg::lq(std::move(matrix));

  stdex::mdspan<double const, stdex::dextents<uni20::index_type, 2>, stdex::layout_left> original_span(original.data(),
                                                                                                       3, 4);
  expect_lq(original_span, l.get_wait(scheduler), q.get_wait(scheduler));
  EXPECT_THROW((void)matrix.get_wait(scheduler), uni20::async::buffer_read_uninitialized);
}

TEST(AsyncQrLqTest, FailurePropagatesToBothFactorsAndPreservesInput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  row_major_matrix_type matrix_value(3, 2);
  initialize_matrix(matrix_value);
  auto const original = matrix_value;
  uni20::async::Async<row_major_matrix_type> matrix = std::move(matrix_value);
  ErrorModeGuard const error_mode;

  auto [q, r] = uni20::linalg::qr(DecliningQrLqBackend{}, matrix);

  EXPECT_THROW((void)q.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)r.get_wait(scheduler), std::runtime_error);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[1, 1]), (original[1, 1]));
}

TEST(AsyncQrLqTest, ConsumingFailurePropagatesToFactorsAndInput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix_type matrix_value(3, 2);
  initialize_matrix(matrix_value);
  async_matrix_type matrix = std::move(matrix_value);
  ErrorModeGuard const error_mode;

  auto [l, q] = uni20::linalg::lq(DecliningQrLqBackend{}, std::move(matrix));

  EXPECT_THROW((void)l.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)q.get_wait(scheduler), std::runtime_error);
  EXPECT_THROW((void)matrix.get_wait(scheduler), std::runtime_error);
}
