#include <uni20/async/debug_scheduler.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;
using vector_type = uni20::Tensor<double, 1>;
using async_matrix_type = uni20::async::Async<matrix_type>;
using async_vector_type = uni20::async::Async<vector_type>;

template <class Tensor>
Tensor make_tensor(uni20::index_type rows, uni20::index_type cols,
                   std::initializer_list<typename Tensor::value_type> values)
{
  Tensor result(rows, cols);
  auto value = values.begin();
  for (uni20::index_type row = 0; row < rows; ++row)
  {
    for (uni20::index_type col = 0; col < cols; ++col)
    {
      result[row, col] = *value;
      ++value;
    }
  }
  return result;
}

vector_type make_vector(std::initializer_list<double> values)
{
  vector_type result(static_cast<uni20::index_type>(values.size()));
  auto value = values.begin();
  for (uni20::index_type index = 0; index < result.extent(0); ++index)
  {
    result[index] = *value;
    ++value;
  }
  return result;
}

template <class T> uni20::async::AsyncTask co_produce_value(uni20::async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
  co_return;
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

TEST(AsyncFixedOutputOpsTest, GemvAwaitsScalarsAndWritesThroughMutableAlias)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_tensor<matrix_type>(2, 2, {1.0, 2.0, 3.0, 4.0});
  async_vector_type input = make_vector({2.0, 3.0});
  async_vector_type output_parent = make_vector({10.0, 20.0});
  auto output = uni20::async::reshape_view(output_parent, 2);
  uni20::async::Async<double> alpha;
  uni20::async::Async<double> beta;

  uni20::async::schedule(co_produce_value(alpha.write(), 0.5));
  uni20::async::schedule(co_produce_value(beta.write(), 2.0));
  uni20::linalg::gemv(uni20::linalg::CpuReferenceBackend{}, output, alpha, matrix, input, beta);

  auto const& result = output_parent.get_wait(scheduler);
  EXPECT_DOUBLE_EQ(result[0], 24.0);
  EXPECT_DOUBLE_EQ(result[1], 49.0);
}

TEST(AsyncFixedOutputOpsTest, ContractNormalizesRawAxesAndAwaitsBeta)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type lhs = make_tensor<matrix_type>(2, 3, {1, 2, 3, 4, 5, 6});
  async_matrix_type rhs = make_tensor<matrix_type>(3, 2, {7, 8, 9, 10, 11, 12});
  async_matrix_type output = make_tensor<matrix_type>(2, 2, {1, 2, 3, 4});
  uni20::async::Async<double> beta;
  std::array<std::pair<std::size_t, std::size_t>, 1> const axes{{{1, 0}}};

  uni20::async::schedule(co_produce_value(beta.write(), 2.0));
  uni20::linalg::contract(uni20::linalg::CpuReferenceBackend{}, output, 1.0, lhs, rhs, axes, beta);

  auto const& result = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ((result[0, 0]), 60.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 68.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 145.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 162.0);
}

TEST(AsyncFixedOutputOpsTest, SetMatrixAwaitsValuesAndHonorsRegion)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_tensor<matrix_type>(3, 3, {9, 9, 9, 9, 9, 9, 9, 9, 9});
  uni20::async::Async<double> diagonal;

  uni20::async::schedule(co_produce_value(diagonal.write(), 4.0));
  uni20::linalg::set_matrix(matrix, diagonal, -1.0, uni20::linalg::MatrixRegion::Upper);

  auto const& result = matrix.get_wait(scheduler);
  EXPECT_DOUBLE_EQ((result[0, 0]), 4.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), -1.0);
  EXPECT_DOUBLE_EQ((result[0, 2]), -1.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 9.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 4.0);
  EXPECT_DOUBLE_EQ((result[1, 2]), -1.0);
  EXPECT_DOUBLE_EQ((result[2, 0]), 9.0);
  EXPECT_DOUBLE_EQ((result[2, 1]), 9.0);
  EXPECT_DOUBLE_EQ((result[2, 2]), 4.0);
}

TEST(AsyncFixedOutputOpsTest, MatrixExponentialAwaitsTimeAndUsesFixedOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type input = make_tensor<matrix_type>(1, 1, {2.0});
  async_matrix_type output = make_tensor<matrix_type>(1, 1, {0.0});
  uni20::async::Async<double> time;

  uni20::async::schedule(co_produce_value(time.write(), 0.5));
  uni20::linalg::matrix_exponential(output, input, time);

  EXPECT_NEAR((output.get_wait(scheduler)[0, 0]), std::exp(1.0), 1.0e-14);
}

TEST(AsyncFixedOutputOpsTest, RejectsFixedOutputQueueAliasBeforeEnrollment)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type matrix = make_tensor<matrix_type>(1, 1, {2.0});
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::linalg::matrix_exponential(matrix, matrix, 0.5), std::runtime_error);
  EXPECT_DOUBLE_EQ((matrix.get_wait(scheduler)[0, 0]), 2.0);
}

TEST(AsyncFixedOutputOpsTest, CopyConstructsHostOutputAndMaterializationObservesAccessor)
{
  using complex_matrix_type = uni20::DenseMatrix<uni20::complex<double>>;
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<complex_matrix_type> input =
      make_tensor<complex_matrix_type>(2, 2, {{1.0, 2.0}, {3.0, -4.0}, {-5.0, 6.0}, {7.0, 8.0}});
  uni20::async::Async<complex_matrix_type> copied;
  auto conjugated = uni20::async::conj(input);

  uni20::copy(copied, input);
  auto materialized = uni20::make_tensor(conjugated);

  EXPECT_EQ((copied.get_wait(scheduler)[1, 0]), uni20::complex<double>(-5.0, 6.0));
  EXPECT_EQ((materialized.get_wait(scheduler)[0, 1]), uni20::complex<double>(3.0, 4.0));
}

TEST(AsyncFixedOutputOpsTest, ConjugateInplaceMutatesOneAsyncEpoch)
{
  using complex_matrix_type = uni20::DenseMatrix<uni20::complex<double>>;
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<complex_matrix_type> matrix = make_tensor<complex_matrix_type>(1, 2, {{1.0, 2.0}, {3.0, -4.0}});

  uni20::conjugate_inplace(matrix);

  auto const& result = matrix.get_wait(scheduler);
  EXPECT_EQ((result[0, 0]), uni20::complex<double>(1.0, -2.0));
  EXPECT_EQ((result[0, 1]), uni20::complex<double>(3.0, 4.0));
}
