#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace
{
using matrix_type = uni20::DenseMatrix<double>;
using async_matrix_type = uni20::async::Async<matrix_type>;

template <class Output, class Lhs, class Rhs>
concept CanAssignProduct =
    requires(Output& output, Lhs const& lhs, Rhs const& rhs) { uni20::linalg::assign_product(output, lhs, rhs); };

template <class Output, class Lhs, class Rhs>
concept CanAddProduct =
    requires(Output& output, Lhs const& lhs, Rhs const& rhs) { uni20::linalg::add_product(output, lhs, rhs); };

template <class Output, class Lhs, class Rhs, class Alpha>
concept CanAssignProductWithAlpha = requires(Output& output, Lhs const& lhs, Rhs const& rhs, Alpha const& alpha) {
  uni20::linalg::assign_product(output, lhs, rhs, alpha);
};

template <class Output, class Lhs, class Rhs, class Alpha>
concept CanAddProductWithAlpha = requires(Output& output, Lhs const& lhs, Rhs const& rhs, Alpha const& alpha) {
  uni20::linalg::add_product(output, lhs, rhs, alpha);
};

template <class Output, class Lhs, class Rhs, class Alpha, class Beta>
concept CanGemm = requires(Output& output, Lhs const& lhs, Rhs const& rhs, Alpha const& alpha, Beta const& beta) {
  uni20::linalg::gemm(output, alpha, lhs, rhs, beta);
};

static_assert(CanAssignProduct<async_matrix_type, async_matrix_type, async_matrix_type>);
static_assert(CanAddProduct<async_matrix_type, async_matrix_type, async_matrix_type>);
static_assert(CanGemm<async_matrix_type, async_matrix_type, async_matrix_type, double, double>);
static_assert(CanGemm<async_matrix_type, async_matrix_type, async_matrix_type, uni20::async::Async<double>,
                      uni20::async::Async<double>>);
static_assert(
    CanAssignProductWithAlpha<async_matrix_type, async_matrix_type, async_matrix_type, uni20::async::Async<double>>);
static_assert(
    CanAddProductWithAlpha<async_matrix_type, async_matrix_type, async_matrix_type, uni20::async::Async<double>>);
static_assert(!CanAssignProduct<matrix_type, async_matrix_type, async_matrix_type>);
static_assert(!CanAssignProduct<async_matrix_type, matrix_type, async_matrix_type>);
static_assert(!CanAssignProduct<async_matrix_type, async_matrix_type, matrix_type>);
static_assert(!CanAddProduct<matrix_type, async_matrix_type, async_matrix_type>);
static_assert(!CanAddProduct<async_matrix_type, matrix_type, async_matrix_type>);
static_assert(!CanAddProduct<async_matrix_type, async_matrix_type, matrix_type>);
static_assert(!CanGemm<matrix_type, async_matrix_type, async_matrix_type, double, double>);
static_assert(!CanGemm<async_matrix_type, matrix_type, async_matrix_type, double, double>);
static_assert(!CanGemm<async_matrix_type, async_matrix_type, matrix_type, double, double>);

template <class Matrix>
Matrix make_matrix(uni20::index_type rows, uni20::index_type cols,
                   std::initializer_list<typename Matrix::value_type> values)
{
  Matrix result(rows, cols);
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

template <class T> uni20::async::AsyncTask produce_value(uni20::async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
  co_return;
}

async_matrix_type schedule_product_from_local_inputs()
{
  async_matrix_type lhs;
  async_matrix_type rhs;
  async_matrix_type output;

  uni20::async::schedule(produce_value(lhs.write(), make_matrix<matrix_type>(2, 3, {1, 2, 3, 4, 5, 6})));
  uni20::async::schedule(produce_value(rhs.write(), make_matrix<matrix_type>(3, 2, {7, 8, 9, 10, 11, 12})));
  uni20::linalg::assign_product(output, lhs, rhs);
  return output;
}

async_matrix_type schedule_update_with_local_async_alpha()
{
  async_matrix_type lhs = make_matrix<matrix_type>(2, 2, {1, 2, 3, 4});
  async_matrix_type rhs = make_matrix<matrix_type>(2, 2, {5, 6, 7, 8});
  async_matrix_type output = make_matrix<matrix_type>(2, 2, {10, 20, 30, 40});
  uni20::async::Async<double> alpha;

  uni20::async::schedule(produce_value(alpha.write(), 0.5));
  uni20::linalg::add_product(output, lhs, rhs, alpha);
  return output;
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

void expect_matrix_product(matrix_type const& result)
{
  ASSERT_EQ(result.rows(), 2);
  ASSERT_EQ(result.cols(), 2);
  EXPECT_DOUBLE_EQ((result[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 154.0);
}
} // namespace

TEST(AsyncMatrixProductTest, AssignProductConstructsOutputAndRetainsPendingInputs)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);

  auto output = schedule_product_from_local_inputs();

  expect_matrix_product(output.get_wait(scheduler));
}

TEST(AsyncMatrixProductTest, ExplicitSelectorResizesExistingOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type lhs = make_matrix<matrix_type>(2, 3, {1, 2, 3, 4, 5, 6});
  async_matrix_type rhs = make_matrix<matrix_type>(3, 2, {7, 8, 9, 10, 11, 12});
  async_matrix_type output = make_matrix<matrix_type>(1, 1, {0});

  uni20::linalg::assign_product(uni20::linalg::CpuReferenceBackend{}, output, lhs, rhs);

  expect_matrix_product(output.get_wait(scheduler));
}

TEST(AsyncMatrixProductTest, AddProductReadsAndUpdatesExistingOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type lhs = make_matrix<matrix_type>(2, 2, {1, 2, 3, 4});
  async_matrix_type rhs = make_matrix<matrix_type>(2, 2, {5, 6, 7, 8});
  async_matrix_type output = make_matrix<matrix_type>(2, 2, {10, 20, 30, 40});

  uni20::linalg::add_product(output, lhs, rhs, 0.5);

  auto const& result = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ((result[0, 0]), 19.5);
  EXPECT_DOUBLE_EQ((result[0, 1]), 31.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 51.5);
  EXPECT_DOUBLE_EQ((result[1, 1]), 65.0);
}

TEST(AsyncMatrixProductTest, GemmAwaitsBothScalarsAndUpdatesFixedOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type lhs = make_matrix<matrix_type>(2, 2, {1, 2, 3, 4});
  async_matrix_type rhs = make_matrix<matrix_type>(2, 2, {5, 6, 7, 8});
  async_matrix_type output = make_matrix<matrix_type>(2, 2, {10, 20, 30, 40});
  uni20::async::Async<double> alpha;
  uni20::async::Async<double> beta;

  uni20::async::schedule(produce_value(alpha.write(), 0.5));
  uni20::async::schedule(produce_value(beta.write(), 2.0));
  uni20::linalg::gemm(uni20::linalg::CpuReferenceBackend{}, output, alpha, lhs, rhs, beta);

  auto const& result = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ((result[0, 0]), 29.5);
  EXPECT_DOUBLE_EQ((result[0, 1]), 51.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 81.5);
  EXPECT_DOUBLE_EQ((result[1, 1]), 105.0);
}

TEST(AsyncMatrixProductTest, AddProductAwaitsAsyncAlphaAndRetainsItsEpoch)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);

  auto output = schedule_update_with_local_async_alpha();

  auto const& result = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ((result[0, 0]), 19.5);
  EXPECT_DOUBLE_EQ((result[0, 1]), 31.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 51.5);
  EXPECT_DOUBLE_EQ((result[1, 1]), 65.0);
}

TEST(AsyncMatrixProductTest, ReadOnlyConjugatedAliasMayShareAnInputQueue)
{
  using scalar_type = uni20::complex<double>;
  using complex_matrix_type = uni20::DenseMatrix<scalar_type>;
  using async_complex_matrix_type = uni20::async::Async<complex_matrix_type>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_complex_matrix_type input =
      make_matrix<complex_matrix_type>(2, 2, {scalar_type{1, 1}, scalar_type{}, scalar_type{}, scalar_type{2, -1}});
  async_complex_matrix_type output;
  auto conjugated = uni20::async::conj(input);

  uni20::linalg::assign_product(output, input, conjugated);

  auto const& result = output.get_wait(scheduler);
  EXPECT_EQ((result[0, 0]), scalar_type(2));
  EXPECT_EQ((result[0, 1]), scalar_type(0));
  EXPECT_EQ((result[1, 0]), scalar_type(0));
  EXPECT_EQ((result[1, 1]), scalar_type(5));
}

TEST(AsyncMatrixProductTest, RejectsOutputAliasOfInputBeforeCreatingBuffers)
{
  using scalar_type = uni20::complex<double>;
  using complex_matrix_type = uni20::DenseMatrix<scalar_type>;
  using async_complex_matrix_type = uni20::async::Async<complex_matrix_type>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_complex_matrix_type output =
      make_matrix<complex_matrix_type>(2, 2, {scalar_type{1}, scalar_type{}, scalar_type{}, scalar_type{1}});
  async_complex_matrix_type rhs =
      make_matrix<complex_matrix_type>(2, 2, {scalar_type{1}, scalar_type{}, scalar_type{}, scalar_type{1}});
  auto output_alias = uni20::async::conj(output);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::linalg::assign_product(output, output_alias, rhs), std::runtime_error);

  auto const& unchanged = output.get_wait(scheduler);
  EXPECT_EQ((unchanged[0, 0]), scalar_type(1));
  EXPECT_EQ((unchanged[1, 1]), scalar_type(1));
}

TEST(AsyncMatrixProductTest, ShapeFailurePropagatesToOutputEpoch)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type lhs = make_matrix<matrix_type>(2, 3, {1, 2, 3, 4, 5, 6});
  async_matrix_type rhs = make_matrix<matrix_type>(4, 2, {1, 2, 3, 4, 5, 6, 7, 8});
  async_matrix_type output;
  ErrorModeGuard const error_mode;

  uni20::linalg::assign_product(output, lhs, rhs);
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), std::runtime_error);
}

TEST(AsyncMatrixProductTest, AddProductRequiresConstructedOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type lhs = make_matrix<matrix_type>(1, 1, {2});
  async_matrix_type rhs = make_matrix<matrix_type>(1, 1, {3});
  async_matrix_type output;

  uni20::linalg::add_product(output, lhs, rhs);
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), uni20::async::buffer_write_uninitialized);
}

TEST(AsyncMatrixProductTest, GemmRequiresConstructedOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_matrix_type lhs = make_matrix<matrix_type>(1, 1, {2});
  async_matrix_type rhs = make_matrix<matrix_type>(1, 1, {3});
  async_matrix_type output;

  uni20::linalg::gemm(output, 1.0, lhs, rhs, 0.0);
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), uni20::async::buffer_write_uninitialized);
}
