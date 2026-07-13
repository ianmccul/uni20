#include <concepts>
#include <gtest/gtest.h>
#include <type_traits>
#include <uni20/async/async_task.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/types.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/tensor.hpp>
#include <utility>

namespace
{

using matrix_type = uni20::DenseMatrix<uni20::complex<double>>;
using conjugated_view_type = uni20::ConjugatedTensorView<matrix_type>;
using async_view_type = uni20::async::Async<conjugated_view_type>;
using double_conjugated_view_type = uni20::ConjugatedTensorView<conjugated_view_type>;
using async_double_conjugated_view_type = uni20::async::Async<double_conjugated_view_type>;
using real_matrix_type = uni20::DenseMatrix<double>;
using const_real_view_type = uni20::ConstTensorView<real_matrix_type>;
using async_const_real_view_type = uni20::async::Async<const_real_view_type>;

static_assert(uni20::TensorView<conjugated_view_type>);
static_assert(!uni20::MutableTensorView<conjugated_view_type>);
static_assert(!std::default_initializable<async_view_type>);
static_assert(!std::constructible_from<async_view_type, conjugated_view_type>);
static_assert(std::same_as<decltype(uni20::async::conj(std::declval<uni20::async::Async<matrix_type> const&>())),
                           async_view_type>);
static_assert(std::same_as<decltype(uni20::async::conj(std::declval<async_view_type const&>())),
                           async_double_conjugated_view_type>);
static_assert(std::same_as<decltype(uni20::async::conj(std::declval<uni20::async::Async<real_matrix_type> const&>())),
                           async_const_real_view_type>);
static_assert(uni20::TensorView<const_real_view_type>);
static_assert(!uni20::MutableTensorView<const_real_view_type>);

matrix_type make_matrix()
{
  matrix_type matrix(2, 2);
  auto span = matrix.mdspan();
  span[0, 0] = {1.0, 2.0};
  span[0, 1] = {3.0, -4.0};
  span[1, 0] = {-5.0, 6.0};
  span[1, 1] = {7.0, 8.0};
  return matrix;
}

} // namespace

TEST(TensorAsyncTest, ConjugatedAliasSharesOwnerAndQueue)
{
  uni20::async::Async<matrix_type> parent(make_matrix());
  auto alias = uni20::async::conj(parent);

  EXPECT_EQ(parent.storage().use_count(), 2);
  EXPECT_EQ(alias.storage().use_count(), 1);
  EXPECT_EQ(&alias.queue(), &parent.queue());

  auto copy = alias;
  EXPECT_EQ(parent.storage().use_count(), 2);
  EXPECT_EQ(alias.storage().use_count(), 2);
  EXPECT_EQ(copy.storage().control_address(), alias.storage().control_address());
  EXPECT_EQ(&copy.queue(), &parent.queue());
}

TEST(TensorAsyncTest, ConjugatedAliasOutlivesParentHandle)
{
  auto make_alias = [] {
    uni20::async::Async<matrix_type> parent(make_matrix());
    return uni20::async::conj(parent);
  };

  auto alias = make_alias();
  uni20::async::DebugScheduler sched;
  sched.schedule([](uni20::async::ReadBuffer<conjugated_view_type> reader) static -> uni20::async::AsyncTask {
    auto const& view = co_await reader;
    auto span = view.mdspan();
    auto const value_00 = span[0, 0];
    auto const value_01 = span[0, 1];
    auto const value_10 = span[1, 0];
    auto const value_11 = span[1, 1];
    EXPECT_EQ(value_00, uni20::complex<double>(1.0, -2.0));
    EXPECT_EQ(value_01, uni20::complex<double>(3.0, 4.0));
    EXPECT_EQ(value_10, uni20::complex<double>(-5.0, -6.0));
    EXPECT_EQ(value_11, uni20::complex<double>(7.0, -8.0));
    co_return;
  }(alias.read()));

  sched.run_all();
}

TEST(TensorAsyncTest, DoubleConjugationRetainsAliasChainAndCancelsTransform)
{
  auto make_alias = [] {
    uni20::async::Async<matrix_type> parent(make_matrix());
    return uni20::async::conj(uni20::async::conj(parent));
  };

  auto alias = make_alias();
  uni20::async::DebugScheduler sched;
  sched.schedule([](uni20::async::ReadBuffer<double_conjugated_view_type> reader) static -> uni20::async::AsyncTask {
    auto const values = (co_await reader).mdspan();
    auto const value_00 = values[0, 0];
    auto const value_01 = values[0, 1];
    auto const value_10 = values[1, 0];
    auto const value_11 = values[1, 1];
    EXPECT_EQ(value_00, uni20::complex<double>(1.0, 2.0));
    EXPECT_EQ(value_01, uni20::complex<double>(3.0, -4.0));
    EXPECT_EQ(value_10, uni20::complex<double>(-5.0, 6.0));
    EXPECT_EQ(value_11, uni20::complex<double>(7.0, 8.0));
    co_return;
  }(alias.read()));

  sched.run_all();
}

TEST(TensorAsyncTest, PendingParentWriterPrecedesConjugatedAliasReader)
{
  uni20::async::Async<matrix_type> parent;
  auto alias = uni20::async::conj(parent);
  uni20::async::DebugScheduler sched;

  sched.schedule([](uni20::async::WriteBuffer<matrix_type> writer) static -> uni20::async::AsyncTask {
    auto& matrix = (co_await writer).emplace(2, 2);
    auto span = matrix.mdspan();
    span[0, 0] = {2.0, 3.0};
    span[0, 1] = {0.0, 0.0};
    span[1, 0] = {0.0, 0.0};
    span[1, 1] = {-4.0, 5.0};
    co_return;
  }(parent.write()));

  sched.schedule([](uni20::async::ReadBuffer<conjugated_view_type> reader) static -> uni20::async::AsyncTask {
    auto const& view = co_await reader;
    auto span = view.mdspan();
    auto const value_00 = span[0, 0];
    auto const value_11 = span[1, 1];
    EXPECT_EQ(value_00, uni20::complex<double>(2.0, -3.0));
    EXPECT_EQ(value_11, uni20::complex<double>(-4.0, -5.0));
    co_return;
  }(alias.read()));

  sched.run_all();
}

TEST(TensorAsyncTest, RealConjugationProducesReadOnlyIdentityAlias)
{
  real_matrix_type matrix(1, 2);
  auto span = matrix.mdspan();
  span[0, 0] = 2.5;
  span[0, 1] = -3.0;

  uni20::async::Async<real_matrix_type> parent(std::move(matrix));
  auto alias = uni20::async::conj(parent);
  EXPECT_EQ(&alias.queue(), &parent.queue());

  uni20::async::DebugScheduler sched;
  sched.schedule([](uni20::async::ReadBuffer<const_real_view_type> reader) static -> uni20::async::AsyncTask {
    auto const values = (co_await reader).mdspan();
    auto const value_0 = values[0, 0];
    auto const value_1 = values[0, 1];
    EXPECT_EQ(value_0, 2.5);
    EXPECT_EQ(value_1, -3.0);
    co_return;
  }(alias.read()));

  sched.run_all();
}
