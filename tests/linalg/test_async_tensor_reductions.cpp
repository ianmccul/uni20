#include <uni20/async/debug_scheduler.hpp>
#include <uni20/common/gtest.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/generated.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace
{

using tensor_type = uni20::Tensor<double, 3>;
using async_tensor_type = uni20::async::Async<tensor_type>;
using scalar_tensor_type = uni20::ScalarTensor<double>;
using async_scalar_tensor_type = uni20::async::Async<scalar_tensor_type>;
using deferred_tensor_type = uni20::test::DeferredHostTensor<double, 3>;
using deferred_scalar_type = uni20::test::DeferredHostTensor<double, 0>;
using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;

template <class Output, class Input>
concept CanSumInto = requires(Output& output, Input const& input) { uni20::sum(output, input); };

template <class Input>
concept CanReturnSum = requires(Input const& input) { uni20::sum(input); };

static_assert(CanSumInto<async_scalar_tensor_type, async_tensor_type>);
static_assert(CanReturnSum<async_tensor_type>);
static_assert(!CanSumInto<scalar_tensor_type, async_tensor_type>);
static_assert(!CanSumInto<async_scalar_tensor_type, tensor_type>);
static_assert(!CanReturnSum<tensor_type const*>);

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

template <class T> uni20::async::AsyncTask co_publish(uni20::async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
  co_return;
}

tensor_type make_input()
{
  tensor_type input(2, 3, 4);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type k = 0; k < 4; ++k)
        input[i, j, k] = static_cast<double>(100 * i + 10 * j + k);
  return input;
}

deferred_tensor_type make_deferred_input()
{
  deferred_tensor_type input(2, 3, 4);
  auto lease = uni20::test::acquire_host_write_access_sync(input);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type k = 0; k < 4; ++k)
        lease.mdspan()[i, j, k] = static_cast<double>(100 * i + 10 * j + k);
  return input;
}

} // namespace

TEST(AsyncTensorReductionTest, FullSumAndHostSumAwaitPendingInput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_tensor_type input;

  uni20::async::schedule(co_publish(input.write(), make_input()));
  auto tensor_result = uni20::sum(input);
  auto host_result = uni20::sum_host(input);

  static_assert(std::same_as<decltype(tensor_result), async_scalar_tensor_type>);
  static_assert(std::same_as<decltype(host_result), uni20::async::Async<double>>);
  EXPECT_DOUBLE_EQ(tensor_result.get_wait(scheduler)[], 1476.0);
  EXPECT_DOUBLE_EQ(host_result.get_wait(scheduler), 1476.0);
}

TEST(AsyncTensorReductionTest, InnerProductAndNormReturnTensorAndHostResults)
{
  using vector_type = uni20::Tensor<double, 1>;
  vector_type lhs_value(2);
  lhs_value[0] = 3.0;
  lhs_value[1] = 4.0;
  vector_type rhs_value(2);
  rhs_value[0] = 2.0;
  rhs_value[1] = -1.0;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<vector_type> lhs = std::move(lhs_value);
  uni20::async::Async<vector_type> rhs = std::move(rhs_value);

  auto inner = uni20::inner_product(lhs, rhs);
  auto inner_host = uni20::inner_product_host(lhs, rhs);
  auto magnitude = uni20::norm(lhs);
  auto magnitude_host = uni20::norm_host(lhs);

  static_assert(std::same_as<decltype(inner), async_scalar_tensor_type>);
  static_assert(std::same_as<decltype(inner_host), uni20::async::Async<double>>);
  static_assert(std::same_as<decltype(magnitude), async_scalar_tensor_type>);
  static_assert(std::same_as<decltype(magnitude_host), uni20::async::Async<double>>);
  EXPECT_DOUBLE_EQ(inner.get_wait(scheduler)[], 2.0);
  EXPECT_DOUBLE_EQ(inner_host.get_wait(scheduler), 2.0);
  EXPECT_DOUBLE_EQ(magnitude.get_wait(scheduler)[], 5.0);
  EXPECT_DOUBLE_EQ(magnitude_host.get_wait(scheduler), 5.0);
}

TEST(AsyncTensorReductionTest, InnerProductAndNormConstructExplicitScalarOutputs)
{
  using vector_type = uni20::Tensor<double, 1>;
  vector_type lhs_value(3);
  lhs_value[0] = 1.0;
  lhs_value[1] = 2.0;
  lhs_value[2] = 3.0;
  vector_type rhs_value(3);
  rhs_value[0] = 4.0;
  rhs_value[1] = 5.0;
  rhs_value[2] = 6.0;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<vector_type> lhs = std::move(lhs_value);
  uni20::async::Async<vector_type> rhs = std::move(rhs_value);
  async_scalar_tensor_type inner_output;
  async_scalar_tensor_type norm_output;

  uni20::inner_product(inner_output, lhs, rhs);
  uni20::norm(norm_output, lhs);

  EXPECT_DOUBLE_EQ(inner_output.get_wait(scheduler)[], 32.0);
  EXPECT_NEAR(norm_output.get_wait(scheduler)[], std::sqrt(14.0), 1.0e-14);
}

TEST(AsyncTensorReductionTest, ComplexInnerProductRetainsConjugateLinearLeftSemantics)
{
  using scalar_type = uni20::complex<double>;
  using vector_type = uni20::Tensor<scalar_type, 1>;
  vector_type lhs_value(2);
  lhs_value[0] = scalar_type{1.0, 2.0};
  lhs_value[1] = scalar_type{-1.0, 1.0};
  vector_type rhs_value(2);
  rhs_value[0] = scalar_type{3.0, -1.0};
  rhs_value[1] = scalar_type{2.0, 4.0};

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<vector_type> lhs = std::move(lhs_value);
  uni20::async::Async<vector_type> rhs = std::move(rhs_value);

  auto result = uni20::inner_product_host(lhs, rhs);

  EXPECT_FLOATING_EQ(result.get_wait(scheduler), (scalar_type{3.0, -13.0}));
}

TEST(AsyncTensorReductionTest, PartialSumPreservesLayoutAndNegativeAxisSemantics)
{
  using row_major_type = uni20::RowMajorTensor<double, 3>;
  row_major_type value(2, 3, 4);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      for (uni20::index_type k = 0; k < 4; ++k)
        value[i, j, k] = static_cast<double>(100 * i + 10 * j + k);

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<row_major_type> input = std::move(value);
  auto result = uni20::sum(input, -1);

  using expected_result_type = uni20::async::Async<uni20::RowMajorTensor<double, 2>>;
  static_assert(std::same_as<decltype(result), expected_result_type>);
  auto const& reduced = result.get_wait(scheduler);
  ASSERT_EQ(reduced.extent(0), 2);
  ASSERT_EQ(reduced.extent(1), 3);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type j = 0; j < 3; ++j)
      EXPECT_DOUBLE_EQ((reduced[i, j]), static_cast<double>(400 * i + 40 * j + 6));
}

TEST(AsyncTensorReductionTest, ExplicitOutputConstructsOrResizesAfterInputIsReady)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_tensor_type input;
  async_scalar_tensor_type scalar_output;
  uni20::async::Async<uni20::RowMajorTensor<double, 2>> partial_output = uni20::RowMajorTensor<double, 2>(1, 1);

  uni20::async::schedule(co_publish(input.write(), make_input()));
  uni20::sum(uni20::linalg::CpuReferenceBackend{}, scalar_output, input);
  uni20::sum(partial_output, input, 1);

  EXPECT_DOUBLE_EQ(scalar_output.get_wait(scheduler)[], 1476.0);
  auto const& partial = partial_output.get_wait(scheduler);
  ASSERT_EQ(partial.extent(0), 2);
  ASSERT_EQ(partial.extent(1), 4);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 4; ++k)
      EXPECT_DOUBLE_EQ((partial[i, k]), static_cast<double>(300 * i + 30 + 3 * k));
}

TEST(AsyncTensorReductionTest, GeneratedInputUsesStorageFallbackAndAccessorSemantics)
{
  auto generated = uni20::ones<double>(2, 3, 4);
  using generated_type = decltype(generated);

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<generated_type> input = std::move(generated);
  auto result = uni20::sum(input, 1);

  static_assert(std::same_as<decltype(result), uni20::async::Async<uni20::Tensor<double, 2>>>);
  auto const& reduced = result.get_wait(scheduler);
  ASSERT_EQ(reduced.extent(0), 2);
  ASSERT_EQ(reduced.extent(1), 4);
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 4; ++k)
      EXPECT_DOUBLE_EQ((reduced[i, k]), 3.0);
}

TEST(AsyncTensorReductionTest, DeferredInputAndOutputsUseMdspecDispatch)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<deferred_tensor_type> input = make_deferred_input();
  uni20::async::Async<deferred_scalar_type> scalar_output =
      deferred_scalar_type(typename deferred_scalar_type::extents_type{});
  uni20::async::Async<deferred_matrix_type> partial_output = deferred_matrix_type(2, 4);

  uni20::sum(scalar_output, input);
  uni20::sum(uni20::linalg::CpuReferenceBackend{}, partial_output, input, 1);
  auto host_output = uni20::sum_host(input);

  auto scalar_lease = uni20::test::acquire_host_read_access_sync(scalar_output.get_wait(scheduler));
  EXPECT_DOUBLE_EQ(scalar_lease.mdspan()[], 1476.0);
  EXPECT_DOUBLE_EQ(host_output.get_wait(scheduler), 1476.0);

  auto partial_lease = uni20::test::acquire_host_read_access_sync(partial_output.get_wait(scheduler));
  for (uni20::index_type i = 0; i < 2; ++i)
    for (uni20::index_type k = 0; k < 4; ++k)
      EXPECT_DOUBLE_EQ((partial_lease.mdspan()[i, k]), static_cast<double>(300 * i + 30 + 3 * k));
}

TEST(AsyncTensorReductionTest, MutableAliasOutputWritesThroughItsParent)
{
  using matrix_type = uni20::Tensor<double, 2>;
  using vector_type = uni20::Tensor<double, 1>;

  matrix_type input_value(2, 3);
  input_value[0, 0] = 1.0;
  input_value[0, 1] = 2.0;
  input_value[0, 2] = 3.0;
  input_value[1, 0] = 4.0;
  input_value[1, 1] = 5.0;
  input_value[1, 2] = 6.0;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<matrix_type> input = std::move(input_value);
  uni20::async::Async<vector_type> parent = vector_type(2);
  auto output = uni20::async::reshape_view(parent, 2);

  uni20::sum(output, input, 1);

  auto const& result = parent.get_wait(scheduler);
  EXPECT_DOUBLE_EQ(result[0], 6.0);
  EXPECT_DOUBLE_EQ(result[1], 15.0);
}

TEST(AsyncTensorReductionTest, ConjugatingInputAccessorIsObserved)
{
  using scalar_type = uni20::complex<double>;
  using matrix_type = uni20::Tensor<scalar_type, 2>;

  matrix_type value(2, 2);
  value[0, 0] = scalar_type{1.0, 2.0};
  value[0, 1] = scalar_type{3.0, -4.0};
  value[1, 0] = scalar_type{-2.0, 1.0};
  value[1, 1] = scalar_type{5.0, 3.0};

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<matrix_type> input = std::move(value);
  auto conjugated = uni20::async::conj(input);
  auto result = uni20::sum(conjugated, 0);

  auto const& reduced = result.get_wait(scheduler);
  EXPECT_FLOATING_EQ(reduced[0], (scalar_type{-1.0, -3.0}));
  EXPECT_FLOATING_EQ(reduced[1], (scalar_type{8.0, 1.0}));
}

TEST(AsyncTensorReductionTest, RejectsOutputInputQueueAliasBeforeBufferEnrollment)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  scalar_tensor_type initial;
  initial[] = 0.0;
  async_scalar_tensor_type value = std::move(initial);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::sum(value, value), std::runtime_error);
  EXPECT_DOUBLE_EQ(value.get_wait(scheduler)[], 0.0);
}

TEST(AsyncTensorReductionTest, AxisErrorsAreReportedBeforeScheduling)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_tensor_type input = make_input();
  ErrorModeGuard const error_mode;

  EXPECT_THROW((void)uni20::sum(input, 1, 1), std::runtime_error);
  EXPECT_THROW((void)uni20::sum(input, 3), std::runtime_error);
  EXPECT_NO_THROW((void)input.get_wait(scheduler));
}

TEST(AsyncTensorReductionTest, FixedAliasShapeFailurePropagatesToOutputEpoch)
{
  using matrix_type = uni20::Tensor<double, 2>;
  using vector_type = uni20::Tensor<double, 1>;

  matrix_type input_value(2, 3);
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<matrix_type> input = std::move(input_value);
  uni20::async::Async<vector_type> parent = vector_type(3);
  auto output = uni20::async::reshape_view(parent, 3);
  ErrorModeGuard const error_mode;

  uni20::sum(output, input, 1);
  scheduler.run_all();

  EXPECT_THROW((void)parent.get_wait(scheduler), std::runtime_error);
}
