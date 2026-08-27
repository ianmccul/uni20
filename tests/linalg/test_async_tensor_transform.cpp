#include <uni20/async/debug_scheduler.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/generated.hpp>
#include <uni20/tensor/tensor.hpp>

#include "deferred_host_tensor.hpp"

#include <gtest/gtest.h>

#include <concepts>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace
{

using vector_type = uni20::Tensor<double, 1>;
using async_vector_type = uni20::async::Async<vector_type>;
using deferred_vector_type = uni20::test::DeferredHostTensor<double, 1>;
using async_deferred_vector_type = uni20::async::Async<deferred_vector_type>;

template <class Output, class Function, class... Inputs>
concept CanAssignTransform = requires(Output& output, Function&& function, Inputs const&... inputs) {
  uni20::assign_transform(output, std::forward<Function>(function), inputs...);
};

template <class Output, class Function, class... Inputs>
concept CanTransformInplace = requires(Output& output, Function&& function, Inputs const&... inputs) {
  uni20::transform_inplace(output, std::forward<Function>(function), inputs...);
};

static_assert(CanAssignTransform<async_vector_type, std::plus<>, async_vector_type, async_vector_type>);
static_assert(CanAssignTransform<async_vector_type, uni20::linalg::constant<double>>);
static_assert(CanTransformInplace<async_vector_type, std::plus<>, async_vector_type>);
static_assert(CanTransformInplace<async_vector_type, std::negate<>>);
static_assert(!CanAssignTransform<vector_type, std::plus<>, async_vector_type, async_vector_type>);
static_assert(!CanAssignTransform<async_vector_type, std::plus<>, vector_type, async_vector_type>);
static_assert(!CanTransformInplace<vector_type, std::plus<>, async_vector_type>);
static_assert(!CanTransformInplace<async_vector_type, std::plus<>, vector_type>);

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

struct move_only_scale
{
    std::unique_ptr<double> scale;

    explicit move_only_scale(double value) : scale(std::make_unique<double>(value)) {}
    move_only_scale(move_only_scale&&) = default;
    move_only_scale& operator=(move_only_scale&&) = default;
    move_only_scale(move_only_scale const&) = delete;
    move_only_scale& operator=(move_only_scale const&) = delete;

    double operator()(double value) const { return *scale * value; }
};

template <class T> uni20::async::AsyncTask co_publish(uni20::async::WriteBuffer<T> output, T value)
{
  co_await output = std::move(value);
  co_return;
}

vector_type make_vector(std::initializer_list<double> values)
{
  vector_type result(static_cast<uni20::index_type>(values.size()));
  uni20::index_type index = 0;
  for (double value : values)
    result[index++] = value;
  return result;
}

deferred_vector_type make_deferred_vector(std::initializer_list<double> values)
{
  deferred_vector_type result(static_cast<uni20::index_type>(values.size()));
  auto lease = uni20::test::acquire_host_write_access_sync(result);
  uni20::index_type index = 0;
  for (double value : values)
    lease.mdspan()[index++] = value;
  return result;
}

async_vector_type schedule_transform_from_local_state()
{
  async_vector_type lhs;
  async_vector_type rhs;
  async_vector_type output;

  uni20::async::schedule(co_publish(lhs.write(), make_vector({1.0, 2.0, 3.0})));
  uni20::async::schedule(co_publish(rhs.write(), make_vector({10.0, 20.0, 30.0})));
  uni20::assign_transform(
      output, [offset = std::make_unique<double>(0.5)](double left, double right) { return left + right + *offset; },
      lhs, rhs);
  return output;
}

} // namespace

TEST(AsyncTensorTransformTest, OverwriteConstructsOutputAndRetainsPendingInputsAndCallable)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);

  auto output = schedule_transform_from_local_state();
  auto const& result = output.get_wait(scheduler);

  ASSERT_EQ(result.extent(0), 3);
  EXPECT_DOUBLE_EQ(result[0], 11.5);
  EXPECT_DOUBLE_EQ(result[1], 22.5);
  EXPECT_DOUBLE_EQ(result[2], 33.5);
}

TEST(AsyncTensorTransformTest, FillOverwritesWithoutReadingExistingValues)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_vector_type values = make_vector({std::numeric_limits<double>::quiet_NaN(), 2.0, -3.0});

  uni20::fill(values, 0.0);

  auto const& result = values.get_wait(scheduler);
  for (uni20::index_type index = 0; index < result.extent(0); ++index)
    EXPECT_DOUBLE_EQ(result[index], 0.0);
}

TEST(AsyncTensorTransformTest, ExplicitSelectorResizesExistingOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_vector_type input = make_vector({1.0, 2.0, 3.0});
  async_vector_type output = make_vector({0.0});

  uni20::assign_transform(uni20::linalg::CpuReferenceBackend{}, output, move_only_scale{4.0}, input);

  auto const& result = output.get_wait(scheduler);
  ASSERT_EQ(result.extent(0), 3);
  EXPECT_DOUBLE_EQ(result[0], 4.0);
  EXPECT_DOUBLE_EQ(result[1], 8.0);
  EXPECT_DOUBLE_EQ(result[2], 12.0);
}

TEST(AsyncTensorTransformTest, UpdateReadsExistingOutputThroughSingleWriter)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_vector_type rhs = make_vector({1.0, 2.0, 3.0});
  async_vector_type output = make_vector({10.0, 20.0, 30.0});

  uni20::transform_inplace(output, [](double current, double value) { return current + 2.0 * value; }, rhs);
  uni20::transform_inplace(output, [](double current) { return -current; });

  auto const& result = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ(result[0], -12.0);
  EXPECT_DOUBLE_EQ(result[1], -24.0);
  EXPECT_DOUBLE_EQ(result[2], -36.0);
}

TEST(AsyncTensorTransformTest, GeneratedAsyncInputUsesBackendNeutralStorage)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  auto generated_value = uni20::ones<double>(3);
  uni20::async::Async<decltype(generated_value)> bias = std::move(generated_value);
  async_vector_type output = make_vector({1.0, 2.0, 3.0});

  uni20::transform_inplace(output, std::plus<>{}, bias);

  auto const& result = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ(result[0], 2.0);
  EXPECT_DOUBLE_EQ(result[1], 3.0);
  EXPECT_DOUBLE_EQ(result[2], 4.0);
}

TEST(AsyncTensorTransformTest, DeferredInputsAndOutputUseTensorLevelDispatch)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_deferred_vector_type lhs = make_deferred_vector({1.0, 2.0, 3.0});
  async_deferred_vector_type rhs = make_deferred_vector({10.0, 20.0, 30.0});
  async_deferred_vector_type output = deferred_vector_type(3);

  uni20::assign_transform(output, std::plus<>{}, lhs, rhs);
  uni20::transform_inplace(output, [](double value) { return 2.0 * value; });

  auto lease = uni20::test::acquire_host_read_access_sync(output.get_wait(scheduler));
  EXPECT_DOUBLE_EQ(lease.mdspan()[0], 22.0);
  EXPECT_DOUBLE_EQ(lease.mdspan()[1], 44.0);
  EXPECT_DOUBLE_EQ(lease.mdspan()[2], 66.0);
}

TEST(AsyncTensorTransformTest, DeferredReshapeAliasPreservesDescriptorAndWritesParent)
{
  using deferred_matrix_type = uni20::test::DeferredHostTensor<double, 2>;

  deferred_matrix_type value(2, 2);
  {
    auto lease = uni20::test::acquire_host_write_access_sync(value);
    lease.mdspan()[0, 0] = 1.0;
    lease.mdspan()[1, 0] = 2.0;
    lease.mdspan()[0, 1] = 3.0;
    lease.mdspan()[1, 1] = 4.0;
  }

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<deferred_matrix_type> parent = std::move(value);
  auto flattened = uni20::async::reshape_view(parent, 4);
  using flattened_type = uni20::async::async_value_t<decltype(flattened)>;
  static_assert(uni20::MutableTensorView<flattened_type>);
  static_assert(!uni20::ImmediateTensorView<flattened_type>);

  uni20::transform_inplace(flattened, [](double value) { return -value; });

  auto lease = uni20::test::acquire_host_read_access_sync(parent.get_wait(scheduler));
  EXPECT_DOUBLE_EQ((lease.mdspan()[0, 0]), -1.0);
  EXPECT_DOUBLE_EQ((lease.mdspan()[1, 0]), -2.0);
  EXPECT_DOUBLE_EQ((lease.mdspan()[0, 1]), -3.0);
  EXPECT_DOUBLE_EQ((lease.mdspan()[1, 1]), -4.0);
}

TEST(AsyncTensorTransformTest, MutableAsyncAliasUpdatesParentStorage)
{
  using matrix_type = uni20::Tensor<double, 2>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  matrix_type matrix(2, 2);
  matrix[0, 0] = 1.0;
  matrix[0, 1] = 2.0;
  matrix[1, 0] = 3.0;
  matrix[1, 1] = 4.0;
  uni20::async::Async<matrix_type> parent = std::move(matrix);
  auto flattened = uni20::async::reshape_view(parent, 4);

  uni20::transform_inplace(flattened, [](double value) { return 3.0 * value; });

  auto const& result = parent.get_wait(scheduler);
  EXPECT_DOUBLE_EQ((result[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 6.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 9.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 12.0);
}

TEST(AsyncTensorTransformTest, MutableAsyncAliasAcceptsFixedShapeOverwrite)
{
  using matrix_type = uni20::Tensor<double, 2>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<matrix_type> parent = matrix_type(2, 2);
  auto flattened = uni20::async::reshape_view(parent, 4);
  async_vector_type input = make_vector({5.0, 6.0, 7.0, 8.0});

  uni20::assign_transform(flattened, [](double value) { return 2.0 * value; }, input);

  auto const& result = parent.get_wait(scheduler);
  EXPECT_DOUBLE_EQ((result[0, 0]), 10.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 12.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 14.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 16.0);
}

TEST(AsyncTensorTransformTest, ReadOnlyInputsMayShareAnEpochQueue)
{
  using scalar_type = uni20::complex<double>;
  using complex_vector_type = uni20::Tensor<scalar_type, 1>;
  using async_complex_vector_type = uni20::async::Async<complex_vector_type>;

  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  complex_vector_type input_value(2);
  input_value[0] = scalar_type{1.0, 2.0};
  input_value[1] = scalar_type{-3.0, 4.0};
  async_complex_vector_type input = std::move(input_value);
  async_complex_vector_type output;
  auto conjugated = uni20::async::conj(input);

  uni20::assign_transform(output, std::plus<>{}, input, conjugated);

  auto const& result = output.get_wait(scheduler);
  EXPECT_EQ(result[0], scalar_type(2.0));
  EXPECT_EQ(result[1], scalar_type(-6.0));
}

TEST(AsyncTensorTransformTest, RejectsOutputInputQueueAliasBeforeBufferEnrollment)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_vector_type output = make_vector({1.0, 2.0});
  async_vector_type rhs = make_vector({3.0, 4.0});
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::assign_transform(output, std::plus<>{}, output, rhs), std::runtime_error);
  EXPECT_THROW(uni20::transform_inplace(output, std::plus<>{}, output), std::runtime_error);

  auto const& unchanged = output.get_wait(scheduler);
  EXPECT_DOUBLE_EQ(unchanged[0], 1.0);
  EXPECT_DOUBLE_EQ(unchanged[1], 2.0);
}

TEST(AsyncTensorTransformTest, ShapeFailurePropagatesToOutputEpoch)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_vector_type lhs = make_vector({1.0, 2.0});
  async_vector_type rhs = make_vector({3.0, 4.0, 5.0});
  async_vector_type output;
  ErrorModeGuard const error_mode;

  uni20::assign_transform(output, std::plus<>{}, lhs, rhs);
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), std::runtime_error);
}

TEST(AsyncTensorTransformTest, CallableFailurePropagatesToOutputEpoch)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_vector_type input = make_vector({1.0});
  async_vector_type output;

  uni20::assign_transform(output, [](double) -> double { throw std::runtime_error("transform failure"); }, input);
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), std::runtime_error);
}

TEST(AsyncTensorTransformTest, UpdateRequiresConstructedOutput)
{
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_vector_type input = make_vector({1.0});
  async_vector_type output;

  uni20::transform_inplace(output, std::plus<>{}, input);
  scheduler.run_all();

  EXPECT_THROW((void)output.get_wait(scheduler), uni20::async::buffer_write_uninitialized);
}
