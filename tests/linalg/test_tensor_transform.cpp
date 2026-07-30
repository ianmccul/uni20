#include <uni20/linalg/linalg.hpp>
#include <uni20/tensor/generated.hpp>
#include <uni20/tensor/tensor.hpp>
#include <uni20/tensor/transform.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace
{

struct affine_transform
{
    double scale;
    double offset;

    double operator()(double value) const { return scale * value + offset; }
};

struct move_only_scale
{
    std::unique_ptr<double> scale;

    move_only_scale(double value) : scale(std::make_unique<double>(value)) {}
    move_only_scale(move_only_scale&&) = default;
    move_only_scale& operator=(move_only_scale&&) = default;
    move_only_scale(move_only_scale const&) = delete;
    move_only_scale& operator=(move_only_scale const&) = delete;

    double operator()(double value) const { return *scale * value; }
};

TEST(TensorTransformTest, CpuReferenceDispatchSupportsVariadicMixedStridedMappings)
{
  using extents_type = stdex::dextents<uni20::index_type, 2>;

  std::vector<double> output_storage(12, 0.0);
  std::vector<double> lhs_storage(12, 0.0);
  std::vector<double> rhs_storage(24, 0.0);
  std::vector<double> bias_storage(12, 0.0);

  stdex::mdspan<double, extents_type, stdex::layout_right> output(output_storage.data(), 3, 4);
  stdex::mdspan<double, extents_type, stdex::layout_left> lhs(lhs_storage.data(), 3, 4);
  auto rhs_mapping =
      stdex::layout_stride::mapping<extents_type>(extents_type{3, 4}, std::array<uni20::index_type, 2>{8, 2});
  stdex::mdspan<double, extents_type, stdex::layout_stride> rhs(rhs_storage.data(), rhs_mapping);
  stdex::mdspan<double, extents_type, stdex::layout_right> bias(bias_storage.data(), 3, 4);

  for (uni20::index_type row = 0; row < 3; ++row)
    for (uni20::index_type column = 0; column < 4; ++column)
    {
      lhs[row, column] = 10.0 * row + column;
      rhs[row, column] = 100.0 + row - column;
      bias[row, column] = 0.5 * column;
    }

  uni20::assign_transform(
      uni20::linalg::CpuReferenceBackend{}, output,
      [](double left, double right, double bias_value) { return left - right + bias_value; }, lhs, rhs, bias);
  uni20::transform_inplace(
      uni20::linalg::CpuReferenceBackend{}, output,
      [](double current, double bias_value) { return current + bias_value; }, bias);

  for (uni20::index_type row = 0; row < 3; ++row)
    for (uni20::index_type column = 0; column < 4; ++column)
      EXPECT_DOUBLE_EQ((output[row, column]), (lhs[row, column] - rhs[row, column] + 2.0 * bias[row, column]));
}

TEST(TensorTransformTest, TensorOverwriteResizesAndReadsGeneratedInput)
{
  uni20::DenseMatrix<double, uni20::RowMajor> input(2, 3);
  uni20::DenseMatrix<double> output;
  auto bias = uni20::ones<double>(2, 3);

  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      input[row, column] = 10.0 * row + column;

  uni20::assign_transform(output, affine_transform{.scale = 2.0, .offset = -1.0}, input);
  uni20::transform_inplace(output, std::plus<>{}, bias);

  EXPECT_EQ(output.rows(), 2);
  EXPECT_EQ(output.cols(), 3);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((output[row, column]), (2.0 * input[row, column]));
}

TEST(TensorTransformTest, UnaryTensorUpdateUsesExistingOutputValue)
{
  uni20::Tensor<double, 1> values(4);
  for (uni20::index_type index = 0; index < 4; ++index)
    values[index] = static_cast<double>(index + 1);

  uni20::transform_inplace(values, [](double value) { return value * value; });

  EXPECT_DOUBLE_EQ(values[0], 1.0);
  EXPECT_DOUBLE_EQ(values[1], 4.0);
  EXPECT_DOUBLE_EQ(values[2], 9.0);
  EXPECT_DOUBLE_EQ(values[3], 16.0);
}

TEST(TensorTransformTest, DeferredTensorsResolveAllLeasesAtTheCpuBoundary)
{
  uni20::test::DeferredHostTensor<double, 1> lhs(3);
  uni20::test::DeferredHostTensor<double, 1> rhs(3);
  uni20::test::DeferredHostTensor<double, 1> output(3);
  lhs.storage() = {1.0, 2.0, 3.0};
  rhs.storage() = {4.0, 5.0, 6.0};

  uni20::assign_transform(output, std::plus<>{}, lhs, rhs);
  uni20::transform_inplace(output, [](double value) { return 2.0 * value; });

  EXPECT_EQ(output.storage(), (std::vector<double>{10.0, 14.0, 18.0}));
}

TEST(TensorTransformTest, DispatchRetainsMoveOnlyCallableState)
{
  uni20::Tensor<double, 1> input(3);
  uni20::Tensor<double, 1> output;
  input[0] = 1.0;
  input[1] = 2.0;
  input[2] = 3.0;

  uni20::assign_transform(output, move_only_scale{4.0}, input);

  EXPECT_DOUBLE_EQ(output[0], 4.0);
  EXPECT_DOUBLE_EQ(output[1], 8.0);
  EXPECT_DOUBLE_EQ(output[2], 12.0);
}

TEST(TensorTransformTest, TypeProbeRequiresConstCallable)
{
  uni20::Tensor<double, 1> input(2);
  uni20::Tensor<double, 1> output(2);
  auto input_span = std::as_const(input).mdspan();
  auto output_span = output.mdspan();

  auto const_callable = [](double value) { return value + 1.0; };
  auto mutable_callable = [count = 0](double value) mutable { return value + count++; };
  auto mutating_input_callable = [](double& value) { return ++value; };

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{},
                                                 uni20::linalg::transform_op{const_callable}, output_span, input_span),
            uni20::linalg::KernelTypeAcceptance::yes);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{},
                                                 uni20::linalg::transform_op{mutable_callable}, output_span,
                                                 input_span),
            uni20::linalg::KernelTypeAcceptance::no);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(uni20::linalg::CpuReferenceBackend{},
                                                 uni20::linalg::transform_op{mutating_input_callable}, output_span,
                                                 input_span),
            uni20::linalg::KernelTypeAcceptance::no);
}

} // namespace
