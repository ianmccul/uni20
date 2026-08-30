#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/linalg/backends/cuda/transform.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>
#include <uni20/tensor/transform.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{

using extents_type = stdex::dextents<uni20::index_type, 2>;
using real_cuda_matrix = uni20::CudaTensor<double, 2>;
using const_real_cuda_mdspec = decltype(std::declval<real_cuda_matrix const&>().mdspec());
using mutable_real_cuda_mdspec = decltype(std::declval<real_cuda_matrix&>().mdspec());

static_assert(uni20::linalg::detail::cuda_reference::is_raw_mutable_cuda_mdspec<mutable_real_cuda_mdspec>);
static_assert(uni20::linalg::detail::cuda_reference::is_raw_const_cuda_mdspec<const_real_cuda_mdspec>);
static_assert(!uni20::linalg::detail::cuda_reference::is_raw_const_cuda_mdspec<mutable_real_cuda_mdspec>);
static_assert(!uni20::linalg::detail::cuda_reference::is_raw_mutable_cuda_mdspec<const_real_cuda_mdspec>);
static_assert(uni20::linalg::detail::cuda_reference::SupportedStatelessUnaryMdspecs<
              mutable_real_cuda_mdspec, const_real_cuda_mdspec, uni20::linalg::negate>);

template <class Tensor>
auto make_strided_mdspec(Tensor& tensor, std::size_t offset, extents_type extents,
                         std::array<uni20::index_type, 2> const& strides)
{
  auto base = tensor.mdspec();
  using base_type = decltype(base);
  using layout_type = stdex::layout_stride;
  using mapping_type = layout_type::mapping<extents_type>;
  using result_type = uni20::mdspec<typename base_type::element_type, extents_type, layout_type,
                                    typename base_type::accessor_type, typename base_type::data_descriptor_type>;
  return result_type{base.data_descriptor().offset_by(offset), mapping_type{extents, strides}, base.accessor()};
}

class CudaTransformTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      auto status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess)
      {
        cudaGetLastError();
        GTEST_SKIP() << "CUDA runtime unavailable: " << cudaGetErrorString(status);
      }
      if (device_count_ == 0) GTEST_SKIP() << "No CUDA devices available";
    }

    int device_count_ = 0;
};

TEST(CudaTransformPlanningTest, UnaryPlanDecodesDifferentPaddedStrides)
{
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  extents_type const extents{2, 3};
  mapping_type output_mapping{extents, std::array<uni20::index_type, 2>{1, 3}};
  mapping_type input_mapping{extents, std::array<uni20::index_type, 2>{4, 1}};
  double value = 0.0;
  stdex::mdspan output{&value, output_mapping};
  stdex::mdspan input{&value, input_mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseUnaryPlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_transform_plan<
              uni20::linalg::detail::cuda_reference::elementwise_arithmetic_maximum_rank>(output, plan, input));
  ASSERT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32);
  auto const& layout = plan.plan_32;
  EXPECT_EQ(layout.compact_rank, 2);
  EXPECT_EQ(layout.element_count, 6);
  EXPECT_EQ(layout.offsets(0), (std::array<std::int32_t, 2>{0, 0}));
  EXPECT_EQ(layout.offsets(1), (std::array<std::int32_t, 2>{1, 4}));
  EXPECT_EQ(layout.offsets(2), (std::array<std::int32_t, 2>{3, 1}));
  EXPECT_EQ(layout.offsets(5), (std::array<std::int32_t, 2>{7, 6}));
}

TEST(CudaTransformPlanningTest, RankZeroUnaryPlanVisitsOneElement)
{
  using scalar_extents = stdex::extents<uni20::index_type>;
  double output_value = 0.0;
  double input_value = 1.0;
  stdex::mdspan<double, scalar_extents> output{&output_value};
  stdex::mdspan<double const, scalar_extents> input{&input_value};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseUnaryPlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_transform_plan<
              uni20::linalg::detail::cuda_reference::elementwise_arithmetic_maximum_rank>(output, plan, input));
  ASSERT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32);
  EXPECT_EQ(plan.plan_32.compact_rank, 0);
  EXPECT_EQ(plan.plan_32.element_count, 1);
  EXPECT_EQ(plan.plan_32.offsets(0), (std::array<std::int32_t, 2>{0, 0}));
}

TEST(CudaTransformPlanningTest, BinaryPlanDecodesThreeIndependentMappings)
{
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  extents_type const extents{2, 3};
  mapping_type output_mapping{extents, std::array<uni20::index_type, 2>{1, 3}};
  mapping_type lhs_mapping{extents, std::array<uni20::index_type, 2>{4, 1}};
  mapping_type rhs_mapping{extents, std::array<uni20::index_type, 2>{5, 2}};
  double value = 0.0;
  stdex::mdspan output{&value, output_mapping};
  stdex::mdspan lhs{&value, lhs_mapping};
  stdex::mdspan rhs{&value, rhs_mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseBinaryPlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_transform_plan<
              uni20::linalg::detail::cuda_reference::elementwise_arithmetic_maximum_rank>(output, plan, lhs, rhs));
  ASSERT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32);
  auto const& layout = plan.plan_32;
  EXPECT_EQ(layout.compact_rank, 2);
  EXPECT_EQ(layout.element_count, 6);
  EXPECT_EQ(layout.offsets(0), (std::array<std::int32_t, 3>{0, 0, 0}));
  EXPECT_EQ(layout.offsets(1), (std::array<std::int32_t, 3>{1, 4, 5}));
  EXPECT_EQ(layout.offsets(2), (std::array<std::int32_t, 3>{3, 1, 2}));
  EXPECT_EQ(layout.offsets(5), (std::array<std::int32_t, 3>{7, 6, 9}));
}

TEST(CudaTransformPlanningTest, RegisteredCallablesHaveExactCudaTypeAcceptance)
{
  using negate_operation = uni20::linalg::transform_op<uni20::linalg::negate>;
  using square_operation = uni20::linalg::transform_op<uni20::linalg::square>;
  using reciprocal_operation = uni20::linalg::transform_op<uni20::linalg::reciprocal>;
  using scale_operation = uni20::linalg::transform_op<uni20::linalg::scale<double>>;
  using integer_scale_operation = uni20::linalg::transform_op<uni20::linalg::scale<int>>;
  using add_operation = uni20::linalg::transform_op<uni20::linalg::add>;
  using subtract_operation = uni20::linalg::transform_op<uni20::linalg::subtract>;
  using multiply_operation = uni20::linalg::transform_op<uni20::linalg::multiply>;
  using divide_operation = uni20::linalg::transform_op<uni20::linalg::divide>;
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, negate_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, scale_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, square_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, reciprocal_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);
  EXPECT_EQ(
      (uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, integer_scale_operation,
                                                      mutable_real_cuda_mdspec, const_real_cuda_mdspec>()),
      uni20::linalg::KernelTypeAcceptance::no);
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, add_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec,
                                                            const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, subtract_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec,
                                                            const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, multiply_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec,
                                                            const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);
  EXPECT_EQ((uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, divide_operation,
                                                            mutable_real_cuda_mdspec, const_real_cuda_mdspec,
                                                            const_real_cuda_mdspec>()),
            uni20::linalg::KernelTypeAcceptance::maybe);

  auto function = [](double value) { return 2.0 * value; };
  using unregistered_operation = uni20::linalg::transform_op<decltype(function)>;
  auto const unregistered =
      uni20::linalg::detail::backend_type_acceptance<uni20::linalg::CudaReferenceBackend, unregistered_operation,
                                                     mutable_real_cuda_mdspec, const_real_cuda_mdspec>();
  EXPECT_EQ(unregistered, uni20::linalg::KernelTypeAcceptance::no);
}

TEST_F(CudaTransformTest, ScalesByRetainedFactor)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host_input(3);
  host_input[0] = 1.0;
  host_input[1] = -2.0;
  host_input[2] = 3.5;

  auto input = uni20::to_device(host_input, 0);
  uni20::CudaTensor<double, 1> output(runtime.device_resources(0), 3);

  uni20::assign_transform(output, uni20::linalg::scale{2.5}, input);

  auto result = uni20::to_host(output);
  EXPECT_DOUBLE_EQ(result[0], 2.5);
  EXPECT_DOUBLE_EQ(result[1], -5.0);
  EXPECT_DOUBLE_EQ(result[2], 8.75);
}

TEST_F(CudaTransformTest, FillsRealAndComplexStridedTensors)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::CudaTensor<double, 2> real(runtime.device_resources(0), 2, 3);
  uni20::CudaTensor<uni20::cfloat, 2> complex(runtime.device_resources(0), 2, 3);

  uni20::fill(real, 2.5);
  uni20::fill(complex, uni20::cfloat{-1.0F, 3.0F});

  auto real_result = uni20::to_host(real);
  auto complex_result = uni20::to_host(complex);
  for (uni20::index_type column = 0; column < 3; ++column)
  {
    for (uni20::index_type row = 0; row < 2; ++row)
    {
      EXPECT_DOUBLE_EQ((real_result[row, column]), 2.5);
      EXPECT_EQ((complex_result[row, column]), (uni20::cfloat{-1.0F, 3.0F}));
    }
  }
}

TEST_F(CudaTransformTest, UpdatesScaleAddAndAxpyInPlace)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host_lhs(3);
  uni20::Tensor<double, 1> host_rhs(3);
  host_lhs[0] = 1.0;
  host_lhs[1] = -2.0;
  host_lhs[2] = 3.0;
  host_rhs[0] = 4.0;
  host_rhs[1] = 5.0;
  host_rhs[2] = -6.0;
  auto lhs = uni20::to_device(host_lhs, 0);
  auto rhs = uni20::to_device(host_rhs, 0);

  uni20::transform_inplace(lhs, uni20::linalg::scale{2.0});
  uni20::transform_inplace(lhs, uni20::linalg::add{}, rhs);
  uni20::transform_inplace(lhs, uni20::linalg::add_scaled{-0.5}, rhs);

  auto result = uni20::to_host(lhs);
  EXPECT_DOUBLE_EQ(result[0], 4.0);
  EXPECT_DOUBLE_EQ(result[1], -1.5);
  EXPECT_DOUBLE_EQ(result[2], 3.0);
}

TEST_F(CudaTransformTest, ScalesComplexValuesByRealAndComplexFactors)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<uni20::cfloat, 1> host_input(2);
  host_input[0] = uni20::cfloat{1.0F, 2.0F};
  host_input[1] = uni20::cfloat{-3.0F, 4.0F};

  auto input = uni20::to_device(host_input, 0);
  uni20::CudaTensor<uni20::cfloat, 1> real_scaled(runtime.device_resources(0), 2);
  uni20::CudaTensor<uni20::cfloat, 1> complex_scaled(runtime.device_resources(0), 2);

  uni20::assign_transform(real_scaled, uni20::linalg::scale{2.0F}, input);
  uni20::assign_transform(complex_scaled, uni20::linalg::scale{uni20::cfloat{0.0F, 1.0F}}, input);

  auto real_result = uni20::to_host(real_scaled);
  auto complex_result = uni20::to_host(complex_scaled);
  EXPECT_EQ(real_result[0], (uni20::cfloat{2.0F, 4.0F}));
  EXPECT_EQ(real_result[1], (uni20::cfloat{-6.0F, 8.0F}));
  EXPECT_EQ(complex_result[0], (uni20::cfloat{-2.0F, 1.0F}));
  EXPECT_EQ(complex_result[1], (uni20::cfloat{-4.0F, -3.0F}));
}

TEST_F(CudaTransformTest, AppliesRegisteredUnaryArithmetic)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host_input(3);
  host_input[0] = 2.0;
  host_input[1] = -4.0;
  host_input[2] = 0.5;

  auto input = uni20::to_device(host_input, 0);
  uni20::CudaTensor<double, 1> squared(runtime.device_resources(0), 3);
  uni20::CudaTensor<double, 1> reciprocals(runtime.device_resources(0), 3);
  uni20::assign_transform(squared, uni20::linalg::square{}, input);
  uni20::assign_transform(reciprocals, uni20::linalg::reciprocal{}, input);

  auto squared_result = uni20::to_host(squared);
  auto reciprocal_result = uni20::to_host(reciprocals);
  EXPECT_DOUBLE_EQ(squared_result[0], 4.0);
  EXPECT_DOUBLE_EQ(squared_result[1], 16.0);
  EXPECT_DOUBLE_EQ(squared_result[2], 0.25);
  EXPECT_DOUBLE_EQ(reciprocal_result[0], 0.5);
  EXPECT_DOUBLE_EQ(reciprocal_result[1], -0.25);
  EXPECT_DOUBLE_EQ(reciprocal_result[2], 2.0);
}

TEST_F(CudaTransformTest, AppliesRegisteredBinaryArithmetic)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host_lhs(3);
  uni20::Tensor<double, 1> host_rhs(3);
  host_lhs[0] = 2.0;
  host_lhs[1] = -4.0;
  host_lhs[2] = 8.0;
  host_rhs[0] = 0.5;
  host_rhs[1] = 2.0;
  host_rhs[2] = -4.0;

  auto lhs = uni20::to_device(host_lhs, 0);
  auto rhs = uni20::to_device(host_rhs, 0);
  uni20::CudaTensor<double, 1> differences(runtime.device_resources(0), 3);
  uni20::CudaTensor<double, 1> products(runtime.device_resources(0), 3);
  uni20::CudaTensor<double, 1> quotients(runtime.device_resources(0), 3);
  uni20::assign_transform(differences, uni20::linalg::subtract{}, lhs, rhs);
  uni20::assign_transform(products, uni20::linalg::multiply{}, lhs, rhs);
  uni20::assign_transform(quotients, uni20::linalg::divide{}, lhs, rhs);

  auto difference_result = uni20::to_host(differences);
  auto product_result = uni20::to_host(products);
  auto quotient_result = uni20::to_host(quotients);
  EXPECT_DOUBLE_EQ(difference_result[0], 1.5);
  EXPECT_DOUBLE_EQ(difference_result[1], -6.0);
  EXPECT_DOUBLE_EQ(difference_result[2], 12.0);
  EXPECT_DOUBLE_EQ(product_result[0], 1.0);
  EXPECT_DOUBLE_EQ(product_result[1], -8.0);
  EXPECT_DOUBLE_EQ(product_result[2], -32.0);
  EXPECT_DOUBLE_EQ(quotient_result[0], 4.0);
  EXPECT_DOUBLE_EQ(quotient_result[1], -2.0);
  EXPECT_DOUBLE_EQ(quotient_result[2], -2.0);
}

TEST_F(CudaTransformTest, AppliesRegisteredComplexArithmetic)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<uni20::cfloat, 1> host_lhs(2);
  uni20::Tensor<uni20::cfloat, 1> host_rhs(2);
  host_lhs[0] = uni20::cfloat{0.0F, 1.0F};
  host_lhs[1] = uni20::cfloat{2.0F, 0.0F};
  host_rhs[0] = uni20::cfloat{0.0F, -1.0F};
  host_rhs[1] = uni20::cfloat{0.5F, 0.0F};

  auto lhs = uni20::to_device(host_lhs, 0);
  auto rhs = uni20::to_device(host_rhs, 0);
  uni20::CudaTensor<uni20::cfloat, 1> squared(runtime.device_resources(0), 2);
  uni20::CudaTensor<uni20::cfloat, 1> reciprocals(runtime.device_resources(0), 2);
  uni20::CudaTensor<uni20::cfloat, 1> products(runtime.device_resources(0), 2);
  uni20::CudaTensor<uni20::cfloat, 1> quotients(runtime.device_resources(0), 2);
  uni20::assign_transform(squared, uni20::linalg::square{}, lhs);
  uni20::assign_transform(reciprocals, uni20::linalg::reciprocal{}, lhs);
  uni20::assign_transform(products, uni20::linalg::multiply{}, lhs, rhs);
  uni20::assign_transform(quotients, uni20::linalg::divide{}, lhs, rhs);

  auto squared_result = uni20::to_host(squared);
  auto reciprocal_result = uni20::to_host(reciprocals);
  auto product_result = uni20::to_host(products);
  auto quotient_result = uni20::to_host(quotients);
  EXPECT_EQ(squared_result[0], (uni20::cfloat{-1.0F, 0.0F}));
  EXPECT_EQ(squared_result[1], (uni20::cfloat{4.0F, 0.0F}));
  EXPECT_EQ(reciprocal_result[0], (uni20::cfloat{0.0F, -1.0F}));
  EXPECT_EQ(reciprocal_result[1], (uni20::cfloat{0.5F, 0.0F}));
  EXPECT_EQ(product_result[0], (uni20::cfloat{1.0F, 0.0F}));
  EXPECT_EQ(product_result[1], (uni20::cfloat{1.0F, 0.0F}));
  EXPECT_EQ(quotient_result[0], (uni20::cfloat{-1.0F, 0.0F}));
  EXPECT_EQ(quotient_result[1], (uni20::cfloat{4.0F, 0.0F}));
}

TEST_F(CudaTransformTest, AddsInputsWithDifferentCanonicalLayouts)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 2> host_lhs(2, 3);
  uni20::RowMajorTensor<double, 2> host_rhs(2, 3);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
    {
      host_lhs[row, column] = 10.0 * row + column;
      host_rhs[row, column] = 100.0 + 2.0 * row + 3.0 * column;
    }

  auto lhs = uni20::to_device(host_lhs, 0);
  auto rhs = uni20::to_device(host_rhs, 0);
  real_cuda_matrix output(runtime.device_resources(0), 2, 3);

  uni20::assign_transform(output, uni20::linalg::add{}, lhs, rhs);

  auto result = uni20::to_host(output);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((result[row, column]), (100.0 + 12.0 * row + 4.0 * column));
}

TEST_F(CudaTransformTest, BinaryAddSharesOneLeaseForRepeatedInput)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::Tensor<double, 1> host_input(3);
  host_input[0] = 1.0;
  host_input[1] = 2.0;
  host_input[2] = 3.0;

  auto input = uni20::to_device(host_input, 0);
  uni20::CudaTensor<double, 1> output(runtime.device_resources(0), 3);
  uni20::assign_transform(output, uni20::linalg::add{}, input, input);

  auto result = uni20::to_host(output);
  EXPECT_DOUBLE_EQ(result[0], 2.0);
  EXPECT_DOUBLE_EQ(result[1], 4.0);
  EXPECT_DOUBLE_EQ(result[2], 6.0);
}

TEST_F(CudaTransformTest, NegatesDifferentCanonicalLayouts)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::RowMajorTensor<double, 2> host_input(2, 3);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      host_input[row, column] = 10.0 * row + column + 1.0;

  auto input = uni20::to_device(host_input, 0);
  real_cuda_matrix output(runtime.device_resources(0), 2, 3);

  uni20::assign_transform(output, uni20::linalg::negate{}, input);

  auto result = uni20::to_host(output);
  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((result[row, column]), -(10.0 * row + column + 1.0));
}

TEST_F(CudaTransformTest, NegatesComplexExecutionValues)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<uni20::cfloat, 1> host_input(3);
  host_input[0] = uni20::cfloat{1.0F, 2.0F};
  host_input[1] = uni20::cfloat{-3.0F, 4.0F};
  host_input[2] = uni20::cfloat{5.0F, -6.0F};

  auto input = uni20::to_device(host_input, 0);
  uni20::CudaTensor<uni20::cfloat, 1> output(runtime.device_resources(0), 3);

  uni20::assign_transform(output, uni20::linalg::negate{}, input);

  auto result = uni20::to_host(output);
  EXPECT_EQ(result[0], (uni20::cfloat{-1.0F, -2.0F}));
  EXPECT_EQ(result[1], (uni20::cfloat{3.0F, -4.0F}));
  EXPECT_EQ(result[2], (uni20::cfloat{-5.0F, 6.0F}));
}

TEST_F(CudaTransformTest, HandlesPaddedMappingsAndBufferOffsets)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host_input(12);
  for (std::size_t index = 0; index < 12; ++index)
    host_input[index] = static_cast<double>(index + 1);
  uni20::Tensor<double, 1> host_output(12);
  for (std::size_t index = 0; index < 12; ++index)
    host_output[index] = 0.0;

  auto input_storage = uni20::to_device(host_input, 0);
  auto output_storage = uni20::to_device(host_output, 0);
  auto output = make_strided_mdspec(output_storage, 1, extents_type{2, 3}, std::array<uni20::index_type, 2>{1, 4});
  auto input =
      make_strided_mdspec(std::as_const(input_storage), 2, extents_type{2, 3}, std::array<uni20::index_type, 2>{4, 1});

  uni20::linalg::dispatch_kernel(uni20::linalg::CudaReferenceBackend{},
                                 uni20::linalg::transform_op{uni20::linalg::negate{}}, output, input);

  auto result = uni20::to_host(output_storage);
  std::array<bool, 12> written{};
  for (auto index : {1U, 2U, 5U, 6U, 9U, 10U})
    written[index] = true;
  std::array<std::size_t, 6> const input_indices{2, 6, 3, 7, 4, 8};
  std::size_t input_index = 0;
  for (std::size_t index = 0; index < 12; ++index)
  {
    if (written[index])
      EXPECT_DOUBLE_EQ(result[index], -host_input[input_indices[input_index++]]);
    else
      EXPECT_DOUBLE_EQ(result[index], 0.0);
  }
}

TEST_F(CudaTransformTest, SameOffsetOverwriteDeclinesBeforeAccess)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  real_cuda_matrix tensor(runtime.device_resources(0), 2, 2);
  auto output = tensor.mdspec();
  auto input = std::as_const(tensor).mdspec();

  auto plan = uni20::linalg::detail::cuda_reference::prepare_stateless_unary<uni20::linalg::negate>(output, input);

  EXPECT_EQ(plan.attempt, uni20::linalg::KernelAttempt::unsupported_instance);
  EXPECT_TRUE(plan.has_work);
}

} // namespace
