#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/linalg/backends/cuda/conjugate_inplace.hpp>
#include <uni20/tensor/conjugate_inplace.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace
{

using complex_type = uni20::complex<double>;
using complex_cuda_matrix_type = uni20::CudaTensor<complex_type, 2>;
using real_cuda_matrix_type = uni20::CudaTensor<double, 2>;
using complex_cuda_mdspec = decltype(std::declval<complex_cuda_matrix_type&>().mdspec());
using real_cuda_mdspec = decltype(std::declval<real_cuda_matrix_type&>().mdspec());
using extents_type = stdex::dextents<uni20::index_type, 2>;

static_assert(uni20::linalg::detail::cuda_reference::SupportedConjugateMdspec<complex_cuda_mdspec>);
static_assert(uni20::linalg::detail::cuda_reference::SupportedConjugateMdspec<real_cuda_mdspec>);

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

class CudaConjugateInplaceTest : public ::testing::Test {
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

TEST(CudaConjugatePlanningTest, OneOperandPlanCompactsAndDecodesPaddedStrides)
{
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  mapping_type mapping{extents_type{2, 3}, std::array<uni20::index_type, 2>{1, 3}};
  complex_type value{};
  stdex::mdspan span{&value, mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseConjugatePlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_conjugate_plan(span, plan));
  ASSERT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32);
  auto const& layout = plan.plan_32;
  EXPECT_EQ(layout.compact_rank, 2);
  EXPECT_EQ(layout.element_count, 6);
  EXPECT_EQ(layout.offsets(0), (std::array<std::int32_t, 1>{0}));
  EXPECT_EQ(layout.offsets(2), (std::array<std::int32_t, 1>{3}));
  EXPECT_EQ(layout.offsets(5), (std::array<std::int32_t, 1>{7}));
}

TEST(CudaConjugatePlanningTest, OneOperandPlanSelects64BitReachableOffsets)
{
  using vector_extents_type = stdex::dextents<uni20::index_type, 1>;
  using mapping_type = stdex::layout_stride::mapping<vector_extents_type>;
  auto const stride = static_cast<uni20::index_type>(std::numeric_limits<std::int32_t>::max()) + 1;
  mapping_type mapping{vector_extents_type{2}, std::array<uni20::index_type, 1>{stride}};
  complex_type value{};
  stdex::mdspan span{&value, mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseConjugatePlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_conjugate_plan(span, plan));
  EXPECT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_64);
  EXPECT_EQ(plan.plan_64.offsets(1), (std::array<std::int64_t, 1>{static_cast<std::int64_t>(stride)}));
}

TEST(CudaConjugatePlanningTest, OneOperandPlanDeclinesNegativeStrides)
{
  using vector_extents_type = stdex::dextents<uni20::index_type, 1>;
  using mapping_type = stdex::layout_stride::mapping<vector_extents_type>;
  mapping_type mapping{vector_extents_type{2}, std::array<uni20::index_type, 1>{-1}};
  complex_type value{};
  stdex::mdspan span{&value, mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseConjugatePlan plan;

  EXPECT_FALSE(uni20::linalg::detail::cuda_reference::try_make_elementwise_conjugate_plan(span, plan));
}

TEST_F(CudaConjugateInplaceTest, ConjugatesComplexDoubleAndFloatStorage)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});

  uni20::Tensor<uni20::cdouble, 1> double_source(2);
  double_source[0] = uni20::cdouble{1.0, 2.0};
  double_source[1] = uni20::cdouble{-3.0, 4.0};
  auto double_device = uni20::to_device(double_source, 0);
  uni20::conjugate_inplace(double_device);
  auto double_result = uni20::to_host(double_device);
  EXPECT_EQ(double_result[0], (uni20::cdouble{1.0, -2.0}));
  EXPECT_EQ(double_result[1], (uni20::cdouble{-3.0, -4.0}));

  uni20::Tensor<uni20::cfloat, 1> float_source(2);
  float_source[0] = uni20::cfloat{5.0F, -6.0F};
  float_source[1] = uni20::cfloat{-7.0F, -8.0F};
  auto float_device = uni20::to_device(float_source, 0);
  uni20::conjugate_inplace(float_device);
  auto float_result = uni20::to_host(float_device);
  EXPECT_EQ(float_result[0], (uni20::cfloat{5.0F, 6.0F}));
  EXPECT_EQ(float_result[1], (uni20::cfloat{-7.0F, 8.0F}));
}

TEST_F(CudaConjugateInplaceTest, HandlesPaddedStridesAndBufferOffsets)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<complex_type, 1> source(10);
  for (std::size_t index = 0; index < 10; ++index)
    source[index] = complex_type{static_cast<double>(index), static_cast<double>(index + 1)};
  auto device = uni20::to_device(source, 0);
  auto view = make_strided_mdspec(device, 1, extents_type{2, 3}, std::array<uni20::index_type, 2>{1, 3});

  uni20::conjugate_inplace(uni20::linalg::CudaReferenceBackend{}, view);

  auto result = uni20::to_host(device);
  std::array<bool, 10> conjugated{};
  for (auto index : {1U, 2U, 4U, 5U, 7U, 8U})
    conjugated[index] = true;
  for (std::size_t index = 0; index < 10; ++index)
  {
    double const expected_imaginary =
        conjugated[index] ? -static_cast<double>(index + 1) : static_cast<double>(index + 1);
    EXPECT_EQ(result[index], (complex_type{static_cast<double>(index), expected_imaginary}));
  }
}

TEST_F(CudaConjugateInplaceTest, HandlesRankZeroAndRowMajorMappings)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});

  complex_type const scalar{2.0, -3.0};
  using cuda_scalar_type = uni20::CudaTensor<complex_type, 0>;
  cuda_scalar_type device_scalar(runtime.device_resources(0), typename cuda_scalar_type::extents_type{});
  {
    auto access = device_scalar.storage().blocking_write_access();
    uni20::cuda::check(cudaMemcpy(access.data(), &scalar, sizeof(scalar), cudaMemcpyHostToDevice),
                       "initialize rank-zero conjugation test", 0);
  }
  uni20::conjugate_inplace(device_scalar);
  complex_type scalar_result;
  {
    auto access = device_scalar.storage().blocking_read_access();
    uni20::cuda::check(cudaMemcpy(&scalar_result, access.data(), sizeof(scalar_result), cudaMemcpyDeviceToHost),
                       "read rank-zero conjugation test", 0);
  }
  EXPECT_EQ(scalar_result, (complex_type{2.0, 3.0}));

  uni20::RowMajorTensor<complex_type, 2> matrix(2, 2);
  matrix[0, 0] = complex_type{1.0, 2.0};
  matrix[0, 1] = complex_type{3.0, -4.0};
  matrix[1, 0] = complex_type{-5.0, 6.0};
  matrix[1, 1] = complex_type{-7.0, -8.0};
  auto device_matrix = uni20::to_device(matrix, 0);
  uni20::conjugate_inplace(device_matrix);
  auto matrix_result = uni20::to_host(device_matrix);
  EXPECT_EQ((matrix_result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((matrix_result[0, 1]), (complex_type{3.0, 4.0}));
  EXPECT_EQ((matrix_result[1, 0]), (complex_type{-5.0, -6.0}));
  EXPECT_EQ((matrix_result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaConjugateInplaceTest, EmptyAndRealOperandsSucceedWithoutElementwiseWork)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  complex_cuda_matrix_type empty(runtime.device_resources(0), 0, 3);
  auto empty_mdspec = empty.mdspec();
  auto empty_plan = uni20::linalg::detail::cuda_reference::prepare_conjugate_inplace(empty_mdspec);
  EXPECT_EQ(empty_plan.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_FALSE(empty_plan.has_work);
  uni20::conjugate_inplace(empty);

  real_cuda_matrix_type real(runtime.device_resources(0), 2, 3);
  auto real_mdspec = real.mdspec();
  auto real_plan = uni20::linalg::detail::cuda_reference::prepare_conjugate_inplace(real_mdspec);
  EXPECT_EQ(real_plan.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_FALSE(real_plan.has_work);
  uni20::conjugate_inplace(real);
}

} // namespace
