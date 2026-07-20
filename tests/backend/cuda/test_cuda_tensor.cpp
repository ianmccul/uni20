#include <uni20/storage/cuda_async_storage.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace
{

using tensor_type = uni20::CudaAsyncTensor<double, 2>;
using direct_tensor_type = uni20::Tensor<double, 2, uni20::CudaAsyncStorage>;
using mutable_span_type = typename tensor_type::mdspan_type;
using const_span_type = typename tensor_type::const_mdspan_type;

static_assert(std::same_as<typename tensor_type::storage_type, uni20::cuda::CudaBuffer<double>>);
static_assert(std::same_as<tensor_type, direct_tensor_type>);
static_assert(std::same_as<typename mutable_span_type::data_handle_type, uni20::cuda::CudaBufferView<double>>);
static_assert(std::same_as<typename const_span_type::data_handle_type, uni20::cuda::CudaBufferView<double const>>);
static_assert(uni20::TensorView<tensor_type>);
static_assert(uni20::MutableTensorView<tensor_type>);
static_assert(uni20::RankedStridedTensorView<tensor_type, 2>);
static_assert(uni20::MutableRankedStridedTensorView<tensor_type, 2>);
static_assert(uni20::OwningTensor<tensor_type>);
static_assert(!std::copy_constructible<tensor_type>);
static_assert(std::move_constructible<tensor_type>);
static_assert(!std::convertible_to<typename mutable_span_type::reference, double>);
static_assert(!std::assignable_from<typename mutable_span_type::reference&, double>);

class CudaTensorTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      int device_count = 0;
      cudaError_t const status = cudaGetDeviceCount(&device_count);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count == 0)
      {
        GTEST_SKIP() << "no CUDA devices are available";
      }
    }
};

TEST_F(CudaTensorTest, ExplicitResourcesConstructionOwnsDeviceBufferAndOpaqueMdspan)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);

  EXPECT_EQ(tensor.rows(), 2);
  EXPECT_EQ(tensor.cols(), 3);
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.storage().size(), 6U);
  EXPECT_EQ(tensor.storage().device(), resources.device());

  auto span = tensor.mdspan();
  EXPECT_EQ(&span.data_handle().buffer(), &tensor.storage());
  EXPECT_EQ(span.data_handle().element_offset(), 0U);
  EXPECT_EQ((span[1, 2].element_offset()), 5U);

  tensor_type const& const_tensor = tensor;
  auto const_span = const_tensor.mdspan();
  EXPECT_EQ(&const_span.data_handle().buffer(), &tensor.storage());
  EXPECT_EQ((const_span[1, 2].element_offset()), 5U);
  EXPECT_EQ(tensor.backend_selector(), uni20::CudaAsyncStorage::backend_selector());
}

TEST_F(CudaTensorTest, ShapeResetKeepsTheOriginalDeviceResources)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  auto* const original_resources = &tensor.storage().resources();

  tensor.reset_shape(tensor_type::extents_type{4, 5});

  EXPECT_EQ(&tensor.storage().resources(), original_resources);
  EXPECT_EQ(tensor.rows(), 4);
  EXPECT_EQ(tensor.cols(), 5);
  EXPECT_EQ(tensor.storage().size(), 20U);
}

TEST_F(CudaTensorTest, DefaultConstructionUsesInstalledRuntimeResources)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::CudaAsyncTensor<double, 2> tensor(3, 4);

  EXPECT_EQ(tensor.rows(), 3);
  EXPECT_EQ(tensor.cols(), 4);
  EXPECT_EQ(&tensor.storage().resources(), &runtime.default_device_resources());
}

} // namespace
