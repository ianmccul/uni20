#include <uni20/async/cuda_task.hpp>
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace
{

using tensor_type = uni20::CudaTensor<double, 2>;
using direct_tensor_type = uni20::Tensor<double, 2, uni20::CudaStorage>;
using mutable_span_type = typename tensor_type::mdspan_type;
using const_span_type = typename tensor_type::const_mdspan_type;
using mutable_device_span_type = decltype(std::declval<tensor_type&>().device_mdspan());
using const_device_span_type = decltype(std::declval<tensor_type const&>().device_mdspan());
using read_lease_type = decltype(uni20::blocking_read_access(std::declval<tensor_type const&>()));
using write_lease_type = decltype(uni20::blocking_write_access(std::declval<tensor_type&>()));
using read_access_type =
    decltype(uni20::read_access(std::declval<tensor_type const&>(), std::declval<uni20::cuda::Stream const&>()));
using write_access_type =
    decltype(uni20::write_access(std::declval<tensor_type&>(), std::declval<uni20::cuda::Stream const&>()));

template <class Tensor>
concept HasMutableMdspan = requires(Tensor& tensor) { tensor.mdspan(); };

template <class Tensor>
concept HasConstMdspan = requires(Tensor const& tensor) { tensor.mdspan(); };

template <class Value>
concept HasStorageObserver = requires(Value& value) { value.storage(); };

template <class Tensor>
concept CanBlockingReadRvalue = requires { uni20::blocking_read_access(std::declval<Tensor&&>()); };

template <class Tensor>
concept CanStreamReadRvalue =
    requires(uni20::cuda::Stream const& stream) { uni20::read_access(std::declval<Tensor&&>(), stream); };

struct DeviceSpanCallCounts
{
    int mutable_calls = 0;
    mutable int const_calls = 0;
};

class CountingCudaTensorView {
  public:
    using tensor_type = uni20::CudaTensor<double, 2>;
    using storage_policy = uni20::CudaStorage;
    using extents_type = typename tensor_type::extents_type;

    CountingCudaTensorView(tensor_type& tensor, DeviceSpanCallCounts& calls) : tensor_(&tensor), calls_(&calls) {}

    [[nodiscard]] auto device_mdspan()
    {
      ++calls_->mutable_calls;
      return tensor_->device_mdspan();
    }

    [[nodiscard]] auto device_mdspan() const
    {
      ++calls_->const_calls;
      return std::as_const(*tensor_).device_mdspan();
    }

    [[nodiscard]] auto backend_selector() const { return tensor_->backend_selector(); }

    [[nodiscard]] auto extents() const -> extents_type const& { return tensor_->extents(); }

    [[nodiscard]] auto extent(std::size_t axis) const { return tensor_->extent(axis); }

  private:
    tensor_type* tensor_;
    DeviceSpanCallCounts* calls_;
};

using owning_read_lease_type = decltype(uni20::blocking_read_access(std::declval<tensor_type&&>()));
using owning_read_access_type =
    decltype(uni20::read_access(std::declval<tensor_type&&>(), std::declval<uni20::cuda::Stream const&>()));

static_assert(std::same_as<typename tensor_type::storage_type, uni20::cuda::CudaBuffer<double>>);
static_assert(std::same_as<tensor_type, direct_tensor_type>);
static_assert(std::same_as<typename mutable_span_type::data_handle_type, double*>);
static_assert(std::same_as<typename const_span_type::data_handle_type, double const*>);
static_assert(std::same_as<typename mutable_span_type::reference, double&>);
static_assert(std::same_as<typename const_span_type::reference, double const&>);
static_assert(std::same_as<typename mutable_device_span_type::data_handle_type, double*>);
static_assert(std::same_as<typename const_device_span_type::data_handle_type, double const*>);
static_assert(!tensor_type::immediately_readable);
static_assert(!tensor_type::immediately_writable);
static_assert(tensor_type::deferred_readable);
static_assert(tensor_type::deferred_writable);
static_assert(uni20::DeviceSpanLike<mutable_device_span_type>);
static_assert(uni20::MutableDeviceSpanLike<mutable_device_span_type>);
static_assert(uni20::DeviceSpanLike<const_device_span_type>);
static_assert(!uni20::SpanLike<mutable_device_span_type>);
static_assert(!uni20::SpanLike<const_device_span_type>);
static_assert(!uni20::TensorView<tensor_type>);
static_assert(!uni20::MutableTensorView<tensor_type>);
static_assert(!HasMutableMdspan<tensor_type>);
static_assert(!HasConstMdspan<tensor_type>);
static_assert(uni20::DeviceTensorView<tensor_type>);
static_assert(uni20::MutableDeviceTensorView<tensor_type>);
static_assert(uni20::RankedStridedDeviceTensorView<tensor_type, 2>);
static_assert(uni20::MutableRankedStridedDeviceTensorView<tensor_type, 2>);
static_assert(uni20::OwningTensor<tensor_type>);
static_assert(uni20::ReadTensorLease<read_lease_type>);
static_assert(uni20::WriteTensorLease<write_lease_type>);
static_assert(std::same_as<typename read_lease_type::storage_policy, uni20::CudaStorage>);
static_assert(std::same_as<typename write_lease_type::storage_policy, uni20::CudaStorage>);
static_assert(std::same_as<typename read_lease_type::mdspan_type::data_handle_type, double const*>);
static_assert(std::same_as<typename write_lease_type::mdspan_type::data_handle_type, double*>);
static_assert(std::same_as<typename read_lease_type::mdspan_type::reference, double const&>);
static_assert(std::same_as<typename write_lease_type::mdspan_type::reference, double&>);
static_assert(
    std::same_as<decltype(std::declval<read_lease_type const&>().storage()), tensor_type::storage_type const&>);
static_assert(!HasStorageObserver<write_lease_type>);
static_assert(uni20::async::TaskAwaitable<read_access_type>);
static_assert(uni20::async::TaskAwaitable<write_access_type>);
static_assert(uni20::ReadTensorLease<owning_read_lease_type>);
static_assert(uni20::async::TaskAwaitable<owning_read_access_type>);
static_assert(CanBlockingReadRvalue<tensor_type>);
static_assert(CanStreamReadRvalue<tensor_type>);
static_assert(uni20::DeviceTensorView<CountingCudaTensorView>);
static_assert(uni20::MutableDeviceTensorView<CountingCudaTensorView>);
static_assert(!uni20::OwningTensor<CountingCudaTensorView>);
static_assert(!CanBlockingReadRvalue<CountingCudaTensorView>);
static_assert(!CanStreamReadRvalue<CountingCudaTensorView>);
static_assert(!std::copy_constructible<tensor_type>);
static_assert(std::move_constructible<tensor_type>);
static_assert(std::convertible_to<typename mutable_span_type::reference, double>);
static_assert(std::assignable_from<typename mutable_span_type::reference, double>);

using complex_tensor_type = uni20::CudaTensor<uni20::complex<double>, 2>;
using conjugated_tensor_type = decltype(uni20::conj(std::declval<complex_tensor_type&>()));
using conjugated_device_span_type = decltype(std::declval<conjugated_tensor_type const&>().device_mdspan());

static_assert(uni20::DeviceTensorView<conjugated_tensor_type>);
static_assert(!uni20::TensorView<conjugated_tensor_type>);
static_assert(!HasConstMdspan<conjugated_tensor_type>);
static_assert(uni20::mdspan_needs_conjugation_v<conjugated_device_span_type>);
static_assert(
    std::same_as<typename conjugated_device_span_type::data_descriptor_type,
                 typename decltype(std::declval<complex_tensor_type const&>().device_mdspan())::data_descriptor_type>);

TEST(CudaPointerAccessorTest, AccessReturnsMappedElementReference)
{
  std::array<double, 3> values{1.0, 2.0, 3.0};
  uni20::cuda::CudaPointerAccessor<double> accessor;

  auto& reference = accessor.access(values.data(), 1);

  EXPECT_EQ(&reference, &values[1]);
  reference = 4.0;
  EXPECT_DOUBLE_EQ(values[1], 4.0);
  EXPECT_EQ(accessor.offset(values.data(), 2), values.data() + 2);
}

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

TEST_F(CudaTensorTest, ExplicitResourcesConstructionOwnsDeviceBufferAndDescriptor)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);

  EXPECT_EQ(tensor.rows(), 2);
  EXPECT_EQ(tensor.cols(), 3);
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.storage().size(), 6U);
  EXPECT_EQ(tensor.storage().device(), resources.device());

  auto span = tensor.device_mdspan();
  EXPECT_EQ(&span.data_descriptor().buffer(), &tensor.storage());
  EXPECT_EQ(span.data_descriptor().element_offset(), 0U);
  EXPECT_EQ(span.mapping()(1, 2), 5U);

  tensor_type const& const_tensor = tensor;
  auto const_span = const_tensor.device_mdspan();
  EXPECT_EQ(&const_span.data_descriptor().buffer(), &tensor.storage());
  EXPECT_EQ(const_span.mapping()(1, 2), 5U);
  EXPECT_EQ(tensor.backend_selector(), uni20::CudaStorage::backend_selector());
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

TEST_F(CudaTensorTest, BlockingAccessResolvesPointerMdspans)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  std::array<double, 6> const input{1, 2, 3, 4, 5, 6};
  std::array<double, 6> output{};

  {
    auto lease = uni20::blocking_write_access(tensor);
    static_assert(uni20::MutableTensorView<decltype(lease)>);
    EXPECT_EQ(lease.backend_selector(), tensor.backend_selector());
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);

    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(lease.mdspan().data_handle(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
                       "cudaMemcpy test host-to-device", resources.device().ordinal());
    uni20::cuda::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize test upload", resources.device().ordinal());
  }

  {
    auto lease = uni20::blocking_read_access(std::as_const(tensor));
    static_assert(uni20::TensorView<decltype(lease)>);
    static_assert(!uni20::MutableTensorView<decltype(lease)>);
    EXPECT_EQ(&lease.storage(), &tensor.storage());

    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "cudaMemcpy test device-to-host", resources.device().ordinal());
  }

  EXPECT_EQ(output, input);
}

TEST_F(CudaTensorTest, StreamOrderedAccessCanBeCoAwaitedAsTensorView)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::async::DebugCudaScheduler scheduler(resources.device());
  tensor_type tensor(resources, 2, 3);
  bool submitted = false;

  auto task = [](tensor_type& value, bool& observed) static -> uni20::async::CudaTask {
    auto stream = co_await uni20::cuda::acquire_stream(value.storage().resources().streams());
    auto lease = co_await uni20::write_access(value, stream);
    static_assert(uni20::MutableTensorView<decltype(lease)>);
    auto const size_bytes =
        static_cast<std::size_t>(lease.mdspan().mapping().required_span_size()) * sizeof(tensor_type::element_type);
    uni20::cuda::check(cudaMemsetAsync(lease.mdspan().data_handle(), 0, size_bytes, stream.native_handle()),
                       "cudaMemsetAsync tensor lease", stream.device());
    observed = true;
    co_return;
  }(tensor, submitted);

  scheduler.schedule(std::move(task), resources.device().ordinal());
  scheduler.run_all();
  EXPECT_TRUE(submitted);
  tensor.storage().synchronize();

  std::array<double, 6> output{};
  output.fill(1.0);
  auto lease = uni20::blocking_read_access(std::as_const(tensor));
  {
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "cudaMemcpy tensor lease verification", resources.device().ordinal());
  }
  EXPECT_EQ(output, (std::array<double, 6>{}));
}

TEST_F(CudaTensorTest, DeferredAcquisitionCapturesEachDeviceSpanOnce)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  DeviceSpanCallCounts calls;
  CountingCudaTensorView view(tensor, calls);

  {
    auto lease = uni20::blocking_read_access(std::as_const(view));
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 0);

  calls = {};
  {
    auto lease = uni20::blocking_write_access(view);
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 1);

  auto stream = resources.streams().acquire();
  calls = {};
  {
    auto acquisition = uni20::read_access(std::as_const(view), stream);
    auto lease = acquisition.await_resume();
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 0);

  calls = {};
  {
    auto acquisition = uni20::write_access(view, stream);
    auto lease = acquisition.await_resume();
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 1);
}

TEST_F(CudaTensorTest, OwningRvalueReadLeaseMovesTheBuffer)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  std::array<double, 6> const input{1, 2, 3, 4, 5, 6};
  std::array<double, 6> output{};

  {
    auto lease = uni20::blocking_write_access(tensor);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(lease.mdspan().data_handle(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
                       "cudaMemcpy owning read test upload", resources.device().ordinal());
  }

  auto lease = uni20::blocking_read_access(std::move(tensor));
  auto moved_lease = std::move(lease);
  {
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(
        cudaMemcpy(output.data(), moved_lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
        "cudaMemcpy owning read test download", resources.device().ordinal());
  }

  EXPECT_EQ(output, input);
}

TEST_F(CudaTensorTest, OwningRvalueStreamReadPublishesCompletion)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  std::array<double, 6> const input{1, 2, 3, 4, 5, 6};
  std::array<double, 6> output{};

  {
    auto lease = uni20::blocking_write_access(tensor);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(lease.mdspan().data_handle(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
                       "cudaMemcpy owning stream read test upload", resources.device().ordinal());
  }

  auto stream = resources.streams().acquire();
  {
    auto acquisition = uni20::read_access(std::move(tensor), stream);
    auto lease = acquisition.await_resume();
    uni20::cuda::check(cudaMemcpyAsync(output.data(), lease.mdspan().data_handle(), sizeof(output),
                                       cudaMemcpyDeviceToHost, stream.native_handle()),
                       "cudaMemcpyAsync owning stream read test download", resources.device().ordinal());
  }

  EXPECT_EQ(output, input);
}

TEST_F(CudaTensorTest, DefaultConstructionUsesInstalledRuntimeResources)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::CudaTensor<double, 2> tensor(3, 4);

  EXPECT_EQ(tensor.rows(), 3);
  EXPECT_EQ(tensor.cols(), 4);
  EXPECT_EQ(&tensor.storage().resources(), &runtime.default_device_resources());
}

} // namespace
