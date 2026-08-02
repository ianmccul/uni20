#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/mdspan/generated_layout.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace
{

using host_matrix_type = uni20::Tensor<double, 2>;
using cuda_matrix_type = uni20::CudaTensor<double, 2>;
using async_cuda_matrix_type = uni20::async::Async<cuda_matrix_type>;
using row_major_host_matrix_type = uni20::RowMajorTensor<double, 2>;
using complex_type = uni20::complex<double>;
using complex_host_matrix_type = uni20::Tensor<complex_type, 2>;
using complex_cuda_matrix_type = uni20::CudaTensor<complex_type, 2>;
using cuda_vector_type = uni20::CudaTensor<double, 1>;
using complex_cuda_conjugated_view = uni20::ConjugatedTensorView<complex_cuda_matrix_type>;
using cuda_extents_type = stdex::dextents<uni20::index_type, 2>;
using resolved_cuda_output_mdspan =
    stdex::mdspan<double, cuda_extents_type, stdex::layout_left, uni20::cuda::CudaPointerAccessor<double>>;
using resolved_cuda_input_mdspan =
    stdex::mdspan<double const, cuda_extents_type, stdex::layout_left, uni20::cuda::CudaPointerAccessor<double const>>;
using resolved_complex_cuda_output_mdspan =
    stdex::mdspan<complex_type, cuda_extents_type, stdex::layout_left, uni20::cuda::CudaPointerAccessor<complex_type>>;
using resolved_complex_cuda_input_mdspan = stdex::mdspan<complex_type const, cuda_extents_type, stdex::layout_left,
                                                         uni20::cuda::CudaPointerAccessor<complex_type const>>;
using conjugated_cuda_descriptor = decltype(std::declval<complex_cuda_conjugated_view const&>().mdspec());
using conjugated_cuda_lease =
    decltype(uni20::acquire_cuda_read_access_sync(std::declval<conjugated_cuda_descriptor const&>()));
using resolved_conjugated_cuda_mdspan = std::remove_cvref_t<decltype(std::declval<conjugated_cuda_lease&>().mdspan())>;

using namespace std::chrono_literals;

template <class Layout, class Mdspec>
auto remap_mdspec(Mdspec const& source, typename Layout::template mapping<typename Mdspec::extents_type> mapping)
{
  using source_type = std::remove_cvref_t<Mdspec>;
  using result_type = uni20::mdspec<typename source_type::element_type, typename source_type::extents_type, Layout,
                                    typename source_type::accessor_type, typename source_type::data_descriptor_type>;
  return result_type{source.data_descriptor(), std::move(mapping), source.accessor()};
}

template <class Tensor>
auto make_strided_matrix_mdspec(Tensor& tensor, std::size_t offset, std::array<uni20::index_type, 2> const& strides)
{
  auto base = tensor.mdspec();
  using base_type = decltype(base);
  using layout_type = stdex::layout_stride;
  using mapping_type = layout_type::mapping<cuda_extents_type>;
  using result_type = uni20::mdspec<typename base_type::element_type, cuda_extents_type, layout_type,
                                    typename base_type::accessor_type, typename base_type::data_descriptor_type>;
  return result_type{base.data_descriptor().offset_by(offset), mapping_type{cuda_extents_type{2, 3}, strides},
                     base.accessor()};
}

template <class Tensor> auto make_vector_mdspec(Tensor& tensor, std::size_t offset, uni20::index_type extent)
{
  auto base = tensor.mdspec();
  using base_type = decltype(base);
  using mapping_type = typename base_type::mapping_type;
  return base_type{base.data_descriptor().offset_by(offset), mapping_type{typename base_type::extents_type{extent}},
                   base.accessor()};
}

struct DescriptorSelectedStoragePolicy
{
    [[nodiscard]] static constexpr auto backend_selector() noexcept
    {
      return uni20::linalg::backend_list<uni20::linalg::CpuReferenceBackend>{uni20::linalg::CpuReferenceBackend{}};
    }
};

template <std::size_t Rank> struct StrideProbe
{
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return Rank; }
    [[nodiscard]] constexpr uni20::index_type extent(std::size_t axis) const noexcept { return extents_[axis]; }
    [[nodiscard]] constexpr uni20::index_type stride(std::size_t axis) const noexcept { return strides_[axis]; }

    std::array<uni20::index_type, Rank> extents_{};
    std::array<uni20::index_type, Rank> strides_{};
};

class CudaDescriptorMatrixView {
  public:
    using storage_policy = DescriptorSelectedStoragePolicy;
    using extents_type = typename cuda_matrix_type::extents_type;

    explicit CudaDescriptorMatrixView(cuda_matrix_type const& tensor) : tensor_(&tensor) {}

    [[nodiscard]] auto mdspec() const { return tensor_->mdspec(); }

    [[nodiscard]] static constexpr auto backend_selector() noexcept { return storage_policy::backend_selector(); }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return tensor_->extents(); }

    [[nodiscard]] auto extent(std::size_t axis) const noexcept { return tensor_->extent(axis); }

  private:
    cuda_matrix_type const* tensor_;
};

template <class AsyncTensor>
concept CanPreserveAsyncDenseDecompositions = requires(AsyncTensor const& matrix) {
  uni20::linalg::singular_values(matrix);
  uni20::linalg::svd_left(matrix);
  uni20::linalg::svd_right(matrix);
  uni20::linalg::svd(matrix);
  uni20::linalg::eigh(matrix);
  uni20::linalg::truncated_svd(matrix);
};

template <class AsyncTensor>
concept CanConsumeAsyncDenseDecompositions = requires(AsyncTensor&& matrix) {
  uni20::linalg::singular_values(std::move(matrix));
  uni20::linalg::svd_left(std::move(matrix));
  uni20::linalg::svd_right(std::move(matrix));
  uni20::linalg::svd(std::move(matrix));
  uni20::linalg::eigh(std::move(matrix));
  uni20::linalg::truncated_svd(std::move(matrix));
};

template <class AsyncTensor>
concept CanFormDeferredAsyncReshape = requires(AsyncTensor& tensor) { uni20::async::reshape_view(tensor, 4); };

static_assert(uni20::TensorView<CudaDescriptorMatrixView>);
static_assert(CanPreserveAsyncDenseDecompositions<async_cuda_matrix_type>);
static_assert(CanConsumeAsyncDenseDecompositions<async_cuda_matrix_type>);
static_assert(CanFormDeferredAsyncReshape<async_cuda_matrix_type>);
static_assert(!std::same_as<typename CudaDescriptorMatrixView::storage_policy, uni20::CudaStorage>);
static_assert(uni20::cuda::BufferMdspec<uni20::tensor_mdspec_t<CudaDescriptorMatrixView>>);
static_assert(uni20::CudaAccessibleAccessor<uni20::cuda::CudaPointerAccessor<double>>);
static_assert(!uni20::HostAccessibleAccessor<uni20::cuda::CudaPointerAccessor<double>>);
static_assert(uni20::CudaAccessibleMdspan<resolved_cuda_output_mdspan>);
static_assert(uni20::CudaAccessibleMdspan<resolved_cuda_input_mdspan>);
static_assert(uni20::MutableMdspanLike<resolved_complex_cuda_output_mdspan>);
static_assert(uni20::CudaAccessibleMdspan<resolved_complex_cuda_input_mdspan>);
static_assert(std::same_as<uni20::remove_proxy_reference_t<typename resolved_complex_cuda_output_mdspan::reference>,
                           complex_type>);
static_assert(
    std::same_as<uni20::logical_value_t<typename resolved_complex_cuda_input_mdspan::reference>, complex_type>);
static_assert(std::same_as<typename resolved_conjugated_cuda_mdspan::accessor_type,
                           uni20::cuda::CudaConjugatingPointerAccessor<double>>);
static_assert(!uni20::cuda::BufferMdspec<resolved_cuda_output_mdspan>);
static_assert(!uni20::linalg::detail::cuda_reference::SupportedCopyMdspans<resolved_cuda_output_mdspan,
                                                                           resolved_cuda_input_mdspan>);
using generated_cuda_descriptor =
    uni20::mdspec<double const, cuda_extents_type, uni20::GeneratedLayout,
                  uni20::cuda::CudaPointerAccessor<double const>, uni20::cuda::CudaBufferView<double const>>;
static_assert(uni20::CudaAccessibleMdspec<generated_cuda_descriptor>);
static_assert(!uni20::linalg::detail::cuda_reference::SupportedCopyMdspans<resolved_cuda_output_mdspan,
                                                                           generated_cuda_descriptor>);

struct CopyGate
{
    std::atomic<bool> open = false;
    std::atomic<bool> entered = false;
};

void CUDART_CB wait_for_copy_gate(void* raw_gate)
{
  auto& gate = *static_cast<CopyGate*>(raw_gate);
  gate.entered.store(true, std::memory_order_release);
  while (!gate.open.load(std::memory_order_acquire))
  {
    std::this_thread::yield();
  }
}

void CUDART_CB set_copy_flag(void* raw_flag)
{
  static_cast<std::atomic<bool>*>(raw_flag)->store(true, std::memory_order_release);
}

bool wait_until(std::atomic<bool> const& flag)
{
  auto const deadline = std::chrono::steady_clock::now() + 5s;
  while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::yield();
  }
  return flag.load(std::memory_order_acquire);
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

class CudaCopyTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    static host_matrix_type make_matrix()
    {
      host_matrix_type matrix(2, 3);
      matrix[0, 0] = 1.0;
      matrix[1, 0] = 2.0;
      matrix[0, 1] = 3.0;
      matrix[1, 1] = 4.0;
      matrix[0, 2] = 5.0;
      matrix[1, 2] = 6.0;
      return matrix;
    }

    static void expect_matrix(host_matrix_type const& matrix)
    {
      ASSERT_EQ(matrix.rows(), 2);
      ASSERT_EQ(matrix.cols(), 3);
      EXPECT_DOUBLE_EQ((matrix[0, 0]), 1.0);
      EXPECT_DOUBLE_EQ((matrix[1, 0]), 2.0);
      EXPECT_DOUBLE_EQ((matrix[0, 1]), 3.0);
      EXPECT_DOUBLE_EQ((matrix[1, 1]), 4.0);
      EXPECT_DOUBLE_EQ((matrix[0, 2]), 5.0);
      EXPECT_DOUBLE_EQ((matrix[1, 2]), 6.0);
    }

    int device_count_ = 0;
};

TEST(CudaCopyPlanningTest, ElementwiseLayoutDecodesIndependentPaddedStrides)
{
  using mapping_type = stdex::layout_stride::mapping<cuda_extents_type>;
  cuda_extents_type const extents{2, 3};
  std::array<uni20::index_type, 2> const output_strides{1, 3};
  std::array<uni20::index_type, 2> const input_strides{1, 2};
  double output_storage[8]{};
  double input_storage[6]{};
  stdex::mdspan output{output_storage, mapping_type{extents, output_strides}};
  stdex::mdspan input{input_storage, mapping_type{extents, input_strides}};
  uni20::linalg::detail::cuda_reference::ElementwiseCopyLayout layout;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_layout(output, input, layout));
  EXPECT_EQ(layout.rank, 2);
  EXPECT_EQ(layout.element_count, 6);
  EXPECT_EQ(layout.offsets(0).output, 0);
  EXPECT_EQ(layout.offsets(0).input, 0);
  EXPECT_EQ(layout.offsets(1).output, 3);
  EXPECT_EQ(layout.offsets(1).input, 2);
  EXPECT_EQ(layout.offsets(2).output, 6);
  EXPECT_EQ(layout.offsets(2).input, 4);
  EXPECT_EQ(layout.offsets(3).output, 1);
  EXPECT_EQ(layout.offsets(3).input, 1);
  EXPECT_EQ(layout.offsets(5).output, 7);
  EXPECT_EQ(layout.offsets(5).input, 5);
}

TEST(CudaCopyPlanningTest, NonpositiveActiveStridesDecline)
{
  using uni20::linalg::detail::cuda_reference::active_strides_are_positive;

  EXPECT_TRUE(active_strides_are_positive(StrideProbe<1>{.extents_ = {3}, .strides_ = {1}}));
  EXPECT_FALSE(active_strides_are_positive(StrideProbe<1>{.extents_ = {3}, .strides_ = {0}}));
  EXPECT_FALSE(active_strides_are_positive(StrideProbe<1>{.extents_ = {3}, .strides_ = {-1}}));
  EXPECT_TRUE(active_strides_are_positive(StrideProbe<1>{.extents_ = {1}, .strides_ = {-1}}));
  EXPECT_TRUE(active_strides_are_positive(StrideProbe<1>{.extents_ = {0}, .strides_ = {-1}}));
  EXPECT_TRUE(active_strides_are_positive(StrideProbe<2>{.extents_ = {0, 3}, .strides_ = {1, -1}}));
}

TEST_F(CudaCopyTest, PageableHostRoundTripUsesExplicitTransferFunctions)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto host = make_matrix();

  auto device = uni20::to_device(host, 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ(device.storage().device().ordinal(), 0);
  expect_matrix(result);
}

TEST_F(CudaCopyTest, FixedOutputPageableTransfersResizeAndRoundTrip)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = make_matrix();
  cuda_matrix_type device(runtime.device_resources(0), 1, 1);
  host_matrix_type result(1, 1);

  uni20::copy(device, source);
  uni20::copy(result, device);

  EXPECT_EQ(device.rows(), source.rows());
  EXPECT_EQ(device.cols(), source.cols());
  expect_matrix(result);
}

TEST_F(CudaCopyTest, ExplicitCudaBackendUsesDescriptorRatherThanStoragePolicy)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto device = uni20::to_device(make_matrix(), 0);
  CudaDescriptorMatrixView input(device);
  host_matrix_type result;

  uni20::copy(uni20::linalg::CudaReferenceBackend{}, result, input);

  expect_matrix(result);
}

TEST_F(CudaCopyTest, DeviceToHostCopyObservesConjugatingAccessor)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device = uni20::to_device(source, 0);
  auto result = uni20::to_host(uni20::conj(device));

  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, DeviceToDeviceCopyExecutesConjugatingAccessorKernel)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device_source = uni20::to_device(source, 0);
  complex_cuda_matrix_type device_output(runtime.device_resources(0), 2, 2);
  auto conjugated = uni20::conj(device_source);
  auto output_span = device_output.mdspec();
  auto input_span = conjugated.mdspec();
  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output_span, input_span);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_EQ(preparation.execution, uni20::linalg::detail::cuda_reference::CopyExecution::elementwise_kernel);

  uni20::copy(device_output, conjugated);

  auto result = uni20::to_host(device_output);
  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, SameBufferConjugatingCopyDeclinesWithoutMutation)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device = uni20::to_device(source, 0);
  auto conjugated = uni20::conj(device);
  auto output_span = device.mdspec();
  auto input_span = conjugated.mdspec();

  EXPECT_EQ(uni20::linalg::detail::cuda_reference::copy(output_span, input_span),
            uni20::linalg::KernelAttempt::unsupported_instance);
  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(uni20::linalg::CudaReferenceBackend{}, uni20::linalg::copy_op{},
                                                  output_span, input_span));
  auto same_input_span = uni20::mdspec_of(std::as_const(device));
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(uni20::linalg::CudaReferenceBackend{}, uni20::linalg::copy_op{},
                                                 output_span, same_input_span));

  auto result = uni20::to_host(device);
  EXPECT_EQ((result[0, 0]), (complex_type{1.0, 2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, 4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, -6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, -8.0}));
}

TEST_F(CudaCopyTest, EmptySameBufferConjugatingCopySucceeds)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  complex_cuda_matrix_type device(runtime.device_resources(0), 1, 1);
  auto conjugated = uni20::conj(device);
  auto output_base = device.mdspec();
  auto input_base = conjugated.mdspec();
  using output_type = decltype(output_base);
  using input_type = decltype(input_base);
  typename output_type::mapping_type output_mapping{typename output_type::extents_type{0, 1}};
  typename input_type::mapping_type input_mapping{typename input_type::extents_type{0, 1}};
  output_type output{output_base.data_descriptor(), output_mapping, output_base.accessor()};
  input_type input{input_base.data_descriptor(), input_mapping, input_base.accessor()};

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output, input);

  EXPECT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_FALSE(preparation.has_work);
  EXPECT_EQ(uni20::linalg::detail::cuda_reference::copy(output, input), uni20::linalg::KernelAttempt::success);
}

TEST_F(CudaCopyTest, EmptySameBufferDifferentOffsetCopySucceeds)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  cuda_matrix_type device(runtime.device_resources(0), 1, 2);
  auto output_base = device.mdspec();
  auto input_base = std::as_const(device).mdspec();
  using output_type = decltype(output_base);
  using input_type = decltype(input_base);
  typename output_type::mapping_type output_mapping{typename output_type::extents_type{0, 1}};
  typename input_type::mapping_type input_mapping{typename input_type::extents_type{0, 1}};
  output_type output{output_base.data_descriptor(), output_mapping, output_base.accessor()};
  input_type input{input_base.data_descriptor().offset_by(1), input_mapping, input_base.accessor()};

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output, input);

  EXPECT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_FALSE(preparation.has_work);
  EXPECT_EQ(uni20::linalg::detail::cuda_reference::copy(output, input), uni20::linalg::KernelAttempt::success);
}

TEST_F(CudaCopyTest, DisjointSameBufferCopyUsesOneAccessState)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::Tensor<double, 1> host(6);
  host[0] = 1.0;
  host[1] = 2.0;
  host[2] = 3.0;
  host[3] = -1.0;
  host[4] = -1.0;
  host[5] = -1.0;
  auto device = uni20::to_device(host, 0);
  auto output = make_vector_mdspec(device, 3, 3);
  auto input = make_vector_mdspec(std::as_const(device), 0, 3);

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output, input);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  ASSERT_TRUE(preparation.has_work);
  ASSERT_EQ(preparation.output_buffer, preparation.input_buffer);
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(output, input), uni20::linalg::KernelAttempt::success);

  auto result = uni20::to_host(device);
  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 3.0);
  EXPECT_DOUBLE_EQ(result[3], 1.0);
  EXPECT_DOUBLE_EQ(result[4], 2.0);
  EXPECT_DOUBLE_EQ(result[5], 3.0);
}

TEST_F(CudaCopyTest, HostToDeviceCopyObservesConjugatingAccessor)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device = uni20::to_device(uni20::conj(source), 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, SameDeviceCopyUsesCudaReferenceFallback)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(0), 2, 3);

  uni20::copy(destination, source);

  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, SameDeviceCopyConvertsBetweenColumnMajorAndRowMajorMappings)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type row_major_storage(runtime.device_resources(0), 2, 3);
  cuda_matrix_type roundtrip(runtime.device_resources(0), 2, 3);

  auto source_span = std::as_const(source).mdspec();
  auto row_major_base = row_major_storage.mdspec();
  using row_major_mapping = stdex::layout_right::mapping<cuda_extents_type>;
  auto row_major_output = remap_mdspec<stdex::layout_right>(row_major_base, row_major_mapping{cuda_extents_type{2, 3}});

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(row_major_output, source_span);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_EQ(preparation.execution, uni20::linalg::detail::cuda_reference::CopyExecution::elementwise_kernel);
  EXPECT_EQ(preparation.elementwise_layout.output_strides[0], 3);
  EXPECT_EQ(preparation.elementwise_layout.output_strides[1], 1);
  EXPECT_EQ(preparation.elementwise_layout.input_strides[0], 1);
  EXPECT_EQ(preparation.elementwise_layout.input_strides[1], 2);
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(row_major_output, source_span),
            uni20::linalg::KernelAttempt::success);

  auto row_major_input_base = std::as_const(row_major_storage).mdspec();
  auto row_major_input =
      remap_mdspec<stdex::layout_right>(row_major_input_base, row_major_mapping{cuda_extents_type{2, 3}});
  auto roundtrip_output = roundtrip.mdspec();
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(roundtrip_output, row_major_input),
            uni20::linalg::KernelAttempt::success);

  expect_matrix(uni20::to_host(roundtrip));
}

TEST_F(CudaCopyTest, SameDeviceCopyHandlesPaddedStridesAndBufferOffsets)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_vector_type padded_storage(runtime.device_resources(0), 10);
  cuda_matrix_type roundtrip(runtime.device_resources(0), 2, 3);
  std::array<uni20::index_type, 2> const padded_strides{1, 3};

  auto source_span = std::as_const(source).mdspec();
  auto padded_output = make_strided_matrix_mdspec(padded_storage, 1, padded_strides);
  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(padded_output, source_span);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_EQ(preparation.execution, uni20::linalg::detail::cuda_reference::CopyExecution::elementwise_kernel);
  EXPECT_EQ(preparation.output_offset, 1);
  EXPECT_EQ(preparation.output_span_size, 8);
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(padded_output, source_span),
            uni20::linalg::KernelAttempt::success);

  auto padded_input = make_strided_matrix_mdspec(std::as_const(padded_storage), 1, padded_strides);
  auto roundtrip_output = roundtrip.mdspec();
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(roundtrip_output, padded_input),
            uni20::linalg::KernelAttempt::success);

  expect_matrix(uni20::to_host(roundtrip));
}

TEST_F(CudaCopyTest, SameDeviceCopyOrdersPendingSourceAndDestinationReuse)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 3});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(0), 2, 3);
  CopyGate gate;

  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto source_predecessor = source.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_copy_gate, &gate),
                       "cudaLaunchHostFunc same-device source predecessor", 0);
  }

  uni20::copy(destination, source);

  std::atomic<bool> destination_reader_completed = false;
  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto destination_reader = destination.storage().read_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &destination_reader_completed),
                       "cudaLaunchHostFunc same-device destination reader", 0);
  }

  bool const predecessor_entered = wait_until(gate.entered);
  if (!predecessor_entered) gate.open.store(true, std::memory_order_release);
  ASSERT_TRUE(predecessor_entered);
  EXPECT_FALSE(destination_reader_completed.load(std::memory_order_acquire));

  gate.open.store(true, std::memory_order_release);
  runtime.device_resources(0).streams().synchronize();
  EXPECT_TRUE(destination_reader_completed.load(std::memory_order_acquire));
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, RowMajorRoundTripPreservesLogicalOrder)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  row_major_host_matrix_type host(2, 3);
  host[0, 0] = 1.0;
  host[0, 1] = 2.0;
  host[0, 2] = 3.0;
  host[1, 0] = 4.0;
  host[1, 1] = 5.0;
  host[1, 2] = 6.0;

  auto result = uni20::to_host(uni20::to_device(host, 0));

  static_assert(std::same_as<typename decltype(result)::layout_type, stdex::layout_right>);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((result[0, 2]), 3.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 4.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 5.0);
  EXPECT_DOUBLE_EQ((result[1, 2]), 6.0);
}

TEST_F(CudaCopyTest, ZeroExtentRoundTripIsAnEmptySuccessfulTransfer)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  host_matrix_type host(3, 0);

  auto device = uni20::to_device(host, 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ(device.rows(), 3);
  EXPECT_EQ(device.cols(), 0);
  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 0);
}

TEST_F(CudaCopyTest, PeerCopyPreservesValues)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);

  auto destination = uni20::to_device(source, 1);

  EXPECT_EQ(destination.storage().device().ordinal(), 1);
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, PeerCopyPreservesSourceAndDestinationLedgers)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 3});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(1), 2, 3);
  CopyGate gate;

  {
    auto stream = runtime.device_resources(1).streams().acquire();
    auto destination_predecessor = destination.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(1);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), wait_for_copy_gate, &gate),
                       "cudaLaunchHostFunc peer destination predecessor", 1);
  }

  uni20::copy(destination, source);

  std::atomic<bool> source_writer_completed = false;
  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto source_writer = source.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &source_writer_completed),
                       "cudaLaunchHostFunc peer source writer", 0);
  }

  std::atomic<bool> destination_reader_completed = false;
  {
    auto stream = runtime.device_resources(1).streams().acquire();
    auto destination_reader = destination.storage().read_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(1);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &destination_reader_completed),
                       "cudaLaunchHostFunc peer destination reader", 1);
  }

  bool const predecessor_entered = wait_until(gate.entered);
  if (!predecessor_entered) gate.open.store(true, std::memory_order_release);
  ASSERT_TRUE(predecessor_entered);
  EXPECT_FALSE(source_writer_completed.load(std::memory_order_acquire));
  EXPECT_FALSE(destination_reader_completed.load(std::memory_order_acquire));

  gate.open.store(true, std::memory_order_release);
  runtime.device_resources(0).streams().synchronize();
  runtime.device_resources(1).streams().synchronize();
  EXPECT_TRUE(source_writer_completed.load(std::memory_order_acquire));
  EXPECT_TRUE(destination_reader_completed.load(std::memory_order_acquire));
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, AsyncCopyConstructsOutputAndCarriesDeviceCompletion)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);
  uni20::async::Async<cuda_matrix_type> output;

  uni20::copy(output, input);

  auto const& device_result = output.get_wait(scheduler);
  expect_matrix(uni20::to_host(device_result));
}

TEST_F(CudaCopyTest, AsyncConjugatingCopyRunsElementwiseKernel)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};
  uni20::async::Async<complex_cuda_matrix_type> input = uni20::to_device(source, 0);
  auto conjugated = uni20::async::conj(input);
  uni20::async::Async<complex_cuda_matrix_type> output;

  uni20::copy(output, conjugated);

  auto result = uni20::to_host(output.get_wait(scheduler));
  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, AsyncPeerCopyRunsOnDestinationDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "async peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);
  uni20::async::Async<cuda_matrix_type> output = cuda_matrix_type(runtime.device_resources(1), 2, 3);

  uni20::copy(output, input);

  auto const& device_result = output.get_wait(scheduler);
  EXPECT_EQ(device_result.storage().device().ordinal(), 1);
  expect_matrix(uni20::to_host(device_result));
}

TEST_F(CudaCopyTest, AsyncCopyRejectsOutputInputEpochAliasing)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::Async<cuda_matrix_type> tensor = uni20::to_device(make_matrix(), 0);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::copy(tensor, tensor), std::runtime_error);
}

} // namespace
