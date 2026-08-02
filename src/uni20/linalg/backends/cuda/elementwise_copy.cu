#include "elementwise_copy.hpp"

#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/storage/cuda_accessor.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{
namespace
{

template <class Plan, class OutputAccessor, class InputAccessor>
__global__ void elementwise_copy_kernel(typename OutputAccessor::data_handle_type output,
                                        typename InputAccessor::data_handle_type input, Plan plan)
{
  OutputAccessor output_accessor{};
  InputAccessor input_accessor{};
  using logical_index_type = typename Plan::logical_index_type;
  using output_offset_type = typename OutputAccessor::offset_type;
  using input_offset_type = typename InputAccessor::offset_type;
  auto index = static_cast<logical_index_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const grid_stride = static_cast<logical_index_type>(blockDim.x * gridDim.x);
  while (index < plan.element_count)
  {
    auto const offsets = plan.offsets(index);
    output_accessor.access(output, static_cast<output_offset_type>(offsets[0])) =
        input_accessor.access(input, static_cast<input_offset_type>(offsets[1]));
    index += grid_stride;
  }
}

template <class Scalar, class InputAccessor, class Plan>
void launch_elementwise_copy(Scalar* output, Scalar const* input, Plan const& plan, cudaStream_t stream, int device)
{
  if (plan.element_count == 0) return;

  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  constexpr unsigned int threads = 256;
  auto required_blocks = (plan.element_count + threads - 1) / threads;
  constexpr std::size_t maximum_blocks = 65535;
  auto const blocks = static_cast<unsigned int>(required_blocks < maximum_blocks ? required_blocks : maximum_blocks);
  elementwise_copy_kernel<Plan, output_accessor, InputAccessor><<<blocks, threads, 0, stream>>>(output, input, plan);
  uni20::cuda::check(cudaGetLastError(), "launch CUDA reference elementwise copy", device);
}

template <class Scalar, class Plan>
void enqueue_elementwise_copy_impl(Scalar* output, Scalar const* input, Plan const& plan,
                                   ElementwiseCopyTransform transform, cudaStream_t stream, int device)
{
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  if constexpr (uni20::Complex<Scalar>)
  {
    if (transform == ElementwiseCopyTransform::conjugate)
    {
      using real_type = uni20::make_real_t<Scalar>;
      using conjugating_accessor = uni20::cuda::CudaConjugatingPointerAccessor<real_type>;
      launch_elementwise_copy<Scalar, conjugating_accessor>(output, input, plan, stream, device);
      return;
    }
  }

  launch_elementwise_copy<Scalar, input_accessor>(output, input, plan, stream, device);
}

} // namespace

#define UNI20_DEFINE_ELEMENTWISE_COPY(Scalar)                                                                          \
  void enqueue_elementwise_copy(Scalar* output, Scalar const* input, ElementwiseCopyPlan32 const& plan,                \
                                ElementwiseCopyTransform transform, cudaStream_t stream, int device)                   \
  {                                                                                                                    \
    enqueue_elementwise_copy_impl(output, input, plan, transform, stream, device);                                     \
  }                                                                                                                    \
                                                                                                                       \
  void enqueue_elementwise_copy(Scalar* output, Scalar const* input, ElementwiseCopyPlan64 const& plan,                \
                                ElementwiseCopyTransform transform, cudaStream_t stream, int device)                   \
  {                                                                                                                    \
    enqueue_elementwise_copy_impl(output, input, plan, transform, stream, device);                                     \
  }

UNI20_DEFINE_ELEMENTWISE_COPY(float)
UNI20_DEFINE_ELEMENTWISE_COPY(double)
UNI20_DEFINE_ELEMENTWISE_COPY(uni20::cfloat)
UNI20_DEFINE_ELEMENTWISE_COPY(uni20::cdouble)

#undef UNI20_DEFINE_ELEMENTWISE_COPY

} // namespace uni20::linalg::detail::cuda_reference
