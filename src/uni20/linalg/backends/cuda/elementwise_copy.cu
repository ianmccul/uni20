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

template <class OutputAccessor, class InputAccessor>
__global__ void elementwise_copy_kernel(typename OutputAccessor::data_handle_type output,
                                        typename InputAccessor::data_handle_type input, ElementwiseCopyLayout layout)
{
  OutputAccessor output_accessor{};
  InputAccessor input_accessor{};
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = blockDim.x * gridDim.x;
  while (index < layout.element_count)
  {
    auto const offsets = layout.offsets(index);
    output_accessor.access(output, offsets.output) = input_accessor.access(input, offsets.input);
    index += stride;
  }
}

template <class Scalar, class InputAccessor>
void launch_elementwise_copy(Scalar* output, Scalar const* input, ElementwiseCopyLayout const& layout,
                             cudaStream_t stream, int device)
{
  if (layout.element_count == 0) return;

  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  constexpr unsigned int threads = 256;
  auto required_blocks = (layout.element_count + threads - 1) / threads;
  constexpr std::size_t maximum_blocks = 65535;
  auto const blocks = static_cast<unsigned int>(required_blocks < maximum_blocks ? required_blocks : maximum_blocks);
  elementwise_copy_kernel<output_accessor, InputAccessor><<<blocks, threads, 0, stream>>>(output, input, layout);
  uni20::cuda::check(cudaGetLastError(), "launch CUDA reference elementwise copy", device);
}

template <class Scalar>
void enqueue_elementwise_copy_impl(Scalar* output, Scalar const* input, ElementwiseCopyLayout const& layout,
                                   ElementwiseCopyTransform transform, cudaStream_t stream, int device)
{
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  if constexpr (uni20::Complex<Scalar>)
  {
    if (transform == ElementwiseCopyTransform::conjugate)
    {
      using real_type = uni20::make_real_t<Scalar>;
      using conjugating_accessor = uni20::cuda::CudaConjugatingPointerAccessor<real_type>;
      launch_elementwise_copy<Scalar, conjugating_accessor>(output, input, layout, stream, device);
      return;
    }
  }

  launch_elementwise_copy<Scalar, input_accessor>(output, input, layout, stream, device);
}

} // namespace

void enqueue_elementwise_copy(float* output, float const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device)
{
  enqueue_elementwise_copy_impl(output, input, layout, transform, stream, device);
}

void enqueue_elementwise_copy(double* output, double const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device)
{
  enqueue_elementwise_copy_impl(output, input, layout, transform, stream, device);
}

void enqueue_elementwise_copy(uni20::cfloat* output, uni20::cfloat const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device)
{
  enqueue_elementwise_copy_impl(output, input, layout, transform, stream, device);
}

void enqueue_elementwise_copy(uni20::cdouble* output, uni20::cdouble const* input, ElementwiseCopyLayout const& layout,
                              ElementwiseCopyTransform transform, cudaStream_t stream, int device)
{
  enqueue_elementwise_copy_impl(output, input, layout, transform, stream, device);
}

} // namespace uni20::linalg::detail::cuda_reference
