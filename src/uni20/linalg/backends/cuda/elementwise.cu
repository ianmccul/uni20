#include "elementwise_conjugate.hpp"
#include "elementwise_copy.hpp"
#include "elementwise_kernel.cuh"
#include "elementwise_negate.hpp"

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/storage/cuda_accessor.hpp>

#include <concepts>
#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{
namespace
{

struct AssignElement
{
    template <class Value> [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Value value) const
    {
      return value;
    }
};

template <class Scalar, class InputAccessor, class Plan>
void launch_elementwise_copy(Scalar* output, Scalar const* input, Plan const& plan, cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  launch_elementwise_kernel<false>(plan, AssignElement{}, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference elementwise copy",
                                   ElementwiseOperand{input, InputAccessor{}});
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

template <class Scalar, class Plan>
void enqueue_elementwise_conjugate_impl(Scalar* output, Plan const& plan, cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  launch_elementwise_kernel<true>(plan, uni20::cuda::CudaConjugate{}, ElementwiseOperand{output, output_accessor{}},
                                  stream, device, "launch CUDA reference elementwise conjugation");
}

template <class Scalar, class Plan>
void enqueue_elementwise_negate_impl(Scalar* output, Scalar const* input, Plan const& plan, cudaStream_t stream,
                                     int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  launch_elementwise_kernel<false>(plan, uni20::linalg::negate{}, ElementwiseOperand{output, output_accessor{}}, stream,
                                   device, "launch CUDA reference elementwise negation",
                                   ElementwiseOperand{input, input_accessor{}});
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

#define UNI20_DEFINE_ELEMENTWISE_CONJUGATE(Scalar)                                                                     \
  void enqueue_elementwise_conjugate(Scalar* output, ElementwiseConjugatePlan32 const& plan, cudaStream_t stream,      \
                                     int device)                                                                       \
  {                                                                                                                    \
    enqueue_elementwise_conjugate_impl(output, plan, stream, device);                                                  \
  }                                                                                                                    \
                                                                                                                       \
  void enqueue_elementwise_conjugate(Scalar* output, ElementwiseConjugatePlan64 const& plan, cudaStream_t stream,      \
                                     int device)                                                                       \
  {                                                                                                                    \
    enqueue_elementwise_conjugate_impl(output, plan, stream, device);                                                  \
  }

UNI20_DEFINE_ELEMENTWISE_CONJUGATE(uni20::cfloat)
UNI20_DEFINE_ELEMENTWISE_CONJUGATE(uni20::cdouble)

#undef UNI20_DEFINE_ELEMENTWISE_CONJUGATE

#define UNI20_DEFINE_ELEMENTWISE_NEGATE(Scalar)                                                                        \
  void enqueue_elementwise_negate(Scalar* output, Scalar const* input, ElementwiseNegatePlan32 const& plan,            \
                                  cudaStream_t stream, int device)                                                     \
  {                                                                                                                    \
    enqueue_elementwise_negate_impl(output, input, plan, stream, device);                                              \
  }                                                                                                                    \
                                                                                                                       \
  void enqueue_elementwise_negate(Scalar* output, Scalar const* input, ElementwiseNegatePlan64 const& plan,            \
                                  cudaStream_t stream, int device)                                                     \
  {                                                                                                                    \
    enqueue_elementwise_negate_impl(output, input, plan, stream, device);                                              \
  }

UNI20_DEFINE_ELEMENTWISE_NEGATE(float)
UNI20_DEFINE_ELEMENTWISE_NEGATE(double)
UNI20_DEFINE_ELEMENTWISE_NEGATE(uni20::cfloat)
UNI20_DEFINE_ELEMENTWISE_NEGATE(uni20::cdouble)

#undef UNI20_DEFINE_ELEMENTWISE_NEGATE

} // namespace uni20::linalg::detail::cuda_reference
