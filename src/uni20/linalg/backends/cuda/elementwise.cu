#include "elementwise_arithmetic.hpp"
#include "elementwise_conjugate.hpp"
#include "elementwise_copy.hpp"
#include "elementwise_kernel.cuh"
#include "elementwise_scale.hpp"

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

template <class Scalar, class Operation, class Plan>
void enqueue_elementwise_unary_impl(Scalar* output, Scalar const* input, Operation operation, Plan const& plan,
                                    cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  launch_elementwise_kernel<false>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference unary arithmetic",
                                   ElementwiseOperand{input, input_accessor{}});
}

template <class Value> [[nodiscard]] auto to_cuda_execution_value(Value value) { return value; }

template <class Real> [[nodiscard]] auto to_cuda_execution_value(uni20::complex<Real> const& value)
{
  return ::cuda::std::complex<Real>{value.real(), value.imag()};
}

template <class Scalar, class Factor, class Plan>
void enqueue_elementwise_scale_impl(Scalar* output, Scalar const* input, Factor factor, Plan const& plan,
                                    cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  auto operation = uni20::linalg::scale{to_cuda_execution_value(factor)};
  launch_elementwise_kernel<false>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference elementwise scaling",
                                   ElementwiseOperand{input, input_accessor{}});
}

template <class Scalar, class Operation, class Plan>
void enqueue_elementwise_binary_impl(Scalar* output, Scalar const* lhs, Scalar const* rhs, Operation operation,
                                     Plan const& plan, cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  launch_elementwise_kernel<false>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference binary arithmetic", ElementwiseOperand{lhs, input_accessor{}},
                                   ElementwiseOperand{rhs, input_accessor{}});
}

} // namespace

template <RegisteredStatelessUnary Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan32 const& plan, cudaStream_t stream, int device)
{
  enqueue_elementwise_unary_impl(output, input, function, plan, stream, device);
}

template <RegisteredStatelessUnary Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan64 const& plan, cudaStream_t stream, int device)
{
  enqueue_elementwise_unary_impl(output, input, function, plan, stream, device);
}

template <RegisteredStatelessBinary Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan32 const& plan, cudaStream_t stream, int device)
{
  enqueue_elementwise_binary_impl(output, lhs, rhs, function, plan, stream, device);
}

template <RegisteredStatelessBinary Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan64 const& plan, cudaStream_t stream, int device)
{
  enqueue_elementwise_binary_impl(output, lhs, rhs, function, plan, stream, device);
}

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

#define UNI20_DEFINE_ELEMENTWISE_SCALE(Scalar, Factor)                                                                 \
  void enqueue_elementwise_scale(Scalar* output, Scalar const* input, Factor factor,                                   \
                                 ElementwiseScalePlan32 const& plan, cudaStream_t stream, int device)                  \
  {                                                                                                                    \
    enqueue_elementwise_scale_impl(output, input, factor, plan, stream, device);                                       \
  }                                                                                                                    \
                                                                                                                       \
  void enqueue_elementwise_scale(Scalar* output, Scalar const* input, Factor factor,                                   \
                                 ElementwiseScalePlan64 const& plan, cudaStream_t stream, int device)                  \
  {                                                                                                                    \
    enqueue_elementwise_scale_impl(output, input, factor, plan, stream, device);                                       \
  }

UNI20_DEFINE_ELEMENTWISE_SCALE(float, float)
UNI20_DEFINE_ELEMENTWISE_SCALE(double, double)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cfloat, float)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cfloat, uni20::cfloat)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cdouble, double)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cdouble, uni20::cdouble)

#undef UNI20_DEFINE_ELEMENTWISE_SCALE

#define UNI20_INSTANTIATE_UNARY(Function, Scalar)                                                                      \
  template void enqueue_elementwise_unary(Scalar*, Scalar const*, Function, ElementwiseUnaryPlan32 const&,             \
                                          cudaStream_t, int);                                                          \
  template void enqueue_elementwise_unary(Scalar*, Scalar const*, Function, ElementwiseUnaryPlan64 const&,             \
                                          cudaStream_t, int)

#define UNI20_INSTANTIATE_UNARY_SCALARS(Function)                                                                      \
  UNI20_INSTANTIATE_UNARY(Function, float);                                                                            \
  UNI20_INSTANTIATE_UNARY(Function, double);                                                                           \
  UNI20_INSTANTIATE_UNARY(Function, uni20::cfloat);                                                                    \
  UNI20_INSTANTIATE_UNARY(Function, uni20::cdouble)

UNI20_INSTANTIATE_UNARY_SCALARS(uni20::linalg::negate);
UNI20_INSTANTIATE_UNARY_SCALARS(uni20::linalg::square);
UNI20_INSTANTIATE_UNARY_SCALARS(uni20::linalg::reciprocal);

#undef UNI20_INSTANTIATE_UNARY_SCALARS
#undef UNI20_INSTANTIATE_UNARY

#define UNI20_INSTANTIATE_BINARY(Function, Scalar)                                                                     \
  template void enqueue_elementwise_binary(Scalar*, Scalar const*, Scalar const*, Function,                            \
                                           ElementwiseBinaryPlan32 const&, cudaStream_t, int);                         \
  template void enqueue_elementwise_binary(Scalar*, Scalar const*, Scalar const*, Function,                            \
                                           ElementwiseBinaryPlan64 const&, cudaStream_t, int)

#define UNI20_INSTANTIATE_BINARY_SCALARS(Function)                                                                     \
  UNI20_INSTANTIATE_BINARY(Function, float);                                                                           \
  UNI20_INSTANTIATE_BINARY(Function, double);                                                                          \
  UNI20_INSTANTIATE_BINARY(Function, uni20::cfloat);                                                                   \
  UNI20_INSTANTIATE_BINARY(Function, uni20::cdouble)

UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::add);
UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::subtract);
UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::multiply);
UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::divide);

#undef UNI20_INSTANTIATE_BINARY_SCALARS
#undef UNI20_INSTANTIATE_BINARY

} // namespace uni20::linalg::detail::cuda_reference
