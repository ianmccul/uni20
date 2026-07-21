#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Checked cuBLAS GEMM wrappers and opaque CUDA-mdspan-to-provider lowering.
 */

#include <uni20/backend/cublas/gemm.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/blas/blas_matrix.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/kernel_attempt.hpp>
#include <uni20/storage/cuda_async_storage.hpp>

#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20::linalg::cublas
{

/// \brief Provider-ready CUDA GEMM operands produced without acquiring execution resources.
template <uni20::cublas::CublasScalar Scalar> struct GemmPlan
{
    blas::BlasWritableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar>> output{};
    blas::BlasReadableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar const>> lhs{};
    blas::BlasReadableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar const>> rhs{};
    blas_int inner = 0;
    int device = -1;
    bool has_work = false;

    /// \brief Return the device resources that own the output allocation.
    [[nodiscard]] uni20::cuda::DeviceResources& resources() const
    {
      CHECK(has_work);
      return output.data.buffer().resources();
    }
};

/// \brief Result of side-effect-free CUDA GEMM operand preparation.
template <uni20::cublas::CublasScalar Scalar> struct GemmPreparation
{
    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    GemmPlan<Scalar> plan{};
};

namespace detail
{

template <class Accessor, class Scalar> struct IsCudaAsyncAccessorFor : std::false_type
{};

template <class ElementType, class Scalar>
struct IsCudaAsyncAccessorFor<uni20::cuda::AsyncAccessor<ElementType>, Scalar>
    : std::bool_constant<std::same_as<std::remove_cv_t<ElementType>, Scalar>>
{};

template <class Accessor, class Scalar>
struct IsCudaAsyncAccessorFor<uni20::conjugated_accessor<Accessor>, Scalar> : IsCudaAsyncAccessorFor<Accessor, Scalar>
{};

template <class Accessor, class Scalar>
inline constexpr bool cuda_async_accessor_for = IsCudaAsyncAccessorFor<std::remove_cvref_t<Accessor>, Scalar>::value;

template <class Handle, class Scalar> struct IsCudaBufferViewFor : std::false_type
{};

template <class ElementType, class Scalar>
struct IsCudaBufferViewFor<uni20::cuda::CudaBufferView<ElementType>, Scalar>
    : std::bool_constant<std::same_as<std::remove_cv_t<ElementType>, Scalar>>
{};

template <class Handle, class Scalar>
inline constexpr bool is_cuda_buffer_view_for = IsCudaBufferViewFor<std::remove_cvref_t<Handle>, Scalar>::value;

template <class Mdspan, class Scalar>
concept readable_cuda_mdspan_for =
    uni20::RankedStridedMdspan<Mdspan, 2> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>, Scalar> &&
    cuda_async_accessor_for<typename std::remove_cvref_t<Mdspan>::accessor_type, Scalar> &&
    is_cuda_buffer_view_for<typename std::remove_cvref_t<Mdspan>::data_handle_type, Scalar>;

template <class Mdspan, class Scalar>
concept writable_cuda_mdspan_for =
    uni20::MutableRankedStridedMdspan<Mdspan, 2> &&
    std::same_as<typename std::remove_cvref_t<Mdspan>::element_type, Scalar> &&
    std::same_as<typename std::remove_cvref_t<Mdspan>::accessor_type, uni20::cuda::AsyncAccessor<Scalar>> &&
    std::same_as<typename std::remove_cvref_t<Mdspan>::data_handle_type, uni20::cuda::CudaBufferView<Scalar>>;

inline int cublas_int(blas_int value)
{
  CHECK(value >= 0 && value <= std::numeric_limits<int>::max(), value);
  return static_cast<int>(value);
}

template <uni20::cublas::CublasScalar Scalar, class LhsHandle, class RhsHandle>
constexpr bool provider_transforms_are_supported(blas::BlasReadableMatrix<Scalar, LhsHandle> lhs,
                                                 blas::BlasReadableMatrix<Scalar, RhsHandle> rhs)
{
  return blas::blas_trans_char_is_supported<Scalar>(lhs.transform) &&
         blas::blas_trans_char_is_supported<Scalar>(rhs.transform);
}

template <uni20::cublas::CublasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
blas_int require_gemm_shape(blas::BlasWritableMatrix<Scalar, OutputHandle> output,
                            blas::BlasReadableMatrix<Scalar, LhsHandle> lhs,
                            blas::BlasReadableMatrix<Scalar, RhsHandle> rhs)
{
  blas_int const lhs_rows = blas::transformed_rows(lhs.rows, lhs.cols, lhs.transform);
  blas_int const lhs_cols = blas::transformed_cols(lhs.rows, lhs.cols, lhs.transform);
  blas_int const rhs_rows = blas::transformed_rows(rhs.rows, rhs.cols, rhs.transform);
  blas_int const rhs_cols = blas::transformed_cols(rhs.rows, rhs.cols, rhs.transform);
  CHECK_EQUAL(lhs_cols, rhs_rows);
  CHECK_EQUAL(output.rows, lhs_rows);
  CHECK_EQUAL(output.cols, rhs_cols);
  return lhs_cols;
}

template <class Scalar, class Handle> std::size_t required_elements(blas::BlasWritableMatrix<Scalar, Handle> matrix)
{
  CHECK(matrix.rows >= 0 && matrix.cols >= 0 && matrix.leading_dimension > 0, matrix.rows, matrix.cols,
        matrix.leading_dimension);
  if (matrix.rows == 0 || matrix.cols == 0) return 0;

  auto const rows = static_cast<std::size_t>(matrix.rows);
  auto const cols = static_cast<std::size_t>(matrix.cols);
  auto const leading_dimension = static_cast<std::size_t>(matrix.leading_dimension);
  CHECK(cols - 1 <= (std::numeric_limits<std::size_t>::max() - rows) / leading_dimension, rows, cols,
        leading_dimension);
  return (cols - 1) * leading_dimension + rows;
}

template <class Scalar, class Handle> std::size_t required_elements(blas::BlasReadableMatrix<Scalar, Handle> matrix)
{
  return required_elements(blas::BlasWritableMatrix<Scalar, Handle>{
      .data = matrix.data, .rows = matrix.rows, .cols = matrix.cols, .leading_dimension = matrix.leading_dimension});
}

template <class View, class Matrix> void require_view_covers_matrix(View const& view, Matrix const& matrix)
{
  auto const offset = view.element_offset();
  auto const size = view.buffer().size();
  auto const required = required_elements(matrix);
  CHECK(offset <= size && required <= size - offset, offset, required, size);
}

template <class Pointer> Pointer offset_pointer(Pointer pointer, std::size_t offset)
{
  if (offset == 0) return pointer;
  CHECK(pointer != nullptr, offset);
  return pointer + offset;
}

template <class Scalar, class Handle>
auto with_data(blas::BlasWritableMatrix<Scalar, Handle> matrix, Scalar* data) -> blas::BlasWritableMatrix<Scalar>
{
  return {.data = data, .rows = matrix.rows, .cols = matrix.cols, .leading_dimension = matrix.leading_dimension};
}

template <class Scalar, class Handle>
auto with_data(blas::BlasReadableMatrix<Scalar, Handle> matrix, Scalar const* data) -> blas::BlasReadableMatrix<Scalar>
{
  return {.data = data,
          .rows = matrix.rows,
          .cols = matrix.cols,
          .leading_dimension = matrix.leading_dimension,
          .transform = matrix.transform};
}

template <uni20::cublas::CublasScalar Scalar, class Handle>
auto readonly_matrix(blas::BlasReadableMatrix<Scalar, Handle> matrix)
    -> blas::BlasReadableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar const>>
{
  return {.data = uni20::cuda::CudaBufferView<Scalar const>{matrix.data},
          .rows = matrix.rows,
          .cols = matrix.cols,
          .leading_dimension = matrix.leading_dimension,
          .transform = matrix.transform};
}

template <uni20::cublas::CublasScalar Scalar, class LhsHandle, class RhsHandle>
GemmPreparation<Scalar>
prepare_staged_gemm(blas::BlasWritableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar>> output,
                    blas::BlasReadableMatrix<Scalar, LhsHandle> lhs, blas::BlasReadableMatrix<Scalar, RhsHandle> rhs)
{
  if (!provider_transforms_are_supported(lhs, rhs))
  {
    return {.attempt = KernelAttempt::unsupported_transform};
  }

  blas_int const inner = require_gemm_shape(output, lhs, rhs);
  if (output.rows == 0 || output.cols == 0)
  {
    return {.attempt = KernelAttempt::success};
  }

  auto& output_buffer = output.data.buffer();
  auto const& lhs_buffer = lhs.data.buffer();
  auto const& rhs_buffer = rhs.data.buffer();
  CHECK(std::addressof(output_buffer) != std::addressof(lhs_buffer) &&
            std::addressof(output_buffer) != std::addressof(rhs_buffer),
        "cuBLAS GEMM output must not share a CUDA buffer with an input");

  int const device = output_buffer.device().ordinal();
  CHECK_EQUAL(lhs_buffer.device().ordinal(), device, "cuBLAS GEMM operands must use one CUDA device");
  CHECK_EQUAL(rhs_buffer.device().ordinal(), device, "cuBLAS GEMM operands must use one CUDA device");
  require_view_covers_matrix(output.data, output);
  require_view_covers_matrix(lhs.data, lhs);
  require_view_covers_matrix(rhs.data, rhs);

  return {.attempt = KernelAttempt::success,
          .plan = {.output = output,
                   .lhs = readonly_matrix(lhs),
                   .rhs = readonly_matrix(rhs),
                   .inner = inner,
                   .device = device,
                   .has_work = true}};
}

} // namespace detail

/// \brief Try provider-ready column-major GEMM through cuBLAS.
/// \return `success` or `unsupported_transform`; invalid dimensions are logic errors.
template <uni20::cublas::CublasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
  requires std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<LhsHandle, Scalar const*> &&
           std::convertible_to<RhsHandle, Scalar const*>
KernelAttempt try_gemm(uni20::cublas::ExecutionLease& execution, blas::BlasWritableMatrix<Scalar, OutputHandle> output,
                       blas::BlasReadableMatrix<Scalar, LhsHandle> lhs, blas::BlasReadableMatrix<Scalar, RhsHandle> rhs,
                       Scalar alpha, Scalar beta)
{
  if (!detail::provider_transforms_are_supported(lhs, rhs)) return KernelAttempt::unsupported_transform;

  blas_int const lhs_cols = detail::require_gemm_shape(output, lhs, rhs);

  if (output.rows == 0 || output.cols == 0) return KernelAttempt::success;

  uni20::cublas::gemm(execution, blas::blas_trans_char<Scalar>(lhs.transform),
                      blas::blas_trans_char<Scalar>(rhs.transform), detail::cublas_int(output.rows),
                      detail::cublas_int(output.cols), detail::cublas_int(lhs_cols), alpha,
                      static_cast<Scalar const*>(lhs.data), detail::cublas_int(lhs.leading_dimension),
                      static_cast<Scalar const*>(rhs.data), detail::cublas_int(rhs.leading_dimension), beta,
                      static_cast<Scalar*>(output.data), detail::cublas_int(output.leading_dimension));
  return KernelAttempt::success;
}

/// \brief Prepare no-copy GEMM from opaque CUDA mdspan operands without acquiring resources.
/// \details Recognized accessors and handles are validated for shape, layout,
///          transform, device, aliasing, and allocation bounds. Preparation has
///          no externally visible side effects, so a non-success attempt is a
///          clean backend decline. A successful plan retains opaque buffer views
///          but does not acquire synchronized access or provider resources.
template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_cuda_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_cuda_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_cuda_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
GemmPreparation<Scalar> prepare_gemm(OutputMdspan&& output, LhsMdspan&& lhs, RhsMdspan&& rhs)
{
  CHECK_EQUAL(lhs.extent(1), rhs.extent(0));
  CHECK_EQUAL(output.extent(0), lhs.extent(0));
  CHECK_EQUAL(output.extent(1), rhs.extent(1));

  if (output.extent(0) == 0 || output.extent(1) == 0)
  {
    return {.attempt = KernelAttempt::success, .plan = {.device = output.data_handle().buffer().device().ordinal()}};
  }

  auto output_stage = blas::try_mdspan_matrix_stage(output);
  auto lhs_stage = blas::try_mdspan_matrix_stage(lhs);
  auto rhs_stage = blas::try_mdspan_matrix_stage(rhs);
  if (!output_stage || !lhs_stage || !rhs_stage)
  {
    return {.attempt = KernelAttempt::unsupported_layout};
  }
  CHECK(!output_stage->needs_conjugation);

  auto const output_matrix = blas::blas_writable_matrix(*output_stage);
  if (output_stage->unit_stride_axis == 0)
  {
    return detail::prepare_staged_gemm(output_matrix, blas::blas_readable_matrix(*lhs_stage),
                                       blas::blas_readable_matrix(*rhs_stage));
  }

  auto lhs_matrix = blas::blas_readable_matrix(*rhs_stage);
  auto rhs_matrix = blas::blas_readable_matrix(*lhs_stage);
  lhs_matrix.transform = blas::compose(blas::MatrixTransform::transpose, lhs_matrix.transform);
  rhs_matrix.transform = blas::compose(blas::MatrixTransform::transpose, rhs_matrix.transform);
  return detail::prepare_staged_gemm(output_matrix, lhs_matrix, rhs_matrix);
}

/// \brief Enqueue one prepared CUDA GEMM and publish its buffer completion state.
/// \details The execution lease must belong to the plan device. Scoped buffer
///          access installs incoming dependencies and records stream-tail
///          completions before this function returns.
template <uni20::cublas::CublasScalar Scalar>
void execute_gemm(uni20::cublas::ExecutionLease& execution, GemmPlan<Scalar> const& plan, Scalar alpha, Scalar beta)
{
  CHECK(plan.has_work);
  CHECK_EQUAL(execution.handle().device(), plan.device, "cuBLAS execution lease must match the operand device");

  auto& output_buffer = plan.output.data.buffer();
  auto const& lhs_buffer = plan.lhs.data.buffer();
  auto const& rhs_buffer = plan.rhs.data.buffer();
  auto output_access = output_buffer.write_synchronized_with(execution.stream());
  auto lhs_access = lhs_buffer.read_synchronized_with(execution.stream());
  auto rhs_access = rhs_buffer.read_synchronized_with(execution.stream());

  auto const raw_output =
      detail::with_data(plan.output, detail::offset_pointer(output_access.data(), plan.output.data.element_offset()));
  auto const raw_lhs =
      detail::with_data(plan.lhs, detail::offset_pointer(lhs_access.data(), plan.lhs.data.element_offset()));
  auto const raw_rhs =
      detail::with_data(plan.rhs, detail::offset_pointer(rhs_access.data(), plan.rhs.data.element_offset()));
  uni20::cublas::gemm(execution, blas::blas_trans_char<Scalar>(raw_lhs.transform),
                      blas::blas_trans_char<Scalar>(raw_rhs.transform), detail::cublas_int(raw_output.rows),
                      detail::cublas_int(raw_output.cols), detail::cublas_int(plan.inner), alpha, raw_lhs.data,
                      detail::cublas_int(raw_lhs.leading_dimension), raw_rhs.data,
                      detail::cublas_int(raw_rhs.leading_dimension), beta, raw_output.data,
                      detail::cublas_int(raw_output.leading_dimension));
}

/// \brief Try no-copy GEMM through the blocking cuBLAS resource-admission path.
/// \details Runtime acceptance is decided before resource acquisition. The call
///          blocks only while acquiring a handle and idle stream; submitted
///          device execution remains asynchronous.
template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires detail::writable_cuda_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           detail::readable_cuda_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           detail::readable_cuda_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
KernelAttempt try_gemm(OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs, RhsMdspan&& rhs, Scalar beta)
{
  auto preparation = prepare_gemm<Scalar>(std::forward<OutputMdspan>(output), std::forward<LhsMdspan>(lhs),
                                          std::forward<RhsMdspan>(rhs));
  if (!kernel_attempt_succeeded(preparation.attempt) || !preparation.plan.has_work)
  {
    return preparation.attempt;
  }

  auto execution = uni20::cublas::execution_pool(preparation.plan.resources()).acquire();
  execute_gemm(execution, preparation.plan, alpha, beta);
  return KernelAttempt::success;
}

/// \brief Run provider-ready column-major GEMM through cuBLAS.
template <uni20::cublas::CublasScalar Scalar, class OutputHandle, class LhsHandle, class RhsHandle>
  requires std::convertible_to<OutputHandle, Scalar*> && std::convertible_to<LhsHandle, Scalar const*> &&
           std::convertible_to<RhsHandle, Scalar const*>
void gemm(uni20::cublas::ExecutionLease& execution, blas::BlasWritableMatrix<Scalar, OutputHandle> output,
          blas::BlasReadableMatrix<Scalar, LhsHandle> lhs, blas::BlasReadableMatrix<Scalar, RhsHandle> rhs,
          Scalar alpha, Scalar beta)
{
  CHECK(kernel_attempt_succeeded(try_gemm(execution, output, lhs, rhs, alpha, beta)));
}

} // namespace uni20::linalg::cublas
