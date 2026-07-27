#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief CUDA-mdspan preparation and buffer-ledger execution for CublasBackend GEMM.
 */

#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/cublas/gemm.hpp>
#include <uni20/linalg/kernel_attempt.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20::linalg::detail::cublas_backend
{

template <uni20::cublas::CublasScalar Scalar> struct GemmPlan
{
    blas::BlasWritableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar>> output{};
    blas::BlasReadableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar const>> lhs{};
    blas::BlasReadableMatrix<Scalar, uni20::cuda::CudaBufferView<Scalar const>> rhs{};
    blas_int inner = 0;
    int device = -1;
    bool has_work = false;

    [[nodiscard]] uni20::cuda::DeviceResources& resources() const
    {
      CHECK(has_work);
      return output.data.buffer().resources();
    }
};

template <uni20::cublas::CublasScalar Scalar> struct GemmPreparation
{
    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    GemmPlan<Scalar> plan{};
};

template <class Accessor, class Scalar> struct IsCudaAccessorFor : std::false_type
{};

template <class ElementType, class Scalar>
struct IsCudaAccessorFor<uni20::cuda::CudaPointerAccessor<ElementType>, Scalar>
    : std::bool_constant<std::same_as<std::remove_cv_t<ElementType>, Scalar>>
{};

template <class Accessor, class Scalar>
struct IsCudaAccessorFor<uni20::conjugated_accessor<Accessor>, Scalar> : IsCudaAccessorFor<Accessor, Scalar>
{};

template <class Accessor, class Scalar>
inline constexpr bool cuda_accessor_for = IsCudaAccessorFor<std::remove_cvref_t<Accessor>, Scalar>::value;

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
    uni20::RankedStridedDeviceSpanLike<Mdspan, 2> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>, Scalar> &&
    cuda_accessor_for<typename std::remove_cvref_t<Mdspan>::accessor_type, Scalar> &&
    is_cuda_buffer_view_for<blas::detail::span_data_t<Mdspan>, Scalar>;

template <class Mdspan, class Scalar>
concept writable_cuda_mdspan_for =
    uni20::MutableDeviceSpanLike<Mdspan> && uni20::RankedStridedDeviceSpanLike<Mdspan, 2> &&
    std::same_as<typename std::remove_cvref_t<Mdspan>::element_type, Scalar> &&
    std::same_as<typename std::remove_cvref_t<Mdspan>::accessor_type, uni20::cuda::CudaPointerAccessor<Scalar>> &&
    is_cuda_buffer_view_for<blas::detail::span_data_t<Mdspan>, Scalar>;

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
  if (!uni20::linalg::cublas::detail::provider_transforms_are_supported(lhs, rhs))
  {
    return {.attempt = KernelAttempt::unsupported_transform};
  }

  blas_int const inner = uni20::linalg::cublas::detail::require_gemm_shape(output, lhs, rhs);
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

template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires writable_cuda_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           readable_cuda_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           readable_cuda_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
GemmPreparation<Scalar> prepare_gemm(OutputMdspan&& output, LhsMdspan&& lhs, RhsMdspan&& rhs)
{
  CHECK_EQUAL(lhs.extent(1), rhs.extent(0));
  CHECK_EQUAL(output.extent(0), lhs.extent(0));
  CHECK_EQUAL(output.extent(1), rhs.extent(1));

  if (output.extent(0) == 0 || output.extent(1) == 0)
  {
    return {.attempt = KernelAttempt::success,
            .plan = {.device = blas::detail::span_data(output).buffer().device().ordinal()}};
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
    return prepare_staged_gemm(output_matrix, blas::blas_readable_matrix(*lhs_stage),
                               blas::blas_readable_matrix(*rhs_stage));
  }

  auto lhs_matrix = blas::blas_readable_matrix(*rhs_stage);
  auto rhs_matrix = blas::blas_readable_matrix(*lhs_stage);
  lhs_matrix.transform = blas::compose(blas::MatrixTransform::transpose, lhs_matrix.transform);
  rhs_matrix.transform = blas::compose(blas::MatrixTransform::transpose, rhs_matrix.transform);
  return prepare_staged_gemm(output_matrix, lhs_matrix, rhs_matrix);
}

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
      with_data(plan.output, offset_pointer(output_access.data(), plan.output.data.element_offset()));
  auto const raw_lhs = with_data(plan.lhs, offset_pointer(lhs_access.data(), plan.lhs.data.element_offset()));
  auto const raw_rhs = with_data(plan.rhs, offset_pointer(rhs_access.data(), plan.rhs.data.element_offset()));
  uni20::linalg::cublas::gemm(execution, raw_output, alpha, raw_lhs, raw_rhs, beta);
}

template <uni20::cublas::CublasScalar Scalar, class OutputMdspan, class LhsMdspan, class RhsMdspan>
  requires writable_cuda_mdspan_for<std::remove_cvref_t<OutputMdspan>, Scalar> &&
           readable_cuda_mdspan_for<std::remove_cvref_t<LhsMdspan>, Scalar> &&
           readable_cuda_mdspan_for<std::remove_cvref_t<RhsMdspan>, Scalar>
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

} // namespace uni20::linalg::detail::cublas_backend
