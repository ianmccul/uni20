#pragma once

/**
 * \file reductions.hpp
 * \ingroup linalg
 * \brief cuBLAS full inner-product and Euclidean-norm reductions.
 */

#include <uni20/backend/cublas/execution.hpp>
#include <uni20/backend/cublas/level1.hpp>
#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/storage/cuda_accessor.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace uni20::linalg
{
namespace detail::cublas_backend
{

template <class Output, class Scalar>
concept HostScalarOutput = (!uni20::MdspecLike<Output>) && std::same_as<std::remove_cvref_t<Output>, Scalar> &&
                           requires(Output& output, Scalar value) { output = value; };

template <class Mdspec, class Scalar>
concept RawReadableCudaReductionMdspec =
    uni20::StridedMdspecLike<Mdspec> && uni20::cuda::BufferMdspec<Mdspec> &&
    uni20::cublas::CublasLevelOneScalar<Scalar> &&
    std::same_as<typename std::remove_cvref_t<Mdspec>::element_type, Scalar const> &&
    std::same_as<typename std::remove_cvref_t<Mdspec>::accessor_type, uni20::cuda::CudaPointerAccessor<Scalar const>>;

template <class Mdspec> [[nodiscard]] std::size_t reduction_element_count(Mdspec const& span)
{
  if (!span.mapping().is_unique() || !span.mapping().is_exhaustive()) return 0;
  auto const required = span.mapping().required_span_size();
  CHECK(std::cmp_greater_equal(required, 0), required);
  return static_cast<std::size_t>(required);
}

template <class Mdspec> void require_reduction_descriptor_range(Mdspec const& span, std::size_t size)
{
  auto const& descriptor = span.data_descriptor();
  auto const offset = descriptor.element_offset();
  auto const buffer_size = descriptor.buffer().size();
  CHECK(offset <= buffer_size && size <= buffer_size - offset, offset, size, buffer_size);
}

template <class Scalar, class LhsMdspec, class RhsMdspec>
KernelAttempt try_inner_product(Scalar& output, LhsMdspec const& lhs, RhsMdspec const& rhs)
{
  static_assert(RawReadableCudaReductionMdspec<LhsMdspec, Scalar>);
  static_assert(RawReadableCudaReductionMdspec<RhsMdspec, Scalar>);
  static_assert(std::remove_cvref_t<LhsMdspec>::rank() == std::remove_cvref_t<RhsMdspec>::rank());
  for (std::size_t axis = 0; axis < std::remove_cvref_t<LhsMdspec>::rank(); ++axis)
    CHECK_EQUAL(lhs.extent(axis), rhs.extent(axis));

  if (!lhs.mapping().is_unique() || !lhs.mapping().is_exhaustive() || !rhs.mapping().is_unique() ||
      !rhs.mapping().is_exhaustive())
    return KernelAttempt::unsupported_layout;
  for (std::size_t axis = 0; axis < std::remove_cvref_t<LhsMdspec>::rank(); ++axis)
  {
    if (lhs.extent(axis) > 1 && lhs.stride(axis) != rhs.stride(axis)) return KernelAttempt::unsupported_layout;
  }

  auto const size = reduction_element_count(lhs);
  CHECK_EQUAL(size, reduction_element_count(rhs));
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) return KernelAttempt::unsupported_instance;
  if (size == 0)
  {
    output = Scalar{};
    return KernelAttempt::success;
  }

  auto const& lhs_descriptor = lhs.data_descriptor();
  auto const& rhs_descriptor = rhs.data_descriptor();
  if (lhs_descriptor.buffer().device() != rhs_descriptor.buffer().device()) return KernelAttempt::incompatible_devices;
  require_reduction_descriptor_range(lhs, size);
  require_reduction_descriptor_range(rhs, size);

  auto execution = uni20::cublas::execution_pool(lhs_descriptor.buffer().resources()).acquire();
  auto lhs_access = lhs_descriptor.buffer().read_synchronized_with(execution.stream());
  auto rhs_access = rhs_descriptor.buffer().read_synchronized_with(execution.stream());
  output = uni20::cublas::dotc(execution, static_cast<int>(size), lhs_access.data() + lhs_descriptor.element_offset(),
                               rhs_access.data() + rhs_descriptor.element_offset());
  lhs_access.release_after_synchronization();
  rhs_access.release_after_synchronization();
  return KernelAttempt::success;
}

template <class Scalar, class InputMdspec>
KernelAttempt try_norm(uni20::make_real_t<Scalar>& output, InputMdspec const& input)
{
  static_assert(RawReadableCudaReductionMdspec<InputMdspec, Scalar>);
  if (!input.mapping().is_unique() || !input.mapping().is_exhaustive()) return KernelAttempt::unsupported_layout;

  auto const size = reduction_element_count(input);
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) return KernelAttempt::unsupported_instance;
  if (size == 0)
  {
    output = uni20::make_real_t<Scalar>{};
    return KernelAttempt::success;
  }

  auto const& descriptor = input.data_descriptor();
  require_reduction_descriptor_range(input, size);
  auto execution = uni20::cublas::execution_pool(descriptor.buffer().resources()).acquire();
  auto access = descriptor.buffer().read_synchronized_with(execution.stream());
  output = uni20::cublas::nrm2(execution, static_cast<int>(size), access.data() + descriptor.element_offset());
  access.release_after_synchronization();
  return KernelAttempt::success;
}

} // namespace detail::cublas_backend

/// \brief Report cuBLAS eligibility for a full CUDA inner product returning a host scalar.
template <class Output, uni20::MdspecLike LhsMdspec, uni20::MdspecLike RhsMdspec>
consteval auto kernel_accepts_types(CublasBackend const&, inner_product_op const&, Output&, LhsMdspec&, RhsMdspec&)
{
  using lhs_type = std::remove_cvref_t<LhsMdspec>;
  using rhs_type = std::remove_cvref_t<RhsMdspec>;
  if constexpr (lhs_type::rank() == rhs_type::rank() &&
                std::same_as<typename lhs_type::value_type, typename rhs_type::value_type> &&
                detail::cublas_backend::HostScalarOutput<Output, typename lhs_type::value_type> &&
                detail::cublas_backend::RawReadableCudaReductionMdspec<LhsMdspec, typename lhs_type::value_type> &&
                detail::cublas_backend::RawReadableCudaReductionMdspec<RhsMdspec, typename lhs_type::value_type>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute a full CUDA inner product through cuBLAS.
template <class Output, uni20::MdspecLike LhsMdspec, uni20::MdspecLike RhsMdspec>
  requires detail::cublas_backend::HostScalarOutput<Output, typename std::remove_cvref_t<LhsMdspec>::value_type> &&
           detail::cublas_backend::RawReadableCudaReductionMdspec<
               LhsMdspec, typename std::remove_cvref_t<LhsMdspec>::value_type> &&
           detail::cublas_backend::RawReadableCudaReductionMdspec<RhsMdspec,
                                                                  typename std::remove_cvref_t<LhsMdspec>::value_type>
KernelAttempt try_kernel(CublasBackend, inner_product_op const&, Output& output, LhsMdspec& lhs, RhsMdspec& rhs)
{
  return detail::cublas_backend::try_inner_product(output, lhs, rhs);
}

/// \brief Report cuBLAS eligibility for a full CUDA norm returning a host scalar.
template <class Output, uni20::MdspecLike InputMdspec>
consteval auto kernel_accepts_types(CublasBackend const&, norm_op const&, Output&, InputMdspec&)
{
  using scalar_type = typename std::remove_cvref_t<InputMdspec>::value_type;
  if constexpr (detail::cublas_backend::HostScalarOutput<Output, uni20::make_real_t<scalar_type>> &&
                detail::cublas_backend::RawReadableCudaReductionMdspec<InputMdspec, scalar_type>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute a full CUDA Euclidean norm through cuBLAS.
template <class Output, uni20::MdspecLike InputMdspec>
  requires detail::cublas_backend::HostScalarOutput<
               Output, uni20::make_real_t<typename std::remove_cvref_t<InputMdspec>::value_type>> &&
           detail::cublas_backend::RawReadableCudaReductionMdspec<InputMdspec,
                                                                  typename std::remove_cvref_t<InputMdspec>::value_type>
KernelAttempt try_kernel(CublasBackend, norm_op const&, Output& output, InputMdspec& input)
{
  using scalar_type = typename std::remove_cvref_t<InputMdspec>::value_type;
  return detail::cublas_backend::try_norm<scalar_type>(output, input);
}

} // namespace uni20::linalg
