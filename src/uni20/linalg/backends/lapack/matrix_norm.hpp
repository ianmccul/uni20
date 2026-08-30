#pragma once

/**
 * \file matrix_norm.hpp
 * \ingroup linalg
 * \brief LAPACK backend for dense matrix norms.
 */

#include "common.hpp"

#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/reduction_output.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::linalg
{
namespace lapack_detail
{

template <class Output, class MatrixMdspan> consteval auto matrix_norm_acceptance()
{
  using scalar_type = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using result_type = uni20::make_real_t<scalar_type>;
  if constexpr (uni20::LapackScalar<scalar_type> && uni20::DefaultAccessorMdspanLike<MatrixMdspan> &&
                reduction_detail::output_is_supported<Output, result_type, 0>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

inline char lapack_matrix_norm(MatrixNorm kind, bool transposed)
{
  switch (kind)
  {
    case MatrixNorm::MaxAbs:
      return 'M';
    case MatrixNorm::One:
      return transposed ? 'I' : '1';
    case MatrixNorm::Infinity:
      return transposed ? '1' : 'I';
    case MatrixNorm::Frobenius:
      return 'F';
  }
  PANIC("invalid MatrixNorm", std::to_underlying(kind));
}

template <class Output, uni20::RankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_matrix_norm(matrix_norm_op const& operation, Output& output, MatrixMdspan& matrix)
{
  using scalar_type = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using result_type = uni20::make_real_t<scalar_type>;

  if constexpr (uni20::Complex<scalar_type>)
  {
    // xLANGE uses |Re(z)| + |Im(z)| for these complex norms. Uni20's
    // public contract uses the mathematical complex magnitude instead.
    if (operation.kind != MatrixNorm::Frobenius) return KernelAttempt::unsupported_instance;
  }

  auto stage = uni20::linalg::blas::try_mdspan_matrix_stage(matrix);
  if (!stage) return KernelAttempt::unsupported_layout;

  auto provider = uni20::linalg::blas::blas_writable_matrix(*stage);
  if (provider.rows == 0 || provider.cols == 0)
  {
    reduction_detail::write_host_output(output, result_type{});
    return KernelAttempt::success;
  }

  char const norm = lapack_matrix_norm(operation.kind, stage->unit_stride_axis == 1);
  std::size_t const work_size = static_cast<std::size_t>(std::max<blas_int>(1, provider.rows));
  std::vector<result_type> work(work_size, result_type{});

  // LANGE is read-only although its Fortran ABI predates const-correct C
  // wrappers. Keep the cast at this provider boundary.
  auto* data = const_cast<scalar_type*>(provider.data);
  result_type const result =
      uni20::lapack::lange(norm, provider.rows, provider.cols, data, provider.leading_dimension, work.data());
  reduction_detail::write_host_output(output, result);
  return KernelAttempt::success;
}

} // namespace lapack_detail

/// \brief Report eligibility for a host-accessible LAPACK matrix norm.
template <class Output, uni20::RankedStridedMdspecLike<2> MatrixMdspec>
  requires reduction_detail::HostOutput<Output> && uni20::HostReadableMdspec<MatrixMdspec>
consteval auto kernel_accepts_types(LapackBackend const&, matrix_norm_op const&, Output&, MatrixMdspec&)
{
  using output_type = reduction_detail::host_output_t<Output>;
  using matrix_span = uni20::host_read_mdspan_t<MatrixMdspec>;
  constexpr auto acceptance = lapack_detail::matrix_norm_acceptance<output_type, matrix_span>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host access and compute a matrix norm through LAPACK.
template <class Output, uni20::RankedStridedMdspecLike<2> MatrixMdspec>
  requires reduction_detail::HostOutput<Output> && uni20::HostReadableMdspec<MatrixMdspec>
KernelAttempt try_kernel(LapackBackend, matrix_norm_op const& operation, Output& output, MatrixMdspec& matrix)
{
  auto matrix_access = acquire_host_read_access_sync(matrix);
  auto matrix_span = matrix_access.mdspan();
  return reduction_detail::with_host_output(output, [&](auto& resolved_output) {
    return lapack_detail::try_matrix_norm(operation, resolved_output, matrix_span);
  });
}

} // namespace uni20::linalg
