#pragma once

/**
 * \file matrix_norm.hpp
 * \ingroup linalg
 * \brief Accessor-respecting reference CPU matrix norms.
 */

#include "reductions.hpp"

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/cpu/detail/compensated_sum.hpp>
#include <uni20/linalg/backends/reduction_output.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cpu_reference
{

template <class Result, class Candidate> void update_matrix_norm_max(Result& result, Candidate candidate)
{
  if (!(candidate == candidate))
  {
    result = candidate;
  }
  else if (candidate > result)
  {
    result = candidate;
  }
}

template <class MatrixMdspan> consteval bool matrix_norm_input_is_supported()
{
  using matrix_type = std::remove_cvref_t<MatrixMdspan>;
  using scalar_type = typename matrix_type::value_type;
  if constexpr (!uni20::RealOrComplex<scalar_type> || !detail::reduction_value_is_readable<MatrixMdspan>())
  {
    return false;
  }
  else
  {
    using result_type = uni20::make_real_t<scalar_type>;
    return requires(result_type result) { backends::cpu::detail::CompensatedRealSum<result_type>{}.add(result); };
  }
}

template <class MatrixMdspan> auto matrix_norm(matrix_norm_op const& op, MatrixMdspan& matrix)
{
  using matrix_type = std::remove_cvref_t<MatrixMdspan>;
  using scalar_type = typename matrix_type::value_type;
  using result_type = uni20::make_real_t<scalar_type>;
  using index_type = typename matrix_type::index_type;

  auto magnitude = [&](index_type row, index_type col) {
    using std::abs;
    return static_cast<result_type>(abs(static_cast<scalar_type>(matrix[row, col])));
  };

  if (op.kind == MatrixNorm::Frobenius)
  {
    result_type result{};
    detail::reference_norm(result, matrix);
    return result;
  }

  result_type result{};
  switch (op.kind)
  {
    case MatrixNorm::MaxAbs:
      for (index_type row = 0; row < matrix.extent(0); ++row)
        for (index_type col = 0; col < matrix.extent(1); ++col)
          update_matrix_norm_max(result, magnitude(row, col));
      return result;

    case MatrixNorm::One:
      for (index_type col = 0; col < matrix.extent(1); ++col)
      {
        backends::cpu::detail::CompensatedRealSum<result_type> sum;
        for (index_type row = 0; row < matrix.extent(0); ++row)
          sum.add(magnitude(row, col));
        update_matrix_norm_max(result, sum.value());
      }
      return result;

    case MatrixNorm::Infinity:
      for (index_type row = 0; row < matrix.extent(0); ++row)
      {
        backends::cpu::detail::CompensatedRealSum<result_type> sum;
        for (index_type col = 0; col < matrix.extent(1); ++col)
          sum.add(magnitude(row, col));
        update_matrix_norm_max(result, sum.value());
      }
      return result;

    case MatrixNorm::Frobenius:
      break;
  }
  PANIC("invalid MatrixNorm", std::to_underlying(op.kind));
}

template <class Output, class MatrixMdspan> consteval auto matrix_norm_acceptance()
{
  using scalar_type = typename MatrixMdspan::value_type;
  using result_type = uni20::make_real_t<scalar_type>;
  if constexpr (matrix_norm_input_is_supported<MatrixMdspan>() &&
                reduction_detail::output_is_supported<Output, result_type, 0>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

} // namespace detail::cpu_reference

/// \brief Report eligibility for a host-accessible matrix norm.
template <class Output, uni20::RankedMdspecLike<2> MatrixMdspec>
  requires reduction_detail::HostOutput<Output> && uni20::HostReadableMdspec<MatrixMdspec>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, matrix_norm_op const&, Output&, MatrixMdspec&)
{
  using output_type = reduction_detail::host_output_t<Output>;
  using matrix_span = uni20::host_read_mdspan_t<MatrixMdspec>;
  constexpr auto acceptance = detail::cpu_reference::matrix_norm_acceptance<output_type, matrix_span>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host access and compute a matrix norm.
template <class Output, uni20::RankedMdspecLike<2> MatrixMdspec>
  requires reduction_detail::HostOutput<Output> && uni20::HostReadableMdspec<MatrixMdspec>
KernelAttempt try_kernel(CpuReferenceBackend, matrix_norm_op const& operation, Output& output, MatrixMdspec& matrix)
{
  auto matrix_access = acquire_host_read_access_sync(matrix);
  auto matrix_span = matrix_access.mdspan();
  return reduction_detail::with_host_output(output, [&](auto& resolved_output) {
    auto result = detail::cpu_reference::matrix_norm(operation, matrix_span);
    reduction_detail::write_host_output(resolved_output, result);
    return KernelAttempt::success;
  });
}

} // namespace uni20::linalg
