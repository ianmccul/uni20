#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Generic CPU GEMM backend for mdspan-like dense matrices.
 */

#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <type_traits>

namespace uni20::linalg
{

namespace detail
{
template <class Mdspan, class Scalar>
concept readable_cpu_gemm_mdspan_for =
    uni20::RankedSpanLike<Mdspan, 2> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>, Scalar>;

template <class Mdspan, class Scalar>
concept writable_cpu_gemm_mdspan_for =
    uni20::MutableRankedSpanLike<Mdspan, 2> &&
    std::same_as<std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>, Scalar>;

template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
concept cpu_gemm_types_supported =
    uni20::Scalar<Scalar> && writable_cpu_gemm_mdspan_for<OutputMdspan, Scalar> &&
    readable_cpu_gemm_mdspan_for<LhsMdspan, Scalar> && readable_cpu_gemm_mdspan_for<RhsMdspan, Scalar>;
} // namespace detail

/// \brief Report compile-time eligibility for generic CPU GEMM dispatch.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
consteval KernelTypeAcceptance kernel_accepts_types(CpuGenericBackend const&, struct gemm_op const&, OutputMdspan&,
                                                    Scalar const&, LhsMdspan&, RhsMdspan&, Scalar const&)
{
  if constexpr (detail::cpu_gemm_types_supported<OutputMdspan, Scalar, LhsMdspan, RhsMdspan>)
  {
    return KernelTypeAcceptance::yes;
  }
  else
  {
    return KernelTypeAcceptance::no;
  }
}

/// \brief Generic accessor-respecting GEMM fallback.
template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
bool try_kernel(CpuGenericBackend, struct gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                RhsMdspan&& rhs, Scalar beta)
{
  static_assert(detail::cpu_gemm_types_supported<std::remove_cvref_t<OutputMdspan>, Scalar,
                                                 std::remove_cvref_t<LhsMdspan>, std::remove_cvref_t<RhsMdspan>>,
                "CPU GEMM try_kernel called for types rejected by kernel_accepts_types");

  using output_type = std::remove_cvref_t<OutputMdspan>;
  using index_type = typename output_type::index_type;

  CHECK_EQUAL(lhs.extent(1), rhs.extent(0));
  CHECK_EQUAL(output.extent(0), lhs.extent(0));
  CHECK_EQUAL(output.extent(1), rhs.extent(1));

  index_type const rows = static_cast<index_type>(output.extent(0));
  index_type const cols = static_cast<index_type>(output.extent(1));
  index_type const inner = static_cast<index_type>(lhs.extent(1));

  for (index_type row = 0; row < rows; ++row)
  {
    for (index_type col = 0; col < cols; ++col)
    {
      Scalar product{};
      for (index_type k = 0; k < inner; ++k)
      {
        product += static_cast<Scalar>(lhs[row, k]) * static_cast<Scalar>(rhs[k, col]);
      }

      if (beta == Scalar{})
      {
        output[row, col] = alpha * product;
      }
      else
      {
        output[row, col] = beta * static_cast<Scalar>(output[row, col]) + alpha * product;
      }
    }
  }

  return true;
}

} // namespace uni20::linalg
