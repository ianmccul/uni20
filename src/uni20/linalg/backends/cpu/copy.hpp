#pragma once

/**
 * \file copy.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for accessor-respecting element copies.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class OutputMdspan, class InputMdspan, std::size_t... Axis>
consteval bool copy_element_is_assignable(std::index_sequence<Axis...>)
{
  using output_index = typename OutputMdspan::index_type;
  using input_index = typename InputMdspan::index_type;
  return requires(OutputMdspan& output, InputMdspan& input, output_index output_i, input_index input_i) {
    output.operator[](((void)Axis, output_i)...) = input.operator[](((void)Axis, input_i)...);
  };
}

template <class OutputMdspan, class InputMdspan, std::size_t... Axis>
void copy_element(OutputMdspan& output, InputMdspan& input,
                  std::array<typename OutputMdspan::index_type, OutputMdspan::rank()> const& index,
                  std::index_sequence<Axis...>)
{
  using input_index = typename InputMdspan::index_type;
  output.operator[](index[Axis]...) = input.operator[](static_cast<input_index>(index[Axis])...);
}

template <std::size_t Axis, class OutputMdspan, class InputMdspan>
void copy_elements(OutputMdspan& output, InputMdspan& input,
                   std::array<typename OutputMdspan::index_type, OutputMdspan::rank()>& index)
{
  if constexpr (Axis == OutputMdspan::rank())
  {
    copy_element(output, input, index, std::make_index_sequence<OutputMdspan::rank()>{});
  }
  else
  {
    using output_index = typename OutputMdspan::index_type;
    for (output_index i = 0; i < static_cast<output_index>(output.extent(Axis)); ++i)
    {
      index[Axis] = i;
      copy_elements<Axis + 1>(output, input, index);
    }
  }
}
} // namespace detail

/// \brief Report compile-time eligibility for reference CPU element-copy dispatch.
template <uni20::MutableSpanLike OutputMdspan, uni20::SpanLike InputMdspan>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, copy_op const&, OutputMdspan&, InputMdspan&)
{
  if constexpr (OutputMdspan::rank() == InputMdspan::rank() &&
                detail::copy_element_is_assignable<OutputMdspan, InputMdspan>(
                    std::make_index_sequence<OutputMdspan::rank()>{}))
  {
    return kernel_types_yes;
  }
  else
  {
    return kernel_types_no;
  }
}

/// \brief Copy every input element through its accessor into the corresponding output element.
/// \pre Input and output have equal extents and do not destructively overlap.
template <class OutputMdspan, class InputMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, copy_op const&, OutputMdspan&& output, InputMdspan&& input)
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  static_assert(output_type::rank() == std::remove_cvref_t<InputMdspan>::rank());

  for (std::size_t axis = 0; axis < output_type::rank(); ++axis)
    CHECK_EQUAL(output.extent(axis), input.extent(axis));

  std::array<typename output_type::index_type, output_type::rank()> index{};
  detail::copy_elements<0>(output, input, index);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
