#pragma once

/**
 * \file copy.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for accessor-respecting element copies.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/cpu/strided_transform.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

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

template <std::size_t Depth, bool ColumnMajor, class OutputMdspan, class InputMdspan>
void copy_elements(OutputMdspan& output, InputMdspan& input,
                   std::array<typename OutputMdspan::index_type, OutputMdspan::rank()>& index)
{
  if constexpr (Depth == OutputMdspan::rank())
  {
    copy_element(output, input, index, std::make_index_sequence<OutputMdspan::rank()>{});
  }
  else
  {
    constexpr std::size_t axis = ColumnMajor ? OutputMdspan::rank() - Depth - 1 : Depth;
    using output_index = typename OutputMdspan::index_type;
    for (output_index i = 0; i < static_cast<output_index>(output.extent(axis)); ++i)
    {
      index[axis] = i;
      copy_elements<Depth + 1, ColumnMajor>(output, input, index);
    }
  }
}
} // namespace detail

/// \brief Report compile-time eligibility for reference CPU element-copy dispatch.
template <uni20::MutableMdspanLike OutputMdspan, uni20::MdspanLike InputMdspan>
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
template <uni20::MutableMdspanLike OutputMdspan, uni20::MdspanLike InputMdspan>
KernelAttempt try_kernel(CpuReferenceBackend, copy_op const&, OutputMdspan&& output, InputMdspan&& input)
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  static_assert(output_type::rank() == std::remove_cvref_t<InputMdspan>::rank());

  if constexpr (output_type::rank() > 0)
  {
    for (std::size_t axis = 0; axis < output_type::rank(); ++axis)
      CHECK_EQUAL(output.extent(axis), input.extent(axis));
  }

  if constexpr (uni20::MutableStridedMdspanLike<output_type> &&
                uni20::StridedMdspanLike<std::remove_cvref_t<InputMdspan>>)
  {
    cpu::detail::transform_strided<false>(
        output, [](auto&& value) -> decltype(auto) { return std::forward<decltype(value)>(value); }, input);
  }
  else
  {
    std::array<typename output_type::index_type, output_type::rank()> index{};
    if constexpr (std::same_as<typename output_type::layout_type, stdex::layout_left>)
      detail::copy_elements<0, true>(output, input, index);
    else
      detail::copy_elements<0, false>(output, input, index);
  }
  return KernelAttempt::success;
}

/// \brief Report eligibility for a host DeviceTensorView element copy.
template <uni20::MutableDeviceTensorView OutputTensor, uni20::DeviceTensorView InputTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<InputTensor>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, copy_op const&, OutputTensor&, InputTensor&)
{
  using output_span = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;
  using input_span = uni20::detail::host_read_tensor_mdspan_t<InputTensor>;
  constexpr auto acceptance =
      detail::backend_type_acceptance<CpuReferenceBackend, copy_op, output_span&, input_span&>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host tensor access and copy through the resulting mdspans.
template <uni20::MutableDeviceTensorView OutputTensor, uni20::DeviceTensorView InputTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<InputTensor>
KernelAttempt try_kernel(CpuReferenceBackend backend, copy_op const& operation, OutputTensor& output,
                         InputTensor const& input)
{
  auto output_access = acquire_host_write_access(output);
  auto input_access = acquire_host_read_access(input);
  auto output_span = output_access.mdspan();
  auto input_span = input_access.mdspan();
  return try_kernel(backend, operation, output_span, input_span);
}

} // namespace uni20::linalg
