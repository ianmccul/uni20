#pragma once

/**
 * \file transform.hpp
 * \ingroup linalg
 * \brief Reference CPU backend for generic elementwise transforms.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/cpu/strided_transform.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{

template <class Mdspan> consteval bool transform_value_is_readable()
{
  using span_type = std::remove_cvref_t<Mdspan>;
  return requires(typename span_type::reference element) { static_cast<typename span_type::value_type>(element); };
}

template <class OutputMdspan, class Function, class... InputMdspans> consteval bool overwrite_transform_is_supported()
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  if constexpr (!uni20::MutableMdspanLike<output_type> || sizeof...(InputMdspans) == 0 ||
                (!(uni20::MdspanLike<std::remove_cvref_t<InputMdspans>> && ...)) ||
                !((output_type::rank() == std::remove_cvref_t<InputMdspans>::rank()) && ...))
  {
    return false;
  }
  else
  {
    return (transform_value_is_readable<InputMdspans>() && ...) &&
           std::invocable<Function const&, typename std::remove_cvref_t<InputMdspans>::value_type...> &&
           requires(typename output_type::reference output, Function const& function) {
             output = std::invoke(function, std::declval<typename std::remove_cvref_t<InputMdspans>::value_type>()...);
           };
  }
}

template <class OutputMdspan, class Function, class... InputMdspans> consteval bool update_transform_is_supported()
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  if constexpr (!uni20::MutableMdspanLike<output_type> ||
                (!(uni20::MdspanLike<std::remove_cvref_t<InputMdspans>> && ...)) ||
                !((output_type::rank() == std::remove_cvref_t<InputMdspans>::rank()) && ...))
  {
    return false;
  }
  else
  {
    return transform_value_is_readable<OutputMdspan>() && (transform_value_is_readable<InputMdspans>() && ...) &&
           std::invocable<Function const&, typename output_type::value_type,
                          typename std::remove_cvref_t<InputMdspans>::value_type...> &&
           requires(typename output_type::reference output, Function const& function) {
             output = std::invoke(function, std::declval<typename output_type::value_type>(),
                                  std::declval<typename std::remove_cvref_t<InputMdspans>::value_type>()...);
           };
  }
}

template <class OutputMdspan, class... InputMdspans>
void check_transform_extents(OutputMdspan const& output, InputMdspans const&... inputs)
{
  if constexpr (sizeof...(InputMdspans) > 0)
  {
    constexpr std::size_t rank = std::remove_cvref_t<OutputMdspan>::rank();
    if constexpr (rank > 0)
    {
      auto check_input = [&](auto const& input) {
        for (std::size_t axis = 0; axis < rank; ++axis)
          CHECK_EQUAL(output.extent(axis), input.extent(axis));
      };
      (check_input(inputs), ...);
    }
  }
}

template <class Mdspan, class Index, std::size_t... Axis>
decltype(auto) transform_element(Mdspan&& span, Index const& index, std::index_sequence<Axis...>)
{
  using span_type = std::remove_cvref_t<Mdspan>;
  using index_type = typename span_type::index_type;
  return std::forward<Mdspan>(span)[static_cast<index_type>(index[Axis])...];
}

template <bool ReadsOutput, class Operation, class OutputMdspan, class Index, class... InputMdspans>
void apply_transform_element(Operation const& operation, OutputMdspan& output, Index const& index,
                             InputMdspans&... inputs)
{
  constexpr auto axes = std::make_index_sequence<std::remove_cvref_t<OutputMdspan>::rank()>{};
  auto&& output_element = transform_element(output, index, axes);
  if constexpr (ReadsOutput)
  {
    using output_value = typename std::remove_cvref_t<OutputMdspan>::value_type;
    output_element = std::invoke(
        operation.function, static_cast<output_value>(output_element),
        static_cast<typename std::remove_cvref_t<InputMdspans>::value_type>(transform_element(inputs, index, axes))...);
  }
  else
  {
    output_element = std::invoke(
        operation.function,
        static_cast<typename std::remove_cvref_t<InputMdspans>::value_type>(transform_element(inputs, index, axes))...);
  }
}

template <std::size_t Axis, bool ReadsOutput, class Operation, class OutputMdspan, class Index, class... InputMdspans>
void transform_elements(Operation const& operation, OutputMdspan& output, Index& index, InputMdspans&... inputs)
{
  if constexpr (Axis == std::remove_cvref_t<OutputMdspan>::rank())
  {
    apply_transform_element<ReadsOutput>(operation, output, index, inputs...);
  }
  else
  {
    using output_index = typename std::remove_cvref_t<OutputMdspan>::index_type;
    for (output_index i = 0; i < static_cast<output_index>(output.extent(Axis)); ++i)
    {
      index[Axis] = i;
      transform_elements<Axis + 1, ReadsOutput>(operation, output, index, inputs...);
    }
  }
}

template <bool ReadsOutput, class Operation, class OutputMdspan, class... InputMdspans>
void reference_transform(Operation const& operation, OutputMdspan& output, InputMdspans&... inputs)
{
  using output_type = std::remove_cvref_t<OutputMdspan>;
  if constexpr (uni20::MutableStridedMdspanLike<output_type> &&
                (uni20::StridedMdspanLike<std::remove_cvref_t<InputMdspans>> && ...))
  {
    if constexpr (ReadsOutput)
    {
      auto invoke_with_values = [&operation](auto&& output_value, auto&&... input_values) {
        return std::invoke(operation.function, static_cast<typename output_type::value_type>(output_value),
                           static_cast<typename std::remove_cvref_t<InputMdspans>::value_type>(input_values)...);
      };
      cpu::detail::transform_strided<true>(output, std::cref(invoke_with_values), inputs...);
    }
    else
    {
      auto invoke_with_values = [&operation](auto&&... input_values) {
        return std::invoke(operation.function,
                           static_cast<typename std::remove_cvref_t<InputMdspans>::value_type>(input_values)...);
      };
      cpu::detail::transform_strided<false>(output, std::cref(invoke_with_values), inputs...);
    }
  }
  else
  {
    std::array<typename output_type::index_type, output_type::rank()> index{};
    transform_elements<0, ReadsOutput>(operation, output, index, inputs...);
  }
}

} // namespace detail

/// \brief Report compile-time eligibility for reference CPU overwrite transforms.
template <class Function, uni20::MutableMdspanLike OutputMdspan, uni20::MdspanLike... InputMdspans>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, transform_op<Function> const&, OutputMdspan&,
                                    InputMdspans&...)
{
  if constexpr (detail::overwrite_transform_is_supported<OutputMdspan, Function, InputMdspans...>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Apply a generic accessor-respecting elementwise overwrite transform.
/// \pre Output and inputs have equal extents and output does not overlap an input.
template <class Function, class OutputMdspan, class... InputMdspans>
KernelAttempt try_kernel(CpuReferenceBackend, transform_op<Function> const& operation, OutputMdspan&& output,
                         InputMdspans&&... inputs)
{
  detail::check_transform_extents(output, inputs...);
  detail::reference_transform<false>(operation, output, inputs...);
  return KernelAttempt::success;
}

/// \brief Report compile-time eligibility for reference CPU update transforms.
template <class Function, uni20::MutableMdspanLike OutputMdspan, uni20::MdspanLike... InputMdspans>
consteval auto kernel_accepts_types(CpuReferenceBackend const&, transform_inplace_op<Function> const&, OutputMdspan&,
                                    InputMdspans&...)
{
  if constexpr (detail::update_transform_is_supported<OutputMdspan, Function, InputMdspans...>())
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Apply a generic accessor-respecting elementwise update transform.
/// \pre Output and inputs have equal extents and output does not overlap an input.
template <class Function, class OutputMdspan, class... InputMdspans>
KernelAttempt try_kernel(CpuReferenceBackend, transform_inplace_op<Function> const& operation, OutputMdspan&& output,
                         InputMdspans&&... inputs)
{
  detail::check_transform_extents(output, inputs...);
  detail::reference_transform<true>(operation, output, inputs...);
  return KernelAttempt::success;
}

/// \brief Report eligibility for host DeviceTensorView overwrite transforms.
template <class Function, uni20::MutableDeviceTensorView OutputTensor, uni20::DeviceTensorView... InputTensors>
  requires uni20::detail::HostWritableTensor<OutputTensor> && (uni20::detail::HostReadableTensor<InputTensors> && ...)
consteval auto kernel_accepts_types(CpuReferenceBackend const&, transform_op<Function> const&, OutputTensor&,
                                    InputTensors&...)
{
  using output_span = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;
  constexpr auto acceptance =
      detail::backend_type_acceptance<CpuReferenceBackend, transform_op<Function>, output_span&,
                                      uni20::detail::host_read_tensor_mdspan_t<InputTensors>&...>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host tensor access and apply an overwrite transform.
template <class Function, uni20::MutableDeviceTensorView OutputTensor, uni20::DeviceTensorView... InputTensors>
  requires uni20::detail::HostWritableTensor<OutputTensor> && (uni20::detail::HostReadableTensor<InputTensors> && ...)
KernelAttempt try_kernel(CpuReferenceBackend backend, transform_op<Function> const& operation, OutputTensor& output,
                         InputTensors const&... inputs)
{
  auto output_access = acquire_host_write_access(output);
  auto input_accesses = std::tuple{acquire_host_read_access(inputs)...};
  return std::apply(
      [&](auto&... input_access) {
        auto output_span = output_access.mdspan();
        return try_kernel(backend, operation, output_span, input_access.mdspan()...);
      },
      input_accesses);
}

/// \brief Report eligibility for host DeviceTensorView update transforms.
template <class Function, uni20::MutableDeviceTensorView OutputTensor, uni20::DeviceTensorView... InputTensors>
  requires uni20::detail::HostWritableTensor<OutputTensor> && (uni20::detail::HostReadableTensor<InputTensors> && ...)
consteval auto kernel_accepts_types(CpuReferenceBackend const&, transform_inplace_op<Function> const&, OutputTensor&,
                                    InputTensors&...)
{
  using output_span = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;
  constexpr auto acceptance =
      detail::backend_type_acceptance<CpuReferenceBackend, transform_inplace_op<Function>, output_span&,
                                      uni20::detail::host_read_tensor_mdspan_t<InputTensors>&...>();
  if constexpr (acceptance == KernelTypeAcceptance::yes)
    return kernel_types_yes;
  else
    return kernel_types_no;
}

/// \brief Resolve host tensor access and apply an update transform.
template <class Function, uni20::MutableDeviceTensorView OutputTensor, uni20::DeviceTensorView... InputTensors>
  requires uni20::detail::HostWritableTensor<OutputTensor> && (uni20::detail::HostReadableTensor<InputTensors> && ...)
KernelAttempt try_kernel(CpuReferenceBackend backend, transform_inplace_op<Function> const& operation,
                         OutputTensor& output, InputTensors const&... inputs)
{
  auto output_access = acquire_host_write_access(output);
  auto input_accesses = std::tuple{acquire_host_read_access(inputs)...};
  return std::apply(
      [&](auto&... input_access) {
        auto output_span = output_access.mdspan();
        return try_kernel(backend, operation, output_span, input_access.mdspan()...);
      },
      input_accesses);
}

} // namespace uni20::linalg
