#pragma once

/**
 * \file copy.hpp
 * \ingroup tensor
 * \brief Backend-dispatched tensor copies and inferred owning materialization.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/cpu/copy.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{
template <class Output, class Input>
concept CopySpans = MutableSpanLike<Output> && SpanLike<Input> &&
                    (std::remove_cvref_t<Output>::rank() == std::remove_cvref_t<Input>::rank());

template <class Output, class Input>
concept CopyTensors = MutableTensorView<Output> && TensorView<Input> &&
                      (tensor_mdspan_t<Output>::rank() == tensor_mdspan_t<Input>::rank());

template <SpanLike Output, SpanLike Input>
[[nodiscard]] constexpr bool copy_extents_match(Output const& output, Input const& input) noexcept
{
  if constexpr (std::remove_cvref_t<Output>::rank() != std::remove_cvref_t<Input>::rank())
  {
    return false;
  }
  else
  {
    for (std::size_t axis = 0; axis < std::remove_cvref_t<Output>::rank(); ++axis)
    {
      if (output.extent(axis) != input.extent(axis)) return false;
    }
    return true;
  }
}

template <class RequestedLayout, class InputMdspan>
using materialized_layout_t =
    std::conditional_t<std::is_void_v<RequestedLayout>, typename InputMdspan::layout_type, RequestedLayout>;
} // namespace detail

/// \brief Copy between fixed-shape mdspan-like operands through an explicit selector.
/// \details Accessor semantics are observed by the selected backend, so a
///          conjugating input remains a lazy view until this explicit copy.
/// \pre Input and output do not destructively overlap.
template <class BackendSelector, class OutputMdspan, class InputMdspan>
  requires detail::CopySpans<OutputMdspan, InputMdspan>
void copy(BackendSelector&& selector, OutputMdspan&& output, InputMdspan&& input)
{
  ERROR_IF(!detail::copy_extents_match(output, input), "copy output shape does not match input shape");
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::copy_op{},
                          std::forward<OutputMdspan>(output), std::forward<InputMdspan>(input));
}

/// \brief Copy into a resizable or already-compatible tensor through an explicit selector.
/// \details Shape preparation occurs before either resolved mdspan is acquired.
/// \pre Input and output do not destructively overlap.
template <class BackendSelector, class OutputTensor, class InputTensor>
  requires detail::CopyTensors<OutputTensor, InputTensor>
void copy(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input)
{
  ensure_shape(output, input.extents());
  auto output_span = output.mdspan();
  auto input_span = input.mdspan();
  copy(std::forward<BackendSelector>(selector), output_span, input_span);
}

/// \brief Copy into a tensor using the operands' default backend selector.
template <class OutputTensor, class InputTensor>
  requires detail::CopyTensors<OutputTensor, InputTensor>
void copy(OutputTensor&& output, InputTensor const& input)
{
  ensure_shape(output, input.extents());
  auto selector = linalg::select_backend(linalg::copy_op{}, output, input);
  auto output_span = output.mdspan();
  auto input_span = input.mdspan();
  copy(selector, output_span, input_span);
}

/// \brief Assign tensor values through a mutable tensor alias descriptor.
/// \details Async alias assignment discovers this function through ADL. The
///          descriptor itself remains unchanged while `copy` writes its values.
template <MutableTensorView Output, TensorView Input>
  requires(tensor_mdspan_t<Output>::rank() == tensor_mdspan_t<Input>::rank())
void assign_through(Output& output, Input const& input)
{
  copy(output, input);
}

/// \brief Materialize a bare mdspan-like view as an inferred owning host tensor.
/// \details The caller supplies a backend selector because a bare mdspan does
///          not carry storage-domain policy. By default the output keeps the
///          source layout type; an explicit `RequestedLayout` may request a
///          different owning layout. The result has the input's compile-time
///          rank with runtime extents on every axis.
template <class RequestedLayout = void, class BackendSelector, SpanLike InputMdspan>
[[nodiscard]] auto make_tensor(BackendSelector&& selector, InputMdspan&& input)
{
  using input_type = std::remove_cvref_t<InputMdspan>;
  using layout_type = detail::materialized_layout_t<RequestedLayout, input_type>;
  using result_type =
      Tensor<std::remove_cv_t<typename input_type::element_type>, input_type::rank(), VectorStorage, layout_type>;

  result_type result(detail::convert_tensor_extents<typename result_type::extents_type>(input.extents()));
  auto output_span = result.mdspan();
  copy(std::forward<BackendSelector>(selector), output_span, std::forward<InputMdspan>(input));
  return result;
}

/// \brief Materialize a tensor view as an inferred owning host tensor.
/// \details This is the explicit eager boundary for lazy views such as
///          `conj(input)`. The result preserves the resolved input layout type
///          unless `RequestedLayout` is supplied, and uses runtime extents on
///          every axis.
template <class RequestedLayout = void, TensorView InputTensor> [[nodiscard]] auto make_tensor(InputTensor const& input)
{
  using input_mdspan = tensor_mdspan_t<InputTensor>;
  using layout_type = detail::materialized_layout_t<RequestedLayout, input_mdspan>;
  using result_type = Tensor<tensor_element_t<InputTensor>, input_mdspan::rank(), VectorStorage, layout_type>;

  result_type result(detail::convert_tensor_extents<typename result_type::extents_type>(input.extents()));
  copy(result, input);
  return result;
}

/// \brief Materialize an owning reshape of a non-owning or generated tensor view.
/// \details The source is first represented by `reshape_view`; materialization
///          then copies its logical values into an owning host tensor. Owning
///          `BasicTensor` operands use the move-aware overload in `reshape.hpp`.
template <StridedTensorView InputTensor, std::integral... Extents>
  requires(!OwningTensor<InputTensor>)
[[nodiscard]] auto reshape(InputTensor const& input, Extents... requested_extents)
{
  auto view = reshape_view(input, requested_extents...);
  return make_tensor(view);
}

} // namespace uni20
