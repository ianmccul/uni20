#pragma once

/**
 * \file copy_into.hpp
 * \ingroup tensor
 * \brief Backend-dispatched copies into existing mdspan and tensor outputs.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backends/cpu/copy.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/output.hpp>

#if UNI20_BACKEND_CUDA
#include <uni20/linalg/backends/cuda/copy.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/storage/vectorstorage.hpp>
#endif

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{
template <class Output, class Input>
concept CopySpans = MutableMdspanLike<Output> && MdspanLike<Input> &&
                    (std::remove_cvref_t<Output>::rank() == std::remove_cvref_t<Input>::rank());

template <class Output, class Input>
concept CopyTensors = MutableDeviceTensorView<Output> && DeviceTensorView<Input> &&
                      (device_tensor_mdspan_t<Output>::rank() == device_tensor_mdspan_t<Input>::rank());

#if UNI20_BACKEND_CUDA
template <class Output, class Input>
inline constexpr bool is_pageable_cuda_transfer = (std::same_as<tensor_storage_policy_t<Output>, CudaStorage> &&
                                                   std::same_as<tensor_storage_policy_t<Input>, VectorStorage>) ||
                                                  (std::same_as<tensor_storage_policy_t<Output>, VectorStorage> &&
                                                   std::same_as<tensor_storage_policy_t<Input>, CudaStorage>);
#endif

template <DeviceMdspanLike Output, DeviceMdspanLike Input>
[[nodiscard]] constexpr bool copy_extents_match(Output const& output, Input const& input) noexcept
{
  if constexpr (std::remove_cvref_t<Output>::rank() != std::remove_cvref_t<Input>::rank())
  {
    return false;
  }
  else
  {
    constexpr std::size_t rank = std::remove_cvref_t<Output>::rank();
    if constexpr (rank > 0)
    {
      for (std::size_t axis = 0; axis < rank; ++axis)
      {
        if (output.extent(axis) != input.extent(axis)) return false;
      }
    }
    return true;
  }
}

template <MdspanLike Span> [[nodiscard]] auto make_const_copy_mdspan(Span const& span)
{
  auto accessor = const_accessor(span.accessor());
  using accessor_type = decltype(accessor);
  using span_type = std::remove_cvref_t<Span>;
  using const_mdspan_type = stdex::mdspan<typename accessor_type::element_type, typename span_type::extents_type,
                                          typename span_type::layout_type, accessor_type>;
  return const_mdspan_type{span.data_handle(), span.mapping(), std::move(accessor)};
}

/// \brief Read-only TensorView facade for explicit-selector bare-mdspan copy.
template <MdspanLike Span, class BackendSelector> class CopyReadTensorView {
  public:
    using source_type = std::remove_cvref_t<Span>;
    using mdspan_type = decltype(make_const_copy_mdspan(std::declval<source_type const&>()));
    using storage_policy = void;
    using index_type = typename mdspan_type::index_type;

    CopyReadTensorView(source_type const& span, BackendSelector const& selector)
        : span_(make_const_copy_mdspan(span)), selector_(std::addressof(selector))
    {}

    [[nodiscard]] auto mdspan() const& -> mdspan_type const& { return span_; }

    [[nodiscard]] auto device_mdspan() const& -> mdspan_type const& { return span_; }

    [[nodiscard]] auto backend_selector() const -> BackendSelector const& { return *selector_; }

    [[nodiscard]] auto extents() const -> typename mdspan_type::extents_type const& { return span_.extents(); }

    [[nodiscard]] index_type extent(std::size_t axis) const { return span_.extent(axis); }

  private:
    mdspan_type span_;
    BackendSelector const* selector_;
};

/// \brief Mutable TensorView facade for explicit-selector bare-mdspan copy.
template <MutableMdspanLike Span, class BackendSelector> class CopyWriteTensorView {
  public:
    using mdspan_type = std::remove_cvref_t<Span>;
    using const_mdspan_type = decltype(make_const_copy_mdspan(std::declval<mdspan_type const&>()));
    using storage_policy = void;
    using index_type = typename mdspan_type::index_type;

    CopyWriteTensorView(mdspan_type& span, BackendSelector const& selector)
        : span_(std::addressof(span)), const_span_(make_const_copy_mdspan(span)), selector_(std::addressof(selector))
    {}

    [[nodiscard]] auto mdspan() & -> mdspan_type& { return *span_; }

    [[nodiscard]] auto mdspan() const& -> const_mdspan_type const& { return const_span_; }

    [[nodiscard]] auto device_mdspan() & -> mdspan_type& { return *span_; }

    [[nodiscard]] auto device_mdspan() const& -> const_mdspan_type const& { return const_span_; }

    [[nodiscard]] auto backend_selector() const -> BackendSelector const& { return *selector_; }

    [[nodiscard]] auto extents() const -> typename mdspan_type::extents_type const& { return span_->extents(); }

    [[nodiscard]] index_type extent(std::size_t axis) const { return span_->extent(axis); }

  private:
    mdspan_type* span_;
    const_mdspan_type const_span_;
    BackendSelector const* selector_;
};
} // namespace detail

/// \brief Copy between fixed-shape mdspan-like operands through an explicit selector.
/// \details Accessor semantics are observed by the selected backend, so a
///          conjugating input remains a lazy view until this explicit copy.
///          The operands are adapted to immediate tensor views before entering
///          operation-tag dispatch.
/// \pre Input and output do not destructively overlap.
template <class BackendSelector, class OutputMdspan, class InputMdspan>
  requires detail::CopySpans<OutputMdspan, InputMdspan>
void copy(BackendSelector&& selector, OutputMdspan&& output, InputMdspan&& input)
{
  ERROR_IF(!detail::copy_extents_match(output, input), "copy output shape does not match input shape");
  using selector_type = std::remove_cvref_t<BackendSelector>;
  using output_type = std::remove_cvref_t<OutputMdspan>;
  using input_type = std::remove_cvref_t<InputMdspan>;
  detail::CopyWriteTensorView<output_type, selector_type> output_view{output, selector};
  detail::CopyReadTensorView<input_type, selector_type> input_view{input, selector};
  static_assert(MutableTensorView<decltype(output_view)>);
  static_assert(TensorView<decltype(input_view)>);
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::copy_op{}, output_view, input_view);
}

/// \brief Copy into a resizable or already-compatible tensor through an explicit selector.
/// \details Shape preparation occurs before either resolved mdspan is acquired.
/// \pre Input and output do not destructively overlap.
template <class BackendSelector, class OutputTensor, class InputTensor>
  requires detail::CopyTensors<OutputTensor, InputTensor>
void copy(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input)
{
  prepare_output(output, input.extents());
  linalg::dispatch_kernel(std::forward<BackendSelector>(selector), linalg::copy_op{}, output, input);
}

/// \brief Copy into a tensor using the operands' default backend selector.
template <class OutputTensor, class InputTensor>
  requires detail::CopyTensors<OutputTensor, InputTensor>
void copy(OutputTensor&& output, InputTensor const& input)
{
  prepare_output(output, input.extents());
#if UNI20_BACKEND_CUDA
  if constexpr (detail::is_pageable_cuda_transfer<OutputTensor, InputTensor>)
  {
    copy(linalg::CudaReferenceBackend{}, output, input);
  }
  else
#endif
  {
    auto selector = linalg::select_backend(linalg::copy_op{}, output, input);
    copy(selector, output, input);
  }
}

/// \brief Assign tensor values through a mutable tensor alias descriptor.
/// \details Async alias assignment discovers this function through ADL. The
///          descriptor itself remains unchanged while `copy` writes its values.
template <MutableDeviceTensorView Output, DeviceTensorView Input>
  requires(device_tensor_mdspan_t<Output>::rank() == device_tensor_mdspan_t<Input>::rank())
void assign_through(Output& output, Input const& input)
{
  copy(output, input);
}

} // namespace uni20
