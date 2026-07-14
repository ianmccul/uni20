#pragma once

/**
 * \file reshape.hpp
 * \ingroup tensor
 * \brief Explicit aliasing, in-place, and owning tensor reshape operations.
 */

#include <uni20/common/mdspan.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/basic_tensor.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/shape.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

enum class ContiguousOrder
{
  row_major,
  column_major
};

template <class Index> [[nodiscard]] constexpr bool positive_stride_equals(Index stride, std::size_t expected)
{
  if constexpr (std::signed_integral<Index>)
    if (stride < 0) return false;
  return std::in_range<std::size_t>(stride) && static_cast<std::size_t>(stride) == expected;
}

template <StridedMdspan Span> [[nodiscard]] bool has_row_major_contiguous_mapping(Span const& span)
{
  std::size_t expected = 1;
  for (std::size_t axis = Span::rank(); axis > 0; --axis)
  {
    auto const extent = static_cast<std::size_t>(span.extent(axis - 1));
    if (extent > 1 && !positive_stride_equals(span.stride(axis - 1), expected)) return false;
    if (extent > 0)
    {
      if (expected > std::numeric_limits<std::size_t>::max() / extent) return false;
      expected *= extent;
    }
  }
  return true;
}

template <StridedMdspan Span> [[nodiscard]] bool has_column_major_contiguous_mapping(Span const& span)
{
  std::size_t expected = 1;
  for (std::size_t axis = 0; axis < Span::rank(); ++axis)
  {
    auto const extent = static_cast<std::size_t>(span.extent(axis));
    if (extent > 1 && !positive_stride_equals(span.stride(axis), expected)) return false;
    if (extent > 0)
    {
      if (expected > std::numeric_limits<std::size_t>::max() / extent) return false;
      expected *= extent;
    }
  }
  return true;
}

template <StridedMdspan Span> [[nodiscard]] auto contiguous_order(Span const& span) -> std::optional<ContiguousOrder>
{
  auto const logical_size = checked_element_count(span.extents());
  if constexpr (requires { span.mapping().is_unique(); })
    if (!span.mapping().is_unique()) return std::nullopt;
  if constexpr (requires { span.mapping().is_exhaustive(); })
    if (!span.mapping().is_exhaustive()) return std::nullopt;
  if constexpr (requires { span.mapping().required_span_size(); })
    if (static_cast<std::size_t>(span.mapping().required_span_size()) != logical_size) return std::nullopt;

  bool const row_major = has_row_major_contiguous_mapping(span);
  bool const column_major = has_column_major_contiguous_mapping(span);
  if constexpr (std::same_as<typename Span::layout_type, stdex::layout_left>)
  {
    if (column_major) return ContiguousOrder::column_major;
  }
  else if constexpr (std::same_as<typename Span::layout_type, stdex::layout_right>)
  {
    if (row_major) return ContiguousOrder::row_major;
  }

  if (row_major) return ContiguousOrder::row_major;
  if (column_major) return ContiguousOrder::column_major;
  return std::nullopt;
}

template <class Extents>
[[nodiscard]] auto make_reshape_mapping(Extents const& extents, ContiguousOrder order) ->
    typename stdex::layout_stride::template mapping<Extents>
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> strides{};
  if (checked_element_count(extents) == 0)
  {
    strides.fill(index_type{1});
    return {extents, strides};
  }

  std::size_t running = 1;

  auto set_stride = [&](std::size_t axis) {
    ERROR_IF(!std::in_range<index_type>(running), "reshape stride is not representable");
    strides[axis] = static_cast<index_type>(running);
    auto const extent = static_cast<std::size_t>(extents.extent(axis));
    auto const factor = extent == 0 ? std::size_t{1} : extent;
    ERROR_IF(running > std::numeric_limits<std::size_t>::max() / factor, "reshape stride overflows size_t");
    running *= factor;
  };

  if (order == ContiguousOrder::row_major)
  {
    for (std::size_t axis = Extents::rank(); axis > 0; --axis)
      set_stride(axis - 1);
  }
  else
  {
    for (std::size_t axis = 0; axis < Extents::rank(); ++axis)
      set_stride(axis);
  }
  return {extents, strides};
}

template <class LayoutPolicy>
inline constexpr bool supported_reshape_layout =
    std::same_as<LayoutPolicy, stdex::layout_left> || std::same_as<LayoutPolicy, stdex::layout_right> ||
    std::same_as<LayoutPolicy, stdex::layout_stride>;

template <class LayoutPolicy, class Extents>
[[nodiscard]] auto make_owning_reshape_mapping(Extents const& extents, ContiguousOrder order) ->
    typename LayoutPolicy::template mapping<Extents>
  requires supported_reshape_layout<LayoutPolicy>
{
  if constexpr (std::same_as<LayoutPolicy, stdex::layout_left>)
  {
    ERROR_IF(order != ContiguousOrder::column_major,
             "column-major tensor layout cannot preserve a row-major reshape sequence");
    return typename LayoutPolicy::template mapping<Extents>{extents};
  }
  else if constexpr (std::same_as<LayoutPolicy, stdex::layout_right>)
  {
    ERROR_IF(order != ContiguousOrder::row_major,
             "row-major tensor layout cannot preserve a column-major reshape sequence");
    return typename LayoutPolicy::template mapping<Extents>{extents};
  }
  else
  {
    return make_reshape_mapping(extents, order);
  }
}

} // namespace detail

/// \brief Tensor-level descriptor owning a reshaped mdspan and backend selector.
/// \details The descriptor aliases the same element storage as its source.
///          Mutability follows the preserved source accessor.
template <SpanLike Mdspan, class StoragePolicy, class BackendSelector> class ReshapedTensor {
  public:
    using mdspan_type = Mdspan;
    using storage_policy = StoragePolicy;
    using backend_selector_type = BackendSelector;
    using element_type = typename mdspan_type::element_type;
    using value_type = typename mdspan_type::value_type;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;
    using layout_type = typename mdspan_type::layout_type;
    using mapping_type = typename mdspan_type::mapping_type;

    /// \brief Construct from a reshaped mdspan and the source backend selector.
    constexpr ReshapedTensor(mdspan_type span, backend_selector_type selector)
        : span_(std::move(span)), selector_(std::move(selector))
    {}

    /// \brief Return the source storage's backend selector.
    [[nodiscard]] constexpr auto backend_selector() const -> backend_selector_type { return selector_; }

    /// \brief Resolve the stored reshaped mdspan descriptor.
    [[nodiscard]] constexpr auto mdspan() const -> mdspan_type { return span_; }

    /// \brief Return the reshaped extents.
    [[nodiscard]] constexpr auto extents() const noexcept -> extents_type const& { return span_.extents(); }

    /// \brief Return one reshaped extent.
    [[nodiscard]] constexpr auto extent(std::size_t axis) const noexcept -> index_type { return span_.extent(axis); }

    /// \brief Evaluate or assign one reshaped element according to accessor mutability.
    template <class... Index>
      requires(sizeof...(Index) == extents_type::rank())
    constexpr decltype(auto) operator[](Index... indices) const
    {
      return span_[indices...];
    }

  private:
    mdspan_type span_;
    [[no_unique_address]] backend_selector_type selector_;
};

/// \brief Return a no-copy reshaped mdspan preserving the source handle and accessor.
/// \details The source must have a unique, exhaustive canonical row-major or
///          column-major mapping. The output preserves that contiguous order.
template <StridedMdspan Span, std::integral... Extents>
[[nodiscard]] auto reshape_view(Span&& source, Extents... requested_extents)
{
  using source_type = std::remove_cvref_t<Span>;
  auto const source_size = detail::checked_element_count(source.extents());
  auto new_extents = detail::make_reshape_extents(source_size, requested_extents...);
  auto const order = detail::contiguous_order(source);
  ERROR_IF(!order.has_value(), "reshape_view requires a unique, exhaustive contiguous source mapping");

  using extents_type = decltype(new_extents);
  using layout_type = stdex::layout_stride;
  using accessor_type = typename source_type::accessor_type;
  using element_type = typename source_type::element_type;
  using result_type = stdex::mdspan<element_type, extents_type, layout_type, accessor_type>;
  auto mapping = detail::make_reshape_mapping(new_extents, *order);
  return result_type{source.data_handle(), std::move(mapping), source.accessor()};
}

/// \brief Return a tensor-level no-copy reshape alias of a tensor lvalue.
/// \details Rvalue tensors are rejected because the returned descriptor does
///          not extend the lifetime of addressable source storage.
template <TensorView Tensor, std::integral... Extents>
  requires(std::is_lvalue_reference_v<Tensor &&> && StridedTensorView<Tensor>)
[[nodiscard]] auto reshape_view(Tensor&& tensor, Extents... requested_extents)
{
  auto span = reshape_view(tensor.mdspan(), requested_extents...);
  using span_type = decltype(span);
  using storage_policy = detail::tensor_storage_policy_t<std::remove_cvref_t<Tensor>>;
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  return ReshapedTensor<span_type, storage_policy, selector_type>{std::move(span), tensor.backend_selector()};
}

/// \brief Change an owning tensor's shape without moving or reallocating its storage.
/// \details In-place reshape preserves the source's canonical contiguous
///          order and is available only when the compile-time rank is
///          unchanged. Existing mdspan descriptors retain their old mappings.
template <Scalar ElementType, class TensorExtents, class StoragePolicy, class LayoutPolicy, class AccessorFactory,
          std::integral... Extents>
  requires(sizeof...(Extents) == TensorExtents::rank() && detail::supported_reshape_layout<LayoutPolicy>)
void reshape_inplace(BasicTensor<ElementType, TensorExtents, StoragePolicy, LayoutPolicy, AccessorFactory>& tensor,
                     Extents... requested_extents)
{
  auto source = tensor.mdspan();
  auto const source_size = detail::checked_element_count(source.extents());
  auto requested = detail::make_reshape_extents(source_size, requested_extents...);
  auto new_extents = detail::convert_reshape_extents<TensorExtents>(requested);
  auto const order = detail::contiguous_order(source);
  ERROR_IF(!order.has_value(), "reshape_inplace requires a unique, exhaustive contiguous source mapping");
  tensor.replace_mapping(detail::make_owning_reshape_mapping<LayoutPolicy>(new_extents, *order));
}

/// \brief Return an owning reshape of a configurable dense tensor.
/// \details Passing an lvalue copies it into the by-value parameter; passing
///          an rvalue transfers it. The resulting tensor then adopts that
///          storage under the new shape without another element copy.
template <Scalar ElementType, class SourceExtents, class StoragePolicy, class LayoutPolicy, class AccessorFactory,
          std::integral... Extents>
  requires detail::supported_reshape_layout<LayoutPolicy>
[[nodiscard]] auto reshape(BasicTensor<ElementType, SourceExtents, StoragePolicy, LayoutPolicy, AccessorFactory> tensor,
                           Extents... requested_extents)
{
  auto source = tensor.mdspan();
  auto const source_size = detail::checked_element_count(source.extents());
  auto new_extents = detail::make_reshape_extents(source_size, requested_extents...);
  auto const order = detail::contiguous_order(source);
  ERROR_IF(!order.has_value(), "reshape requires a unique, exhaustive contiguous source mapping");

  using source_type = BasicTensor<ElementType, SourceExtents, StoragePolicy, LayoutPolicy, AccessorFactory>;
  using result_type = typename source_type::template rebind_extents_type<decltype(new_extents)>;
  auto mapping = detail::make_owning_reshape_mapping<LayoutPolicy>(new_extents, *order);
  auto accessor_factory = std::move(tensor).release_accessor_factory();
  auto storage = std::move(tensor).release_storage();
  return result_type::adopt_storage(std::move(mapping), std::move(storage), std::move(accessor_factory));
}

} // namespace uni20
