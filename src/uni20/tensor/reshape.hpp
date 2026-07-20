#pragma once

/**
 * \file reshape.hpp
 * \ingroup tensor
 * \brief Explicit aliasing, in-place, and owning tensor reshape operations.
 */

#include <uni20/common/trace.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/tensor/basic_tensor.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/shape.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class LayoutPolicy>
concept CanonicalReshapeLayout =
    std::same_as<LayoutPolicy, stdex::layout_left> || std::same_as<LayoutPolicy, stdex::layout_right>;

template <class Span>
concept CanonicallyLaidOutMdspan =
    StridedMdspan<Span> && CanonicalReshapeLayout<typename std::remove_cvref_t<Span>::layout_type>;

template <class T>
concept CanonicallyLaidOutOwningTensor =
    OwningTensor<T> && MutableStridedTensorView<T> && CanonicalReshapeLayout<typename tensor_mdspan_t<T>::layout_type>;

template <class T, class NewExtents>
using reshape_rebind_t = typename std::remove_cvref_t<T>::template rebind_extents_type<NewExtents>;

template <class T, class NewExtents>
concept StorageTransferReshapableTensor =
    CanonicallyLaidOutOwningTensor<T> &&
    requires(std::remove_cvref_t<T>&& tensor, typename std::remove_cvref_t<T>::storage_type storage,
             typename std::remove_cvref_t<T>::accessor_factory_type accessor_factory,
             typename reshape_rebind_t<T, NewExtents>::mapping_type mapping) {
      { std::move(tensor).release_storage() } -> std::same_as<typename std::remove_cvref_t<T>::storage_type>;
      {
        std::move(tensor).release_accessor_factory()
      } -> std::same_as<typename std::remove_cvref_t<T>::accessor_factory_type>;
      {
        reshape_rebind_t<T, NewExtents>::adopt_storage(std::move(mapping), std::move(storage),
                                                       std::move(accessor_factory))
      } -> std::same_as<reshape_rebind_t<T, NewExtents>>;
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

template <CanonicalReshapeLayout LayoutPolicy, StridedMdspan Span>
[[nodiscard]] bool has_canonical_contiguous_mapping(Span const& span)
{
  auto const logical_size = checked_element_count(span.extents());
  if constexpr (requires { span.mapping().is_unique(); })
    if (!span.mapping().is_unique()) return false;
  if constexpr (requires { span.mapping().is_exhaustive(); })
    if (!span.mapping().is_exhaustive()) return false;
  if constexpr (requires { span.mapping().required_span_size(); })
  {
    auto const required_size = span.mapping().required_span_size();
    if (!std::in_range<std::size_t>(required_size) || static_cast<std::size_t>(required_size) != logical_size)
      return false;
  }

  // An empty or scalar index space has no observable traversal-order distinction.
  if (logical_size <= 1) return true;

  if constexpr (std::same_as<LayoutPolicy, stdex::layout_left>)
    return has_column_major_contiguous_mapping(span);
  else
    return has_row_major_contiguous_mapping(span);
}

template <CanonicalReshapeLayout LayoutPolicy, StridedMdspan Span>
void require_canonical_contiguous_mapping(Span const& span)
{
  if constexpr (std::same_as<LayoutPolicy, stdex::layout_left>)
  {
    ERROR_IF(!has_canonical_contiguous_mapping<LayoutPolicy>(span),
             "reshape requires a unique, exhaustive, canonical column-major source mapping");
  }
  else
  {
    ERROR_IF(!has_canonical_contiguous_mapping<LayoutPolicy>(span),
             "reshape requires a unique, exhaustive, canonical row-major source mapping");
  }
}

template <CanonicalReshapeLayout LayoutPolicy, StridedMdspan Span, std::integral... Extents>
[[nodiscard]] auto make_reshape_view(Span&& source, Extents... requested_extents)
{
  using source_type = std::remove_cvref_t<Span>;
  require_canonical_contiguous_mapping<LayoutPolicy>(source);
  auto const source_size = checked_element_count(source.extents());
  auto new_extents = make_reshape_extents(source_size, requested_extents...);
  using extents_type = decltype(new_extents);
  using accessor_type = typename source_type::accessor_type;
  using element_type = typename source_type::element_type;
  using result_type = stdex::mdspan<element_type, extents_type, LayoutPolicy, accessor_type>;
  using mapping_type = typename LayoutPolicy::template mapping<extents_type>;
  return result_type{source.data_handle(), mapping_type{new_extents}, source.accessor()};
}

template <CanonicalReshapeLayout LayoutPolicy, TensorView Tensor, std::integral... Extents>
  requires(std::is_lvalue_reference_v<Tensor &&> && StridedTensorView<Tensor>)
[[nodiscard]] auto make_tensor_reshape_view(Tensor&& tensor, Extents... requested_extents);

} // namespace detail

/// \brief Return a no-copy mdspan reshape preserving a canonical source layout.
/// \details Automatic order selection is available only when the source's
///          static layout is `layout_left` or `layout_right`.
template <detail::CanonicallyLaidOutMdspan Span, std::integral... Extents>
[[nodiscard]] auto reshape_view(Span&& source, Extents... requested_extents)
{
  using layout_type = typename std::remove_cvref_t<Span>::layout_type;
  return detail::make_reshape_view<layout_type>(std::forward<Span>(source), requested_extents...);
}

/// \brief Return a no-copy column-major reshape of a compatible strided mdspan.
/// \details Singleton-axis strides do not select an order. All other strides
///          must describe a unique, exhaustive canonical column-major mapping.
template <StridedMdspan Span, std::integral... Extents>
[[nodiscard]] auto reshape_view_left(Span&& source, Extents... requested_extents)
{
  return detail::make_reshape_view<stdex::layout_left>(std::forward<Span>(source), requested_extents...);
}

/// \brief Return a no-copy row-major reshape of a compatible strided mdspan.
/// \details Singleton-axis strides do not select an order. All other strides
///          must describe a unique, exhaustive canonical row-major mapping.
template <StridedMdspan Span, std::integral... Extents>
[[nodiscard]] auto reshape_view_right(Span&& source, Extents... requested_extents)
{
  return detail::make_reshape_view<stdex::layout_right>(std::forward<Span>(source), requested_extents...);
}

/// \brief Tensor-level descriptor owning a reshaped mdspan and backend selector.
/// \details The descriptor aliases the same element storage as its source.
///          Mutability follows the preserved source accessor.
template <SpanLike Mdspan, class StoragePolicy, class BackendSelector> class ReshapedTensor {
  public:
    using mdspan_type = Mdspan;
    using storage_policy = StoragePolicy;
    using backend_selector_type = BackendSelector;
    using async_alias_tag = void;
    using element_type = typename mdspan_type::element_type;
    using value_type = typename mdspan_type::value_type;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;
    using layout_type = typename mdspan_type::layout_type;
    using mapping_type = typename mdspan_type::mapping_type;
    using const_accessor_type = const_accessor_t<typename mdspan_type::accessor_type>;
    using const_element_type = std::add_const_t<std::remove_const_t<element_type>>;
    using const_mdspan_type = stdex::mdspan<const_element_type, extents_type, layout_type, const_accessor_type>;

    /// \brief Construct from a reshaped mdspan and source backend selector.
    constexpr ReshapedTensor(mdspan_type span, backend_selector_type selector)
        : span_(std::move(span)), selector_(std::move(selector))
    {}

    /// \brief Return the source storage's backend selector.
    [[nodiscard]] constexpr auto backend_selector() const -> backend_selector_type { return selector_; }

    /// \brief Resolve the stored reshaped mdspan descriptor for mutable access.
    [[nodiscard]] constexpr auto mdspan() & -> mdspan_type { return span_; }

    /// \brief Resolve the stored reshaped mdspan descriptor for read-only access.
    [[nodiscard]] constexpr auto mdspan() const& -> const_mdspan_type
    {
      return const_mdspan_type{span_.data_handle(), span_.mapping(), const_accessor(span_.accessor())};
    }

    /// \brief Return the reshaped extents.
    [[nodiscard]] constexpr auto extents() const noexcept -> extents_type const& { return span_.extents(); }

    /// \brief Return one reshaped extent.
    [[nodiscard]] constexpr auto extent(std::size_t axis) const noexcept -> index_type { return span_.extent(axis); }

    /// \brief Evaluate or assign one reshaped element according to accessor mutability.
    template <class... Index>
      requires(sizeof...(Index) == extents_type::rank())
    constexpr decltype(auto) operator[](Index... indices)
    {
      return span_[indices...];
    }

    /// \brief Evaluate one reshaped element through the read-only accessor.
    template <class... Index>
      requires detail::MdspanIndexPack<mdspan_type, Index...>
    constexpr decltype(auto) operator[](Index... indices) const
    {
      return const_access(span_, std::move(indices)...);
    }

  private:
    mdspan_type span_;
    [[no_unique_address]] backend_selector_type selector_;
};

/// \brief Tensor-level descriptor that resolves a reshape through its parent tensor.
/// \details The descriptor aliases the same element storage as its source and
///          preserves source mutability. Parent indirection allows an async
///          alias to bind reserved storage before the parent value is constructed.
template <class Tensor, detail::CanonicalReshapeLayout LayoutPolicy, std::integral... RequestedExtents>
  requires TensorView<Tensor> && StridedTensorView<Tensor>
class IndirectReshapedTensorView {
  public:
    using tensor_type = std::remove_reference_t<Tensor>;
    using storage_policy = detail::tensor_storage_policy_t<std::remove_cv_t<tensor_type>>;
    using backend_selector_type = std::remove_cvref_t<decltype(std::declval<tensor_type const&>().backend_selector())>;
    using requested_extents_type = std::tuple<RequestedExtents...>;
    using mdspan_type = decltype(detail::make_reshape_view<LayoutPolicy>(std::declval<tensor_type&>().mdspan(),
                                                                         std::declval<RequestedExtents>()...));
    using const_mdspan_type = decltype(detail::make_reshape_view<LayoutPolicy>(
        std::declval<tensor_type const&>().mdspan(), std::declval<RequestedExtents>()...));
    using async_alias_tag = void;
    using element_type = typename mdspan_type::element_type;
    using value_type = typename mdspan_type::value_type;
    using extents_type = typename mdspan_type::extents_type;
    using index_type = typename mdspan_type::index_type;
    using layout_type = typename mdspan_type::layout_type;
    using mapping_type = typename mdspan_type::mapping_type;

    /// \brief Bind and immediately validate a reshape of an existing tensor.
    IndirectReshapedTensorView(tensor_type& tensor, RequestedExtents... requested_extents)
        : IndirectReshapedTensorView(std::addressof(tensor), requested_extents...)
    {
      static_cast<void>(this->mdspan());
    }

    /// \brief Bind a reshape to externally retained tensor storage.
    /// \warning The pointer may identify reserved but unconstructed storage;
    ///          callers must not resolve the view until the parent epoch is readable.
    explicit IndirectReshapedTensorView(tensor_type* tensor, RequestedExtents... requested_extents)
        : tensor_(tensor), requested_extents_(requested_extents...)
    {
      CHECK(tensor_ != nullptr);
    }

    /// \brief Return the source storage's backend selector.
    [[nodiscard]] constexpr decltype(auto) backend_selector() const { return this->base().backend_selector(); }

    /// \brief Resolve the reshaped mdspan descriptor with source mutability.
    [[nodiscard]] auto mdspan() & -> mdspan_type
    {
      return std::apply(
          [this](auto... requested_extents) {
            return detail::make_reshape_view<LayoutPolicy>(this->base().mdspan(), requested_extents...);
          },
          requested_extents_);
    }

    /// \brief Resolve the reshaped mdspan descriptor for read-only access.
    [[nodiscard]] auto mdspan() const& -> const_mdspan_type
    {
      return std::apply(
          [this](auto... requested_extents) {
            return detail::make_reshape_view<LayoutPolicy>(this->base().mdspan(), requested_extents...);
          },
          requested_extents_);
    }

    /// \brief Return the reshaped extents.
    [[nodiscard]] auto extents() const -> extents_type { return this->mdspan().extents(); }

    /// \brief Return one reshaped extent.
    [[nodiscard]] auto extent(std::size_t axis) const -> index_type { return this->mdspan().extent(axis); }

    /// \brief Evaluate or assign one reshaped element according to source mutability.
    template <class... Index>
      requires(sizeof...(Index) == extents_type::rank())
    decltype(auto) operator[](Index... indices)
    {
      return this->mdspan()[indices...];
    }

    /// \brief Evaluate one reshaped element through the read-only accessor.
    template <class... Index>
      requires(sizeof...(Index) == extents_type::rank())
    decltype(auto) operator[](Index... indices) const
    {
      return this->mdspan()[indices...];
    }

    /// \brief Return the tensor referenced by this descriptor.
    [[nodiscard]] tensor_type& base() noexcept { return *std::launder(tensor_); }

    /// \brief Return the tensor referenced by this descriptor.
    [[nodiscard]] tensor_type const& base() const noexcept { return *std::launder(tensor_); }

  private:
    tensor_type* tensor_;
    requested_extents_type requested_extents_;
};

namespace detail
{

template <CanonicalReshapeLayout LayoutPolicy, TensorView Tensor, std::integral... Extents>
  requires(std::is_lvalue_reference_v<Tensor &&> && StridedTensorView<Tensor>)
[[nodiscard]] auto make_tensor_reshape_view(Tensor&& tensor, Extents... requested_extents)
{
  auto span = make_reshape_view<LayoutPolicy>(tensor.mdspan(), requested_extents...);
  using span_type = decltype(span);
  using storage_policy = tensor_storage_policy_t<std::remove_cvref_t<Tensor>>;
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  return ReshapedTensor<span_type, storage_policy, selector_type>{std::move(span), tensor.backend_selector()};
}

} // namespace detail

/// \brief Return a tensor-level no-copy reshape preserving a canonical source layout.
/// \details Rvalue tensors are rejected because the returned descriptor does
///          not extend the lifetime of addressable source storage.
template <TensorView Tensor, std::integral... Extents>
  requires(std::is_lvalue_reference_v<Tensor &&> && StridedTensorView<Tensor> &&
           detail::CanonicalReshapeLayout<typename tensor_mdspan_t<Tensor>::layout_type>)
[[nodiscard]] auto reshape_view(Tensor&& tensor, Extents... requested_extents)
{
  using layout_type = typename tensor_mdspan_t<Tensor>::layout_type;
  return detail::make_tensor_reshape_view<layout_type>(std::forward<Tensor>(tensor), requested_extents...);
}

/// \brief Return a tensor-level no-copy column-major reshape of a strided source.
template <TensorView Tensor, std::integral... Extents>
  requires(std::is_lvalue_reference_v<Tensor &&> && StridedTensorView<Tensor>)
[[nodiscard]] auto reshape_view_left(Tensor&& tensor, Extents... requested_extents)
{
  return detail::make_tensor_reshape_view<stdex::layout_left>(std::forward<Tensor>(tensor), requested_extents...);
}

/// \brief Return a tensor-level no-copy row-major reshape of a strided source.
template <TensorView Tensor, std::integral... Extents>
  requires(std::is_lvalue_reference_v<Tensor &&> && StridedTensorView<Tensor>)
[[nodiscard]] auto reshape_view_right(Tensor&& tensor, Extents... requested_extents)
{
  return detail::make_tensor_reshape_view<stdex::layout_right>(std::forward<Tensor>(tensor), requested_extents...);
}

/// \brief Change a canonical owning tensor's shape without reallocating storage.
/// \details The compile-time rank is unchanged. Existing copied mdspan
///          descriptors retain their old mappings.
template <detail::CanonicallyLaidOutOwningTensor TensorType, std::integral... Extents>
  requires(sizeof...(Extents) == tensor_extents_t<TensorType>::rank() &&
           requires(TensorType& tensor, typename TensorType::mapping_type mapping) {
             tensor.replace_mapping(std::move(mapping));
           })
void reshape_inplace(TensorType& tensor, Extents... requested_extents)
{
  using tensor_extents = tensor_extents_t<TensorType>;
  using layout_type = typename tensor_mdspan_t<TensorType>::layout_type;
  auto const source_size = detail::checked_element_count(tensor.extents());
  auto requested = detail::make_reshape_extents(source_size, requested_extents...);
  auto new_extents = detail::convert_reshape_extents<tensor_extents>(requested);
  using mapping_type = typename layout_type::template mapping<tensor_extents>;
  tensor.replace_mapping(mapping_type{new_extents});
}

/// \brief Return an owning reshape of a canonical dense tensor.
/// \details Passing an lvalue copies its allocation; passing an rvalue transfers
///          it. The source's compile-time canonical layout is preserved.
template <class TensorType, std::integral... Extents>
  requires detail::StorageTransferReshapableTensor<TensorType, stdex::dextents<index_type, sizeof...(Extents)>>
[[nodiscard]] auto reshape(TensorType tensor, Extents... requested_extents)
{
  auto const source_size = detail::checked_element_count(tensor.extents());
  auto new_extents = detail::make_reshape_extents(source_size, requested_extents...);

  using source_type = std::remove_cvref_t<TensorType>;
  using result_type = typename source_type::template rebind_extents_type<decltype(new_extents)>;
  using layout_type = typename tensor_mdspan_t<TensorType>::layout_type;
  using mapping_type = typename layout_type::template mapping<decltype(new_extents)>;
  auto accessor_factory = std::move(tensor).release_accessor_factory();
  auto storage = std::move(tensor).release_storage();
  return result_type::adopt_storage(mapping_type{new_extents}, std::move(storage), std::move(accessor_factory));
}

} // namespace uni20
