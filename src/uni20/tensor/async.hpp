#pragma once

/**
 * \file async.hpp
 * \ingroup tensor
 * \brief Async aliases for lazy tensor-level views.
 */

#include <uni20/async/async.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/reshape.hpp>

namespace uni20::async
{

/// \brief Return a lazy conjugating alias of an async complex tensor.
/// \details The result retains the parent tensor storage and shares its exact
///          epoch queue. Awaiting the alias therefore observes the same causal
///          timeline as awaiting the parent.
template <uni20::DeviceTensorView Tensor>
  requires uni20::Complex<uni20::tensor_element_t<Tensor>>
[[nodiscard]] auto conj(Async<Tensor> const& tensor)
{
  using view_type = uni20::ConjugatedTensorView<Tensor>;
  return make_async_alias<view_type>(tensor, tensor.storage().storage_address());
}

/// \brief Return a lazy read-only identity alias of an async real tensor.
template <uni20::DeviceTensorView Tensor>
  requires(!uni20::Complex<uni20::tensor_element_t<Tensor>>)
[[nodiscard]] auto conj(Async<Tensor> const& tensor)
{
  using view_type = uni20::ConstTensorView<Tensor>;
  return make_async_alias<view_type>(tensor, tensor.storage().storage_address());
}

/// \brief Return a mutable structural reshape alias of an async tensor.
/// \details The descriptor retains the parent storage and shares its exact
///          epoch queue. Shape and layout validation occurs when the shared
///          parent epoch first becomes readable.
template <uni20::MutableStridedTensorView Tensor, std::integral... Extents>
  requires uni20::detail::CanonicalReshapeLayout<typename uni20::tensor_mdspan_t<Tensor>::layout_type>
[[nodiscard]] auto reshape_view(Async<Tensor>& tensor, Extents... requested_extents)
{
  using layout_type = typename uni20::tensor_mdspan_t<Tensor>::layout_type;
  using view_type = uni20::IndirectReshapedTensorView<Tensor, layout_type, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a read-only structural reshape alias of an async tensor.
template <uni20::StridedTensorView Tensor, std::integral... Extents>
  requires uni20::detail::CanonicalReshapeLayout<typename uni20::tensor_mdspan_t<Tensor>::layout_type>
[[nodiscard]] auto reshape_view(Async<Tensor> const& tensor, Extents... requested_extents)
{
  using layout_type = typename uni20::tensor_mdspan_t<Tensor>::layout_type;
  using view_type = uni20::IndirectReshapedTensorView<Tensor const, layout_type, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a mutable column-major structural reshape alias.
template <uni20::MutableStridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_left(Async<Tensor>& tensor, Extents... requested_extents)
{
  using view_type = uni20::IndirectReshapedTensorView<Tensor, stdex::layout_left, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a read-only column-major structural reshape alias.
template <uni20::StridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_left(Async<Tensor> const& tensor, Extents... requested_extents)
{
  using view_type =
      uni20::IndirectReshapedTensorView<Tensor const, stdex::layout_left, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a mutable row-major structural reshape alias.
template <uni20::MutableStridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_right(Async<Tensor>& tensor, Extents... requested_extents)
{
  using view_type = uni20::IndirectReshapedTensorView<Tensor, stdex::layout_right, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

/// \brief Return a read-only row-major structural reshape alias.
template <uni20::StridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto reshape_view_right(Async<Tensor> const& tensor, Extents... requested_extents)
{
  using view_type =
      uni20::IndirectReshapedTensorView<Tensor const, stdex::layout_right, std::remove_cvref_t<Extents>...>;
  return make_async_alias_from_parent<view_type>(tensor, requested_extents...);
}

} // namespace uni20::async
