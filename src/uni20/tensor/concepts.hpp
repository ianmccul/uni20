/**
 * \file concepts.hpp
 * \ingroup tensor
 * \brief Concepts for tensor-level objects that resolve mdspan kernel operands.
 */

#pragma once

#include <uni20/mdspan/concepts.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Tensor-level object that exposes a readable mdspan and backend selector.
/// \details A tensor owns storage or provides another durable storage policy.
///          Leaf kernels receive the resolved result of `mdspan()`, not the
///          tensor object itself.
template <class T>
concept TensorView = requires(std::remove_reference_t<T> const& tensor) {
  tensor.backend_selector();
  tensor.mdspan();
} && SpanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan())>;

/// \brief Tensor-level object that also exposes a writable resolved mdspan.
template <class T>
concept MutableTensorView =
    TensorView<T> && MutableSpanLike<decltype(std::declval<std::remove_reference_t<T>&>().mdspan())>;

/// \brief Tensor-level object whose readable resolved mdspan is strided.
template <class T>
concept StridedTensorView =
    TensorView<T> && StridedMdspan<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan())>;

/// \brief Mutable tensor-level object whose resolved mdspan is strided.
template <class T>
concept MutableStridedTensorView = MutableTensorView<T> && StridedTensorView<T> &&
                                   MutableStridedMdspan<decltype(std::declval<std::remove_reference_t<T>&>().mdspan())>;

/// \brief Tensor-level object whose readable resolved mdspan has a specified static rank.
/// \tparam T Tensor-like type under test.
/// \tparam Rank Required rank of the resolved mdspan.
template <class T, std::size_t Rank>
concept RankedTensorView =
    TensorView<T> && RankedSpanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan()), Rank>;

/// \brief Mutable tensor-level object whose writable resolved mdspan has a specified static rank.
/// \tparam T Tensor-like type under test.
/// \tparam Rank Required rank of the resolved mdspan.
template <class T, std::size_t Rank>
concept MutableRankedTensorView = RankedTensorView<T, Rank> && MutableTensorView<T>;

/// \brief Strided tensor-level object whose resolved mdspan has a specified static rank.
template <class T, std::size_t Rank>
concept RankedStridedTensorView = RankedTensorView<T, Rank> && StridedTensorView<T>;

/// \brief Mutable strided tensor-level object whose resolved mdspan has a specified static rank.
template <class T, std::size_t Rank>
concept MutableRankedStridedTensorView = MutableRankedTensorView<T, Rank> && MutableStridedTensorView<T>;

} // namespace uni20
