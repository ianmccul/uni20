/**
 * \file concepts.hpp
 * \ingroup tensor
 * \brief Concepts for tensor-level objects that resolve mdspan kernel operands.
 */

#pragma once

#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{

namespace detail
{
template <class T> using tensor_type_t = std::remove_reference_t<T>;

template <class T> using tensor_const_mdspan_t = decltype(std::declval<tensor_type_t<T> const&>().mdspan());

template <class T> using tensor_mutable_mdspan_t = decltype(std::declval<tensor_type_t<T>&>().mdspan());
} // namespace detail

/// \brief Tensor-level object that exposes a readable mdspan and backend selector.
/// \details A tensor view combines mdspan access with storage/execution policy.
///          Non-owning views remain subject to their documented source lifetime;
///          leaf kernels receive the resolved result of `mdspan()`, not the
///          tensor object itself.
template <class T>
concept TensorView = requires(std::remove_reference_t<T> const& tensor) {
  typename std::remove_cvref_t<detail::tensor_const_mdspan_t<T>>::extents_type;
  tensor.backend_selector();
  tensor.mdspan();
  {
    tensor.extents()
  } -> std::convertible_to<typename std::remove_cvref_t<detail::tensor_const_mdspan_t<T>>::extents_type>;
  {
    tensor.extent(std::size_t{})
  } -> std::convertible_to<typename std::remove_cvref_t<detail::tensor_const_mdspan_t<T>>::index_type>;
} && SpanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan())>;

/// \brief Opt-in marker for tensor types that own their storage and lifetime.
/// \details Specialize this variable template for a tensor type only when
///          moving an object transfers ownership of the storage observed by
///          its resolved mdspan. Tensor view and proxy types must not opt in.
template <class T> inline constexpr bool enable_owning_tensor = false;

/// \brief Tensor-level object that owns the storage exposed by its mdspan.
/// \details Ownership is a property of the cvref-stripped object type. The
///          concept does not imply that a particular expression is mutable or
///          may be consumed; consuming operations impose those requirements
///          separately.
template <class T>
concept OwningTensor = TensorView<T> && enable_owning_tensor<std::remove_cvref_t<T>>;

/// \brief Resolved readable mdspan type exposed by a tensor-level object.
template <TensorView T> using tensor_mdspan_t = std::remove_cvref_t<detail::tensor_const_mdspan_t<T>>;

/// \brief Scalar value type exposed by a tensor-level object.
template <TensorView T> using tensor_element_t = std::remove_cv_t<typename tensor_mdspan_t<T>::element_type>;

/// \brief Extents type exposed by a tensor-level object.
template <TensorView T> using tensor_extents_t = typename tensor_mdspan_t<T>::extents_type;

/// \brief Tensor-level object that also exposes a writable resolved mdspan.
template <class T>
concept MutableTensorView =
    TensorView<T> && MutableSpanLike<decltype(std::declval<std::remove_reference_t<T>&>().mdspan())>;

/// \brief Resolved writable mdspan type exposed by a mutable tensor-level object.
template <MutableTensorView T> using mutable_tensor_mdspan_t = std::remove_cvref_t<detail::tensor_mutable_mdspan_t<T>>;

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
