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

template <class Tensor, class = void> struct TensorStoragePolicy
{
    using type = void;
};

template <class Tensor> struct TensorStoragePolicy<Tensor, std::void_t<typename Tensor::storage_policy>>
{
    using type = typename Tensor::storage_policy;
};

template <class T> using tensor_const_mdspan_t = decltype(std::declval<tensor_type_t<T> const&>().mdspan());

template <class T> using tensor_mutable_mdspan_t = decltype(std::declval<tensor_type_t<T>&>().mdspan());
} // namespace detail

/// \brief Storage-policy type exposed by a tensor-level object, or `void`.
template <class Tensor>
using tensor_storage_policy_t = typename detail::TensorStoragePolicy<std::remove_cvref_t<Tensor>>::type;

/// \brief Return a tensor's immediate or descriptor-backed multidimensional metadata.
/// \details This operation prefers `mdspec()` when the tensor exposes it
///          and otherwise returns `mdspan()`. It does not acquire, copy, or
///          transform tensor data.
/// \param tensor Tensor-level object whose multidimensional metadata is requested.
/// \return The result of `tensor.mdspec()`.
template <class Tensor>
  requires requires(Tensor& tensor) { tensor.mdspec(); }
[[nodiscard]] constexpr decltype(auto) mdspec_of(Tensor& tensor) noexcept(noexcept(tensor.mdspec()))
{
  return tensor.mdspec();
}

/// \brief Return an immediate tensor's mdspan as its mdspec representation.
/// \details This fallback applies when the tensor does not expose
///          `mdspec()`. An ordinary `MdspanLike` is already the immediate
///          case of `MdspecLike`, so no wrapper is introduced.
/// \param tensor Tensor-level object whose multidimensional metadata is requested.
/// \return The result of `tensor.mdspan()`.
template <class Tensor>
  requires(
      !requires(Tensor& tensor) { tensor.mdspec(); } && requires(Tensor& tensor) { tensor.mdspan(); })
[[nodiscard]] constexpr decltype(auto) mdspec_of(Tensor& tensor) noexcept(noexcept(tensor.mdspan()))
{
  return tensor.mdspan();
}

namespace detail
{
template <class T> using normalized_const_mdspec_t = decltype(mdspec_of(std::declval<tensor_type_t<T> const&>()));

template <class T> using normalized_mutable_mdspec_t = decltype(mdspec_of(std::declval<tensor_type_t<T>&>()));
} // namespace detail

/// \brief Tensor-level object exposing readable multidimensional metadata and a backend selector.
/// \details A model exposes an immediate or descriptor-backed `MdspecLike`
///          through `mdspec_of()`. Explicit access acquisition resolves that
///          metadata into an `ImmediateTensorView` governed by an RAII lease.
///          The policy-bearing tensor object does not itself model `MdspecLike`.
template <class T>
concept TensorView = (!MdspecLike<T>) && requires(std::remove_reference_t<T> const& tensor) {
  typename std::remove_cvref_t<detail::normalized_const_mdspec_t<T>>::extents_type;
  tensor.backend_selector();
  mdspec_of(tensor);
  {
    tensor.extents()
  } -> std::convertible_to<typename std::remove_cvref_t<detail::normalized_const_mdspec_t<T>>::extents_type>;
  {
    tensor.extent(std::size_t{})
  } -> std::convertible_to<typename std::remove_cvref_t<detail::normalized_const_mdspec_t<T>>::index_type>;
} && MdspecLike<detail::normalized_const_mdspec_t<T>>;

/// \brief TensorView whose mdspec is already an mdspan.
/// \details An immediate tensor view exposes `mdspan()` and requires no
///          descriptor-to-handle resolution. It remains distinct from
///          `MdspanLike`: the tensor object exposes an mdspan rather than
///          modeling the mdspan contract directly.
template <class T>
concept ImmediateTensorView = TensorView<T> && requires(std::remove_reference_t<T> const& tensor) {
  tensor.mdspan();
} && MdspanLike<detail::tensor_const_mdspan_t<T>> && MdspanLike<detail::normalized_const_mdspec_t<T>>;

/// \brief Opt-in marker for tensor types that own their storage and lifetime.
/// \details Specialize this variable template for a tensor type only when
///          moving an object transfers ownership of the storage observed by
///          its resolved mdspan. Tensor view and proxy types must not opt in.
template <class T> inline constexpr bool enable_owning_tensor = false;

/// \brief Tensor-level object that owns the storage described by its multidimensional view.
/// \details Ownership is a property of the cvref-stripped object type. The
///          concept does not imply that a particular expression is mutable or
///          may be consumed; consuming operations impose those requirements
///          separately.
template <class T>
concept OwningTensor = TensorView<T> && enable_owning_tensor<std::remove_cvref_t<T>>;

/// \brief Resolved readable mdspan type exposed by a tensor-level object.
template <ImmediateTensorView T>
using immediate_tensor_mdspan_t = std::remove_cvref_t<detail::tensor_const_mdspan_t<T>>;

/// \brief Normalized readable MdspecLike type exposed by a tensor-level object.
template <TensorView T> using tensor_mdspec_t = std::remove_cvref_t<detail::normalized_const_mdspec_t<T>>;

/// \brief Scalar value type exposed by a tensor-level object.
template <TensorView T> using tensor_element_t = std::remove_cv_t<typename tensor_mdspec_t<T>::element_type>;

/// \brief Extents type exposed by a tensor-level object.
template <TensorView T> using tensor_extents_t = typename tensor_mdspec_t<T>::extents_type;

/// \brief TensorView whose writable mdspec supports eventual element assignment.
template <class T>
concept MutableTensorView = TensorView<T> && requires(std::remove_reference_t<T>& tensor) { mdspec_of(tensor); } &&
                            MutableMdspecLike<detail::normalized_mutable_mdspec_t<T>>;

/// \brief MutableTensorView whose writable mdspec is already an mdspan.
template <class T>
concept MutableImmediateTensorView =
    MutableTensorView<T> && ImmediateTensorView<T> &&
    requires(std::remove_reference_t<T>& tensor) { tensor.mdspan(); } &&
    MutableMdspanLike<detail::tensor_mutable_mdspan_t<T>> && MutableMdspanLike<detail::normalized_mutable_mdspec_t<T>>;

/// \brief Normalized writable MdspecLike type exposed by a mutable tensor-level object.
template <MutableTensorView T>
using mutable_tensor_mdspec_t = std::remove_cvref_t<detail::normalized_mutable_mdspec_t<T>>;

/// \brief TensorView whose readable mdspec has a specified static rank.
template <class T, std::size_t Rank>
concept RankedTensorView = TensorView<T> && RankedMdspecLike<tensor_mdspec_t<T>, Rank>;

/// \brief TensorView whose readable mdspec is strided.
template <class T>
concept StridedTensorView = TensorView<T> && StridedMdspecLike<tensor_mdspec_t<T>>;

/// \brief MutableTensorView whose readable and writable mdspecs are strided.
template <class T>
concept MutableStridedTensorView =
    MutableTensorView<T> && StridedTensorView<T> && MutableStridedMdspecLike<mutable_tensor_mdspec_t<T>>;

/// \brief StridedTensorView with a specified static rank.
template <class T, std::size_t Rank>
concept RankedStridedTensorView = RankedTensorView<T, Rank> && StridedTensorView<T>;

/// \brief MutableTensorView with a specified static rank.
template <class T, std::size_t Rank>
concept MutableRankedTensorView =
    RankedTensorView<T, Rank> && MutableTensorView<T> && MutableRankedMdspecLike<mutable_tensor_mdspec_t<T>, Rank>;

/// \brief TensorView whose readable mdspec has rank zero.
template <class T>
concept ScalarTensorView = RankedTensorView<T, 0>;

/// \brief MutableTensorView whose writable mdspec has rank zero.
template <class T>
concept MutableScalarTensorView = MutableRankedTensorView<T, 0>;

/// \brief Mutable strided TensorView with a specified static rank.
template <class T, std::size_t Rank>
concept MutableRankedStridedTensorView = MutableRankedTensorView<T, Rank> && MutableStridedTensorView<T>;

/// \brief Resolved writable mdspan type exposed by a mutable tensor-level object.
template <MutableImmediateTensorView T>
using mutable_immediate_tensor_mdspan_t = std::remove_cvref_t<detail::tensor_mutable_mdspan_t<T>>;

/// \brief ImmediateTensorView whose readable mdspan is strided.
template <class T>
concept StridedImmediateTensorView =
    ImmediateTensorView<T> && StridedTensorView<T> &&
    StridedMdspanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan())>;

/// \brief MutableImmediateTensorView whose readable and writable mdspans are strided.
template <class T>
concept MutableStridedImmediateTensorView =
    MutableImmediateTensorView<T> && MutableStridedTensorView<T> && StridedImmediateTensorView<T> &&
    MutableStridedMdspanLike<decltype(std::declval<std::remove_reference_t<T>&>().mdspan())>;

/// \brief ImmediateTensorView whose readable mdspan has a specified static rank.
/// \tparam T Tensor-like type under test.
/// \tparam Rank Required rank of the resolved mdspan.
template <class T, std::size_t Rank>
concept RankedImmediateTensorView =
    ImmediateTensorView<T> && RankedTensorView<T, Rank> &&
    RankedMdspanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan()), Rank>;

/// \brief MutableImmediateTensorView whose writable mdspan has a specified static rank.
/// \tparam T Tensor-like type under test.
/// \tparam Rank Required rank of the resolved mdspan.
template <class T, std::size_t Rank>
concept MutableRankedImmediateTensorView =
    RankedImmediateTensorView<T, Rank> && MutableRankedTensorView<T, Rank> && MutableImmediateTensorView<T> &&
    MutableRankedMdspanLike<detail::tensor_mutable_mdspan_t<T>, Rank>;

/// \brief Tensor-level object whose readable resolved mdspan has rank zero.
template <class T>
concept ScalarImmediateTensorView = RankedImmediateTensorView<T, 0>;

/// \brief Mutable tensor-level object whose writable resolved mdspan has rank zero.
template <class T>
concept MutableScalarImmediateTensorView = MutableRankedImmediateTensorView<T, 0>;

/// \brief Strided tensor-level object whose resolved mdspan has a specified static rank.
template <class T, std::size_t Rank>
concept RankedStridedImmediateTensorView =
    RankedImmediateTensorView<T, Rank> && RankedStridedTensorView<T, Rank> && StridedImmediateTensorView<T>;

/// \brief Mutable strided tensor-level object whose resolved mdspan has a specified static rank.
template <class T, std::size_t Rank>
concept MutableRankedStridedImmediateTensorView =
    MutableRankedImmediateTensorView<T, Rank> && MutableRankedStridedTensorView<T, Rank> &&
    MutableStridedImmediateTensorView<T>;

/// \brief Retrieve the normalized multidimensional strides of a tensor-level object.
/// \tparam T The strided tensor view type.
/// \param tensor Tensor metadata whose strides will be returned.
/// \return A std::array containing the strides for each rank.
template <StridedTensorView T> [[nodiscard]] auto strides(T const& tensor) { return strides(mdspec_of(tensor)); }

/// \brief Assign tensor values through a mutable tensor alias descriptor.
/// \details This declaration is the tensor customization used by capability-aware
///          async write proxies. Its definition delegates to `uni20::copy`.
template <MutableTensorView Output, TensorView Input>
  requires(tensor_mdspec_t<Output>::rank() == tensor_mdspec_t<Input>::rank())
void assign_through(Output& output, Input const& input);

} // namespace uni20
