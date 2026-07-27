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

template <class Tensor> using tensor_storage_policy_t = typename TensorStoragePolicy<std::remove_cvref_t<Tensor>>::type;

template <class T> using tensor_const_mdspan_t = decltype(std::declval<tensor_type_t<T> const&>().mdspan());

template <class T> using tensor_mutable_mdspan_t = decltype(std::declval<tensor_type_t<T>&>().mdspan());

template <class Tensor>
  requires requires(Tensor& tensor) { tensor.device_mdspan(); }
[[nodiscard]] decltype(auto) tensor_device_mdspan(Tensor& tensor)
{
  return tensor.device_mdspan();
}

template <class Tensor>
  requires(
      !requires(Tensor& tensor) { tensor.device_mdspan(); } && requires(Tensor& tensor) { tensor.mdspan(); })
[[nodiscard]] decltype(auto) tensor_device_mdspan(Tensor& tensor)
{
  return tensor.mdspan();
}

template <class T>
using normalized_const_device_mdspan_t = decltype(tensor_device_mdspan(std::declval<tensor_type_t<T> const&>()));

template <class T>
using normalized_mutable_device_mdspan_t = decltype(tensor_device_mdspan(std::declval<tensor_type_t<T>&>()));
} // namespace detail

/// \brief Tensor-level object that exposes a readable mdspan and backend selector.
/// \details A tensor view combines mdspan access with storage/execution policy.
///          Non-owning views remain subject to their documented source lifetime.
///          Backend adapters may accept the tensor object and lower `mdspan()`
///          at their provider boundary.
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
} && MdspanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan())>;

/// \brief Tensor-level object whose multidimensional data may require acquisition.
/// \details An ordinary `TensorView` is the immediate case. A deferred model
///          exposes `device_mdspan()` returning a `DeviceMdspanLike`; explicit
///          read or write access resolves that metadata into a `TensorView`
///          whose lifetime is governed by an RAII lease.
template <class T>
concept DeviceTensorView = requires(std::remove_reference_t<T> const& tensor) {
  typename std::remove_cvref_t<detail::normalized_const_device_mdspan_t<T>>::extents_type;
  tensor.backend_selector();
  detail::tensor_device_mdspan(tensor);
  {
    tensor.extents()
  } -> std::convertible_to<typename std::remove_cvref_t<detail::normalized_const_device_mdspan_t<T>>::extents_type>;
  {
    tensor.extent(std::size_t{})
  } -> std::convertible_to<typename std::remove_cvref_t<detail::normalized_const_device_mdspan_t<T>>::index_type>;
} && DeviceMdspanLike<detail::normalized_const_device_mdspan_t<T>>;

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
concept OwningTensor = DeviceTensorView<T> && enable_owning_tensor<std::remove_cvref_t<T>>;

/// \brief Resolved readable mdspan type exposed by a tensor-level object.
template <TensorView T> using tensor_mdspan_t = std::remove_cvref_t<detail::tensor_const_mdspan_t<T>>;

/// \brief Normalized readable DeviceMdspanLike type exposed by a tensor-level object.
template <DeviceTensorView T>
using device_tensor_mdspan_t = std::remove_cvref_t<detail::normalized_const_device_mdspan_t<T>>;

/// \brief Scalar value type exposed by a tensor-level object.
template <DeviceTensorView T>
using tensor_element_t = std::remove_cv_t<typename device_tensor_mdspan_t<T>::element_type>;

/// \brief Extents type exposed by a tensor-level object.
template <DeviceTensorView T> using tensor_extents_t = typename device_tensor_mdspan_t<T>::extents_type;

/// \brief Tensor-level object that also exposes a writable resolved mdspan.
template <class T>
concept MutableTensorView =
    TensorView<T> && MutableMdspanLike<decltype(std::declval<std::remove_reference_t<T>&>().mdspan())>;

/// \brief DeviceTensorView whose normalized non-const device mdspan supports writes.
template <class T>
concept MutableDeviceTensorView = DeviceTensorView<T> && requires(std::remove_reference_t<T>& tensor) {
  detail::tensor_device_mdspan(tensor);
} && MutableDeviceMdspanLike<detail::normalized_mutable_device_mdspan_t<T>>;

/// \brief DeviceTensorView whose normalized readable device mdspan has a specified static rank.
template <class T, std::size_t Rank>
concept RankedDeviceTensorView = DeviceTensorView<T> && RankedDeviceMdspanLike<device_tensor_mdspan_t<T>, Rank>;

/// \brief DeviceTensorView whose normalized readable multidimensional metadata is strided.
template <class T>
concept StridedDeviceTensorView = DeviceTensorView<T> && StridedDeviceMdspanLike<device_tensor_mdspan_t<T>>;

/// \brief Mutable device tensor view whose normalized readable and writable device mdspans are strided.
template <class T>
concept MutableStridedDeviceTensorView =
    MutableDeviceTensorView<T> && StridedDeviceTensorView<T> &&
    MutableStridedDeviceMdspanLike<std::remove_cvref_t<detail::normalized_mutable_device_mdspan_t<T>>>;

/// \brief Strided device tensor view with a specified static rank.
template <class T, std::size_t Rank>
concept RankedStridedDeviceTensorView = RankedDeviceTensorView<T, Rank> && StridedDeviceTensorView<T>;

/// \brief Mutable device tensor view with a specified static rank.
template <class T, std::size_t Rank>
concept MutableRankedDeviceTensorView =
    RankedDeviceTensorView<T, Rank> && MutableDeviceTensorView<T> &&
    MutableRankedDeviceMdspanLike<detail::normalized_mutable_device_mdspan_t<T>, Rank>;

/// \brief DeviceTensorView whose readable device mdspan has rank zero.
template <class T>
concept ScalarDeviceTensorView = RankedDeviceTensorView<T, 0>;

/// \brief Mutable DeviceTensorView whose writable device mdspan has rank zero.
template <class T>
concept MutableScalarDeviceTensorView = MutableRankedDeviceTensorView<T, 0>;

/// \brief Mutable strided device tensor view with a specified static rank.
template <class T, std::size_t Rank>
concept MutableRankedStridedDeviceTensorView =
    MutableRankedDeviceTensorView<T, Rank> && MutableStridedDeviceTensorView<T>;

/// \brief Resolved writable mdspan type exposed by a mutable tensor-level object.
template <MutableTensorView T> using mutable_tensor_mdspan_t = std::remove_cvref_t<detail::tensor_mutable_mdspan_t<T>>;

/// \brief Tensor-level object whose readable resolved mdspan is strided.
template <class T>
concept StridedTensorView =
    TensorView<T> && StridedMdspanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan())>;

/// \brief Mutable tensor-level object whose resolved mdspan is strided.
template <class T>
concept MutableStridedTensorView =
    MutableTensorView<T> && StridedTensorView<T> &&
    MutableStridedMdspanLike<decltype(std::declval<std::remove_reference_t<T>&>().mdspan())>;

/// \brief Tensor-level object whose readable resolved mdspan has a specified static rank.
/// \tparam T Tensor-like type under test.
/// \tparam Rank Required rank of the resolved mdspan.
template <class T, std::size_t Rank>
concept RankedTensorView =
    TensorView<T> && RankedMdspanLike<decltype(std::declval<std::remove_reference_t<T> const&>().mdspan()), Rank>;

/// \brief Mutable tensor-level object whose writable resolved mdspan has a specified static rank.
/// \tparam T Tensor-like type under test.
/// \tparam Rank Required rank of the resolved mdspan.
template <class T, std::size_t Rank>
concept MutableRankedTensorView = RankedTensorView<T, Rank> && MutableTensorView<T> &&
                                  MutableRankedMdspanLike<detail::tensor_mutable_mdspan_t<T>, Rank>;

/// \brief Tensor-level object whose readable resolved mdspan has rank zero.
template <class T>
concept ScalarTensorView = RankedTensorView<T, 0>;

/// \brief Mutable tensor-level object whose writable resolved mdspan has rank zero.
template <class T>
concept MutableScalarTensorView = MutableRankedTensorView<T, 0>;

/// \brief Strided tensor-level object whose resolved mdspan has a specified static rank.
template <class T, std::size_t Rank>
concept RankedStridedTensorView = RankedTensorView<T, Rank> && StridedTensorView<T>;

/// \brief Mutable strided tensor-level object whose resolved mdspan has a specified static rank.
template <class T, std::size_t Rank>
concept MutableRankedStridedTensorView = MutableRankedTensorView<T, Rank> && MutableStridedTensorView<T>;

/// \brief Assign tensor values through a mutable tensor alias descriptor.
/// \details This declaration is the tensor customization used by capability-aware
///          async write proxies. Its definition delegates to `uni20::copy`.
template <MutableDeviceTensorView Output, DeviceTensorView Input>
  requires(device_tensor_mdspan_t<Output>::rank() == device_tensor_mdspan_t<Input>::rank())
void assign_through(Output& output, Input const& input);

} // namespace uni20
