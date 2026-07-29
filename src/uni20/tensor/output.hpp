#pragma once

/**
 * \file output.hpp
 * \ingroup tensor
 * \brief Shape preparation contracts for tensor operation outputs.
 */

#include <uni20/common/trace.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{
template <class Extents>
concept TensorExtentsLike = requires(Extents const& extents) {
  { std::remove_cvref_t<Extents>::rank() } -> std::convertible_to<std::size_t>;
  extents.extent(std::size_t{});
};

template <TensorExtentsLike LhsExtents, TensorExtentsLike RhsExtents>
[[nodiscard]] constexpr bool tensor_extents_equal(LhsExtents const& lhs, RhsExtents const& rhs) noexcept
{
  if constexpr (std::remove_cvref_t<LhsExtents>::rank() != std::remove_cvref_t<RhsExtents>::rank())
  {
    return false;
  }
  else
  {
    for (std::size_t axis = 0; axis < std::remove_cvref_t<LhsExtents>::rank(); ++axis)
    {
      if (lhs.extent(axis) != rhs.extent(axis)) return false;
    }
    return true;
  }
}

template <class TargetExtents, TensorExtentsLike SourceExtents, std::size_t... Axis>
[[nodiscard]] TargetExtents convert_tensor_extents(SourceExtents const& source, std::index_sequence<Axis...>)
{
  for (std::size_t axis = 0; axis < TargetExtents::rank(); ++axis)
  {
    auto const fixed_extent = TargetExtents::static_extent(axis);
    ERROR_IF(fixed_extent != stdex::dynamic_extent && !std::cmp_equal(source.extent(axis), fixed_extent),
             "tensor extent does not match the destination's static extent");
  }
  using index_type = typename TargetExtents::index_type;
  return TargetExtents(static_cast<index_type>(source.extent(Axis))...);
}

template <class TargetExtents, TensorExtentsLike SourceExtents>
[[nodiscard]] TargetExtents convert_tensor_extents(SourceExtents const& source)
{
  static_assert(TargetExtents::rank() == std::remove_cvref_t<SourceExtents>::rank(),
                "tensor output rank does not match the required shape");
  return convert_tensor_extents<TargetExtents>(source, std::make_index_sequence<TargetExtents::rank()>{});
}
} // namespace detail

/// \brief Mutable tensor output that can replace its shape and storage.
/// \details A resizable output provides `reset_shape(extents)`, whose contract
///          discards old values and leaves the object with the requested shape.
template <class T>
concept ResizableTensorOutput =
    MutableDeviceTensorView<T> &&
    requires(std::remove_reference_t<T>& output, tensor_extents_t<T> const& extents) { output.reset_shape(extents); };

/// \brief Deferred storage that may contain a mutable Tensor output.
template <class T>
concept TensorOutputStorage = requires(std::remove_reference_t<T>& storage) {
  typename std::remove_cvref_t<T>::element_type;
  { storage.constructed() } -> std::convertible_to<bool>;
  { storage.get() } -> std::same_as<typename std::remove_cvref_t<T>::element_type*>;
} && MutableDeviceTensorView<typename std::remove_cvref_t<T>::element_type>;

/// \brief Tensor value type contained by deferred output storage.
template <TensorOutputStorage Storage>
using tensor_output_storage_t = typename std::remove_cvref_t<Storage>::element_type;

/// \brief Require an existing tensor output to have the specified shape.
/// \details This operation validates only and never modifies the output,
///          including when the output owns replaceable storage.
template <DeviceTensorView Output, detail::TensorExtentsLike RequiredExtents>
void require_output(Output const& output, RequiredExtents const& required)
{
  static_assert(tensor_extents_t<Output>::rank() == std::remove_cvref_t<RequiredExtents>::rank(),
                "tensor output rank does not match the required shape");
  ERROR_IF(!detail::tensor_extents_equal(output.extents(), required),
           "tensor output shape does not match the required extents");
}

/// \brief Prepare a mutable tensor output with the specified shape.
/// \details Resizable outputs retain their current storage and values when the
///          shape already matches, and call `reset_shape` otherwise. Fixed
///          outputs validate through `require_output` and never rebind storage.
/// \warning This operation may modify the output. A dispatch backend must call
///          it only after every check that could produce a clean decline has
///          succeeded and the backend is committed to the operation.
/// \return Reference to the prepared output.
template <MutableDeviceTensorView Output, detail::TensorExtentsLike RequiredExtents>
Output& prepare_output(Output& output, RequiredExtents const& required)
{
  static_assert(tensor_extents_t<Output>::rank() == std::remove_cvref_t<RequiredExtents>::rank(),
                "tensor output rank does not match the required shape");
  if (detail::tensor_extents_equal(output.extents(), required)) return output;

  if constexpr (ResizableTensorOutput<Output>)
  {
    auto const converted = detail::convert_tensor_extents<tensor_extents_t<Output>>(required);
    output.reset_shape(converted);
    CHECK(detail::tensor_extents_equal(output.extents(), required));
  }
  else
  {
    require_output(output, required);
  }
  return output;
}

/// \brief Prepare a mutable Tensor output with compatible shape and storage.
/// \details A matching output is retained. Otherwise a replaceable output is
///          reconstructed with the required extents and placement.
/// \warning This operation may modify the output. A dispatch backend must call
///          it only after every check that could produce a clean decline has
///          succeeded and the backend is committed to the operation.
/// \tparam Placement Storage-policy placement requirement.
/// \return Reference to the prepared output.
template <MutableDeviceTensorView Output, detail::TensorExtentsLike RequiredExtents, class Placement>
  requires requires(Output& output, tensor_extents_t<Output> const& extents, Placement const& placement) {
    { output.storage_is_compatible(placement) } -> std::convertible_to<bool>;
    output.reset_shape(extents);
    output.replace(extents, placement);
  }
Output& prepare_output(Output& output, RequiredExtents const& required, Placement const& placement)
{
  static_assert(tensor_extents_t<Output>::rank() == std::remove_cvref_t<RequiredExtents>::rank(),
                "tensor output rank does not match the required shape");
  if (detail::tensor_extents_equal(output.extents(), required) && output.storage_is_compatible(placement))
    return output;

  auto const converted = detail::convert_tensor_extents<tensor_extents_t<Output>>(required);
  if (output.storage_is_compatible(placement))
    output.reset_shape(converted);
  else
    output.replace(converted, placement);

  CHECK(detail::tensor_extents_equal(output.extents(), required));
  CHECK(output.storage_is_compatible(placement));
  return output;
}

/// \brief Construct or resize a deferred Tensor output to the required shape.
/// \warning This operation may construct or modify the output. A dispatch
///          backend must call it only after every check that could produce a
///          clean decline has succeeded and the backend is committed.
/// \return Reference to the prepared output value.
template <TensorOutputStorage Storage, detail::TensorExtentsLike RequiredExtents>
  requires std::constructible_from<typename std::remove_cvref_t<Storage>::element_type,
                                   tensor_extents_t<typename std::remove_cvref_t<Storage>::element_type> const&>
auto prepare_output(Storage& storage, RequiredExtents const& required) ->
    typename std::remove_cvref_t<Storage>::element_type&
{
  using output_type = typename std::remove_cvref_t<Storage>::element_type;
  if (storage.constructed()) return prepare_output(*storage, required);

  auto const converted = detail::convert_tensor_extents<tensor_extents_t<output_type>>(required);
  return storage.emplace(converted);
}

/// \brief Construct or replace a deferred Tensor output with compatible shape and storage.
/// \warning This operation may construct or modify the output. A dispatch
///          backend must call it only after every check that could produce a
///          clean decline has succeeded and the backend is committed.
/// \tparam Placement Storage-policy placement requirement.
/// \return Reference to the prepared output value.
template <TensorOutputStorage Storage, detail::TensorExtentsLike RequiredExtents, class Placement>
  requires std::constructible_from<typename std::remove_cvref_t<Storage>::element_type, Placement const&,
                                   tensor_extents_t<typename std::remove_cvref_t<Storage>::element_type> const&> &&
               requires(typename std::remove_cvref_t<Storage>::element_type& output,
                        tensor_extents_t<typename std::remove_cvref_t<Storage>::element_type> const& extents,
                        Placement const& placement) { prepare_output(output, extents, placement); }
auto prepare_output(Storage& storage, RequiredExtents const& required, Placement const& placement) ->
    typename std::remove_cvref_t<Storage>::element_type&
{
  using output_type = typename std::remove_cvref_t<Storage>::element_type;
  if (storage.constructed()) return prepare_output(*storage, required, placement);

  auto const converted = detail::convert_tensor_extents<tensor_extents_t<output_type>>(required);
  return storage.emplace(placement, converted);
}

} // namespace uni20
