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
[[nodiscard]] constexpr TargetExtents convert_tensor_extents(SourceExtents const& source, std::index_sequence<Axis...>)
{
  using index_type = typename TargetExtents::index_type;
  return TargetExtents(static_cast<index_type>(source.extent(Axis))...);
}

template <class TargetExtents, TensorExtentsLike SourceExtents>
[[nodiscard]] constexpr TargetExtents convert_tensor_extents(SourceExtents const& source)
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
    MutableTensorView<T> &&
    requires(std::remove_reference_t<T>& output, tensor_extents_t<T> const& extents) { output.reset_shape(extents); };

/// \brief Validate that a tensor already has the required shape.
/// \details This operation never resizes, including for owning tensors. It is
///          the shape contract for update operations that read the old output.
template <TensorView Output, detail::TensorExtentsLike RequiredExtents>
void require_shape(Output const& output, RequiredExtents const& required)
{
  static_assert(tensor_extents_t<Output>::rank() == std::remove_cvref_t<RequiredExtents>::rank(),
                "tensor output rank does not match the required shape");
  ERROR_IF(!detail::tensor_extents_equal(output.extents(), required),
           "tensor output shape does not match the required extents");
}

/// \brief Ensure that a mutable tensor output has the required shape.
/// \details Resizable outputs retain their current storage and values when the
///          shape already matches, and call `reset_shape` otherwise. Fixed
///          outputs validate through `require_shape` and never rebind storage.
template <MutableTensorView Output, detail::TensorExtentsLike RequiredExtents>
void ensure_shape(Output&& output, RequiredExtents const& required)
{
  static_assert(tensor_extents_t<Output>::rank() == std::remove_cvref_t<RequiredExtents>::rank(),
                "tensor output rank does not match the required shape");
  if (detail::tensor_extents_equal(output.extents(), required)) return;

  if constexpr (ResizableTensorOutput<Output>)
  {
    auto const converted = detail::convert_tensor_extents<tensor_extents_t<Output>>(required);
    output.reset_shape(converted);
    CHECK(detail::tensor_extents_equal(output.extents(), required));
  }
  else
  {
    require_shape(output, required);
  }
}

} // namespace uni20
