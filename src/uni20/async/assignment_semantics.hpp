/// \file assignment_semantics.hpp
/// \brief Assignment-semantic traits for async write buffers and proxies.

#pragma once

#include <type_traits>

namespace uni20::async
{

/// \brief Selects how write-proxy assignment updates a stored value.
enum class assignment_semantics
{
    rebind,
    write_through,
};

/// \brief Trait mapping a type to its async write-assignment behavior.
/// \tparam T Stored value type.
///
/// The default behavior is `assignment_semantics::rebind`, where write-proxy
/// assignment reconstructs/rebinds the stored object.
template <typename T>
struct assignment_semantics_of : std::integral_constant<assignment_semantics, assignment_semantics::rebind>
{};

/// \brief Convenience variable for accessing `assignment_semantics_of`.
/// \tparam T Stored value type.
template <typename T>
inline constexpr assignment_semantics assignment_semantics_v = assignment_semantics_of<std::remove_cvref_t<T>>::value;

/// \brief `true` when async write assignment should mutate through an existing object.
/// \tparam T Stored value type.
template <typename T>
inline constexpr bool write_through_assignment_v = assignment_semantics_v<T> == assignment_semantics::write_through;

} // namespace uni20::async
