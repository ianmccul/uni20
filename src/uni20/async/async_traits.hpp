/**
 * \file async_traits.hpp
 * \ingroup async_api
 * \brief Payload classification and write-through customization for asynchronous values.
 */

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace uni20::async
{

template <typename T> class Async;

/// \brief Strip `Async<T>` to `T` for assignment and expression type deduction.
template <typename T> struct async_value_type
{
    using type = T;
};

template <typename T> struct async_value_type<Async<T>>
{
    using type = T;
};

/// \brief Extract the underlying value type from an immediate value or `Async<T>`.
template <typename T> using async_value_t = typename async_value_type<std::remove_cvref_t<T>>::type;

namespace detail
{

template <typename T, typename = void> struct default_is_async_alias : std::false_type
{};

template <typename T>
struct default_is_async_alias<T, std::void_t<typename std::remove_cvref_t<T>::async_alias_tag>> : std::true_type
{};

} // namespace detail

/// \brief Identifies payload descriptors requiring shared async alias identity.
/// \details Ordinary payloads are independent values. A durable alias
///          descriptor may declare `async_alias_tag`, or specialize this trait,
///          so `Async<T>` copies retain the descriptor storage, lifetime owner,
///          and exact epoch queue.
template <typename T> struct is_async_alias : detail::default_is_async_alias<std::remove_cvref_t<T>>
{};

/// \brief True when `T` is an owner-bound async alias descriptor.
template <typename T> inline constexpr bool is_async_alias_v = is_async_alias<std::remove_cvref_t<T>>::value;

namespace detail
{

// Poison ordinary lookup so only an ADL-visible write-through operation can
// make an alias assignable.
void assign_through() = delete;

template <typename Target, typename Source>
concept async_alias_assignable_from = is_async_alias_v<Target> && requires(Target& target, Source&& source) {
  { assign_through(target, std::forward<Source>(source)) } -> std::same_as<void>;
};

template <typename Target, typename Source>
concept async_alias_assignment_source = async_alias_assignable_from<Target, async_value_t<Source> const&>;

template <typename Target, typename Source>
  requires async_alias_assignable_from<Target, Source>
void assign_async_alias(Target& target, Source&& source)
{
  assign_through(target, std::forward<Source>(source));
}

} // namespace detail
} // namespace uni20::async
