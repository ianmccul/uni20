#pragma once

/**
 * \file backend_selector.hpp
 * \ingroup linalg
 * \brief Ordered backend selector values shared by tensor storage and linalg dispatch.
 */

#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Backend value for BLAS dense linalg kernels.
struct BlasBackend
{
    friend constexpr bool operator==(BlasBackend const&, BlasBackend const&) = default;
};

/// \brief Backend value for the generic host CPU dense linalg oracle.
struct CpuGenericBackend
{
    friend constexpr bool operator==(CpuGenericBackend const&, CpuGenericBackend const&) = default;
};

/// \brief Ordered list of backend values tried by the dispatch walk.
template <class... Backends> struct backend_list
{
    std::tuple<Backends...> entries;

    constexpr explicit backend_list(Backends... backends) : entries(std::move(backends)...) {}

    friend constexpr bool operator==(backend_list const&, backend_list const&) = default;
};

template <class... Backends> backend_list(Backends...) -> backend_list<std::remove_cvref_t<Backends>...>;

template <class T> struct is_backend_list : std::false_type
{};

template <class... Backends> struct is_backend_list<backend_list<Backends...>> : std::true_type
{};

template <class T> inline constexpr bool is_backend_list_v = is_backend_list<std::remove_cvref_t<T>>::value;

/// \brief Return an explicit backend list unchanged.
template <class... Backends>
[[nodiscard]] constexpr auto normalize_backend_selector(backend_list<Backends...> selector) -> backend_list<Backends...>
{
  return selector;
}

/// \brief Treat a single backend value as a one-entry backend list.
template <class Backend>
  requires(!is_backend_list_v<Backend>)
[[nodiscard]] constexpr auto normalize_backend_selector(Backend&& backend)
{
  return backend_list{std::forward<Backend>(backend)};
}

} // namespace uni20::linalg
