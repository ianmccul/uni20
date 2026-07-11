#pragma once

/**
 * \file dispatch.hpp
 * \ingroup linalg
 * \brief Minimal operation-tag backend dispatch helpers for dense linalg.
 */

#include <uni20/common/trace.hpp>

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Type-level backend eligibility for an operation and argument set.
enum class KernelTypeAcceptance
{
  no,
  maybe,
  yes
};

/// \brief Ordered list of backend values tried by the dispatch walk.
template <class... Backends> struct backend_list
{
    std::tuple<Backends...> entries;

    constexpr explicit backend_list(Backends... backends) : entries(std::move(backends)...) {}
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

namespace detail
{
template <class T> extern std::remove_reference_t<T> kernel_type_probe_object;

template <class T> constexpr std::remove_reference_t<T>& kernel_type_probe_arg() noexcept
{
  return kernel_type_probe_object<T>;
}

template <class Backend, class Op, class... Args> consteval bool backend_has_try_kernel()
{
  return requires(Backend backend, Op op) {
    { try_kernel(backend, op, kernel_type_probe_arg<Args>()...) } -> std::same_as<bool>;
  };
}
} // namespace detail

/// \brief Return the type-level acceptance for one backend and operation.
template <class Backend, class Op, class... Args> consteval KernelTypeAcceptance kernel_type_acceptance()
{
  if constexpr (requires {
                  {
                    kernel_accepts_types(detail::kernel_type_probe_arg<Backend const&>(),
                                         detail::kernel_type_probe_arg<Op const&>(),
                                         detail::kernel_type_probe_arg<Args>()...)
                  } -> std::same_as<KernelTypeAcceptance>;
                })
  {
    constexpr auto acceptance =
        kernel_accepts_types(detail::kernel_type_probe_arg<Backend const&>(),
                             detail::kernel_type_probe_arg<Op const&>(), detail::kernel_type_probe_arg<Args>()...);
    if constexpr (acceptance == KernelTypeAcceptance::no)
    {
      return KernelTypeAcceptance::no;
    }
    else
    {
      static_assert(detail::backend_has_try_kernel<Backend, Op, Args...>(),
                    "kernel_accepts_types accepted these types, but try_kernel is not available");
      return acceptance;
    }
  }
  else if constexpr (detail::backend_has_try_kernel<Backend, Op, Args...>())
  {
    return KernelTypeAcceptance::maybe;
  }
  else
  {
    return KernelTypeAcceptance::no;
  }
}

template <class Backends, class Op, class... Args> inline constexpr bool any_kernel_type_eligible_v = false;

template <class... Backends, class Op, class... Args>
inline constexpr bool any_kernel_type_eligible_v<backend_list<Backends...>, Op, Args...> =
    ((kernel_type_acceptance<Backends, Op, Args...>() != KernelTypeAcceptance::no) || ...);

namespace detail
{
template <std::size_t Index, class... Backends, class Op, class... Args>
bool dispatch_kernel_at(backend_list<Backends...> const& backends, Op op, Args&&... args)
{
  if constexpr (Index == sizeof...(Backends))
  {
    return false;
  }
  else
  {
    using backend_type = std::tuple_element_t<Index, std::tuple<Backends...>>;
    constexpr auto acceptance = kernel_type_acceptance<backend_type, Op, Args&&...>();

    if constexpr (acceptance == KernelTypeAcceptance::yes)
    {
      bool const success = try_kernel(std::get<Index>(backends.entries), op, std::forward<Args>(args)...);
      CHECK(success);
      return true;
    }
    else if constexpr (acceptance == KernelTypeAcceptance::maybe)
    {
      if (try_kernel(std::get<Index>(backends.entries), op, std::forward<Args>(args)...))
      {
        return true;
      }
    }

    return dispatch_kernel_at<Index + 1>(backends, op, std::forward<Args>(args)...);
  }
}
} // namespace detail

/// \brief Try each eligible backend until one performs the operation.
template <class... Backends, class Op, class... Args>
bool dispatch_kernel(backend_list<Backends...> const& backends, Op op, Args&&... args)
{
  return detail::dispatch_kernel_at<0>(backends, op, std::forward<Args>(args)...);
}

} // namespace uni20::linalg
