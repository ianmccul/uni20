#pragma once

/**
 * \file dispatch.hpp
 * \ingroup linalg
 * \brief Minimal operation-tag backend dispatch helpers for dense linalg.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/dispatch_error.hpp>
#include <uni20/linalg/dispatch_error_presentation.hpp>

#include <concepts>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::linalg
{

namespace detail
{
template <class T> using kernel_type_probe_arg_t = std::remove_reference_t<T>&;

template <class T> struct is_kernel_acceptance : std::false_type
{};

template <KernelTypeAcceptance Acceptance> struct is_kernel_acceptance<KernelAcceptance<Acceptance>> : std::true_type
{};

template <class Backend, class Op, class... Args> consteval bool backend_has_try_kernel()
{
  return requires {
    {
      try_kernel(std::declval<Backend const&>(), std::declval<Op const&>(),
                 std::declval<kernel_type_probe_arg_t<Args>>()...)
    } -> std::same_as<bool>;
  };
}

/// \brief Query one backend's type-level acceptance.
/// \details Argument values are not inspected. Their deduced types are queried
///          through unevaluated lvalue expressions so narrowly constrained
///          backend gates can be detected safely.
template <class Backend, class Op, class... Args> consteval KernelTypeAcceptance backend_type_acceptance()
{
  using backend_type = std::remove_cvref_t<Backend>;
  using op_type = std::remove_cvref_t<Op>;
  if constexpr (requires {
                  kernel_accepts_types(std::declval<backend_type const&>(), std::declval<op_type const&>(),
                                       std::declval<kernel_type_probe_arg_t<Args>>()...);
                })
  {
    using result_type = std::remove_cvref_t<decltype(kernel_accepts_types(
        std::declval<backend_type const&>(), std::declval<op_type const&>(),
        std::declval<kernel_type_probe_arg_t<Args>>()...))>;
    if constexpr (!is_kernel_acceptance<result_type>::value)
    {
      static_assert(is_kernel_acceptance<result_type>::value,
                    "kernel_accepts_types must return kernel_types_no, kernel_types_maybe, or kernel_types_yes");
      return KernelTypeAcceptance::no;
    }
    else
    {
      constexpr auto acceptance = result_type::value;
      if constexpr (acceptance == KernelTypeAcceptance::no)
      {
        return KernelTypeAcceptance::no;
      }
      else
      {
        static_assert(backend_has_try_kernel<backend_type, op_type, Args...>(),
                      "kernel_accepts_types accepted these types, but try_kernel is not available");
        return acceptance;
      }
    }
  }
  return KernelTypeAcceptance::no;
}
} // namespace detail

namespace detail
{
template <class BackendSelector>
using normalized_backend_selector_t =
    std::remove_cvref_t<decltype(normalize_backend_selector(std::declval<BackendSelector>()))>;

template <class Op, class... Args, class... Backends>
consteval KernelTypeAcceptance probe_backend_list_types(std::type_identity<backend_list<Backends...>>)
{
  constexpr bool any_yes = ((backend_type_acceptance<Backends, Op, Args...>() == KernelTypeAcceptance::yes) || ...);
  if constexpr (any_yes)
  {
    return KernelTypeAcceptance::yes;
  }

  constexpr bool any_maybe = ((backend_type_acceptance<Backends, Op, Args...>() == KernelTypeAcceptance::maybe) || ...);
  if constexpr (any_maybe)
  {
    return KernelTypeAcceptance::maybe;
  }

  return KernelTypeAcceptance::no;
}
} // namespace detail

/// \brief Probe whether a backend selector can dispatch an operation for the argument types.
/// \details Returns `yes` if any backend accepts all runtime instances, `maybe`
///          if at least one backend may accept an instance, and `no` otherwise.
///          Argument values and backend state are not inspected. A single backend
///          value is normalized to a one-entry backend list.
template <class BackendSelector, class Op, class... Args>
constexpr KernelTypeAcceptance probe_dispatch_kernel(BackendSelector&&, Op const&, Args&&...)
{
  using backends_type = detail::normalized_backend_selector_t<BackendSelector>;
  static_assert(is_backend_list_v<backends_type>, "backend selectors must normalize to backend_list");
  return detail::probe_backend_list_types<std::remove_cvref_t<Op>, Args...>(std::type_identity<backends_type>{});
}

namespace detail
{
template <class BackendList, class Op, class... Args>
concept KernelDispatchTypesAccepted =
    probe_backend_list_types<std::remove_cvref_t<Op>, Args...>(std::type_identity<BackendList>{}) !=
    KernelTypeAcceptance::no;

template <class Entity>
concept NamedKernelDispatchEntity = requires {
  { std::remove_cvref_t<Entity>::name } -> std::convertible_to<std::string_view>;
};

template <KernelDispatchFailure Failure, class... Backends, class Op, class... Args>
KernelDispatchError make_kernel_dispatch_error(backend_list<Backends...> const&, Op const&, Args&&...)
{
  static_assert(NamedKernelDispatchEntity<Op>, "kernel operation tags must define a static name");
  static_assert((NamedKernelDispatchEntity<Backends> && ...), "kernel backends must define a static name");

  std::vector<KernelBackendAttempt> attempts;
  attempts.reserve(sizeof...(Backends));
  auto append_attempt = [&]<class Backend>() {
    constexpr auto acceptance = backend_type_acceptance<Backend, Op, Args...>();
    attempts.push_back(KernelBackendAttempt{
        .backend = std::string(std::remove_cvref_t<Backend>::name),
        .type_acceptance = acceptance,
        .attempted =
            Failure == KernelDispatchFailure::all_candidates_declined && acceptance != KernelTypeAcceptance::no,
    });
  };
  (append_attempt.template operator()<Backends>(), ...);

  return KernelDispatchError(std::string(std::remove_cvref_t<Op>::name), Failure, std::move(attempts));
}

template <std::size_t Index, class... Backends, class Op, class... Args>
bool try_dispatch_kernel_at(backend_list<Backends...> const& backends, Op op, Args&&... args)
{
  if constexpr (Index == sizeof...(Backends))
  {
    return false;
  }
  else
  {
    using backend_type = std::tuple_element_t<Index, std::tuple<Backends...>>;
    constexpr auto acceptance = backend_type_acceptance<backend_type, Op, Args...>();

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

    return try_dispatch_kernel_at<Index + 1>(backends, op, std::forward<Args>(args)...);
  }
}

template <class... Backends, class Op, class... Args>
void dispatch_backend_list(backend_list<Backends...> const& backends, Op op, Args&&... args)
{
  if (!try_dispatch_kernel_at<0>(backends, op, std::forward<Args>(args)...))
  {
    trace::raise(make_kernel_dispatch_error<KernelDispatchFailure::all_candidates_declined>(backends, op, args...));
  }
}
} // namespace detail

/// \brief Normalize a selector and try each eligible backend until one performs the operation.
template <class BackendSelector, class Op, class... Args>
  requires detail::KernelDispatchTypesAccepted<detail::normalized_backend_selector_t<BackendSelector>, Op, Args...>
bool try_dispatch_kernel(BackendSelector&& selector, Op op, Args&&... args)
{
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  return detail::try_dispatch_kernel_at<0>(backends, op, std::forward<Args>(args)...);
}

/// \brief Normalize a selector and dispatch or report that every eligible backend declined.
template <class BackendSelector, class Op, class... Args>
  requires detail::KernelDispatchTypesAccepted<detail::normalized_backend_selector_t<BackendSelector>, Op, Args...>
void dispatch_kernel(BackendSelector&& selector, Op op, Args&&... args)
{
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  detail::dispatch_backend_list(backends, op, std::forward<Args>(args)...);
}

/// \brief Dynamically dispatch an operation, raising even type-level rejection at runtime.
/// \details This boundary is intended for Python bindings and runtime-erased interfaces that must remain callable
///          when the configured backend list has no implementation for the concrete argument types.
template <class BackendSelector, class Op, class... Args>
void dynamic_dispatch_kernel(BackendSelector&& selector, Op op, Args&&... args)
{
  using backends_type = detail::normalized_backend_selector_t<BackendSelector>;
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  if constexpr (!detail::KernelDispatchTypesAccepted<backends_type, Op, Args...>)
  {
    trace::raise(detail::make_kernel_dispatch_error<KernelDispatchFailure::no_eligible_backend>(backends, op, args...));
  }
  else
  {
    detail::dispatch_backend_list(backends, op, std::forward<Args>(args)...);
  }
}

} // namespace uni20::linalg
