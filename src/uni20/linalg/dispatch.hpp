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

/// \brief Query one backend's type-level acceptance.
/// \details Argument values are not inspected. Their deduced types are queried
///          through probe lvalues so narrowly constrained backend gates can be
///          detected safely.
template <class Backend, class Op, class... Args>
constexpr KernelTypeAcceptance backend_type_acceptance(Backend const&, Op const&, Args&&...)
{
  using backend_type = std::remove_cvref_t<Backend>;
  using op_type = std::remove_cvref_t<Op>;
  if constexpr (requires {
                  {
                    kernel_accepts_types(kernel_type_probe_arg<backend_type const&>(),
                                         kernel_type_probe_arg<op_type const&>(), kernel_type_probe_arg<Args>()...)
                  } -> std::same_as<KernelTypeAcceptance>;
                })
  {
    constexpr auto acceptance =
        kernel_accepts_types(kernel_type_probe_arg<backend_type const&>(), kernel_type_probe_arg<op_type const&>(),
                             kernel_type_probe_arg<Args>()...);
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
  return KernelTypeAcceptance::no;
}
} // namespace detail

/// \brief Probe whether an ordered backend list can dispatch an operation for the argument types.
/// \details Returns `yes` if any backend accepts all runtime instances, `maybe`
///          if at least one backend may accept an instance, and `no` otherwise.
///          Argument values and backend state are not inspected.
template <class... Backends, class Op, class... Args>
constexpr KernelTypeAcceptance probe_dispatch_kernel(backend_list<Backends...> const&, Op const&, Args&&...)
{
  constexpr bool any_yes =
      ((detail::backend_type_acceptance(detail::kernel_type_probe_arg<Backends const&>(),
                                        detail::kernel_type_probe_arg<std::remove_cvref_t<Op> const&>(),
                                        detail::kernel_type_probe_arg<Args>()...) == KernelTypeAcceptance::yes) ||
       ...);
  if constexpr (any_yes)
  {
    return KernelTypeAcceptance::yes;
  }

  constexpr bool any_maybe =
      ((detail::backend_type_acceptance(detail::kernel_type_probe_arg<Backends const&>(),
                                        detail::kernel_type_probe_arg<std::remove_cvref_t<Op> const&>(),
                                        detail::kernel_type_probe_arg<Args>()...) == KernelTypeAcceptance::maybe) ||
       ...);
  if constexpr (any_maybe)
  {
    return KernelTypeAcceptance::maybe;
  }

  return KernelTypeAcceptance::no;
}

namespace detail
{
template <class BackendList, class Op, class... Args>
concept KernelDispatchTypesAccepted =
    probe_dispatch_kernel(kernel_type_probe_arg<BackendList const&>(),
                          kernel_type_probe_arg<std::remove_cvref_t<Op> const&>(),
                          kernel_type_probe_arg<Args>()...) != KernelTypeAcceptance::no;

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
    constexpr auto acceptance = backend_type_acceptance(
        kernel_type_probe_arg<Backend const&>(), kernel_type_probe_arg<Op const&>(), kernel_type_probe_arg<Args>()...);
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
    constexpr auto acceptance =
        backend_type_acceptance(kernel_type_probe_arg<backend_type const&>(), kernel_type_probe_arg<Op const&>(),
                                kernel_type_probe_arg<Args>()...);

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
} // namespace detail

/// \brief Try each eligible backend until one performs the operation.
template <class... Backends, class Op, class... Args>
  requires detail::KernelDispatchTypesAccepted<backend_list<Backends...>, Op, Args...>
bool try_dispatch_kernel(backend_list<Backends...> const& backends, Op op, Args&&... args)
{
  return detail::try_dispatch_kernel_at<0>(backends, op, std::forward<Args>(args)...);
}

/// \brief Dispatch an operation or report that every eligible backend declined it.
template <class... Backends, class Op, class... Args>
  requires detail::KernelDispatchTypesAccepted<backend_list<Backends...>, Op, Args...>
void dispatch_kernel(backend_list<Backends...> const& backends, Op op, Args&&... args)
{
  if (!try_dispatch_kernel(backends, op, std::forward<Args>(args)...))
  {
    trace::raise(
        detail::make_kernel_dispatch_error<KernelDispatchFailure::all_candidates_declined>(backends, op, args...));
  }
}

/// \brief Dynamically dispatch an operation, raising even type-level rejection at runtime.
/// \details This boundary is intended for Python bindings and runtime-erased interfaces that must remain callable
///          when the configured backend list has no implementation for the concrete argument types.
template <class... Backends, class Op, class... Args>
void dynamic_dispatch_kernel(backend_list<Backends...> const& backends, Op op, Args&&... args)
{
  if constexpr (!detail::KernelDispatchTypesAccepted<backend_list<Backends...>, Op, Args...>)
  {
    trace::raise(detail::make_kernel_dispatch_error<KernelDispatchFailure::no_eligible_backend>(backends, op, args...));
  }
  else
  {
    dispatch_kernel(backends, op, std::forward<Args>(args)...);
  }
}

} // namespace uni20::linalg
