#pragma once

/**
 * \file dispatch.hpp
 * \ingroup linalg
 * \brief Operation-tag backend dispatch over tensor-view operands.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/linalg/dispatch_diagnostics.hpp>
#include <uni20/linalg/dispatch_error.hpp>
#include <uni20/linalg/dispatch_error_presentation.hpp>

#include <array>
#include <concepts>
#include <optional>
#include <span>
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
    } -> std::same_as<KernelAttempt>;
  };
}

/// \brief Query one backend's type-level acceptance.
/// \details Argument values are not inspected. Their deduced types are queried
///          through unevaluated lvalue expressions so narrowly constrained
///          backend gates can be detected safely. Runtime dispatch invokes
///          candidates with the same stable lvalue arguments.
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

template <class Backend, class BackendList> struct PrependBackend;

template <class Backend, class... Backends> struct PrependBackend<Backend, backend_list<Backends...>>
{
    using type = backend_list<Backend, Backends...>;
};

template <class BackendList, class Op, class... Args> struct KernelTypeCandidates;

template <class Op, class... Args> struct KernelTypeCandidates<backend_list<>, Op, Args...>
{
    using type = backend_list<>;
};

template <class Backend, class... Backends, class Op, class... Args>
struct KernelTypeCandidates<backend_list<Backend, Backends...>, Op, Args...>
{
  private:
    using tail = typename KernelTypeCandidates<backend_list<Backends...>, Op, Args...>::type;

  public:
    using type = std::conditional_t<backend_type_acceptance<Backend, Op, Args...>() == KernelTypeAcceptance::no, tail,
                                    typename PrependBackend<Backend, tail>::type>;
};

template <class Op, class... Args, class Backend> constexpr auto candidate_backend_tuple(Backend&& backend)
{
  using backend_type = std::remove_cvref_t<Backend>;
  if constexpr (backend_type_acceptance<backend_type, Op, Args...>() == KernelTypeAcceptance::no)
  {
    return std::tuple<>{};
  }
  else
  {
    return std::tuple<backend_type>{std::forward<Backend>(backend)};
  }
}

template <class Tuple, std::size_t... Index>
constexpr auto backend_list_from_tuple(Tuple&& tuple, std::index_sequence<Index...>)
{
  using tuple_type = std::remove_cvref_t<Tuple>;
  return backend_list<std::tuple_element_t<Index, tuple_type>...>{std::get<Index>(std::forward<Tuple>(tuple))...};
}

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

/// \brief Ordered backend-list type whose entries accept the operation argument types.
/// \details Backends with `yes` or `maybe` type acceptance are retained. A
///          missing `kernel_accepts_types` customization is a hard `no` and is
///          omitted. Runtime values and backend state are not inspected.
template <class BackendSelector, class Op, class... Args>
using kernel_type_candidates_t =
    typename detail::KernelTypeCandidates<detail::normalized_backend_selector_t<BackendSelector>,
                                          std::remove_cvref_t<Op>, Args...>::type;

/// \brief Filter a backend selector to candidates accepting the call's argument types.
/// \details The result type is determined entirely at compile time. Retained
///          backend values preserve their original order and state so callers
///          such as conformance tests can exercise each candidate independently.
template <class BackendSelector, class Op, class... Args>
constexpr auto kernel_type_candidates(BackendSelector&& selector, Op const&, Args&&...)
{
  using op_type = std::remove_cvref_t<Op>;
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  auto candidates = std::apply(
      []<class... Backends>(Backends&&... backend) {
        return std::tuple_cat(detail::candidate_backend_tuple<op_type, Args...>(std::forward<Backends>(backend))...);
      },
      std::move(backends.entries));
  auto result = detail::backend_list_from_tuple(std::move(candidates),
                                                std::make_index_sequence<std::tuple_size_v<decltype(candidates)>>{});
  static_assert(std::same_as<decltype(result), kernel_type_candidates_t<BackendSelector, Op, Args...>>);
  return result;
}

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

template <class... Backends, class Op, class... Args>
std::vector<KernelBackendAttempt>
make_kernel_backend_attempts(backend_list<Backends...> const&,
                             std::span<std::optional<KernelAttempt> const> runtime_results, Op const&, Args&&...)
{
  static_assert(NamedKernelDispatchEntity<Op>, "kernel operation tags must define a static name");
  static_assert((NamedKernelDispatchEntity<Backends> && ...), "kernel backends must define a static name");
  CHECK(runtime_results.empty() || runtime_results.size() == sizeof...(Backends));

  std::vector<KernelBackendAttempt> attempts;
  attempts.reserve(sizeof...(Backends));
  std::size_t index = 0;
  auto append_attempt = [&]<class Backend>() {
    constexpr auto acceptance = backend_type_acceptance<Backend, Op, Args...>();
    attempts.push_back(KernelBackendAttempt{
        .backend = std::string(std::remove_cvref_t<Backend>::name),
        .type_acceptance = acceptance,
        .runtime_result = runtime_results.empty() ? std::nullopt : runtime_results[index],
    });
    ++index;
  };
  (append_attempt.template operator()<Backends>(), ...);
  return attempts;
}

template <class... Backends, class Op, class... Args>
dispatch_diagnostics::event
make_kernel_dispatch_diagnostic(backend_list<Backends...> const& backends,
                                std::span<std::optional<KernelAttempt> const> runtime_results, Op const& op,
                                Args&&... args)
{
  return dispatch_diagnostics::event{
      .operation = std::string(std::remove_cvref_t<Op>::name),
      .backend_attempts = make_kernel_backend_attempts(backends, runtime_results, op, args...),
  };
}

template <KernelDispatchFailure Failure, class... Backends, class Op, class... Args>
KernelDispatchError make_kernel_dispatch_error(backend_list<Backends...> const& backends,
                                               std::span<std::optional<KernelAttempt> const> runtime_results,
                                               Op const& op, Args&&... args)
{
  auto diagnostic = make_kernel_dispatch_diagnostic(backends, runtime_results, op, args...);
  return KernelDispatchError(std::move(diagnostic.operation), Failure, std::move(diagnostic.backend_attempts));
}

template <std::size_t Index, class... Backends, class Op, class RecordAttempt, class... Args>
bool try_dispatch_kernel_at(backend_list<Backends...> const& backends, Op const& op, RecordAttempt& record_attempt,
                            Args&&... args)
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
      KernelAttempt const attempt = try_kernel(std::get<Index>(backends.entries), op, args...);
      record_attempt(Index, attempt);
      CHECK(kernel_attempt_succeeded(attempt));
      return true;
    }
    else if constexpr (acceptance == KernelTypeAcceptance::maybe)
    {
      KernelAttempt const attempt = try_kernel(std::get<Index>(backends.entries), op, args...);
      record_attempt(Index, attempt);
      if (kernel_attempt_succeeded(attempt))
      {
        return true;
      }
    }

    return try_dispatch_kernel_at<Index + 1>(backends, op, record_attempt, args...);
  }
}

template <class... Backends, class Op, class... Args>
void dispatch_backend_list(backend_list<Backends...> const& backends, Op const& op, Args&&... args)
{
  bool const diagnostics_enabled = dispatch_diagnostics::enabled();
  std::array<std::optional<KernelAttempt>, sizeof...(Backends)> runtime_results{};
  auto record_attempt = [&](std::size_t index, KernelAttempt attempt) { runtime_results[index] = attempt; };
  bool const succeeded = try_dispatch_kernel_at<0>(backends, op, record_attempt, args...);
  auto const results = std::span<std::optional<KernelAttempt> const>(runtime_results);

  if (succeeded)
  {
    if (diagnostics_enabled)
    {
      auto diagnostic = make_kernel_dispatch_diagnostic(backends, results, op, args...);
      dispatch_diagnostics::detail::emit(diagnostic);
    }
    return;
  }

  if (diagnostics_enabled)
  {
    auto diagnostic = make_kernel_dispatch_diagnostic(backends, results, op, args...);
    dispatch_diagnostics::detail::emit(diagnostic);
    trace::raise(KernelDispatchError(std::move(diagnostic.operation), KernelDispatchFailure::all_candidates_declined,
                                     std::move(diagnostic.backend_attempts)));
  }

  trace::raise(
      make_kernel_dispatch_error<KernelDispatchFailure::all_candidates_declined>(backends, results, op, args...));
}
} // namespace detail

/// \brief Normalize a selector and try each eligible backend until one performs the operation.
/// \details A backend returning a decline result must preserve every argument
///          and produce no externally visible side effect, so a later backend
///          receives the original operation instance intact. Tensor operands
///          are TensorView or DeviceTensorView models; selected backends perform
///          execution-domain acquisition and mdspan lowering.
template <class BackendSelector, class Op, class... Args>
  requires detail::KernelDispatchTypesAccepted<detail::normalized_backend_selector_t<BackendSelector>, Op, Args...>
bool try_dispatch_kernel(BackendSelector&& selector, Op op, Args&&... args)
{
  auto backends = normalize_backend_selector(std::forward<BackendSelector>(selector));
  if (dispatch_diagnostics::enabled())
  {
    constexpr auto backend_count = std::tuple_size_v<decltype(backends.entries)>;
    std::array<std::optional<KernelAttempt>, backend_count> runtime_results{};
    auto record_attempt = [&](std::size_t index, KernelAttempt attempt) { runtime_results[index] = attempt; };
    bool const succeeded = detail::try_dispatch_kernel_at<0>(backends, op, record_attempt, args...);
    auto diagnostic = detail::make_kernel_dispatch_diagnostic(
        backends, std::span<std::optional<KernelAttempt> const>(runtime_results), op, args...);
    dispatch_diagnostics::detail::emit(diagnostic);
    return succeeded;
  }

  auto ignore_attempt = [](std::size_t, KernelAttempt) {};
  return detail::try_dispatch_kernel_at<0>(backends, op, ignore_attempt, args...);
}

/// \brief Normalize a selector and dispatch or report that every eligible backend declined.
/// \details Each runtime decline has the same argument-preservation and
///          no-side-effect contract as `try_dispatch_kernel`. Tensor operands
///          remain tensor views until a selected backend lowers them.
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
    if (dispatch_diagnostics::enabled())
    {
      auto diagnostic = detail::make_kernel_dispatch_diagnostic(
          backends, std::span<std::optional<KernelAttempt> const>{}, op, args...);
      dispatch_diagnostics::detail::emit(diagnostic);
      trace::raise(KernelDispatchError(std::move(diagnostic.operation), KernelDispatchFailure::no_eligible_backend,
                                       std::move(diagnostic.backend_attempts)));
    }
    trace::raise(detail::make_kernel_dispatch_error<KernelDispatchFailure::no_eligible_backend>(
        backends, std::span<std::optional<KernelAttempt> const>{}, op, args...));
  }
  else
  {
    detail::dispatch_backend_list(backends, op, std::forward<Args>(args)...);
  }
}

} // namespace uni20::linalg
