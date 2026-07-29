#pragma once

/**
 * \file dispatch.hpp
 * \ingroup linalg
 * \brief Coroutine-aware kernel dispatch with an ordinary blocking fallback.
 */

#include <uni20/async/async_task.hpp>
#include <uni20/linalg/async/kernel_task.hpp>
#include <uni20/linalg/dispatch.hpp>

#include <array>
#include <optional>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{

template <class Backend, class Op, class... Args> consteval bool backend_has_kernel_task_factory()
{
  if constexpr (requires {
                  try_make_kernel_task(std::declval<Backend const&>(), std::declval<Op const&>(),
                                       std::declval<kernel_type_probe_arg_t<Args>>()...);
                })
  {
    using result_type = decltype(try_make_kernel_task(std::declval<Backend const&>(), std::declval<Op const&>(),
                                                      std::declval<kernel_type_probe_arg_t<Args>>()...));
    static_assert(is_kernel_task_attempt<result_type>,
                  "try_make_kernel_task must return KernelTaskAttempt<ConcreteTask>");
    return true;
  }
  return false;
}

template <std::size_t Index, class... Backends, class Op, std::size_t ResultCount, class... Args>
async::AsyncTask co_dispatch_kernel_at(backend_list<Backends...> const& backends, Op const& op,
                                       std::array<std::optional<KernelAttempt>, ResultCount>& runtime_results,
                                       bool& succeeded, Args&... args)
{
  static_assert(ResultCount == sizeof...(Backends));
  if constexpr (Index == sizeof...(Backends))
  {
    co_return;
  }
  else
  {
    using backend_type = std::tuple_element_t<Index, std::tuple<Backends...>>;
    constexpr auto acceptance = backend_type_acceptance<backend_type, Op, Args&...>();

    if constexpr (acceptance != KernelTypeAcceptance::no)
    {
      KernelAttempt attempt;
      if constexpr (backend_has_kernel_task_factory<backend_type, Op, Args&...>())
      {
        auto task_attempt = try_make_kernel_task(std::get<Index>(backends.entries), op, args...);
        attempt = task_attempt.attempt();
        runtime_results[Index] = attempt;
        if (kernel_attempt_succeeded(attempt) && task_attempt.has_task())
        {
          co_await std::move(task_attempt).take_task();
        }
      }
      else
      {
        attempt = try_kernel(std::get<Index>(backends.entries), op, args...);
        runtime_results[Index] = attempt;
      }

      if constexpr (acceptance == KernelTypeAcceptance::yes)
      {
        CHECK(kernel_attempt_succeeded(attempt));
      }

      if (kernel_attempt_succeeded(attempt))
      {
        succeeded = true;
        co_return;
      }
    }

    co_await co_dispatch_kernel_at<Index + 1>(backends, op, runtime_results, succeeded, args...);
  }
}

} // namespace detail

/// \brief Dispatch a kernel from a coroutine without requiring coroutine wrappers for blocking backends.
/// \details Each backend may optionally provide `try_make_kernel_task`,
///          returning a `KernelTaskAttempt` whose concrete task is awaited
///          before dispatch completes. A backend without that customization is
///          invoked through its ordinary `try_kernel` implementation on the
///          current scheduler thread. Runtime declines submit no work and
///          permit fallback; an operation-declared replaceable output may
///          remain provisionally prepared for the next backend.
/// \note Operation arguments are stable lvalues owned by the calling coroutine.
///       This avoids copying arbitrary operands and matches the ordinary dispatch
///       probe and invocation contract.
template <class BackendSelector, class Op, class... Args>
  requires detail::KernelDispatchTypesAccepted<detail::normalized_backend_selector_t<BackendSelector>, Op, Args&...>
async::AsyncTask co_dispatch_kernel(BackendSelector selector, Op op, Args&... args)
{
  auto backends = normalize_backend_selector(std::move(selector));
  constexpr auto backend_count = std::tuple_size_v<decltype(backends.entries)>;
  std::array<std::optional<KernelAttempt>, backend_count> runtime_results{};
  bool succeeded = false;

  co_await detail::co_dispatch_kernel_at<0>(backends, op, runtime_results, succeeded, args...);

  auto const results = std::span<std::optional<KernelAttempt> const>(runtime_results);
  if (succeeded)
  {
    if (dispatch_diagnostics::enabled())
    {
      auto diagnostic = detail::make_kernel_dispatch_diagnostic(backends, results, op, args...);
      dispatch_diagnostics::detail::emit(diagnostic);
    }
    co_return;
  }

  if (dispatch_diagnostics::enabled())
  {
    auto diagnostic = detail::make_kernel_dispatch_diagnostic(backends, results, op, args...);
    dispatch_diagnostics::detail::emit(diagnostic);
    trace::raise(KernelDispatchError(std::move(diagnostic.operation), KernelDispatchFailure::all_candidates_declined,
                                     std::move(diagnostic.backend_attempts)));
  }

  trace::raise(detail::make_kernel_dispatch_error<KernelDispatchFailure::all_candidates_declined>(backends, results, op,
                                                                                                  args...));
}

} // namespace uni20::linalg
