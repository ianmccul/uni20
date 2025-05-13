/// \file awaiters.hpp
/// \brief Utilities for awaiting on Async<T>: co_await overload, all(), try_await().
/// \ingroup async_api

#pragma once

#include "async.hpp"
#include "buffers.hpp"

#include <array>
#include <coroutine>
#include <functional>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::async
{

/// \brief Awaitable that waits for *all* provided awaiters to complete.
/// @tparam Aw Awaitable types.
template <typename... Aw> struct AllAwaiter
{
    std::tuple<Aw...> bufs_;                  ///< Underlying awaiters
    std::array<bool, sizeof...(Aw)> ready_{}; ///< Readiness flags

    /// \brief Check if all awaiters are ready.
    /// \return true if no suspension is required.
    bool await_ready() noexcept { return await_ready_impl(std::make_index_sequence<sizeof...(Aw)>{}); }

    /// \brief Suspend the coroutine on awaiters not yet ready.
    /// \tparam Promise The coroutine’s promise type.
    /// \param h Coroutine handle to suspend.
    template <typename Promise> void await_suspend(std::coroutine_handle<Promise> h) noexcept
    {
      await_suspend_impl(h, std::make_index_sequence<sizeof...(Aw)>{});
    }

    /// \brief Resume all awaiters and collect their results.
    /// \return Tuple of each await_resume() value.
    auto await_resume()
    {
      return std::apply([](auto&... w) { return std::make_tuple(w.await_resume()...); }, bufs_);
    }

  private:
    template <std::size_t... I> bool await_ready_impl(std::index_sequence<I...>) noexcept
    {
      ((ready_[I] = std::get<I>(bufs_).await_ready()), ...);
      return (ready_[I] && ...);
    }

    template <typename Promise, std::size_t... I>
    void await_suspend_impl(std::coroutine_handle<Promise> h, std::index_sequence<I...>) noexcept
    {
      ((ready_[I] ? void() : std::get<I>(bufs_).await_suspend(h)), ...);
    }
};

/// \brief Helper to construct an AllAwaiter.
/// @tparam Aw Awaitable types.
template <typename... Aw> struct AllBuilder
{
    std::tuple<Aw...> bufs_; ///< Underlying awaiters

    /// \brief Store the awaiters.
    /// \param aw The awaitables to wait on.
    explicit AllBuilder(Aw&&... aw) noexcept : bufs_(std::forward<Aw>(aw)...) {}

    /// \brief rvalue overload for `co_await all(...)`.
    /// \return An AllAwaiter configured with the awaiters.
    auto operator co_await() && noexcept { return AllAwaiter<Aw...>{std::move(bufs_), {}}; }

    /// \brief lvalue overload if `auto x = all(...); co_await x;`.
    /// \return An AllAwaiter configured with the awaiters.
    auto operator co_await() & noexcept { return AllAwaiter<Aw...>{bufs_, {}}; }
};

/// \brief Build an awaitable that waits for *all* of the provided awaitables.
/// @param aw Awaitable arguments.
/// @return An object supporting `co_await`.
template <typename... Aw> auto all(Aw&&... aw) noexcept { return AllBuilder<Aw...>{std::forward<Aw>(aw)...}; }

//------------------------------------------------------------------------------
// Non-blocking await: try_await
//------------------------------------------------------------------------------

/// \brief Detects if `T` has a member `operator co_await()`.
template <typename T>
concept HasMemberCoAwait = requires(T t) { t.operator co_await(); };

/// \brief Detects if a free `operator co_await(t)` exists.
/// \note Excludes types that already have a member operator.
template <typename T>
concept HasFreeCoAwait = (!HasMemberCoAwait<T> && requires(T t) { operator co_await(t); });

/// \brief Get the “true” awaiter for an object outside a coroutine.
/// Handles member and free `operator co_await`, else forwards.
/// @tparam U Awaitable or awaiter type.
/// @param u The object to transform.
/// @return The awaiter object to use.
template <typename U> decltype(auto) get_awaiter(U&& u) noexcept
{
  if constexpr (HasMemberCoAwait<U>)
  {
    return std::forward<U>(u).operator co_await();
  }
  else if constexpr (HasFreeCoAwait<U>)
  {
    return operator co_await(std::forward<U>(u));
  }
  else
  {
    return std::forward<U>(u);
  }
}

/// \brief Awaiter wrapper that returns an optional result.
/// @tparam Awt Either an awaiter type or a reference to one.
template <typename Awt> struct TryAwaiter
{
    Awt awaiter_; ///< either a reference or an owned awaiter

    bool await_ready() const noexcept { return awaiter_.await_ready(); }

    template <typename Promise> void await_suspend(std::coroutine_handle<Promise> h) noexcept
    {
      awaiter_.await_suspend(h);
    }

    auto await_resume() noexcept
    {
      using R = decltype(awaiter_.await_resume());
      // If R is a reference, StoreT is reference_wrapper<remove_reference_t<R>>,
      // otherwise it's just R.
      using StoreT =
          std::conditional_t<std::is_lvalue_reference_v<R>, std::reference_wrapper<std::remove_reference_t<R>>, R>;

      if (!awaiter_.await_ready())
      {
        // no value yet → empty optional
        return std::optional<StoreT>{};
      }

      if constexpr (std::is_lvalue_reference_v<R>)
      {
        // need to wrap the reference
        return std::optional<StoreT>{std::ref(awaiter_.await_resume())};
      }
      else
      {
        // plain value
        return std::optional<StoreT>{awaiter_.await_resume()};
      }
    }
};

/// \brief Builder for `try_await`, capturing the awaiter by reference or value.
/// @tparam Aw The exact type of your awaiter (often `ReadBuffer<T>&` or similar).
template <typename Aw> struct TryBuilder
{
    Aw awaiter_; ///< holds either a reference or an owned awaiter

    /// \brief Perfect-forward constructor.
    /// \param aw The awaiter you want to wrap.
    explicit constexpr TryBuilder(Aw aw) noexcept : awaiter_(std::forward<Aw>(aw)) {}

    /// \brief co_await on an lvalue builder
    auto operator co_await() & noexcept { return TryAwaiter<Aw>{std::forward<Aw>(awaiter_)}; }

    /// \brief co_await on an rvalue builder
    auto operator co_await() && noexcept { return TryAwaiter<Aw>{std::forward<Aw>(awaiter_)}; }
};

/// \brief Build a non-blocking awaiter that returns `optional<T>` instead of suspending.
///
/// If you pass an lvalue awaiter, it’s stored by reference.
/// If you pass a prvalue awaiter (e.g. a proxy type you’ve designed), it’s stored by value.
///
/// Example:
/// ```cpp
/// ReadBuffer<int> rbuf = I.read();
/// // ok: Aw = ReadBuffer<int>&, builder holds rbuf by reference
/// auto maybe_i = co_await try_await(rbuf);
///
/// // also ok if AsyncProxyAwaiter is a small, copyable awaiter
/// auto proxy = AsyncProxyAwaiter{/*…*/};
/// auto maybe_v = co_await try_await(proxy);
/// ```
///
/// \param aw An awaitable (must outlive the coroutine for references).
/// \returns A TryBuilder<Aw> that will create a TryAwaiter when co_awaited.
template <typename Aw> constexpr auto try_await(Aw&& aw) noexcept
{
  // Aw deduced as:
  //   - ReadBuffer<T>& if aw is an lvalue ReadBuffer<T>
  //   - ProxyType  if aw is a prvalue ProxyType
  return TryBuilder<Aw>{std::forward<Aw>(aw)};
}

} // namespace uni20::async
