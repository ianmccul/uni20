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

/// \brief Awaitable that waits for *all* provided awaiters to complete. This meets the AsyncTaskFactoryAwaitable
/// concept.
/// \tparam Aw Awaitable types, that must meet the AsyncTaskAwaitable concept
/// \note the child await_resume() functions must return a non-void.
/// \todo We currently don't allow nested waiting on AsyncTaskFactoryAwaitable children. This could be supported, if it
/// was useful.
template <AsyncTaskAwaitable... Aw>
  requires((!std::is_void_v<decltype(std::declval<Aw>().await_resume())> && ...))
struct AllAwaiter
{
    std::tuple<Aw...> bufs_;                  ///< Underlying awaiters
    std::array<bool, sizeof...(Aw)> ready_{}; ///< Readiness flags

    /// \brief Check if all awaiters are ready.
    /// \return true if no suspension is required.
    bool await_ready() noexcept { return await_ready_impl(std::make_index_sequence<sizeof...(Aw)>{}); }

    /// \brief Return the number of awaiters, needed by the AsyncTaskFactory model.
    /// \note It is safe to over-allocate: unused AsyncTasks will be returned in the factory destructor.
    int num_awaiters() const noexcept { return sizeof...(Aw); }

    /// \brief Suspend the coroutine on awaiters not yet ready.
    /// \tparam Promise The coroutine’s promise type.
    /// \param f AsyncTaskFactory that provides one AsyncTask per sub-awaitable
    void await_suspend(AsyncTaskFactory f) noexcept
    {
      await_suspend_impl(std::move(f), std::make_index_sequence<sizeof...(Aw)>{});
    }

    /// \brief Resume all awaiters and collect their results.
    /// \return Tuple of each await_resume() value. Make sure we preserve the exact type
    /// returned by the client awaiters, so references are preserved.
    auto await_resume()
    {
      return std::apply([](auto&&... w) -> decltype(auto) { return std::forward_as_tuple(w.await_resume()...); },
                        bufs_);
    }

  private:
    template <std::size_t... I> bool await_ready_impl(std::index_sequence<I...>) noexcept
    {
      ((ready_[I] = std::get<I>(bufs_).await_ready()), ...);
      DEBUG_TRACE(ready_);
      return (ready_[I] && ...);
    }

    template <std::size_t... I> void await_suspend_impl(AsyncTaskFactory f, std::index_sequence<I...>) noexcept
    {
      ((ready_[I] ? void() : std::get<I>(bufs_).await_suspend(f.take_next())), ...);
    }
};

namespace detail
{
template <typename T> struct MapToRefOrValue
{
    static_assert(std::is_trivially_move_constructible_v<T>,
                  "Prvalue passed to all(...) must be trivially move constructible");
    using type = T;
};

template <typename T> struct MapToRefOrValue<T&&>
{
    static_assert(std::is_trivially_move_constructible_v<T>,
                  "Rvalue passed to all(...) must be trivially move constructible");
    using type = T;
};

template <typename T> struct MapToRefOrValue<T&>
{
    using type = T&;
};
} // namespace detail

/// \brief Build an awaitable that waits for *all* of the provided awaitables.
/// @param aw Awaitable arguments.
/// @return An object supporting `co_await`.
template <AsyncTaskAwaitable... Aw>
  requires((!std::is_void_v<decltype(std::declval<Aw>().await_resume())> && ...))
auto all(Aw&&... aw) noexcept
{
  return AllAwaiter<typename detail::MapToRefOrValue<Aw>::type...>(
      std::tuple<typename detail::MapToRefOrValue<Aw>::type...>(std::forward<Aw>(aw)...));
}

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
    Awt& awaiter_;

    bool await_ready() const noexcept { return true; }

    void await_suspend(AsyncTask t) noexcept { awaiter_.await_suspend(std::move(t)); }

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
/// \returns A TryAwaiter<Aw>
template <typename Aw> constexpr auto try_await(Aw& aw) noexcept { return TryAwaiter<Aw>{aw}; }

template <typename Aw> constexpr auto try_await(Aw&& aw) noexcept
{
  static_assert(std::is_trivially_move_constructible_v<Aw>, "try_await(x) on prvalues requires T to be safely movable");
  return TryAwaiter<Aw>{std::move(aw)};
}

namespace detail
{
/// \brief Awaitable that writes a value to a WriteBuffer<T>.
///
/// This awaiter performs a write of a value to an Async<T> object by
/// acquiring a WriteBuffer<T>, suspending if needed, and assigning the
/// provided value at resume.
///
/// \tparam Buffer A WriteBuffer<T> or compatible wrapper, passed by value or reference.
/// \tparam Value The type of the value to be written. Moved into the buffer at resume.
///
/// \note Use this when you want to write a computed value directly into an Async<T>
///       in one expression. Especially useful when the WriteBuffer is a temporary.
///       `co_await write_to(buffer, v)` is equivalent to `auto& vo = co_await buffer; vo = std::move(v)`
///
/// \warning The WriteBuffer must not be reused after passing to write_to. It is moved
///          into the coroutine and consumed during co_await.
template <typename T, typename Value> class WriteToAwaiter {
  public:
    WriteToAwaiter(WriteBuffer<T>&& buffer, Value&& value)
        : buffer_(std::move(buffer)), value_(std::forward<Value>(value))
    {
      static_assert(std::is_convertible_v<Value, T>, "Value must be convertible to T for WriteBuffer<T>");
    }

    bool await_ready() const noexcept { return buffer_.await_ready(); }

    auto await_suspend(AsyncTask&& t) noexcept { return buffer_.await_suspend(std::move(t)); }

    void await_resume() { buffer_.await_resume() = std::move(value_); }

  private:
    WriteBuffer<T> buffer_;
    std::remove_reference_t<Value> value_;
};
} // namespace detail

/// \brief Create a coroutine awaiter that writes a value to a WriteBuffer<T>.
///
/// Constructs an awaitable object that can be co_awaited to write a given value
/// into an Async<T> via its WriteBuffer. Ensures lifetime safety and correct
/// causal ordering when the buffer is a temporary.
///
/// \param buffer A WriteBuffer<T>, moved into the coroutine.
/// \param value The value to assign, moved into the awaiter frame.
///
/// \return An awaitable object that performs the write upon resumption.
template <typename T, typename Value> auto write_to(WriteBuffer<T>&& buffer, Value value)
{
  return detail::WriteToAwaiter<T, Value>(std::move(buffer), std::move(value));
}

} // namespace uni20::async
