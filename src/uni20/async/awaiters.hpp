/// \file awaiters.hpp
/// \brief Utilities for awaiting on Async<T>: co_await overload, all(), try_await().

#pragma once

#include "async.hpp"
#include "buffers.hpp"

#include <array>
#include <concepts>
#include <coroutine>
#include <functional>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::async
{

/// \brief Always-ready awaitable owning an immediate operation value.
/// \details This gives immediate values the read-buffer shape expected by
///          async kernels. The value is copied or moved into the coroutine and
///          `release()` is a no-op.
template <typename T> struct ValueAwaiter
{
    using value_type = T;

    T value;

    /// \brief Report that the immediate value never suspends.
    [[nodiscard]] bool await_ready() const noexcept { return true; }

    /// \brief No-op suspension hook required by the awaiter protocol.
    void await_suspend(BasicTask) const noexcept {}

    /// \brief Return a borrowed reference when the awaiter is retained.
    [[nodiscard]] T const& await_resume() const& noexcept { return value; }

    /// \brief Move the value when the awaiter itself is consumed.
    [[nodiscard]] T await_resume() && noexcept { return std::move(value); }

    /// \brief Match the read-buffer release surface without owning an epoch.
    void release() const noexcept {}
};

namespace detail
{
template <typename T> using member_read_awaiter_t = std::remove_cvref_t<decltype(std::declval<T>().read())>;

template <typename T>
concept AsyncReadProvider = requires(T&& value) { std::forward<T>(value).read(); } &&
                            TaskAwaitable<member_read_awaiter_t<T>> && requires(member_read_awaiter_t<T>& awaiter) {
                              typename member_read_awaiter_t<T>::value_type;
                              { awaiter.await_ready() } -> std::convertible_to<bool>;
                              awaiter.await_resume();
                            };
} // namespace detail

/// \brief Wrap an immediate value in an always-ready read awaiter.
template <typename T>
  requires(!detail::AsyncReadProvider<T>)
[[nodiscard]] auto read(T&& value) -> ValueAwaiter<std::remove_cvref_t<T>>
{
  return ValueAwaiter<std::remove_cvref_t<T>>{std::forward<T>(value)};
}

/// \brief Obtain the read awaiter exposed by an async-like value.
template <typename T>
  requires detail::AsyncReadProvider<T>
[[nodiscard]] decltype(auto) read(T&& value)
{
  return std::forward<T>(value).read();
}

/// \brief Awaitable that waits for *all* provided awaiters to complete. This meets the TaskFactoryAwaitable
/// concept.
/// \tparam Aw Registration-style awaiters that return `void` from `await_suspend(BasicTask)`.
/// \note the child await_resume() functions must return a non-void.
/// \todo We currently don't allow nested waiting on TaskFactoryAwaitable children. This could be supported, if it
/// was useful.
template <TaskFactoryChildAwaitable... Aw>
  requires((!std::is_void_v<decltype(std::declval<Aw>().await_resume())> && ...))
struct AllAwaiter //: public AsyncAwaiter
{
    std::tuple<Aw...> bufs_;                  ///< Underlying awaiters
    std::array<bool, sizeof...(Aw)> ready_{}; ///< Readiness flags
    int pending_ = sizeof...(Aw);             ///< Number of awaiters that still need to suspend

    /// \brief Check if all awaiters are ready.
    /// \return true if no suspension is required.
    [[nodiscard]] bool await_ready() noexcept { return await_ready_impl(std::make_index_sequence<sizeof...(Aw)>{}); }

    /// \brief Return the number of awaiters, needed by the TaskFactory model.
    /// \note It is safe to over-allocate: unused BasicTasks will be returned in the factory destructor.
    [[nodiscard]] int num_awaiters() const noexcept { return pending_; }

#if UNI20_DEBUG_DAG
    /// \brief Visits child awaiter dependencies exposed to the debug DAG layer.
    /// \tparam F Callable accepting `(NodeInfo const*, TaskRegistry::EpochTaskRole)`.
    /// \param visit Callback invoked for each child dependency.
    template <typename F> void debug_each_dependency(F&& visit) const
    {
      std::apply([&](auto const&... awaiters) { (debug_visit_dependency(awaiters, visit), ...); }, bufs_);
    }
#endif

    /// \brief Suspend the coroutine on awaiters not yet ready.
    /// \tparam Promise The coroutine’s promise type.
    /// \param f TaskFactory that provides one BasicTask per sub-awaitable
    void await_suspend(TaskFactory f) noexcept
    {
      await_suspend_impl(std::move(f), std::make_index_sequence<sizeof...(Aw)>{});
    }

    /// \brief Resume all awaiters and collect their results.
    /// \return Tuple of each await_resume() value. Make sure we preserve the exact type
    /// returned by the client awaiters, so references are preserved.
    [[nodiscard]] auto await_resume()
    {
      return std::apply([](auto&&... w) -> decltype(auto) { return std::forward_as_tuple(w.await_resume()...); },
                        bufs_);
    }

  private:
#if UNI20_DEBUG_DAG
    template <typename A, typename F> static void debug_visit_dependency(A const& awaiter, F& visit)
    {
      if constexpr (requires {
                      awaiter.node();
                      awaiter.debug_task_role();
                    })
      {
        visit(awaiter.node(), awaiter.debug_task_role());
      }
    }
#endif

    template <std::size_t... I> bool await_ready_impl(std::index_sequence<I...>) noexcept
    {
      pending_ = 0;
      ((ready_[I] = std::get<I>(bufs_).await_ready(), pending_ += ready_[I] ? 0 : 1), ...);
      return pending_ == 0;
    }

    template <std::size_t... I> void await_suspend_impl(TaskFactory f, std::index_sequence<I...>) noexcept
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
/// \param aw Awaitable arguments.
/// \return An object supporting `co_await`.
template <TaskFactoryChildAwaitable... Aw>
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

/// \brief for an awaitable `a`, return the actual Awaitable object returned by co_await
/// \tparam U Awaitable or awaiter type.
/// \param u The object to transform.
/// \return The awaiter object to use.
/// \note Within a coroutine, this type may be transformed into something else via
///       the promise_type::await_transform() function.
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
/// \tparam Awt Either an awaiter type or a reference to one.
template <typename Awt> struct TryAwaiter
{
    using Awaiter = std::remove_reference_t<Awt>;
    using StoredAwaiter = std::conditional_t<std::is_lvalue_reference_v<Awt>, Awaiter*, Awt>;

    StoredAwaiter awaiter_;

    constexpr explicit TryAwaiter(Awt aw) noexcept : awaiter_(store(std::forward<Awt>(aw))) {}

    [[nodiscard]] bool await_ready() const noexcept { return true; }

    void await_suspend(BasicTask t) noexcept { access().await_suspend(std::move(t)); }

    [[nodiscard]] auto await_resume() noexcept
    {
      auto& inner = access();
      using R = decltype(inner.await_resume());
      using StoreT =
          std::conditional_t<std::is_lvalue_reference_v<R>, std::reference_wrapper<std::remove_reference_t<R>>, R>;

      if (!inner.await_ready())
      {
        return std::optional<StoreT>{};
      }

      if constexpr (std::is_lvalue_reference_v<R>)
      {
        return std::optional<StoreT>{std::ref(inner.await_resume())};
      }
      else
      {
        return std::optional<StoreT>{inner.await_resume()};
      }
    }

  private:
    static constexpr StoredAwaiter store(Awt aw) noexcept
    {
      if constexpr (std::is_lvalue_reference_v<Awt>)
      {
        return std::addressof(aw);
      }
      else
      {
        return std::move(aw);
      }
    }

    [[nodiscard]] constexpr Awaiter& access() noexcept
    {
      if constexpr (std::is_lvalue_reference_v<Awt>)
      {
        return *awaiter_;
      }
      else
      {
        return awaiter_;
      }
    }

    [[nodiscard]] constexpr Awaiter const& access() const noexcept
    {
      if constexpr (std::is_lvalue_reference_v<Awt>)
      {
        return *awaiter_;
      }
      else
      {
        return awaiter_;
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
/// \return A `TryAwaiter` wrapper around the supplied awaitable.
template <typename Aw> constexpr auto try_await(Aw& aw) noexcept { return TryAwaiter<Aw&>{aw}; }

/// \brief Build a non-blocking awaiter from an rvalue awaitable.
/// \tparam Aw Awaitable type.
/// \param aw Awaitable value moved into the try-await wrapper.
/// \return A `TryAwaiter` that owns the awaitable value.
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
/// acquiring a WriteBuffer<T>, suspending if needed, and emplacing the
/// provided value at resume.
///
/// \tparam Buffer A WriteBuffer<T> or compatible wrapper, passed by value or reference.
/// \tparam Value The type of the value to be written. Moved into the buffer at resume.
///
/// \note Use this when you want to write a computed value directly into an Async<T>
///       in one expression. Especially useful when the WriteBuffer is a temporary.
///       `co_await write_to(buffer, v)` is equivalent to `co_await buffer = std::move(v)`
///
/// \warning The WriteBuffer must not be reused after passing to write_to. For named
///          buffers, prefer `write_to(buffer.transfer(), value)` to make that
///          ownership transfer explicit.
template <typename T, typename Value> class WriteToAwaiter {
  public:
    WriteToAwaiter(WriteBuffer<T>&& buffer, Value&& value)
        : buffer_(std::move(buffer)), value_(std::forward<Value>(value))
    {
      static_assert(std::constructible_from<T, Value&&>, "Value must be constructible as T for WriteBuffer<T>");
    }

    [[nodiscard]] bool await_ready() const noexcept { return buffer_.await_ready(); }

    auto await_suspend(BasicTask&& t) noexcept { return buffer_.await_suspend(std::move(t)); }

    void await_resume() { buffer_.emplace_assert(std::move(value_)); }

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
