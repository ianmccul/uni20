#pragma once
#include "async.hpp"
#include "buffers.hpp"

namespace uni20::async
{

/// FutureValue<T> owns a write-capable handle (WriteBuffer<T>) to an Async<T> value.
/// It allows writing the value once, while supporting standard Async<T> reading and awaiting.
template <typename T> class FutureValue {
  public:
    using value_type = T;

    /// Construct a new, uninitialized FutureValue.
    FutureValue() : async_{}, write_buf_{async_.write()} {}

    // move ctor and move-assign should "just work"
    FutureValue(FutureValue&&) noexcept = default;
    FutureValue& operator=(FutureValue&&) noexcept = default;

    // copy-asssignment doesn't make sense
    FutureValue(FutureValue const&) = delete;

    /// Access as an Async<T> (read-only use).
    Async<T> const& async() const noexcept { return async_; }

    /// Get a ReadBuffer<T> from the underlying Async.
    ReadBuffer<T> read() const { return async_.read(); }

    /// Get the WriteBuffer<T> — allows a single write, then must be released.
    [[nodiscard]] WriteBuffer<T> write() noexcept { return std::move(write_buf_); }

    /// Since we are guaranteed that the write is immediate, we don't need to wait
    template <typename U>
    FutureValue& operator=(U&& v)
      requires std::constructible_from<T, U&&>
    {
      write_buf_.emplace_assert(std::forward<U>(v));
      write_buf_.release();
      return *this;
    }

    // Assigning from an Async<T> is possible; this launches a coroutine do to the copy
    template <typename U> FutureValue& operator=(Async<U> const& v)
    {
      async_assign(v.read(), std::move(write_buf_));
      return *this;
    }

    // Assigning from an Async<T> is possible; this launches a coroutine do to the copy
    template <typename U> FutureValue& operator=(Async<U>&& v)
    {
      async_move(std::move(v), std::move(write_buf_));
      return *this;
    }

    Async<T>& value() { return async_; }
    Async<T> const& value() const { return async_; }

  private:
    Async<T> async_;
    WriteBuffer<T> write_buf_;
};

template <typename T> class Defer {
  public:
    using value_type = T;

    // Movable
    Defer(Defer&&) = default;

    // But not construcible or copyable or move-assignable
    Defer() = delete;
    Defer(Defer const&) = delete;
    Defer& operator=(Defer&&) = delete;
    Defer& operator=(Defer const&) = delete;

    explicit Defer(Async<T>& w) : writer_(w.write()) {}

    explicit Defer(WriteBuffer<T>&& w) : writer_(std::move(w)) {}

    /// \brief Write immediately without suspending — asserts write readiness.
    template <typename U>
      requires std::constructible_from<T, U&&>
    void write_assert(U&& val)
    {
      writer_.emplace_assert(std::forward<U>(val));
    }

    template <typename U>
      requires std::assignable_from<T&, async_value_t<U>>
    void operator=(U&& val)
    {
      async_assign(std::forward<U>(val), std::move(writer_));
    }

    /// \brief Get the WriteBuffer for coroutine-based use.
    [[nodiscard]] WriteProxy<T> write() { return writer_.write(); }

    /// Release the reference count;
    void release() { writer_.release(); }

  private:
    WriteBuffer<T> writer_;
};

// CTAD guide
template <typename T> Defer(Async<T>&) -> Defer<T>;

/// free function to get a deferred writer on `a`
template <typename T> Defer<T> defer_write(Async<T>& a) { return Defer<T>(a); }

} // namespace uni20::async
