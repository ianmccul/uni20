#pragma once
/// \file future_value.hpp
/// \brief Write-once async wrappers (`FutureValue` and `Defer`) built on `Async<T>`.
#include "async.hpp"
#include "buffers.hpp"

namespace uni20::async
{

/// \brief Write-once async value wrapper that retains a pending write buffer.
/// \tparam T Stored value type.
template <typename T> class FutureValue {
  public:
    using value_type = T;

    /// \brief Construct a new, uninitialized FutureValue.
    FutureValue() : async_{}, write_buf_{async_.write()} {}

    /// \brief Move constructor.
    FutureValue(FutureValue&&) noexcept = default;
    /// \brief Move assignment.
    FutureValue& operator=(FutureValue&&) noexcept = default;

    /// \brief Non-copyable.
    FutureValue(FutureValue const&) = delete;

    /// \brief Access the wrapped Async value as read-only.
    /// \return Const reference to the wrapped Async value.
    Async<T> const& async() const noexcept { return async_; }

    /// \brief Get a read buffer from the wrapped Async value.
    /// \return Read buffer for the wrapped Async value.
    ReadBuffer<T> read() const { return async_.read(); }

    /// \brief Get the retained write buffer.
    /// \return Write buffer that can be used once before release.
    [[nodiscard]] WriteBuffer<T> write() noexcept { return std::move(write_buf_); }

    /// \brief Assign a value immediately into the retained write buffer.
    /// \tparam U Source type constructible as `T`.
    /// \param v Source value.
    /// \return Reference to `*this`.
    template <typename U> FutureValue& operator=(U&& v) requires std::constructible_from<T, U&&>
    {
      write_buf_.emplace_assert(std::forward<U>(v));
      write_buf_.release();
      return *this;
    }

    /// \brief Assign from another async source by scheduling an async copy.
    /// \tparam U Source value type.
    /// \param v Source async value.
    /// \return Reference to `*this`.
    template <typename U> FutureValue& operator=(Async<U> const& v)
    {
      async_assign(v.read(), std::move(write_buf_));
      return *this;
    }

    /// \brief Move-assign from another async source by scheduling an async move.
    /// \tparam U Source value type.
    /// \param v Source async value.
    /// \return Reference to `*this`.
    template <typename U> FutureValue& operator=(Async<U>&& v)
    {
      async_move(std::move(v), std::move(write_buf_));
      return *this;
    }

    /// \brief Mutable access to the wrapped Async value.
    /// \return Reference to the underlying Async.
    Async<T>& value() { return async_; }
    /// \brief Const access to the wrapped Async value.
    /// \return Const reference to the underlying Async.
    Async<T> const& value() const { return async_; }

  private:
    Async<T> async_;
    WriteBuffer<T> write_buf_;
};

/// \brief Move-only deferred writer helper for a single `WriteBuffer<T>`.
/// \tparam T Value type written through the deferred handle.
template <typename T> class Defer {
  public:
    using value_type = T;

    /// \brief Move constructor.
    Defer(Defer&&) = default;

    /// \brief No default construction.
    Defer() = delete;
    /// \brief Non-copyable.
    Defer(Defer const&) = delete;
    /// \brief Non-move-assignable.
    Defer& operator=(Defer&&) = delete;
    /// \brief Non-copyable.
    Defer& operator=(Defer const&) = delete;

    /// \brief Construct from an Async writer endpoint.
    /// \param w Async value whose write buffer should be retained.
    explicit Defer(Async<T>& w) : writer_(w.write()) {}

    /// \brief Construct from an existing write buffer.
    /// \param w Write buffer to retain.
    explicit Defer(WriteBuffer<T>&& w) : writer_(std::move(w)) {}

    /// \brief Write immediately without suspending — asserts write readiness.
    template <typename U>
    requires std::constructible_from<T, U&&>
    void write_assert(U&& val) { writer_.emplace_assert(std::forward<U>(val)); }

    /// \brief Schedule assignment through the deferred writer.
    /// \tparam U Source type.
    /// \param val Source expression to assign.
    template <typename U>
    requires std::assignable_from<T&, async_value_t<U>>
    void operator=(U&& val) { async_assign(std::forward<U>(val), std::move(writer_)); }

    /// \brief Get the WriteBuffer for coroutine-based use.
    [[nodiscard]] WriteProxy<T> write() { return writer_.write(); }

    /// \brief Release the retained write buffer early.
    void release() { writer_.release(); }

  private:
    WriteBuffer<T> writer_;
};

/// \brief CTAD guide for constructing `Defer<T>` from `Async<T>&`.
template <typename T> Defer(Async<T>&) -> Defer<T>;

/// \brief Create a deferred writer helper for an Async value.
/// \tparam T Stored value type.
/// \param a Async value to write.
/// \return Deferred writer helper.
template <typename T> Defer<T> defer_write(Async<T>& a) { return Defer<T>(a); }

} // namespace uni20::async
