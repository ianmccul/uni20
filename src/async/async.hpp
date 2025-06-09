/// \file async.hpp
/// \brief The Async<T> container: coroutine‐safe asynchronous read/write.
/// \ingroup async_api

// NOTE: Immediately-invoked coroutine lambdas must not capture variables.
// Captures (by reference or value) are stored in the lambda frame, which is destroyed
// after the lambda returns. If the coroutine suspends, any captured state becomes dangling.
// Instead, pass all needed data via parameters, which are safely moved into the coroutine frame.

#pragma once

#include "buffers.hpp"
#include "epoch_queue.hpp"
#include <memory>

namespace uni20::async
{

class DebugScheduler;

namespace detail
{
/// \brief Internal reference-counted data and coordination for Async<T>
///
/// Holds the actual value of type T and the epoch queue used to
/// coordinate asynchronous read and write operations.
///
/// Instances are managed by shared_ptr and never copied or moved directly.
/// All access is mediated through owning Async<T> or active buffers.
template <typename T> struct AsyncImpl
{
    T value_;          ///< Stored data
    EpochQueue queue_; ///< Coordination structure

    AsyncImpl() = default;

    explicit AsyncImpl(const T& val) : value_(val) {}
    explicit AsyncImpl(T&& val) : value_(std::move(val)) {}

    AsyncImpl(const AsyncImpl&) = delete;
    AsyncImpl& operator=(const AsyncImpl&) = delete;
};

} // namespace detail

/// \brief Async<T> is a move-only container for asynchronously accessed data.
///
/// `Async<T>` stores a value of type `T` and mediates access through
/// epoch-based coordination. The value and access queue are jointly
/// refcounted by internal shared state, allowing buffer handles
/// to outlive the owning Async container.
///
/// Copying is disabled: deep copy must be performed via explicit kernels.
///
/// \note Buffers maintain shared ownership of the internal state, so
/// `ReadBuffer<T>` and `WriteBuffer<T>` may safely outlive the Async.
///
/// \note The value of T must be copyable or movable as appropriate for construction.
template <typename T> class Async {
  public:
    using value_type = T;

    /// \brief Default-constructs a T value and an empty access queue.
    Async() : impl_(std::make_shared<detail::AsyncImpl<T>>()) {}

    /// \brief Constructs with a copy of an initial value that can be implicitly converted to T
    template <typename U>
      requires std::convertible_to<U, T>
    Async(U&& val) : impl_(std::make_shared<detail::AsyncImpl<T>>(std::forward<U>(val)))
    {}

    /// \brief Explicit conversion ctor for cases where there is an explicit, but no implicit, conversion
    ///        from U to T
    template <typename U>
      requires std::constructible_from<T, U> && (!std::convertible_to<U, T>)
    explicit Async(U&& u) : impl_(std::make_shared<detail::AsyncImpl<T>>(static_cast<T>(std::forward<U>(u))))
    {}

    /// \brief Construct a new Async<T> by copying the value from another Async<T>.
    ///
    /// \note This constructor schedules a coroutine that reads the current or eventual value of `rhs`
    ///       and writes it into the initial epoch of the newly constructed `*this`.
    ///
    /// \warning This is not a structural copy — it does not replicate the state or dependencies of `rhs`.
    ///          Coroutine handles, epoch queues, and computation histories are not copied.
    ///
    /// \see `async_assign` for explicit value-level copy scheduling.
    Async(const Async& rhs) : Async() { async_assign(rhs, *this); }

    /// \brief Copy-assign from another Async<T>, overwriting this instance's value timeline.
    ///
    /// \note This operator first resets the internal epoch queue of `*this` by move-assigning a fresh `Async<T>`.
    ///       It then schedules a coroutine that awaits `rhs` and writes its result to `*this`.
    ///
    /// \warning This operation does not preserve prior epochs or dependencies of `*this`.
    ///          If you wish to serialize with prior writes, use `async_assign(rhs, *this)` directly.
    ///
    /// \code
    ///   Async<T> x, y;
    ///   x = y;              // copies value from y into x, resets x's causal history
    ///
    ///   x = Async<T>{};     // explicitly reset x
    ///   async_assign(y, x); // equivalent to copy-assignment
    /// \endcode
    ///
    /// \see Async::operator=(Async&&) for structural replacement
    /// \see async_assign for explicit value-copy semantics
    Async& operator=(const Async& rhs)
    {
      if (this != &rhs)
      {
        *this = Async<T>{}; // reset the epoch queue
        async_assign(rhs, *this);
      }
      return *this;
    }

    /// \note `Async<T>` supports standard move construction and assignment. These operations
    ///       transfer the handle (logical reference to the async value), not the value itself.
    ///       To schedule a value transfer from one async object to another, use `async_move(...)` explicitly.
    ///       The results of both operations are rather similar.
    Async(Async&&) noexcept = default;
    Async& operator=(Async&&) noexcept = default;

    ~Async() = default;

    /// \brief Begin an asynchronous read of the value.
    /// \return A ReadBuffer<T> which may be co_awaited.
    ReadBuffer<T> read() const noexcept
    {
      DEBUG_CHECK(impl_);
      return ReadBuffer<T>(impl_->queue_.create_read_context(impl_));
    }

    /// \brief Begin an asynchronous write to the value.
    /// \return A WriteBuffer<T> which may be co_awaited.
    WriteBuffer<T> write() noexcept
    {
      DEBUG_CHECK(impl_);
      return WriteBuffer<T>(impl_->queue_.create_write_context(impl_));
    }

    // Used by FutureValue<T>; TODO: make private and friend
    std::tuple<WriteBuffer<T>, ReadBuffer<T>> prepend_epoch()
    {
      DEBUG_TRACE("Prepending epoch!");
      // std::abort();
      DEBUG_CHECK(impl_);
      return impl_->queue_.prepend_epoch(impl_);
    }

    template <typename Sched> T& get_wait(Sched& sched)
    {
      DEBUG_CHECK(impl_);
      while (impl_->queue_.has_pending_writers())
      {
        TRACE("Has pending writers");
        sched.run();
      }
      return impl_->value_;
    }

    T const& get_wait() const;

    // TODO: we could have a version that returns a ref-counted proxy, which enables reference rather than copy

    /// \brief Block until the value is available, and move it into the return value (which is a write operation)
    T move_from_wait();

    // FiXME: this is a hack for debugging
    void set(T const& x)
    {
      DEBUG_CHECK(impl_);
      impl_->value_ = x;
    }

    // FiXME: this is a hack for debugging
    T value() const
    {
      DEBUG_CHECK(impl_);
      return impl_->value_;
    }

    /// \return Direct reference to stored value (for diagnostics only).
    T const& unsafe_value_ref() const
    {
      DEBUG_CHECK(impl_);
      return impl_->value_;
    }
    T& unsafe_value_ref()
    {
      DEBUG_CHECK(impl_);
      return impl_->value_;
    }

    /// \return Access to underlying implementation (shared with buffers).
    std::shared_ptr<detail::AsyncImpl<T>> const& impl() const { return impl_; }

  private:
    std::shared_ptr<detail::AsyncImpl<T>> impl_;
};

template <typename T> ReadBuffer<T> read(Async<T> const& a) { return a.read(); }
template <typename T> WriteBuffer<T> write(Async<T>& a) { return a.write(); }

} // namespace uni20::async

#include "async-impl.hpp"
