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

template <typename T> using AsyncImplPtr = std::shared_ptr<AsyncImpl<T>>;

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
    /// \brief Default-constructs a T value and an empty access queue.
    Async() : impl_(std::make_shared<AsyncImpl<T>>()) {}

    /// \brief Constructs with a copy of an initial value.
    Async(const T& val) : impl_(std::make_shared<AsyncImpl<T>>(val)) {}

    /// \brief Constructs with a moved initial value.
    Async(T&& val) : impl_(std::make_shared<AsyncImpl<T>>(std::move(val))) {}

    Async(const Async&) = delete;
    Async& operator=(const Async&) = delete;

    Async(Async&&) noexcept = default;
    Async& operator=(Async&&) noexcept = default;

    ~Async() = default;

    /// \brief Begin an asynchronous read of the value.
    /// \return A ReadBuffer<T> which may be co_awaited.
    ReadBuffer<T> read() noexcept { return ReadBuffer<T>(impl_->queue_.create_read_context(impl_)); }

    /// \brief Begin an asynchronous write to the value.
    /// \return A WriteBuffer<T> which may be co_awaited.
    WriteBuffer<T> write() noexcept { return WriteBuffer<T>(impl_->queue_.create_write_context(impl_)); }

    template <typename Sched> T& get_wait(Sched& sched)
    {
      while (impl_->queue_.has_pending_writers())
      {
        TRACE("Has pending writers");
        sched.run();
      }
      return impl_->value_;
    }

    // FiXME: this is a hack for debugging
    void set(T const& x) { impl_->value_ = x; }

    // FiXME: this is a hack for debugging
    T value() const { return impl_->value_; }

    /// \return Direct reference to stored value (for diagnostics only).
    T const& unsafe_value_ref() const { return impl_->value_; }
    T& unsafe_value_ref() { return impl_->value_; }

    /// \return Access to underlying implementation (shared with buffers).
    std::shared_ptr<AsyncImpl<T>> const& impl() const { return impl_; }

  private:
    std::shared_ptr<AsyncImpl<T>> impl_;
};

} // namespace uni20::async
