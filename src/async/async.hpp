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

namespace uni20::async
{

/// \brief Asynchronous value container supporting ordered R/W via coroutines.
/// \tparam T The type of the stored value.
template <typename T> class Async {
  public:
    /// \brief Default-construct the stored value.
    /// \post Stored value is value-initialized; no pending operations.
    constexpr Async() noexcept = default;

    /// \brief Construct the stored value from \p v.
    /// \param v Initial value to store.
    constexpr Async(T v) noexcept : data_(std::move(v)) {}

    /// \brief Acquire a snapshot‐read gate awaitable.
    /// \note The returned ReadBuffer<T> must outlive any references bound to the value
    ///       returned by co_await, to ensure the snapshot remains valid.
    /// \return A ReadBuffer<T> that suspends until all prior writes complete and then
    ///         yields a copy of the stored value.
    // ReadBuffer<T> read() noexcept { return ReadBuffer<T>(this, queue_.new_reader()); }
    ReadBuffer<T> read() noexcept { return ReadBuffer<T>(queue_.create_read_context(this)); }

    /// \brief Acquire an in‐place write gate awaitable.
    /// \note The returned WriteBuffer<T> must outlive any references bound to its
    ///       await_resume() result, and any assignment through that reference must
    ///       occur before the WriteBuffer is destroyed.
    /// \return A WriteBuffer<T> that suspends until it’s safe to write (after all
    ///         prior reads and writes), then yields a mutable reference to the value.
    WriteBuffer<T> write() noexcept { return WriteBuffer<T>(queue_.create_write_context(this)); }

    /// \brief Blocking access: drive \p sched until all prior writes finish.
    /// \note In coroutine context it is much better to co_await on a read() or
    ///       write() buffer.
    /// \tparam Sched Scheduler type (must implement run()).
    /// \param sched Scheduler instance used to advance coroutines.
    /// \return Reference to the stored value.
    template <typename Sched> T& get_wait(Sched& sched)
    {
      while (queue_.has_pending_writers())
      {
        TRACE("Has pending writers");
        sched.run();
      }
      return data_;
    }

    // FiXME: this is a hack for debugging
    void set(T const& x) { data_ = x; }

    // FiXME: this is a hack for debugging
    T value() const { return data_; }

  private:
    friend class ReadBuffer<T>;
    friend class EpochContextReader<T>;
    friend class WriteBuffer<T>;
    friend class EpochContextWriter<T>;

    /// \brief Pointer to the stored data.
    /// \return Address of the contained T.
    T* data() noexcept { return &data_; }

    T data_;           ///< The stored value.
    EpochQueue queue_; ///< Underlying R/W synchronization queue.
};

} // namespace uni20::async
