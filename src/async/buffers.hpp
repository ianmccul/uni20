/// \file buffers.hpp
/// \brief Awaitable gates for Async<T>: snapshot‐reads and in‐place writes.
/// \ingroup async_api

#pragma once

#include "async_task.hpp"
#include "common/trace.hpp"
#include "epoch_context.hpp"

#include <coroutine>
#include <utility>

namespace uni20::async
{

template <typename T> class Async;

/// \brief RAII awaitable for snapshot‐reads of an Async<T>.
/// @tparam T The stored value type.
template <typename T> class ReadBuffer {
  public:
    /// \brief Construct a read gate awaitable.
    /// @param parent Pointer to the Async<T> container.
    /// @param epoch  Pointer to the EpochContext for this read.
    ReadBuffer(Async<T>* parent, EpochContext* epoch) noexcept : parent_(parent), epoch_(epoch) {}

    ReadBuffer(ReadBuffer const&) = delete;
    ReadBuffer& operator=(ReadBuffer const&) = delete;

    /// \brief Move-construct, transferring ownership.
    /// Suppresses the destructor action in @p other.
    ReadBuffer(ReadBuffer&& other) noexcept : parent_(other.parent_), epoch_(other.epoch_), done_(other.done_)
    {
      other.done_ = true;
    }

    /// \brief Move-assign, releasing any held gate first.
    ReadBuffer& operator=(ReadBuffer&& other) noexcept
    {
      if (!done_)
      {
        epoch_->mark_reader_done();
        parent_->queue_.on_reader_done(epoch_);
      }
      parent_ = other.parent_;
      epoch_ = other.epoch_;
      done_ = other.done_;
      other.done_ = true;
      return *this;
    }

    /// \brief Ready-check: true if the writer has completed.
    /// \return True ⇒ await_suspend() will not be called.
    bool await_ready() const noexcept
    {
      DEBUG_CHECK(!done_, "ReadBuffer used after release()!");
      DEBUG_TRACE("ReadBuffer await_ready()", epoch_->readers_ready(), epoch_);
      return epoch_->readers_ready();
    }

    /// \brief Suspend this coroutine and enqueue as a reader.
    /// \tparam Promise The coroutine’s promise type.
    /// \param h The coroutine handle to suspend.
    template <typename Promise> void await_suspend(std::coroutine_handle<Promise> h) noexcept
    {
      auto hh = std::coroutine_handle<AsyncTask::promise_type>::from_address(h.address());
      epoch_->add_reader(hh);
      TRACE("Suspending ReadBuffer", epoch_);
    }

    /// \brief Resume and return a const reference to the stored value.
    /// \return Reference to the stored T inside Async<T>.
    T const& await_resume() const noexcept
    {
      TRACE("Resuming ReadBuffer", epoch_);
      return *parent_->data();
    }

    /// \brief Manually release the read gate without suspension.
    void release() noexcept
    {
      epoch_->mark_reader_done();
      parent_->queue_.on_reader_done(epoch_);
      done_ = true;
    }

    /// \brief Destructor: auto-release if not already done.
    ~ReadBuffer()
    {
      TRACE("Destroying ReadBuffer", done_);
      if (!done_)
      {
        TRACE(epoch_);
        epoch_->mark_reader_done();
        parent_->queue_.on_reader_done(epoch_);
      }
    }

  private:
    Async<T>* parent_;    ///< The Async<T> being read.
    EpochContext* epoch_; ///< EpochContext managing ordering.
    bool done_ = false;   ///< True if released or moved-from.
};

/// \brief RAII awaitable for in‐place writes to an Async<T>.
/// @tparam T The stored value type.
template <typename T> class WriteBuffer {
  public:
    /// \brief Construct a write gate awaitable.
    /// @param parent Pointer to the Async<T> container.
    /// @param epoch  Pointer to the EpochContext for this write.
    WriteBuffer(Async<T>* parent, EpochContext* epoch) noexcept : parent_(parent), epoch_(epoch) {}

    WriteBuffer(WriteBuffer const&) = delete;
    WriteBuffer& operator=(WriteBuffer const&) = delete;

    /// \brief Move-construct, transferring ownership.
    WriteBuffer(WriteBuffer&& other) noexcept : parent_(other.parent_), epoch_(other.epoch_), done_(other.done_)
    {
      other.done_ = true;
    }

    /// \brief Move-assign, releasing any held gate first.
    WriteBuffer& operator=(WriteBuffer&& other) noexcept
    {
      if (!done_)
      {
        epoch_->mark_writer_done();
        parent_->queue_.on_writer_done(epoch_);
      }
      parent_ = other.parent_;
      epoch_ = other.epoch_;
      done_ = other.done_;
      other.done_ = true;
      return *this;
    }

    /// \brief Ready-check: true if it's this write’s turn.
    /// \return True ⇒ await_suspend() will not be called.
    bool await_ready() const noexcept
    {
      DEBUG_CHECK(!done_, "WriteBuffer used after release()!");
      DEBUG_CHECK(!epoch_->writer_has_task(), "Unexpected: double bind!");
      TRACE("WriteBuffer await_ready()", parent_->queue_.is_front(epoch_), epoch_);
      return parent_->queue_.is_front(epoch_);
    }

    /// \brief Suspend this coroutine until it can write.
    /// \tparam Promise The coroutine’s promise type.
    /// \param h The coroutine handle to suspend.
    template <typename Promise> void await_suspend(std::coroutine_handle<Promise> h) noexcept
    {
      auto hh = std::coroutine_handle<AsyncTask::promise_type>::from_address(h.address());
      epoch_->bind_writer(hh);
      parent_->queue_.on_writer_bound(epoch_);
      TRACE("Suspending WriteBuffer", epoch_);
    }

    /// \brief Resume and return a reference to the stored value.
    /// \return Reference to the stored T inside Async<T>.
    T& await_resume() const noexcept
    {
      TRACE("Resuming WriteBuffer", epoch_);
      return *parent_->data();
    }

    /// \brief Manually release the write gate without suspension.
    void release() noexcept
    {
      epoch_->mark_writer_done();
      parent_->queue_.on_writer_done(epoch_);
      done_ = true;
    }

    /// \brief Destructor: auto-release if not already done.
    ~WriteBuffer()
    {
      TRACE("Destroying WriteBuffer", done_);
      if (!done_)
      {
        TRACE(epoch_);
        epoch_->mark_writer_done();
        parent_->queue_.on_writer_done(epoch_);
      }
    }

  private:
    Async<T>* parent_;    ///< The Async<T> being written.
    EpochContext* epoch_; ///< EpochContext managing ordering.
    bool done_ = false;   ///< True if released or moved-from.
};

} // namespace uni20::async
