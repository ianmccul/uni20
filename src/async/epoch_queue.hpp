/// \file epoch_queue.hpp
/// \brief FIFO queue enforcing writer→readers→next-writer ordering.
/// \ingroup async_core

#pragma once

#include "common/trace.hpp"
#include "epoch_context.hpp"
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

namespace uni20::async
{

/// \brief Coordinates multiple epochs of read/write gates.
/// \ingroup async_core
class EpochQueue {
  public:
    EpochQueue() : current_(std::make_shared<EpochContext>()) { TRACE_MODULE(ASYNC, "EpochQueue Constructor", this); }

    ~EpochQueue() { TRACE_MODULE(ASYNC, "EpochQueue Destructor", this); }

    template <typename T> EpochContextReader<T> create_read_context(shared_storage<T> storage) const
    {
      TRACE_MODULE(ASYNC, "EpochQueue::create_read_context", this);
      // if (!current_->has_writer() && storage.constructed()) current_->start();
      return EpochContextReader<T>(storage, current_);
    }

    template <typename T> EpochContextWriter<T> create_write_context(shared_storage<T> storage)
    {
      TRACE_MODULE(ASYNC, "EpochQueue::create_write_context", this);
      if (current_->has_writer())
      {
        std::shared_ptr<EpochContext> next = std::make_shared<EpochContext>();
        current_->set_next_epoch(next);
        current_ = next;
      }
      return EpochContextWriter<T>(storage, current_);
    }

    std::shared_ptr<EpochContext> latest() const noexcept { return current_; }

    bool has_pending_writers() const noexcept { return current_ && current_->has_writer(); }

  private:
    std::shared_ptr<EpochContext> current_;
};

// FIXME: we can probably fix the oddities with the ordering of getting the buffers by
// pre-constructing the WriteBuffer.
// If the first epoch goes away then we cancel() it
class ReverseEpochQueue {
  public:
    ReverseEpochQueue() : first_(std::make_shared<EpochContext>())
    {
      TRACE("Constructing ReverseEpochQueue EpochContext", first_.get());
    }

    // move-only
    ReverseEpochQueue(ReverseEpochQueue&) = delete;
    ReverseEpochQueue& operator=(ReverseEpochQueue&) = delete;

    ReverseEpochQueue(ReverseEpochQueue&&) = default;
    ReverseEpochQueue& operator=(ReverseEpochQueue&&) = default;

    ~ReverseEpochQueue() = default;

    /// \brief Construct a ReverseEpochQueue from a given EpochContext.
    explicit ReverseEpochQueue(std::shared_ptr<EpochContext> first) : first_(std::move(first))
    {
      TRACE_MODULE(ASYNC, "ReverseEpochQueue Constructor", this);
      DEBUG_CHECK(!first_->has_writer());
    }

    template <typename T> EpochContextReader<T> create_read_context(shared_storage<T> storage)
    {
      TRACE_MODULE(ASYNC, "ReverseEpochQueue::create_read_context", this);
      DEBUG_CHECK(first_);
      if (first_->has_writer()) first_ = EpochContext::make_previous(first_);
      return EpochContextReader<T>(storage, first_);
    }

    template <typename T> EpochContextWriter<T> create_write_context(shared_storage<T> storage)
    {
      TRACE_MODULE(ASYNC, "ReverseEpochQueue::create_write_context", this);
      DEBUG_CHECK(first_);
      if (first_->has_writer()) first_ = EpochContext::make_previous(first_);
      // CHECK(!first_->has_writer());
      return EpochContextWriter<T>(storage, first_);
    }

    /// Start running the queue. After it starts, we can no longer access the queue
    void start()
    {
      DEBUG_CHECK(first_);
      // CHECK(first_->has_writer());

      // FIXME: we need to start() here in such a way that if we don't have an initial writer then
      // we set an error state (which will generally be absorbed by the backprop accumulation knowing
      // how to handle missing data)
      first_->writer_acquire();
      first_->writer_cancel();
      first_->start();
      first_.reset();
    }

    /// Returns true if the queue has been started.  In that case we can no longer access the initial epoch.
    bool is_started() const { return !first_; }

  private:
    std::shared_ptr<EpochContext> first_;
};

} // namespace uni20::async
