/// \file epoch_queue.hpp
/// \brief FIFO queue enforcing writer→readers→next-writer ordering.

#pragma once

#include <uni20/common/trace.hpp>
#include "epoch_context.hpp"
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

namespace uni20::async
{

/// \brief Coordinates multiple epochs of read/write gates.
class EpochQueue {
  public:
    /// \brief Construct a queue with one initial epoch.
    EpochQueue() : current_(std::make_shared<EpochContext>()) { TRACE_MODULE(ASYNC, "EpochQueue Constructor", this); }

    /// \brief Destroy the epoch queue.
    ~EpochQueue() { TRACE_MODULE(ASYNC, "EpochQueue Destructor", this); }

    /// \brief Create a read context bound to the current epoch.
    /// \tparam T Stored value type.
    /// \param storage Shared value storage.
    /// \return Reader handle for the current epoch.
    template <typename T> [[nodiscard]] EpochContextReader<T> create_read_context(shared_storage<T> storage) const
    {
      TRACE_MODULE(ASYNC, "EpochQueue::create_read_context", this);
      // if (!current_->has_writer() && storage.constructed()) current_->start();
      return EpochContextReader<T>(storage, current_);
    }

    /// \brief Create a write context, advancing to a new epoch when needed.
    /// \tparam T Stored value type.
    /// \param storage Shared value storage.
    /// \return Writer handle for the selected epoch.
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

    /// \brief Returns the latest epoch context.
    /// \return Shared pointer to the current epoch.
    [[nodiscard]] std::shared_ptr<EpochContext> latest() const noexcept { return current_; }

    /// \brief Reports whether the latest epoch already has a writer.
    /// \return `true` when a writer is pending in the current epoch.
    [[nodiscard]] bool has_pending_writers() const noexcept { return current_ && current_->has_writer(); }

  private:
    std::shared_ptr<EpochContext> current_;
};

// FIXME: we can probably fix the oddities with the ordering of getting the buffers by
// pre-constructing the WriteBuffer.
// If the first epoch goes away then we cancel() it
/// \brief Reverse-order epoch queue used by reverse-mode gradient accumulation.
class ReverseEpochQueue {
  public:
    /// \brief Construct a reverse queue with one initial epoch.
    ReverseEpochQueue() : first_(std::make_shared<EpochContext>())
    {
      TRACE("Constructing ReverseEpochQueue EpochContext", first_.get());
    }

    /// \brief Non-copyable.
    ReverseEpochQueue(ReverseEpochQueue&) = delete;
    /// \brief Non-copyable.
    ReverseEpochQueue& operator=(ReverseEpochQueue&) = delete;

    /// \brief Move constructor.
    ReverseEpochQueue(ReverseEpochQueue&&) = default;
    /// \brief Move assignment.
    ReverseEpochQueue& operator=(ReverseEpochQueue&&) = default;

    /// \brief Destructor.
    ~ReverseEpochQueue() = default;

    /// \brief Construct a ReverseEpochQueue from a given EpochContext.
    explicit ReverseEpochQueue(std::shared_ptr<EpochContext> first) : first_(std::move(first))
    {
      TRACE_MODULE(ASYNC, "ReverseEpochQueue Constructor", this);
      DEBUG_CHECK(!first_->has_writer());
    }

    /// \brief Create a read context from the earliest active reverse epoch.
    /// \tparam T Stored value type.
    /// \param storage Shared value storage.
    /// \return Reader handle for reverse traversal.
    template <typename T> [[nodiscard]] EpochContextReader<T> create_read_context(shared_storage<T> storage)
    {
      TRACE_MODULE(ASYNC, "ReverseEpochQueue::create_read_context", this);
      DEBUG_CHECK(first_);
      if (first_->has_writer()) first_ = EpochContext::make_previous(first_);
      return EpochContextReader<T>(storage, first_);
    }

    /// \brief Create a write context from the earliest active reverse epoch.
    /// \tparam T Stored value type.
    /// \param storage Shared value storage.
    /// \return Writer handle for reverse traversal.
    template <typename T> EpochContextWriter<T> create_write_context(shared_storage<T> storage)
    {
      TRACE_MODULE(ASYNC, "ReverseEpochQueue::create_write_context", this);
      DEBUG_CHECK(first_);
      if (first_->has_writer()) first_ = EpochContext::make_previous(first_);
      // CHECK(!first_->has_writer());
      return EpochContextWriter<T>(storage, first_);
    }

    /// \brief Start running the reverse queue.
    /// \details After start, the initial epoch is consumed and cannot be accessed again.
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

    /// \brief Reports whether the reverse queue has already started.
    /// \return `true` once `start()` has consumed the initial epoch.
    [[nodiscard]] bool is_started() const { return !first_; }

  private:
    std::shared_ptr<EpochContext> first_;
};

} // namespace uni20::async
