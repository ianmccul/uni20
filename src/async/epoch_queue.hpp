/// \file epoch_queue.hpp
/// \brief FIFO queue enforcing writer→readers→next-writer ordering.
/// \ingroup async_core

#pragma once

#include "common/trace.hpp"
#include "epoch_context.hpp"
#include <deque>
#include <mutex>
#include <utility>

namespace uni20::async
{

/// \brief Coordinates multiple epochs of read/write gates.
/// \ingroup async_core
class EpochQueue {
  public:
    /// \brief Default-construct an empty epoch queue.
    EpochQueue() = default;

    /// \brief Start a new reader epoch and return a RAII reader handle.
    /// \tparam T The data type managed by the Async container.
    /// \param parent Pointer to the owning Async<T> instance.
    /// \return A new EpochContextReader<T> bound to the back of the queue.
    template <typename T> EpochContextReader<T> create_read_context(detail::AsyncImplPtr<T> const& parent)
    {
      if (queue_.empty())
      {
        queue_.emplace_back(nullptr, /*writer_already_done=*/true);
      }
      return EpochContextReader<T>(parent, &queue_.back());
    }

    /// \brief Check if there are pending writers ahead of reads.
    /// \return true if any writer is still pending.
    bool has_pending_writers() const noexcept
    {
      std::lock_guard lock(mtx_);
      return queue_.size() > 1 || (queue_.size() == 1 && !queue_.front().writer_is_done());
    }

    /// \brief Start a new writer epoch and return a RAII writer handle.
    /// \tparam T The data type managed by the Async container.
    /// \param parent Pointer to the owning Async<T> instance.
    /// \return A new EpochContextWriter<T> bound to the back of the queue.
    template <typename T> EpochContextWriter<T> create_write_context(detail::AsyncImplPtr<T> const& parent)
    {
      std::lock_guard lock(mtx_);

      if (queue_.empty())
      {
        queue_.emplace_back(nullptr, /*writer_already_done=*/false);
      }
      else
      {
        // create the new epoch first, so we can inherit from the previous
        queue_.emplace_back(&queue_.back(), /*writer_already_done=*/false);

        // Now that we have a new epoch, prune drained epochs from the front of the queue
        // (this cannot drain our recently added epoch, since the writer is not done)
        while (queue_.front().writer_is_done() && queue_.front().reader_is_empty())
        {
          bool writer_required = queue_.front().writer_is_required();
          queue_.pop_front();
          if (writer_required) queue_.front().writer_require();
        }
      }
      return EpochContextWriter<T>(parent, &queue_.back());
    }

    /// \brief Prepend a new epoch to the front of the queue.
    /// \return {EpochContextWriter, EpochContextReader} to the new front epoch.
    /// \pre The current front epoch must not have a writer bound.
    template <typename T> struct EpochPair
    {
        EpochContextWriter<T> writer;
        EpochContextReader<T> reader;
    };

    template <typename T> EpochPair<T> prepend_epoch(detail::AsyncImplPtr<T> const& parent)
    {
      std::lock_guard lock(mtx_);
      if (queue_.empty())
      {
        queue_.emplace_front(nullptr, std::integral_constant<bool, true>{});
      }
      else
      {
        DEBUG_CHECK(!queue_.front().writer_has_task());
        queue_.emplace_front(&queue_.front(), std::integral_constant<bool, true>{});
      }
      return {EpochContextWriter<T>(parent, &queue_.front()), EpochContextReader<T>(parent, &queue_.front())};
    }

    /// \brief Called when a writer coroutine is bound to its epoch.
    /// \param e Epoch to which the writer was bound.
    void on_writer_bound(EpochContext* e) noexcept
    {
      std::unique_lock lock(mtx_);
      if (e == &queue_.front())
      {
        auto task = e->writer_take_task();
        lock.unlock();
        AsyncTask::reschedule(std::move(task));
      }
      else
      {
        TRACE_MODULE(ASYNC, "Writer bound to non-front epoch, deferring");
      }
    }

    /// \brief Conditionally enqueue or immediately schedule a reader task.
    ///
    /// This is called by EpochContextReader to register a suspended coroutine
    /// associated with an epoch. If the epoch is already at the front of the
    /// queue and the writer has completed, the task is scheduled immediately
    /// without being enqueued.
    ///
    /// \param e The EpochContext associated with the reader.
    /// \param task The suspended coroutine representing the reader.
    ///
    /// \note This method must only be called after the reader has been acquired.
    ///       The queue mutex is held during the readiness check to ensure atomicity.
    ///       If the epoch is not ready, the task is stored inside the epoch
    ///       until the queue advances and schedules it.
    void enqueue_reader(EpochContext* e, AsyncTask&& task)
    {
      std::unique_lock lock(mtx_);
      if (e == &queue_.front() && e->reader_is_ready())
      {
        lock.unlock(); // drop the mutex before entering the scheduler
        bool MaybeCancel = e->reader_error() && (e->reader_exception() == nullptr);
        if (!MaybeCancel) task.written();
        AsyncTask::reschedule(std::move(task));
      }
      else
      {
        lock.unlock(); // important to drop the mutex here; if the task is cancelled then it might advance the epoch
        e->reader_enqueue(std::move(task));
      }
    }

    /// \brief Called when a write gate is released (writer done).
    /// \param e Epoch whose writer has completed.
    void on_writer_done(EpochContext* e) noexcept
    {
      TRACE_MODULE(ASYNC, "Writer has finished", e, &queue_.front());
      std::unique_lock lock(mtx_);
      if (e != &queue_.front())
      {
        TRACE_MODULE(ASYNC, "Finished writer is not at the front of the queue; not advancing");
        return;
      }
      // If readers are waiting, schedule them first
      auto readers = e->reader_take_tasks();
      TRACE_MODULE(ASYNC, readers.size());
      if (!readers.empty())
      {
        TRACE_MODULE(ASYNC, "Finished writer results in some readers getting rescheduled");
        lock.unlock();
        bool MaybeCancel = e->reader_error() && (e->reader_exception() == nullptr);
        for (auto&& task : readers)
        {
          TRACE_MODULE(ASYNC, "Rescheduling", &task);
          if (!MaybeCancel) task.written();
          AsyncTask::reschedule(std::move(task));
        }
        return;
      }
      TRACE_MODULE(ASYNC, e, queue_.size(), e->reader_is_empty());
      if (e->reader_is_empty() && queue_.size() > 1)
      {
        // If we have writer_required() then carry it forward to the new epoch
        bool writer_required = e->writer_is_required();
        queue_.pop_front();
        if (writer_required) queue_.front().writer_require();
        lock.unlock();
        advance();
      }
    }

    /// \brief Called when the last reader of an EpochContext has been released.
    ///
    /// Invoked by EpochContextReader only when the reference count reaches zero.
    /// If this epoch is the front of the queue and the writer is also done,
    /// the epoch can be safely removed.
    ///
    /// \param e Epoch whose final reader has been released.
    /// \note It is possible for all readers to be released before the writer is fired,
    ///       if a ReadBuffer is destroyed or released before `await_suspend()` occurs.
    void on_all_readers_released(EpochContext* e) noexcept
    {
      TRACE_MODULE(ASYNC, "readers finished - we might be able to advance the epoch");
      DEBUG_CHECK(e->reader_is_empty());
      std::unique_lock lock(mtx_);
      //  Only pop if this is the front epoch and fully done, and there are other epochs waiting
      if (&queue_.front() != e || !e->writer_is_done() || queue_.size() == 1)
      {
        return;
      }
      bool writer_required = e->writer_is_required();
      queue_.pop_front();
      if (writer_required) queue_.front().writer_require();
      lock.unlock();
      this->advance();
    }

    /// \brief Check if a given epoch is at the front of the queue.
    /// \param e Epoch to test.
    /// \return true if \p e is the front epoch.
    bool is_front(const EpochContext* e) const noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_CHECK(!queue_.empty(), "EpochQueue is empty");
      return e == &queue_.front();
    }

  private:
    /// \brief Advance the queue by scheduling next writer/readers as appropriate.
    void advance() noexcept
    {
      TRACE_MODULE(ASYNC, "advance!");
      while (true)
      {
        std::unique_lock lock(mtx_);
        if (queue_.empty()) return;
        auto* e = &queue_.front();

        TRACE_MODULE(ASYNC, e->writer_has_task());
        e->show();

        // Phase 1: schedule writer if not yet fired
        if (e->writer_has_task())
        {
          auto task = e->writer_take_task();
          lock.unlock();
          AsyncTask::reschedule(std::move(task));
          return;
        }

        // Phase 2: schedule readers if writer done
        if (e->reader_is_ready())
        {
          auto readers = e->reader_take_tasks();
          if (!readers.empty())
          {
            lock.unlock();
            bool MaybeCancel = e->reader_error() && (e->reader_exception() == nullptr);
            for (auto&& task : readers)
            {
              if (!MaybeCancel) task.written();
              AsyncTask::reschedule(std::move(task));
            }
            return;
          }
        }

        // Phase 3: pop epoch if both writer and readers are done, and there are more epochs to come
        if (e->writer_is_done() && e->reader_is_empty() && queue_.size() > 1)
        {
          bool writer_required = e->writer_is_required();
          queue_.pop_front();
          if (writer_required) queue_.front().writer_require();

          continue;
        }

        // Nothing more to do
        return;
      }
    }

    mutable std::mutex mtx_;         ///< Protects queue_.
    std::deque<EpochContext> queue_; ///< Epoch queue.
};

} // namespace uni20::async
