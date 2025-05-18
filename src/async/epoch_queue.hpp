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

    /// \brief Start a new reader in the current (back) epoch.
    /// \return Pointer to the associated EpochContext.
    EpochContext* new_reader() noexcept
    {
      std::lock_guard lock(mtx_);
      if (queue_.empty())
      {
        queue_.emplace_back(/*writer_already_done=*/true);
      }
      TRACE("new_reader, queue size:", queue_.size());
      queue_.back().create_reader();
      return &queue_.back();
    }

    /// \brief Check if there are pending writers ahead of reads.
    /// \return true if any writer is still pending.
    bool has_pending_writers() const noexcept
    {
      std::lock_guard lock(mtx_);
      TRACE(queue_.size());
      return queue_.size() > 1 || (queue_.size() == 1 && !queue_.front().writer_done());
    }

    /// \brief Start a new writer epoch.
    /// \return Pointer to the new EpochContext.
    EpochContext* new_writer() noexcept
    {
      std::lock_guard lock(mtx_);
      queue_.emplace_back(/*writer_already_done=*/false);
      TRACE("new_writer, queue size:", queue_.size());
      return &queue_.back();
    }

    /// \brief Called when a writer coroutine is bound to its epoch.
    /// \param e Epoch to which the writer was bound.
    void on_writer_bound(EpochContext* e) noexcept
    {
      std::unique_lock lock(mtx_);
      if (e == &queue_.front())
      {
        TRACE("Scheduling writer for front epoch");
        auto wh = e->take_writer();
        lock.unlock();
        AsyncTask::reschedule(std::move(wh));
      }
      else
      {
        TRACE("Writer bound to non-front epoch, deferring");
      }
    }

    /// \brief Called when a write gate is released (writer done).
    /// \param e Epoch whose writer has completed.
    void on_writer_done(EpochContext* e) noexcept
    {
      e->mark_writer_done();
      std::unique_lock lock(mtx_);
      if (e != &queue_.front())
      {
        return;
      }
      // If readers are waiting, schedule them first
      auto readers = e->take_readers();
      if (!readers.empty())
      {
        lock.unlock();
        for (auto&& rh : readers)
        {
          AsyncTask::reschedule(std::move(rh));
        }
        return;
      }
      // No readers: pop this epoch and advance
      queue_.pop_front();
      lock.unlock();
      advance();
    }

    /// \brief Called when a reader gate is released.
    /// \param e Epoch whose one reader has completed.
    void on_reader_done(EpochContext* e) noexcept
    {
      e->mark_reader_done();
      std::unique_lock lock(mtx_);
      // Only pop if this is the front epoch and fully done
      if (&queue_.front() != e || !e->writer_done() || !e->readers_empty())
      {
        return;
      }
      queue_.pop_front();
      lock.unlock();
      advance();
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
      while (true)
      {
        std::unique_lock lock(mtx_);
        if (queue_.empty()) return;
        auto* e = &queue_.front();

        // Phase 1: schedule writer if not yet fired
        if (e->writer_has_task())
        {
          auto wh = e->take_writer();
          lock.unlock();
          AsyncTask::reschedule(std::move(wh));
          return;
        }

        // Phase 2: schedule readers if writer done
        if (e->readers_ready())
        {
          auto readers = e->take_readers();
          if (!readers.empty())
          {
            lock.unlock();
            for (auto&& rh : readers)
            {
              AsyncTask::reschedule(std::move(rh));
            }
            return;
          }
        }

        // Phase 3: pop epoch if both writer and readers are done
        if (e->writer_done() && e->readers_empty())
        {
          queue_.pop_front();
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
