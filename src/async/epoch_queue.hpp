/// \file epoch_queue.hpp
/// \brief FIFO queue enforcing writer→readers→next-writer ordering.
/// \ingroup async_core

#pragma once

#include "common/trace.hpp"
#include "epoch_context.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

namespace uni20::async
{

/// \brief Coordinates multiple epochs of read/write gates.
/// \ingroup async_core
class EpochQueue {
  private:
    struct Node
    {
        Node(EpochContext const* prev, bool writer_already_done) : ctx(prev, writer_already_done) {}
        Node(bool writer_already_done, bool value_initialized) : ctx(nullptr, writer_already_done, value_initialized) {}
        Node(EpochContext const* next, std::integral_constant<bool, true> reverse)
            : ctx(next, std::integral_constant<bool, true>{})
        {}

        EpochContext ctx;
        std::unique_ptr<Node> next;
    };

  public:
    /// \brief Default-construct an empty epoch queue.
    /// \ingroup async_core
    EpochQueue() = default;

#if UNI20_DEBUG_DAG
    /// \brief Initialize debug metadata for DAG visualization.
    /// \param value Pointer to the associated value stored in Async<T>.
    template <typename V> void initialize_node(V const* value) noexcept
    {
      if (node_) return;
      global_counter_.fetch_add(1, std::memory_order_relaxed);
      node_ = NodeInfo::create(value);
    }

    /// \brief Access the debug node for this queue.
    NodeInfo const* node() const noexcept { return node_; }
#endif

    /// \brief Establish the initial epoch state before any readers or writers are created.
    /// \param value_initialized True if the initial epoch already contains a valid value.
    /// \ingroup async_core
    void initialize(bool value_initialized)
    {
      std::lock_guard lock(mtx_);
      if (bootstrapped_) return;

      bootstrapped_ = true;
      initial_value_initialized_ = value_initialized;
      initial_writer_pending_ = !value_initialized;
    }

    /// \brief Start a new reader epoch and return a RAII reader handle.
    /// \tparam T The data type managed by the Async container.
    /// \param parent Pointer to the owning Async<T> instance.
    /// \return A new EpochContextReader<T> bound to the back of the queue.
    /// \ingroup async_core
    template <typename T>
    EpochContextReader<T> create_read_context(std::shared_ptr<T> const& value, std::shared_ptr<EpochQueue> const& self)
    {
      std::lock_guard lock(mtx_);
      CHECK(bootstrapped_, "EpochQueue must be initialized before use");
      ensure_initial_epoch_locked();
      CHECK(tail_, "EpochQueue must retain a tail epoch");
      return EpochContextReader<T>(value, self, &tail_->ctx);
    }

    /// \brief Check if there are pending writers ahead of reads.
    /// \return true if any writer is still pending.
    /// \ingroup async_core
    bool has_pending_writers() const noexcept
    {
      std::lock_guard lock(mtx_);
      if (!head_) return false;
      if (!head_->ctx.writer_is_done()) return true;
      return static_cast<bool>(head_->next);
    }

    /// \brief Start a new writer epoch and return a RAII writer handle.
    /// \tparam T The data type managed by the Async container.
    /// \param parent Pointer to the owning Async<T> instance.
    /// \return A new EpochContextWriter<T> bound to the back of the queue.
    /// \ingroup async_core
    template <typename T>
    EpochContextWriter<T> create_write_context(std::shared_ptr<T> const& value, std::shared_ptr<EpochQueue> const& self)
    {
      std::lock_guard lock(mtx_);

      CHECK(bootstrapped_, "EpochQueue must be initialized before use");
      ensure_initial_epoch_locked();
      CHECK(tail_, "EpochQueue must retain a tail epoch");

      if (initial_writer_pending_ && !tail_->ctx.writer_is_done())
      {
        initial_writer_pending_ = false;
        return EpochContextWriter<T>(value, self, &tail_->ctx);
      }

      auto new_node = std::make_unique<Node>(&tail_->ctx, /*writer_already_done=*/false);
      Node* new_tail = new_node.get();
      tail_->ctx.set_next(&new_tail->ctx);
      tail_->next = std::move(new_node);
      tail_ = new_tail;
      prune_front_locked();
      return EpochContextWriter<T>(value, self, &tail_->ctx);
    }

    /// \brief Prepend a new epoch to the front of the queue.
    /// \return {EpochContextWriter, EpochContextReader} to the new front epoch.
    /// \pre The current front epoch must not have a writer bound.
    /// \ingroup async_core
    template <typename T> struct EpochPair
    {
        EpochContextWriter<T> writer;
        EpochContextReader<T> reader;
    };

    /// \brief Create a new epoch at the front of the queue for reverse-mode operations.
    /// \tparam T The data type managed by the Async container.
    /// \param parent Pointer to the owning Async<T> instance.
    /// \return Writer/reader handles for the new epoch.
    /// \ingroup async_core
    template <typename T>
    EpochPair<T> prepend_epoch(std::shared_ptr<T> const& value, std::shared_ptr<EpochQueue> const& self)
    {
      std::lock_guard lock(mtx_);
      CHECK(bootstrapped_, "EpochQueue must be initialized before use");
      if (!head_)
      {
        head_ = std::make_unique<Node>(nullptr, std::integral_constant<bool, true>{});
        tail_ = head_.get();
      }
      else
      {
        DEBUG_CHECK(!head_->ctx.writer_has_task());
        auto new_node = std::make_unique<Node>(&head_->ctx, std::integral_constant<bool, true>{});
        new_node->ctx.set_next(&head_->ctx);
        new_node->next = std::move(head_);
        head_ = std::move(new_node);
        if (!tail_) tail_ = head_.get();
      }
      return {EpochContextWriter<T>(value, self, &head_->ctx), EpochContextReader<T>(value, self, &head_->ctx)};
    }

    /// \brief Called when a writer coroutine is bound to its epoch.
    /// \param e Epoch to which the writer was bound.
    /// \ingroup async_core
    void on_writer_bound(EpochContext* e) noexcept
    {
      std::unique_lock lock(mtx_);
      if (head_ && e == &head_->ctx)
      {
        auto task = e->writer_take_task();
        if (task)
        {
          lock.unlock();
          AsyncTask::reschedule(std::move(task));
        }
      }
      else
      {
        TRACE_MODULE(ASYNC, "Writer bound to non-front epoch, deferring", e, head_ ? &head_->ctx : nullptr);
      }
    }

    /// \brief Conditionally enqueue or immediately schedule a reader task.
    /// \details Called by EpochContextReader to register a suspended coroutine associated with an epoch. If the
    ///          epoch is already at the front of the queue and the writer has completed, the task is scheduled
    ///          immediately without being enqueued.
    /// \param e The EpochContext associated with the reader.
    /// \param task The suspended coroutine representing the reader.
    /// \note This method must only be called after the reader has been acquired. The queue mutex is held during
    ///       the readiness check to ensure atomicity. If the epoch is not ready, the task is stored inside the
    ///       epoch until the queue advances and schedules it.
    /// \ingroup async_core
    void enqueue_reader(EpochContext* e, AsyncTask&& task)
    {
      std::unique_lock lock(mtx_);
      if (head_ && e == &head_->ctx && e->reader_is_ready())
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
        // The writer may have completed after we released the queue mutex but
        // before the reader task was enqueued. In that case no subsequent
        // queue activity would reschedule the task, so explicitly advance here
        // to probe the queue again.
        this->advance();
      }
    }

    /// \brief Called when a write gate is released (writer done).
    /// \param e Epoch whose writer has completed.
    /// \ingroup async_core
    void on_writer_done(EpochContext* e) noexcept
    {
      TRACE_MODULE(ASYNC, "Writer has finished", e, head_ ? &head_->ctx : nullptr);
      std::unique_lock lock(mtx_);
      if (!head_ || e != &head_->ctx)
      {
        TRACE_MODULE(ASYNC, "Finished writer is not at the front of the queue; will probe queue front", e,
                     head_ ? &head_->ctx : nullptr);
        lock.unlock();
        this->advance();
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
      TRACE_MODULE(ASYNC, e, head_ ? 1 : 0, e->reader_is_empty());
      if (e->reader_is_empty() && head_ && head_->next)
      {
        bool writer_required = e->writer_is_required();
        pop_front_locked();
        if (head_ && writer_required) head_->ctx.writer_require();
        lock.unlock();
        advance();
      }
    }

    /// \brief Called when the last reader of an EpochContext has been released.
    /// \details Invoked by EpochContextReader only when the reference count reaches zero. If this epoch is the
    ///          front of the queue and the writer is also done, the epoch can be safely removed.
    /// \param e Epoch whose final reader has been released.
    /// \note It is possible for all readers to be released before the writer is fired if a ReadBuffer is destroyed
    ///       or released before await_suspend() occurs.
    /// \ingroup async_core
    void on_all_readers_released(EpochContext* e) noexcept
    {
      DEBUG_CHECK(e->reader_is_empty());
      std::unique_lock lock(mtx_);
      TRACE_MODULE(ASYNC, "readers finished - we might be able to advance the epoch");
      //  Only pop if this is the front epoch and fully done, and there are other epochs waiting
      if (!head_ || &head_->ctx != e || !e->writer_is_done() || !head_->next)
      {
        return;
      }
      bool writer_required = e->writer_is_required();
      pop_front_locked();
      if (head_ && writer_required) head_->ctx.writer_require();
      lock.unlock();
      this->advance();
    }

    /// \brief Check if a given epoch is at the front of the queue.
    /// \param e Epoch to test.
    /// \return true if \p e is the front epoch.
    /// \ingroup async_core
    bool is_front(const EpochContext* e) const noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_CHECK(head_, "EpochQueue is empty");
      return head_ && e == &head_->ctx;
    }

  private:
    /// \brief Advance the queue by scheduling next writer/readers as appropriate.
    /// \ingroup internal
    void advance() noexcept
    {
      while (true)
      {
        std::unique_lock lock(mtx_);
        if (!head_) return;
        auto* e = &head_->ctx;

        TRACE_MODULE(ASYNC, e, e->writer_has_task());
        e->show();

        // Phase 1: schedule writer if not yet fired
        if (e->writer_has_task())
        {
          auto task = e->writer_take_task();
          // The task might already have been taken by another thread, in which case it will be null here
          if (task)
          {
            lock.unlock();
            AsyncTask::reschedule(std::move(task));
            return;
          }
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
        if (e->writer_is_done() && e->reader_is_empty() && head_ && head_->next)
        {
          bool writer_required = e->writer_is_required();
          pop_front_locked();
          if (head_ && writer_required) head_->ctx.writer_require();

          continue;
        }

        // Nothing more to do
        return;
      }
    }

    /// \brief Remove completed epochs at the front while the queue mutex is held.
    /// \ingroup internal
    void prune_front_locked()
    {
      while (head_ && head_->ctx.writer_is_done() && head_->ctx.reader_is_empty() && head_->next)
      {
        bool writer_required = head_->ctx.writer_is_required();
        pop_front_locked();
        if (head_ && writer_required) head_->ctx.writer_require();
      }
    }

    /// \brief Pop the front epoch while holding the queue mutex.
    /// \ingroup internal
    void pop_front_locked()
    {
      auto old_head = std::move(head_);
      if (!old_head)
      {
        tail_ = nullptr;
        return;
      }
      head_ = std::move(old_head->next);
      if (!head_)
      {
        tail_ = nullptr;
      }
    }

    /// \brief Lazily create the bootstrap epoch if none exists.
    /// \ingroup internal
    void ensure_initial_epoch_locked()
    {
      if (head_) return;

      head_ = std::make_unique<Node>(initial_value_initialized_, initial_value_initialized_);
      tail_ = head_.get();

      if (initial_value_initialized_)
      {
        head_->ctx.writer_has_written();
      }
      else
      {
        head_->ctx.writer_require();
      }
    }

    mutable std::mutex mtx_;                ///< Protects queue_.
    std::unique_ptr<Node> head_;            ///< Head of the epoch list.
    Node* tail_ = nullptr;                  ///< Tail pointer for O(1) append.
    bool bootstrapped_ = false;             ///< True once initialize() has been called.
    bool initial_writer_pending_ = false;   ///< True if the initial epoch still expects its first writer.
    bool initial_value_initialized_ = true; ///< Tracks bootstrap initialization state.

#if UNI20_DEBUG_DAG
    inline static std::atomic<uint64_t> global_counter_ = 0;
    NodeInfo const* node_ = nullptr;
#endif
};

#if UNI20_DEBUG_DAG
template <typename T> inline NodeInfo const* EpochContextReader<T>::node() const
{
  return queue_ ? queue_->node() : nullptr;
}
#endif

template <typename T> inline void EpochContextReader<T>::suspend(AsyncTask&& t)
{
  TRACE_MODULE(ASYNC, "suspend", &t, epoch_);
  DEBUG_PRECONDITION(epoch_);
  queue_->enqueue_reader(epoch_, std::move(t));
}

template <typename T> inline bool EpochContextReader<T>::is_front() const noexcept
{
  DEBUG_PRECONDITION(epoch_, this);
  return queue_->is_front(epoch_);
}

template <typename T> inline void EpochContextReader<T>::release() noexcept
{
  if (epoch_)
  {
    if (epoch_->reader_release()) queue_->on_all_readers_released(epoch_);
    epoch_ = nullptr;
  }
}

#if UNI20_DEBUG_DAG
template <typename T> inline NodeInfo const* EpochContextWriter<T>::node() const
{
  return queue_ ? queue_->node() : nullptr;
}
#endif

template <typename T> inline bool EpochContextWriter<T>::ready() const noexcept
{
  DEBUG_PRECONDITION(epoch_);
  return queue_->is_front(epoch_);
}

template <typename T> inline void EpochContextWriter<T>::suspend(AsyncTask&& t)
{
  DEBUG_PRECONDITION(epoch_);
  TRACE_MODULE(ASYNC, "suspend", &t, epoch_, epoch_->counter_);
  epoch_->writer_bind(std::move(t));
  queue_->on_writer_bound(epoch_);
}

template <typename T> inline T& EpochContextWriter<T>::data() const noexcept
{
  DEBUG_PRECONDITION(value_);                     // the value must exist
  DEBUG_PRECONDITION(queue_->is_front(epoch_));   // we must be at the front of the queue
  DEBUG_PRECONDITION(!epoch_->writer_is_done());  // writer still holds the gate
  accessed_ = true;
  return *value_;
}

template <typename T> inline void EpochContextWriter<T>::release() noexcept
{
  if (epoch_)
  {
    if (accessed_ && !marked_written_)
    {
      epoch_->writer_has_written();
      marked_written_ = true;
    }
    if (epoch_->writer_release()) queue_->on_writer_done(epoch_);
    epoch_ = nullptr;
    accessed_ = false;
    marked_written_ = false;
  }
}

} // namespace uni20::async
