/// \file epoch_queue.hpp
/// \brief FIFO queue enforcing writer→readers→next-writer ordering.
/// \ingroup async_core

#pragma once

#include "common/trace.hpp"
#include "epoch_context_decl.hpp"
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
  private:
    struct Node
    {
        explicit Node(std::shared_ptr<EpochState> state) : state(std::move(state)) {}

        std::shared_ptr<EpochState> state;
        std::weak_ptr<Node> next;
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
    EpochContextReader<T> create_read_context(std::shared_ptr<detail::StorageBuffer<T>> const& storage,
                                              std::shared_ptr<EpochQueue> const& self)
    {
      std::lock_guard lock(mtx_);
      CHECK(bootstrapped_, "EpochQueue must be initialized before use");
      ensure_initial_epoch_locked();
      CHECK(tail_, "EpochQueue must retain a tail epoch");
      return EpochContextReader<T>(storage, self, tail_->state);
    }

    /// \brief Check if there are pending writers ahead of reads.
    /// \return true if any writer is still pending.
    /// \ingroup async_core
    bool has_pending_writers() const noexcept
    {
      std::lock_guard lock(mtx_);
      if (!head_) return false;
      if (!head_->state->ctx.writer_is_done()) return true;
      return nodes_.size() > 1;
    }

    /// \brief Start a new writer epoch and return a RAII writer handle.
    /// \tparam T The data type managed by the Async container.
    /// \param parent Pointer to the owning Async<T> instance.
    /// \return A new EpochContextWriter<T> bound to the back of the queue.
    /// \ingroup async_core
    template <typename T>
    EpochContextWriter<T> create_write_context(std::shared_ptr<detail::StorageBuffer<T>> const& storage,
                                               std::shared_ptr<EpochQueue> const& self)
    {
      std::lock_guard lock(mtx_);

      CHECK(bootstrapped_, "EpochQueue must be initialized before use");
      ensure_initial_epoch_locked();
      CHECK(tail_, "EpochQueue must retain a tail epoch");

      if (initial_writer_pending_ && !tail_->state->ctx.writer_is_done())
      {
        initial_writer_pending_ = false;
        return EpochContextWriter<T>(storage, self, tail_->state);
      }

      auto new_state = std::make_shared<EpochState>(&tail_->state->ctx, /*writer_already_done=*/false);
      auto new_node = std::make_shared<Node>(new_state);
      tail_->state->ctx.set_next(new_state.get());
      tail_->next = new_node;
      nodes_.push_back(new_node);
      tail_ = std::move(new_node);

      // we need to prune here, since otherwise we deadlock with no tasks running.  Presumably, what
      // happens is there is no mechanism to reschedule the task as the front of the queue is an empty
      // context.
      this->prune_front_locked();

      return EpochContextWriter<T>(storage, self, tail_->state);
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
    EpochPair<T> prepend_epoch(std::shared_ptr<detail::StorageBuffer<T>> const& storage,
                               std::shared_ptr<EpochQueue> const& self)
    {
      std::lock_guard lock(mtx_);
      CHECK(bootstrapped_, "EpochQueue must be initialized before use");
      if (!head_)
      {
        head_ = std::make_shared<Node>(std::make_shared<EpochState>(nullptr, std::integral_constant<bool, true>{}));
        tail_ = head_;
        nodes_.push_front(head_);
      }
      else
      {
        DEBUG_CHECK(!head_->state->ctx.writer_has_task());
        auto new_state = std::make_shared<EpochState>(&head_->state->ctx, std::integral_constant<bool, true>{});
        auto new_node = std::make_shared<Node>(new_state);
        new_state->ctx.set_next(head_->state.get());
        new_node->next = head_;
        head_ = std::move(new_node);
        if (!tail_) tail_ = head_;
        nodes_.push_front(head_);
      }
      return {EpochContextWriter<T>(storage, self, head_->state), EpochContextReader<T>(storage, self, head_->state)};
    }

    /// \brief Called when a writer coroutine is bound to its epoch.
    /// \param state Epoch to which the writer was bound.
    /// \ingroup async_core
    void on_writer_bound(std::shared_ptr<EpochState> const& state) noexcept
    {
      std::unique_lock lock(mtx_);
      if (head_ && state.get() == head_->state.get())
      {
        auto task = state->ctx.writer_take_task();
        if (task)
        {
          lock.unlock();
          AsyncTask::reschedule(std::move(task));
        }
      }
      else
      {
        TRACE_MODULE(ASYNC, "Writer bound to non-front epoch, deferring", state.get(),
                     head_ ? head_->state.get() : nullptr);
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
    void enqueue_reader(std::shared_ptr<EpochState> const& state, AsyncTask&& task)
    {
      std::vector<AsyncTask> ready_readers;
      bool maybe_cancel = false;
      bool scheduled_now = false;

      {
        std::unique_lock lock(mtx_);
        bool const at_front = head_ && state.get() == head_->state.get();
        bool const ready = at_front && state->ctx.reader_is_ready();

        TRACE_MODULE(ASYNC, "enqueue_reader", state.get(), at_front, ready);

        state->ctx.reader_enqueue(std::move(task));

        if (ready)
        {
          ready_readers = state->ctx.reader_take_tasks();
          maybe_cancel = state->ctx.reader_error() && (state->ctx.reader_exception() == nullptr);
          state->ctx.enter_draining_phase();
          scheduled_now = true;
        }
      }

      if (scheduled_now)
      {
        for (auto&& reader : ready_readers)
        {
          if (!maybe_cancel) reader.written();
          AsyncTask::reschedule(std::move(reader));
        }
      }
      else
      {
        this->advance();
      }
    }

    /// \brief Called when a write gate is released (writer done).
    /// \param state Epoch whose writer has completed.
    /// \ingroup async_core
    void on_writer_done(std::shared_ptr<EpochState> const& state) noexcept
    {
      TRACE_MODULE(ASYNC, "Writer has finished", state.get(), head_ ? head_->state.get() : nullptr);
      std::vector<AsyncTask> readers;
      bool maybe_cancel = false;

      {
        std::unique_lock lock(mtx_);
        state->ctx.enter_ready_phase();
        if (!head_ || state.get() != head_->state.get())
        {
          lock.unlock();
          this->advance();
          return;
        }

        readers = state->ctx.reader_take_tasks();
        maybe_cancel = state->ctx.reader_error() && (state->ctx.reader_exception() == nullptr);
        TRACE_MODULE(ASYNC, readers.size());

        if (!readers.empty())
        {
          state->ctx.enter_draining_phase();
        }
        else if (state->ctx.reader_is_empty() && head_ && nodes_.size() > 1)
        {
          bool writer_required = state->ctx.writer_is_required();
          this->pop_front_locked();
          if (head_ && writer_required) head_->state->ctx.writer_require();
          lock.unlock();
          advance();
          return;
        }
      }

      if (!readers.empty())
      {
        TRACE_MODULE(ASYNC, "Finished writer results in some readers getting rescheduled");
        for (auto&& task : readers)
        {
          TRACE_MODULE(ASYNC, "Rescheduling", &task);
          if (!maybe_cancel) task.written();
          AsyncTask::reschedule(std::move(task));
        }
      }
    }

    /// \brief Called when the last reader of an EpochContext has been released.
    /// \details Invoked by EpochContextReader only when the reference count reaches zero. If this epoch is the
    ///          front of the queue and the writer is also done, the epoch can be safely removed.
    /// \param state Epoch whose final reader has been released.
    /// \note It is possible for all readers to be released before the writer is fired if a ReadBuffer is destroyed
    ///       or released before await_suspend() occurs.
    /// \note We need to check state->ctx.reader_is_empty() after acquiring the lock, to avoid a race condition where
    ///       this epoch is current and another reader gets added.
    /// \ingroup async_core
    void on_all_readers_released(std::shared_ptr<EpochState> const& state) noexcept
    {
      std::unique_lock lock(mtx_);
      if (state->ctx.reader_is_empty())
      {
        TRACE_MODULE(ASYNC, "readers finished - we might be able to advance the epoch");
        //  Only pop if this is the front epoch and fully done, and there are other epochs waiting
        if (!head_ || head_->state.get() != state.get() || !state->ctx.writer_is_done() || nodes_.size() < 2)
        {
          return;
        }
        bool writer_required = state->ctx.writer_is_required();
        this->pop_front_locked();
        if (head_ && writer_required) head_->state->ctx.writer_require();
        lock.unlock();
        this->advance();
      }
    }

    /// \brief Check if a given epoch is at the front of the queue.
    /// \param e Epoch to test.
    /// \return true if \p e is the front epoch.
    /// \ingroup async_core
    bool is_front(EpochState const* state) const noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_CHECK(head_, "EpochQueue is empty");
      return head_ && state == head_->state.get();
    }

  private:
    /// \brief Advance the queue by scheduling next writer/readers as appropriate.
    /// \ingroup internal
    void advance() noexcept
    {
      while (true)
      {
        TRACE_MODULE(ASYNC, "advance()");
        std::vector<AsyncTask> ready_readers;
        AsyncTask writer_task;
        bool maybe_cancel = false;

        {
          std::unique_lock lock(mtx_);
          if (!head_) return;
          auto* e = &head_->state->ctx;

          TRACE_MODULE(ASYNC, e, e->writer_has_task());

          if (e->writer_has_task())
          {
            writer_task = e->writer_take_task();
            if (writer_task)
            {
              lock.unlock();
              AsyncTask::reschedule(std::move(writer_task));
              return;
            }
          }

          if (e->reader_is_ready())
          {
            ready_readers = e->reader_take_tasks();
            maybe_cancel = e->reader_error() && (e->reader_exception() == nullptr);
            if (!ready_readers.empty())
            {
              e->enter_draining_phase();
              lock.unlock();
              for (auto&& task : ready_readers)
              {
                if (!maybe_cancel) task.written();
                AsyncTask::reschedule(std::move(task));
              }
              return;
            }
          }

          if (e->writer_is_done() && e->reader_is_empty() && head_ && nodes_.size() > 1)
          {
            bool writer_required = e->writer_is_required();
            this->pop_front_locked();
            if (head_ && writer_required) head_->state->ctx.writer_require();

            continue;
          }

          return;
        }
      }
    }

    /// \brief Remove completed epochs at the front while the queue mutex is held.
    /// \ingroup internal
    void prune_front_locked()
    {
      while (head_ && head_->state->ctx.writer_is_done() && head_->state->ctx.reader_is_empty() && nodes_.size() > 1)
      {
        bool writer_required = head_->state->ctx.writer_is_required();
        this->pop_front_locked();
        if (head_ && writer_required) head_->state->ctx.writer_require();
      }
    }

    /// \brief Pop the front epoch while holding the queue mutex.
    /// \ingroup internal
    void pop_front_locked()
    {
      if (nodes_.empty())
      {
        head_.reset();
        tail_.reset();
        return;
      }

      nodes_.pop_front();
      if (nodes_.empty())
      {
        head_.reset();
        tail_.reset();
        return;
      }

      head_ = nodes_.front();
      tail_ = nodes_.back();
    }

    /// \brief Lazily create the bootstrap epoch if none exists.
    /// \ingroup internal
    void ensure_initial_epoch_locked()
    {
      if (head_) return;

      head_ =
          std::make_shared<Node>(std::make_shared<EpochState>(initial_value_initialized_, initial_value_initialized_));
      tail_ = head_;
      nodes_.push_back(head_);

      if (initial_value_initialized_)
      {
        head_->state->ctx.writer_has_written();
      }
      else
      {
        head_->state->ctx.writer_require();
      }
    }

    mutable std::mutex mtx_;     ///< Protects queue_.
    std::shared_ptr<Node> head_; ///< Head of the epoch list.
    std::shared_ptr<Node> tail_; ///< Tail pointer for O(1) append.
    std::deque<std::shared_ptr<Node>> nodes_;
    bool bootstrapped_ = false;             ///< True once initialize() has been called.
    bool initial_writer_pending_ = false;   ///< True if the initial epoch still expects its first writer.
    bool initial_value_initialized_ = true; ///< Tracks bootstrap initialization state.

#if UNI20_DEBUG_DAG
    inline static std::atomic<uint64_t> global_counter_ = 0;
    NodeInfo const* node_ = nullptr;
#endif
};

} // namespace uni20::async
