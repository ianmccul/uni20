/// \file epoch_context.hpp
/// \brief Manages one “generation” of write/read ordering in an Async<T>
/// \ingroup async_core
/// \note these are the declarations that do not depend on epoch_queue.hpp,
///       so we can avoid circular dependencies.

#pragma once

#include "async_errors.hpp"
#include "async_node.hpp"
#include "async_task_promise.hpp"
#include "shared_storage.hpp"
#include "task_registry.hpp"
#include <atomic>
#include <coroutine>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <vector>

namespace uni20::async
{

template <typename T> class Async;
template <typename T> class EpochContextReader;
template <typename T> class EpochContextWriter;

/// \brief Tracks read/write scheduling for a single async epoch in Async<T>.
///
/// An `EpochContext` coordinates synchronization for one logical "generation" of an `Async<T>` value.
/// It governs when a writer may produce the value and when readers may consume it.
/// The new design allows multiple writers. If there is more than one writer then they must be performing
/// a cumulative sum operation (such as adding gradients in backpropogation). In the current implemention
/// writers run sequentially in any order, but this could be replaced with a parallel reduction.
///
/// ### Key Design Goals
/// - Self-contained: manages its own state transitions.
/// - Linear chaining: each epoch links to at most one `next_epoch_`.
/// - No external state required (i.e., no manual EpochQueue manipulation).
/// - Error and cancellation propagation through epochs.
///
/// ### Lifecycle & State Machine
///
/// Each epoch moves through the following phases:
///
/// - **Pending**: Epoch is inactive, no readers/writers started.
/// - **Started**: Epoch has been started (usually by prior epoch), waiting for writers.
/// - **Writing**: Writer(s) active; awaiting completion.
/// - **Reading**: Value is ready; readers may consume it.
/// - **Finished**: All readers done; next epoch (if present) is started.
///
/// Transitions are internally synchronized and follow precise rules
/// (see `Phase transition rules` in code for exhaustive list).
///
/// ### Ownership and Lifetime
/// - EpochContexts are heap-allocated and shared via `std::shared_ptr`.
/// - Each epoch owns a reference to its successor (`next_epoch_`) if one is created.
/// - The lifetime of an epoch is extended by:
///   - Outstanding readers or writers (RAII handles)
///   - Backward reference from the next epoch
///
/// ### Interaction
/// External code should not call `EpochContext` methods directly. Instead, use:
/// - `EpochQueue::create_write_context()` → yields a writer for the current epoch
/// - `EpochQueue::create_read_context()`  → yields a reader for the current epoch
///
/// These return RAII handles (`EpochContextReader<T>`, `EpochContextWriter<T>`)
/// that encapsulate access and ensure phase transitions occur safely.
///
/// ### Exception and Cancellation Propagation
/// - Writers can set `std::exception_ptr` or mark the epoch cancelled.
/// - These propagate to `next_epoch_`, and readers will resume with the same error state.
///
/// \invariant See inline invariants and assertions in the code for phase and state guarantees.
///
/// \thread_safety Thread-safe for all operations via internal locking.
///

// =============================================================================================
// EpochContext Invariants
// =============================================================================================
//
// Exactly one of the following must hold at all times:
//
//   * phase_ == Pending
//
//   * phase_ == Started
//      && total_writers_ == 0
//
//   * phase_ == Writing
//       && num_writers_ > 0
//
//   * phase_ == Reading
//       && num_writers_ == 0
//       && (num_readers_ > 0 || next_epoch_ == nullptr)
//
//   * phase_ == Finished
//
// Normally, num_writers_ <= 1. (Not sure if there is a use for > 1, as a transient?)
//
// ---------------------------------------------------------------------------------------------
//
// Destruction invariants:
//
//   * Destruction while (num_writers_ > 0 || num_readers_ > 0)
//       is erroneous (assertion failure).
//
// ---------------------------------------------------------------------------------------------
//
// Phase transition rules:
//
//   * Pending → Started
//       Triggered by start_epoch()
//
//   * Started → Writing
//       Triggered by total_writers_ > 0.
//
//   * Writing → Reading
//       Triggered when num_writers_ drops to to 0.
//       (i.e. writer has finished).
//       \pre num_writers_ = 0
//
//   * Reading → Finished
//       Triggered when num_readers_ drops to 0 && next_epoch_ != nullptr
//       or next_epoch_ is set and num_readers == 0
//       Invokes next_epoch_->start() and passes on cancelled_ and eptr_
//       \pre num_readers_ == 0 && next_epoch_ != nullptr
//
//   * Writing → Finished
//       Triggered when num_writers_ drops to to 0 && num_readers_ == 0 && next_eopch_ != nullptr
//       (i.e. writer has finished, and there are no readers).
//       \pre num_writers_ = 0 && num_readers_ == 0 && next_epoch_ != nullptr
//
// Note that immediate transitions are possible.
//
// ---------------------------------------------------------------------------------------------
//
// Registration rules:
//
//   * Writers can only be added when phase_ == Pending || phase_ == Started.
//   * Readers can only be added when phase_ != Finished.
//
// =============================================================================================
//
// Cancellation:
// An Epoch chain can be cancelled at any time. This indicates that the final result of this
// calculation is not going to be used and Reader tasks will either need to handle the cancel()
// or they will get an exception.
//
// Initialization status is handled separately.  The EpochContext simply handles writers and readers.

// WriteBuffer participates in epoch scheduling only; value construction state is tracked by shared_storage<T>.
// If a writer is released without emplacing/writing a value, the storage can remain unconstructed for this epoch.
//
// If you co_await on a ReadBuffer that points to an EpochContext that is in the cancelled state, then
// it will either throw an exception, or cancel the coroutine (if .or_cancel() modifier was used).
//
// If there is a next_epoch_, then the cancellation and exception status is propogated forwards.
//
// co_await on a WriteBuffer on a cancelled or exception EpochContext passes the error into the coroutine.
//

class EpochQueue;
class ReverseEpochQueue;

class EpochContext {
  public:
    enum class Phase
    {
      Pending, ///< The epoch is not yet active (i.e. it is in the future)
      Started, ///< The epoch has started but is waiting for a writer task
      Writing, ///< The epoch is active and waiting for the writers to finish, with no writer currently running
      Reading, ///< The epoch is active and waiting for readers to finish
      Finished ///< The epoch is not active, and readers have finished
    };

    struct TaskState
    {
        AsyncTask task;
        bool cancel_on_exception;
    };

    struct TaskStateBuf
    {
        AsyncTask task;
        bool cancel_on_exception;
        void const* buf;
    };

    struct DebugSnapshot
    {
        int generation{0};
        Phase phase{Phase::Pending};
        EpochContext const* next_epoch{nullptr};
        std::vector<std::coroutine_handle<>> reader_tasks{};
        std::vector<std::coroutine_handle<>> writer_tasks{};
    };

    EpochContext() { TaskRegistry::register_epoch_context(this); }

    /// Construct an EpochContext given an existing next epoch.
    /// \note this is backwards propogation, the counter is initialized to next->counter_ - 1
    explicit EpochContext(std::shared_ptr<EpochContext> next) : counter_(next->counter_ - 1), next_epoch_(next)
    {
      TaskRegistry::register_epoch_context(this);
    }

    /// Helper factory to construct the previous epoch in a reverse-mode chain
    static std::shared_ptr<EpochContext> make_previous(std::shared_ptr<EpochContext> next)
    {
      return std::make_shared<EpochContext>(std::move(next));
    }

    ~EpochContext()
    {
      {
        std::lock_guard lock(mtx_);
        CHECK_EQUAL(num_writers_, 0);
        CHECK_EQUAL(num_readers_, 0);
      }
      TaskRegistry::destroy_epoch_context(this);
    }

    /// Set the next EpochContext in the chain.
    void set_next_epoch(std::shared_ptr<EpochContext> next)
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::set_next_epoch", this, next.get(), counter_);
      DEBUG_CHECK(!next_epoch_); // make sure that we haven't already set the next epoch pointer
      DEBUG_CHECK(phase_ <= Phase::Reading);

      next->counter_ = counter_ + 1; // increment the epoch counter
      next_epoch_ = std::move(next);
      if (phase_ == Phase::Reading && num_readers_ == 0) this->advance_finished_locked(std::move(lock));
    }

    // start executing the epoch, if it has not already started
    void start()
    {
      std::unique_lock lock(mtx_);
      //      DEBUG_CHECK(phase_ == Phase::Pending);
      if (phase_ == Phase::Pending) this->advance_start_locked(std::move(lock));
    }

    // start executing the epoch, assuming that we've already written
    // precondition: no existing writers
    void start_reading()
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::start_reading", this, counter_);
      DEBUG_CHECK(phase_ == Phase::Pending);
      DEBUG_CHECK(total_writers_ == 0);
      ++total_writers_;
      this->advance_start_locked(std::move(lock));
    }

    // start executing the epoch, propogating the exception and cancellation state from a previous epoch
    void start(std::exception_ptr eptr)
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::start", this, counter_);
      DEBUG_CHECK(phase_ == Phase::Pending, this);
      eptr_ = eptr;
      this->advance_start_locked(std::move(lock));
    }

    /// \brief Return a point-in-time snapshot of epoch state and queued tasks.
    DebugSnapshot debug_snapshot() const
    {
      std::lock_guard lock(mtx_);
      DebugSnapshot snapshot;
      snapshot.generation = counter_;
      snapshot.phase = phase_;
      snapshot.next_epoch = next_epoch_.get();

      snapshot.reader_tasks.reserve(reader_tasks_.size());
      for (auto const& task_state : reader_tasks_)
      {
        if (auto h = task_state.task.coroutine_handle()) snapshot.reader_tasks.push_back(h);
      }

      snapshot.writer_tasks.reserve(writer_tasks_.size());
      for (auto const& task_state : writer_tasks_)
      {
        if (auto h = task_state.task.coroutine_handle()) snapshot.writer_tasks.push_back(h);
      }

      return snapshot;
    }

    /// \brief Return coroutine handles currently queued as readers in this epoch context.
    std::vector<std::coroutine_handle<>> reader_task_handles() const
    {
      auto snapshot = this->debug_snapshot();
      return std::move(snapshot.reader_tasks);
    }

    /// \brief Return coroutine handles currently queued as writers in this epoch context.
    std::vector<std::coroutine_handle<>> writer_task_handles() const
    {
      auto snapshot = this->debug_snapshot();
      return std::move(snapshot.writer_tasks);
    }

  private:
    // Writer interface
    // internal private used only by EpochContextWriter<T>
    template <typename T> friend class EpochContextWriter;
    friend void propagate_unhandled_writer_exception(EpochContext* epoch, std::exception_ptr eptr) noexcept;

    /// \brief Acquire the writer role for this epoch.
    /// \pre phase_ <= Phase::Started
    void writer_acquire() noexcept
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_acquire", this, counter_, phase_);
      // TODO: if we have reenterant writers, then we might be able to acquire writers in Writing phase
      DEBUG_CHECK(phase_ <= Phase::Started);

      ++num_writers_;
      ++total_writers_;
      if (phase_ == Phase::Started) this->advance_writing_locked(std::move(lock));
    }

    bool has_writer() const noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::has_writer", this, counter_, phase_, total_writers_);
      return total_writers_ > 0 || phase_ >= Phase::Writing;
    }

    /// \brief Clear any inherited errors. Use when a writer is going to overwrite the stored data
    void writer_clear_errors() noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_clear_errors", this, counter_);
      DEBUG_CHECK(phase_ <= Phase::Writing);
      eptr_ = nullptr;
      // cancelled_ = false;
      // inherit_error_state_ = false;
    }

    /// \brief Bind a coroutine to act as the writer.
    /// \param task The coroutine task to register.
    /// \pre Must follow writer_acquire()
    /// \ingroup async_core
    void writer_bind(AsyncTask&& task, bool cancel_on_exception) noexcept
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_bind", this, counter_, task.h_, cancel_on_exception, phase_,
                         num_writers_, writer_active_);
      DEBUG_CHECK(phase_ <= Phase::Writing);
      DEBUG_CHECK(phase_ != Phase::Started); // if this Epoch was started, it should have transitioned to Writing by now

      // If the Epoch is running, then execute the task immediately
      if (phase_ == Phase::Writing && !writer_active_)
      {
        this->execute_writer_locked(std::move(lock), TaskState{std::move(task), cancel_on_exception});
        return;
      }

      // Otherwise, store the task for later
      writer_tasks_.emplace_back(std::move(task), cancel_on_exception);
    }

    /// \brief See if it is possible to run a writer immediately
    /// \note Returns true if the caller writer is now active
    bool writer_try_make_active() noexcept
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_try_make_active", this, counter_, phase_, writer_active_);
      DEBUG_CHECK(phase_ <= Phase::Writing);
      if (phase_ < Phase::Writing || writer_active_) return false;
      writer_active_ = true;
      return true;
    }

    void writer_set_exception(std::exception_ptr e) noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_set_exception", this, counter_);
      DEBUG_CHECK(phase_ <= Phase::Reading);
      if (!eptr_) eptr_ = e;
    }

    // void writer_set_cancel() noexcept
    // {
    //   std::lock_guard lock(mtx_);
    //   DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_set_cancel", this, counter_);
    //   DEBUG_CHECK(phase_ == Phase::Writing);
    //   cancelled_ = true;
    // }

    /// \brief Release a writer without it executing
    /// \pre Must follow writer_acquire()
    /// \return true if this call released the final writer.
    /// \ingroup async_core
    bool writer_cancel() noexcept
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_release", this, counter_, phase_);
      DEBUG_CHECK(phase_ <= Phase::Writing);
      DEBUG_CHECK(num_writers_ >= 1);

      --num_writers_;

      if (num_writers_ == 0 && phase_ == Phase::Writing)
      {
        DEBUG_CHECK(!writer_active_);
        this->advance_reading_locked(std::move(lock));
        return true;
      }
      return false;
    }

    /// \brief Release a writer that is currently active.
    /// \note This is called when a currently running coroutine releases a write buffer
    /// \pre We must be in the Writing phase here (otherwise we would not have an active coroutine).
    /// \return true if this call released the final writer.
    bool writer_release_active() noexcept
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::writer_release", this, counter_, phase_);
      DEBUG_CHECK(phase_ == Phase::Writing);
      DEBUG_CHECK(num_writers_ >= 1);
      DEBUG_CHECK(writer_active_);

      writer_active_ = false;
      --num_writers_;

      if (num_writers_ == 0 && phase_ == Phase::Writing)
      {
        this->advance_reading_locked(std::move(lock));
        return true;
      }

      if (!writer_tasks_.empty())
      {
        auto NextTask = std::move(writer_tasks_.back());
        writer_tasks_.pop_back();
        this->execute_writer_locked(std::move(lock), std::move(NextTask));
      }

      return false;
    }

  private:
    // reader interface
    // internal private used only by EpochContextReader<T>
    template <typename T> friend class EpochContextReader;

    /// \brief Reserve a reader slot for this epoch.
    ///
    /// \note Each call increases the reference count for readers. Must be
    ///       matched with a corresponding call to reader_release().
    /// \ingroup internal
    void reader_acquire() noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_acquire", this, counter_);
      DEBUG_CHECK(phase_ <= Phase::Reading);

      ++num_readers_;
    }

    /// \brief Returns true if a reader tasks can be executed immediately
    bool reader_ready() const noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_ready", this, counter_);
      DEBUG_CHECK(phase_ <= Phase::Reading);

      return phase_ == Phase::Reading;
    }

    void reader_bind(AsyncTask&& h, bool cancel_on_exception)
    {
      std::unique_lock lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_bind", this, counter_, phase_);
      DEBUG_CHECK(phase_ <= Phase::Reading);

      if (phase_ == Phase::Reading)
      {
        DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_bind is scheduling the task immediately", this, counter_);
        std::exception_ptr my_eptr = eptr_;
        lock.unlock();
        if (my_eptr)
        {
          if (cancel_on_exception)
            h.set_cancel_on_resume();
          else
            h.exception_on_resume(my_eptr);
        }
        AsyncTask::reschedule(std::move(h));
      }
      else
      {
        DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_bind is adding the reader task", h.h_);
        reader_tasks_.emplace_back(std::move(h), cancel_on_exception);
      }
    }

    /// \brief Returns the exception object.
    std::exception_ptr reader_exception() const noexcept
    {
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_exception", this, counter_);
      DEBUG_PRECONDITION(phase_ == Phase::Reading);

      return eptr_;
    }

    // bool reader_cancelled() const noexcept
    // {
    //   DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_bind", this, counter_);
    //   DEBUG_PRECONDITION(phase_ == Phase::Reading);
    //
    //   return cancelled_;
    // }
    //
    bool reader_release() noexcept
    {
      std::lock_guard lock(mtx_);
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::reader_release", this, counter_);
      DEBUG_CHECK(phase_ <= Phase::Reading);
      DEBUG_CHECK(num_readers_ >= 1);

      --num_readers_;

      if (phase_ == Phase::Reading && num_readers_ == 0 && next_epoch_ != nullptr)
      {
        phase_ = Phase::Finished;
        next_epoch_->start(eptr_);
      }

      return num_readers_ == 0;
    }

    // Helper functions
    void advance_start_locked(std::unique_lock<std::mutex> lock)
    {
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::advance_start_locked", this, counter_);
      DEBUG_PRECONDITION(phase_ == Phase::Pending);
      phase_ = Phase::Started;

      if (total_writers_ > 0) this->advance_writing_locked(std::move(lock));
    }

    void advance_writing_locked(std::unique_lock<std::mutex> lock) noexcept
    {
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::advance_writing_locked", this, counter_);
      DEBUG_PRECONDITION(phase_ == Phase::Started);
      phase_ = Phase::Writing;

      if (!writer_tasks_.empty())
      {
        auto NextTask = std::move(writer_tasks_.back());
        writer_tasks_.pop_back();
        this->execute_writer_locked(std::move(lock), std::move(NextTask));
        return;
      }

      if (num_writers_ == 0) this->advance_reading_locked(std::move(lock));
    }

    void advance_reading_locked(std::unique_lock<std::mutex> lock) noexcept
    {
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::advance_reading_locked", this, counter_);
      DEBUG_PRECONDITION(phase_ == Phase::Writing);
      phase_ = Phase::Reading;

      if (num_readers_ > 0)
      {
        this->execute_readers_locked(std::move(lock));
        return;
      }

      if (num_readers_ == 0 && next_epoch_ != nullptr) this->advance_finished_locked(std::move(lock));
    }

    void advance_finished_locked(std::unique_lock<std::mutex> lock) noexcept
    {
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::advance_finished_locked", this, counter_);
      DEBUG_PRECONDITION(phase_ == Phase::Reading && next_epoch_);
      phase_ = Phase::Finished;

      if (next_epoch_)
      {
        next_epoch_->start(eptr_);
      }
    }

    void execute_writer_locked(std::unique_lock<std::mutex> lock, TaskState task)
    {
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::execute_writer_locked", this, counter_);
      DEBUG_CHECK(phase_ == Phase::Writing);
      DEBUG_CHECK(!writer_active_);

      if (eptr_)
      {
        if (task.cancel_on_exception)
          task.task.set_cancel_on_resume();
        else
          task.task.exception_on_resume(eptr_);
      }
      writer_active_ = true;
      lock.unlock();
      AsyncTask::reschedule(std::move(task.task));
    }

    void execute_readers_locked(std::unique_lock<std::mutex> lock)
    {
      DEBUG_TRACE_MODULE(ASYNC, "EpochContext::execute_readers_locked", this, counter_);

      // When we reschedule (or cancel) a task, we need to assume that the coroutine frame may get
      // destroyed at any time, which might result in the last reference to this EpochContext itself
      // disappearing.  This means that we cannot reference *this at all once we start rescheduling
      // tasks.  Hence copy all of the data structures we need into local storage and release the lock.
      std::exception_ptr my_eptr = eptr_;
      std::vector<TaskState> my_reader_tasks;
      std::swap(reader_tasks_, my_reader_tasks);
      lock.unlock();

      for (auto&& task : my_reader_tasks)
      {
        if (my_eptr)
        {
          if (task.cancel_on_exception)
            task.task.set_cancel_on_resume();
          else
            task.task.exception_on_resume(my_eptr);
        }
        AsyncTask::reschedule(std::move(task.task));
      }
    }

    friend class EpochQueue;
    friend class ReverseEpochQueue;

    // All of the data is protected by the mutex
    mutable std::mutex mtx_;

    // Exception pointer - if this is set, then any attempt to get a buffer throws the exception.
    // This is also propogated to future epochs
    std::exception_ptr eptr_{nullptr};

    // Epoch counter. This is mainly for debugging
    int counter_{0};

    // Pointer to the following epoch
    std::shared_ptr<EpochContext> next_epoch_;

    Phase phase_{Phase::Pending};

    // Writer interface
    int total_writers_{0};
    int num_writers_{0}; ///< number of active writers
    std::vector<TaskState> writer_tasks_;
    bool writer_active_{false};

    // Reader interface
    int num_readers_{0};
    std::vector<TaskState> reader_tasks_;

    // NOTE: we should be able to use atomic operations for most data.
    // Data that is only written once in some phase, and read only in a later phase,
    // do not need any protection at all because it is sequenced by the phase change.
    // std::atomic<Phase> phase_{Phase::Pending};
    // std::atomic<int> num_writers_{0};
    // std::atomic<int> num_readers_{0};
    // std::shared_ptr<EpochContext> next_epoch_;  // shared_ptr for lifetime management
    // std::atomic<EpochContext*> next_epoch_ptr_; // copy of next_epoch_ in an atomic pointer
    //
    // bool inherit_error_state{true};       // written only before phase_ = Reading
    // bool cancelled_{false};               // written only before phase_ = Reading
    // std::exception_ptr eptr_{nullptr};    // written only before phase_ = Reading
    // AsyncTask writer_tasks_;               // written only before phase_ = Writing
    // std::vector<AsyncTask> reader_tasks_; // protected by mutex
    // std::mutex reader_tasks_mtx_;         // protects reader_tasks_
    //
    // starting next_epoch_ is gated on phase_ transitioning to Finished.
    // This is a competition between set_next_epoch() and release_writer().
    //
    // void set_next_epoch(std::shared_ptr<EpochContext> next) {
    //     DEBUG_TRACE_MODULE(ASYNC, "EpochContext::set_next_epoch", this, next.get(), counter_);
    //
    //     DEBUG_CHECK(!next_epoch_); // not set before
    //     DEBUG_CHECK(phase_.load(std::memory_order_relaxed) <= Phase::Reading);
    //
    //     // Establish ownership first
    //     next->counter_ = counter_ + 1;
    //     next_epoch_ = std::move(next);
    //
    //     // Publish raw pointer with release semantics
    //     next_epoch_ptr_.store(next_epoch_.get(), std::memory_order_release);
    //
    //     // If readers are already done, start the next epoch now
    //     if (num_readers_ == 0) {
    //         Phase expected = Phase::Reading;
    //         if (phase_.compare_exchange_strong(expected, Phase::Finished, std::memory_order_acq_rel)) {
    //             auto* next_ptr = next_epoch_ptr_.load(std::memory_order_acquire);
    //             if (next_ptr)
    //                 next_ptr->start(eptr_, cancelled_);
    //         }
    //     }
    // }
    //
    // void reader_release() noexcept
    // {
    //   int readers = --num_readers_;
    //
    //   if (readers == 0)
    //   {
    //     auto* next_ptr = next_epoch_ptr_.load(std::memory_order_acquire);
    //     if (next_ptr && phase_.load(std::memory_order_relaxed) == Phase::Reading)
    //     {
    //       Phase expected = Phase::Reading;
    //       if (phase_.compare_exchange_strong(expected, Phase::Finished, std::memory_order_acq_rel))
    //       {
    //         next_ptr->start(eptr_, cancelled_);
    //       }
    //     }
    //   }
    // }
};

inline void propagate_unhandled_writer_exception(EpochContext* epoch, std::exception_ptr eptr) noexcept
{
  if (!epoch || !eptr) return;
  epoch->writer_set_exception(std::move(eptr));
}

constexpr std::string_view format_as(uni20::async::EpochContext::Phase p)
{
  using Phase = uni20::async::EpochContext::Phase;
  switch (p)
  {
    case Phase::Pending:
      return "Pending";
    case Phase::Started:
      return "Started";
    case Phase::Writing:
      return "Writing";
    case Phase::Reading:
      return "Reading";
    case Phase::Finished:
      return "Finished";
  }
  return "Unknown";
}

/// \brief RAII handle representing the writer of a value in an EpochContext.
///
/// On construction, this marks the epoch as having a writer. When the buffer
/// is set and the writer releases the value, waiting readers may resume.
///
/// `EpochContextWriter` does not represent a coroutine or suspendable operation.
/// It is a scoped object used to enforce single-writer discipline and ensure
/// writer→readers sequencing.
///
/// \note Writer must eventually call `.release()` to trigger reader advancement.
///
/// \see EpochContext, EpochContextReader

template <typename T> class EpochContextReader {
  public:
    /// \brief Default-constructed inactive reader (no effect).
    /// \ingroup async_core
    EpochContextReader() = delete;

    /// \brief Copy constructor retaining the reader reference count.
    /// \ingroup async_core
    EpochContextReader(EpochContextReader const& other) : storage_(other.storage_), epoch_(other.epoch_)
    {
      epoch_->reader_acquire();
    }

    /// \brief Copy assignment retaining the reader reference count.
    /// \param other Reader being copied from.
    /// \return Reference to *this after copy.
    /// \ingroup async_core
    EpochContextReader& operator=(EpochContextReader const& other)
    {
      if (this != &other)
      {
        if (other.epoch_) other.epoch_->reader_acquire();
        this->release();
        storage_ = other.storage_;
        epoch_ = other.epoch_;
      }
      return *this;
    }

    /// \brief Construct a new reader handle for a given parent and epoch.
    /// \param value Shared pointer to the stored value.
    /// \param epoch Pointer to the epoch being tracked.
    /// \ingroup async_core
    EpochContextReader(shared_storage<T> storage, std::shared_ptr<EpochContext> epoch) noexcept
        : storage_(std::move(storage)), epoch_(std::move(epoch))
    {
      epoch_->reader_acquire();
    }

    /// \brief Move constructor. Transfers ownership and nulls the source.
    /// \param other Reader being moved from.
    /// \ingroup async_core
    EpochContextReader(EpochContextReader&& other) noexcept
        : storage_(std::move(other.storage_)), epoch_(std::move(other.epoch_))
    {}

    /// \brief Move assignment. Releases prior ownership and transfers.
    /// \param other Reader being moved from.
    /// \return Reference to *this after transfer.
    /// \ingroup async_core
    EpochContextReader& operator=(EpochContextReader&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::move(other.storage_);
        epoch_ = std::move(other.epoch_);
      }
      return *this;
    }

    /// \brief Destructor. Signals epoch reader completion if still active.
    /// \ingroup async_core
    ~EpochContextReader() { this->release(); }

#if UNI20_DEBUG_DAG
    /// \brief Get the debug node pointer of the object
    NodeInfo const* node() const;
#endif

    /// \brief Suspend a coroutine task as a reader of this epoch.
    /// \param t The coroutine task to register.
    /// \ingroup async_core
    void suspend(AsyncTask&& t, bool cancel_on_exception)
    {
      TRACE_MODULE(ASYNC, "EpochContextReader::suspend", this, t.h_, epoch_.get(), epoch_->counter_);
      epoch_->reader_bind(std::move(t), cancel_on_exception);
    }

    /// \brief Check whether the reader is ready to resume.
    /// \return True if all prerequisites for this epoch are satisfied.
    /// \ingroup async_core
    bool ready() const noexcept { return epoch_->reader_ready(); }

    /// \brief Access the stored value inside the parent Async<T>.
    /// \return Reference to the T value.
    /// \pre The value must be ready. Should only be called after await_ready() returns true.
    /// \throws buffer_read_uninitialized if no value was constructed for this epoch.
    /// \ingroup async_core
    T const& data() const
    {
      DEBUG_PRECONDITION(epoch_->reader_ready());
      DEBUG_TRACE_MODULE(ASYNC, "EpochContextReader::data", epoch_.get(), epoch_->counter_);
      if (auto e = epoch_->reader_exception(); e) std::rethrow_exception(e);
      auto* ptr = storage_.get();
      if (!ptr)
      {
#if UNI20_DEBUG_ASYNC_TASKS
        constexpr char const* reason = "EpochContextReader::data access on uninitialized storage";
        auto const mode = TaskRegistry::dump_mode();
        if (mode != TaskRegistry::DumpMode::None)
        {
          auto const epoch_counter = epoch_ ? epoch_->counter_ : -1;
          TRACE_STACK(reason, this, epoch_.get(), epoch_counter);
          if (mode == TaskRegistry::DumpMode::Full)
            TaskRegistry::dump();
          else
            TaskRegistry::dump_epoch_context(epoch_.get(), reason);
        }
#endif
        throw buffer_read_uninitialized{};
      }
      return *ptr;
    }

    // \brief Optionally retrieves the value stored in this buffer, if available.
    ///
    /// \return A pointer to the value, if it is written, otherwise returns nullptr.
    /// \throws Any exception stored in the writer, if the buffer is in an exception state.
    /// \pre The buffer must be ready for reading (i.e., the write gate is closed).
    /// \ingroup async_core
    T const* data_maybe() const
    {
      DEBUG_PRECONDITION(epoch_->reader_ready());
      if (auto e = epoch_->reader_exception(); e) std::rethrow_exception(e);
      return storage_.get();
    }

    /// \brief Optionally retrieves the value stored in this buffer, if available.
    ///
    /// \return A copy of the value if it was written; `std::nullopt` if the buffer was cancelled.
    /// \throws Any exception stored in the writer, if the write failed exceptionally.
    /// \pre The buffer must be ready for reading (i.e., the write gate is closed).
    /// \post If a value is returned, it is a full copy and independent of the internal buffer.
    /// \ingroup async_core
    std::optional<T> data_option() const
    {
      DEBUG_PRECONDITION(epoch_->reader_ready());
      if (auto e = epoch_->reader_exception(); e) std::rethrow_exception(e);
      auto* ptr = storage_.get();
      if (!ptr) return std::nullopt;
      return *ptr;
    }

    /// \brief Release the reader.
    /// \ingroup async_core
    void release() noexcept
    {
      if (epoch_)
      {
        epoch_->reader_release();
        epoch_.reset();
      }
    }

    /// \brief Wait for the epoch to become available on the global scheduler and return a reference to the value.
    /// \ingroup async_core
    T const& get_wait() const;

    /// \brief Wait for the epoch to become available on the provided scheduler and return a reference to the value.
    /// \param sched Scheduler used to drive readiness.
    /// \ingroup async_core
    T const& get_wait(IScheduler& sched) const;

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return epoch_; }

  private:
    shared_storage<T> storage_;
    std::shared_ptr<EpochContext> epoch_{}; ///< Epoch currently tracked.
};

/// \brief RAII-scoped representation of a writer’s participation in an epoch.
///
/// Constructed by EpochQueue and passed into a WriteBuffer<T>, this class manages
/// the binding and release of a writer coroutine to a single EpochContext. It
/// ensures that the write gate is completed exactly once, either manually via
/// release() or automatically on destruction.
///
/// \ingroup async_core
template <typename T> class EpochContextWriter {
  public:
    EpochContextWriter() = delete;

    /// \brief Construct an active writer.
    /// \param value Shared pointer to the stored value.
    /// \param epoch Epoch tracked by this writer.
    /// \ingroup async_core
    EpochContextWriter(shared_storage<T> storage, std::shared_ptr<EpochContext> epoch) noexcept
        : storage_(std::move(storage)), epoch_(std::move(epoch))
    {
      TRACE_MODULE(ASYNC, "EpochContextWriter Constructor", this, epoch_.get());
      epoch_->writer_acquire();
    }

    /// \brief Copy constructor acquiring a writer reference for diagnostics.
    /// \ingroup async_core
    // EpochContextWriter(EpochContextWriter const& other) : storage_(other.storage_), epoch_(other.epoch_)
    // {
    //   epoch_->writer_acquire_duplicate(static_cast<void const*>(this));
    // }

    explicit operator bool() const noexcept { return epoch_ != nullptr; }

    EpochContextWriter& operator=(EpochContextWriter const&) = delete;

    /// \brief Move constructor transferring the writer handle.
    /// \param other Writer being moved from.
    /// \ingroup async_core
    EpochContextWriter(EpochContextWriter&& other) noexcept
        : storage_(std::move(other.storage_)), epoch_(std::move(other.epoch_)), acquired_(std::move(other.acquired_))
    {
      other.epoch_.reset();
    }

    /// \brief Move assignment transferring the writer handle.
    /// \param other Writer being moved from.
    /// \return Reference to *this after transfer.
    /// \ingroup async_core
    EpochContextWriter& operator=(EpochContextWriter&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        storage_ = std::move(other.storage_);
        epoch_ = std::move(other.epoch_);
        acquired_ = other.acquired_;
        other.epoch_.reset();
      }
      return *this;
    }

    /// \brief Destructor releasing any held writer epoch.
    /// \ingroup async_core
    ~EpochContextWriter() { this->release(); }

#if UNI20_DEBUG_DAG
    /// \brief Get the debug node pointer of the object
    NodeInfo const* node() const;
#endif

    /// \brief Check whether the writer may proceed immediately.
    /// \note If true, then the writer is now active
    bool ready() const noexcept
    {
      if (acquired_) return true;
      acquired_ = epoch_->writer_try_make_active();
      return acquired_;
    }

    /// \brief Suspend the writer task and submit to the epoch context
    /// \param t The coroutine to bind and schedule.
    void suspend(AsyncTask&& t, bool cancel_on_exception)
    {
      DEBUG_CHECK(!acquired_);
      epoch_->writer_bind(std::move(t), cancel_on_exception);
    }

    void resume() const noexcept { acquired_ = true; }

    /// \brief Access the stored data while holding the writer gate.
    /// \return Mutable reference to the stored value.
    /// \throws buffer_write_uninitialized if storage is not yet constructed.
    T& data() const
    {
      auto* ptr = storage_.get();
      if (!ptr)
      {
#if UNI20_DEBUG_ASYNC_TASKS
        constexpr char const* reason = "EpochContextWriter::data access on uninitialized storage";
        auto const mode = TaskRegistry::dump_mode();
        if (mode != TaskRegistry::DumpMode::None)
        {
          auto const epoch_counter = epoch_ ? epoch_->counter_ : -1;
          TRACE_STACK(reason, this, epoch_.get(), epoch_counter);
          if (mode == TaskRegistry::DumpMode::Full)
            TaskRegistry::dump();
          else
            TaskRegistry::dump_epoch_context(epoch_.get(), reason);
        }
#endif
        throw buffer_write_uninitialized{};
      }
      return *ptr;
    }

    template <typename... Args>
    requires std::constructible_from<T, Args...> T& emplace(Args&&... args)
    {
      try
      {
        return storage_.emplace(std::forward<Args>(args)...);
      }
      catch (...)
      {
#if UNI20_DEBUG_ASYNC_TASKS
        constexpr char const* reason = "EpochContextWriter::emplace failed while constructing storage";
        auto const mode = TaskRegistry::dump_mode();
        if (mode != TaskRegistry::DumpMode::None)
        {
          auto const epoch_counter = epoch_ ? epoch_->counter_ : -1;
          TRACE_STACK(reason, this, epoch_.get(), epoch_counter);
          if (mode == TaskRegistry::DumpMode::Full)
            TaskRegistry::dump();
          else
            TaskRegistry::dump_epoch_context(epoch_.get(), reason);
        }
#endif
        throw;
      }
    }

    shared_storage<T>& storage() { return storage_; }

    void set_exception(std::exception_ptr e) const noexcept { epoch_->writer_set_exception(e); }

    std::shared_ptr<EpochContext> epoch_context_shared() const noexcept { return epoch_; }

    /// \brief Report whether the associated value is currently initialized.
    /// \return true if the epoch reports initialized storage.
    /// \ingroup async_core
    bool value_is_initialized() const noexcept { return storage_.constructed(); }

    /// \brief Wait for the epoch to become available, and then return a prvalue-reference to the value.
    /// \ingroup async_core
    /// \brief Finalize this write gate, if not already done.
    void release() noexcept
    {
      if (epoch_)
      {
        if (acquired_)
          epoch_->writer_release_active();
        else
          epoch_->writer_cancel(); // releasing without ever
        epoch_.reset();
        storage_.reset();
      }
    }

    /// \brief Wait for the epoch to become available, and then return a prvalue-reference to the value.
    /// \ingroup async_core
    T&& move_from_wait();

  private:
    mutable shared_storage<T> storage_;
    std::shared_ptr<EpochContext> epoch_;
    mutable bool acquired_{false};
};

} // namespace uni20::async
