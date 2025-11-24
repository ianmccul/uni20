/// \file epoch_context.hpp
/// \brief Manages one “generation” of write/read ordering in an Async<T>
/// \ingroup async_core

#pragma once

#include "async_node.hpp"
#include "async_task_promise.hpp"
#include "storage_buffer.hpp"
#include <atomic>
#include <coroutine>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

namespace uni20::async
{

template <typename T> class Async;
template <typename T> class EpochContextReader;

class EpochQueue;
struct EpochState;

// This needs a rethink.We want to track if the data has been initialized.That is, if we require_read at some point,
//     then we want to track and see if it was.We need to know this,
//     so that operator+= can detect whether we are adding to it,
//     or assigning a new value.

/// \brief Per-epoch state for an Async<T> value, coordinating one writer and multiple readers.
///
/// \details
/// The EpochContext tracks synchronization state for one generation of an Async<T>.
/// One writer coroutine may be bound, and multiple readers.
/// Writer progress is tracked via three flags:
///
///   - \c writer_task_set_ : set once the writer coroutine is registered.
///   - \c writer_done_     : set when the write gate has been released.
///   - \c writer_required_ : true if readers must be destroyed when no write occurs (e.g., canceled).
///   - \c eptr_            : if writer_done_ && writer_required_ then the writer has finished, but not
///                           written to the buffer. In this case, eptr_ == null indicates that the write
///                           was cancelled, which readers may detect and handle, OR if eptr_ != null
///                           then an exception was thrown and this will be passed on to readers.
///                           eptr_ is not itself atomic, but is fenced by writer_required_
///
/// \note
/// Epochs are constructed in a chain. The previous epoch may pass forward `writer_required_`,
/// which means that the existing value is undetermined/invalid, and the writer is required
/// to co_await on the buffer, or we enter an error state.
/// The `counter_` tracks generation number for debugging.
/// \ingroup async_core
class EpochContext {
  public:
    EpochContext() = delete;

    /// \brief Construct a new epoch.
    /// \param prev Pointer to the previous epoch (if any).
    /// \param writer_already_done If true, this epoch begins in the "bootstrap" state.
    /// \param initial_value_initialized Whether the initial value should be treated as initialized.
    /// \post If \p writer_already_done is true, this epoch is considered immediately readable.
    /// \ingroup async_core
    EpochContext(EpochContext const* prev, bool writer_already_done,
                 bool initial_value_initialized = true) noexcept
        : eptr_(prev ? prev->eptr_ : nullptr), writer_done_{writer_already_done}, writer_required_(false),
          counter_(prev ? prev->counter_ + 1 : 0)
    {
      bool initialized = prev ? prev->value_initialized_for_next() : initial_value_initialized;
      value_initialized_.store(initialized, std::memory_order_relaxed);
      value_initialized_for_next_.store(initialized, std::memory_order_relaxed);
      writer_has_written_.store(writer_already_done && initialized, std::memory_order_relaxed);
      TRACE_MODULE(ASYNC, "Creating new forwards epoch", this, counter_);
    }

    /// \brief Construct a reverse-mode epoch, linked to the next one in time.
    /// \param next Pointer to the next epoch (i.e., earlier in forward time).
    /// \param reverse Tag indicating that this is a reverse-mode epoch.
    /// \ingroup async_core
    EpochContext(EpochContext const* next, std::integral_constant<bool, true> reverse)
        : writer_done_{false}, writer_required_(true), counter_(next ? next->counter_ - 1 : 0)
    {
      value_initialized_.store(false, std::memory_order_relaxed);
      value_initialized_for_next_.store(false, std::memory_order_relaxed);
      TRACE_MODULE(ASYNC, "Creating new reverse epoch", this, counter_);
    }

    // reader interface
  private:
    template <typename T> friend class EpochContextReader;
    /// \brief Reserve a reader slot for this epoch.
    ///
    /// \note Each call increases the reference count for readers. Must be
    ///       matched with a corresponding call to reader_release().
    /// \ingroup internal
    void reader_acquire() noexcept
    {
      // DEBUG_TRACE_MODULE(ASYNC, "reader_acquire()", this);
      created_readers_.fetch_add(1, std::memory_order_relaxed);
    }

    /// \brief Signal that one reader has completed.
    ///
    /// \note Decreases the reader reference count. When all readers are released,
    ///       the epoch may be advanced by the queue.
    /// \return true if this call released the final reader handle.
    /// \ingroup internal
    bool reader_release() noexcept
    {
      // DEBUG_TRACE_MODULE(ASYNC, "reader_release()", this, created_readers_.load(std::memory_order_acquire),
      // reader_handles_.size());
      return created_readers_.fetch_sub(1, std::memory_order_acq_rel) == 1;
    }

  public:
    /// \brief Enqueue a reader coroutine to be resumed when the epoch is active.
    ///
    /// \param h The suspended reader task.
    /// \pre Must be called after reader_acquire().
    /// \note Stored coroutines are resumed when the epoch advances.
    /// \ingroup async_core
    void reader_enqueue(AsyncTask&& h)
    {
      std::lock_guard lock(reader_mtx_);
      reader_handles_.push_back(std::move(h));
    }

    /// \brief Are readers allowed to run?
    /// \return true if the writer is done.
    /// \ingroup async_core
    bool reader_is_ready() const noexcept { return this->writer_is_done(); }

    /// \brief Returns true if reading from the buffer right now would be an error condition
    /// \return true if the buffer is in an error state.
    /// \ingroup async_core
    bool reader_error() const noexcept
    {
      return this->reader_is_ready() && writer_required_.load(std::memory_order_acquire);
    }

    /// \brief returns the exception pointer set by the writer (nullptr, if no exception)
    /// \return Exception pointer captured during writing, or nullptr if none.
    /// \ingroup async_core
    std::exception_ptr reader_exception() const noexcept { return eptr_; }

    // EpochQueue interface

    /// \brief Extract all pending reader handles.
    /// \return Vector of reader handles.
    /// \ingroup async_core
    std::vector<AsyncTask> reader_take_tasks() noexcept
    {
      TRACE_MODULE(ASYNC, created_readers_.load(std::memory_order_acquire));
      std::lock_guard lock(reader_mtx_);
      std::vector<AsyncTask> v;
      v.swap(reader_handles_);
      return v;
    }

    /// \brief Check if all reader slots have been released.
    ///
    /// \note Intended for use by EpochQueue only during synchronized advancement.
    ///       Not valid for concurrent polling by external threads.
    ///
    /// \pre Caller must hold exclusive access to the epoch queue.
    /// \return true if there are no outstanding readers.
    /// \ingroup internal
    bool reader_is_empty() const noexcept { return created_readers_.load(std::memory_order_acquire) == 0; }

    /// \brief Link this epoch to its successor in the queue.
    /// \param next Next epoch in the chain.
    /// \ingroup internal
    void set_next(EpochState* next) noexcept;

    /// \brief Report whether the underlying value is currently initialized for this epoch.
    /// \return true if the stored value can be safely read.
    /// \ingroup async_core
    bool value_is_initialized() const noexcept { return value_initialized_.load(std::memory_order_acquire); }

    /// \brief Return the initialization state that should be inherited by the next epoch.
    /// \return true if the next epoch should treat the value as initialized.
    /// \ingroup internal
    bool value_initialized_for_next() const noexcept
    {
      return value_initialized_for_next_.load(std::memory_order_acquire);
    }

    /// \brief Emit debugging information for this epoch.
    /// \ingroup internal
    void show()
    {
      DEBUG_TRACE_MODULE(ASYNC, this, created_readers_.load(std::memory_order_acquire), reader_handles_.size(),
                         writer_done_.load(std::memory_order_acquire),
                         writer_task_set_.load(std::memory_order_acquire));
    }

    // Writer interface

    // internal private used only by EpochContextWriter<T>

    /// \brief Acquire the writer role for this epoch.
    /// \ingroup internal
    void writer_acquire() noexcept
    {
      // DEBUG_TRACE_MODULE(ASYNC, "writer_acquire", this);
      created_writers_.fetch_add(1, std::memory_order_relaxed);
    }

    /// \brief Bind a coroutine to act as the writer.
    /// \param task The coroutine task to register.
    /// \pre Must follow writer_acquire(), and only be called once.
    /// \ingroup async_core
    void writer_bind(AsyncTask&& task) noexcept
    {
      std::lock_guard lock(writer_task_mtx_);
      if (writer_done_.load(std::memory_order_acquire))
      {
        fail_writer_binding("EpochContext::writer_bind called after writer completed\n");
      }
      if (writer_task_set_.load(std::memory_order_acquire))
      {
        fail_writer_binding("EpochContext::writer_bind called twice for the same epoch\n");
      }
      writer_task_ = std::move(task);
      eptr_ = nullptr; // reset the exception pointer
      writer_task_set_.store(true, std::memory_order_release);
    }

    /// \brief Mark the writer as complete, releasing the epoch to readers.
    /// \pre Must follow writer_acquire(), and only be called once.
    /// \return true if this call released the final writer.
    /// \ingroup async_core
    bool writer_release() noexcept
    {
      DEBUG_CHECK(!writer_done_.load(std::memory_order_acquire));
      bool done = created_writers_.fetch_sub(1, std::memory_order_acq_rel) == 1;

      if (done)
      {
        /// Mark the writer as done, allowing readers to proceed.
        writer_done_.store(true, std::memory_order_release);
        this->propagate_value_initialized(value_initialized_.load(std::memory_order_acquire));
      }
      return done;
    }

    // External Writer interface

    // writer interface

    /// \brief Check if a writer coroutine has been bound.
    /// \return true if a writer task was assigned via await_suspend().
    /// \ingroup async_core
    bool writer_has_task() const noexcept { return writer_task_set_.load(std::memory_order_acquire); }

    /// \brief Check if the writer has released the write gate.
    /// \return true if the epoch is ready for readers.
    /// \ingroup async_core
    bool writer_is_done() const noexcept { return writer_done_.load(std::memory_order_acquire); }

    /// \brief Mark this epoch as requiring write; readers will be destroyed if no write occurs.
    /// \note This sets writer_required_ = true. If writer_release() is called without a prior write,
    ///       all reader coroutines will be destroyed.
    /// \ingroup async_core
    void writer_require() noexcept
    {
      writer_required_.store(true, std::memory_order_release);
      value_initialized_.store(false, std::memory_order_release);
      this->propagate_value_initialized(false);
    }

    /// \brief Mark this epoch as successfully written.
    /// \ingroup async_core
    void writer_has_written() noexcept
    {
      writer_required_.store(false, std::memory_order_release);
      writer_has_written_.store(true, std::memory_order_release);
      value_initialized_.store(true, std::memory_order_release);
      this->propagate_value_initialized(true);
    }

    /// \brief Determine whether the writer is still required to produce a value.
    /// \return true if readers expect a write.
    /// \ingroup async_core
    bool writer_is_required() const noexcept { return writer_required_.load(std::memory_order_acquire); }

    /// \brief Record an exception thrown by the writer coroutine.
    /// \param e Exception pointer to propagate to readers.
    /// \ingroup async_core
    void writer_set_exception(std::exception_ptr e) noexcept
    {
      eptr_ = e;
      writer_required_.store(true, std::memory_order_release);
      this->propagate_value_initialized(false);
    }

    /// \brief Transfer ownership of the bound writer coroutine.
    /// \return The bound writer coroutine (may be null if another thread has already taken it).
    /// \ingroup async_core
    AsyncTask writer_take_task() noexcept
    {
      DEBUG_TRACE_MODULE(ASYNC, "writer_take_task", this, counter_);
      bool expected = true;
      if (!writer_task_set_.compare_exchange_strong(expected, false, std::memory_order_acquire,
                                                    std::memory_order_relaxed))
      {
        // No task was set, or another thread already claimed it
        return {};
        DEBUG_TRACE_MODULE(ASYNC, "writer_take_task: no task set!", this, counter_);
      }
      return std::move(writer_task_);
    }

    // Informational interface

    /// \brief Generation counter associated with this epoch.
    /// \return Sequential identifier for debugging.
    /// \ingroup async_core
    uint64_t counter() const noexcept { return counter_; }

  private:
    /// \brief Propagate initialization state to the following epoch.
    /// \param value Initialization flag to propagate.
    /// \ingroup internal
    void propagate_value_initialized(bool value) noexcept;

    /// \brief Adopt initialization state from the previous epoch.
    /// \param value Initialization flag inherited from predecessor.
    /// \ingroup internal
    void inherit_value_initialized(bool value) noexcept
    {
      if (!writer_required_.load(std::memory_order_acquire))
      {
        value_initialized_.store(value, std::memory_order_release);
        value_initialized_for_next_.store(value, std::memory_order_relaxed);
      }
    }

    std::atomic<int> created_readers_{0};
    std::mutex reader_mtx_;
    std::vector<AsyncTask> reader_handles_;
    std::exception_ptr eptr_{nullptr}; /// set by the writer, if we have a current exception to pass on to the readers

    AsyncTask writer_task_;                    ///< Coroutine task (if bound).
    std::atomic<int> created_writers_{0};      ///< number of active writers (normally max 1)
    std::atomic<bool> writer_task_set_{false}; ///< Set if task has been bound.
    std::atomic<bool> writer_done_{false};     ///< Set when writer releases gate.
    std::atomic<bool> writer_required_{false}; ///< Flag that we want ReadBuffers to get dropped if we don't write
    std::atomic<bool> writer_has_written_{false};
    std::atomic<bool> value_initialized_{false};
    std::atomic<bool> value_initialized_for_next_{false};
    std::atomic<EpochState*> next_{nullptr};
    std::mutex writer_task_mtx_;

    [[noreturn]] static void fail_writer_binding(const char* message)
    {
      std::fputs(message, stderr);
      std::abort();
    }

  public:
    int64_t counter_ = 0;
};

class buffer_cancelled : public std::exception {
  public:
    const char* what() const noexcept override { return "ReadBuffer was cancelled: no value written"; }
};

struct EpochState : std::enable_shared_from_this<EpochState>
{
    EpochState(EpochContext const* prev, bool writer_already_done, bool initial_value_initialized = true)
        : ctx(prev, writer_already_done, initial_value_initialized)
    {
    }

    EpochState(bool writer_already_done, bool value_initialized)
        : ctx(nullptr, writer_already_done, value_initialized)
    {
    }

    EpochState(EpochContext const* next, std::integral_constant<bool, true> reverse)
        : ctx(next, std::integral_constant<bool, true>{})
    {
    }

    EpochContext ctx;
};

inline void EpochContext::set_next(EpochState* next) noexcept
{
  next_.store(next, std::memory_order_release);
  if (next) next->ctx.inherit_value_initialized(value_initialized_for_next_.load(std::memory_order_acquire));
}

inline void EpochContext::propagate_value_initialized(bool value) noexcept
{
  value_initialized_for_next_.store(value, std::memory_order_release);
  if (auto* next = next_.load(std::memory_order_acquire)) { next->ctx.inherit_value_initialized(value); }
}

/// \brief RAII-scoped representation of a reader's participation in an EpochContext.
///
/// This type is constructed by the EpochQueue and passed to a ReadBuffer<T> instance
/// to track a single reader within an epoch. It allows the reader to register for
/// suspension, test readiness, and upon destruction, notify the queue of completion.
///
/// \note This type is move-only. Reader ownership must be transferred or dropped exactly once.
/// \pre The epoch and parent must remain valid for the lifetime of the reader.
/// \ingroup async_core
template <typename T> class EpochContextReader {
  public:
    /// \brief Default-constructed inactive reader (no effect).
    /// \ingroup async_core
    EpochContextReader() = default;

    /// \brief Copy constructor retaining the reader reference count.
    /// \ingroup async_core
      EpochContextReader(EpochContextReader const& other)
          : storage_(other.storage_), queue_(other.queue_), epoch_(other.epoch_)
    {
      if (epoch_) epoch_->ctx.reader_acquire();
    }

    /// \brief Copy assignment retaining the reader reference count.
    /// \param other Reader being copied from.
    /// \return Reference to *this after copy.
    /// \ingroup async_core
    EpochContextReader& operator=(EpochContextReader const& other)
    {
      if (this != &other)
      {
        if (other.epoch_) other.epoch_->ctx.reader_acquire();
        this->release();
        storage_ = other.storage_;
        queue_ = other.queue_;
        epoch_ = other.epoch_;
      }
      return *this;
    }

    /// \brief Construct a new reader handle for a given parent and epoch.
    /// \param value Shared pointer to the stored value.
    /// \param queue Shared pointer to the associated epoch queue.
    /// \param epoch Pointer to the epoch being tracked.
    /// \ingroup async_core
    EpochContextReader(std::shared_ptr<detail::StorageBuffer<T>> const& storage, std::shared_ptr<EpochQueue> const& queue,
                       std::shared_ptr<EpochState> epoch) noexcept
        : storage_(storage), queue_(queue), epoch_(std::move(epoch))
    {
      if (epoch_) epoch_->ctx.reader_acquire();
    }

    /// \brief Move constructor. Transfers ownership and nulls the source.
    /// \param other Reader being moved from.
    /// \ingroup async_core
    EpochContextReader(EpochContextReader&& other) noexcept
        : storage_(std::move(other.storage_)), queue_(std::move(other.queue_)), epoch_(std::move(other.epoch_))
    {
      other.epoch_.reset();
    }

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
        queue_ = std::move(other.queue_);
        epoch_ = std::move(other.epoch_);
        other.epoch_.reset();
      }
      return *this;
    }

    /// \brief Destructor. Signals epoch reader completion if still active.
    /// \post If owning an epoch, marks reader done and notifies the queue.
    /// \ingroup async_core
    ~EpochContextReader() { this->release(); }

#if UNI20_DEBUG_DAG
    /// \brief Get the debug node pointer of the object
    NodeInfo const* node() const;
#endif

    /// \brief Suspend a coroutine task as a reader of this epoch.
    /// \param t The coroutine task to register.
    /// \ingroup async_core
    void suspend(AsyncTask&& t);

    /// \brief Check whether the reader is ready to resume.
    /// \return True if all prerequisites for this epoch are satisfied.
    /// \ingroup async_core
    bool ready() const noexcept
    {
      DEBUG_PRECONDITION(epoch_, this);
      return epoch_->ctx.reader_is_ready();
    }

    /// \brief Access the stored value inside the parent Async<T>.
    /// \return Reference to the T value.
    /// \pre The value must be ready. Should only be called after await_ready() returns true.
    /// \ingroup async_core
    T const& data() const
    {
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->ctx.reader_is_ready());
      DEBUG_TRACE_MODULE(ASYNC, epoch_.get(), epoch_->ctx.reader_error(), epoch_->ctx.counter_);
      if (epoch_->ctx.reader_error())
      {
        if (auto e = epoch_->ctx.reader_exception(); e)
          std::rethrow_exception(e);
        else
          throw buffer_cancelled();
      }
      auto* ptr = storage_ ? storage_->get() : nullptr;
      DEBUG_CHECK(ptr);
      return *ptr;
    }

    T const& data_assert() const
    {
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->ctx.reader_is_ready());
      if (epoch_->ctx.reader_error())
      {
        if (auto e = epoch_->ctx.reader_exception(); e)
          std::rethrow_exception(e);
        else
        {
          PANIC("buffer cancelled but not caught");
        }
      }
      auto* ptr = storage_ ? storage_->get() : nullptr;
      DEBUG_CHECK(ptr);
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
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->ctx.reader_is_ready());
      if (epoch_->ctx.reader_error())
      {
        if (auto e = epoch_->ctx.reader_exception(); e)
          std::rethrow_exception(e);
        else
          return nullptr;
      }
      return storage_ ? storage_->get() : nullptr;
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
      DEBUG_PRECONDITION(epoch_, this);
      DEBUG_PRECONDITION(epoch_->ctx.reader_is_ready());
      if (epoch_->ctx.reader_error())
      {
        if (auto e = epoch_->ctx.reader_exception(); e)
          std::rethrow_exception(e);
        else
          return std::nullopt;
      }
        auto* ptr = storage_ ? storage_->get() : nullptr;
        DEBUG_CHECK(ptr);
        return *ptr;
    }

    /// \brief Check whether this epoch is at the front of the queue.
    /// \return True if the associated epoch is the head of the epoch queue.
    /// \ingroup async_core
    bool is_front() const noexcept;

    /// \brief Release the reader, notifying the queue when appropriate.
    /// \ingroup async_core
    void release() noexcept;

    /// \brief Wait for the epoch to become available on the global scheduler, and then return a reference to the value
    /// \ingroup async_core
    T const& get_wait() const;

    /// \brief Wait for the epoch to become available on the given scheduler, and then return a reference to the value
    /// \param sched Scheduler used to drive readiness.
    /// \ingroup async_core
    T const& get_wait(IScheduler& sched) const;

  private:
    std::shared_ptr<detail::StorageBuffer<T>> storage_;
    std::shared_ptr<EpochQueue> queue_;
    std::shared_ptr<EpochState> epoch_{}; ///< Epoch currently tracked.
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
    /// \brief Construct an active writer.
    /// \param value Shared pointer to the stored value.
    /// \param queue Shared pointer to the epoch queue.
    /// \param epoch Epoch tracked by this writer.
    /// \ingroup async_core
    EpochContextWriter(std::shared_ptr<detail::StorageBuffer<T>> const& storage,
                       std::shared_ptr<EpochQueue> const& queue, std::shared_ptr<EpochState> epoch) noexcept
        : storage_(storage), queue_(queue), epoch_(std::move(epoch))
    {
      if (epoch_) epoch_->ctx.writer_acquire();
    }

    /// \brief Copy constructor acquiring a writer reference for diagnostics.
    /// \ingroup async_core
    EpochContextWriter(EpochContextWriter const& other)
        : storage_(other.storage_), queue_(other.queue_), epoch_(other.epoch_)
    {
      if (epoch_) epoch_->ctx.writer_acquire();
    }

    EpochContextWriter& operator=(EpochContextWriter const&) = delete;

    /// \brief Move constructor transferring the writer handle.
    /// \param other Writer being moved from.
    /// \ingroup async_core
      EpochContextWriter(EpochContextWriter&& other) noexcept
          : storage_(std::move(other.storage_)), queue_(std::move(other.queue_)), epoch_(std::move(other.epoch_))
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
        queue_ = std::move(other.queue_);
        epoch_ = std::move(other.epoch_);
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
    /// \return true if the writer is at the front of the queue.
    /// \ingroup async_core
    bool ready() const noexcept;

    /// \brief Suspend the writer task and submit to the epoch queue.
    /// \param t The coroutine to bind and schedule.
    /// \ingroup async_core
    void suspend(AsyncTask&& t);

    /// \brief Access the stored data while holding the writer gate.
    /// \return Mutable reference to the stored value.
    /// \ingroup async_core
    T& data() const noexcept;

    /// \brief Finalize this write gate, if not already done.
    ///
    /// \note This may be called explicitly or automatically by the destructor.
    ///       It is idempotent and safe to call more than once.
    /// \ingroup async_core
    void release() noexcept;

    /// \brief Require the writer to produce a value, canceling pending readers if omitted.
    /// \ingroup async_core
    void writer_require() noexcept
    {
      DEBUG_PRECONDITION(epoch_);
      epoch_->ctx.writer_require();
    }

    /// \brief Report whether the associated value is currently initialized.
    /// \return true if the epoch reports initialized storage.
    /// \ingroup async_core
    bool value_is_initialized() const noexcept
    {
      return epoch_ && epoch_->ctx.value_is_initialized();
    }

    /// \brief Wait for the epoch to become available, and then return a prvalue-reference to the value.
    /// \ingroup async_core
    T&& move_from_wait() const;

  private:
    std::shared_ptr<detail::StorageBuffer<T>> storage_;
    std::shared_ptr<EpochQueue> queue_;
    std::shared_ptr<EpochState> epoch_{};
    mutable bool accessed_ = false;
    mutable bool marked_written_ = false;
};

} // namespace uni20::async
